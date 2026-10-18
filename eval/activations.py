import os
import torch

def _get_layers(model):
    inner = model.model
    if hasattr(inner._module, 'language_model'):
        return inner.language_model.layers
    return inner.layers

def _describe_hook(label, value):
    def fmt(v):
        return f'{type(v).__name__}{tuple(v.shape)}' if hasattr(v, 'shape') else type(v).__name__
    if isinstance(value, (tuple, list)):
        print(f'[activations] {label}: {type(value).__name__} len={len(value)} -> {[fmt(v) for v in value]}')
    else:
        print(f'[activations] {label}: {fmt(value)}')


REDUCED_VARIANTS = ('masked_mean', 'last_eot', 'last_prompt')


def variant_cache_path(cache_path, variant):
    return cache_path[:-len('.pt')] + f'_{variant}.pt'


def coverage_cache_path(cache_path):
    return cache_path[:-len('.pt')] + '_coverage.pt'


def steering_cache_path(cache_dir, model_name, source, resid):
    """Positional steering cache path. `_content` marks caches built only from each prompt's
    content onward (no pre-prompt chat template, no padding)."""
    suffix = '_resid' if resid else '_attn'
    return f'{cache_dir}{model_name}_steering_cache_{source}{suffix}_content.pt'


def steering_coverage(data_handler):
    """[P] fraction of add/sub steering pairs that are both past their content start at each position."""
    masks = data_handler.steering_content_masks()
    if masks['add'].shape != masks['sub'].shape:
        raise ValueError(
            f"positional steering needs paired add/sub steering prompts of equal shape, "
            f"got add {tuple(masks['add'].shape)} vs sub {tuple(masks['sub'].shape)}."
        )
    return (masks['add'] & masks['sub']).float().mean(0)


def _masked_token_mean(acts, mask):
    """[n, P, H] activations -> [H]: mean over each example's masked-in positions, then over examples."""
    m = mask.to(torch.float32).unsqueeze(-1)
    per_example = (acts.to(torch.float32) * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return per_example.mean(dim=0).to(acts.dtype)


def _masked_position_mean(acts, mask):
    """[n, P, H] activations -> ([P, H] mean over the examples masked in at each position, [P] counts)."""
    m = mask.to(torch.float32).unsqueeze(-1)
    counts = m.sum(dim=0)
    means = (acts.to(torch.float32) * m).sum(dim=0) / counts.clamp(min=1)
    return means, counts.squeeze(-1)


def _positional_diff(add_acts, sub_acts, add_mask, sub_mask):
    """[P, H] add − sub per-position mean over content tokens; 0 where either side has none."""
    add_mean, add_n = _masked_position_mean(add_acts, add_mask)
    sub_mean, sub_n = _masked_position_mean(sub_acts, sub_mask)
    diff = (add_mean - sub_mean) * ((add_n > 0) & (sub_n > 0)).to(torch.float32).unsqueeze(-1)
    return diff.to(add_acts.dtype)


def _token_mean(acts, positions):
    """[n, P, H] activations -> [H]: each example's activation at its own position, averaged over examples."""
    return acts[torch.arange(acts.shape[0]), positions].to(torch.float32).mean(dim=0).to(acts.dtype)


def steering_reps_cache(model, data_handler, batch_size=9, variant=None):
    """Positional steering cache [layers, P, H], or for a `variant` in REDUCED_VARIANTS
    the [layers, H] add − sub difference of: the per-example mean over content tokens
    (masked_mean), or the activation at each example's last-EOT / last-prompt token
    (last_eot / last_prompt). Only tokens from each prompt's content start onward are
    used: the pre-prompt chat template and padding never enter a vector. All are built
    from one activation pass and cached side by side."""
    if variant is not None and variant not in REDUCED_VARIANTS:
        raise ValueError(f"unknown steering cache variant {variant!r}")
    model_name = data_handler.config.args.model_id.split('/')[-1]
    source = data_handler.config.args.source
    resid = getattr(data_handler.config.args, 'resid', False)
    cache_dir = data_handler.config.get_output_prefix()
    cache_path = steering_cache_path(cache_dir, model_name, source, resid)

    coverage_path = coverage_cache_path(cache_path)
    if not os.path.exists(coverage_path):
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(steering_coverage(data_handler).cpu(), coverage_path)

    if os.path.exists(cache_path):
        cached_len = torch.load(cache_path, map_location='cpu').shape[1]
        if cached_len != data_handler.gen_max_len:
            print(f'[activations] {cache_path} has P={cached_len} but prompts are now padded to '
                  f'{data_handler.gen_max_len} (e.g. tokenization changed); rebuilding every steering cache.')
            os.remove(cache_path)
            for v in REDUCED_VARIANTS:
                if os.path.exists(variant_cache_path(cache_path, v)):
                    os.remove(variant_cache_path(cache_path, v))
            if os.path.exists(coverage_path):
                os.remove(coverage_path)
            torch.save(steering_coverage(data_handler).cpu(), coverage_path)

    wanted_path = variant_cache_path(cache_path, variant) if variant else cache_path
    if os.path.exists(wanted_path):
        return torch.load(wanted_path, map_location=model.device)

    source_toks = data_handler.steering_qs_toks['add']
    base_toks = data_handler.steering_qs_toks['sub']
    layers = _get_layers(model)
    num_layers = len(layers)
    steer = [[] for _ in range(num_layers)]
    base = [[] for _ in range(num_layers)]

    is_gemma = 'gemma' in model.config._name_or_path.lower() and getattr(model.config, 'model_type', '') != 'gemma4_unified'
    print(f'[activations] resid={resid} is_gemma={is_gemma} layer_class={type(layers[0]).__name__}')

    for i in range(0, source_toks['input_ids'].shape[0], batch_size):
        s_slice = {
            'input_ids': source_toks['input_ids'][i:i+batch_size].to(model.device),
            'attention_mask': source_toks['attention_mask'][i:i+batch_size].to(model.device)
        }
        b_slice = {
            'input_ids': base_toks['input_ids'][i:i+batch_size].to(model.device),
            'attention_mask': base_toks['attention_mask'][i:i+batch_size].to(model.device)
        }

        probe = None
        with model.trace(s_slice) as _:
            for idx, layer in enumerate(layers):
                if i == 0 and idx == 0:
                    probe = (layer.output if resid else layer.self_attn.o_proj.output).save()
                if resid:
                    raw = layer.output[0] if is_gemma else layer.output
                    steer[idx].append(raw.detach().cpu().save())
                else:
                    steer[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
        if i == 0:
            _describe_hook('layers[0] raw hook', probe)
            print(f'[activations] stored slice shape: {tuple(steer[0][-1].shape)}')
        with model.trace(b_slice) as _:
            for idx, layer in enumerate(layers):
                if resid:
                    raw = layer.output[0] if is_gemma else layer.output
                    base[idx].append(raw.detach().cpu().save())
                else:
                    base[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())

    os.makedirs(cache_dir, exist_ok=True)

    positions = data_handler.steering_token_positions()
    content_masks = data_handler.steering_content_masks()
    reducers = {
        'masked_mean': (lambda a: _masked_token_mean(a, content_masks['add']),
                        lambda b: _masked_token_mean(b, content_masks['sub'])),
        'last_eot': (lambda a: _token_mean(a, positions['add']['last-EOT']),
                     lambda b: _token_mean(b, positions['sub']['last-EOT'])),
        'last_prompt': (lambda a: _token_mean(a, positions['add']['last-prompt']),
                        lambda b: _token_mean(b, positions['sub']['last-prompt'])),
    }
    reduced = {}
    for name, (reduce_add, reduce_sub) in reducers.items():
        reduced[name] = torch.stack([
            reduce_add(torch.cat(steer[i], dim=0)) - reduce_sub(torch.cat(base[i], dim=0))
            for i in range(num_layers)
        ])
        path = variant_cache_path(cache_path, name)
        torch.save(reduced[name], path)
        print(f'Saved {name} steering cache {tuple(reduced[name].shape)}: {path}')

    if os.path.exists(cache_path):
        result = torch.load(cache_path, map_location=model.device)
    else:
        result = torch.stack([
            _positional_diff(torch.cat(steer[i], dim=0), torch.cat(base[i], dim=0),
                             content_masks['add'], content_masks['sub'])
            for i in range(num_layers)
        ])
        if resid:
            print(f'[activations] resid steering cache shape: {result.shape}')
        torch.save(result, cache_path)
        print(f'Saved steering cache: {cache_path}')

    if variant:
        return reduced[variant].to(model.device)
    return result
