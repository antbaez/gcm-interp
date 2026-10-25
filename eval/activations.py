import math
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


def steering_cache_path(cache_dir, model_name, source):
    """Positional steering cache path. `_content` marks caches built only from each prompt's
    content onward (no pre-prompt chat template, no padding)."""
    return f'{cache_dir}{model_name}_steering_cache_{source}_content.pt'


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


def hook_acts_dir(config):
    model_name = config.args.model_id.split('/')[-1]
    return f'./activations/{model_name}/from_{config.args.base}_to_{config.args.source}/pt/'


def _hook_layer(model, layer_frac):
    layers = _get_layers(model)
    layer_idx = min(math.floor(len(layers) * layer_frac + 0.5), len(layers) - 1)
    is_gemma = 'gemma' in model.config._name_or_path.lower() and getattr(model.config, 'model_type', '') != 'gemma4_unified'
    return layers, layer_idx, is_gemma


def _trace_layer(model, layer, ids, mask, is_gemma):
    with model.trace({'input_ids': ids, 'attention_mask': mask}) as _:
        raw = layer.output[0] if is_gemma else layer.output
        batch = raw.detach().cpu().save()
    return batch


def _stop_token_ids(model):
    ids = {model.tokenizer.eos_token_id}
    gen_config = getattr(getattr(model, '_model', None), 'generation_config', None)
    eos = getattr(gen_config, 'eos_token_id', None)
    if eos is not None:
        ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
    ids.discard(None)
    return ids


def _hook_sets(data_handler, split):
    """(label, file name, tokenized prompts, content-positions thunk) per prompt set a hook
    covers: the add (source) and sub (base) steering prompts for split='train', or the active
    eval split's prompts (named after its test file: prose-valid on validation runs,
    prose-test on -split test runs) for split='eval', or that split's target-concept
    counterpart (DataHandler.target_split(): verse-valid / verse-test) for split='target'."""
    args = data_handler.config.args
    if split == 'target':
        name, toks, get_positions = data_handler.target_split()
        return [('target', name, toks, get_positions)]
    if split == 'eval':
        return [('eval', args.test_dataset, data_handler.base_qs_toks['test'], data_handler.test_token_positions)]
    return [(side, name, data_handler.steering_qs_toks[side],
             lambda side=side: data_handler.steering_token_positions()[side])
            for side, name in (('add', args.source), ('sub', args.base))]


def save_prompt_activations(model, data_handler, n=100, batch_size=9, layer_frac=2/3, split='train'):
    """Saves layer round(layer_frac * num_layers)'s activations at every real (non-padding)
    token of the first `n` (None: all) prompts of each _hook_sets() set, one bf16 file per set
    under hook_acts_dir(). Each file holds per-prompt [T_i, H] tensors, their input_ids, and the
    content_start / content_end indices of each (tokens before content_start are pre-prompt
    chat template, tokens from content_end on are post-prompt template)."""
    config = data_handler.config
    out_dir = hook_acts_dir(config)
    layers, layer_idx, is_gemma = _hook_layer(model, layer_frac)
    os.makedirs(out_dir, exist_ok=True)

    for side, name, toks, get_positions in _hook_sets(data_handler, split):
        path = f'{out_dir}{name}_layer={layer_idx}.pt'
        if os.path.exists(path):
            print(f'[hook_acts] {path} exists, skipping')
            continue
        positions = get_positions()
        count = toks['input_ids'].shape[0] if n is None else min(n, toks['input_ids'].shape[0])
        acts, input_ids, starts, ends = [], [], [], []
        for i in range(0, count, batch_size):
            j = min(i + batch_size, count)
            ids = toks['input_ids'][i:j].to(model.device)
            mask = toks['attention_mask'][i:j].to(model.device)
            batch = _trace_layer(model, layers[layer_idx], ids, mask, is_gemma)
            mask = mask.bool().cpu()
            for b in range(batch.shape[0]):
                real = mask[b].nonzero().squeeze(-1)
                acts.append(batch[b][real].to(torch.bfloat16))
                input_ids.append(ids[b].cpu()[real])
                starts.append(int(positions['content-start'][i + b]) - int(real[0]))
                ends.append(int(positions['last-prompt'][i + b]) + 1 - int(real[0]))
        torch.save({'acts': acts, 'input_ids': input_ids, 'content_start': torch.tensor(starts),
                    'content_end': torch.tensor(ends),
                    'layer': layer_idx}, path)
        print(f'[hook_acts] Saved {count} prompts at layer {layer_idx}/{len(layers)} ({side}): {path}')


def save_response_activations(model, data_handler, n=100, batch_size=9, max_new_tokens=512, layer_frac=2/3,
                              split='train'):
    """Greedily generates an unsteered response to each of the first `n` (None: all) prompts
    of each _hook_sets() set, then reruns prompt + response and saves layer
    round(layer_frac * num_layers)'s activations at the response tokens only (up to, not
    including, the first stop token), one bf16 file per set under hook_acts_dir() as
    <name>_response_layer=<L>.pt. Each file holds per-response [R_i, H] tensors, the response
    and prompt input_ids, the decoded responses, and whether each hit a stop token before
    max_new_tokens."""
    config = data_handler.config
    out_dir = hook_acts_dir(config)
    layers, layer_idx, is_gemma = _hook_layer(model, layer_frac)
    stop_ids = _stop_token_ids(model)
    pad_id = model.tokenizer.eos_token_id
    os.makedirs(out_dir, exist_ok=True)

    for side, name, toks, _ in _hook_sets(data_handler, split):
        path = f'{out_dir}{name}_response_layer={layer_idx}.pt'
        if os.path.exists(path):
            print(f'[hook_acts] {path} exists, skipping')
            continue
        count = toks['input_ids'].shape[0] if n is None else min(n, toks['input_ids'].shape[0])
        acts, response_ids, prompt_ids, texts, finished = [], [], [], [], []
        for i in range(0, count, batch_size):
            j = min(i + batch_size, count)
            ids = toks['input_ids'][i:j].to(model.device)
            mask = toks['attention_mask'][i:j].to(model.device)
            with model.generate({'input_ids': ids, 'attention_mask': mask}, pad_token_id=pad_id,
                                use_cache=True, do_sample=False, top_p=None, top_k=None,
                                temperature=None, max_new_tokens=max_new_tokens) as _:
                out = model.generator.output.save()
            out = out.cpu()
            ids, mask = ids.cpu(), mask.bool().cpu()

            prompts, responses = [], []
            for b in range(out.shape[0]):
                generated = out[b, ids.shape[1]:].tolist()
                stop = next((k for k, t in enumerate(generated) if t in stop_ids), None)
                finished.append(stop is not None)
                responses.append(torch.tensor(generated[:stop], dtype=ids.dtype))
                prompts.append(ids[b][mask[b]])

            seq_len = max(len(p) + len(r) for p, r in zip(prompts, responses))
            full_ids = torch.full((len(prompts), seq_len), pad_id, dtype=ids.dtype)
            full_mask = torch.zeros((len(prompts), seq_len), dtype=mask.dtype)
            for b, (p, r) in enumerate(zip(prompts, responses)):
                seq = torch.cat([p, r])
                full_ids[b, seq_len - len(seq):] = seq
                full_mask[b, seq_len - len(seq):] = 1
            batch = _trace_layer(model, layers[layer_idx], full_ids.to(model.device),
                                 full_mask.long().to(model.device), is_gemma)
            for b, (p, r) in enumerate(zip(prompts, responses)):
                acts.append(batch[b, seq_len - len(r):].to(torch.bfloat16).clone())
                response_ids.append(r)
                prompt_ids.append(p)
                texts.append(model.tokenizer.decode(r))
        torch.save({'acts': acts, 'input_ids': response_ids, 'prompt_ids': prompt_ids, 'text': texts,
                    'finished': torch.tensor(finished), 'max_new_tokens': max_new_tokens,
                    'layer': layer_idx}, path)
        print(f'[hook_acts] Saved {count} responses ({sum(finished)} stopped before max_new_tokens) '
              f'at layer {layer_idx}/{len(layers)} ({side}): {path}')


def steered_acts_path(config, layer_frac=2/3, model=None, layer_idx=None):
    """<hook_acts_dir>/steered/<steering_type>/<test_dataset>_N=<N>_steer-layer=<steered>_layer=<hooked>.pt"""
    if layer_idx is None:
        layer_idx = _hook_layer(model, layer_frac)[1]
    steered = 'all' if getattr(config.args, 'global_steer', False) else config.args.steer_layer
    return (f'{hook_acts_dir(config)}steered/{config.args.steering_type}/'
            f'{config.args.test_dataset}_N={config.args.N:g}_steer-layer={steered}_layer={layer_idx}.pt')


def save_steered_activations(model, data_handler, outputs, steer_kwargs, path, batch_size=9, layer_frac=2/3):
    """Saves layer round(layer_frac * num_layers)'s activations under steering for every eval
    prompt (base_qs_toks['test']) and its steered response. `outputs` are generate_with_patches()
    rows (padded prompt + generated tokens) in eval-prompt order; each is rerun through
    trace_with_patches() with `steer_kwargs` (patch_activations, layer_ids, N, normalize,
    steering_type, coverage, projection_dirs) and the eval split's steer mask, so the
    prompt is steered exactly as at prefill. The file holds per-prompt prompt_acts [T_i, H] (real
    prompt tokens, with content_start / content_end) and acts [R_i, H] (response tokens up to,
    not including, the first stop token), plus prompt_ids, input_ids (response), text, finished,
    N, steering_type, steer_layers and layer."""
    from eval.generation import trace_with_patches
    config = data_handler.config
    _, layer_idx, _ = _hook_layer(model, layer_frac)
    stop_ids = _stop_token_ids(model)
    pad_id = model.tokenizer.eos_token_id
    toks = data_handler.base_qs_toks['test']
    prompt_len = toks['input_ids'].shape[1]
    positions = data_handler.test_token_positions()
    steer_mask = data_handler.test_steer_mask.cpu()
    if len(outputs) != toks['input_ids'].shape[0]:
        raise ValueError(f'{len(outputs)} steered outputs for {toks["input_ids"].shape[0]} eval prompts')

    prompt_acts, acts, prompt_ids, response_ids, texts, finished, starts, ends = [], [], [], [], [], [], [], []
    for i in range(0, len(outputs), batch_size):
        rows = [o.cpu() for o in outputs[i:i + batch_size]]
        responses, stops = [], []
        for row in rows:
            generated = row[prompt_len:].tolist()
            stop = next((k for k, t in enumerate(generated) if t in stop_ids), None)
            stops.append(stop is not None)
            responses.append(generated[:stop])
        seq_len = prompt_len + max(len(r) for r in responses)
        ids = torch.full((len(rows), seq_len), pad_id, dtype=rows[0].dtype)
        attn = torch.zeros((len(rows), seq_len), dtype=toks['attention_mask'].dtype)
        mask = torch.zeros((len(rows), seq_len), dtype=torch.bool)
        for b, (row, r) in enumerate(zip(rows, responses)):
            ids[b, :prompt_len + len(r)] = row[:prompt_len + len(r)]
            attn[b, :prompt_len] = toks['attention_mask'][i + b]
            attn[b, prompt_len:prompt_len + len(r)] = 1
            mask[b, :prompt_len] = steer_mask[i + b]
        batch = trace_with_patches(model, ids.to(model.device), attn.to(model.device), capture_idx=layer_idx,
                                   steer_mask=mask, **steer_kwargs)
        for b, r in enumerate(responses):
            real = attn[b, :prompt_len].bool().nonzero().squeeze(-1)
            prompt_acts.append(batch[b, real].to(torch.bfloat16))
            acts.append(batch[b, prompt_len:prompt_len + len(r)].to(torch.bfloat16).clone())
            prompt_ids.append(ids[b, real])
            response_ids.append(torch.tensor(r, dtype=ids.dtype))
            texts.append(model.tokenizer.decode(r))
            starts.append(int(positions['content-start'][i + b]) - int(real[0]))
            ends.append(int(positions['last-prompt'][i + b]) + 1 - int(real[0]))
        finished += stops

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'prompt_acts': prompt_acts, 'acts': acts, 'prompt_ids': prompt_ids, 'input_ids': response_ids,
                'text': texts, 'finished': torch.tensor(finished), 'content_start': torch.tensor(starts),
                'content_end': torch.tensor(ends), 'N': steer_kwargs['N'],
                'steering_type': steer_kwargs['steering_type'], 'steer_layers': list(steer_kwargs['layer_ids']),
                'layer': layer_idx}, path)
    print(f'[hook_acts] Saved steered activations for {len(outputs)} prompts ({sum(finished)} stopped before '
          f'max_new_tokens) at layer {layer_idx}: {path}')


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
    cache_dir = data_handler.config.get_output_prefix()
    cache_path = steering_cache_path(cache_dir, model_name, source)

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
    print(f'[activations] is_gemma={is_gemma} layer_class={type(layers[0]).__name__}')

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
                    probe = layer.output.save()
                raw = layer.output[0] if is_gemma else layer.output
                steer[idx].append(raw.detach().cpu().save())
        if i == 0:
            _describe_hook('layers[0] raw hook', probe)
            print(f'[activations] stored slice shape: {tuple(steer[0][-1].shape)}')
        with model.trace(b_slice) as _:
            for idx, layer in enumerate(layers):
                raw = layer.output[0] if is_gemma else layer.output
                base[idx].append(raw.detach().cpu().save())

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
        print(f'[activations] steering cache shape: {result.shape}')
        torch.save(result, cache_path)
        print(f'Saved steering cache: {cache_path}')

    if variant:
        return reduced[variant].to(model.device)
    return result
