import os
import gc
import torch

def _get_layers(model):
    inner = model.model
    if hasattr(inner._module, 'language_model'):
        return inner.language_model.layers
    return inner.layers

def mean_ablations_cache(model, data_handler, batch_size=10, key='desired'):
    toks = data_handler.source_qs_toks[key]
    attn_layer_cache = [[] for _ in range(len(_get_layers(model)))]
    for i in range(0, toks['input_ids'].shape[0], batch_size):
        input_slice = {
            'input_ids': toks['input_ids'][i:i+batch_size].to(model.device),
            'attention_mask': toks['attention_mask'][i:i+batch_size].to(model.device)
        }
        with model.trace(input_slice) as _:
            for idx, layer in enumerate(_get_layers(model)):
                attn_layer_cache[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
    attn_cache = [torch.cat(attns_in_layer, dim=0).mean(dim=0).to(model.device) for attns_in_layer in attn_layer_cache]
    return torch.stack(attn_cache)

def steering_reps_cache(model, data_handler, batch_size=10, mean=True):
    config = data_handler.config
    steering_type = config.args.steering_type
    patch_prefix = config.get_output_prefix()
    cache_path = f"{patch_prefix}/steering_cache_{steering_type}.pt"

    if os.path.exists(cache_path):
        print(f"Loading steering cache from {cache_path}")
        return torch.load(cache_path)

    print(f"Loading patching reps for steer (steering_type={steering_type})")
    gc.collect()
    torch.cuda.empty_cache()

    source_toks = data_handler.steering_qs_toks['add']
    base_toks = data_handler.steering_qs_toks['sub']
    num_layers = len(_get_layers(model))
    steer = [[] for _ in range(num_layers)]
    base = [[] for _ in range(num_layers)]

    for i in range(0, source_toks['input_ids'].shape[0], batch_size):
        s_slice = {
            'input_ids': source_toks['input_ids'][i:i+batch_size].to(model.device),
            'attention_mask': source_toks['attention_mask'][i:i+batch_size].to(model.device)
        }
        b_slice = {
            'input_ids': base_toks['input_ids'][i:i+batch_size].to(model.device),
            'attention_mask': base_toks['attention_mask'][i:i+batch_size].to(model.device)
        }

        with model.trace(s_slice) as _:
            for idx, layer in enumerate(_get_layers(model)):
                steer[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
        with model.trace(b_slice) as _:
            for idx, layer in enumerate(_get_layers(model)):
                base[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())

    # Masks on CPU, aligned with the cached (CPU) activations.
    source_mask = source_toks['attention_mask'].detach().cpu()  # [N, Sa]
    base_mask = base_toks['attention_mask'].detach().cpu()       # [N, Sb]

    if steering_type == 'last-token':
        # Last real token of each example. Left-padding => it sits at index -1.
        cache = []
        for i in range(num_layers):
            s = torch.cat(steer[i], dim=0)[:, -1, :].mean(0)  # [H]
            b = torch.cat(base[i], dim=0)[:, -1, :].mean(0)   # [H]
            cache.append(s - b)
        result = torch.stack(cache)  # [layers, H]


        cache2 = []
        for i in range(num_layers):
            s = torch.cat(steer[i], dim=0).mean(0)  # [H]
            b = torch.cat(base[i], dim=0).mean(0)   # [H]
            cache2.append(s - b)
        result2 = torch.stack(cache2)
        print(result2.shape)

    elif steering_type == 'mean':
        # Masked mean over each example's non-padding tokens, then mean over examples.
        s_denom = source_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [N, 1]
        b_denom = base_mask.sum(dim=1, keepdim=True).clamp(min=1)
        s_w = source_mask.unsqueeze(-1)  # [N, Sa, 1]
        b_w = base_mask.unsqueeze(-1)    # [N, Sb, 1]
        cache = []
        for i in range(num_layers):
            s = ((torch.cat(steer[i], dim=0) * s_w).sum(dim=1) / s_denom).mean(0)  # [H]
            b = ((torch.cat(base[i], dim=0) * b_w).sum(dim=1) / b_denom).mean(0)   # [H]
            cache.append(s - b)
        result = torch.stack(cache)  # [layers, H]

    elif steering_type == 'positional':
        cache = []
        for i in range(num_layers):
            s = torch.cat(steer[i], dim=0).mean(0)
            b = torch.cat(base[i], dim=0).mean(0)
            cache.append(s - b)
        result = torch.stack(cache)

    else:
        raise ValueError(f"Unknown steering_type: {steering_type}")

    print('Stacked steering cache ', f"shape={result.shape}")
    os.makedirs(patch_prefix, exist_ok=True)
    torch.save(result, cache_path)
    print(f"Saved steering cache to {cache_path}")
    return result