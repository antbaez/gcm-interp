import os
import gc
import torch

def mean_ablations_cache(model, data_handler, batch_size=10, key='desired'):
    toks = data_handler.source_qs_toks[key]
    attn_layer_cache = [[] for _ in range(len(model.model.layers))]
    for i in range(0, toks['input_ids'].shape[0], batch_size):
        input_slice = {
            'input_ids': toks['input_ids'][i:i+batch_size].to(model.device),
            'attention_mask': toks['attention_mask'][i:i+batch_size].to(model.device)
        }
        with model.trace(input_slice) as _:
            for idx, layer in enumerate(model.model.layers):
                attn_layer_cache[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
    attn_cache = [torch.cat(attns_in_layer, dim=0).mean(dim=0).to(model.device) for attns_in_layer in attn_layer_cache]
    return torch.stack(attn_cache)

def steering_reps_cache(model, data_handler, batch_size=10, mean=True):
    config = data_handler.config
    patch_prefix = config.get_output_prefix()
    cache_path = f"{patch_prefix}/steering_cache.pt"

    if os.path.exists(cache_path):
        print(f"Loading steering cache from {cache_path}")
        return torch.load(cache_path)

    print("Loading patching reps for steer")
    gc.collect()
    torch.cuda.empty_cache()

    source_toks = data_handler.steering_qs_toks['add']
    base_toks = data_handler.steering_qs_toks['sub']
    add_max_real = int(source_toks['attention_mask'].sum(dim=1).max().item())
    sub_max_real = int(base_toks['attention_mask'].sum(dim=1).max().item())
    source_toks = {k: v[:, -add_max_real:] for k, v in source_toks.items()}
    base_toks = {k: v[:, -sub_max_real:] for k, v in base_toks.items()}
    add_total = source_toks['input_ids'].shape[1]
    sub_total = base_toks['input_ids'].shape[1]
    add_real = int(source_toks['attention_mask'][0].sum().item())
    sub_real = int(base_toks['attention_mask'][0].sum().item())
    print(f"[steering-vec] add: total_len={add_total}, real_len={add_real} | sub: total_len={sub_total}, real_len={sub_real}")
    num_layers = len(model.model.layers)
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
            for idx, layer in enumerate(model.model.layers):
                steer[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
        with model.trace(b_slice) as _:
            for idx, layer in enumerate(model.model.layers):
                base[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())

    # Keep only positions that are real tokens in every example of both sets.
    # With left-padding, real tokens occupy the last N positions, so slicing
    # to [-min_real_len:] strips all left-padding positions.
    source_min_real = int(source_toks['attention_mask'].sum(dim=1).min().item())
    base_min_real = int(base_toks['attention_mask'].sum(dim=1).min().item())
    min_real_len = min(source_min_real, base_min_real)
    print(f'source_min_real={source_min_real}, base_min_real={base_min_real}, min_real_len={min_real_len}')

    if mean:
        cache = [torch.cat(steer[i], dim=0).mean(0) - torch.cat(base[i], dim=0).mean(0) for i in range(num_layers)]
        print(f'cache shape before padding filter: {cache[0].shape}')
        cache = [c[-min_real_len:] for c in cache]
        print(f'cache shape after padding filter: {cache[0].shape}')
    else:
        cache = [torch.cat(steer[i], dim=0) - torch.cat(base[i], dim=0) for i in range(num_layers)]
        print('########### Steering cache after ########### ', f"cache_layer_shape={cache[0].shape}")

    result = torch.stack(cache)
    print('Stacked steering cache ', f"shape={result.shape}")
    os.makedirs(patch_prefix, exist_ok=True)
    torch.save(result, cache_path)
    print(f"Saved steering cache to {cache_path}")
    return result