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
    steering_type = config.args.steering_type
    patch_prefix = config.get_output_prefix()
    # One cache per steering type: the three types collapse the position axis
    # differently, so they have different shapes and must not share a file.
    cache_path = f"{patch_prefix}/steering_cache_{steering_type}.pt"

    if os.path.exists(cache_path):
        print(f"Loading steering cache from {cache_path}")
        return torch.load(cache_path)

    print(f"Loading patching reps for steer (steering_type={steering_type})")
    gc.collect()
    torch.cuda.empty_cache()

    source_toks = data_handler.steering_qs_toks['add']
    base_toks = data_handler.steering_qs_toks['sub']
    # Drop columns that are pure padding for every example (left-padding => real
    # tokens are flush right, so the rightmost max_real columns hold all content).
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
        # Right-aligned cache, natural (un-flipped) order: cache[P-1] => the LAST
        # real token, cache[0] => the leftmost included token (the P-th token from
        # the end). With left-padding, real tokens are already flush-right, so the
        # last min_real_len columns of every example are all real tokens — no
        # per-example offset arithmetic needed. Generation slices from the right
        # (contribution[-actual_P:]) so cache[P-1] lands on the prompt's last token.
        source_min_real = int(source_mask.sum(dim=1).min().item())
        base_min_real = int(base_mask.sum(dim=1).min().item())
        min_real_len = min(source_min_real, base_min_real)
        print(f'source_min_real={source_min_real}, base_min_real={base_min_real}, min_real_len={min_real_len}')
        cache = []
        for i in range(num_layers):
            s_full = torch.cat(steer[i], dim=0)  # [N, Sa, H]
            b_full = torch.cat(base[i], dim=0)   # [N, Sb, H]
            # Take last min_real_len real tokens. Index 0 = leftmost token in window,
            # index P-1 = last real token. No flip needed; generation slices from
            # the right to match positions right-to-left.
            s_aligned = s_full[:, -min_real_len:, :]  # [N, P, H]
            b_aligned = b_full[:, -min_real_len:, :]
            cache.append((s_aligned - b_aligned).mean(0))  # [P, H]
        result = torch.stack(cache)  # [layers, P, H]

        # Previous left-aligned implementation:
        # # Leading-pad count per example = index of its first real token.
        # s_off = (source_mask == 0).sum(dim=1).tolist()  # [N]
        # b_off = (base_mask == 0).sum(dim=1).tolist()
        # cache = []
        # for i in range(num_layers):
        #     s_full = torch.cat(steer[i], dim=0)  # [N, Sa, H]
        #     b_full = torch.cat(base[i], dim=0)   # [N, Sb, H]
        #     s_aligned = torch.stack([s_full[n, s_off[n]:s_off[n]+min_real_len] for n in range(s_full.shape[0])])
        #     b_aligned = torch.stack([b_full[n, b_off[n]:b_off[n]+min_real_len] for n in range(b_full.shape[0])])
        #     cache.append((s_aligned - b_aligned).mean(0))  # [P, H]
        # result = torch.stack(cache)  # [layers, P, H]

    else:
        raise ValueError(f"Unknown steering_type: {steering_type}")

    print('Stacked steering cache ', f"shape={result.shape}")
    os.makedirs(patch_prefix, exist_ok=True)
    torch.save(result, cache_path)
    print(f"Saved steering cache to {cache_path}")
    return result