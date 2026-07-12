import os
import torch

def _get_layers(model):
    inner = model.model
    if hasattr(inner._module, 'language_model'):
        return inner.language_model.layers
    return inner.layers

def mean_ablations_cache(model, data_handler, batch_size=9, key='desired'):
    toks = data_handler.source_qs_toks[key]
    layers = _get_layers(model)
    attn_layer_cache = [[] for _ in range(len(layers))]
    for i in range(0, toks['input_ids'].shape[0], batch_size):
        input_slice = {
            'input_ids': toks['input_ids'][i:i+batch_size].to(model.device),
            'attention_mask': toks['attention_mask'][i:i+batch_size].to(model.device)
        }
        with model.trace(input_slice) as _:
            for idx, layer in enumerate(layers):
                attn_layer_cache[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
    attn_cache = [torch.cat(attns_in_layer, dim=0).mean(dim=0).to(model.device) for attns_in_layer in attn_layer_cache]
    return torch.stack(attn_cache)

def steering_reps_cache(model, data_handler, batch_size=9, key='desired', mean=True):
    model_name = data_handler.config.args.model_id.split('/')[-1]
    source = data_handler.config.args.source
    resid = getattr(data_handler.config.args, 'resid', False)
    cache_dir = data_handler.config.get_output_prefix()
    suffix = '_resid' if resid else ''
    cache_path = f'{cache_dir}{model_name}_steering_cache_{source}{suffix}.pt'

    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=model.device)

    source_toks = data_handler.steering_qs_toks['add']
    base_toks = data_handler.steering_qs_toks['sub']
    layers = _get_layers(model)
    num_layers = len(layers)
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
            for idx, layer in enumerate(layers):
                if resid:
                    steer[idx].append(layer.output.save())
                else:
                    steer[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
        if resid:
            if i == 0:
                raw = steer[0][-1]
                try:
                    print(f'[activations] layer.output shape: {raw.shape}')
                except Exception:
                    print(f'[activations] layer.output type: {type(raw)}')
                    print(f'[activations] layer.output[0] shape: {raw[0].shape}')
            for idx in range(num_layers):
                raw = steer[idx][-1]
                try:
                    steer[idx][-1] = raw.detach().cpu()
                except Exception:
                    steer[idx][-1] = raw[0].detach().cpu()
        with model.trace(b_slice) as _:
            for idx, layer in enumerate(layers):
                if resid:
                    base[idx].append(layer.output.save())
                else:
                    base[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
        if resid:
            for idx in range(num_layers):
                raw = base[idx][-1]
                try:
                    base[idx][-1] = raw.detach().cpu()
                except Exception:
                    base[idx][-1] = raw[0].detach().cpu()

    if mean:
        cache = [torch.cat(steer[i], dim=0).mean(0) - torch.cat(base[i], dim=0).mean(0) for i in range(num_layers)]
    else:
        cache = [torch.cat(steer[i], dim=0) - torch.cat(base[i], dim=0) for i in range(num_layers)]
    os.makedirs(cache_dir, exist_ok=True)
    result = torch.stack(cache)
    if resid:
        print(f'[activations] resid steering cache shape: {result.shape}')
    torch.save(result, cache_path)
    print(f'Saved steering cache: {cache_path}')
    return result