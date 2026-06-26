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
    cache_dir = data_handler.config.get_output_prefix()
    cache_path = f'{cache_dir}{model_name}_steering_cache_{source}.pt'

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
                steer[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
        with model.trace(b_slice) as _:
            for idx, layer in enumerate(layers):
                base[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())

    if mean:
        cache = [torch.cat(steer[i], dim=0).mean(0) - torch.cat(base[i], dim=0).mean(0) for i in range(num_layers)]
    else:
        cache = [torch.cat(steer[i], dim=0) - torch.cat(base[i], dim=0) for i in range(num_layers)]
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(torch.stack(cache), cache_path)
    print(f'Saved steering cache: {cache_path}')
    return torch.stack(cache)