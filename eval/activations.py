import os
import torch

def mean_ablations_cache(model, data_handler, batch_size=9, key='desired'):
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

def steering_reps_cache(model, data_handler, batch_size=9, key='desired', mean=True):
    model_name = data_handler.config.args.model_id.split('/')[-1]
    source = data_handler.config.args.source
    cache_dir = f'results/{model_name}/{source}'
    cache_path = f'{cache_dir}/{model_name}_steering_cache_{source}.pt'

    if os.path.exists(cache_path):
        print(f'Loading steering cache from {cache_path}')
        return torch.load(cache_path, map_location=model.device)

    source_toks = data_handler.steering_qs_toks['add']
    base_toks = data_handler.steering_qs_toks['sub']
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

    if mean:
        print('########### Mean steering cache ########### ', source_toks['input_ids'].shape[0], steer[0][0].shape, base[0][0].shape, len(steer[0]), len(base))
        cache = [torch.cat(steer[i], dim=0).mean(0) - torch.cat(base[i], dim=0).mean(0) for i in range(num_layers)]
        print('########### Mean steering cache after ########### ', cache[0].shape)
    else:
        cache = [torch.cat(steer[i], dim=0) - torch.cat(base[i], dim=0) for i in range(num_layers)]
        print('########### Steering cache after ########### ', cache[0].shape)
    print('Stacked steering cache ', torch.stack(cache).shape)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(torch.stack(cache), cache_path)
    print(f'Saved steering cache to {cache_path}')
    return torch.stack(cache)