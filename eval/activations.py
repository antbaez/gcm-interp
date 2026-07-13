import os
import torch

def _get_layers(model):
    inner = model.model
    if hasattr(inner._module, 'language_model'):
        return inner.language_model.layers
    return inner.layers

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

        is_gemma = 'gemma' in model.config._name_or_path.lower()
        print('is gemma', is_gemma)
        with model.trace(s_slice) as _:
            for idx, layer in enumerate(layers):
                if resid:
                    raw = layer.output[0] if is_gemma else layer.output
                    steer[idx].append(raw.detach().cpu().save())
                else:
                    steer[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())
        if resid and i == 0:
            print(f'[activations] layer.output shape: {steer[0][-1].shape}')
        with model.trace(b_slice) as _:
            for idx, layer in enumerate(layers):
                if resid:
                    raw = layer.output[0] if is_gemma else layer.output
                    base[idx].append(raw.detach().cpu().save())
                else:
                    base[idx].append(layer.self_attn.o_proj.output.detach().cpu().save())

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