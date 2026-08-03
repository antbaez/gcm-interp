import os
from tqdm import tqdm
import torch
from pathlib import Path

def _get_layers(model):
    inner = model.model
    if hasattr(inner._module, 'language_model'):
        return inner.language_model.layers
    return inner.layers

def select_gen_qs_toks(config, batch_handler):
    if config.args.eval_test:
        return batch_handler.base_qs_toks['test']
    else:
        raise ValueError("eval_test must be True.")
def _get_steering_vector(patch_activations, layer_idx, sl, steering_type):
    if steering_type in ('last_token', 'last-token', 'last'):
        return patch_activations[layer_idx][-1, sl]
    elif steering_type in ('all_tokens', 'all-tokens', 'mean'):
        return patch_activations[layer_idx][:, sl].mean(dim=0)
    elif steering_type == 'positional':
        return patch_activations[layer_idx][:, sl]
    else:
        raise ValueError(f"Unknown steering_type: {steering_type!r}")

_config_printed_types = set()


def generate_with_patches(model, gen_toks, patch_activations, topk_df, N, DIM, max_new_tokens=256, normalize=True, steering_type='last_token', kv_caching=False, resid=False):
    patch_activations = patch_activations.to(model.device)
    layer_ids = topk_df['layer'].unique()
    tuple_output = resid and 'gemma' in model.config._name_or_path.lower() and getattr(model.config, 'model_type', '') != 'gemma4_unified'
    if steering_type not in _config_printed_types:
        print(gen_toks['input_ids'].shape, " normalize:", normalize, " steering type:", steering_type, " kv_caching:", kv_caching, " resid:", resid)
        _config_printed_types.add(steering_type)

    gen_kwargs = dict(pad_token_id=model.tokenizer.eos_token_id, do_sample=False,
                      top_p=None, top_k=None, temperature=None, max_new_tokens=max_new_tokens)

    if kv_caching:
        # No model.all() — interventions apply to prefill only; decoding uses KV cache
        with model.generate(gen_toks, use_cache=True, **gen_kwargs) as tracer:
            for i, layer_idx in enumerate(layer_ids):
                layer = _get_layers(model)[layer_idx]
                sv = _get_steering_vector(patch_activations, layer_idx, slice(None), steering_type)
                if normalize:
                    sv = sv / (torch.norm(sv, dim=-1, keepdim=True) + 1e-12)
                if resid:
                    if tuple_output:
                        layer.output = (layer.output[0] + N * sv,)
                    else:
                        layer.output += N * sv
                else:
                    layer.self_attn.o_proj.output += N * sv
            generated = model.generator.output.save()
    else:
        # model.all() reapplies interventions on every decoding step; use_cache=False required
        with model.generate(gen_toks, use_cache=False, **gen_kwargs) as tracer:
            with model.all():
                for i, layer_idx in enumerate(layer_ids):
                    layer = _get_layers(model)[layer_idx]
                    sv = _get_steering_vector(patch_activations, layer_idx, slice(None), steering_type)
                    if normalize:
                        sv = sv / (torch.norm(sv, dim=-1, keepdim=True) + 1e-12)
                    if resid:
                        if tuple_output:
                            layer.output = (layer.output[0] + N * sv,)
                        else:
                            layer.output = layer.output + N * sv
                    else:
                        layer.self_attn.o_proj.output = layer.self_attn.o_proj.output + N * sv
            generated = model.generator.output.save()

    return generated

def decode_responses(model, inputs, originals, edited, base, answers=None):
    decoded = []
    for i in tqdm(range(len(originals)), desc="Decoding Responses"):
        query = model.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
        orig = model.tokenizer.decode(originals[i], skip_special_tokens=True).split(query)[-1]
        edit = model.tokenizer.decode(edited[i], skip_special_tokens=True).split(query)[-1]
        to_append = {
            'query': query,
            f'old_{base}': orig,
            f'edit_{base}': edit
        }
        if answers is not None:
            to_append['answer'] = answers[i]
        decoded.append(to_append)
    assert len(decoded) > 0, "No responses decoded. Check the generation process."
    return decoded