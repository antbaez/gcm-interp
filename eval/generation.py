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
POSITIONAL_STEERING_TYPES = ('positional', 'unweighted-positional', 'positional-projected')
WEIGHTED_POSITIONAL_TYPES = ('positional', 'positional-projected')
PROJECTED_STEERING_TYPES = {'positional-projected': 'masked_mean'}
STEERING_TYPE_VARIANTS = {
    'all_tokens': 'masked_mean', 'all-tokens': 'masked_mean', 'mean': 'masked_mean',
    'last-EOT': 'last_eot',
    'last-prompt': 'last_prompt',
}


def _get_steering_vector(patch_activations, layer_idx, sl, steering_type):
    if steering_type in ('last_token', 'last-token', 'last'):
        return patch_activations[layer_idx][-1, sl]
    elif steering_type in STEERING_TYPE_VARIANTS:
        if patch_activations.dim() != 2:
            raise ValueError(
                f"steering_type {steering_type!r} needs the [layers, H] "
                f"'{STEERING_TYPE_VARIANTS[steering_type]}' cache (steering_reps_cache(..., "
                f"variant=...)), got shape {tuple(patch_activations.shape)}."
            )
        return patch_activations[layer_idx][sl]
    elif steering_type in POSITIONAL_STEERING_TYPES:
        return patch_activations[layer_idx][:, sl]
    else:
        raise ValueError(f"Unknown steering_type: {steering_type!r}")

_config_printed_types = set()


def positional_weights(steering_type, coverage, patch_activations):
    """[P, 1] per-position weights for the WEIGHTED_POSITIONAL_TYPES (None for every other type)."""
    if steering_type not in WEIGHTED_POSITIONAL_TYPES:
        return None
    if coverage is None:
        raise ValueError(
            f"steering_type {steering_type!r} requires `coverage`: the per-position "
            "fraction of add/sub steering pairs that are both non-padding."
        )
    if coverage.shape[0] != patch_activations.shape[1]:
        raise ValueError(
            f"coverage length {coverage.shape[0]} does not match the steering "
            f"vector's sequence axis {patch_activations.shape[1]}."
        )
    return coverage.to(device=patch_activations.device, dtype=patch_activations.dtype).unsqueeze(-1)


def _prepare_steering_vector(patch_activations, layer_idx, steering_type, normalize, pos_weights,
                             projection_dirs=None):
    if steering_type in WEIGHTED_POSITIONAL_TYPES and pos_weights is None:
        raise ValueError(f"steering_type {steering_type!r} needs pos_weights (see positional_weights()).")
    if steering_type in PROJECTED_STEERING_TYPES and projection_dirs is None:
        raise ValueError(
            f"steering_type {steering_type!r} needs projection_dirs: the [layers, H] "
            f"'{PROJECTED_STEERING_TYPES[steering_type]}' steering cache."
        )
    sv = _get_steering_vector(patch_activations, layer_idx, slice(None), steering_type)
    if normalize:
        sv = sv / (torch.norm(sv, dim=-1, keepdim=True) + 1e-12)
    if pos_weights is not None:
        sv = sv * pos_weights
    if steering_type in PROJECTED_STEERING_TYPES:
        d = projection_dirs[layer_idx].to(device=sv.device, dtype=sv.dtype)
        d = d / (torch.norm(d) + 1e-12)
        sv = (sv @ d).unsqueeze(-1) * d
    return sv


def generate_with_patches(model, gen_toks, patch_activations, layer_ids, N, max_new_tokens=256, normalize=True, steering_type='last_token', resid=False, coverage=None, projection_dirs=None, steer_mask=None):
    """steer_mask: [batch, seq_len] bool, True where the vector is added (each prompt's content
    onward, never the pre-prompt chat template or padding); None steers every position."""
    patch_activations = patch_activations.to(model.device)
    mask = None if steer_mask is None else steer_mask.to(device=model.device, dtype=patch_activations.dtype).unsqueeze(-1)
    pos_weights = positional_weights(steering_type, coverage, patch_activations)
    tuple_output = resid and 'gemma' in model.config._name_or_path.lower() and getattr(model.config, 'model_type', '') != 'gemma4_unified'
    if steering_type not in _config_printed_types:
        print(gen_toks['input_ids'].shape, " normalize:", normalize, " steering type:", steering_type, " resid:", resid)
        _config_printed_types.add(steering_type)

    gen_kwargs = dict(pad_token_id=model.tokenizer.eos_token_id, do_sample=False,
                      top_p=None, top_k=None, temperature=None, max_new_tokens=max_new_tokens)

    # No model.all() — interventions apply to prefill only; decoding uses KV cache
    with model.generate(gen_toks, use_cache=True, **gen_kwargs) as tracer:
        for i, layer_idx in enumerate(layer_ids):
            layer = _get_layers(model)[layer_idx]
            sv = _prepare_steering_vector(patch_activations, layer_idx, steering_type, normalize, pos_weights, projection_dirs)
            if mask is not None:
                sv = sv * mask
            if resid:
                if tuple_output:
                    layer.output = (layer.output[0] + N * sv,)
                else:
                    layer.output += N * sv
            else:
                layer.self_attn.o_proj.output += N * sv
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