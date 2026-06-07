import os
from tqdm import tqdm
import torch
from pathlib import Path

def select_gen_qs_toks(config, batch_handler):
    if config.args.eval_train:
        # print("Evaluating on training set.")
        return batch_handler.base_qs_toks['desired']
    elif config.args.eval_test:
        return batch_handler.base_qs_toks['test']
    elif config.args.eval_transfer:
        print("Evaluating on eval_test dataset.")
        return batch_handler.eval_transfer['queries']
    else:
        raise ValueError("Either eval_train or eval_test must be True.")

def generate_with_patches(model, gen_toks, patch_activations, topk_df, N, ablation_type, DIM, max_new_tokens=256, normalize=True, steering_type=None):
    if steering_type is None:
        raise ValueError("steering_type must be specified: 'last-token', 'mean', or 'positional'")
    patch_activations = patch_activations['desired'].to(model.device)
    P = patch_activations.shape[1]

    # Precompute the per-head steering contributions OUTSIDE the trace, so the traced
    # body below contains only plain tensor assignments. pandas indexing / proxy
    # math inside the trace gets routed through nnsight's tracing hacks and is fragile.
    interventions = []  # list of (layer_idx, slice, contribution)
    for layer_idx in topk_df['layer'].unique():
        head_ids_for_layer = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
        for head_idx in head_ids_for_layer:
            sl = slice(DIM * head_idx, DIM * (head_idx + 1))

            if steering_type == 'last-token':
                sv = patch_activations[layer_idx][-1, sl]            # [dim]
            elif steering_type == 'mean':
                sv = patch_activations[layer_idx][:, sl].mean(dim=0) # [dim]
            elif steering_type == 'positional':
                sv = patch_activations[layer_idx][:, sl]             # [P, dim]

            if normalize:
                sv = sv / (torch.norm(sv, dim=-1, keepdim=True) + 1e-12)

            interventions.append((int(layer_idx), sl, N * sv))

    with model.generate(
        gen_toks,
        pad_token_id=model.tokenizer.eos_token_id,
        use_cache=True,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        max_new_tokens=max_new_tokens
    ) as tracer:
        # No model.all(): interventions placed directly in the generate body apply
        # only to the FIRST forward pass — the prefill over the full prompt. Decode
        # steps are left untouched, so only prompt-token attention outputs are steered.
        # Those steered prompt activations are written into the KV cache, so the rest
        # of generation still reflects the steering without re-applying it each step.
        for layer_idx, sl, contribution in interventions:
            if ablation_type == 'mean':
                model.model.layers[layer_idx].self_attn.o_proj.output[..., :P, sl] = contribution
            else:
                model.model.layers[layer_idx].self_attn.o_proj.output[..., :P, sl] += contribution

        generated = model.generator.output.save()
    return generated

def decode_responses(model, inputs, originals, edited, base, answers=None):
    decoded = []
    for i in range(len(originals)):
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
