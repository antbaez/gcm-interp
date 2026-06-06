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
    layer_ids = topk_df['layer'].unique()

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
        with model.all():
            # nnsight re-runs this block once per forward
            # call and yields a concrete tensor on .output access, so a plain Python
            # shape check reliably separates the two phases:
            #   prefill — seq_len == full prompt length (> 1)
            #   decode  — seq_len == 1 (one new token; KV cache handles the rest)
            #
            # We only steer the prefill. Steering o_proj at layer L modifies the
            # residual stream, which is the input to layer L+1. That means the K/V
            # projections at all deeper layers are computed from the steered residual,
            # so the KV cache that is built during prefill already reflects the
            # steering. Decode steps attend over that steered cache with no extra work.
            if model.model.layers[layer_ids[0]].self_attn.o_proj.output.shape[1] == 1:
                pass  # decode step — nothing to do
            else:
                for layer_idx in layer_ids:
                    head_ids_for_layer = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
                    layer = model.model.layers[layer_idx]
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

                        if ablation_type == 'mean':
                            layer.self_attn.o_proj.output[..., :patch_activations.shape[1], sl] = N * sv
                        else:
                            layer.self_attn.o_proj.output[..., :patch_activations.shape[1], sl] += N * sv

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
