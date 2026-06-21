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
    # last-token / mean caches are [layers, dim] (single direction); positional is
    # [layers, P, dim] (one direction per position). P only exists for positional.
    positional = (steering_type == 'positional')
    P = patch_activations.shape[1] if positional else None

    total_len = gen_toks['input_ids'].shape[1]
    if 'attention_mask' in gen_toks:
        prompt_starts = (gen_toks['attention_mask'] == 0).sum(dim=1).int().tolist()
    else:
        prompt_starts = [0] * gen_toks['input_ids'].shape[0]
    real_lens = [total_len - ps for ps in prompt_starts]
    print(f"[steering] total_len={total_len}, real_len={min(real_lens)}-{max(real_lens)}, P={P}")

    # Precompute the per-head steering contributions OUTSIDE the trace, so the traced
    # body below contains only plain tensor assignments. pandas indexing / proxy
    # math inside the trace gets routed through nnsight's tracing hacks and is fragile.
    interventions = []  # list of (layer_idx, slice, contribution)
    for layer_idx in topk_df['layer'].unique():
        head_ids_for_layer = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
        for head_idx in head_ids_for_layer:
            sl = slice(DIM * head_idx, DIM * (head_idx + 1))

            if positional:
                sv = patch_activations[layer_idx][:, sl]   # [P, dim]
            else:
                sv = patch_activations[layer_idx][sl]      # [dim]  (last-token or mean)

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
            out = model.model.layers[layer_idx].self_attn.o_proj.output
            for i, ps in enumerate(prompt_starts):
                if contribution.dim() == 1:
                    # last-token / mean: one direction applied to every real position.
                    if ablation_type == 'mean':
                        out[i, ps:total_len, sl] = contribution
                    else:
                        out[i, ps:total_len, sl] = out[i, ps:total_len, sl] + contribution
                else:
                    # positional right-aligned: contribution[P-1] -> last real token,
                    # contribution[0] -> leftmost cached position. Slicing the last
                    # actual_P entries (contribution[-actual_P:]) lines them up
                    # directly with the rightmost window [start_pos:total_len], so the
                    # last token always gets contribution[P-1]. If the prompt is longer
                    # than P, the uncovered tokens to the left get contribution[0]
                    # (the leftmost cached vector).
                    actual_P = min(P, total_len - ps)
                    start_pos = total_len - actual_P
                    if ablation_type == 'mean':
                        out[i, start_pos:total_len, sl] = contribution[-actual_P:]
                    else:
                        out[i, start_pos:total_len, sl] = out[i, start_pos:total_len, sl] + contribution[-actual_P:]
                    if total_len - ps > P:
                        tail = contribution[0]  # leftmost cached vector
                        if ablation_type == 'mean':
                            out[i, ps:start_pos, sl] = tail
                        else:
                            out[i, ps:start_pos, sl] = out[i, ps:start_pos, sl] + tail

                    # Previous left-aligned implementation (and used torch.flip — now removed):
                    # actual_P = min(P, total_len - ps)
                    # c = contribution[:actual_P]
                    # if ablation_type == 'mean':
                    #     out[i, ps:ps + actual_P, sl] = c
                    # else:
                    #     out[i, ps:ps + actual_P, sl] = out[i, ps:ps + actual_P, sl] + c
                    # if total_len - ps > P:
                    #     tail = contribution[-1]  # [dim]
                    #     if ablation_type == 'mean':
                    #         out[i, ps + actual_P:total_len, sl] = tail
                    #     else:
                    #         out[i, ps + actual_P:total_len, sl] = out[i, ps + actual_P:total_len, sl] + tail

        generated = model.generator.output.save()
    return generated

def decode_responses(model, inputs, originals, edited, base, answers=None):
    decoded = []
    for i in range(len(originals)):
        query = model.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
        orig = model.tokenizer.decode(originals[i], skip_special_tokens=True).split(query)[-1]
        edit = model.tokenizer.decode(edited[i], skip_special_tokens=True).split(query)[-1]
        raw_orig = model.tokenizer.decode(originals[i], skip_special_tokens=False)
        raw_edit = model.tokenizer.decode(edited[i], skip_special_tokens=False)
        to_append = {
            'query': query,
            f'old_{base}': orig,
            f'edit_{base}': edit,
            f'raw_old_{base}': raw_orig,
            f'raw_edit_{base}': raw_edit,
        }
        if answers is not None:
            to_append['answer'] = answers[i]
        decoded.append(to_append)
    assert len(decoded) > 0, "No responses decoded. Check the generation process."
    return decoded
