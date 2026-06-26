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
    if config.args.eval_train:
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
    print("patch activations", patch_activations.shape, steering_type)
    positional = (steering_type == 'positional')
    total_len = gen_toks['input_ids'].shape[1]
    print(f"[steering] total_len={total_len}, steering_type={steering_type}, normalize={normalize}")

    interventions = []
    for layer_idx in topk_df['layer'].unique():
        head_ids = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
        for head_idx in head_ids:
            sl = slice(DIM * head_idx, DIM * (head_idx + 1))
            if positional:
                sv = patch_activations[layer_idx][:, sl]  # [P, dim]
            else:
                sv = patch_activations[layer_idx][sl]     # [dim]
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
        print("contribution", interventions[0][2].shape)
        out_debug = _get_layers(model)[interventions[0][0]].self_attn.o_proj.output.save()
        for layer_idx, sl, contribution in interventions:
            out = _get_layers(model)[layer_idx].self_attn.o_proj.output
            # print(f"[steer] layer={layer_idx}, slice={sl}, contribution shape={contribution.shape}")
            if contribution.dim() == 1:
                if ablation_type == 'mean':
                    out[..., sl] = contribution
                else:
                    out[..., sl] = out[..., sl] + contribution
            else:
                P = contribution.shape[0]
                if total_len <= P:
                    c = contribution[-total_len:]
                    if ablation_type == 'mean':
                        out[..., sl] = c
                    else:
                        out[..., sl] = out[..., sl] + c
                else:
                    if ablation_type == 'mean':
                        out[:, -P:, sl] = contribution
                    else:
                        out[:, -P:, sl] = out[:, -P:, sl] + contribution

        generated = model.generator.output.save()
    print(f"[debug] out shape on prefill: {out_debug.shape} [batch, seq_len, hidden_dim]")
    return generated

    # no kv cache
    #     with model.all():
    #         for layer_idx, sl, contribution in interventions:
    #             out = _get_layers(model)[layer_idx].self_attn.o_proj.output
    #             # print(f"[steer] layer={layer_idx}, slice={sl}, contribution shape={contribution.shape}")
    #             if contribution.dim() == 1:
    #                 if ablation_type == 'mean':
    #                     out[..., sl] = contribution
    #                 else:
    #                     out[..., :total_len, sl] = out[..., :total_len, sl] + contribution
    #             else:
    #                 # print("positional")
    #                 P = contribution.shape[0]
    #                 if total_len <= P:
    #                     c = contribution[-total_len:]
    #                     if ablation_type == 'mean':
    #                         out[..., sl] = c
    #                     else:
    #                         out[..., sl] = out[..., sl] + c
    #                 else:
    #                     if ablation_type == 'mean':
    #                         out[:, -P:, sl] = contribution
    #                     else:
    #                         out[:, -P:, sl] = out[:, -P:, sl] + contribution

    #         generated = model.generator.output.save()
    # return generated




def decode_responses(model, inputs, originals, edited, base, answers=None):
    decoded = []
    for i in range(len(originals)):
        query = model.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
        orig = model.tokenizer.decode(originals[i], skip_special_tokens=True).split(query)[-1]
        edit = model.tokenizer.decode(edited[i], skip_special_tokens=True).split(query)[-1]
        to_append = {
            'query': query,
            f'old_{base}': orig,
            f'edit_{base}': edit,
        }
        if answers is not None:
            to_append['answer'] = answers[i]
        decoded.append(to_append)
    assert len(decoded) > 0, "No responses decoded. Check the generation process."
    print(decoded[:3])
    return decoded
