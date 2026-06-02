import os
from tqdm import tqdm
import torch
from pathlib import Path

def select_gen_qs_toks(config, batch_handler):
    if config.args.eval_train:
        # print("Evaluating on training set.")
        return batch_handler.base_qs_toks['desired']
    elif config.args.eval_test:
        print("Evaluating on test set. ", batch_handler.base_qs_toks['test']['input_ids'].shape[0])
        return batch_handler.base_qs_toks['test']
    elif config.args.eval_transfer:
        print("Evaluating on eval_test dataset.")
        return batch_handler.eval_transfer['queries']
    else:
        raise ValueError("Either eval_train or eval_test must be True.")
def generate_with_patches(model, gen_toks, patch_activations, topk_df, N, ablation_type, DIM, max_new_tokens=256, normalize=True, steering_pos='last-token', steering_type='add'):
    if steering_type == 'positional' and steering_pos == 'last-token':
        raise ValueError("steering_type='positional' is incompatible with steering_pos='last-token': positional steering requires all token positions.")
    patch_activations = patch_activations['desired'].to(model.device)
    layer_ids = topk_df['layer'].unique()
    head_ids = [topk_df[topk_df['layer'] == layer_idx]['neuron'].unique() for layer_idx in layer_ids]
    # print(f"Generating for ", gen_toks['input_ids'].shape, " with normalization set to ", normalize, " steering type ", steering_pos)
    with model.generate(
        gen_toks,
        pad_token_id=model.tokenizer.eos_token_id,
        use_cache=False, 
        do_sample=False,  
        top_p=None, 
        top_k=None,
        temperature=None, 
        max_new_tokens=max_new_tokens
    ) as tracer:
        with model.all():
            for layer_idx in layer_ids:
                head_ids = topk_df[topk_df['layer'] == layer_idx]['neuron'].unique()
                layer = model.model.layers[layer_idx]
                for head_idx in head_ids:
                    sl = slice(DIM * head_idx, DIM * (head_idx + 1))


                    if steering_type == 'last-token':
                        # steering vector is last token of diffmean vector
                        steering_vector = patch_activations[layer_idx][-1, sl]
                    elif steering_type == 'mean':
                        # steering vector is mean token of diffmean vector
                        steering_vector = patch_activations[layer_idx][:, sl].mean(dim=0)
                    elif steering_type == 'positional':
                        # steering vector is diffmean vector
                        steering_vector = patch_activations[layer_idx][:, sl]

                    if normalize:
                        # print('Normalizing')
                        steering_vector = steering_vector / (torch.norm(steering_vector, dim=-1, keepdim=True) + 1e-12)
                    if ablation_type == 'mean':
                        if steering_pos == 'last-token':
                            layer.self_attn.o_proj.output[..., patch_activations.shape[1] - 1, sl] = N * steering_vector
                        elif steering_pos == 'all-tokens':
                            layer.self_attn.o_proj.output[..., :patch_activations.shape[1], sl] = N * steering_vector
                    elif ablation_type == 'steer':

                        if steering_pos == 'last-token':
                            # add steering vector only to final token of prompt
                            layer.self_attn.o_proj.output[..., patch_activations.shape[1] - 1, sl] += N * steering_vector
                        elif steering_pos == 'all-tokens':
                            # add steering vector to all tokens in prompt
                            layer.self_attn.o_proj.output[..., :patch_activations.shape[1], sl] += N * steering_vector

          
          
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