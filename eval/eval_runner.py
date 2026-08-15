from asyncio import log
# from setup import set_seed
from eval.logits_handler import load_logits, get_top_k_layer_and_head
from eval.activations import steering_reps_cache
from eval.generation import select_gen_qs_toks, generate_with_patches, decode_responses
import os
import gc
import json
import pandas as pd
from tqdm import tqdm
import sys
import torch
import time
sys.path.append('../')  # Adjust path to import modules correctly
from utils.batch_handler import BatchHandler
from utils.model_handler import ModelHandler
def load_patching_reps(data_handler, model_handler, mean=True):
    model = model_handler.model
    return steering_reps_cache(model, data_handler, mean=mean)

def save_prompt_responses(responses, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for entry in responses:
            for k, v in entry.items():
                f.write(f"{v}\n")
            f.write('-' * 40 + '\n')
    with open(path.replace('.txt', '.json'), 'w') as jf:
        json.dump(responses, jf)

def save_top_k(reps_type, config, model_handler, topk, logits, logit_metric):
    topk_df = get_top_k_layer_and_head(logits, topk)

    os.makedirs(config.get_output_prefix(), exist_ok=True)
    topk_df.to_csv(f"{config.get_output_prefix()}/{logit_metric}_{reps_type}_{topk}.csv", index=False)
    return topk_df

def generate_baseline(config, data_handler, model_handler):
    """Unsteered generation on the eval test set. Independent of steering_type,
    so callers looping over steering types should compute this once per
    dataset and pass it into run_eval() rather than letting each call redo it."""
    model = model_handler.model
    model.eval()
    batch_handler = BatchHandler(config, data_handler)
    len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
    original_outputs = []
    baseline_total = len(range(0, len_gen_qs, config.args.batch_size))
    print(f"\nBASELINE GENERATION — {baseline_total} batches")
    for batch_num, idx in enumerate(range(0, len_gen_qs, config.args.batch_size), start=1):
        _t0 = time.time()
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        with model.generate(gen_qs_toks,
        pad_token_id=model.tokenizer.eos_token_id,
        use_cache=True,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        max_new_tokens=config.args.max_new_tokens) as _:
            op = model.generator.output.save()
        original_outputs += op.cpu().numpy().tolist()
        batch_handler.update()
        print(f"  Baseline batch {batch_num}/{baseline_total} — done in {time.time()-_t0:.1f}s")
    return original_outputs

def run_eval(config, data_handler, model_handler, batch_handler, which_patch, topk_vals=None, N=None, original_outputs=None):
    # set_seed()
    print(f"Starting evaluation — task: {config.args.source} -> {config.args.base}, test: {config.args.test_dataset}, N={config.args.steering_n}, topk={config.args.topk_vals}")

    model = model_handler.model
    resid = getattr(config.args, 'resid', False)
    global_steer = getattr(config.args, 'global_steer', False)
    # Head localization (top-k ATP selection) only happens in attention mode and not --global.
    # Otherwise whole layers are steered: a single-layer sweep over
    # -layer_range_start/-layer_range_end (default) or every layer at once (--global).
    head_selection = (not resid) and (not global_steer)
    layer_sweep = (not global_steer) and (not head_selection)
    if head_selection:
        if os.path.exists(f"{config.get_output_prefix()}/heads/numerator_1_{which_patch}.pt"):
            logits = torch.load(f"{config.get_output_prefix()}/heads/numerator_1_{which_patch}.pt")
        else:
            logits = load_logits(config, data_handler, which_patch, model_handler)
    else:
        logits = None
    patching_reps = load_patching_reps(data_handler, model_handler)
    ablation = data_handler.config.args.ablation
    reps_types = ['targeted']

    steering_coverage = None
    if config.args.steering_type == 'weighted-pos':
        steering_coverage = data_handler.steering_qs_toks['add']['attention_mask'].float().mean(0)
        print(f"[eval] weighted-pos coverage: {steering_coverage.shape[0]} positions, "
              f"mean={steering_coverage.mean():.4f}, min={steering_coverage.min():.4f}")

    if topk_vals is None:
        topk_vals = config.args.topk_vals
    if N is not None:
        config.args.N = N

    # Decide what the inner sweep ranges over. Layer sweeps steer one layer at a
    # time over [-layer_range_start, -layer_range_end) of all layers; the swept
    # value is written into the filename as `layer=<idx>`. Global / head-selection
    # runs keep `topk=<val>`.
    if layer_sweep:
        sweep_axis = 'layer'
        explicit_layers = getattr(config.args, 'layers', None)
        if explicit_layers:
            # A pinned layer set (e.g. the one chosen on validation) replaces the
            # sweep entirely, so the held-out split is only ever generated once.
            sweep_vals = list(explicit_layers)
            print(f"Steering pinned layers: {sweep_vals}")
        else:
            num_layers = model_handler.num_layers
            range_start = getattr(config.args, 'layer_range_start', 0.0)
            range_end = getattr(config.args, 'layer_range_end', 2/3)
            layer_lo, layer_hi = round(num_layers * range_start), round(num_layers * range_end)
            sweep_vals = list(range(layer_lo, layer_hi))
            print(f"Single-layer sweep over layers {layer_lo}..{layer_hi - 1} "
                  f"({len(sweep_vals)} layers, range=[{range_start:g}, {range_end:g}))")
    elif global_steer:
        # Every layer is steered at once, so there is no swept index; the
        # filename slot is a fixed `layer=all` sentinel.
        sweep_axis = 'layer'
        sweep_vals = ['all']
    else:
        sweep_axis = 'topk'
        sweep_vals = topk_vals
    logit_metric = 'numerator_1'

    decoded_responses = {}
    pre_patch_logits = None
    model.eval()
    if original_outputs is None:
        original_outputs = generate_baseline(config, data_handler, model_handler)
    print("\nSTEERING GENERATION")
    for N in config.args.steering_n:
        config.args.N = N
        # Format N without a trailing ".0" for whole numbers (keeps existing
        # integer-N filenames unchanged) while still rendering fractional
        # values like 0.5 for --global runs.
        n_str = f"{N:g}"
        for reps_type in tqdm(reps_types, desc="Reps Types"):
            decoded_responses[reps_type] = {}
            for sweep_val in tqdm(sweep_vals, desc=f"{sweep_axis} values"):
                slot = f"{sweep_axis}={sweep_val}"
                norm_dir   = "normalized" if config.args.normalize else "unnormalized"
                stream_dir = "residuals" if resid else "attention"
                scope_dir  = "global" if global_steer else "local"
                steer_eval_dir = f"{config.get_output_prefix()}/{norm_dir}/{stream_dir}/{scope_dir}/{config.args.steering_type}"
                new_stem = f"{steer_eval_dir}/N={n_str}_{ablation}_{slot}_{config.args.test_dataset}_gen"
                old_stem = f"{steer_eval_dir}/{n_str}_{reps_type}_{ablation}_{sweep_val}_{config.args.test_dataset}_gen"
                existing_stem = (
                    new_stem if os.path.exists(f"{new_stem}.txt") and os.path.exists(f"{new_stem}.json")
                    else old_stem if os.path.exists(f"{old_stem}.txt") and os.path.exists(f"{old_stem}.json")
                    else None
                )
                if existing_stem:
                    with open(f"{existing_stem}.json", 'r') as jf:
                        decoded_responses[reps_type][sweep_val] = json.load(jf)

                    for item_iix, item in enumerate(decoded_responses[reps_type][sweep_val]):
                        query = item['query']
                        item[f'old_{config.args.base}'] = model.tokenizer.decode(original_outputs[item_iix], skip_special_tokens=True).split(query)[-1]
                    gen_file = f"{new_stem}.txt"
                    save_prompt_responses(decoded_responses[reps_type][sweep_val], gen_file)
                    print(f"Skipping evaluation for {slot}, N={config.args.N} as gen files already exist.")
                    continue
                decoded_responses[reps_type][sweep_val] = []
                gen_file = f"{new_stem}.txt"

                csv_path = f"{config.get_output_prefix()}/{logit_metric}_{reps_type}_{sweep_val}.csv"
                if head_selection and os.path.exists(gen_file) and os.path.exists(gen_file.replace('.txt', '.json')) and os.path.exists(csv_path):
                    print(f"Skipping generation as all relevant files exist.")
                    continue
                if layer_sweep:
                    topk_df = pd.DataFrame({'layer': [sweep_val]})
                elif global_steer:
                    topk_df = pd.DataFrame({'layer': list(range(model_handler.num_layers))})
                elif not os.path.exists(csv_path):
                    topk_df = save_top_k(reps_type, config, model_handler, sweep_val, logits, logit_metric)
                else:
                    topk_df = pd.read_csv(csv_path)

                batch_handler = BatchHandler(config, data_handler)
                len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
                steer_total = len(range(0, len_gen_qs, config.args.batch_size))
                for batch_num, idx in enumerate(range(0, len_gen_qs, config.args.batch_size), start=1):
                    _t0 = time.time()
                    gen_qs_toks = select_gen_qs_toks(config, batch_handler)
                    edited_outputs = generate_with_patches(model, gen_qs_toks, patching_reps, topk_df, config.args.N, model_handler.dim, max_new_tokens=config.args.max_new_tokens, normalize=config.args.normalize, steering_type=config.args.steering_type, resid=resid, coverage=steering_coverage)
                    decoded = decode_responses(model, gen_qs_toks, original_outputs[idx:idx+config.args.batch_size], edited_outputs, config.args.base)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if len(decoded_responses[reps_type][sweep_val]) == 0:
                        decoded_responses[reps_type][sweep_val] = decoded
                    else:
                        decoded_responses[reps_type][sweep_val] += decoded
                    batch_handler.update()
                    print(f"  Steering batch {batch_num}/{steer_total} — N={config.args.N}, {slot} — done in {time.time()-_t0:.1f}s")

                os.makedirs(steer_eval_dir, exist_ok=True)
                save_prompt_responses(decoded_responses[reps_type][sweep_val], gen_file)
    print("Evaluation complete.")
