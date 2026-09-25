from asyncio import log
# from setup import set_seed
from eval.activations import steering_reps_cache
from eval.generation import select_gen_qs_toks, generate_with_patches, decode_responses
import hashlib
import os
import gc
import json
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

def baseline_cache_path(config):
    """On-disk baseline for the current dataset + eval split, beside its config.yml.

    Lives at the task-dir level (not under <norm>/<stream>/<scope>/<method>/) and
    doesn't end in `_gen.json`, so neither the judge's gen-file discovery nor
    select_best_config.py's held-out pruning ever picks it up.
    """
    return os.path.join(config.get_output_prefix(), "baseline", f"baseline_{config.args.test_dataset}.json")


def _baseline_key(config, data_handler):
    """Everything the unsteered outputs depend on, so a stale cache is never reused."""
    input_ids = select_gen_qs_toks(config, data_handler)['input_ids']
    return {
        "model_id": config.args.model_id,
        "test_dataset": config.args.test_dataset,
        "max_new_tokens": config.args.max_new_tokens,
        # Batches are left-padded to their longest prompt, so batch composition
        # is part of what the greedy outputs depend on.
        "batch_size": config.args.batch_size,
        "input_hash": hashlib.sha1(input_ids.cpu().numpy().tobytes()).hexdigest(),
        "n": int(input_ids.shape[0]),
    }


def generate_baseline(config, data_handler, model_handler):
    """Unsteered generation on the eval test set. Independent of steering_type,
    so callers looping over steering types should compute this once per
    dataset and pass it into run_eval() rather than letting each call redo it.

    The outputs are saved to baseline_cache_path() and reloaded on later runs
    whenever the model, split, prompts, max_new_tokens and batch size all match.
    """
    cache_path = baseline_cache_path(config)
    key = _baseline_key(config, data_handler)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("key") == key and len(cached.get("outputs", [])) == key["n"]:
                print(f"\nBASELINE GENERATION — skipped, loaded {cache_path}")
                return cached["outputs"]
            print(f"\nBaseline cache {cache_path} doesn't match this run's settings; regenerating")
        except (OSError, ValueError) as e:
            print(f"\nCouldn't read baseline cache {cache_path} ({e}); regenerating")

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

    # Write to a temp file and rename, so an interrupted run never leaves a
    # truncated cache behind for the next one to trip over.
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump({"key": key, "outputs": original_outputs}, f)
    os.replace(tmp_path, cache_path)
    print(f"Saved baseline: {cache_path}")
    return original_outputs

def run_eval(config, data_handler, model_handler, batch_handler, N=None, original_outputs=None, baseline_fn=None):
    """Steered generation for every (N, layer) condition of the current steering_type.

    The baseline and the steering vector are only produced once a condition
    actually needs them, so a run whose gen files all exist does no generation.
    `baseline_fn` (e.g. a per-dataset memoized generate_baseline) is used instead
    of calling generate_baseline() directly when given.
    """
    # set_seed()
    print(f"Starting evaluation — task: {config.args.source} -> {config.args.base}, test: {config.args.test_dataset}, N={config.args.steering_n}")

    model = model_handler.model
    resid = getattr(config.args, 'resid', False)
    global_steer = getattr(config.args, 'global_steer', False)
    # Whole layers are steered in both streams: a single-layer sweep over
    # -layer_range_start/-layer_range_end (default) or every layer at once (--global).
    layer_sweep = not global_steer
    ablation = data_handler.config.args.ablation

    def get_baseline():
        nonlocal original_outputs
        if original_outputs is None:
            original_outputs = (baseline_fn() if baseline_fn is not None
                                else generate_baseline(config, data_handler, model_handler))
        return original_outputs

    patching_reps = None
    def get_patching_reps():
        nonlocal patching_reps
        if patching_reps is None:
            patching_reps = load_patching_reps(data_handler, model_handler)
        return patching_reps
    reps_types = ['targeted']

    steering_coverage = None
    if config.args.steering_type == 'weighted-pos':
        steering_coverage = data_handler.steering_qs_toks['add']['attention_mask'].float().mean(0)
        print(f"[eval] weighted-pos coverage: {steering_coverage.shape[0]} positions, "
              f"mean={steering_coverage.mean():.4f}, min={steering_coverage.min():.4f}")

    if N is not None:
        config.args.N = N

    # Decide what the inner sweep ranges over. Layer sweeps steer one layer at a
    # time over [-layer_range_start, -layer_range_end) of all layers; the swept
    # value is written into the filename as `layer=<idx>`.
    sweep_axis = 'layer'
    if layer_sweep:
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
    else:
        # Every layer is steered at once, so there is no swept index; the
        # filename slot is a fixed `layer=all` sentinel.
        sweep_vals = ['all']

    decoded_responses = {}
    pre_patch_logits = None
    model.eval()
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

                    # A current-format file that already pairs every response with
                    # its baseline needs nothing: skip without touching the baseline.
                    old_key = f'old_{config.args.base}'
                    if existing_stem == new_stem and all(old_key in item for item in decoded_responses[reps_type][sweep_val]):
                        print(f"Skipping evaluation for {slot}, N={config.args.N} as gen files already exist.")
                        continue

                    # Legacy stem or missing baseline field: rewrite it under the
                    # current stem with the baseline filled in (loaded from disk
                    # when cached).
                    original_outputs = get_baseline()
                    for item_iix, item in enumerate(decoded_responses[reps_type][sweep_val]):
                        query = item['query']
                        item[f'old_{config.args.base}'] = model.tokenizer.decode(original_outputs[item_iix], skip_special_tokens=True).split(query)[-1]
                    gen_file = f"{new_stem}.txt"
                    save_prompt_responses(decoded_responses[reps_type][sweep_val], gen_file)
                    print(f"Skipping evaluation for {slot}, N={config.args.N} as gen files already exist.")
                    continue
                decoded_responses[reps_type][sweep_val] = []
                gen_file = f"{new_stem}.txt"
                original_outputs = get_baseline()
                patching_reps = get_patching_reps()

                if layer_sweep:
                    steer_layers = [sweep_val]
                else:
                    steer_layers = list(range(model_handler.num_layers))

                batch_handler = BatchHandler(config, data_handler)
                len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
                steer_total = len(range(0, len_gen_qs, config.args.batch_size))
                for batch_num, idx in enumerate(range(0, len_gen_qs, config.args.batch_size), start=1):
                    _t0 = time.time()
                    gen_qs_toks = select_gen_qs_toks(config, batch_handler)
                    edited_outputs = generate_with_patches(model, gen_qs_toks, patching_reps, steer_layers, config.args.N, max_new_tokens=config.args.max_new_tokens, normalize=config.args.normalize, steering_type=config.args.steering_type, resid=resid, coverage=steering_coverage)
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
