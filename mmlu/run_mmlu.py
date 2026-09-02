import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'judge-evals'))

import numpy as np
import torch
from utils.config import Config
from utils.model_handler import ModelHandler
import mmlu_scoring as scoring
from selection_utils import scan_conditions

WORKDIRS_ROOT = os.path.join(REPO_ROOT, 'judge-evals', 'workdirs')
NORM_MODE = 'normalized'


def parse_mmlu_args():
    parser = argparse.ArgumentParser(description='MMLU steering-corruption eval', add_help=False)
    parser.add_argument('--fraction', type=float, default=0.1,
                         help='Fraction of MMLU questions to sample per subject')
    parser.add_argument('--mmlu_data', type=str, default='mmlu/data/mmlu_test.jsonl')
    parser.add_argument('--best_configs', type=str, default='judge-evals/best_configs.json')
    parser.add_argument('--mmlu_results_dir', type=str, default='mmlu/results')
    parser.add_argument('--mmlu_seed', type=int, default=42)
    parser.add_argument('--mmlu_batch_size', type=int, default=25)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                         help="Which steering configs to score: 'test' (default) scores the single "
                              "best config per method, pinned in best_configs.json; 'val' scores every "
                              "condition from the validation sweep instead, one MMLU accuracy per "
                              "(N, layer) actually judged during validation")
    parser.add_argument('--print_examples', action='store_true',
                         help='Print 5 random example question/prediction pairs after each eval')
    known, remaining = parser.parse_known_args()
    # Config() below re-parses sys.argv with its own argparser, which would
    # error on our flags above — strip them out first.
    sys.argv = [sys.argv[0]] + remaining
    return known


def main():
    mmlu_args = parse_mmlu_args()

    # Seed once, up front, so sample_mmlu()'s subject loop draws from one
    # continuously-advancing RNG stream instead of reseeding per subject.
    random.seed(mmlu_args.mmlu_seed)
    np.random.seed(mmlu_args.mmlu_seed)
    torch.manual_seed(mmlu_args.mmlu_seed)
    torch.cuda.manual_seed_all(mmlu_args.mmlu_seed)

    print('Parsing config...')
    config = Config()
    # -source/-base are nargs='+' lists; run_mmlu_eval.sh always passes exactly
    # one of each, matching run.py's per-dataset unwrapping.
    config.args.source = config.args.source[0]
    config.args.base = config.args.base[0]

    print('Loading model...')
    model_handler = ModelHandler(config)
    model = model_handler.model
    model.eval()

    scope = 'global' if config.args.global_steer else 'local'

    model_name = config.args.model_id.split('/')[-1]
    source, base = config.args.source, config.args.base
    task = f"from_{source}_to_{base}"
    config.set_output_prefix()

    rows_by_subject = scoring.load_mmlu(mmlu_args.mmlu_data)
    sampled_rows = scoring.sample_mmlu(rows_by_subject, mmlu_args.fraction)
    print(f"MMLU: {len(sampled_rows)} sampled questions across {len(rows_by_subject)} subjects "
          f"(fraction={mmlu_args.fraction}, seed={mmlu_args.mmlu_seed})")

    answer_token_ids = scoring.resolve_answer_token_ids(model_handler.tokenizer)
    print(f"Answer letter token ids: {dict(zip(scoring.LETTERS, answer_token_ids))}")

    batches = scoring.build_batches(
        model_handler.tokenizer, model_handler.template_kwargs, config.args.device,
        sampled_rows, mmlu_args.mmlu_batch_size,
    )

    best_configs = {}
    if mmlu_args.split == 'test':
        with open(mmlu_args.best_configs) as f:
            best_configs = json.load(f)

    # --- Baseline (unsteered), once per model — dedup'd across the per-task
    # jobs that all reach this model via skip-if-exists on the output file.
    baseline_path = f"{mmlu_args.mmlu_results_dir}/{model_name}/baseline_mmlu_accuracy.json"
    if not os.path.exists(baseline_path):
        print("\n=== Baseline (unsteered) MMLU ===")
        result = scoring.evaluate_mmlu(model, batches, answer_token_ids, print_examples=mmlu_args.print_examples)
        result.update({"model": model_name, "fraction": mmlu_args.fraction, "seed": mmlu_args.mmlu_seed})
        scoring.atomic_write_json(baseline_path, result)
        print(f"Baseline accuracy: {result['accuracy']:.3f} ({result['n_correct']}/{result['n_samples']})")
    else:
        print(f"\nBaseline already exists at {baseline_path}, skipping.")

    # --- Steered configs. '--split test' (default) scores the single best config
    # per (stream, steering_type) pinned in best_configs.json; '--split val' scores
    # every (N, layer) condition actually judged during the validation sweep
    # instead, written under a 'val/' subdirectory so it never collides with (or
    # gets skipped by) the pinned test-split result at the same path. Always
    # normalized.
    task_configs = best_configs.get(model_name, {}).get(task, {})
    for stream in ('residuals', 'attention'):
        if mmlu_args.split == 'test':
            stream_configs = task_configs.get(stream, {}).get(scope, {})
            conditions = [
                (steering_type, entry['N'], entry['layer'])
                for steering_type, entry in stream_configs.items()
            ]
        else:
            method_root = Path(WORKDIRS_ROOT) / model_name / task / NORM_MODE / stream / scope
            conditions = []
            if method_root.is_dir():
                for method_dir in sorted(p for p in method_root.iterdir() if p.is_dir()):
                    steering_type = method_dir.name
                    for cond in scan_conditions(method_dir, test_file=f"{base}-test"):
                        layer = 'all' if cond['value'] == 'all' else int(cond['value'])
                        conditions.append((steering_type, cond['N'], layer))
        if not conditions:
            print(f"WARNING: no {mmlu_args.split}-split configs found for {model_name}/{task}/{stream}/"
                  f"{scope}, skipping {stream}")
            continue

        resid = stream == 'residuals'
        cache_suffix = '_resid' if resid else '_attn'
        cache_path = f"{config.get_output_prefix()}{model_name}_steering_cache_{source}{cache_suffix}.pt"
        if not os.path.exists(cache_path):
            print(f"WARNING: steering cache not found at {cache_path}, skipping {stream} configs for {task}")
            continue
        patch_activations = torch.load(cache_path, map_location=model.device)

        split_dir = '' if mmlu_args.split == 'test' else 'val/'
        for steering_type, N, layer in conditions:
            # 'global' entries pin layer='all' (every layer steered at once, per
            # eval_runner.py's global_steer mode) rather than a single swept index.
            layer_ids = list(range(model_handler.num_layers)) if layer == 'all' else [layer]
            out_path = (f"{mmlu_args.mmlu_results_dir}/{model_name}/{task}/{stream}/{scope}/"
                        f"{steering_type}/{split_dir}N={N:g}_layer={layer}_mmlu_accuracy.json")
            if os.path.exists(out_path):
                print(f"Already scored: {out_path}, skipping.")
                continue
            print(f"\n=== {task} / {stream} / {scope} / {steering_type} / {mmlu_args.split}: "
                  f"N={N:g} layer={layer} ===")
            result = scoring.evaluate_mmlu(
                model, batches, answer_token_ids,
                patch_activations=patch_activations, layer_ids=layer_ids, N=N,
                steering_type=steering_type, resid=resid, normalize=config.args.normalize,
                print_examples=mmlu_args.print_examples,
            )
            result.update({
                "model": model_name, "task": task, "stream": stream, "scope": scope,
                "steering_type": steering_type, "split": mmlu_args.split,
                "N": N, "layer": layer, "fraction": mmlu_args.fraction, "seed": mmlu_args.mmlu_seed,
            })
            scoring.atomic_write_json(out_path, result)
            print(f"Accuracy: {result['accuracy']:.3f} ({result['n_correct']}/{result['n_samples']})")


if __name__ == "__main__":
    main()
