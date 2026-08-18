import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.config import Config
from utils.model_handler import ModelHandler
import mmlu_scoring as scoring


def parse_mmlu_args():
    parser = argparse.ArgumentParser(description='MMLU steering-corruption eval', add_help=False)
    parser.add_argument('--fraction', type=float, default=0.1,
                         help='Fraction of MMLU questions to sample per subject')
    parser.add_argument('--mmlu_data', type=str, default='mmlu/data/mmlu_test.jsonl')
    parser.add_argument('--best_configs', type=str, default='judge-evals/best_configs.json')
    parser.add_argument('--mmlu_results_dir', type=str, default='mmlu/results')
    parser.add_argument('--mmlu_seed', type=int, default=42)
    parser.add_argument('--mmlu_batch_size', type=int, default=25)
    known, remaining = parser.parse_known_args()
    # Config() below re-parses sys.argv with its own argparser, which would
    # error on our flags above — strip them out first.
    sys.argv = [sys.argv[0]] + remaining
    return known


def main():
    mmlu_args = parse_mmlu_args()

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

    model_name = config.args.model_id.split('/')[-1]
    source, base = config.args.source, config.args.base
    task = f"from_{source}_to_{base}"
    config.set_output_prefix()

    rows_by_subject = scoring.load_mmlu(mmlu_args.mmlu_data)
    sampled_rows = scoring.sample_mmlu(rows_by_subject, mmlu_args.fraction, seed=mmlu_args.mmlu_seed)
    print(f"MMLU: {len(sampled_rows)} sampled questions across {len(rows_by_subject)} subjects "
          f"(fraction={mmlu_args.fraction}, seed={mmlu_args.mmlu_seed})")

    answer_token_ids = scoring.resolve_answer_token_ids(model_handler.tokenizer)
    print(f"Answer letter token ids: {dict(zip(scoring.LETTERS, answer_token_ids))}")

    batches = scoring.build_batches(
        model_handler.tokenizer, model_handler.template_kwargs, config.args.device,
        sampled_rows, mmlu_args.mmlu_batch_size,
    )

    with open(mmlu_args.best_configs) as f:
        best_configs = json.load(f)

    # --- Baseline (unsteered), once per model — dedup'd across the per-task
    # jobs that all reach this model via skip-if-exists on the output file.
    baseline_path = f"{mmlu_args.mmlu_results_dir}/{model_name}/baseline_mmlu_accuracy.json"
    if not os.path.exists(baseline_path):
        print("\n=== Baseline (unsteered) MMLU ===")
        result = scoring.evaluate_mmlu(model, batches, answer_token_ids)
        result.update({"model": model_name, "fraction": mmlu_args.fraction, "seed": mmlu_args.mmlu_seed})
        scoring.atomic_write_json(baseline_path, result)
        print(f"Baseline accuracy: {result['accuracy']:.3f} ({result['n_correct']}/{result['n_samples']})")
    else:
        print(f"\nBaseline already exists at {baseline_path}, skipping.")

    # --- Steered configs: every (stream, steering_type) pinned in best_configs
    # for this model/task. Scope is always local/normalized — the only held-out
    # configs that exist.
    task_configs = best_configs.get(model_name, {}).get(task, {})
    for stream in ('residuals', 'attention'):
        stream_configs = task_configs.get(stream, {})
        if not stream_configs:
            continue
        resid = stream == 'residuals'
        cache_suffix = '_resid' if resid else '_attn'
        cache_path = f"{config.get_output_prefix()}{model_name}_steering_cache_{source}{cache_suffix}.pt"
        if not os.path.exists(cache_path):
            print(f"WARNING: steering cache not found at {cache_path}, skipping {stream} configs for {task}")
            continue
        patch_activations = torch.load(cache_path, map_location=model.device)

        for steering_type, entry in stream_configs.items():
            N, layer = entry['N'], entry['layer']
            out_path = (f"{mmlu_args.mmlu_results_dir}/{model_name}/{task}/{stream}/local/"
                        f"{steering_type}/N={N:g}_layer={layer}_mmlu_accuracy.json")
            if os.path.exists(out_path):
                print(f"Already scored: {out_path}, skipping.")
                continue
            print(f"\n=== {task} / {stream} / {steering_type}: N={N:g} layer={layer} ===")
            result = scoring.evaluate_mmlu(
                model, batches, answer_token_ids,
                patch_activations=patch_activations, layer_ids=[layer], N=N,
                steering_type=steering_type, resid=resid, normalize=config.args.normalize,
            )
            result.update({
                "model": model_name, "task": task, "stream": stream, "steering_type": steering_type,
                "N": N, "layer": layer, "fraction": mmlu_args.fraction, "seed": mmlu_args.mmlu_seed,
            })
            scoring.atomic_write_json(out_path, result)
            print(f"Accuracy: {result['accuracy']:.3f} ({result['n_correct']}/{result['n_samples']})")


if __name__ == "__main__":
    main()
