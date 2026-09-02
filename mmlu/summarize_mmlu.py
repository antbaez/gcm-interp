"""Aggregates every mmlu/results/**/*_mmlu_accuracy.json into one CSV.

Usage: python mmlu/summarize_mmlu.py [--results_dir mmlu/results] [--out mmlu/results/mmlu_summary.csv]
"""
import argparse
import csv
import glob
import json
import os

FIELDS = ["model", "task", "stream", "scope", "steering_type", "split", "N", "layer",
          "fraction", "n_samples", "accuracy", "baseline_accuracy", "delta"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    out_path = args.out or os.path.join(args.results_dir, 'mmlu_summary.csv')

    baselines = {}
    for path in glob.glob(os.path.join(args.results_dir, '*', 'baseline_mmlu_accuracy.json')):
        with open(path) as f:
            data = json.load(f)
        baselines[data['model']] = data['accuracy']

    rows = []
    # scope dir (local/global) used to be hardcoded 'local'; glob both now that
    # run_mmlu.py writes global-scope results too. `**` (recursive) covers both
    # the flat test-split layout (.../<steering_type>/N=..._mmlu_accuracy.json)
    # and the val-split layout one directory deeper
    # (.../<steering_type>/val/N=..._mmlu_accuracy.json).
    pattern = os.path.join(args.results_dir, '*', '*', '*', '*', '*', '**', 'N=*_layer=*_mmlu_accuracy.json')
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            data = json.load(f)
        baseline_acc = baselines.get(data['model'])
        rows.append({
            "model": data['model'], "task": data['task'], "stream": data['stream'],
            # Older result files predate the scope/split fields and are local/test-only.
            "scope": data.get('scope', 'local'), "split": data.get('split', 'test'),
            "steering_type": data['steering_type'], "N": data['N'], "layer": data['layer'],
            "fraction": data['fraction'], "n_samples": data['n_samples'], "accuracy": data['accuracy'],
            "baseline_accuracy": baseline_acc,
            "delta": (data['accuracy'] - baseline_acc) if baseline_acc is not None else None,
        })

    os.makedirs(args.results_dir, exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
