"""Summarizes MMLU baseline-accuracy variance across --mmlu_seed values, per
--fraction, from the output of scripts/run_mmlu_seed_variance.sh.

Usage: python mmlu/analyze_seed_variance.py --model_id <full-hf-model-id>
       [--results_root mmlu/results/seed_variance] [--out_csv <path>]
"""
import argparse
import csv
import glob
import json
import os
import statistics

FIELDS = ["fraction", "n_seeds", "n_samples", "mean_accuracy", "std_accuracy",
          "min_accuracy", "max_accuracy", "range_accuracy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True,
                         help='Full model id, e.g. allenai/OLMo-2-1124-13B-DPO')
    parser.add_argument('--results_root', type=str,
                         default=os.path.join(os.path.dirname(__file__), 'results', 'seed_variance'))
    parser.add_argument('--out_csv', type=str, default=None)
    args = parser.parse_args()
    out_csv = args.out_csv or os.path.join(args.results_root, 'seed_variance_summary.csv')

    model_name = args.model_id.split('/')[-1]
    pattern = os.path.join(args.results_root, 'frac=*', 'seed=*', model_name, 'baseline_mmlu_accuracy.json')

    by_fraction = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            result = json.load(f)
        by_fraction.setdefault(result['fraction'], []).append(result)

    if not by_fraction:
        print(f"No baseline_mmlu_accuracy.json files found under {pattern}")
        return

    rows = []
    header = f"{'fraction':>10} {'n_seeds':>8} {'n_samples':>10} {'mean_acc':>10} {'std_acc':>9} {'min_acc':>9} {'max_acc':>9} {'range':>9}"
    print(header)
    for fraction in sorted(by_fraction):
        entries = by_fraction[fraction]
        accs = [e['accuracy'] for e in entries]
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
        min_acc, max_acc = min(accs), max(accs)
        n_samples = entries[0]['n_samples']
        print(f"{fraction:>10} {len(entries):>8} {n_samples:>10} {mean_acc:>10.4f} "
              f"{std_acc:>9.4f} {min_acc:>9.4f} {max_acc:>9.4f} {(max_acc - min_acc):>9.4f}")
        rows.append({
            "fraction": fraction, "n_seeds": len(entries), "n_samples": n_samples,
            "mean_accuracy": mean_acc, "std_accuracy": std_acc,
            "min_accuracy": min_acc, "max_accuracy": max_acc, "range_accuracy": max_acc - min_acc,
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tmp_path = f"{out_csv}.tmp"
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, out_csv)
    print(f"\nWrote summary to {out_csv}")


if __name__ == "__main__":
    main()
