"""
Prints a per-config summary table of average judge_rating values
(fluency, judge, relevance) from the judge-evals workdir tree.

Usage:
    python judge-evals/analyze_judge_results.py [--workdirs_root <path>]

Default workdirs_root: judge-evals/workdirs  (relative to repo root)
"""

import argparse
import json
import re
import sys
from pathlib import Path


METRICS = ["fluency", "judge", "relevance"]


def avg(vals):
    return round(sum(vals) / len(vals), 3) if vals else None


def load_ratings(config_dir, metric):
    fpath = config_dir / f"{metric}_ratings.jsonl"
    ratings = []
    if fpath.exists():
        for line in fpath.read_text().splitlines():
            if line.strip():
                v = json.loads(line).get("judge_rating")
                if v is not None:
                    ratings.append(float(v))
    return ratings


def analyze_task(task_dir):
    """Return sorted list of row dicts for one task directory."""
    eval_root = task_dir / "atp" / "eval"
    if not eval_root.exists():
        return []

    rows = []
    for stype_dir in sorted(eval_root.iterdir()):
        for config_dir in sorted(stype_dir.iterdir()):
            m = re.match(r"steer_\w+_N=(\d+)_k=([\d.]+)", config_dir.name)
            if not m:
                continue
            N, k = int(m.group(1)), float(m.group(2))
            row = {"type": stype_dir.name, "N": N, "k": k}
            for metric in METRICS:
                ratings = load_ratings(config_dir, metric)
                row[metric] = avg(ratings)
                row[f"{metric}_n"] = len(ratings)
            rows.append(row)

    rows.sort(key=lambda r: (r["type"], r["N"], r["k"]))
    return rows


def fmt(v):
    return f"{v:>8.3f}" if v is not None else "     N/A"


def table_lines(task_name, rows):
    out = []
    if not rows:
        out.append("  (no data)")
        out.append("")
        return out

    header = f"  {'type':<12} {'N':>4} {'k':>6}  {'fluency':>8} {'judge':>8} {'relevance':>9}  {'n':>4}"
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))

    cur_type = None
    for r in rows:
        if r["type"] != cur_type:
            if cur_type is not None:
                out.append("")
            cur_type = r["type"]
        n = r.get("judge_n", "?")
        out.append(
            f"  {r['type']:<12} {r['N']:>4} {r['k']:>6} "
            f" {fmt(r['fluency'])} {fmt(r['judge'])} {fmt(r['relevance'])}  {n:>4}"
        )
    out.append("")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdirs_root",
        default=str(Path(__file__).parent / "workdirs"),
        help="Path to judge-evals/workdirs",
    )
    parser.add_argument(
        "--accuracy_dir",
        default=None,
        help="Directory to save judge_results_analysis.txt",
    )
    args = parser.parse_args()

    workdirs_root = Path(args.workdirs_root)
    if not workdirs_root.exists():
        print(f"workdirs_root not found: {workdirs_root}", file=sys.stderr)
        sys.exit(1)

    lines = []

    found_any = False
    for model_dir in sorted(workdirs_root.iterdir()):
        for task_dir in sorted(model_dir.iterdir()):
            rows = analyze_task(task_dir)
            if not rows:
                continue
            found_any = True
            lines.append(f"\n{'=' * 60}")
            lines.append(f"  {model_dir.name}  /  {task_dir.name}")
            lines.append(f"{'=' * 60}")
            lines.extend(table_lines(task_dir.name, rows))

    if not found_any:
        print("No judge results found.", file=sys.stderr)
        sys.exit(1)

    output = "\n".join(lines)
    print(output)

    if args.accuracy_dir:
        out_path = Path(args.accuracy_dir) / "judge_results_analysis.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
