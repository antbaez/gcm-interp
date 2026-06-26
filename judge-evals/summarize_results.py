"""
Summarize judge accuracy results into a CSV.

Reads judge_ratings.jsonl, fluency_ratings.jsonl, and relevance_ratings.jsonl
from the workdirs directory tree and computes per-condition w_rf and wo_rf
pass rates.

Usage:
    python summarize_results.py [--workdirs_dir DIR] [--accuracy_dir DIR]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import BASE_DIR

WORKDIRS_JSONL = BASE_DIR / "judge-evals" / "workdirs"
ACCURACY_DIR   = BASE_DIR / "judge-evals" / "accuracy"


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_records(workdirs_dir: Path) -> pd.DataFrame:
    records = []
    for judge_path in sorted(workdirs_dir.rglob("judge_ratings.jsonl")):
        rel   = judge_path.relative_to(workdirs_dir)
        parts = rel.parts
        # Expected layout: (model, from_to, norm_mode, cache_mode, steering_type, exp_dir, filename)
        if len(parts) < 7:
            continue

        model         = parts[0]
        from_to       = parts[1]
        norm_mode     = parts[2]   # "normalized" or "unnormalized"
        cache_mode    = parts[3]   # "cache" or "no_cache"
        steering_type = parts[4]   # e.g. "positional", "last-token"

        ft = re.match(r"^from_(.+)_to_(.+)$", from_to)
        if not ft:
            continue
        source, base = ft.group(1), ft.group(2)

        judge_recs = list(_read_jsonl(judge_path))
        if not judge_recs:
            continue

        flu_path = judge_path.parent / "fluency_ratings.jsonl"
        rel_path = judge_path.parent / "relevance_ratings.jsonl"
        flu_recs = list(_read_jsonl(flu_path)) if flu_path.exists() else []
        rel_recs = list(_read_jsonl(rel_path)) if rel_path.exists() else []

        first = judge_recs[0]
        N    = first.get("N")
        topk = first.get("topk")
        if N is None or topk is None:
            continue

        flu_by_query = {r.get("data_path_query"): r.get("judge_rating") for r in flu_recs}
        rel_by_query = {r.get("data_path_query"): r.get("judge_rating") for r in rel_recs}

        w_passes  = []
        wo_passes = []
        for rec in judge_recs:
            jp_rating = rec.get("judge_rating")
            jp_pass   = bool(jp_rating == 5)
            query     = rec.get("data_path_query", "")
            flu       = flu_by_query.get(query)
            rel       = rel_by_query.get(query)
            w_passes.append(jp_pass and flu == 2 and rel == 2)
            wo_passes.append(jp_pass)

        n = len(w_passes)
        condition = f"{cache_mode} / {steering_type}"
        records.append(dict(
            model=model, source=source, base=base,
            dataset=f"{source} → {base}",
            N=int(N), topk=float(topk),
            norm_mode=norm_mode,
            cache_mode=cache_mode,
            steering_type=steering_type,
            condition=condition,
            pass_rate=sum(w_passes) / n,
            pass_rate_wo_rf=sum(wo_passes) / n,
        ))

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdirs_dir", default=str(WORKDIRS_JSONL))
    parser.add_argument("--accuracy_dir", default=str(ACCURACY_DIR))
    args = parser.parse_args()

    workdirs_dir = Path(args.workdirs_dir)
    accuracy_dir = Path(args.accuracy_dir)
    accuracy_dir.mkdir(parents=True, exist_ok=True)

    df = collect_records(workdirs_dir)
    if df.empty:
        print("No rating files found.")
        return

    csv_path = accuracy_dir / "results_summary.csv"
    df.sort_values(["model", "dataset", "norm_mode", "cache_mode", "steering_type", "N", "topk"]) \
      .to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
