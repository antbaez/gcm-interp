"""
Summarize judge accuracy results into a CSV.

Reads judge_ratings.jsonl, fluency_ratings.jsonl, and relevance_ratings.jsonl
from the workdirs directory tree and computes per-condition w_rf, wo_rf, and
fluency-only (no relevance check) pass rates.

Usage:
    python summarize_results.py [--workdirs_dir DIR] [--accuracy_dir DIR]
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import BASE_DIR

WORKDIRS_JSONL = BASE_DIR / "judge-evals" / "workdirs"
ACCURACY_DIR   = BASE_DIR / "judge-evals" / "accuracy"


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _process_judge_file(judge_path: Path, workdirs_dir: Path, stream_mode: str | None) -> dict | None:
    rel   = judge_path.relative_to(workdirs_dir)
    parts = rel.parts
    # Expected layout: (model, from_to, norm_mode, stream_mode, scope, steering_type, exp_dir, filename)
    if len(parts) < 8:
        return None

    model         = parts[0]
    from_to       = parts[1]
    norm_mode     = parts[2]   # "normalized" or "unnormalized"
    stream_mode_i = parts[3]   # "attention" or "residuals"
    scope         = parts[4]   # "global" or "local"
    steering_type = parts[5]   # e.g. "positional", "last-token"
    exp_dir       = parts[6]   # e.g. "N=10_steer_layer=all_harmless-test" (val)
                                # or "N=225_steer_layer=all_harmless-heldout-test" (held-out)
    split         = "test" if exp_dir.endswith("-heldout-test") else "val"

    if stream_mode is not None and stream_mode_i != stream_mode:
        return None

    ft = re.match(r"^from_(.+)_to_(.+)$", from_to)
    if not ft:
        return None
    source, base = ft.group(1), ft.group(2)

    judge_recs = list(_read_jsonl(judge_path))
    if not judge_recs:
        return None

    flu_path = judge_path.parent / "fluency_ratings.jsonl"
    rel_path = judge_path.parent / "relevance_ratings.jsonl"
    flu_recs = list(_read_jsonl(flu_path)) if flu_path.exists() else []
    rel_recs = list(_read_jsonl(rel_path)) if rel_path.exists() else []

    first = judge_recs[0]
    N = first.get("N")
    # Ratings written before the topk -> layer rename carry the swept value under
    # the old key; the value itself is the same layer index either way.
    layer_raw = first.get("layer", first.get("topk"))
    if N is None or layer_raw is None:
        return None

    # Global runs use `layer=all` (every layer steered), which has no single
    # index, so `layer` stays None there.
    try:
        layer = int(round(float(layer_raw)))
    except (TypeError, ValueError):
        layer = None

    flu_by_query = {r.get("data_path_query"): r.get("judge_rating") for r in flu_recs}
    rel_by_query = {r.get("data_path_query"): r.get("judge_rating") for r in rel_recs}

    w_passes      = []
    wo_passes     = []
    no_rel_passes = []
    for rec in judge_recs:
        jp_rating = rec.get("judge_rating")
        jp_pass   = bool(jp_rating == 5)
        query     = rec.get("data_path_query", "")
        flu       = flu_by_query.get(query)
        rel       = rel_by_query.get(query)
        w_passes.append(jp_pass and flu == 2 and rel == 2)
        wo_passes.append(jp_pass)
        no_rel_passes.append(jp_pass and flu == 2)

    n = len(w_passes)
    condition = steering_type
    # N may be fractional for --global runs (e.g. 0.5); int(N) would
    # truncate it, so cast to float and only narrow to int when exact.
    N_val = float(N)
    if N_val.is_integer():
        N_val = int(N_val)
    return dict(
        model=model, source=source, base=base,
        dataset=f"{source} → {base}",
        N=N_val, layer=layer,
        scope=scope,
        norm_mode=norm_mode,
        stream_mode=stream_mode_i,
        split=split,
        steering_type=steering_type,
        condition=condition,
        pass_rate=sum(w_passes) / n,
        pass_rate_wo_rf=sum(wo_passes) / n,
        pass_rate_no_rel=sum(no_rel_passes) / n,
    )


def collect_records(workdirs_dir: Path, stream_mode: str | None = None, max_workers: int = 32) -> pd.DataFrame:
    judge_paths = sorted(workdirs_dir.rglob("judge_ratings.jsonl"))

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process_judge_file, p, workdirs_dir, stream_mode) for p in judge_paths]
        for future in tqdm(futures, desc="Reading rating files", unit="file"):
            record = future.result()
            if record is not None:
                records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdirs_dir", default=str(WORKDIRS_JSONL))
    parser.add_argument("--accuracy_dir", default=str(ACCURACY_DIR))
    parser.add_argument("--stream_mode", default=None,
                        help="Restrict to 'attention' or 'residuals'; omit for both")
    args = parser.parse_args()

    workdirs_dir = Path(args.workdirs_dir)
    accuracy_dir = Path(args.accuracy_dir)
    accuracy_dir.mkdir(parents=True, exist_ok=True)

    df = collect_records(workdirs_dir, stream_mode=args.stream_mode)
    if df.empty:
        print("No rating files found.")
        return

    # A stream-filtered summary gets its own file: the two streams are summarized
    # by separate runs, and sharing one filename would make whichever ran last
    # clobber the other's rows. The unfiltered run keeps the combined name.
    csv_path = accuracy_dir / (
        "results_summary.csv" if args.stream_mode is None
        else f"results_summary_{args.stream_mode}.csv"
    )
    df.sort_values(["model", "dataset", "norm_mode", "stream_mode", "split", "steering_type", "N", "layer"]) \
      .to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
