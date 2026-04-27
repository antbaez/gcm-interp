"""
Compute per-condition accuracies from judge outputs.

Reads the three per-workdir rating files (fluency_ratings.jsonl,
relevance_ratings.jsonl, judge_ratings.jsonl), merges them on row-level
metadata, and writes per-condition accuracy JSONs into an accuracy/ directory
tree.

Called directly by run_judge.py (phase 3). Can also be run standalone:
    python compute_accuracies.py --workdir judge-evals/workdirs/...
    python compute_accuracies.py [--jp_path FILE] [--flu_path FILE] [--rel_path FILE]
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import ROW_KEY_COLS, GROUP_COLS, load_jsonl_or_json


# ---------------------------------------------------------------------------
# Rating extraction
# ---------------------------------------------------------------------------

def extract_first_int(text) -> int | None:
    """Extract the first integer from judge output text."""
    if not isinstance(text, str):
        return None
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None



def fix_empty_response_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set jp_rating = 1 for any row where post-intervention-response is empty.
    Empty generations are never successful steers regardless of task.
    Rating 1 is the mid-scale value for both the 1-5 tasks and the 1-3
    sycophancy scale, so it safely avoids being counted as a pass on either.
    """
    if "post-intervention-response" not in df.columns or "jp_rating" not in df.columns:
        return df

    is_empty = df["post-intervention-response"].astype(str).str.strip() == ""
    if not is_empty.any():
        return df

    df = df.copy()
    df.loc[is_empty, "jp_rating"] = 1
    print(f"  Rating overrides: {is_empty.sum()} empty responses → jp_rating=1")
    return df


# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------

def load_or_cache(json_path: str, cache_name: str) -> pd.DataFrame:
    """Load from JSON, or use CSV cache if available.
    Cache file is placed next to the JSON source.
    """
    parent = os.path.dirname(json_path) or "."
    csv_cache = os.path.join(parent, cache_name)

    if os.path.exists(csv_cache):
        print(f"Loading cached {csv_cache}")
        return pd.read_csv(csv_cache)

    if not os.path.exists(json_path):
        print(f"Not found: {json_path}, returning empty DataFrame")
        return pd.DataFrame()

    print(f"Loading {json_path}")
    items = load_jsonl_or_json(json_path)
    df = pd.DataFrame(items)

    before = len(df)
    df = df.drop_duplicates()
    print(f"Loaded {before} rows, {len(df)} after deduplication")

    df.to_csv(csv_cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Shared compute logic
# ---------------------------------------------------------------------------

def _compute_and_write(
    jp_df: pd.DataFrame,
    rf_flu_df: pd.DataFrame,
    rf_rel_df: pd.DataFrame,
    output_dir: str,
):
    """
    Merge ratings DataFrames, group by condition, and write per-condition
    accuracy JSON files into output_dir.

    Called by both main() (batch mode) and compute_accuracy_for_workdir().
    """
    if jp_df.empty and rf_flu_df.empty and rf_rel_df.empty:
        print("No judge outputs found. Nothing to compute.")
        return

    # Extract / rename ratings
    if not jp_df.empty:
        jp_df["jp_rating"] = jp_df["judge_output"].apply(extract_first_int)

    if not rf_flu_df.empty and "judge_rating" in rf_flu_df.columns:
        rf_flu_df = rf_flu_df.rename(columns={"judge_rating": "fluency_rating"})

    if not rf_rel_df.empty and "judge_rating" in rf_rel_df.columns:
        rf_rel_df = rf_rel_df.rename(columns={"judge_rating": "relevance_rating"})


    # Override jp_rating=1 for all empty post-intervention responses
    if not jp_df.empty:
        print("Applying empty response rating overrides...")
        jp_df = fix_empty_response_ratings(jp_df)

    print("Merging dataframes on metadata columns...")
    if not jp_df.empty:
        available_keys = [c for c in ROW_KEY_COLS if c in jp_df.columns]
        merged = jp_df[available_keys + ["jp_rating"]].copy()
    elif not rf_flu_df.empty:
        available_keys = [c for c in ROW_KEY_COLS if c in rf_flu_df.columns]
        merged = rf_flu_df[available_keys + ["fluency_rating"]].copy()
        merged["jp_rating"] = float("nan")
    else:
        available_keys = [c for c in ROW_KEY_COLS if c in rf_rel_df.columns]
        merged = rf_rel_df[available_keys + ["relevance_rating"]].copy()
        merged["jp_rating"] = float("nan")

    if "fluency_rating" not in merged.columns:
        if not rf_flu_df.empty and "fluency_rating" in rf_flu_df.columns:
            flu_keys = [c for c in ROW_KEY_COLS if c in rf_flu_df.columns]
            merged = merged.merge(
                rf_flu_df[flu_keys + ["fluency_rating"]], on=flu_keys, how="left"
            )
        else:
            merged["fluency_rating"] = float("nan")

    if "relevance_rating" not in merged.columns:
        if not rf_rel_df.empty and "relevance_rating" in rf_rel_df.columns:
            rel_keys = [c for c in ROW_KEY_COLS if c in rf_rel_df.columns]
            merged = merged.merge(
                rf_rel_df[rel_keys + ["relevance_rating"]], on=rel_keys, how="left"
            )
        else:
            merged["relevance_rating"] = float("nan")

    # Compute jp_pass:
    #   - sycophancy tasks use a 1-3 scale; pass = 3
    #   - all other tasks use a 1-5 scale; pass = 5
    has_jp = "jp_rating" in merged.columns and merged["jp_rating"].notna().any()
    if has_jp:
        is_syco = merged["SOURCE"].str.contains("sycophancy", na=False)
        merged["jp_pass"] = (
            (is_syco  & (merged["jp_rating"] == 3)) |
            (~is_syco & (merged["jp_rating"] == 5))
        )

    os.makedirs(output_dir, exist_ok=True)
    ratings_path = os.path.join(output_dir, "merged_ratings.csv")
    merged.to_csv(ratings_path, index=False)
    print(f"Saved {ratings_path} ({len(merged)} rows)")

    print("Computing per-condition accuracies...")
    available_group = [c for c in GROUP_COLS if c in merged.columns]
    grouped = merged.groupby(available_group)

    for _, group in tqdm(grouped):
        row = group.iloc[0]
        base_dir = os.path.join(
            output_dir,
            str(row["MODEL_ID"]),
            f"from_{row['SOURCE']}_to_{row['BASE']}",
            str(row["METHOD"]),
            str(row["EVAL_SUB_DIR"]),
            str(row["STEER_SUB_DIR"]),
        )
        os.makedirs(base_dir, exist_ok=True)
        fn_base = (
            f"{row['N']}_{row['REPS']}_{row['STEERING_METHOD']}_topk_{row['topk']}"
        )

        if has_jp:
            acc_without = float(group["jp_pass"].mean())
            path_wo = os.path.join(
                base_dir, f"{fn_base}_gen_accuracy_wo_rf.json.accuracy.json"
            )
            with open(path_wo, "w") as f:
                json.dump({"q1": acc_without}, f, indent=2)

            acc_with = float(
                (
                    group["jp_pass"]
                    & (group["fluency_rating"] == 2)
                    & (group["relevance_rating"] == 2)
                ).mean()
            )
            path_w = os.path.join(
                base_dir, f"{fn_base}_gen_accuracy_w_rf.json.accuracy.json"
            )
            with open(path_w, "w") as f:
                json.dump({"q1": acc_with}, f, indent=2)
        else:
            acc_rf = float(
                (
                    (group["fluency_rating"] == 2)
                    & (group["relevance_rating"] == 2)
                ).mean()
            )
            path_rf = os.path.join(base_dir, f"{fn_base}_gen_rf_pass_rate.json")
            with open(path_rf, "w") as f:
                json.dump({"rf_pass_rate": acc_rf}, f, indent=2)

    print("Done.")


def compute_accuracy_for_workdir(
    workdir: Path,
    accuracy_dir: Path,
    skip_judge: bool = False,
):
    """
    Read per-file ratings from workdir and compute per-condition accuracy files.

    Expects these files in workdir (written by run_judge.py phase 2):
      - fluency_ratings.jsonl
      - relevance_ratings.jsonl
      - judge_ratings.jsonl   (only if skip_judge is False)
    """
    flu_path  = workdir / "fluency_ratings.jsonl"
    rel_path  = workdir / "relevance_ratings.jsonl"
    jp_path   = workdir / "judge_ratings.jsonl"

    jp_df     = load_or_cache(str(jp_path),  "jp_ratings.csv")  if (not skip_judge and jp_path.exists())  else pd.DataFrame()
    rf_flu_df = load_or_cache(str(flu_path), "flu_ratings.csv") if flu_path.exists() else pd.DataFrame()
    rf_rel_df = load_or_cache(str(rel_path), "rel_ratings.csv") if rel_path.exists() else pd.DataFrame()

    _compute_and_write(jp_df, rf_flu_df, rf_rel_df, str(accuracy_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute accuracies from judge outputs")
    p.add_argument("--workdir", default=None,
                    help="Per-file workdir containing *_ratings.jsonl files "
                         "(reads fluency/relevance/judge_ratings.jsonl from there)")
    p.add_argument("--jp_path",
                    default="judge_prompts.judge_prompt.judge_outputs.json",
                    help="Path to behavioral judge outputs (JSONL); ignored when --workdir is set")
    p.add_argument("--flu_path",
                    default="relevance_fluency_prompts.fluency_prompt.judge_outputs.json",
                    help="Path to fluency judge outputs (JSONL); ignored when --workdir is set")
    p.add_argument("--rel_path",
                    default="relevance_fluency_prompts.relevance_prompt.judge_outputs.json",
                    help="Path to relevance judge outputs (JSONL); ignored when --workdir is set")
    p.add_argument("--output_dir", default="accuracy",
                    help="Root directory for accuracy JSON outputs")
    return p.parse_args()


def main():
    args = parse_args()

    if hasattr(args, "workdir") and args.workdir:
        compute_accuracy_for_workdir(
            Path(args.workdir), Path(args.output_dir),
            skip_judge=not Path(args.workdir, "judge_ratings.jsonl").exists(),
        )
        return

    jp_df     = load_or_cache(args.jp_path,  "jp_ratings.csv")
    rf_flu_df = load_or_cache(args.flu_path, "rf_fluency_ratings.csv")
    rf_rel_df = load_or_cache(args.rel_path, "rf_relevance_ratings.csv")

    _compute_and_write(jp_df, rf_flu_df, rf_rel_df, args.output_dir)


if __name__ == "__main__":
    main()