"""
Convert *_gen.json files to per-file evaluation CSVs.

Each gen.json produces its own CSV saved alongside it (replacing
_gen.json with _eval.csv). This keeps each condition separate for
easier debugging instead of merging everything into one giant file.

Usage:
    # Convert a single file explicitly
    python merge_outputs.py --input path/to/N_targeted_steer_X_gen.json [--output path.csv]

    # Discover and convert all matching files (one CSV per gen.json)
    python merge_outputs.py [--runs_dir DIR] [--data_dir DIR]
        [--model_name M] [--source S] [--base B] [--algos A1 A2]
        [--eval_subdir SUBDIR] [--steer_subdir SUBDIR]
"""

import argparse
import json
import math
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import BASE_DIR, RUNS_DIR, DATA_DIR, GEN_RE


def extract_path_metadata(path: str) -> dict:
    """Pull model/task/method metadata from the directory structure."""
    parts = Path(path).parts
    runs_idx = parts.index("results")

    model_id = parts[runs_idx + 1]
    from_to = parts[runs_idx + 2]
    _, source, _, base = from_to.split("_")

    method = parts[runs_idx + 3]
    valid_methods = {"acp", "atp", "atp-zero", "probes", "random"}
    if method not in valid_methods:
        raise ValueError(f"Unexpected METHOD: {method} in path: {path}")

    eval_sub_dir = parts[runs_idx + 4]
    steer_sub_dir = parts[runs_idx + 5]
    sub_dir = parts[runs_idx + 6]
    if sub_dir != "eval":
        raise ValueError(f"Unexpected SUB_DIR: {sub_dir} in path: {path}")

    filename = parts[runs_idx + 7]
    m = GEN_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    return {
        "MODEL_ID": model_id,
        "SOURCE": source,
        "BASE": base,
        "METHOD": method,
        "EVAL_SUB_DIR": eval_sub_dir,
        "STEER_SUB_DIR": steer_sub_dir,
        **m.groupdict(),
        "filename": filename,
    }


def load_test_queries(data_dir: str, model_id: str, source: str, base: str) -> list[str]:
    """Load the user-turn text from the test JSONL."""
    logits_path = f"{data_dir}/{model_id}/{source}/{base}-test.jsonl"
    queries = []
    with open(logits_path) as f:
        for line in f:
            obj = json.loads(line)
            user_content = next(
                m["content"] for m in reversed(obj["prompt"]) if m["role"] == "user"
            )
            queries.append(user_content)
    return queries


def validate_record(record: dict):
    """Raise on NaN or non-primitive values."""
    for key, value in record.items():
        if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
            raise ValueError(f"NaN value for '{key}'")
        if not isinstance(value, (str, int, float)):
            raise TypeError(
                f"Invalid type for '{key}': {type(value).__name__} ({value})"
            )


def discover_gen_files(
    runs_dir: str,
    model_name: str | None = None,
    source: str | None = None,
    base: str | None = None,
    algos: list[str] | None = None,
    eval_subdir: str | None = None,
    steer_subdir: str | None = None,
) -> list[str]:
    """
    Glob for *_gen.json files, optionally filtered by model/task/algo/eval_subdir.

    Uses single-level wildcards (*) for each path component to avoid duplicates
    that arise from recursive (**) globbing.
    """
    model_part = model_name or "*"
    task_part = f"from_{source}_to_{base}" if (source and base) else "*"
    eval_part = eval_subdir or "*"
    steer_part = steer_subdir or "*"
    algo_parts = algos or ["*"]

    gen_files = []
    for algo in algo_parts:
        pattern = (
            f"{runs_dir}/{model_part}/{task_part}/{algo}"
            f"/{eval_part}/{steer_part}/eval/*_gen.json"
        )
        gen_files.extend(glob.glob(pattern))

    # Deduplicate and filter by filename pattern
    seen = set()
    result = []
    for f in sorted(gen_files):
        if f not in seen and GEN_RE.search(Path(f).name):
            seen.add(f)
            result.append(f)
    return result


def gen_to_csv(gen_path: str, data_dir: str, output_path: str):
    """Convert a single gen.json file to a CSV with metadata columns."""
    try:
        meta = extract_path_metadata(gen_path)
    except ValueError as e:
        raise ValueError(f"Cannot process {gen_path}: {e}") from e

    model_id = meta["MODEL_ID"]
    source = meta["SOURCE"]
    base = meta["BASE"]

    with open(gen_path) as f:
        items = json.load(f)

    test_queries = load_test_queries(data_dir, model_id, source, base)

    old_key = f"old_{base}"
    edit_key = f"edit_{base}"

    records = []
    for i, item in enumerate(items):
        record = {
            "query": str(item["query"]).strip().replace("\r", "\n"),
            "post-intervention-response": str(item[edit_key]).strip().replace("\r", "\n"),
            "original-response": str(item[old_key]).strip().replace("\r", "\n"),
            "filename": meta["filename"].strip().replace("\r", "\n"),
            "data_path_query": test_queries[i].strip().replace("\r", "\n"),
            "MODEL_ID": meta["MODEL_ID"],
            "SOURCE": source,
            "BASE": base,
            "METHOD": meta["METHOD"],
            "EVAL_SUB_DIR": meta["EVAL_SUB_DIR"],
            "STEER_SUB_DIR": meta["STEER_SUB_DIR"],
            "N": meta["N"],
            "REPS": meta["REPS"],
            "STEERING_METHOD": meta["STEERING_METHOD"],
            "topk": meta["topk"],
            "TEST_FILE": meta["TEST_FILE"],
        }
        validate_record(record)
        records.append(record)

    df = pd.DataFrame(records)

    if df.isna().any().any():
        nan_rows = df[df.isna().any(axis=1)]
        raise ValueError(f"NaNs detected in dataframe!\n{nan_rows.to_string(index=False)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {output_path}  ({len(df)} rows)")


def default_csv_path(gen_path: str) -> str:
    """Derive the default CSV output path from a gen.json path."""
    p = Path(gen_path)
    stem = p.stem  # e.g. "1_targeted_steer_0.01_sycophancy-single_gen"
    if stem.endswith("_gen"):
        stem = stem[:-4]
    return str(p.parent / f"{stem}_eval.csv")


def main():
    parser = argparse.ArgumentParser(description="Convert gen.json files to per-file CSVs")
    parser.add_argument("--input", default=None,
                        help="Single gen.json file to convert")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (only used with --input)")
    parser.add_argument("--runs_dir", default=str(RUNS_DIR))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--base", default=None)
    parser.add_argument("--algos", nargs="*", default=None)
    parser.add_argument("--eval_subdir", default=None,
                        help="Filter by eval subdirectory (e.g. sycophancy-single_eval)")
    parser.add_argument("--steer_subdir", default=None,
                        help="Filter by steer subdirectory (e.g. sycophancy-long_steer)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip gen.json files whose CSV already exists")
    args = parser.parse_args()

    data_dir = args.data_dir

    if args.input:
        output = args.output or default_csv_path(args.input)
        if args.skip_existing and Path(output).exists():
            print(f"Skipping {args.input} (CSV already exists)")
            return
        gen_to_csv(args.input, data_dir, output)
        return

    # Discovery mode: convert each matching gen.json to its own CSV
    gen_files = discover_gen_files(
        args.runs_dir, args.model_name, args.source, args.base, args.algos,
        args.eval_subdir, args.steer_subdir,
    )
    print(f"Found {len(gen_files)} gen files")

    skipped = 0
    processed = 0
    errors = 0
    for gpath in gen_files:
        output = default_csv_path(gpath)
        if args.skip_existing and Path(output).exists():
            skipped += 1
            continue
        try:
            gen_to_csv(gpath, data_dir, output)
            processed += 1
        except (ValueError, FileNotFoundError) as e:
            print(f"  ERROR processing {gpath}: {e}")
            errors += 1

    print(f"\nDone: {processed} converted, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
