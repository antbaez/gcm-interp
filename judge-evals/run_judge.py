"""
Run the full judge evaluation pipeline.

Strategy:
  Phase 1 (per file, fast): convert each gen.json → CSV → build prompt CSVs
  Phase 2:
    - Single-eval files: token matching only (no model loaded)
    - Long-eval files:   load judge model ONCE, batch all prompts from all
                         workdirs into one inference per mode (fluency /
                         relevance / behavioral), then fan results back out
                         to per-workdir JSONL files
  Phase 3 (aggregate):  compute per-condition accuracies from per-workdir ratings

Usage examples:

  # All verse-long eval for atp
  python run_judge.py --eval_subdir verse-long_eval --algos atp

  # Specific model + task
  python run_judge.py \\
      --model_name Qwen1.5-14B-Chat \\
      --source sycophancy-long \\
      --base non-sycophantic \\
      --eval_subdir sycophancy-single_eval \\
      --algos atp

  # Everything for a model
  python run_judge.py --model_name Qwen1.5-14B-Chat

  # All results
  python run_judge.py --all

  # Skip behavioral judge (only fluency + relevance)
  python run_judge.py --eval_subdir sycophancy-long_eval --skip_judge

  # Generate plots after evaluation
  python run_judge.py --eval_subdir verse-long_eval --plots
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from config import BASE_DIR, RUNS_DIR, DATA_DIR, TOKENIZER_MODEL_NAME
from merge_outputs import (
    discover_gen_files,
    extract_path_metadata,
    gen_to_csv,
)
from build_prompts import (
    load_tokenizer,
    build_fluency_prompts,
    build_relevance_prompts,
    build_judge_prompts,
)
from compute_single_accuracies import compute_accuracy_for_file
from compute_accuracies import compute_accuracy_for_workdir

ACCURACY_DIR = BASE_DIR / "judge-evals" / "accuracy"
WORKDIRS_ROOT = BASE_DIR / "judge-evals" / "workdirs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gen_workdir(gen_path: str) -> Path:
    """Derive a per-file workdir under judge-evals/workdirs/."""
    p = Path(gen_path)
    stem = p.stem
    if stem.endswith("_gen"):
        stem = stem[:-4]
    try:
        rel = p.parent.relative_to(RUNS_DIR)
    except ValueError:
        rel = Path(p.parent.name)
    return WORKDIRS_ROOT / rel / stem


def accuracy_paths(meta: dict) -> tuple[Path, Path]:
    base_dir = (
        ACCURACY_DIR
        / meta["MODEL_ID"]
        / f"from_{meta['SOURCE']}_to_{meta['BASE']}"
        / meta["METHOD"]
        / meta["EVAL_SUB_DIR"]
        / meta["STEER_SUB_DIR"]
    )
    fn_base = f"{meta['N']}_{meta['REPS']}_{meta['STEERING_METHOD']}_topk_{meta['topk']}"
    wo_rf = base_dir / f"{fn_base}_gen_accuracy_wo_rf.json.accuracy.json"
    w_rf  = base_dir / f"{fn_base}_gen_accuracy_w_rf.json.accuracy.json"
    return wo_rf, w_rf


def accuracy_exists(meta: dict) -> bool:
    wo_rf, w_rf = accuracy_paths(meta)
    return wo_rf.exists() and w_rf.exists()


def is_single_eval(meta: dict) -> bool:
    return "single" in meta.get("EVAL_SUB_DIR", "")


# ---------------------------------------------------------------------------
# Phase 1: per-file CSV + prompt building
# ---------------------------------------------------------------------------

def phase1_prepare(
    gen_files: list[str],
    data_dir: str,
    skip_judge: bool,
    force: bool,
) -> list[tuple[Path, dict, str]]:
    """
    For each gen file: convert to CSV and build prompt CSVs.
    Tokenizer is loaded ONCE and reused for all long-eval files.

    Returns list of (workdir, meta, gen_path) for files that need evaluation.
    Single-eval files get prompt CSVs skipped (not needed for token matching).
    """
    to_process = []
    for gen_path in gen_files:
        try:
            meta = extract_path_metadata(gen_path)
        except ValueError as e:
            print(f"Skipping {gen_path}: {e}")
            continue
        if not force and accuracy_exists(meta):
            print(f"  Skip (done): {Path(gen_path).name}")
            continue
        to_process.append((gen_path, meta))

    if not to_process:
        return []

    # Only need tokenizer for long-eval files
    long_files = [(gp, m) for gp, m in to_process if not is_single_eval(m)]
    tokenizer = None
    if long_files:
        print(f"\nPhase 1: {len(to_process)} files to prepare "
              f"({len(long_files)} long-eval). Loading tokenizer once...")
        tokenizer = load_tokenizer(TOKENIZER_MODEL_NAME)
        print("Tokenizer loaded.\n")
    else:
        print(f"\nPhase 1: {len(to_process)} files to prepare (all single-eval, no tokenizer needed).\n")

    prepared = []
    errors = 0

    for gen_path, meta in to_process:
        workdir = gen_workdir(gen_path)
        workdir.mkdir(parents=True, exist_ok=True)
        name = Path(gen_path).stem

        # Step 1: gen.json → eval_output.csv
        eval_csv = workdir / "eval_output.csv"
        if not eval_csv.exists():
            try:
                gen_to_csv(gen_path, data_dir, str(eval_csv))
            except (ValueError, FileNotFoundError) as e:
                print(f"  ERROR converting {name}: {e}")
                errors += 1
                continue

        # Single-eval: no prompt CSVs needed (token matching only)
        if is_single_eval(meta):
            prepared.append((workdir, meta, gen_path))
            continue

        # Long-eval: build fluency + relevance + judge prompts
        df = pd.read_csv(eval_csv, keep_default_na=False,
                         dtype={"post-intervention-response": str,
                                "original-response": str,
                                "query": str,
                                "data_path_query": str})

        rf_csv = workdir / "relevance_fluency_prompts.csv"
        if not rf_csv.exists():
            print(f"  Building fluency+relevance prompts: {name}")
            rf_df = build_fluency_prompts(df.copy(), tokenizer)
            rf_df = build_relevance_prompts(rf_df, tokenizer)
            rf_df.to_csv(rf_csv, index=False)

        if not skip_judge:
            jp_csv = workdir / "judge_prompts.csv"
            if not jp_csv.exists():
                print(f"  Building judge prompts: {name}")
                try:
                    jp_df = build_judge_prompts(df.copy(), tokenizer)
                    jp_df.to_csv(jp_csv, index=False)
                except (ValueError, AssertionError) as e:
                    print(f"  ERROR building judge prompts for {name}: {e}")
                    errors += 1
                    continue

        prepared.append((workdir, meta, gen_path))

    print(f"\nPhase 1 done: {len(prepared)} files prepared, {errors} errors")
    return prepared


# ---------------------------------------------------------------------------
# Phase 2: evaluate (single-eval via token match; long-eval via judge model)
# ---------------------------------------------------------------------------

def _evaluate_all_workdirs_batched(
    llm,
    long_items: list[tuple[Path, dict, str]],
    batch_size: int,
    skip_judge: bool,
):
    """
    Collect all prompts from all workdirs into one batch per mode, run vLLM once
    per mode, then write results back to per-workdir JSONL files.
    """
    from evaluator import get_sampling_params, generate_in_batches
    from config import PASSTHROUGH_COLS, extract_rating
    from compute_accuracies import extract_first_int

    dtype = {"post-intervention-response": str, "original-response": str,
             "query": str, "data_path_query": str}

    modes_config = [
        ("fluency",   "fluency_prompt",   "fluency_ratings.jsonl",   extract_rating),
        ("relevance", "relevance_prompt",  "relevance_ratings.jsonl", extract_rating),
    ]
    if not skip_judge:
        modes_config.append(
            ("judge", "judge_prompt", "judge_ratings.jsonl", extract_first_int)
        )

    for mode, prompt_col, out_filename, rate_fn in modes_config:
        # Collect prompts + metadata from all workdirs that still need this mode
        all_rows = []   # list of (workdir, row_dict)
        for wd, _meta, _gp in long_items:
            out_path = wd / out_filename
            if out_path.exists():
                continue
            if mode in ("fluency", "relevance"):
                csv_path = wd / "relevance_fluency_prompts.csv"
            else:
                csv_path = wd / "judge_prompts.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, keep_default_na=False, dtype=dtype)
            if prompt_col not in df.columns:
                continue
            for _, row in df.iterrows():
                all_rows.append((wd, row.to_dict()))

        if not all_rows:
            print(f"  [{mode}] Nothing to evaluate (all done).")
            continue

        print(f"  [{mode}] Evaluating {len(all_rows)} prompts across "
              f"{len({wd for wd, _ in all_rows})} workdirs...")

        prompts = [row[prompt_col] for _, row in all_rows]
        sp = get_sampling_params()

        # Single large batched inference
        outputs = []
        for batch in generate_in_batches(llm, prompts, sp, batch_size):
            outputs.extend(batch)

        # Group results back by workdir and write JSONL
        from collections import defaultdict
        wd_results: dict[Path, list[dict]] = defaultdict(list)
        for (wd, row), output in zip(all_rows, outputs):
            item = {
                **{col: row[col] for col in PASSTHROUGH_COLS if col in row},
                prompt_col: row[prompt_col],
                "judge_output": output,
                "judge_rating": rate_fn(output),
            }
            wd_results[wd].append(item)

        written = 0
        for wd, results in wd_results.items():
            out_path = wd / out_filename
            with open(out_path, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
        print(f"  [{mode}] Done. Wrote {written} files.")


def phase2_evaluate(
    prepared: list[tuple[Path, dict, str]],
    data_dir: str,
    batch_size: int,
    skip_judge: bool,
) -> list[tuple[Path, dict, str]]:
    """
    Evaluate all files.
    - Single-eval: token matching (no model), accuracy written directly to ACCURACY_DIR.
    - Long-eval: load judge model ONCE, batch all prompts across workdirs into one
                 inference per mode, then write results to per-workdir JSONL files.

    Returns the long-eval items (needed for phase 3).
    """
    single_items = [(wd, m, gp) for wd, m, gp in prepared if is_single_eval(m)]
    long_items   = [(wd, m, gp) for wd, m, gp in prepared if not is_single_eval(m)]

    # --- Single-eval: token matching, no model ---
    if single_items:
        print(f"\nPhase 2a: Token matching for {len(single_items)} single-eval files")
        for _wd, _meta, gen_path in single_items:
            compute_accuracy_for_file(Path(gen_path), data_dir, ACCURACY_DIR)

    # --- Long-eval: load model once, batch all prompts across workdirs ---
    if long_items:
        print(f"\nPhase 2b: Judge model evaluation for {len(long_items)} long-eval files")
        from evaluator import make_llm
        llm = make_llm()
        _evaluate_all_workdirs_batched(llm, long_items, batch_size, skip_judge)

    return long_items


# ---------------------------------------------------------------------------
# Phase 3: compute accuracies from per-workdir ratings
# ---------------------------------------------------------------------------

def phase3_accuracies(long_items: list[tuple[Path, dict, str]], skip_judge: bool):
    """Compute per-condition accuracies for all long-eval workdirs."""
    print(f"\nPhase 3: Computing accuracies for {len(long_items)} long-eval workdirs")
    for wd, _meta, _gp in long_items:
        print(f"  {wd.name}")
        compute_accuracy_for_workdir(wd, ACCURACY_DIR, skip_judge)


# ---------------------------------------------------------------------------
# Optional: plots
# ---------------------------------------------------------------------------

def step_plots(plots_args: list[str] | None = None):
    plots_script = str(BASE_DIR / "plots.py")
    if not os.path.exists(plots_script):
        print(f"Skipping plots ({plots_script} not found)")
        return
    import subprocess
    cmd = [sys.executable, plots_script] + (plots_args or [])
    print(f"\n{'=' * 60}\n  Generating plots\n{'=' * 60}")
    print(f"  cmd: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"\nFAILED: plots (exit code {result.returncode})")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run judge evaluation pipeline (model loads once, per-file storage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_name",   default=None)
    p.add_argument("--source",       default=None)
    p.add_argument("--base",         default=None)
    p.add_argument("--algos",        nargs="*", default=None)
    p.add_argument("--eval_subdir",  default=None,
                   help="e.g. sycophancy-single_eval")
    p.add_argument("--steer_subdir", default=None,
                   help="e.g. sycophancy-long_steer")
    p.add_argument("--all",          action="store_true")
    p.add_argument("--skip_judge",   action="store_true",
                   help="Skip behavioral judge (fluency + relevance only)")
    p.add_argument("--force",        action="store_true",
                   help="Re-process even if accuracy files exist")
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--plots",        action="store_true")
    p.add_argument("--plots_args",   nargs="*", default=None)
    p.add_argument("--runs_dir",     default=str(RUNS_DIR))
    p.add_argument("--data_dir",     default=str(DATA_DIR))
    return p.parse_args()


def main():
    args = parse_args()

    if not args.all and not any([
        args.model_name, args.source, args.base,
        args.algos, args.eval_subdir, args.steer_subdir,
    ]):
        print("Error: specify at least one filter or --all. Run --help for examples.")
        sys.exit(1)

    gen_files = discover_gen_files(
        args.runs_dir, args.model_name, args.source, args.base, args.algos,
        args.eval_subdir, args.steer_subdir,
    )
    print(f"Found {len(gen_files)} gen files")
    print(f"Accuracy dir: {ACCURACY_DIR}\n")

    # Phase 1: convert + build prompts (fast, no GPU)
    print("=" * 60)
    print("  PHASE 1: Convert gen files + build prompt CSVs")
    print("=" * 60)
    prepared = phase1_prepare(gen_files, args.data_dir, args.skip_judge, args.force)

    if not prepared:
        print("Nothing to evaluate.")
    else:
        # Phase 2: evaluate (single-eval token match + long-eval model)
        print("\n" + "=" * 60)
        print("  PHASE 2: Evaluate")
        print("=" * 60)
        long_items = phase2_evaluate(prepared, args.data_dir, args.batch_size, args.skip_judge)

        # Phase 3: compute per-condition accuracies for long-eval
        if long_items:
            print("\n" + "=" * 60)
            print("  PHASE 3: Compute accuracies")
            print("=" * 60)
            phase3_accuracies(long_items, args.skip_judge)

    print(f"\nPipeline complete. Accuracy files: {ACCURACY_DIR}")

    if args.plots:
        step_plots(args.plots_args)


if __name__ == "__main__":
    main()
