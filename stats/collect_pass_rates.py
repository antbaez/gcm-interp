"""
Per-prompt pass/fail comparison between steering methods (steering_types), for
every model / dataset / norm_mode / stream_mode combination found
under `results/` and/or `judge-evals/workdirs/` — restricted to the `local`
(per-layer/per-head localized) scope, never `global`.

Two splits, selected with `--split`:

- `test` (default) — the held-out `<base>-heldout-test.jsonl` run, which was
  generated once at the config `select_best_config.py` chose on validation.
  There is exactly one condition per method, so nothing is maximized here and
  the pass rate is an unbiased estimate. This is the split to run
  `run_mcnemar_test.py` on, which is the whole point of this script.
- `val` — the N/layer sweep on `<base>-test.jsonl`. For each method it scans
  every condition directory and reports the one with the highest w_rf pass
  rate. That maximum is selection-biased: the same 50 prompts choose the config
  and score it.

Output is one row per prompt with a pass column per method, written to a CSV
under `stats/pass_results/` mirroring the `results/` path.

The pass rule lives in ../judge-evals/selection_utils.py and mirrors
judge-evals/compute_accuracies.py (_compute_and_write): empty responses are
forced to fail, and non-sycophancy tasks pass at judge_rating == 5.

Usage (from the repo root):
    python stats/collect_pass_rates.py
    python stats/collect_pass_rates.py --split val
    python stats/collect_pass_rates.py --model qwen3 --dataset harmful
    python stats/collect_pass_rates.py --model gemma-4-12B-it --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

STATS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STATS_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "judge-evals"))

from selection_utils import scan_conditions, select_best, condition_label

RESULTS_ROOT = REPO_ROOT / "results"
WORKDIRS_ROOT = REPO_ROOT / "judge-evals" / "workdirs"
PASS_RESULTS_ROOT = STATS_ROOT / "pass_results"

SCOPE = "local"

# task dir -> base name, which names both split files.
TASK_BASES = {
    "from_harmful-long_to_harmless": "harmless",
    "from_non-sycophantic-long_to_sycophancy": "sycophancy",
    "from_verse-long_to_prose": "prose",
}

# Short tags accepted by --model/--dataset, so the pipeline can pass the same
# tags it uses everywhere else. Full directory names are also accepted.
MODEL_DIRS = {
    "olmo": "OLMo-2-1124-13B-DPO",
    "qwen": "Qwen1.5-14B-Chat",
    "qwen3": "Qwen3-14B",
    "gemma4": "gemma-4-12B-it",
    "llama": "Llama-3.1-8B-Instruct",
}

DATASET_TASKS = {
    "harmful": "from_harmful-long_to_harmless",
    "sycophancy": "from_non-sycophantic-long_to_sycophancy",
    "verse": "from_verse-long_to_prose",
}


def resolve_filter(tag_arg: str | None, mapping: dict) -> set[str] | None:
    """Expand a comma-separated tag list into the directory names it selects.

    None or "all" means no filtering. Unknown values pass through unchanged so a
    full directory name (e.g. `gemma-4-12B-it`) works as well as a tag.
    """
    if not tag_arg or tag_arg == "all":
        return None
    return {mapping.get(t.strip(), t.strip()) for t in tag_arg.split(",")}


def best_config_for_method(method_root: Path, test_file: str | None):
    """Return per-prompt rows for the condition to report for one method.

    On the validation split this is an argmax over the whole sweep; on the
    held-out split only one condition exists, so this just picks it up.
    """
    conditions = scan_conditions(method_root, test_file=test_file)
    chosen = select_best(conditions)
    if chosen is None:
        return None, None, None, 0
    return chosen["rows"], condition_label(chosen), chosen["rate"], len(conditions)


def discover_local_combos(model_filter=None, dataset_filter=None, scope=SCOPE):
    """Find every <model>/<task>/<norm_mode>/<stream_mode>
    combination that has a `scope` dir (`local` by default) under results/
    and/or workdirs/, along with the steering methods available under it.

    Checked against both roots, not just results/: a combo whose results/ gen
    files were deleted (or never generated) but whose workdirs/ judging
    survived still has data worth reporting, and only checking results/ would
    silently drop it even though judge_ratings.jsonl is still sitting right
    there.
    """
    combos = {}
    for root in (RESULTS_ROOT, WORKDIRS_ROOT):
        for scope_dir in sorted(root.glob("*/*/*/*/" + scope)):
            model, task, norm_mode, stream_mode, _ = scope_dir.relative_to(root).parts
            if model_filter and model not in model_filter:
                continue
            if dataset_filter and task not in dataset_filter:
                continue
            methods = {p.name for p in scope_dir.iterdir() if p.is_dir()}
            if methods:
                key = (model, task, norm_mode, stream_mode)
                combos.setdefault(key, set()).update(methods)

    return [(*key, sorted(methods)) for key, methods in sorted(combos.items())]


def split_test_file(task: str, split: str) -> str | None:
    """The test-file stem naming this task's split, or None for unknown tasks
    (in which case every condition is considered, as before)."""
    base = TASK_BASES.get(task)
    if base is None:
        return None
    return f"{base}-heldout-test" if split == "test" else f"{base}-test"


def build_comparison_csv(model, task, norm_mode, stream_mode, methods, split):
    """Compute the per-prompt pass/fail comparison across `methods` for one
    combo, reading judge ratings from the mirrored workdirs tree. Returns the
    output DataFrame, or None if no method had any judge ratings."""
    workdir_base = WORKDIRS_ROOT / model / task / norm_mode / stream_mode / SCOPE
    test_file = split_test_file(task, split)

    per_method = {}
    for method in methods:
        rows, label, rate, n_conds = best_config_for_method(workdir_base / method, test_file)
        if rows is None:
            print(f"  skipping method '{method}': no {split} judge ratings under {workdir_base / method}")
            continue
        if split == "test":
            print(f"  {method}: {label}  (w_rf pass rate = {rate:.3f})")
            if n_conds > 1:
                # More than one held-out condition means something swept the test
                # split, which puts the selection bias right back in.
                print(f"    WARNING: {n_conds} held-out conditions found for '{method}'; "
                      f"reporting the best of them is NOT an unbiased estimate")
        else:
            print(f"  {method}: best of {n_conds} = {label}  (val w_rf pass rate = {rate:.3f})")
        per_method[method] = {r["prompt"]: r["pass"] for r in rows}

    if not per_method:
        return None

    all_prompts = sorted(set().union(*(set(d) for d in per_method.values())))
    out_rows = []
    for prompt in all_prompts:
        row = {"prompt": prompt}
        for method in per_method:
            row[f"{method}_pass"] = per_method[method].get(prompt)
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help=f"Restrict to model tag(s) or dir name(s), comma-separated, or 'all' "
                             f"(tags: {', '.join(MODEL_DIRS)}). Default: all models found")
    parser.add_argument("--dataset", default=None,
                        help=f"Restrict to dataset tag(s) or task dir(s), comma-separated, or 'all' "
                             f"(tags: {', '.join(DATASET_TASKS)}). Default: all datasets found")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="test (default): the single held-out run on <base>-heldout-test (unbiased). "
                             "val: best of the N/layer sweep on <base>-test (selection-biased)")
    parser.add_argument("--output-dir", default=str(PASS_RESULTS_ROOT), help="Root dir for output CSVs")
    parser.add_argument("--dry-run", action="store_true", help="List combos that would be processed, without writing CSVs")
    parser.add_argument("--no-force", dest="force", action="store_false",
                        help="Skip combos whose output CSV already exists instead of regenerating them")
    parser.set_defaults(force=True)
    args = parser.parse_args()

    combos = discover_local_combos(
        model_filter=resolve_filter(args.model, MODEL_DIRS),
        dataset_filter=resolve_filter(args.dataset, DATASET_TASKS),
    )
    if not combos:
        raise SystemExit(f"No '{SCOPE}' scope combinations found under {RESULTS_ROOT} or {WORKDIRS_ROOT}")

    output_root = Path(args.output_dir)
    n_written = 0
    n_skipped = 0
    # Validation keeps the original filename so existing CSVs stay valid; the
    # held-out split gets its own so the two are never confused downstream.
    stem = SCOPE if args.split == "val" else f"{SCOPE}_test"

    print("n=200")

    for model, task, norm_mode, stream_mode, methods in combos:
        rel = Path(model) / task / norm_mode / stream_mode
        out_path = output_root / rel / f"{stem}.csv"
        print(f"\n{model}/{task}")

        if args.dry_run:
            continue

        if out_path.exists() and not args.force:
            print(f"  already exists, skipping -> {out_path}")
            n_skipped += 1
            continue

        df = build_comparison_csv(model, task, norm_mode, stream_mode, methods, args.split)
        if df is None:
            print(f"  no judge ratings found for any method, skipping")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        n_written += 1

    if not args.dry_run:
        print(f"\nWrote {n_written} CSV(s), skipped {n_skipped} already-existing under {output_root}")


if __name__ == "__main__":
    main()
