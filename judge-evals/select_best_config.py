"""
Pick the best (N, layer) per model / dataset / steering method on the
*validation* split, so the held-out test split can be evaluated once at that
single config instead of being swept.

The N/layer sweep under `judge-evals/workdirs/` is validation data: scoring it
and reporting its maximum on the same prompts is a selection-biased estimate.
This script does the selection half — it reads the sweep, takes the argmax w_rf
pass rate per (model, dataset, steering_type), and writes `best_configs.json`.
`run_steering.sh --split test` then pins those values and generates on
`<base>-heldout-test.jsonl`, which nothing in the sweep ever touched.

Ties go to the smallest N, then the layer closest to the middle of the swept
range (see selection_utils.select_best).

Selecting a new config also invalidates any held-out run generated at the old
one, so by default this prunes held-out conditions whose N/layer no longer match
the selection (see prune_stale_heldout). Without that, a second held-out
condition accumulates every time the validation maximum moves, and
`stats/collect_pass_rates.py` reports the best of them — putting the
selection bias the held-out split exists to avoid straight back in.

Selections are merged into `best_configs.json` rather than replacing it, so a run
scoped to one model/dataset leaves every other combination's entry alone. The
pipeline runs one such scoped call per job, concurrently, so the read and write
are held under a lock (see merge_best_configs).

Usage:
    python judge-evals/select_best_config.py
    python judge-evals/select_best_config.py --model olmo,qwen3 --dataset harmful
    python judge-evals/select_best_config.py --no-prune
    python judge-evals/select_best_config.py --model qwen3 --dataset harmful --pending-out /tmp/pending
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
from pathlib import Path

JUDGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = JUDGE_DIR.parent
# selection_utils sits alongside this script, but the import has to work when
# run as `python judge-evals/select_best_config.py` from the repo root too.
sys.path.insert(0, str(JUDGE_DIR))

from selection_utils import parse_condition_dir, scan_conditions, select_best, condition_label

WORKDIRS_ROOT = JUDGE_DIR / "workdirs"
RESULTS_ROOT = REPO_ROOT / "results"
ACCURACY_ROOT = JUDGE_DIR / "accuracy"
DEFAULT_OUTPUT = JUDGE_DIR / "best_configs.json"

# Model tag -> results/ dir name (mirrors run_steering.sh's MODEL_ID case).
MODEL_DIRS = {
    "olmo": "OLMo-2-1124-13B-DPO",
    "qwen": "Qwen1.5-14B-Chat",
    "qwen3": "Qwen3-14B",
    "gemma4": "gemma-4-12B-it",
    "llama": "Llama-3.1-8B-Instruct",
}

# Dataset tag -> (task dir, base name). Base names the split files.
DATASET_TASKS = {
    "harmful":    ("from_harmful-long_to_harmless",             "harmless"),
    "sycophancy": ("from_non-sycophantic-long_to_sycophancy",   "sycophancy"),
    "verse":      ("from_verse-long_to_prose",                  "prose"),
}

NORM_MODE = "normalized"
STREAMS = ("residuals", "attention")
SCOPES = ("local", "global")

# Accuracy files are named from run_judge.py's fn_base
# (`<N>_<reps>_<ablation>_layer_<value>_gen_accuracy_*`), which is a different
# shape from the condition dirs selection_utils parses. The legacy `topk_` slot
# is still matched so files written before the rename are still prunable.
ACCURACY_FILE_RE = re.compile(
    r"^(?P<N>\d+(?:\.\d+)?)_[^_]+_(?:steer|mean)_(?:layer|topk)_(?P<value>all|\d+(?:\.\d+)?)_gen_accuracy_"
)


def _same_config(meta: dict, chosen: dict) -> bool:
    """Whether a parsed condition names the same N/layer as the selection."""
    return meta["N"] == chosen["N"] and str(meta["value"]) == str(chosen["value"])


def prune_stale_heldout(model_dir: str, task: str, base: str, method: str,
                        chosen: dict, dry_run: bool, stream: str, scope: str) -> list[Path]:
    """Delete held-out conditions whose N/layer no longer match the selection.

    Works off what is on disk rather than off the previous best_configs.json, so
    it stays correct if that file was edited, deleted, or written by an older
    run. Each tree is scanned independently: an interrupted earlier prune leaves
    the trees disagreeing, and scanning them separately repairs that instead of
    stopping at the first one that already looks clean.
    """
    test_file = f"{base}-heldout-test"
    rel = Path(model_dir) / task / NORM_MODE / stream / scope / method
    removed = []

    # results/: the generated .txt/.json pair, which is what judging discovers.
    for gen_json in sorted((RESULTS_ROOT / rel).glob(f"*_{test_file}_gen.json")):
        meta = parse_condition_dir(gen_json.name[: -len("_gen.json")])
        if meta is None or _same_config(meta, chosen):
            continue
        for path in (gen_json, gen_json.with_suffix(".txt")):
            if path.exists():
                removed.append(path)
                if not dry_run:
                    path.unlink()

    # workdirs/: what collect_pass_rates.py actually reads. Deleting only
    # the results/ pair would leave the stale condition visible to it.
    for cond_dir in sorted(p for p in (WORKDIRS_ROOT / rel).glob(f"*_{test_file}") if p.is_dir()):
        meta = parse_condition_dir(cond_dir.name)
        if meta is None or _same_config(meta, chosen):
            continue
        removed.append(cond_dir)
        if not dry_run:
            shutil.rmtree(cond_dir)

    # accuracy/: feeds summarize_results.py only. The current layout mirrors
    # results/ and workdirs/ (`<norm>/<stream>/<scope>/<method>/`). Trees written
    # before that layout landed had no norm or stream level and are residual by
    # definition, so their shallower shapes are only swept for the residual
    # stream — globbing them for attention would delete another stream's files.
    # Files whose name does not parse are left alone rather than guessed at.
    acc_task_dir = ACCURACY_ROOT / model_dir / task
    acc_globs = [f"{NORM_MODE}/{stream}/{scope}/{method}/{test_file}/*.json"]
    if stream == "residuals":
        acc_globs += [f"{method}/{test_file}/*.json", f"*/{method}/{test_file}/*.json"]
    acc_files = sorted({p for g in acc_globs for p in acc_task_dir.glob(g)})
    for acc_file in acc_files:
        m = ACCURACY_FILE_RE.match(acc_file.name)
        if m is None:
            continue
        if float(m["N"]) == chosen["N"] and str(m["value"]) == str(chosen["value"]):
            continue
        removed.append(acc_file)
        if not dry_run:
            acc_file.unlink()

    return removed


def heldout_exists(model_dir: str, task: str, base: str, method: str, chosen: dict,
                   stream: str, scope: str) -> bool:
    """Whether a held-out generation already exists at the selected config.

    Matches by parsing what is on disk rather than rebuilding the filename, for
    the same reason prune_stale_heldout does: the ablation tag and N formatting
    are decided at generation time.
    """
    rel = Path(model_dir) / task / NORM_MODE / stream / scope / method
    for gen_json in (RESULTS_ROOT / rel).glob(f"*_{base}-heldout-test_gen.json"):
        meta = parse_condition_dir(gen_json.name[: -len("_gen.json")])
        if meta is not None and _same_config(meta, chosen):
            return True
    return False


def heldout_judged(model_dir: str, task: str, base: str, method: str, chosen: dict,
                   stream: str, scope: str) -> bool:
    """Whether the held-out generation at the selected config has actually been
    judged — i.e. its workdir has a non-empty judge_ratings.jsonl, which is what
    collect_pass_rates.py and select_best_config.py's own validation-side
    scanning both require. Generation existing is not enough: it can be left
    over from an earlier run whose judging step never completed."""
    rel = Path(model_dir) / task / NORM_MODE / stream / scope / method
    test_file = f"{base}-heldout-test"
    conditions = scan_conditions(WORKDIRS_ROOT / rel, test_file=test_file)
    return any(_same_config(c, chosen) for c in conditions)


def merge_best_configs(out_path: Path, updates: dict):
    """Merge `updates` into the JSON at `out_path`, leaving other entries alone.

    Jobs for different model/dataset combinations run concurrently and all write
    this one file, so the read and the write are held under a single lock — two
    jobs that both read the old contents would otherwise each write back a copy
    missing the other's entry. The write goes through a temporary file so a
    reader never sees a half-written file.

    Merging happens per steering method rather than per task: a run where one
    method has no judged conditions yet should not drop that method's earlier
    selection.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_name(out_path.name + ".lock")

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            merged = {}
            if out_path.exists():
                with open(out_path) as f:
                    merged = json.load(f)

            for model_dir, tasks in updates.items():
                for task, streams in tasks.items():
                    for stream, scopes in streams.items():
                        for scope, methods in scopes.items():
                            merged.setdefault(model_dir, {}).setdefault(task, {}) \
                                  .setdefault(stream, {}).setdefault(scope, {}).update(methods)

            tmp_path = out_path.with_name(out_path.name + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(merged, f, indent=2, sort_keys=True)
            os.replace(tmp_path, out_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return merged


def expand_tags(tag_arg: str, mapping: dict, kind: str) -> list[str]:
    if tag_arg == "all":
        return list(mapping.keys())
    tags = [t.strip() for t in tag_arg.split(",")]
    unknown = [t for t in tags if t not in mapping]
    if unknown:
        raise SystemExit(f"Unknown {kind}(s) {unknown}. Must be one of: {', '.join(mapping)}, all")
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="all", help=f"model tag(s) or 'all' ({', '.join(MODEL_DIRS)})")
    parser.add_argument("--dataset", default="all", help=f"dataset tag(s) or 'all' ({', '.join(DATASET_TASKS)})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write best_configs.json")
    parser.add_argument("--stream", default="residuals", choices=STREAMS,
                        help="Which steering stream to select for. The two have separate "
                             "workdir trees and separate entries in best_configs.json, so a "
                             "selection for one never affects the other.")
    parser.add_argument("--scope", default="local", choices=SCOPES,
                        help="Which steering scope to select for: 'local' (single-layer sweep, "
                             "the default) or 'global' (every layer at once). Separate workdir "
                             "trees and separate entries in best_configs.json, like --stream.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selections and prunes without writing or deleting anything")
    parser.add_argument("--no-prune", dest="prune", action="store_false",
                        help="Keep held-out conditions that no longer match the selection")
    parser.add_argument("--pending-out", default=None,
                        help="Write '<model_tag> <dataset_tag>' for each combination whose "
                             "selected config has no held-out generation yet, or has generation "
                             "but no judged ratings. The file is always created, so callers can "
                             "test it for emptiness rather than existence.")
    parser.set_defaults(prune=True)
    args = parser.parse_args()

    model_tags = expand_tags(args.model, MODEL_DIRS, "model")
    dataset_tags = expand_tags(args.dataset, DATASET_TASKS, "dataset")

    best = {}
    pending = []
    for model_tag in model_tags:
        model_dir = MODEL_DIRS[model_tag]
        for dataset_tag in dataset_tags:
            task, base = DATASET_TASKS[dataset_tag]
            val_split = f"{base}-test"
            scope_root = (
                WORKDIRS_ROOT / model_dir / task / NORM_MODE / args.stream / args.scope
            )

            print(f"\n=== {model_tag} / {dataset_tag} / {args.stream} / {args.scope} "
                  f"(validation split: {val_split}) ===")
            if not scope_root.is_dir():
                print("  no validation workdirs, skipping")
                continue

            for method_dir in sorted(p for p in scope_root.iterdir() if p.is_dir()):
                method = method_dir.name
                conditions = scan_conditions(method_dir, test_file=val_split)
                if not conditions:
                    print(f"  {method}: no judged conditions, skipping")
                    continue

                chosen = select_best(conditions)
                ties = sum(1 for c in conditions if c["rate"] == chosen["rate"])
                print(f"  {method}: {condition_label(chosen)}  "
                      f"val w_rf = {chosen['rate']:.3f} (n={chosen['n']}, "
                      f"{len(conditions)} conditions, {ties} tied at best)")

                # Keyed by the directory names run.py already computes
                # (model dir + `from_<source>_to_<base>` + stream + scope) so
                # --split test can look a config up without needing the short
                # model/dataset tags.
                best.setdefault(model_dir, {}).setdefault(task, {}) \
                    .setdefault(args.stream, {}).setdefault(args.scope, {})[method] = {
                    "N": chosen["N"],
                    "layer": int(chosen["value"]) if chosen["value"] != "all" else "all",
                    "val_pass_rate": chosen["rate"],
                    "val_n": chosen["n"],
                    "val_split": val_split,
                    "n_conditions": len(conditions),
                    "n_tied_at_best": ties,
                    "model_tag": model_tag,
                    "dataset_tag": dataset_tag,
                    "stream": args.stream,
                    "scope": args.scope,
                }

                if args.prune:
                    stale = prune_stale_heldout(
                        model_dir, task, base, method, chosen, args.dry_run, args.stream, args.scope
                    )
                    for path in stale:
                        print(f"    {'would remove' if args.dry_run else 'removed'} stale held-out: "
                              f"{path.relative_to(REPO_ROOT)}")

                # Checked after pruning, so a leftover run at a superseded config
                # is already gone and cannot be mistaken for this one.
                if not heldout_exists(model_dir, task, base, method, chosen, args.stream, args.scope):
                    print(f"    held-out run needed at {condition_label(chosen)}: generation missing")
                    if (model_tag, dataset_tag) not in pending:
                        pending.append((model_tag, dataset_tag))
                elif not heldout_judged(model_dir, task, base, method, chosen, args.stream, args.scope):
                    print(f"    held-out run needed at {condition_label(chosen)}: judging missing")
                    if (model_tag, dataset_tag) not in pending:
                        pending.append((model_tag, dataset_tag))

    if args.pending_out and not args.dry_run:
        pending_path = Path(args.pending_out)
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("".join(f"{m} {d}\n" for m, d in pending))

    if not best:
        # Not an error: the pipeline runs this straight after judging, and a
        # combination with nothing judged yet should not fail the whole job.
        print("\nNo validation results found — nothing selected.")
        return

    if args.dry_run:
        print("\n(dry run, not writing)")
        return

    out_path = Path(args.output)
    merge_best_configs(out_path, best)
    n_entries = sum(
        len(methods)
        for tasks in best.values()
        for streams in tasks.values()
        for methods in streams.values()
    )
    print(f"\nMerged {n_entries} selection(s) into {out_path}")


if __name__ == "__main__":
    main()
