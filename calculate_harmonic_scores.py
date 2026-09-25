"""
Harmonic-mean scores from saved judge outputs, per model / dataset / steering method.

For every prompt, the three judge ratings are put on a common 0-2 scale:

    concept    judge_ratings.jsonl      1-5 -> 1,2,3 -> 0, 4 -> 1, 5 -> 2
    fluency    fluency_ratings.jsonl    0-2 as-is
    relevance  relevance_ratings.jsonl  0-2 as-is

and combined as their harmonic mean, 3 / (1/c + 1/f + 1/r), taken as 0 when any
of the three is 0. The reported score is that per-prompt harmonic mean averaged
over the condition's prompts, alongside the mean of each component.

As in selection_utils.py / compute_accuracies.py, an empty steered response
counts as concept 1 (-> 0). A rating that didn't parse or is out of range
(concept not 1-5, fluency/relevance not 0-2) counts as 0 and stays in n; the
`unparsed` column counts prompts with at least one such rating. A prompt missing
from fluency_ratings.jsonl or relevance_ratings.jsonl is dropped from n and
counted in the `dropped` column.

Reads the layout run_harmonic_tests.py saves under harmonic_results/judge_scores/
(the default), which mirrors judge-evals/workdirs/, so --root can point at either:

    <root>/<model>/<task>/<norm>/<stream>/<scope>/<method>/N=<N>_<ablation>_layer=<L>_<test_file>/

One row is printed per condition, so a method with several saved (N, layer)
configs gets one row each.

Dataset variants (e.g. verse-varied, sycophancy-unaligned; see
run_harmonic_tests.py) are reported under their tag, with their base dataset
in the `base` column. `delta_vs_base` is the variant's harmonic mean minus its
base dataset's at the same model, method, N and layer on the matching split
(`-` when that base condition isn't scored, and for base datasets themselves).
--datasets filters rows by tag (`variants` / `all` expand as in
run_harmonic_tests.py); base rows used for the deltas are read regardless.

Usage:
    python calculate_harmonic_scores.py
    python calculate_harmonic_scores.py --out harmonic_scores.csv
    python calculate_harmonic_scores.py --root judge-evals/workdirs --split heldout-test
    python calculate_harmonic_scores.py --datasets variants
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "judge-evals"))

from selection_utils import parse_condition_dir, read_jsonl
from run_harmonic_tests import DATASET_TASKS, VARIANT_BASE, parse_datasets

DEFAULT_ROOT = REPO_ROOT / "harmonic_results" / "judge_scores"

# Task dir -> short dataset tag, incl. dataset variants; unknown tasks are
# reported under their task dir name.
TASK_DATASETS = {task: tag for tag, (task, _) in DATASET_TASKS.items()}

FIELDS = ["model", "dataset", "base", "method", "N", "layer", "test_file", "n", "unparsed", "dropped",
          "harmonic_mean", "delta_vs_base", "concept", "fluency", "relevance", "path"]


def map_concept(rating) -> int | None:
    """1-5 concept rating -> 0-2 (1,2,3 -> 0, 4 -> 1, 5 -> 2); None if out of range."""
    if not isinstance(rating, int) or not 1 <= rating <= 5:
        return None
    return max(rating - 3, 0)


def zero_to_two(rating) -> int | None:
    return rating if isinstance(rating, int) and 0 <= rating <= 2 else None


def harmonic_mean(*scores: int) -> float:
    if any(s == 0 for s in scores):
        return 0.0
    return len(scores) / sum(1 / s for s in scores)


def score_condition(cond_dir: Path) -> dict | None:
    """Per-condition mean harmonic score and component means, or None if unjudged."""
    judge_recs = read_jsonl(cond_dir / "judge_ratings.jsonl")
    if not judge_recs:
        return None
    flu_by_query = {r.get("data_path_query"): r.get("judge_rating")
                    for r in read_jsonl(cond_dir / "fluency_ratings.jsonl")}
    rel_by_query = {r.get("data_path_query"): r.get("judge_rating")
                    for r in read_jsonl(cond_dir / "relevance_ratings.jsonl")}

    rows, unparsed, dropped = [], 0, 0
    for rec in judge_recs:
        query = rec.get("data_path_query")
        if query not in flu_by_query or query not in rel_by_query:
            dropped += 1
            continue
        response = rec.get("post-intervention-response", "")
        concept_rating = rec.get("judge_rating")
        # Empty generations are never a successful steer, regardless of rating.
        if not isinstance(response, str) or response.strip() == "":
            concept_rating = 1
        scores = (map_concept(concept_rating),
                  zero_to_two(flu_by_query[query]),
                  zero_to_two(rel_by_query[query]))
        # Unparsed / out-of-range ratings (run_judge writes -1 or None) score 0.
        if None in scores:
            unparsed += 1
        c, f, r = (0 if x is None else x for x in scores)
        rows.append((c, f, r, harmonic_mean(c, f, r)))

    n = len(rows)
    mean = (lambda i: sum(row[i] for row in rows) / n) if n else (lambda i: None)
    return {"n": n, "unparsed": unparsed, "dropped": dropped, "harmonic_mean": mean(3),
            "concept": mean(0), "fluency": mean(1), "relevance": mean(2)}


def collect(root: Path, split: str | None) -> list[dict]:
    results = []
    for ratings in sorted(root.glob("**/judge_ratings.jsonl")):
        cond_dir = ratings.parent
        meta = parse_condition_dir(cond_dir.name)
        if meta is None or meta["axis"] != "layer":
            continue
        if split is not None and not meta["test_file"].endswith(split):
            continue
        parts = cond_dir.relative_to(root).parts
        # <model>/<task>/<norm>/<stream>/<scope>/<method>/<condition>
        if len(parts) != 7:
            print(f"Skipping {cond_dir} (unexpected directory depth)", file=sys.stderr)
            continue
        scores = score_condition(cond_dir)
        if scores is None:
            continue
        dataset = TASK_DATASETS.get(parts[1], parts[1])
        results.append({
            "model": parts[0],
            "dataset": dataset,
            "base": VARIANT_BASE.get(dataset, dataset),
            "method": parts[5],
            "N": meta["N"],
            "layer": meta["value"],
            "test_file": meta["test_file"],
            **scores,
            "path": str(cond_dir.relative_to(root)),
        })
    add_base_deltas(results)
    results.sort(key=lambda r: (r["model"], r["base"], r["dataset"] != r["base"], r["dataset"],
                                r["method"], r["N"], r["layer"]))
    return results


def split_suffix(r: dict) -> str:
    """The split part of a test file stem, e.g. 'heldout-test' for prose-varied-heldout-test."""
    base_name = DATASET_TASKS[r["dataset"]][1] if r["dataset"] in DATASET_TASKS else ""
    return r["test_file"][len(base_name) + 1:] if r["test_file"].startswith(base_name + "-") else r["test_file"]


def add_base_deltas(results: list[dict]):
    """Set each variant row's harmonic-mean difference from its base dataset's matching row."""
    key = lambda r, dataset: (r["model"], dataset, r["method"], r["N"], r["layer"], split_suffix(r))
    base_scores = {key(r, r["dataset"]): r["harmonic_mean"]
                   for r in results if r["dataset"] not in VARIANT_BASE}
    for r in results:
        base_score = (base_scores.get(key(r, r["base"])) if r["dataset"] in VARIANT_BASE else None)
        r["delta_vs_base"] = (r["harmonic_mean"] - base_score
                              if base_score is not None and r["harmonic_mean"] is not None else None)


def fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def fmt_delta(x) -> str:
    return "-" if x is None else f"{x:+.3f}"


def print_table(results: list[dict]):
    dw = max([11] + [len(r["dataset"]) for r in results])
    header = (f"{'model':<22} {'dataset':<{dw}} {'method':<11} {'N':>6} {'layer':>5} "
              f"{'n':>4} {'unpar':>5} {'drop':>4}  {'harmonic':>8} {'Δ base':>7} "
              f"{'concept':>7} {'fluency':>7} {'relev':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<22} {r['dataset']:<{dw}} {r['method']:<11} {r['N']:>6g} {r['layer']:>5} "
              f"{r['n']:>4} {r['unparsed']:>5} {r['dropped']:>4}  {fmt(r['harmonic_mean']):>8} "
              f"{fmt_delta(r['delta_vs_base']):>7} {fmt(r['concept']):>7} "
              f"{fmt(r['fluency']):>7} {fmt(r['relevance']):>7}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Judge-score tree to read (default: harmonic_results/judge_scores)")
    parser.add_argument("--split", default=None,
                        help="Only conditions whose test file ends with this, e.g. 'heldout-test' "
                             "(default: every condition found)")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset tags to report, incl. variants; 'variants' = every "
                             "variant, 'all' = base datasets + every variant (default: everything found)")
    parser.add_argument("--out", type=Path, default=None, help="Also write the table as CSV here")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"{args.root} not found; run run_harmonic_tests.py first or pass --root")
    results = collect(args.root, args.split)
    if args.datasets is not None:
        wanted = set(parse_datasets(args.datasets))
        results = [r for r in results if r["dataset"] in wanted]
    if not results:
        raise SystemExit(f"No judged conditions found under {args.root}")

    print_table(results)
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
