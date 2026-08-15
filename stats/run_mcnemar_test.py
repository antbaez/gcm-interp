"""
McNemar's test comparing steering methods within each model/dataset's
per-prompt comparison CSV (produced by collect_pass_rates.py).

Always runs two comparisons per CSV: mean vs last, and positional vs last.
The thing you vary is which model(s) / dataset(s) to pull CSVs for, using
the same tag conventions as run_steering.sh (comma-separated tags, or "all").

Each CSV has a "prompt" column plus one "<method>_pass" column per steering
method (True/False per prompt), all for the same model/dataset/norm_mode/
stream_mode combination. McNemar's test only uses the discordant
pairs (prompts where the two methods disagree): a_only = pass-in-A/fail-in-B,
b_only = fail-in-A/pass-in-B, where A is always mean/positional and B is
always "last" (see COMPARISONS). Both tests are one-sided for the hypothesis
that A beats B (H1: a_only > b_only, i.e. mean/positional passes strictly
more of the discordant prompts than last) — this is a directional design
question (does the richer aggregation beat the naive last-token baseline?),
decided before looking at results, not picked post hoc from whichever
direction looked significant. It reports both the exact one-sided binomial
p-value (used regardless of sample size, as in statsmodels' default
`exact=True`) and a continuity-corrected one-sided normal-approximation
z-test for reference.

Default --model is olmo,qwen3,llama (the models with results so far); pass
--model all to also include qwen. Results are saved to stats/mcnemar_results/
by default; pass --output-dir to save elsewhere.

Usage (from the repo root):
    python stats/run_mcnemar_test.py
    python stats/run_mcnemar_test.py --model olmo,qwen3 --dataset harmful
    python stats/run_mcnemar_test.py --model all --dataset all
"""

import argparse
import math
from pathlib import Path

import pandas as pd

STATS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STATS_ROOT.parent
PASS_RESULTS_ROOT = STATS_ROOT / "pass_results"
DEFAULT_OUTPUT_DIR = STATS_ROOT / "mcnemar_results"

# Model tag -> results/ dir name (mirrors run_steering.sh's MODEL_ID case).
MODEL_DIRS = {
    "olmo": "OLMo-2-1124-13B-DPO",
    "qwen": "Qwen1.5-14B-Chat",
    "qwen3": "Qwen3-14B",
    "llama": "Llama-3.1-8B-Instruct",
    # "gemma4": "gemma-4-12B-it",  # disabled, uncomment to re-enable
}

# Dataset tag -> "from_<source>_to_<base>" task dir (mirrors run_steering.sh's D_SOURCE/D_BASE).
DATASET_TASKS = {
    "harmful": "from_harmful-long_to_harmless",
    "sycophancy": "from_non-sycophantic-long_to_sycophancy",
    "verse": "from_verse-long_to_prose",
}

# Hardcoded comparisons: (method_a, method_b), run for every resolved CSV.
COMPARISONS = [("mean", "last"), ("positional", "last")]

ALPHA = 0.05

NORM_MODE = "normalized"
STREAM_MODE = "residuals"
SCOPE = "local"


def mcnemar_exact_p(wins_a: int, wins_b: int) -> float:
    """One-sided exact binomial McNemar p-value for H1: wins_a > wins_b,
    i.e. P(X >= wins_a) under X ~ Binomial(wins_a + wins_b, 0.5). wins_a < wins_b
    (the "wrong" direction) correctly yields p > 0.5, never a spuriously low p."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    cum = sum(math.comb(n, i) for i in range(wins_a, n + 1))
    return min(cum * (0.5 ** n), 1.0)


def mcnemar_z(wins_a: int, wins_b: int) -> tuple[float, float]:
    """Continuity-corrected one-sided normal-approximation z-stat and p-value
    for H1: wins_a > wins_b (signed, unlike the two-sided chi-square stat this
    replaces — a negative z means wins_a < wins_b, i.e. evidence points the
    other way and p is > 0.5)."""
    n = wins_a + wins_b
    if n == 0:
        return 0.0, 1.0
    diff = wins_a - wins_b
    corrected = math.copysign(max(abs(diff) - 1, 0), diff)
    z = corrected / math.sqrt(n)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return z, p


def run_mcnemar_pair(df: pd.DataFrame, method_a: str, method_b: str):
    """Paired McNemar's test between two method columns of the same CSV.
    Returns None if either method's column is missing."""
    col_a, col_b = f"{method_a}_pass", f"{method_b}_pass"
    if col_a not in df.columns or col_b not in df.columns:
        return None

    a, b = df[col_a], df[col_b]
    valid = a.notna() & b.notna()
    a, b = a[valid].astype(bool), b[valid].astype(bool)

    n = len(a)
    both_pass = int((a & b).sum())
    both_fail = int((~a & ~b).sum())
    a_only = int((a & ~b).sum())  # pass in A, fail in B
    b_only = int((~a & b).sum())  # fail in A, pass in B

    exact_p = mcnemar_exact_p(a_only, b_only)
    z_stat, z_p = mcnemar_z(a_only, b_only)

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n": n,
        "pass_rate_a": a.mean() if n else float("nan"),
        "pass_rate_b": b.mean() if n else float("nan"),
        "both_pass": both_pass,
        "both_fail": both_fail,
        "a_only_pass": a_only,
        "b_only_pass": b_only,
        "exact_p": exact_p,
        "z_stat": z_stat,
        "z_p": z_p,
    }


def simulate_at_n(result: dict, target_n: int) -> dict:
    """Rescale a real McNemar result to a hypothetical sample size, keeping
    the same pass-rate proportions and recomputing p-values from the scaled
    discordant/concordant counts."""
    factor = target_n / result["n"]
    both_pass = round(result["both_pass"] * factor)
    both_fail = round(result["both_fail"] * factor)
    a_only = round(result["a_only_pass"] * factor)
    b_only = round(result["b_only_pass"] * factor)

    exact_p = mcnemar_exact_p(a_only, b_only)
    z_stat, z_p = mcnemar_z(a_only, b_only)

    return {
        **result,
        "n": both_pass + both_fail + a_only + b_only,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "a_only_pass": a_only,
        "b_only_pass": b_only,
        "exact_p": exact_p,
        "z_stat": z_stat,
        "z_p": z_p,
    }


def print_p_value_table(entries: list[tuple[str, dict]]) -> None:
    """Print (model_tag, mcnemar-result) pairs as a column-aligned table —
    column widths are computed from the actual strings each call, not fixed."""
    rows = []
    for model_tag, r in entries:
        # Significant at alpha (one-sided, H1: mean/positional beats last) -> checkmark,
        # otherwise -> x. method_a is always mean/positional, method_b is always "last"
        # (see COMPARISONS), regardless of the left/right print order.
        mark = "✓" if r["exact_p"] < ALPHA else "✗"
        # "last" always prints on the left, the other method on the right,
        # regardless of which one is method_a/method_b internally.
        if r["method_b"] == "last":
            left, left_rate, right, right_rate = r["method_b"], r["pass_rate_b"], r["method_a"], r["pass_rate_a"]
        else:
            left, left_rate, right, right_rate = r["method_a"], r["pass_rate_a"], r["method_b"], r["pass_rate_b"]
        rows.append((model_tag, left, left_rate, right, right_rate, r["exact_p"], mark))

    # Pad method names to the widest on each side ("mean" vs "positional") so the
    # "(rate)" parens — and thus the rate digits — line up vertically across rows.
    left_name_width = max(len(row[1]) for row in rows)
    right_name_width = max(len(row[3]) for row in rows)

    columns = []
    for model_tag, left, left_rate, right, right_rate, exact_p, mark in rows:
        label = (f"{left:<{left_name_width}} ({left_rate:.3f}) vs. "
                 f"{right:<{right_name_width}} ({right_rate:.3f})")
        columns.append((model_tag, label, f"p={exact_p:.4g}", mark))

    widths = [max(len(row[i]) for row in columns) for i in range(3)]
    for model_tag, label, pval, mark in columns:
        print(f"  {model_tag:<{widths[0]}}  {label:<{widths[1]}}  "
              f"{pval:<{widths[2]}}  {mark}")


def print_summary(rows: list[dict], dataset_tags: list[str], n_label: str) -> None:
    """Print the alpha/n header, one column-aligned table per dataset (grouping
    every model together), and the last-vs-mean/positional win tally. Shared by
    the real run and every --simulate-n target so both look identical."""
    if not rows:
        return

    print(f"alpha={ALPHA}  n={n_label}  sided=one-sided (H1: mean/positional beats last)")

    for dataset_tag in dataset_tags:
        dataset_entries = [(r["model_tag"], r) for r in rows if r["dataset_tag"] == dataset_tag]
        if dataset_entries:
            print(f"\n=== {dataset_tag} ===")
            print_p_value_table(dataset_entries)

    # method_a is always mean/positional and method_b is always "last" (see COMPARISONS).
    # exact_p is now one-sided for H1: a_only > b_only, which is mathematically
    # equivalent to pass_rate_a > pass_rate_b (both rates share the same n, and
    # both_pass cancels out of the comparison) — so exact_p < ALPHA already implies
    # the direction; no separate pass-rate check needed the way the old two-sided
    # p-value required.
    def _beats(r):
        return r["exact_p"] < ALPHA

    # One case per (model, dataset) — counts if EITHER mean or positional beats last,
    # so 3 datasets x 3 models = 9 total cases (not 18, which would double-count a
    # model/dataset combo where both methods beat last).
    combos = {}
    for r in rows:
        combos.setdefault((r["model_tag"], r["dataset_tag"]), []).append(r)
    beaten = sum(1 for combo_rows in combos.values() if any(_beats(r) for r in combo_rows))
    print(f"\nlast beaten by mean or positional in {beaten}/{len(combos)} cases")

    # Highest pass rate of the three methods (last/mean/positional) per (model, dataset)
    # combo, regardless of significance. Ties (equal max rate) count for every method
    # that hit it, so the two counts below can sum to more than len(combos).
    mean_highest = 0
    positional_highest = 0
    for combo_rows in combos.values():
        rates = {"last": combo_rows[0]["pass_rate_b"]}
        for r in combo_rows:
            rates[r["method_a"]] = r["pass_rate_a"]
        best = max(rates.values())
        if rates.get("mean") == best:
            mean_highest += 1
        if rates.get("positional") == best:
            positional_highest += 1
    print(f"mean has the highest pass rate in {mean_highest}/{len(combos)} cases")
    print(f"positional has the highest pass rate in {positional_highest}/{len(combos)} cases")


def expand_tags(tag_arg: str, mapping: dict, kind: str) -> list[str]:
    if tag_arg == "all":
        return list(mapping.keys())
    tags = tag_arg.split(",")
    unknown = [t for t in tags if t not in mapping]
    if unknown:
        raise SystemExit(f"Unknown {kind}(s) {unknown}. Must be one of: {', '.join(mapping)}, all")
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="olmo,qwen3,llama",
                         help=f"model tag(s), comma-separated, or 'all' ({', '.join(MODEL_DIRS)})")
    parser.add_argument("--dataset", default="all",
                         help=f"dataset tag(s), comma-separated, or 'all' ({', '.join(DATASET_TASKS)})")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                         help="Root dir to save per-combo McNemar result CSVs")
    parser.add_argument("--simulate-n", type=int, nargs="+", default=None,
                         help="Instead of the real sample size, rescale each result's counts to these "
                              "hypothetical sample sizes (same pass-rate proportions, recomputed p-values)")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                         help="test (default): the held-out run selected on validation (unbiased). "
                              "val: the sweep's best-of conditions (selection-biased)")
    args = parser.parse_args()

    model_tags = expand_tags(args.model, MODEL_DIRS, "model")
    dataset_tags = expand_tags(args.dataset, DATASET_TASKS, "dataset")
    output_root = Path(args.output_dir)
    # Encodes which split these results are from in every output filename (val vs
    # test read different input CSVs and must never overwrite each other's output).
    stem = SCOPE if args.split == "val" else f"{SCOPE}_test"
    # Collects every combo's rows so the alpha/n header, per-dataset tables, and the
    # compiled summary.csv can all be built after the loop below. Real and simulated
    # results are kept separate (sim_all_rows is one list per --simulate-n target) so
    # a simulated run is never mixed into the real summary.csv.
    all_rows = []
    sim_all_rows = {target_n: [] for target_n in (args.simulate_n or [])}

    for dataset_tag in dataset_tags:
        task = DATASET_TASKS[dataset_tag]

        for model_tag in model_tags:
            model_dir = MODEL_DIRS[model_tag]
            csv_path = PASS_RESULTS_ROOT / model_dir / task / NORM_MODE / STREAM_MODE / f"{stem}.csv"

            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            rows = [r for r in (run_mcnemar_pair(df, a, b) for a, b in COMPARISONS) if r is not None]
            if not rows:
                continue

            if args.simulate_n:
                for target_n in args.simulate_n:
                    sim_rows = [simulate_at_n(r, target_n) for r in rows]
                    sim_all_rows[target_n].extend(
                        {**r, "model_tag": model_tag, "dataset_tag": dataset_tag,
                         "model_dir": model_dir, "task": task} for r in sim_rows
                    )

                    if output_root:
                        out_path = output_root / model_dir / task / f"{stem}_mcnemar_simn{target_n}.csv"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(sim_rows).to_csv(out_path, index=False)
                continue

            all_rows.extend({**r, "model_tag": model_tag, "dataset_tag": dataset_tag,
                              "model_dir": model_dir, "task": task} for r in rows)

            results_df = pd.DataFrame(rows)
            if output_root:
                out_path = output_root / model_dir / task / f"{stem}_mcnemar.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(out_path, index=False)

    if args.simulate_n:
        for target_n in args.simulate_n:
            print_summary(sim_all_rows[target_n], dataset_tags, f"{target_n} (simulated)")
            print()
    elif all_rows:
        distinct_ns = sorted({r["n"] for r in all_rows})
        n_str = str(distinct_ns[0]) if len(distinct_ns) == 1 else f"{distinct_ns[0]}-{distinct_ns[-1]}"
        print_summary(all_rows, dataset_tags, n_str)
    else:
        print(f"alpha={ALPHA}  sided=one-sided (H1: mean/positional beats last)")

    if output_root and all_rows:
        # Rebuilt from every combo found on disk each time (not merged incrementally),
        # so concurrent/repeated runs can only add rows, never lose ones already compiled.
        # Written atomically (temp file + rename) so a torn write is never observable.
        summary_df = pd.DataFrame(all_rows)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / f"{stem}_mcnemar_summary.csv"
        tmp_path = summary_path.with_suffix(".csv.tmp")
        summary_df.to_csv(tmp_path, index=False)
        tmp_path.replace(summary_path)


if __name__ == "__main__":
    main()
