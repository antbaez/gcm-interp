"""
Summarize judge accuracy results into a CSV and heatmap visualizations.

Reads judge_ratings.jsonl, fluency_ratings.jsonl, and relevance_ratings.jsonl
from the workdirs directory tree, computes per-condition w_rf pass rates, and
produces per-(model, dataset) heatmaps comparing cache/no_cache × last/positional.

Usage:
    python summarize_results.py [--workdirs_dir DIR] [--accuracy_dir DIR] [--diff]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import BASE_DIR

WORKDIRS_JSONL = BASE_DIR / "judge-evals" / "workdirs"
ACCURACY_DIR   = BASE_DIR / "judge-evals" / "accuracy"

SOURCE_TO_TAG = {
    "harmful-long":         "harmful",
    "non-sycophantic-long": "sycophancy",
    "verse-long":           "verse",
    "paragraph-long":       "paragraph",
}


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
        # Expected layout: (model, from_to, cache_mode, steering_type, exp_dir, filename)
        if len(parts) < 6:
            continue

        model         = parts[0]
        from_to       = parts[1]
        cache_mode    = parts[2]   # "cache" or "no_cache"
        steering_type = parts[3]   # e.g. "positional", "last-token"

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

        w_passes = []
        for rec in judge_recs:
            jp_rating = rec.get("judge_rating")
            jp_pass   = bool(jp_rating == 5)
            query     = rec.get("data_path_query", "")
            flu       = flu_by_query.get(query)
            rel       = rel_by_query.get(query)
            w_passes.append(jp_pass and flu == 2 and rel == 2)

        n = len(w_passes)
        condition = f"{cache_mode} / {steering_type}"
        records.append(dict(
            model=model, source=source, base=base,
            dataset=f"{source} → {base}",
            N=int(N), topk=float(topk),
            cache_mode=cache_mode,
            steering_type=steering_type,
            condition=condition,
            pass_rate=sum(w_passes) / n,
        ))

    return pd.DataFrame(records)


LAST_TOKEN_NAMES = {"last", "last-token", "last_token"}

def compute_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-last pass_rates with (other − last), per cache_mode."""
    join_cols = ["model", "dataset", "source", "base", "N", "topk", "cache_mode"]
    last = (
        df[df["steering_type"].isin(LAST_TOKEN_NAMES)]
        .set_index(join_cols)["pass_rate"]
        .rename("last_pass_rate")
    )
    out = df.join(last, on=join_cols)
    mask = ~out["steering_type"].isin(LAST_TOKEN_NAMES)
    out.loc[mask, "pass_rate"] = out.loc[mask, "pass_rate"] - out.loc[mask, "last_pass_rate"]
    return out[mask].drop(columns=["last_pass_rate"])


def make_heatmaps(df: pd.DataFrame, figures_dir: Path, diff: bool = False, norm_label: str = "normalized"):
    """
    One figure per (model, dataset).
    Grid rows = cache_mode, grid columns = steering_type.
    Each subplot shows N (y-axis) × topk (x-axis) pass rates.
    """
    import matplotlib as mpl

    cache_modes    = sorted(df["cache_mode"].unique())     # rows
    steering_types = sorted(df["steering_type"].unique())  # columns
    cmap = "RdYlGn" if diff else "YlGn"
    vmin, vmax = (-1, 1) if diff else (0, 1)

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        source, base = dataset.split(" → ")
        short_model  = next(
            (n for n in ("OLMo", "Qwen3", "Qwen", "Gemma", "Llama") if n.lower() in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))
        mode_label   = "diff (other − last)" if diff else norm_label
        short_title  = f"{short_model}  |  {short_source}  |  {mode_label}  (w_rf)"

        n_vals    = sorted(group["N"].unique())
        topk_vals = sorted(group["topk"].unique())
        topk_labels = [str(t) for t in topk_vals]
        n_rows = len(cache_modes)
        n_cols = len(steering_types)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(0.8 + 0.75 * len(topk_vals) * n_cols,
                     1.2 + 0.55 * len(n_vals) * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(short_title, fontsize=16)

        for row_i, cache in enumerate(cache_modes):
            for col_i, steer in enumerate(steering_types):
                ax  = axes[row_i][col_i]
                sub = group[(group["cache_mode"] == cache) &
                            (group["steering_type"] == steer)]

                matrix = (
                    sub.pivot_table(index="N", columns="topk", values="pass_rate")
                       .reindex(index=n_vals, columns=topk_vals)
                )

                sns.heatmap(matrix, ax=ax, vmin=vmin, vmax=vmax,
                            annot=False, cmap=cmap, linewidths=0.5, cbar=False)

                for r_i in range(matrix.shape[0]):
                    for c_i in range(matrix.shape[1]):
                        val = matrix.iat[r_i, c_i]
                        if pd.isna(val):
                            continue
                        ax.text(c_i + 0.5, r_i + 0.5, f"{val:.2f}",
                                ha="center", va="center", color="black", fontsize=9)

                # Column header (steering type) on top row only
                if row_i == 0:
                    ax.set_title(steer, fontsize=13, fontweight="bold", pad=6)
                # Row header (cache mode) on left column only
                if col_i == 0:
                    ax.set_ylabel(cache, fontsize=12, fontweight="bold", labelpad=8)
                else:
                    ax.set_ylabel("")
                ax.set_yticklabels(n_vals, rotation=0, fontsize=10)
                ax.set_xticklabels(
                    topk_labels if row_i == n_rows - 1 else [""] * len(topk_vals),
                    rotation=45, ha="right", fontsize=10,
                )
                ax.set_xlabel("topk" if row_i == n_rows - 1 else "", fontsize=11)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)

        fig.text(0.995, 1.01, "rows = N\ncolumns = topk",
                 ha="right", va="top", fontsize=11, family="monospace", clip_on=False,
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        suffix   = "_diff" if diff else ""
        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}{suffix}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap: {fig_path.name}")


def make_simple_heatmaps(df: pd.DataFrame, figures_simple_dir: Path, diff: bool = False, norm_label: str = "normalized"):
    """
    One figure per (model, dataset): rows = conditions, columns = topk.
    Each cell = max pass rate over all N values (w_rf).
    """
    import matplotlib as mpl

    cmap = "RdYlGn" if diff else "YlGn"
    vmin, vmax = (-1, 1) if diff else (0, 1)

    max_over_n = (
        df.groupby(
            ["model", "dataset", "source", "base",
             "cache_mode", "steering_type", "condition", "topk"],
            as_index=False,
        )["pass_rate"].max()
    )

    if diff:
        max_over_n = compute_diff(max_over_n.assign(N=0))
        max_over_n = max_over_n.drop(columns=["N"], errors="ignore")

    combos = sorted(max_over_n["condition"].unique())

    for (model, dataset), group in max_over_n.groupby(["model", "dataset"]):
        source, base = dataset.split(" → ")
        short_model  = next(
            (n for n in ("OLMo", "Qwen3", "Qwen", "Gemma", "Llama") if n.lower() in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))
        mode_label   = "diff (other − last)" if diff else norm_label
        short_title  = f"{short_model}  |  {short_source}  |  {mode_label}  (max over N, w_rf)"

        topk_vals   = sorted(group["topk"].unique())
        topk_labels = [str(t) for t in topk_vals]

        matrix = (
            group.pivot_table(index="condition", columns="topk", values="pass_rate")
                 .reindex(index=combos, columns=topk_vals)
        )

        fig, ax = plt.subplots(
            figsize=(1.2 + 0.65 * len(topk_vals), 0.6 + 0.55 * len(combos)),
            constrained_layout=True,
        )
        fig.suptitle(short_title, fontsize=14)

        sns.heatmap(matrix, ax=ax, vmin=vmin, vmax=vmax,
                    annot=False, cmap=cmap, linewidths=0.5, cbar=False)

        for r_i in range(matrix.shape[0]):
            for c_i in range(matrix.shape[1]):
                val = matrix.iat[r_i, c_i]
                if pd.isna(val):
                    continue
                ax.text(c_i + 0.5, r_i + 0.5, f"{val:.2f}",
                        ha="center", va="center", color="black", fontsize=9)

        ax.set_yticklabels(combos, rotation=0, fontsize=11)
        ax.set_xticklabels(topk_labels, rotation=45, ha="right", fontsize=11)
        ax.set_xlabel("topk", fontsize=12)
        ax.set_ylabel("")

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)

        suffix   = "_diff" if diff else ""
        fig_path = figures_simple_dir / f"heatmap_{short_model}_{short_source}_simple{suffix}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap: {fig_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdirs_dir", default=str(WORKDIRS_JSONL))
    parser.add_argument("--accuracy_dir", default=str(ACCURACY_DIR),
                        help="Directory for the CSV summary and heatmap PNGs")
    parser.add_argument("--diff", action="store_true",
                        help="Plot other−last difference instead of raw pass rates")
    args = parser.parse_args()

    workdirs_dir = Path(args.workdirs_dir)
    accuracy_dir = Path(args.accuracy_dir)
    if args.diff:
        accuracy_dir = accuracy_dir.parent / f"{accuracy_dir.name}-diff"
    figures_dir        = accuracy_dir / "figures"
    figures_simple_dir = accuracy_dir / "figures_simple"
    for d in (accuracy_dir, figures_dir, figures_simple_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = collect_records(workdirs_dir)
    if df.empty:
        print("No rating files found.")
        return

    csv_path = accuracy_dir / "results_summary.csv"
    df.sort_values(["model", "dataset", "cache_mode", "steering_type", "N", "topk"]) \
      .to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path.name}  ({len(df)} rows)")

    CANONICAL_BASES = {"harmless", "sycophancy", "prose"}
    canonical = df["base"].apply(lambda b: b.split("_")[-1] in CANONICAL_BASES)
    plot_df = compute_diff(df[canonical]) if args.diff else df[canonical]
    make_heatmaps(plot_df, figures_dir, diff=args.diff)
    make_simple_heatmaps(df[canonical], figures_simple_dir, diff=args.diff)


if __name__ == "__main__":
    main()
