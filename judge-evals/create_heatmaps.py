"""
Generate heatmap figures from a results_summary.csv produced by summarize_results.py.

Usage:
    python create_heatmaps.py [--csv PATH] [--diff]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import BASE_DIR

ACCURACY_DIR = BASE_DIR / "judge-evals" / "accuracy"

SOURCE_TO_TAG = {
    "harmful-long":         "harmful",
    "non-sycophantic-long": "sycophancy",
    "verse-long":           "verse",
    "paragraph-long":       "paragraph",
}

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
    Grid rows = cache_mode + wo_rf, grid columns = steering_type.
    Each subplot shows N (y-axis) × topk (x-axis) pass rates.
    """
    import matplotlib as mpl

    cache_modes    = sorted(df["cache_mode"].unique())
    steering_types = sorted(df["steering_type"].unique())
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
        n_rows = len(cache_modes) + 1  # cache_mode rows + wo_rf row
        n_cols = len(steering_types)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(0.8 + 0.75 * len(topk_vals) * n_cols,
                     1.2 + 0.55 * len(n_vals) * n_rows),
            squeeze=False,
            gridspec_kw={"hspace": 0.1},
        )
        fig.suptitle(short_title, fontsize=16)

        all_rows = [(cache, "pass_rate") for cache in cache_modes] + [(None, "pass_rate_wo_rf")]

        for row_i, (cache, val_col) in enumerate(all_rows):
            is_worf_row = cache is None
            is_last_row = row_i == n_rows - 1
            for col_i, steer in enumerate(steering_types):
                ax = axes[row_i][col_i]
                sub = (group[group["steering_type"] == steer] if is_worf_row
                       else group[(group["cache_mode"] == cache) &
                                  (group["steering_type"] == steer)])

                matrix = (
                    sub.pivot_table(index="N", columns="topk", values=val_col)
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

                if row_i == 0:
                    ax.set_title(steer, fontsize=13, fontweight="bold", pad=6)
                if col_i == 0:
                    ax.set_ylabel("wo_rf" if is_worf_row else "w_rf",
                                  fontsize=12, fontweight="bold", labelpad=8)
                else:
                    ax.set_ylabel("")
                ax.set_yticklabels(n_vals, rotation=0, fontsize=10)
                ax.set_xticklabels(
                    topk_labels if is_last_row else [""] * len(topk_vals),
                    rotation=45, ha="right", fontsize=10,
                )
                ax.set_xlabel("topk" if is_last_row else "", fontsize=11)

        sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        cax = fig.add_axes([0.92, 0.1, 0.012, 0.8])  # [left, bottom, width, height] in figure coords
        fig.colorbar(sm, cax=cax)
        fig.text(0.995, 1.01, "rows = N\ncolumns = topk",
                 ha="right", va="top", fontsize=11, family="monospace", clip_on=False,
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        suffix   = "_diff" if diff else ""
        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}_full{suffix}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")


def make_simple_heatmaps(df: pd.DataFrame, figures_dir: Path, diff: bool = False, norm_label: str = "normalized"):
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

        combo_labels = [c.split(" / ", 1)[-1] for c in combos]

        fig = plt.figure(figsize=(1.2 + 0.65 * len(topk_vals), 0.6 + 0.4 * len(combos)))
        fig.suptitle(short_title, fontsize=14)
        gs = mpl.gridspec.GridSpec(len(combos), 1, figure=fig, hspace=0.1, top=0.82)
        axes = [fig.add_subplot(gs[i]) for i in range(len(combos))]

        for i, (combo, label) in enumerate(zip(combos, combo_labels)):
            ax = axes[i]
            row_df = (matrix.loc[[combo]] if combo in matrix.index
                      else pd.DataFrame([[np.nan] * len(topk_vals)],
                                        columns=topk_vals, index=[combo]))

            sns.heatmap(row_df, ax=ax, vmin=vmin, vmax=vmax,
                        annot=False, cmap=cmap, linewidths=0.5, cbar=False)

            for c_i in range(row_df.shape[1]):
                val = row_df.iat[0, c_i]
                if pd.isna(val):
                    continue
                ax.text(c_i + 0.5, 0.5, f"{val:.2f}",
                        ha="center", va="center", color="black", fontsize=9)

            ax.set_yticklabels([label], rotation=0, fontsize=11)
            ax.set_ylabel("")
            if i == len(combos) - 1:
                ax.set_xticklabels(topk_labels, rotation=45, ha="right", fontsize=11)
                ax.set_xlabel("topk", fontsize=12)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")
                ax.tick_params(axis="x", bottom=False)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)

        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ACCURACY_DIR / "results_summary.csv"),
                        help="Path to results_summary.csv from summarize_results.py")
    parser.add_argument("--diff", action="store_true",
                        help="Plot other−last difference instead of raw pass rates")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}  —  run summarize_results.py first")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    figures_dir      = BASE_DIR / "figures"
    figures_full_dir = BASE_DIR / "figures_full"
    for d in (figures_dir, figures_full_dir):
        d.mkdir(parents=True, exist_ok=True)

    CANONICAL_BASES = {"harmless", "sycophancy", "prose"}
    canonical = df["base"].apply(lambda b: b.split("_")[-1] in CANONICAL_BASES)
    base_df = df[canonical & (df["cache_mode"] == "cache")]

    make_heatmaps(base_df, figures_full_dir, diff=False)
    make_simple_heatmaps(base_df, figures_dir, diff=args.diff)


if __name__ == "__main__":
    main()
