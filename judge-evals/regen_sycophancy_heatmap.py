"""
Regenerate sycophancy heatmaps only, reading from results_summary.csv.
Filters to source=non-sycophantic-long, base=sycophancy for both models.
Outputs heatmap PNGs with _new suffix.
"""

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

RF_TITLES = {
    "wo_rf": "wo_rf (no quality filter)",
    "w_rf":  "w_rf (quality filter)",
}
RF_ORDER = ["wo_rf", "w_rf"]


def make_heatmaps_new(df: pd.DataFrame, accuracy_dir: Path):
    import matplotlib as mpl

    combos = sorted(df["steering_type"].unique())

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        source, base = dataset.split(" → ")
        short_model  = next(
            (n for n in ("OLMo", "Qwen", "SOLAR") if n.lower() in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))
        short_title  = f"{short_model}  |  {short_source}"

        n_vals    = sorted(group["N"].unique())
        topk_vals = sorted(group["topk"].unique())
        topk_labels = [str(t) for t in topk_vals]
        n_rows = len(combos)
        n_cols = len(RF_ORDER)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.2 + 0.60 * len(topk_vals) * n_cols,
                     1.0 + 0.50 * len(n_vals) * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(short_title, fontsize=18)

        for row_i, combo in enumerate(combos):
            for col_i, rf in enumerate(RF_ORDER):
                ax = axes[row_i][col_i]
                sub = group[(group["steering_type"] == combo) &
                            (group["rf_type"] == rf)]

                matrix = (
                    sub.pivot_table(index="N", columns="topk", values="pass_rate")
                       .reindex(index=n_vals, columns=topk_vals)
                )

                sns.heatmap(
                    matrix,
                    ax=ax,
                    vmin=0, vmax=1,
                    annot=False,
                    cmap="YlGn",
                    linewidths=0.5,
                    cbar=False,
                )
                for r_i in range(matrix.shape[0]):
                    for c_i in range(matrix.shape[1]):
                        val = matrix.iat[r_i, c_i]
                        if pd.isna(val):
                            continue
                        ax.text(
                            c_i + 0.5, r_i + 0.5, f"{val:.2f}",
                            ha="center", va="center",
                            color="black", fontsize=9,
                        )

                if row_i == 0:
                    ax.set_title(RF_TITLES[rf], fontsize=15)
                ax.set_xlabel("", fontsize=13)
                if col_i == 0:
                    ax.set_ylabel(combo, fontsize=15, fontweight="bold", labelpad=8)
                else:
                    ax.set_ylabel("")
                ax.set_yticklabels(n_vals, rotation=0, fontsize=11)
                ax.set_xticklabels(topk_labels, rotation=45, ha="right", fontsize=11)

        sm = mpl.cm.ScalarMappable(
            cmap="YlGn", norm=mpl.colors.Normalize(vmin=0, vmax=1)
        )
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)

        fig.text(
            0.995, 1.01, "rows = N\ncolumns = topk",
            ha="right", va="top", fontsize=13, family="monospace",
            clip_on=False,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )

        fig_path = accuracy_dir / f"heatmap_{short_model}_{short_source}_new.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap: {fig_path.name}")


df = pd.read_csv(ACCURACY_DIR / "results_summary.csv")

mask = (df["source"] == "non-sycophantic-long") & (df["base"] == "sycophancy")
filtered = df[mask]

print(f"Rows selected: {len(filtered)}")
print(filtered[["model", "dataset", "N", "topk", "steering_type", "rf_type", "pass_rate"]].to_string(index=False))

make_heatmaps_new(filtered, ACCURACY_DIR)
