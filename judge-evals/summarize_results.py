"""
Summarize judge accuracy results into a CSV and heatmap visualizations.

Reads all *_gen_accuracy_{wo_rf,w_rf}.json.accuracy.json files from the
accuracy directory, collects them into a flat CSV, then produces per-
(model, dataset) heatmaps with axes N × topk, one subplot per steering combo.

Usage:
    python summarize_results.py [--accuracy_dir DIR]
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import BASE_DIR

ACCURACY_DIR = BASE_DIR / "judge-evals" / "accuracy"

ACC_RE = re.compile(
    r"^(?P<N>\d+)_(?P<REPS>random|targeted)_(?P<STEERING_METHOD>steer|mean)"
    r"_topk_(?P<topk>[\d.]+)"
    r"_(?P<STEERING_TYPE>[^_]+)"
    r"_(?P<STEERING_POS>[^_]+)"
    r"_gen_accuracy_(?P<rf_type>wo_rf|w_rf)\.json\.accuracy\.json$"
)


def collect_records(accuracy_dir: Path) -> pd.DataFrame:
    pattern = str(accuracy_dir / "**" / "*.json.accuracy.json")
    records = []
    for path in glob.glob(pattern, recursive=True):
        p = Path(path)
        m = ACC_RE.match(p.name)
        if not m:
            continue
        parts = p.parts
        try:
            acc_idx = parts.index("accuracy")
        except ValueError:
            continue
        if acc_idx + 4 >= len(parts):
            continue
        model_id = parts[acc_idx + 1]
        from_to  = parts[acc_idx + 2]
        ft = re.match(r"^from_(.+)_to_(.+)$", from_to)
        if not ft:
            continue
        source, base = ft.group(1), ft.group(2)
        with open(path) as f:
            data = json.load(f)
        records.append({
            "model":         model_id,
            "source":        source,
            "base":          base,
            "dataset":       f"{source} → {base}",
            "N":             int(m.group("N")),
            "topk":          float(m.group("topk")),
            "steering_type": m.group("STEERING_TYPE"),
            "steering_pos":  m.group("STEERING_POS"),
            "rf_type":       m.group("rf_type"),
            "pass_rate":     data.get("q1", float("nan")),
        })
    return pd.DataFrame(records)


COMBO_LABELS = {
    "last-token\nall-tokens":  "type: last-token\npos: all-tokens",
    "last-token\nlast-token":  "type: last-token\npos: last-token",
    "mean\nall-tokens":        "type: mean\npos: all-tokens",
    "mean\nlast-token":        "type: mean\npos: last-token",
    "positional\nall-tokens":  "type: positional\npos: all-tokens",
}

LEGEND_TEXT = (
    "wo_rf — judge pass rate without quality filter\n"
    "w_rf — judge pass rate with quality filter (fluency + relevance = 2)"
)


def make_heatmaps(df: pd.DataFrame, accuracy_dir: Path):
    combos = sorted(
        (df["steering_type"] + "\n" + df["steering_pos"]).unique()
    )
    combo_display = [COMBO_LABELS.get(c, c) for c in combos]

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        source, base = dataset.split(" → ")
        short_model  = next(
            (n for n in ("OLMo", "Qwen", "SOLAR") if n.lower() in model.lower()),
            model.split("/")[-1],
        )
        short_source = re.sub(r"-(long|single)$", "", source)
        short_title  = f"{short_model}  |  {short_source} → {base}"

        n_vals    = sorted(group["N"].unique())
        topk_vals = sorted(group["topk"].unique())
        n_rows, n_cols = len(n_vals), len(topk_vals)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5 * n_cols, (0.7 * len(combos) + 1.2) * n_rows + 0.6),
            squeeze=False,
        )
        fig.suptitle(short_title, fontsize=12)
        fig.text(
            0.5, -0.02, LEGEND_TEXT,
            ha="center", va="top", fontsize=8,
            family="monospace",
            transform=fig.transFigure,
        )

        for row_i, N in enumerate(n_vals):
            for col_i, topk in enumerate(topk_vals):
                ax = axes[row_i][col_i]
                subgroup = group[(group["N"] == N) & (group["topk"] == topk)]

                matrix = pd.DataFrame(np.nan, index=combos, columns=["wo_rf", "w_rf"])
                for _, r in subgroup.iterrows():
                    combo = f"{r.steering_type}\n{r.steering_pos}"
                    matrix.loc[combo, r["rf_type"]] = r["pass_rate"]
                matrix.index = combo_display

                sns.heatmap(
                    matrix,
                    ax=ax,
                    vmin=0, vmax=1,
                    annot=False,
                    cmap="YlGn",
                    linewidths=0.5,
                    cbar=(col_i == n_cols - 1),
                )
                # Annotate manually: seaborn 0.12 + matplotlib >= 3.8 misaligns
                # get_facecolors() against masked cells, dropping some labels.
                for r_i in range(matrix.shape[0]):
                    for c_i in range(matrix.shape[1]):
                        val = matrix.iat[r_i, c_i]
                        if pd.isna(val):
                            continue
                        ax.text(
                            c_i + 0.5, r_i + 0.5, f"{val:.2f}",
                            ha="center", va="center",
                            color="black", fontsize=10,
                        )
                ax.set_title(f"N={N}  topk={topk}", fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_yticklabels(
                    ax.get_yticklabels() if col_i == 0 else [],
                    rotation=0, fontsize=8,
                )

        plt.tight_layout()
        fig_path = accuracy_dir / f"heatmap_{short_model}_{short_source}_to_{base}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap: {fig_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy_dir", default=str(ACCURACY_DIR))
    args = parser.parse_args()

    accuracy_dir = Path(args.accuracy_dir)

    df = collect_records(accuracy_dir)
    if df.empty:
        print("No accuracy files found.")
        return

    csv_path = accuracy_dir / "results_summary.csv"
    df.sort_values(["model", "dataset", "N", "topk", "steering_type", "steering_pos", "rf_type"]) \
      .to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path.name}  ({len(df)} rows)")

    make_heatmaps(df, accuracy_dir)


if __name__ == "__main__":
    main()
