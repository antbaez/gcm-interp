"""
Summarize judge accuracy results into a CSV and heatmap visualizations.

Reads judge_ratings.jsonl, fluency_ratings.jsonl, and relevance_ratings.jsonl
from the workdirs_jsonl directory tree, computes per-condition pass rates, and
produces per-(model, dataset) heatmaps with axes N × topk, one subplot per
steering type.

Usage:
    python summarize_results.py [--workdirs_dir DIR]
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

WORKDIRS_JSONL = BASE_DIR / "judge-evals" / "workdirs_jsonl"


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
        # Expected layout: (model, from_to, method, "eval", steering_type, exp_dir, filename)
        if len(parts) < 7 or parts[3] != "eval":
            continue

        model         = parts[0]
        from_to       = parts[1]
        steering_type = parts[4]

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

        is_syco      = "sycophancy" in source
        flu_by_query = {r.get("data_path_query"): r.get("judge_rating") for r in flu_recs}
        rel_by_query = {r.get("data_path_query"): r.get("judge_rating") for r in rel_recs}

        wo_passes, w_passes = [], []
        for rec in judge_recs:
            jp_rating = rec.get("judge_rating")
            jp_pass   = bool((jp_rating == 3) if is_syco else (jp_rating == 5))
            query     = rec.get("data_path_query", "")
            flu       = flu_by_query.get(query)
            rel       = rel_by_query.get(query)
            wo_passes.append(jp_pass)
            w_passes.append(jp_pass and flu == 2 and rel == 2)

        n = len(wo_passes)
        base_rec = dict(
            model=model, source=source, base=base,
            dataset=f"{source} → {base}",
            N=int(N), topk=float(topk),
            steering_type=steering_type,
        )
        records.append({**base_rec, "rf_type": "wo_rf", "pass_rate": sum(wo_passes) / n})
        records.append({**base_rec, "rf_type": "w_rf",  "pass_rate": sum(w_passes)  / n})

    return pd.DataFrame(records)


RF_TITLES = {
    "wo_rf": "wo_rf (no quality filter)",
    "w_rf":  "w_rf (fluency + relevance = 2)",
}
RF_ORDER = ["wo_rf", "w_rf"]


def make_heatmaps(df: pd.DataFrame, accuracy_dir: Path):
    import matplotlib as mpl

    combos = sorted(df["steering_type"].unique())

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
        topk_labels = [str(t) for t in topk_vals]
        n_rows = len(combos)            # one block per steering type
        n_cols = len(RF_ORDER)          # wo_rf | w_rf

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.2 + 0.80 * len(topk_vals) * n_cols,
                     1.0 + 0.68 * len(n_vals) * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(short_title, fontsize=16)

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
                            color="black", fontsize=9,
                        )

                if row_i == 0:
                    ax.set_title(RF_TITLES[rf], fontsize=13)
                # Every block carries its own x-axis (ticks + label).
                ax.set_xlabel("topk", fontsize=11)
                if col_i == 0:
                    ax.set_ylabel(combo, fontsize=13, fontweight="bold",
                                  labelpad=8)
                else:
                    ax.set_ylabel("")
                ax.set_yticklabels(n_vals, rotation=0, fontsize=9)
                ax.set_xticklabels(topk_labels, rotation=45, ha="right",
                                   fontsize=9)

        # Shared colorbar on the right.
        sm = mpl.cm.ScalarMappable(
            cmap="YlGn", norm=mpl.colors.Normalize(vmin=0, vmax=1)
        )
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)

        # Axis-orientation legend, top-right.
        fig.text(
            0.995, 0.995, "rows = N\ncolumns = topk",
            ha="right", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )

        fig_path = accuracy_dir / f"heatmap_{short_model}_{short_source}_to_{base}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap: {fig_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdirs_dir", default=str(WORKDIRS_JSONL))
    args = parser.parse_args()

    workdirs_dir = Path(args.workdirs_dir)

    df = collect_records(workdirs_dir)
    if df.empty:
        print("No rating files found.")
        return

    csv_path = workdirs_dir / "results_summary.csv"
    df.sort_values(["model", "dataset", "N", "topk", "steering_type", "rf_type"]) \
      .to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path.name}  ({len(df)} rows)")

    make_heatmaps(df, workdirs_dir)


if __name__ == "__main__":
    main()
