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

ACCURACY_DIR          = BASE_DIR / "judge-evals" / "accuracy"
ACCURACY_RESIDUAL_DIR = BASE_DIR / "judge-evals" / "accuracy_residual"

SOURCE_TO_TAG = {
    "harmful-long":                    "harmful",
    "non-sycophantic-long":            "sycophancy",
    "verse-long":                      "verse",
    "paragraph-long":                  "paragraph",
    "non-sycophantic-haiku-long":       "sycophancy-haiku",
    "non-sycophantic-poem-long":        "sycophancy-poem",
    "non-sycophantic-haiku-concise-long": "sycophancy-haiku-concise",
    "non-sycophantic-poem-concise-long":  "sycophancy-poem-concise",
}

CANONICAL_SYCOPHANCY_BASES = {
    "sycophancy-haiku",
    "sycophancy-poem",
    "sycophancy-haiku-concise",
    "sycophancy-poem-concise",
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


def make_heatmaps(df: pd.DataFrame, figures_dir: Path, diff: bool = False, norm_label: str = "normalized", cache_suffix: str = "", compare_df: pd.DataFrame = None, padding: bool = False, resid: bool = False):
    """
    One figure per (model, dataset).
    Grid rows = cache_mode + wo_rf (or cache comparison when compare_df provided), one heatmap per row.
    Each heatmap: rows = steering_type, columns = N (topk is fixed at 1.0).
    """
    import matplotlib as mpl

    cache_modes    = sorted(df["cache_mode"].unique())
    steering_types = sorted(df["steering_type"].unique())
    cmap = "RdYlGn" if diff else "YlGn"
    vmin, vmax = (-1, 1) if diff else (0, 1)

    compare_groups = {}
    if compare_df is not None:
        for (model, dataset), grp in compare_df.groupby(["model", "dataset"]):
            compare_groups[(model, dataset)] = grp

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        source, base = dataset.split(" → ")
        short_model  = next(
            (n for n in ("OLMo", "Qwen3", "Qwen", "Gemma", "Llama") if n.lower() in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))
        stream_label = "residual" if resid else "attention"
        short_title  = f"{short_model}  |  {short_source}  |  {stream_label}  (w_rf)"

        n_vals = sorted(group["N"].unique())
        n_rows = len(cache_modes) + 1  # cache_mode rows + extra row
        n_cols = 1

        w_rf_max = group.groupby("steering_type")["pass_rate"].max()
        best_steers = (set(w_rf_max[w_rf_max == w_rf_max.max()].index)
                       if not w_rf_max.empty else set())

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.2 + 0.7 * len(n_vals),
                     1.0 + 0.5 * len(steering_types) * n_rows),
            squeeze=False,
            gridspec_kw={"hspace": 0.35},
        )
        fig.suptitle(short_title, fontsize=16)

        compare_group = compare_groups.get((model, dataset))
        if compare_group is not None:
            main_label    = "normal"  if padding else cache_modes[0] if len(cache_modes) == 1 else None
            compare_label = "padding" if padding else "cache"
            all_rows = (
                [(cache, "pass_rate", group, main_label if padding else cache) for cache in cache_modes] +
                [(None, "pass_rate", compare_group, compare_label)]
            )
        else:
            all_rows = (
                [(cache, "pass_rate", group, "w_rf") for cache in cache_modes] +
                [(None, "pass_rate_wo_rf", group, "wo_rf")]
            )

        for row_i, (cache, val_col, src, row_label) in enumerate(all_rows):
            is_last_row = row_i == n_rows - 1
            ax  = axes[row_i][0]
            sub = src if cache is None else src[src["cache_mode"] == cache]

            matrix = (
                sub.pivot_table(index="steering_type", columns="N", values=val_col)
                   .reindex(index=steering_types, columns=n_vals)
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

            ax.set_title(row_label, fontsize=12, fontweight="bold", loc="left", pad=6)
            ax.set_ylabel("")
            ax.set_yticklabels(steering_types, rotation=0, fontsize=10)
            for tick in ax.get_yticklabels():
                if tick.get_text() in best_steers:
                    tick.set_fontweight("bold")

            if is_last_row:
                ax.set_xticklabels([str(n) for n in n_vals], rotation=0, fontsize=10)
                ax.set_xlabel("N", fontsize=11)
            else:
                ax.set_xticklabels([""] * len(n_vals))
                ax.set_xlabel("")
                ax.tick_params(axis="x", bottom=False)

        sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        fig.colorbar(sm, cax=cax)
        fig.text(0.5, -0.02, f"{norm_label}  |  topk=1.0  |  bold = best",
                 ha="center", va="top", fontsize=10, family="monospace")

        suffix   = "_diff" if diff else ""
        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}{suffix}{cache_suffix}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")


def make_simple_heatmaps(df: pd.DataFrame, figures_dir: Path, diff: bool = False, norm_label: str = "normalized", cache_suffix: str = ""):
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

        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}{cache_suffix}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")


def generate_for_stream(args, resid: bool, norm_mode: str, cache_suffix: str,
                         cache_mode: str):
    figures_root = BASE_DIR / "figures"
    if resid:
        full_dir = figures_root / "residuals"
        csv_path = ACCURACY_RESIDUAL_DIR / "results_summary.csv"
    else:
        full_dir = figures_root / ("attention-padding" if args.padding else "attention")
        csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}  —  run summarize_results.py first  "
              f"(skipping {'residual' if resid else 'attention'} results)")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    full_dir.mkdir(parents=True, exist_ok=True)

    if args.sycophancy:
        base_mask = df["base"].isin(CANONICAL_SYCOPHANCY_BASES)
    else:
        CANONICAL_BASES = {"harmless", "sycophancy", "prose"}
        base_mask = df["base"].apply(lambda b: b.split("_")[-1] in CANONICAL_BASES)
    base_df = df[
        base_mask &
        (df["norm_mode"] == norm_mode) &
        (df["cache_mode"] == cache_mode)
    ]

    compare_df = None
    if resid:
        pass  # no compare_df for resid mode
    elif args.nocache:
        compare_df = df[
            base_mask &
            (df["norm_mode"] == norm_mode) &
            (df["cache_mode"] == "cache")
        ]
    elif args.padding:
        padding_csv = ACCURACY_PADDING_DIR / "results_summary.csv"
        if not padding_csv.exists():
            print(f"Padding CSV not found: {padding_csv}  —  run summarize_results.py with accuracy_padding/ first")
            return
        padding_raw = pd.read_csv(padding_csv)
        if args.sycophancy:
            pad_mask = padding_raw["base"].isin(CANONICAL_SYCOPHANCY_BASES)
        else:
            CANONICAL_BASES = {"harmless", "sycophancy", "prose"}
            pad_mask = padding_raw["base"].apply(lambda b: b.split("_")[-1] in CANONICAL_BASES)
        compare_df = padding_raw[
            pad_mask &
            (padding_raw["norm_mode"] == norm_mode) &
            (padding_raw["cache_mode"] == cache_mode)
        ]
        common_pairs = compare_df[["model", "dataset"]].drop_duplicates()
        base_df = base_df.merge(common_pairs, on=["model", "dataset"], how="inner")

    base_df = base_df[base_df["topk"] == 1.0]
    if compare_df is not None:
        compare_df = compare_df[compare_df["topk"] == 1.0]

    norm_label = norm_mode
    make_heatmaps(base_df, full_dir, diff=args.diff, norm_label=norm_label, cache_suffix=cache_suffix, compare_df=compare_df, padding=args.padding, resid=resid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ACCURACY_DIR / "results_summary.csv"),
                        help="Path to results_summary.csv from summarize_results.py")
    parser.add_argument("--diff", action="store_true",
                        help="Plot other−last difference instead of raw pass rates")
    parser.add_argument("--unnormalized", action="store_true",
                        help="Plot unnormalized data; outputs to figures_unnormalized / figures_full_unnormalized")
    parser.add_argument("--norm_mode", default=None,
                        choices=["normalized", "unnormalized"],
                        help="Which normalization condition to plot (overridden by --unnormalized)")
    parser.add_argument("--cache_mode", default="cache",
                        choices=["cache", "no_cache"],
                        help="Which cache condition to plot (default: cache)")
    parser.add_argument("--sycophancy", action="store_true",
                        help="Plot only non-sycophantic haiku and poem datasets")
    parser.add_argument("--nocache", action="store_true",
                        help="Plot no-cache results; appends _nocache to output filenames")
    parser.add_argument("--padding", action="store_true",
                        help="Compare accuracy/ (top row) vs accuracy_padding/ (bottom row)")
    parser.add_argument("--resid", action="store_true",
                        help="Plot only residual-stream results from accuracy_residual/ "
                             "(by default both attention and residual results are plotted)")
    args = parser.parse_args()

    norm_mode     = "unnormalized" if args.unnormalized else (args.norm_mode or "normalized")
    cache_suffix  = "_nocache" if args.nocache else ""
    cache_mode    = "no_cache" if args.nocache else args.cache_mode

    stream_modes = [True] if args.resid else [False, True]
    for resid in stream_modes:
        generate_for_stream(args, resid, norm_mode, cache_suffix, cache_mode)


if __name__ == "__main__":
    main()
