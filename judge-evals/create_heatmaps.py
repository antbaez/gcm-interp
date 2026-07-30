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

# N values to drop from heatmaps, keyed by a lowercase substring matched against
# the "model" column; "default" applies to any model not otherwise listed.
HIDDEN_N_BY_MODEL = {
    "default": {2, 500},
    "gemma": set(),
    "qwen3": {2, 500},
    "llama": {2, 100, 250, 500},
}


def hidden_n_for_model(model: str) -> set:
    model_lower = model.lower()
    for key, hidden in HIDDEN_N_BY_MODEL.items():
        if key != "default" and key in model_lower:
            return hidden
    return HIDDEN_N_BY_MODEL["default"]


def filter_hidden_n(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose N value is hidden for that row's model (see HIDDEN_N_BY_MODEL)."""
    if df.empty or "N" not in df.columns:
        return df
    mask = ~df.apply(lambda r: r["N"] in hidden_n_for_model(r["model"]), axis=1)
    return df[mask]


def make_heatmaps(df: pd.DataFrame, figures_dir: Path, diff: bool = False, norm_label: str = "normalized", cache_suffix: str = "", compare_df: pd.DataFrame = None, padding: bool = False, resid: bool = False):
    """
    One figure per (model, dataset). Used for global runs (all layers steered).
    Grid rows = cache_mode + wo_rf (or cache comparison when compare_df provided), one heatmap per row.
    Each heatmap: rows = steering_type, columns = N.
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
        short_title  = f"{short_model}  |  {short_source}  (w_rf)"

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
        fig.suptitle(short_title, fontsize=18)

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
                            ha="center", va="center", color="black", fontsize=11)

            ax.set_title(row_label.capitalize(), fontsize=16, fontweight="bold", loc="left", pad=6)
            ax.set_ylabel("")
            ax.set_yticklabels(steering_types, rotation=0, fontsize=12)
            for tick in ax.get_yticklabels():
                if tick.get_text() in best_steers:
                    tick.set_fontweight("bold")

            if is_last_row:
                ax.set_xticklabels([f"{n:g}" for n in n_vals], rotation=0, fontsize=12)
                ax.set_xlabel("N", fontsize=15)
            else:
                ax.set_xticklabels([""] * len(n_vals))
                ax.set_xlabel("")
                ax.tick_params(axis="x", bottom=False)

        sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        cax = fig.add_axes([0.92, 0.1, 0.005, 0.8])
        fig.colorbar(sm, cax=cax)

        suffix   = "_diff" if diff else ""
        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}{suffix}{cache_suffix}_global.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")


def make_layer_heatmaps(df: pd.DataFrame, figures_dir: Path, norm_label: str = "normalized",
                        cache_suffix: str = "", resid: bool = False):
    """
    One figure per (model, dataset) for single-layer-sweep runs.
    One subplot per steering_type, laid out in a row; each heatmap has rows = layer,
    columns = N, cells = w_rf pass rate. Every swept layer is shown; the best layer per
    steering type (max pass rate over N) is bolded in that subplot's labels.
    """
    import matplotlib as mpl

    cmap = "YlGn"
    vmin, vmax = 0, 1
    steering_types = sorted(df["steering_type"].unique())

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        source, base = dataset.split(" → ")
        short_model  = next(
            (n for n in ("OLMo", "Qwen3", "Qwen", "Gemma", "Llama") if n.lower() in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))

        n_vals     = sorted(group["N"].unique())
        layer_vals = sorted(int(l) for l in group["layer"].dropna().unique())
        n_types    = len(steering_types)

        st_max = group.groupby("steering_type")["pass_rate"].max()
        best_steers = (set(st_max[st_max == st_max.max()].index)
                       if not st_max.empty else set())

        fig, axes = plt.subplots(
            1, n_types,
            figsize=(1.6 + 0.7 * len(n_vals) * n_types,
                     1.0 + 0.35 * len(layer_vals)),
            squeeze=False,
            gridspec_kw={"wspace": 0.2},
        )
        fig.suptitle(f"{short_model}  |  {short_source}  (w_rf)",
                     fontsize=17, y=1.02)

        for col_i, st in enumerate(steering_types):
            ax  = axes[0][col_i]
            sub = group[group["steering_type"] == st]
            matrix = (
                sub.pivot_table(index="layer", columns="N", values="pass_rate")
                   .reindex(index=layer_vals, columns=n_vals)
            )

            sns.heatmap(matrix, ax=ax, vmin=vmin, vmax=vmax,
                        annot=False, cmap=cmap, linewidths=0.5, cbar=False)

            for r_i in range(matrix.shape[0]):
                for c_i in range(matrix.shape[1]):
                    val = matrix.iat[r_i, c_i]
                    if pd.isna(val):
                        continue
                    ax.text(c_i + 0.5, r_i + 0.5, f"{val:.2f}",
                            ha="center", va="center", color="black", fontsize=10)

            best_layer = None
            best_n = None
            if not sub.empty:
                per_layer_max = sub.groupby("layer")["pass_rate"].max()
                if not per_layer_max.empty:
                    best_layer = int(per_layer_max.idxmax())
                    best_row = sub[sub["layer"] == best_layer]
                    best_n = best_row.loc[best_row["pass_rate"].idxmax(), "N"]

            ax.set_title(st.capitalize(), fontsize=15,
                         fontweight="bold" if st in best_steers else "normal",
                         loc="left", pad=6)

            # Layer ticks on every subplot; the "layer" axis label only on the leftmost.
            ax.set_ylabel("Layer (L)" if col_i == 0 else "", fontsize=14)
            ax.set_yticklabels([str(l) for l in layer_vals], rotation=0, fontsize=10)
            for tick in ax.get_yticklabels():
                if best_layer is not None and tick.get_text() == str(best_layer):
                    tick.set_fontweight("bold")

            ax.set_xticklabels([f"{n:g}" for n in n_vals], rotation=0, fontsize=11)
            ax.set_xlabel("Steering Coefficient (N)", fontsize=15, labelpad=12)
            for tick in ax.get_xticklabels():
                if best_n is not None and tick.get_text() == f"{best_n:g}":
                    tick.set_fontweight("bold")

        sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        cax = fig.add_axes([0.92, 0.1, 0.005, 0.8])
        fig.colorbar(sm, cax=cax)

        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}{cache_suffix}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")


def make_layer_heatmaps_agg(df: pd.DataFrame, figures_dir: Path, agg: str, norm_label: str = "normalized",
                            cache_suffix: str = "", resid: bool = False):
    """
    Like make_layer_heatmaps, but collapses the N axis into a single aggregated
    value per layer: agg='max' takes the max pass rate over N, agg='avg' takes
    the mean pass rate over N. One figure per (model, dataset); one subplot per
    steering_type; each heatmap has rows = layer, a single column = the aggregate.
    """
    import matplotlib as mpl

    cmap = "YlGn"
    vmin, vmax = 0, 1
    steering_types = sorted(df["steering_type"].unique())
    agg_fn    = "max" if agg == "max" else "mean"
    agg_label = "max over N" if agg == "max" else "avg over N"

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        source, base = dataset.split(" → ")
        short_model  = next(
            (n for n in ("OLMo", "Qwen3", "Qwen", "Gemma", "Llama") if n.lower() in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))

        layer_vals = sorted(int(l) for l in group["layer"].dropna().unique())
        n_types    = len(steering_types)

        st_max = group.groupby("steering_type")["pass_rate"].max()
        best_steers = (set(st_max[st_max == st_max.max()].index)
                       if not st_max.empty else set())

        n_rows_shown = len(layer_vals) + (1 if agg == "avg" else 0)
        fig, axes = plt.subplots(
            1, n_types,
            figsize=(1.6 + 1.2 * n_types, 1.0 + 0.35 * n_rows_shown),
            squeeze=False,
            gridspec_kw={"wspace": 0.25},
        )
        fig.suptitle(f"{short_model}  |  {short_source}  ({agg_label}, w_rf)",
                     fontsize=17)

        for col_i, st in enumerate(steering_types):
            ax  = axes[0][col_i]
            sub = group[group["steering_type"] == st]
            agg_series = sub.groupby("layer")["pass_rate"].agg(agg_fn).reindex(layer_vals)

            best_n_by_layer = {}
            if agg == "max" and not sub.empty:
                idx = sub.groupby("layer")["pass_rate"].idxmax()
                best_n_by_layer = sub.loc[idx].set_index("layer")["N"].to_dict()

            best_layer = None
            if not agg_series.dropna().empty:
                best_layer = int(agg_series.idxmax())

            row_labels = [str(l) for l in layer_vals]
            values     = list(agg_series.values)
            if agg == "avg":
                row_labels = row_labels + ["avg"]
                values     = values + [agg_series.mean()]

            matrix = pd.DataFrame({agg_label: values}, index=row_labels)

            sns.heatmap(matrix, ax=ax, vmin=vmin, vmax=vmax,
                        annot=False, cmap=cmap, linewidths=0.5, cbar=False)

            for r_i in range(matrix.shape[0]):
                val = matrix.iat[r_i, 0]
                if pd.isna(val):
                    continue
                best_n = None
                is_best_layer_row = agg == "max" and r_i < len(layer_vals) and layer_vals[r_i] == best_layer
                if agg == "max" and r_i < len(layer_vals):
                    best_n = best_n_by_layer.get(layer_vals[r_i])
                if best_n is None:
                    ax.text(0.5, r_i + 0.5, f"{val:.2f}",
                            ha="center", va="center", color="black", fontsize=10)
                else:
                    ax.text(0.5, r_i + 0.42, f"{val:.2f}",
                            ha="center", va="center", color="black", fontsize=10)
                    ax.text(0.5, r_i + 0.72, f"N={best_n:g}",
                            ha="center", va="center", color="black", fontsize=10 * 2 / 3,
                            fontweight="bold" if is_best_layer_row else "normal")

            if agg == "avg":
                ax.axhline(len(layer_vals), color="black", linewidth=1.5)

            ax.set_title(st.capitalize(), fontsize=15,
                         fontweight="bold" if st in best_steers else "normal",
                         loc="left", pad=6)

            ax.set_ylabel("Layer" if col_i == 0 else "", fontsize=14)
            ax.set_yticklabels(row_labels, rotation=0, fontsize=10)
            for tick in ax.get_yticklabels():
                if best_layer is not None and tick.get_text() == str(best_layer):
                    tick.set_fontweight("bold")
                elif tick.get_text() == "avg":
                    tick.set_fontweight("bold")

            ax.set_xticklabels([agg_label], rotation=0, fontsize=11)
            ax.set_xlabel("")

        sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        cax = fig.add_axes([0.92, 0.1, 0.005, 0.8])
        fig.colorbar(sm, cax=cax)

        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}_{agg}{cache_suffix}.png"
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
    base_df = filter_hidden_n(base_df)

    compare_df = None
    if resid:
        pass  # no compare_df for resid mode
    elif args.nocache:
        compare_df = df[
            base_mask &
            (df["norm_mode"] == norm_mode) &
            (df["cache_mode"] == "cache")
        ]
        compare_df = filter_hidden_n(compare_df)
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
        compare_df = filter_hidden_n(compare_df)
        common_pairs = compare_df[["model", "dataset"]].drop_duplicates()
        base_df = base_df.merge(common_pairs, on=["model", "dataset"], how="inner")

    # Runs are split by scope: `local` = single-layer sweep over the middle third
    # (layer-axis heatmaps), `global` = all layers at once (N × steering-type
    # heatmaps). Older summaries without a `scope` column fall back to the legacy
    # `layer`-present / `topk == 1.0` heuristic. Figures are separated on disk into
    # `<stream>/local/` and `<stream>/global/` subdirectories.
    has_scope = "scope" in base_df.columns
    local_dir  = full_dir / "local"
    global_dir = full_dir / "global"

    if has_scope:
        layer_source_df = base_df[(base_df["scope"] == "local") & base_df["layer"].notna()]
    elif "layer" in base_df.columns:
        layer_source_df = base_df[base_df["layer"].notna()]
        base_df = base_df[base_df["layer"].isna()]
    else:
        layer_source_df = base_df.iloc[0:0]
        print("no `layer` column in data  —  nothing to plot for local heatmaps")

    if not layer_source_df.empty:
        agg_requested = args.max or args.avg
        if not agg_requested:
            local_dir.mkdir(parents=True, exist_ok=True)
            make_layer_heatmaps(layer_source_df, local_dir, norm_label=norm_mode,
                                cache_suffix=cache_suffix, resid=resid)

        base_stream_name = "residuals" if resid else ("attention-padding" if args.padding else "attention")
        for agg, flag in (("max", args.max), ("avg", args.avg)):
            if not flag:
                continue
            agg_dir = figures_root / f"{base_stream_name}_{agg}" / "local"
            agg_dir.mkdir(parents=True, exist_ok=True)
            make_layer_heatmaps_agg(layer_source_df, agg_dir, agg=agg, norm_label=norm_mode,
                                     cache_suffix=cache_suffix, resid=resid)

    if not args.global_scope:
        return

    if has_scope:
        base_df = base_df[base_df["scope"] == "global"]
        if compare_df is not None and "scope" in compare_df.columns:
            compare_df = compare_df[compare_df["scope"] == "global"]
    else:
        base_df = base_df[base_df["topk"] == 1.0]
        if compare_df is not None:
            compare_df = compare_df[compare_df["topk"] == 1.0]

    if base_df.empty:
        return

    norm_label = norm_mode
    global_dir.mkdir(parents=True, exist_ok=True)
    make_heatmaps(base_df, global_dir, diff=args.diff, norm_label=norm_label, cache_suffix=cache_suffix, compare_df=compare_df, padding=args.padding, resid=resid)


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
    parser.add_argument("--global", dest="global_scope", action="store_true",
                        help="Also create the global (N × steering-type) heatmaps; "
                             "by default only the local layer-sweep heatmaps are created")
    parser.add_argument("--max", action="store_true",
                        help="Additionally create layer-sweep figures with cells aggregated via "
                             "max over N, saved to figures/<stream>_max/local/")
    parser.add_argument("--avg", action="store_true",
                        help="Additionally create layer-sweep figures with cells aggregated via "
                             "mean over N, saved to figures/<stream>_avg/local/")
    args = parser.parse_args()

    norm_mode     = "unnormalized" if args.unnormalized else (args.norm_mode or "normalized")
    cache_suffix  = "_nocache" if args.nocache else ""
    cache_mode    = "no_cache" if args.nocache else args.cache_mode

    stream_modes = [True] if args.resid else [False, True]
    for resid in stream_modes:
        generate_for_stream(args, resid, norm_mode, cache_suffix, cache_mode)


if __name__ == "__main__":
    main()
