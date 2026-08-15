"""
Generate heatmap figures from a results_summary.csv produced by summarize_results.py.

Usage:
    python create_heatmaps.py [--csv PATH] [--diff]
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
})

from config import BASE_DIR

ACCURACY_DIR          = BASE_DIR / "judge-evals" / "accuracy"
ACCURACY_RESIDUAL_DIR = BASE_DIR / "judge-evals" / "accuracy"

SOURCE_TO_TAG = {
    "harmful-long":                    "harmful",
    "non-sycophantic-long":            "sycophancy",
    "verse-long":                      "verse",
}

LAST_TOKEN_NAMES = {"last", "last-token", "last_token"}

# Display label for filenames/titles, matched against the lowercased "model"
# column via substring; order matters — "gemma-4" must be checked before the
# generic "gemma" so gemma-4-12B-it isn't mislabeled as plain Gemma.
SHORT_MODEL_NEEDLES = [
    ("OLMo", "olmo"),
    ("Qwen3", "qwen3"),
    ("Qwen", "qwen"),
    ("Gemma4", "gemma-4"),
    ("Gemma", "gemma"),
    ("Llama", "llama"),
]

# Row order for the "paper" figures (make_paper_heatmaps): models not in this
# list (e.g. gemma-4) are excluded entirely.
PAPER_MODEL_ORDER = ["OLMo", "Qwen3", "Llama"]

# Display label for the paper figures' row labels (distinct from SHORT_MODEL_NEEDLES,
# which is also used for filenames elsewhere and should stay unchanged).
PAPER_ROW_LABEL = {"OLMo": "Olmo", "Qwen3": "Qwen"}

# Layers dropped from the paper figures' y-axis for these models.
PAPER_HIDDEN_LAYERS = {"OLMo": {26}, "Qwen3": {26}}

# N values to drop from heatmaps, keyed by a lowercase substring matched against
# the "model" column; "default" applies to any model not otherwise listed.
HIDDEN_N_BY_MODEL = {
    "default": {2, 500},
    "gemma": set(),
    "qwen3": {2, 500},
    "llama": {2, 250, 500},
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


def make_heatmaps(df: pd.DataFrame, figures_dir: Path, diff: bool = False, norm_label: str = "normalized", compare_df: pd.DataFrame = None, padding: bool = False, resid: bool = False, multi: bool = False):
    """
    One figure per (model, dataset). Used for global runs (all layers steered).
    Grid rows = w_rf + wo_rf (or a padding comparison when compare_df provided), one heatmap per row.
    Each heatmap: rows = steering_type, columns = N.
    """
    import matplotlib as mpl

    steering_types = sorted(df["steering_type"].unique())
    cmap = "RdYlGn" if diff else "YlGn"
    vmin, vmax = (-1, 1) if diff else (0, 1)

    compare_groups = {}
    if compare_df is not None:
        for (model, dataset), grp in compare_df.groupby(["model", "dataset"]):
            compare_groups[(model, dataset)] = grp

    def _process_group(model, dataset, group):
        source, base = dataset.split(" → ")
        short_model  = next(
            (label for label, needle in SHORT_MODEL_NEEDLES if needle in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))
        short_title  = f"{short_model}  |  {short_source}  (w_rf)"

        n_vals = sorted(group["N"].unique())
        n_rows = 2  # main row + wo_rf (or padding-comparison) row
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
            all_rows = [
                ("pass_rate", group, "normal"),
                ("pass_rate", compare_group, "padding"),
            ]
        else:
            all_rows = [
                ("pass_rate", group, "w_rf"),
                ("pass_rate_wo_rf", group, "wo_rf"),
            ]

        for row_i, (val_col, src, row_label) in enumerate(all_rows):
            is_last_row = row_i == n_rows - 1
            ax  = axes[row_i][0]
            sub = src

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
        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}{suffix}_global.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")

    groups = list(df.groupby(["model", "dataset"]))
    if multi:
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda item: _process_group(item[0][0], item[0][1], item[1]), groups))
    else:
        for (model, dataset), group in groups:
            _process_group(model, dataset, group)


def make_layer_heatmaps(df: pd.DataFrame, figures_dir: Path, norm_label: str = "normalized",
                        resid: bool = False, multi: bool = False,
                        no_rel: bool = False, wo_rf: bool = False):
    """
    One figure per (model, dataset) for single-layer-sweep runs.
    One subplot per steering_type, laid out in a row; each heatmap has rows = layer,
    columns = N, cells = w_rf pass rate. Every swept layer is shown; the best layer per
    steering type (max pass rate over N) is bolded in that subplot's labels.

    When no_rel is set, a second row of heatmaps is added below using the
    fluency-only pass rate (judge pass + fluency check, no relevance check).

    When wo_rf is set, a second row of heatmaps is added below showing
    wo_rf − w_rf (how much the pass rate gains from dropping the fluency/relevance
    check) — always >= 0 since w_rf's pass condition is a strict subset of wo_rf's.
    This row uses its own (dynamic, per-figure) color scale since the gains are
    typically much smaller than the raw pass rates in the other rows.
    """
    import matplotlib as mpl

    cmap = "YlGn"
    vmin, vmax = 0, 1
    steering_types = sorted(df["steering_type"].unique())

    if no_rel and "pass_rate_no_rel" not in df.columns:
        print("--no-rel requested but `pass_rate_no_rel` column not found in the summary CSV "
              "— rerun summarize_results.py to add it. Skipping the fluency-only row.")
        no_rel = False

    if wo_rf and "pass_rate_wo_rf" not in df.columns:
        print("--wo_rf requested but `pass_rate_wo_rf` column not found in the summary CSV "
              "— rerun summarize_results.py to add it. Skipping the wo_rf row.")
        wo_rf = False

    WO_RF_DIFF_COL = "pass_rate_wo_rf_minus_w_rf"
    if wo_rf:
        df = df.copy()
        df[WO_RF_DIFF_COL] = df["pass_rate_wo_rf"] - df["pass_rate"]

    row_specs = [("pass_rate", "w_rf", False)]
    if no_rel:
        row_specs.append(("pass_rate_no_rel", "fluency only", False))
    if wo_rf:
        row_specs.append((WO_RF_DIFF_COL, "wo_rf − w_rf", True))
    n_rows = len(row_specs)

    def _process_group(model, dataset, group):
        source, base = dataset.split(" → ")
        short_model  = next(
            (label for label, needle in SHORT_MODEL_NEEDLES if needle in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))

        n_vals     = sorted(group["N"].unique())
        layer_vals = sorted(int(l) for l in group["layer"].dropna().unique())
        n_types    = len(steering_types)

        fig, axes = plt.subplots(
            n_rows, n_types,
            figsize=(1.6 + 0.7 * len(n_vals) * n_types,
                     (1.0 + 0.35 * len(layer_vals)) * n_rows),
            squeeze=False,
            gridspec_kw={"wspace": 0.2, "hspace": 0.5},
        )
        title_tag = " / ".join(label for _, label, _ in row_specs)
        fig.suptitle(f"{short_model}  |  {short_source}  ({title_tag})",
                     fontsize=17, y=1.02)

        row_scales = {}
        for row_i, (val_col, row_label, dynamic_scale) in enumerate(row_specs):
            is_last_row = row_i == n_rows - 1

            if dynamic_scale:
                row_vals = group[val_col].dropna()
                row_vmin, row_vmax = 0, (row_vals.max() if not row_vals.empty else 0) or 1e-6
            else:
                row_vmin, row_vmax = vmin, vmax
            row_scales[row_i] = (row_vmin, row_vmax)

            st_max = group.groupby("steering_type")[val_col].max()
            best_steers = (set(st_max[st_max == st_max.max()].index)
                           if not st_max.empty else set())

            for col_i, st in enumerate(steering_types):
                ax  = axes[row_i][col_i]
                sub = group[group["steering_type"] == st]
                matrix = (
                    sub.pivot_table(index="layer", columns="N", values=val_col)
                       .reindex(index=layer_vals, columns=n_vals)
                )

                sns.heatmap(matrix, ax=ax, vmin=row_vmin, vmax=row_vmax,
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
                    per_layer_max = sub.groupby("layer")[val_col].max()
                    if not per_layer_max.empty:
                        best_layer = int(per_layer_max.idxmax())
                        best_row = sub[sub["layer"] == best_layer]
                        best_n = best_row.loc[best_row[val_col].idxmax(), "N"]

                if row_i == 0:
                    ax.set_title(st.capitalize(), fontsize=15,
                                 fontweight="bold" if st in best_steers else "normal",
                                 loc="left", pad=6)
                elif st in best_steers:
                    ax.set_title(" ", fontsize=15, loc="left", pad=6)

                # Layer ticks on every subplot; the row label + "layer" axis label
                # only on the leftmost column.
                ylabel = "Layer (L)" if n_rows == 1 else f"{row_label}\nLayer (L)"
                ax.set_ylabel(ylabel if col_i == 0 else "", fontsize=14)
                ax.set_yticklabels([str(l) for l in layer_vals], rotation=0, fontsize=10)
                for tick in ax.get_yticklabels():
                    if best_layer is not None and tick.get_text() == str(best_layer):
                        tick.set_fontweight("bold")

                if is_last_row:
                    ax.set_xticklabels([f"{n:g}" for n in n_vals], rotation=0, fontsize=11)
                    ax.set_xlabel("Steering Coefficient (N)", fontsize=15, labelpad=12)
                else:
                    ax.set_xticklabels([""] * len(n_vals))
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", bottom=False)
                for tick in ax.get_xticklabels():
                    if best_n is not None and tick.get_text() == f"{best_n:g}":
                        tick.set_fontweight("bold")

        # Static-scale rows (w_rf, fluency-only) share one colorbar spanning their
        # combined vertical extent; a dynamic-scale row (wo_rf − w_rf) gets its own,
        # since its range differs from the shared 0..1 scale.
        static_row_idxs  = [i for i, spec in enumerate(row_specs) if not spec[2]]
        dynamic_row_idxs = [i for i, spec in enumerate(row_specs) if spec[2]]

        def _row_y_span(idxs):
            positions = [axes[i][0].get_position() for i in idxs]
            return min(p.y0 for p in positions), max(p.y1 for p in positions)

        if static_row_idxs:
            y0, y1 = _row_y_span(static_row_idxs)
            sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            cax = fig.add_axes([0.92, y0, 0.005, y1 - y0])
            fig.colorbar(sm, cax=cax)

        for row_i in dynamic_row_idxs:
            y0, y1 = _row_y_span([row_i])
            row_vmin, row_vmax = row_scales[row_i]
            sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=row_vmin, vmax=row_vmax))
            cax = fig.add_axes([0.92, y0, 0.005, y1 - y0])
            fig.colorbar(sm, cax=cax)

        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")

    groups = list(df.groupby(["model", "dataset"]))
    if multi:
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda item: _process_group(item[0][0], item[0][1], item[1]), groups))
    else:
        for (model, dataset), group in groups:
            _process_group(model, dataset, group)


def make_paper_heatmaps(df: pd.DataFrame, figures_dir: Path, norm_label: str = "normalized",
                        resid: bool = False, multi: bool = False):
    """
    One figure per dataset, for the paper: rows stacked by model in PAPER_MODEL_ORDER
    (OLMo, Qwen3, Llama), columns = steering_type, cells = layer x N w_rf pass-rate
    heatmap (same data as make_layer_heatmaps' w_rf row). X-axis labels only appear
    on the bottom row.

    Each row can sweep a different N range per model, so subplots are placed by hand
    (fig.add_axes, in inches) rather than via plt.subplots — that keeps every cell the
    same physical size (CELL_W x CELL_H) instead of stretching narrower rows to fill a
    shared grid column/row width.
    """

    cmap = "YlGn"
    vmin, vmax = 0, 1
    steering_types = sorted(df["steering_type"].unique())
    n_types = len(steering_types)

    CELL_W, CELL_H       = 0.55, 0.30
    COL_GAP, ROW_GAP      = 0.45, 0.55
    LEFT_MARGIN           = 1.3
    RIGHT_MARGIN          = 0.3
    TOP_MARGIN            = 0.95
    BOTTOM_MARGIN         = 0.5

    def short_model_label(model):
        return next(
            (label for label, needle in SHORT_MODEL_NEEDLES if needle in model.lower()),
            model.split("/")[-1],
        )

    df = df.copy()
    df["short_model"] = df["model"].apply(short_model_label)
    df = df[df["short_model"].isin(PAPER_MODEL_ORDER)]

    def _process_dataset(dataset, group):
        source, base = dataset.split(" → ")
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))

        rows = [(label, group[group["short_model"] == label]) for label in PAPER_MODEL_ORDER]
        rows = [
            (label, g[~g["layer"].isin(PAPER_HIDDEN_LAYERS[label])] if label in PAPER_HIDDEN_LAYERS else g)
            for label, g in rows
        ]
        rows = [(label, g) for label, g in rows if not g.empty]
        n_rows = len(rows)
        if n_rows == 0:
            return

        row_layer_vals = [sorted(int(l) for l in g["layer"].dropna().unique()) for _, g in rows]
        row_n_vals     = [sorted(g["N"].unique()) for _, g in rows]
        row_heights    = [len(lv) * CELL_H for lv in row_layer_vals]
        row_widths     = [len(nv) * CELL_W for nv in row_n_vals]

        fig_width  = LEFT_MARGIN + n_types * max(row_widths) + (n_types - 1) * COL_GAP + RIGHT_MARGIN
        fig_height = TOP_MARGIN + sum(row_heights) + ROW_GAP * (n_rows - 1) + BOTTOM_MARGIN

        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(short_source.capitalize(), fontsize=22,
                     y=(fig_height - 0.2) / fig_height)

        y_cursor = fig_height - TOP_MARGIN
        for row_i, (model_label, mgroup) in enumerate(rows):
            is_last_row = row_i == n_rows - 1
            layer_vals = row_layer_vals[row_i]
            n_vals     = row_n_vals[row_i]
            row_h      = row_heights[row_i]
            row_w      = row_widths[row_i]
            y_bottom   = y_cursor - row_h

            for col_i, st in enumerate(steering_types):
                x_left = LEFT_MARGIN + col_i * (row_w + COL_GAP)
                ax = fig.add_axes([x_left / fig_width, y_bottom / fig_height,
                                    row_w / fig_width, row_h / fig_height])
                sub = mgroup[mgroup["steering_type"] == st]
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
                                ha="center", va="center", color="black", fontsize=12)

                # Derived from `matrix` (the same mean-aggregated values that are
                # displayed) rather than raw `sub`, so the bolded cell always matches
                # the highest number actually shown — duplicate condition rows
                # (val-sweep + held-out runs sharing a (layer, N)) get averaged by
                # the pivot_table the same way for both the annotation and the bold
                # pick. Ties broken by lower N, then lower layer.
                best_layer = None
                best_n = None
                flat = matrix.stack()
                if not flat.empty:
                    max_val = flat.max()
                    tied = [idx for idx, v in flat.items() if v == max_val]
                    best_layer, best_n = min(tied, key=lambda t: (t[1], t[0]))
                    best_layer = int(best_layer)

                if row_i == 0:
                    ax.set_title(st.capitalize(), fontsize=17, loc="center", pad=6)

                row_display_label = PAPER_ROW_LABEL.get(model_label, model_label)
                ax.set_ylabel(f"{row_display_label}\nLayer (L)" if col_i == 0 else "", fontsize=13)
                ax.set_yticklabels([str(l) for l in layer_vals], rotation=0, fontsize=11)
                for tick in ax.get_yticklabels():
                    if best_layer is not None and tick.get_text() == str(best_layer):
                        tick.set_fontweight("bold")

                ax.set_xticklabels([f"{n:g}" for n in n_vals], rotation=0, fontsize=13)
                ax.set_xlabel("Steering Coefficient (N)" if is_last_row else "",
                              fontsize=14, labelpad=10)
                for tick in ax.get_xticklabels():
                    if best_n is not None and tick.get_text() == f"{best_n:g}":
                        tick.set_fontweight("bold")

            y_cursor = y_bottom - ROW_GAP

        fig_path = figures_dir / f"heatmap_{short_source}_paper.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")

    groups = list(df.groupby("dataset"))
    if multi:
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda item: _process_dataset(item[0], item[1]), groups))
    else:
        for dataset, group in groups:
            _process_dataset(dataset, group)


def make_layer_heatmaps_agg(df: pd.DataFrame, figures_dir: Path, agg: str, norm_label: str = "normalized",
                            resid: bool = False, multi: bool = False):
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

    def _process_group(model, dataset, group):
        source, base = dataset.split(" → ")
        short_model  = next(
            (label for label, needle in SHORT_MODEL_NEEDLES if needle in model.lower()),
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

        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}_{agg}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")

    groups = list(df.groupby(["model", "dataset"]))
    if multi:
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda item: _process_group(item[0][0], item[0][1], item[1]), groups))
    else:
        for (model, dataset), group in groups:
            _process_group(model, dataset, group)


def generate_for_stream(args, resid: bool, norm_mode: str):
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
    df = df[~df["model"].str.lower().str.contains("gemma-4", na=False)]
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    full_dir.mkdir(parents=True, exist_ok=True)

    CANONICAL_BASES = {"harmless", "sycophancy", "prose"}
    base_mask = df["base"].apply(lambda b: b.split("_")[-1] in CANONICAL_BASES)
    base_df = df[
        base_mask &
        (df["norm_mode"] == norm_mode)
    ]
    base_df = filter_hidden_n(base_df)

    compare_df = None
    if resid:
        pass  # no compare_df for resid mode
    elif args.padding:
        padding_csv = ACCURACY_PADDING_DIR / "results_summary.csv"
        if not padding_csv.exists():
            print(f"Padding CSV not found: {padding_csv}  —  run summarize_results.py with accuracy_padding/ first")
            return
        padding_raw = pd.read_csv(padding_csv)
        padding_raw = padding_raw[~padding_raw["model"].str.lower().str.contains("gemma-4", na=False)]
        pad_mask = padding_raw["base"].apply(lambda b: b.split("_")[-1] in CANONICAL_BASES)
        compare_df = padding_raw[
            pad_mask &
            (padding_raw["norm_mode"] == norm_mode)
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
        if args.paper:
            paper_dir = full_dir / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            make_paper_heatmaps(layer_source_df, paper_dir, norm_label=norm_mode,
                                resid=resid, multi=args.multi)
            return

        agg_requested = args.max or args.avg
        if not agg_requested:
            local_dir.mkdir(parents=True, exist_ok=True)
            make_layer_heatmaps(layer_source_df, local_dir, norm_label=norm_mode,
                                resid=resid, multi=args.multi,
                                no_rel=args.no_rel, wo_rf=args.wo_rf)

            paper_dir = full_dir / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            make_paper_heatmaps(layer_source_df, paper_dir, norm_label=norm_mode,
                                resid=resid, multi=args.multi)

        base_stream_name = "residuals" if resid else ("attention-padding" if args.padding else "attention")
        for agg, flag in (("max", args.max), ("avg", args.avg)):
            if not flag:
                continue
            agg_dir = figures_root / f"{base_stream_name}_{agg}" / "local"
            agg_dir.mkdir(parents=True, exist_ok=True)
            make_layer_heatmaps_agg(layer_source_df, agg_dir, agg=agg, norm_label=norm_mode,
                                     resid=resid, multi=args.multi)
    elif args.paper:
        return

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
    make_heatmaps(base_df, global_dir, diff=args.diff, norm_label=norm_label, compare_df=compare_df, padding=args.padding, resid=resid, multi=args.multi)


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
    parser.add_argument("--padding", action="store_true",
                        help="Compare accuracy/ (top row) vs accuracy_padding/ (bottom row)")
    parser.add_argument("--attention", action="store_true",
                        help="Plot only attention-head results "
                             "(by default only residual-stream results are plotted)")
    parser.add_argument("--global", dest="global_scope", action="store_true",
                        help="Also create the global (N × steering-type) heatmaps; "
                             "by default only the local layer-sweep heatmaps are created")
    parser.add_argument("--max", action="store_true",
                        help="Additionally create layer-sweep figures with cells aggregated via "
                             "max over N, saved to figures/<stream>_max/local/")
    parser.add_argument("--avg", action="store_true",
                        help="Additionally create layer-sweep figures with cells aggregated via "
                             "mean over N, saved to figures/<stream>_avg/local/")
    parser.add_argument("--multi", action="store_true",
                        help="Generate figures in parallel using a thread pool "
                             "(default: single-threaded)")
    parser.add_argument("--no-rel", dest="no_rel", action="store_true",
                        help="Add a second row of layer-sweep heatmaps using the fluency-only "
                             "pass rate (drops the relevance check); requires results_summary.csv "
                             "to have been generated with the `pass_rate_no_rel` column")
    parser.add_argument("--wo_rf", dest="wo_rf", action="store_true",
                        help="Add a second row of layer-sweep heatmaps using the wo_rf "
                             "pass rate (drops both the relevance and fluency checks)")
    parser.add_argument("--paper", action="store_true",
                        help="Only (re)generate the paper heatmaps (figures/<stream>/paper/), "
                             "skipping local/global/agg figures")
    args = parser.parse_args()

    norm_mode = "unnormalized" if args.unnormalized else (args.norm_mode or "normalized")

    stream_modes = [False] if args.attention else [True]
    for resid in stream_modes:
        generate_for_stream(args, resid, norm_mode)


if __name__ == "__main__":
    main()
