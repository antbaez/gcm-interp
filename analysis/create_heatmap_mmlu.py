"""
Same as create_heatmaps.py, but each cell is pass_rate divided by the MMLU
accuracy of that condition (from mmlu/results/mmlu_summary.csv), instead of
raw pass_rate — i.e. how much steered behavior you get per unit of retained
general capability.

Usage:
    python analysis/create_heatmap_mmlu.py [--csv PATH] [--diff]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "judge-evals"))
from config import BASE_DIR

ACCURACY_DIR = BASE_DIR / "judge-evals" / "accuracy"

MMLU_SUMMARY_PATH = BASE_DIR / "mmlu" / "results" / "mmlu_summary.csv"

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


def divide_by_mmlu_accuracy(df: pd.DataFrame, stream: str) -> pd.DataFrame:
    """Merge in each condition's MMLU accuracy (from mmlu/summarize_mmlu.py's
    mmlu_summary.csv) and divide `pass_rate` by it in place. Joined on
    model/task/scope/steering_type/split/N/layer; conditions with no matching
    MMLU run get `pass_rate` set to NaN (blank cell)."""
    if not MMLU_SUMMARY_PATH.exists():
        print(f"MMLU summary not found: {MMLU_SUMMARY_PATH}  —  "
              f"run mmlu/summarize_mmlu.py first. All cells will be blank.")
        df = df.copy()
        df["pass_rate"] = float("nan")
        return df

    mmlu = pd.read_csv(MMLU_SUMMARY_PATH)
    mmlu = mmlu[mmlu["stream"] == stream]
    mmlu["layer_key"] = mmlu["layer"].astype(str)

    df = df.copy()
    df["task"] = "from_" + df["source"] + "_to_" + df["base"]
    df["layer_key"] = df["layer"].apply(lambda v: "all" if pd.isna(v) else str(int(v)))

    df = df.merge(
        mmlu[["model", "task", "scope", "steering_type", "split", "N", "layer_key", "accuracy"]],
        on=["model", "task", "scope", "steering_type", "split", "N", "layer_key"],
        how="left",
    )
    df["pass_rate"] = df["pass_rate"] / df["accuracy"]
    return df.drop(columns=["task", "layer_key", "accuracy"])


def make_heatmaps(df: pd.DataFrame, figures_dir: Path, diff: bool = False, norm_label: str = "normalized", resid: bool = False, wo_rf: bool = False):
    """
    One figure per (model, dataset). Used for global runs (all layers steered).
    Grid rows = w_rf (default), plus a wo_rf row when wo_rf=True. Each heatmap:
    rows = steering_type, columns = N.
    """
    import matplotlib as mpl

    if wo_rf and "pass_rate_wo_rf" not in df.columns:
        print("--wo_rf requested but `pass_rate_wo_rf` column not found in the summary CSV "
              "— rerun summarize_results.py to add it. Skipping the wo_rf row.")
        wo_rf = False

    steering_types = sorted(df["steering_type"].unique())
    cmap = "RdYlGn" if diff else "YlGn"
    # pass_rate is now pass_rate / mmlu_accuracy, not bounded to [0, 1].
    pass_rate_max = df["pass_rate"].max()
    vmin, vmax = (-1, 1) if diff else (0, pass_rate_max if pd.notna(pass_rate_max) else 1)

    def _process_group(model, dataset, group):
        source, base = dataset.split(" → ")
        short_model  = next(
            (label for label, needle in SHORT_MODEL_NEEDLES if needle in model.lower()),
            model.split("/")[-1],
        )
        short_source = SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))
        short_title  = f"{short_model}  |  {short_source}  (w_rf)"

        n_vals = sorted(group["N"].unique())
        n_cols = 1

        w_rf_max = group.groupby("steering_type")["pass_rate"].max()
        best_steers = (set(w_rf_max[w_rf_max == w_rf_max.max()].index)
                       if not w_rf_max.empty else set())

        all_rows = [("pass_rate", group, "w_rf")]
        if wo_rf:
            all_rows.append(("pass_rate_wo_rf", group, "wo_rf"))
        n_rows = len(all_rows)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.2 + 0.7 * len(n_vals),
                     1.0 + 0.5 * len(steering_types) * n_rows + 1),
            squeeze=False,
            gridspec_kw={"hspace": 0.35},
        )
        fig.suptitle(short_title, fontsize=18)

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
        fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal",
                     location="bottom", fraction=0.025, pad=0.165)

        suffix   = "_diff" if diff else ""
        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}{suffix}_global.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        _process_group(model, dataset, group)


def make_layer_heatmaps(df: pd.DataFrame, figures_dir: Path, norm_label: str = "normalized",
                        resid: bool = False, no_rel: bool = False, wo_rf: bool = False):
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
    # pass_rate is now pass_rate / mmlu_accuracy, not bounded to [0, 1].
    pass_rate_max = df["pass_rate"].max()
    vmin, vmax = 0, pass_rate_max if pd.notna(pass_rate_max) else 1
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
                     (1.0 + 0.35 * len(layer_vals)) * n_rows + 1),
            squeeze=False,
            gridspec_kw={"wspace": 0.08, "hspace": 0.5},
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
                    per_layer_max = sub.groupby("layer")[val_col].max().dropna()
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
        # combined axes; a dynamic-scale row (wo_rf − w_rf) gets its own,
        # since its range differs from the shared 0..1 scale.
        static_row_idxs  = [i for i, spec in enumerate(row_specs) if not spec[2]]
        dynamic_row_idxs = [i for i, spec in enumerate(row_specs) if spec[2]]

        def _row_axes(idxs):
            return [axes[i][j] for i in idxs for j in range(n_types)]

        if static_row_idxs:
            sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            fig.colorbar(sm, ax=_row_axes(static_row_idxs), orientation="horizontal",
                         location="bottom", fraction=0.02, pad=0.235)

        for row_i in dynamic_row_idxs:
            row_vmin, row_vmax = row_scales[row_i]
            sm  = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=row_vmin, vmax=row_vmax))
            fig.colorbar(sm, ax=_row_axes([row_i]), orientation="horizontal",
                         location="bottom", fraction=0.02, pad=0.235)

        fig_path = figures_dir / f"heatmap_{short_model}_{short_source}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path.name}")

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        _process_group(model, dataset, group)


def make_paper_heatmaps(df: pd.DataFrame, figures_dir: Path, norm_label: str = "normalized",
                        resid: bool = False):
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
    # pass_rate is now pass_rate / mmlu_accuracy, not bounded to [0, 1].
    pass_rate_max = df["pass_rate"].max()
    vmin, vmax = 0, pass_rate_max if pd.notna(pass_rate_max) else 1
    steering_types = sorted(df["steering_type"].unique())
    n_types = len(steering_types)

    CELL_W, CELL_H       = 0.55, 0.30
    COL_GAP, ROW_GAP      = 0.20, 0.55
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
        fig_height = TOP_MARGIN + sum(row_heights) + ROW_GAP * (n_rows - 1) + BOTTOM_MARGIN + 1

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
                # sharing a (layer, N) get averaged by the pivot_table the same way
                # for both the annotation and the bold pick. Ties broken by lower N,
                # then lower layer.
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

    for dataset, group in df.groupby("dataset"):
        _process_dataset(dataset, group)


def generate_for_stream(args, resid: bool, norm_mode: str):
    figures_root = BASE_DIR / "figures_mmlu"
    stream = "residuals" if resid else "attention"
    full_dir = figures_root / stream
    # summarize_results.py writes one summary per stream; an explicit --csv wins,
    # and the unsuffixed combined summary is the fallback for older trees.
    csv_path = Path(args.csv) if args.csv else ACCURACY_DIR / f"results_summary_{stream}.csv"
    if not csv_path.exists() and not args.csv:
        csv_path = ACCURACY_DIR / "results_summary.csv"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}  —  run summarize_results.py first  "
              f"(skipping {'residual' if resid else 'attention'} results)")
        return

    df = pd.read_csv(csv_path)
    df = df[~df["model"].str.lower().str.contains("gemma-4", na=False)]
    if "split" in df.columns:
        df = df[df["split"] == "val"]
    else:
        print(f"WARNING: no `split` column in {csv_path.name} — rerun summarize_results.py "
              f"to exclude held-out-test rows from these heatmaps.")
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    df = divide_by_mmlu_accuracy(df, stream)

    full_dir.mkdir(parents=True, exist_ok=True)

    CANONICAL_BASES = {"harmless", "sycophancy", "prose"}
    base_mask = df["base"].apply(lambda b: b.split("_")[-1] in CANONICAL_BASES)
    base_df = df[
        base_mask &
        (df["norm_mode"] == norm_mode)
    ]
    # The combined summary holds both streams, so restrict to this one; a
    # per-stream summary is already filtered and the mask is a no-op there.
    if "stream_mode" in base_df.columns:
        base_df = base_df[base_df["stream_mode"] == stream]
    base_df = filter_hidden_n(base_df)

    # Runs are split by scope: `local` = single-layer sweep over the middle third
    # (layer-axis heatmaps), `global` = all layers at once (N × steering-type
    # heatmaps). Summaries predating the `scope` column fall back to whether a
    # layer index is present. Figures are separated on disk into
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
            make_paper_heatmaps(layer_source_df, paper_dir, norm_label=norm_mode, resid=resid)
            return

        local_dir.mkdir(parents=True, exist_ok=True)
        make_layer_heatmaps(layer_source_df, local_dir, norm_label=norm_mode,
                            resid=resid, no_rel=args.no_rel, wo_rf=args.wo_rf)
    elif args.paper:
        return

    if not args.global_scope:
        return

    if not has_scope:
        return
    base_df = base_df[base_df["scope"] == "global"]

    if base_df.empty:
        return

    norm_label = norm_mode
    global_dir.mkdir(parents=True, exist_ok=True)
    make_heatmaps(base_df, global_dir, diff=args.diff, norm_label=norm_label, resid=resid, wo_rf=args.wo_rf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to a results_summary CSV from summarize_results.py "
                             "(default: the per-stream summary for each stream plotted)")
    parser.add_argument("--diff", action="store_true",
                        help="Plot other−last difference instead of raw pass rates")
    parser.add_argument("--unnormalized", action="store_true",
                        help="Plot unnormalized data; outputs to figures_unnormalized / figures_full_unnormalized")
    parser.add_argument("--attention", action="store_true",
                        help="Plot attention-stream results instead of residual-stream results "
                             "(the two are mutually exclusive; residual-stream is the default)")
    parser.add_argument("--global", dest="global_scope", action="store_true",
                        help="Also create the global (N × steering-type) heatmaps; "
                             "by default only the local layer-sweep heatmaps are created")
    parser.add_argument("--no-rel", dest="no_rel", action="store_true",
                        help="Add a second row of layer-sweep heatmaps using the fluency-only "
                             "pass rate (drops the relevance check); requires results_summary.csv "
                             "to have been generated with the `pass_rate_no_rel` column")
    parser.add_argument("--wo_rf", dest="wo_rf", action="store_true",
                        help="Add a second row (to both layer-sweep and global heatmaps) using "
                             "the wo_rf pass rate (drops both the relevance and fluency checks)")
    parser.add_argument("--paper", action="store_true",
                        help="Only (re)generate the paper heatmaps (figures_mmlu/<stream>/paper/), "
                             "skipping local/global figures")
    args = parser.parse_args()

    norm_mode = "unnormalized" if args.unnormalized else "normalized"

    generate_for_stream(args, not args.attention, norm_mode)


if __name__ == "__main__":
    main()
