"""
Scatter plots of judge pass rate (x) vs. MMLU accuracy (y) on the validation
split, one 3x3-subplot figure (model x dataset) per stream per scope, with
steering methods (steering_type) color-coded. By default, generates one PNG
per stream x scope combination (residuals/attention x local/global), named
pareto_<stream>_<scope>.png, all written flat into figures_pareto/.

Usage:
    python analysis/pareto_frontier.py [--unnormalized]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    "harmful-long":         "harmful",
    "non-sycophantic-long": "sycophancy",
    "verse-long":           "verse",
}

# Same needles as create_heatmaps.py / create_heatmap_mmlu.py, kept in sync
# for consistent filenames across the analysis/ scripts.
SHORT_MODEL_NEEDLES = [
    ("OLMo", "olmo"),
    ("Qwen3", "qwen3"),
    ("Qwen", "qwen"),
    ("Gemma4", "gemma-4"),
    ("Gemma", "gemma"),
    ("Llama", "llama"),
]

CANONICAL_BASES = {"harmless", "sycophancy", "prose"}


def short_model_label(model: str) -> str:
    return next(
        (label for label, needle in SHORT_MODEL_NEEDLES if needle in model.lower()),
        model.split("/")[-1],
    )


def short_source_label(source: str) -> str:
    return SOURCE_TO_TAG.get(source, re.sub(r"-(long|single)$", "", source))


def build_merged_df(stream: str, norm_mode: str) -> pd.DataFrame | None:
    """Join validation-split pass rates (results_summary_<stream>.csv) with
    validation-split MMLU accuracy (mmlu_summary.csv) on
    model/task/scope/steering_type/split/N/layer, matching the same join keys
    as create_heatmap_mmlu.py's divide_by_mmlu_accuracy(). Only conditions
    with a matching MMLU run are kept (inner join)."""
    csv_path = ACCURACY_DIR / f"results_summary_{stream}.csv"
    if not csv_path.exists():
        csv_path = ACCURACY_DIR / "results_summary.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}  —  run summarize_results.py first  "
              f"(skipping {stream} results)")
        return None

    df = pd.read_csv(csv_path)
    df = df[~df["model"].str.lower().str.contains("gemma-4", na=False)]
    if "split" in df.columns:
        df = df[df["split"] == "val"]
    else:
        print(f"WARNING: no `split` column in {csv_path.name} — rerun summarize_results.py "
              f"to exclude held-out-test rows from this plot.")

    base_mask = df["base"].apply(lambda b: b.split("_")[-1] in CANONICAL_BASES)
    df = df[base_mask & (df["norm_mode"] == norm_mode)]
    if "stream_mode" in df.columns:
        df = df[df["stream_mode"] == stream]

    if not MMLU_SUMMARY_PATH.exists():
        print(f"MMLU summary not found: {MMLU_SUMMARY_PATH}  —  "
              f"run mmlu/summarize_mmlu.py first. Skipping {stream}.")
        return None

    mmlu = pd.read_csv(MMLU_SUMMARY_PATH)
    mmlu = mmlu[(mmlu["stream"] == stream) & (mmlu["split"] == "val")]
    mmlu["layer_key"] = mmlu["layer"].astype(str)

    df = df.copy()
    df["task"] = "from_" + df["source"] + "_to_" + df["base"]
    df["layer_key"] = df["layer"].apply(lambda v: "all" if pd.isna(v) else str(int(v)))

    merged = df.merge(
        mmlu[["model", "task", "scope", "steering_type", "split", "N", "layer_key",
              "accuracy", "baseline_accuracy"]],
        on=["model", "task", "scope", "steering_type", "split", "N", "layer_key"],
        how="inner",
    )
    return merged


def pareto_frontier_points(x, y):
    """Non-dominated (x, y) pairs for maximizing both — scan in decreasing-x
    order, keeping a point only if it beats every higher/equal-x point's y
    seen so far, then return them sorted ascending by x for a step plot."""
    pts = sorted(zip(x, y), key=lambda p: (-p[0], -p[1]))
    frontier = []
    max_y = -float("inf")
    for px, py in pts:
        if py > max_y:
            frontier.append((px, py))
            max_y = py
    frontier.sort(key=lambda p: p[0])
    return frontier


MODEL_ORDER = [label for label, _ in SHORT_MODEL_NEEDLES]
SOURCE_ORDER = list(SOURCE_TO_TAG.values())


def _ordered(present, preferred_order):
    ordered = [v for v in preferred_order if v in present]
    ordered += sorted(v for v in present if v not in ordered)
    return ordered


def _plot_on_ax(ax, model_label, source_label, group, steering_types, palette):
    right_ends = []  # (x, y, color) of each method's rightmost frontier point
    left_ends = []   # (x, y, color) of each method's leftmost frontier point
    for steering_type in steering_types:
        sub = group[group["steering_type"] == steering_type]
        if sub.empty:
            continue
        color = palette[steering_type]

        frontier = pareto_frontier_points(sub["pass_rate"], sub["accuracy"])
        frontier_set = set(frontier)
        is_frontier = [(px, py) in frontier_set
                       for px, py in zip(sub["pass_rate"], sub["accuracy"])]
        dominated = sub[~pd.Series(is_frontier, index=sub.index)]

        ax.scatter(dominated["pass_rate"], dominated["accuracy"],
                   color=color, marker="o", s=25, alpha=0.2)

        fx, fy = zip(*frontier)
        ax.plot(fx, fy, color=color, linewidth=4, alpha=0.5, zorder=3)
        ax.scatter(fx, fy, color=color, marker="o", s=25, alpha=0.95,
                   zorder=4, label=steering_type.capitalize())
        right_ends.append((fx[-1], fy[-1], color))
        left_ends.append((fx[0], fy[0], color))

    ax.set_xlabel("Steering Success Rate", fontsize=14)
    ax.set_ylabel("MMLU Accuracy", fontsize=14)
    ax.set_title(f"{model_label}  |  {source_label}", fontsize=16)
    ax.set_xlim(-0.02, 1.02)
    ax.relim()
    ax.autoscale_view()
    return right_ends, left_ends


def _draw_edge_lines(ax, right_ends, left_ends):
    # Drop each frontier's rightmost point straight down to the x-axis, and
    # extend each frontier's leftmost point horizontally out to the left
    # edge, so the step plot reads as a complete frontier rather than
    # dangling in empty space past the last (highest pass-rate) condition
    # or starting abruptly at the first plotted point. Called after any
    # cross-subplot axis-limit sharing so the lines reach the final edges.
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    for x, y, color in right_ends:
        ax.plot([x, x], [ymin, y], color=color, linewidth=4, alpha=0.5,
                zorder=3)
    for x, y, color in left_ends:
        ax.plot([xmin, x], [y, y], color=color, linewidth=4, alpha=0.5,
                zorder=3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def make_pareto_plots(df: pd.DataFrame, fig_path: Path, stream: str, scope: str):
    steering_types = sorted(df["steering_type"].unique())
    palette = dict(zip(steering_types, sns.color_palette("tab10", len(steering_types))))

    df = df.copy()
    df["model_label"] = df["model"].apply(short_model_label)
    df["source"] = df["dataset"].apply(lambda d: d.split(" → ")[0])
    df["source_label"] = df["source"].apply(short_source_label)

    models = _ordered(df["model_label"].unique(), MODEL_ORDER)
    sources = _ordered(df["source_label"].unique(), SOURCE_ORDER)

    n_rows, n_cols = len(models), len(sources)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                              squeeze=False)

    handles, labels = [], []
    visible = [[False] * n_cols for _ in range(n_rows)]
    edge_ends = {}  # (row, col) -> (right_ends, left_ends)
    for row, model_label in enumerate(models):
        for col, source_label in enumerate(sources):
            ax = axes[row][col]
            group = df[(df["model_label"] == model_label) &
                       (df["source_label"] == source_label)]
            if group.empty:
                ax.set_visible(False)
                continue
            edge_ends[(row, col)] = _plot_on_ax(
                ax, model_label, source_label, group, steering_types, palette)
            visible[row][col] = True
            if not handles:
                handles, labels = ax.get_legend_handles_labels()

    # Share y-axis limits across each row (same model, so MMLU accuracy is
    # on a comparable scale across its datasets/columns).
    for row in range(n_rows):
        cols_in_row = [col for col in range(n_cols) if visible[row][col]]
        if len(cols_in_row) <= 1:
            continue
        ylims = [axes[row][col].get_ylim() for col in cols_in_row]
        row_ymin = min(lo for lo, hi in ylims)
        row_ymax = max(hi for lo, hi in ylims)
        for col in cols_in_row:
            axes[row][col].set_ylim(row_ymin, row_ymax)

    # Draw the frontier edge-extension lines now that each row's y-limits
    # are final, so they reach all the way to the (possibly shared) bottom
    # and left edges instead of the pre-sharing per-subplot limits.
    for (row, col), (right_ends, left_ends) in edge_ends.items():
        _draw_edge_lines(axes[row][col], right_ends, left_ends)

    # Keep the x-axis label only on each column's bottommost visible subplot,
    # and the y-axis label only on each row's leftmost visible subplot, to
    # avoid repeating them on every subplot.
    for col in range(n_cols):
        rows_in_col = [row for row in range(n_rows) if visible[row][col]]
        for row in rows_in_col[:-1]:
            axes[row][col].set_xlabel("")
    for row in range(n_rows):
        cols_in_row = [col for col in range(n_cols) if visible[row][col]]
        for col in cols_in_row[1:]:
            axes[row][col].set_ylabel("")

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02), fontsize=13)

    stream_label = "Residual Stream" if stream == "residuals" else "Attention Stream"
    fig.suptitle(f"{stream_label}, {scope.capitalize()}", fontsize=18)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def generate_for_stream(stream: str, norm_mode: str, figures_dir: Path):
    merged = build_merged_df(stream, norm_mode)
    if merged is None or merged.empty:
        print(f"No overlapping pass-rate / MMLU-accuracy rows for stream={stream} — skipping.")
        return

    for scope in ("local", "global"):
        scope_df = merged[merged["scope"] == scope]
        if scope_df.empty:
            continue
        fig_path = figures_dir / f"pareto_{stream}_{scope}.png"
        make_pareto_plots(scope_df, fig_path, stream, scope)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unnormalized", action="store_true",
                        help="Plot unnormalized data instead of normalized")
    args = parser.parse_args()

    norm_mode = "unnormalized" if args.unnormalized else "normalized"
    figures_dir = BASE_DIR / "figures_pareto"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stream in ("residuals", "attention"):
        generate_for_stream(stream, norm_mode, figures_dir)


if __name__ == "__main__":
    main()
