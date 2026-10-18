"""
Cosine similarity between a positional steering vector's per-position vectors and its two
single-vector counterparts, for every model x dataset combo, drawn as 3x3 grids.

    python mech_analysis/cosine_sim/cosine_sim.py [--model olmo,qwen3,llama] [--dataset all]
                                                  [--best_configs best_configs_harmonic_floor.json]

Each combo has one cached [layers, positions, hidden] steering tensor in results/, and the
three steering types are collapses of it at a single layer (eval/generation.py):

    mean-padding    cache[layer].mean(0)    one vector
    last            cache[layer, -1]        one vector, the final token only
    positional      cache[layer]            num_positions vectors, one per end-aligned slot

The layer is the positional method's selected layer in --best_configs; nothing is averaged
over layers. Padding slots have an exactly-zero diff, so the mean over all positions points
the same way as a mean over real tokens only.

Positions are plotted relative to the final token (0 = last, negative counting back). Prompts
are left-padded, so the leading slots are padding.

Cosine is scale-invariant, so neither the L2 normalization applied at generation time nor
the r_k weighting changes anything here. The one exception is a slot whose diff is exactly
zero -- every add/sub row is padding there, or every add/sub pair is token-identical up to
it. Its direction is unrecoverable, so it is reported as NaN rather than as 0.

Cosine is signed rather than squared: a steering vector's sign is meaningful (+v steers
toward the concept, -v away), so anti-aligned positions should read as negative.

r_k is the share of steering rows (add and sub pooled) with a real token at slot k, recomputed
by re-tokenizing the add/sub prompts named in the combo's config.yml the way DataHandler does.
Positions whose r_k is below --coverage (default 90%) are left out of the figures, since below
that a position's vector is a diff-of-means over a handful of long examples padded out with
padding activations. The npz still holds every position.

Figures go to cosine_sim_figures/ next to this file, raw arrays to cosine_sim_results/:
    vs_mean_last_grid.png        per-position cosine against the mean/last vectors
    pairwise_grid.png            P x P cosine between every pair of positions
    {model}_{dataset}.npz        the raw arrays for one combo
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_BEST_CONFIGS = REPO_ROOT / "best_configs_harmonic_floor.json"
LAYER_METHOD = "positional"

MODELS = {
    "olmo": ("OLMo-2-1124-13B-DPO", "allenai/OLMo-2-1124-13B-DPO", "OLMo-2 13B"),
    "qwen3": ("Qwen3-14B", "Qwen/Qwen3-14B", "Qwen3 14B"),
    "llama": ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1 8B"),
}
DATASETS = {
    "harmful": ("harmful-long", "harmless", "Harmful"),
    "sycophancy": ("non-sycophantic-long", "sycophancy", "Sycophancy"),
    "verse": ("verse-long", "prose", "Verse"),
}

# Each model's add_generation_prompt=True suffix, shared by every steering prompt of every
# dataset (verified against the tokenizers), keyed by position relative to the last token.
# Anything further back is prompt-dependent and intentionally left unlabeled.
CHAT_SUFFIX_TOKENS = {
    "olmo": {0: ">\\n", -1: "|", -2: "assistant", -3: "|", -4: "<"},
    "qwen3": {0: "\\n\\n", -1: "</think>", -2: "\\n\\n", -3: "<think>", -4: "\\n",
              -5: "assistant", -6: "<|im_start|>", -7: "\\n", -8: "<|im_end|>"},
    "llama": {0: "\\n\\n", -1: "<|end_header_id|>", -2: "assistant",
              -3: "<|start_header_id|>", -4: "<|eot_id|>"},
}


def expand(raw, table):
    tags = list(table) if raw == "all" else [t.strip() for t in raw.split(",")]
    for t in tags:
        if t not in table:
            raise SystemExit(f"unknown tag {t!r}; choose from {', '.join(table)}, all")
    return tags


def cache_dir(mtag, dtag):
    source, base, _ = DATASETS[dtag]
    return RESULTS_ROOT / MODELS[mtag][0] / f"from_{source}_to_{base}"


def load_cache(mtag, dtag):
    source = DATASETS[dtag][0]
    path = cache_dir(mtag, dtag) / f"{MODELS[mtag][0]}_steering_cache_{source}_resid_content.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def selected_layer(best_configs, mtag, dtag):
    source, base, _ = DATASETS[dtag]
    cfg = best_configs[MODELS[mtag][0]][f"from_{source}_to_{base}"]["residuals"]["local"]
    return int(cfg[LAYER_METHOD]["layer"])


def real_frac(mtag, dtag, P, tokenizer):
    """r_k over the add and sub rows pooled: DataHandler's templating, then left-padded to P,
    so slot k is real for a row iff its length >= P - k."""
    cfg = dict(line.split(":", 1) for line in
               (cache_dir(mtag, dtag) / "config.yml").read_text().splitlines() if ":" in line)
    kwargs = {"enable_thinking": False} if mtag == "qwen3" else {}
    lens = []
    for side in ("add", "sub"):
        rows = [json.loads(line) for line in
                (REPO_ROOT / cfg[f"steering_{side}_path"].strip()).read_text().splitlines()
                if line.strip()]
        assistant_exists = any(m["role"] == "assistant" for m in rows[0]["prompt"])
        for r in rows:
            n = len(r["prompt"]) - 1 if assistant_exists else max(len(r["prompt"]), 1)
            text = tokenizer.apply_chat_template(r["prompt"][:n], add_generation_prompt=True,
                                                 tokenize=False, **kwargs)
            lens.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    lens = np.array(lens)
    if lens.max() > P:
        raise ValueError(f"{mtag}/{dtag}: a {lens.max()}-token steering prompt exceeds cache "
                         f"P={P}; tokenization doesn't match the cache")
    return (lens[None, :] >= (P - np.arange(P))[:, None]).mean(axis=1)


def unit(x):
    """Unit-normalize along the last dim; rows that are exactly zero stay zero."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def position_cosines(x, real_frac):
    """Per-position cos-vs-mean / cos-vs-last, mean-vs-last, P x P pairwise cosine, and dead
    mask for one [P, h] layer slice."""
    x = x.double()
    dead = (x.norm(dim=1) <= 0).numpy()
    xn = unit(x)
    mean_v = unit(x.mean(0))
    last_v = unit(x[-1])
    cos_mean = (xn @ mean_v).numpy()
    cos_last = (xn @ last_v).numpy()
    cos_mat = (xn @ xn.T).numpy()
    cos_mean[dead] = np.nan
    cos_last[dead] = np.nan
    cos_mat[dead, :] = np.nan
    cos_mat[:, dead] = np.nan
    return {"real_frac": real_frac, "cos_vs_mean": cos_mean, "cos_vs_last": cos_last,
            "cos_matrix": cos_mat, "mean_vs_last": float(mean_v @ last_v), "dead": dead}


def draw_mean_cosine(ax, r, coverage, suffix_tokens):
    """Per-position cosine against the mean/last vectors, trimmed to r_k >= coverage."""
    P = len(r["dead"])
    keep = (~r["dead"]) & (r["real_frac"] >= coverage)
    kidx = np.arange(P)[keep]
    rel = kidx - (P - 1)
    ax.plot(rel, r["cos_vs_mean"][keep], marker="o", ms=4, lw=1.6, color="#1f77b4",
            label="cos(Mean Token, Positional)")
    ax.plot(rel, r["cos_vs_last"][keep], marker="o", ms=4, lw=1.6, color="#d62728",
            label="cos(Last Token, Positional)")
    ax.axhline(r["mean_vs_last"], color="k", ls=":", lw=1.2,
               label="cos(Mean Token, Last Token)")
    ax.axhline(0, color="grey", lw=0.8, zorder=0)
    ax.set(ylim=(min(0, np.nanmin(r["cos_vs_mean"][keep]),
                     np.nanmin(r["cos_vs_last"][keep])) - 0.02, 1.02),
           xlim=(rel[0] - 0.5, rel[-1] + 0.5))
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(max(1, round(len(rel) / 12))))
    # drawn as rotated annotations below the numeric ticks rather than baked into the tick
    # labels themselves, since the token text is wider than the 1-unit tick spacing
    for xi, tok in suffix_tokens.items():
        if rel[0] <= xi <= rel[-1]:
            ax.annotate(tok, xy=(xi, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -18), textcoords="offset points",
                        rotation=45, rotation_mode="anchor", ha="right", va="top",
                        fontsize=8, annotation_clip=False)
    return kidx


def draw_pairwise(ax, r, coverage, vlo, vhi, cmap):
    """P x P cosine between positions, trimmed to r_k >= coverage."""
    P = len(r["dead"])
    keep = (~r["dead"]) & (r["real_frac"] >= coverage)
    kidx = np.arange(P)[keep]
    sub = r["cos_matrix"][np.ix_(kidx, kidx)]
    n = len(kidx)
    im = ax.imshow(sub, vmin=vlo, vmax=vhi, cmap=cmap, origin="upper")
    rel_kidx = kidx - (P - 1)
    step = max(1, round(n / 8))
    ticks = np.arange(n)[::-1][::step][::-1]
    ax.set_xticks(ticks, labels=rel_kidx[ticks])
    ax.set_yticks(ticks, labels=rel_kidx[ticks])
    ax.tick_params(labelsize=14)
    return im


def grid_axes(models, datasets, w, h):
    fig, axes = plt.subplots(len(models), len(datasets), figsize=(w * len(datasets),
                             h * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            if i == 0:
                axes[i][j].set_title(DATASETS[d][2], fontsize=20)
    return fig, axes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="olmo,qwen3,llama")
    p.add_argument("--dataset", default="all")
    p.add_argument("--best_configs", default=str(DEFAULT_BEST_CONFIGS),
                   help=f"best-configs json; the {LAYER_METHOD} method's selected layer is used")
    p.add_argument("--coverage", type=float, default=0.90,
                   help="drop positions whose r_k is below this from the figures (default "
                        "0.90, i.e. 90%% of the steering rows have a real token there); "
                        "the npz still holds every position")
    p.add_argument("--threads", type=int, default=4,
                   help="torch CPU threads; its default (one per core) thrashes on "
                        "shared many-core nodes, and these matrices are small")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    from transformers import AutoTokenizer

    models = expand(args.model, MODELS)
    datasets = expand(args.dataset, DATASETS)
    best_configs = json.loads(Path(args.best_configs).read_text())

    here = Path(__file__).resolve().parent
    out_dir, fig_dir = here / "cosine_sim_results", here / "cosine_sim_figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    results = {}
    for m in models:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[m][1], token=os.environ.get("HF_TOKEN"))
        for d in datasets:
            cache = load_cache(m, d)
            L, P, h = cache.shape
            layer = selected_layer(best_configs, m, d)
            rf = real_frac(m, d, P, tokenizer)
            r = position_cosines(cache[layer], rf)
            r["layer"] = layer
            results[(m, d)] = r

            np.savez(out_dir / f"{m}_{d}.npz", position=np.arange(P), layer=np.array(layer),
                     cos_vs_mean=r["cos_vs_mean"], cos_vs_last=r["cos_vs_last"],
                     cos_matrix=r["cos_matrix"], mean_vs_last=np.array(r["mean_vs_last"]),
                     real_frac=rf)

            keep = (~r["dead"]) & (rf >= args.coverage)
            kidx = np.arange(P)[keep]
            sub = r["cos_matrix"][np.ix_(kidx, kidx)]
            off = sub[~np.eye(len(kidx), dtype=bool)]
            print(f"{m}/{d}: layer {layer}/{L}, P={P}, h={h}, {int(r['dead'].sum())} empty, "
                  f"{len(kidx)} plotted at r_k >= {args.coverage:.0%} "
                  f"(rel {kidx[0] - (P - 1)}..0)")
            print(f"  cos(mean, last) = {r['mean_vs_last']:+.4f}")
            for name in ("mean", "last"):
                c = r[f"cos_vs_{name}"]
                print(f"  positional vs {name}: max {np.nanmax(c[keep]):+.3f}  "
                      f"min {np.nanmin(c[keep]):+.3f}")
            print(f"  pairwise cosine (off-diagonal, plotted set): mean {off.mean():+.3f}  "
                  f"min {off.min():+.3f}  max {off.max():+.3f}")

    cov_label = f"r_k >= {args.coverage:.1f}"

    # --- per-position cosine against the two single-vector methods ---
    fig, axes = grid_axes(models, datasets, 6.0, 5.2)
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            ax = axes[i][j]
            draw_mean_cosine(ax, results[(m, d)], args.coverage, CHAT_SUFFIX_TOKENS[m])
            r = results[(m, d)]
            sub = f"layer {r['layer']}, cos(Mean Token, Last Token) = {r['mean_vs_last']:+.2f}"
            ax.set_title(f"{DATASETS[d][2]}\n{sub}" if i == 0 else sub,
                         fontsize=16 if i == 0 else 11)
            if j == 0:
                ax.set_ylabel(f"{MODELS[m][2]}\nCosine Similarity", fontsize=13)
            if i == len(models) - 1:
                ax.set_xlabel("Position (From Last Token)", fontsize=13, labelpad=75)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(f"Cosine Similarity Between Steering Vectors ({cov_label})", fontsize=18)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3,
               fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=4.5)
    fig.savefig(fig_dir / "vs_mean_last_grid.png", dpi=300)
    plt.close(fig)

    # --- P x P cosine between every pair of positions, one shared color scale ---
    subs = []
    for r in results.values():
        keep = (~r["dead"]) & (r["real_frac"] >= args.coverage)
        subs.append(r["cos_matrix"][np.ix_(keep, keep)].ravel())
    allv = np.concatenate(subs)
    vlo, vhi = min(float(np.nanmin(allv)), 0.0), float(np.nanmax(allv))
    # viridis instead of a diverging map -- these are nearly all positive, so a sequential
    # colormap uses its full range instead of wasting the half meant for negative values
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("lightgrey")
    fig, axes = grid_axes(models, datasets, 5.2, 5.0)
    im = None
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            ax = axes[i][j]
            im = draw_pairwise(ax, results[(m, d)], args.coverage, vlo, vhi, cmap)
            if j == 0:
                ax.set_ylabel(f"{MODELS[m][2]}\nPosition (From Last Token)", fontsize=17)
            if i == len(models) - 1:
                ax.set_xlabel("Position (From Last Token)", fontsize=17)
    fig.suptitle(f"Pairwise Cosine Similarity Between Positional Steering Vectors "
                 f"({cov_label})", fontsize=22)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    cax = fig.add_axes((0.25, 0.045, 0.5, 0.012))
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Cosine Similarity", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    fig.savefig(fig_dir / "pairwise_grid.png", dpi=300)
    plt.close(fig)

    print(f"wrote {fig_dir}/vs_mean_last_grid.png and pairwise_grid.png")
    print(f"wrote {out_dir}/<model>_<dataset>.npz")


if __name__ == "__main__":
    main()
