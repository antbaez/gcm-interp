"""
Cosine-similarity geometry of the three steering-vector collapses.

All three `steering_type` modes are different collapses of one cached
`[layers, positions, hidden]` tensor, applied at generation time in
eval/generation.py (_get_steering_vector):

    last          -> cache[layer, -1]          [H]
    mean          -> cache[layer].mean(0)      [H]
    positional[p] -> cache[layer, p]           [H], one per position

This script loads those caches from results/ and reports, per model and
dataset, how similar the collapses are in direction:

    cos(last, mean)            one scalar
    cos(last, positional[p])   a curve over positions
    cos(mean, positional[p])   a curve over positions

Each is computed per layer and then averaged across all layers (the standard
deviation across layers is reported alongside, since a mean cosine near zero
can mean either "orthogonal at every layer" or "sign-flipping across layers").

Cosine is scale-invariant, so the L2 normalization applied in
_prepare_steering_vector does not affect any number here.

Prompts are left-padded, so low positions are pad-region and the final
position is the last real token. cos(last, positional[P-1]) == 1 by
construction and is a useful sanity check.

Outputs (default under analysis/cosine/):
    cosine_summary.csv      one row per model x dataset
    cosine_positional.csv   one row per model x dataset x position
    positional_grid.png

Usage (from the repo root):
    python steering_cosines.py
    python steering_cosines.py --model llama,olmo --dataset harmful
    python steering_cosines.py --no-plots
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUT = REPO_ROOT / "analysis" / "cosine"

MODEL_DIRS = {
    "olmo": "OLMo-2-1124-13B-DPO",
    "qwen": "Qwen1.5-14B-Chat",
    "qwen3": "Qwen3-14B",
    "gemma": "gemma-3-12b-it",
    "gemma4": "gemma-4-12B-it",
    "llama": "Llama-3.1-8B-Instruct",
}

DATASET_SPECS = {
    "harmful": ("harmful-long", "harmless"),
    "sycophancy": ("non-sycophantic-long", "sycophancy"),
    "verse": ("verse-long", "prose"),
}


def expand_tags(raw, table, what):
    if raw is None or raw == "all":
        return list(table)
    out = []
    for tag in raw.split(","):
        tag = tag.strip()
        if tag not in table:
            sys.exit(f"Error: unknown {what} '{tag}'. Must be one of: {', '.join(table)}, all")
        if tag not in out:
            out.append(tag)
    return out


def find_cache(model_dir, source, base):
    model_root = RESULTS_ROOT / model_dir
    if not model_root.is_dir():
        return None
    filename = f"{model_dir}_steering_cache_{source}_resid.pt"
    exact = model_root / f"from_{source}_to_{base}" / filename
    if exact.is_file():
        return exact
    matches = sorted(model_root.glob(f"from_{source}_to_{base}*/{filename}"))
    return matches[0] if matches else None


def collapse_cosines(cache):
    cache = cache.to(torch.float32)
    last = cache[:, -1]
    mean = cache.mean(dim=1)

    cos_last_mean = F.cosine_similarity(last, mean, dim=-1)
    cos_last_pos = F.cosine_similarity(last.unsqueeze(1), cache, dim=-1)
    cos_mean_pos = F.cosine_similarity(mean.unsqueeze(1), cache, dim=-1)

    norm_by_pos = cache.norm(dim=-1).mean(dim=0)
    active = norm_by_pos > 0
    first_active = int(active.nonzero()[0].item()) if active.any() else cache.shape[1]

    return {
        "n_layers": cache.shape[0],
        "n_positions": cache.shape[1],
        "hidden": cache.shape[2],
        "n_active": int(active.sum().item()),
        "first_active": first_active,
        "norm_by_pos": norm_by_pos,
        "last_mean": (cos_last_mean.mean().item(), cos_last_mean.std().item()),
        "last_pos": (cos_last_pos.mean(dim=0), cos_last_pos.std(dim=0)),
        "mean_pos": (cos_mean_pos.mean(dim=0), cos_mean_pos.std(dim=0)),
    }


def plot_one(ax, stats, title):
    P = stats["n_positions"]
    start = stats["first_active"]
    positions = range(start, P)
    lp_mu, lp_sd = stats["last_pos"]
    mp_mu, mp_sd = stats["mean_pos"]
    lm_mu, _ = stats["last_mean"]

    if start > 0:
        ax.axvspan(0, start - 0.5, color="0.88", zorder=0)
        ax.text(start / 2, -0.92, f"pad\n(Δ = 0)", ha="center", va="bottom",
                fontsize=6.5, color="0.35")

    for mu, sd, colour, label in (
        (lp_mu, lp_sd, "tab:blue", "cos(last, positional[p])"),
        (mp_mu, mp_sd, "tab:orange", "cos(mean, positional[p])"),
    ):
        mu = mu[start:].numpy()
        sd = sd[start:].numpy()
        ax.plot(positions, mu, color=colour, linewidth=1.6, label=label)
        ax.fill_between(positions, mu - sd, mu + sd, color=colour, alpha=0.18, linewidth=0)

    ax.axhline(lm_mu, color="tab:green", linestyle="--", linewidth=1.2,
               label=f"cos(last, mean) = {lm_mu:.3f}")
    ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=0)
    ax.set_xlim(0, P - 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"{title}   (P={P}, active={stats['n_active']})", fontsize=10)
    ax.set_xlabel("position p (left-padded; p = P-1 is last real token)", fontsize=8)
    ax.set_ylabel("mean cosine across layers", fontsize=8)
    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="olmo,qwen3,llama")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    models = expand_tags(args.model, MODEL_DIRS, "model")
    datasets = expand_tags(args.dataset, DATASET_SPECS, "dataset")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    for mtag in models:
        model_dir = MODEL_DIRS[mtag]
        for dtag in datasets:
            source, base = DATASET_SPECS[dtag]
            path = find_cache(model_dir, source, base)
            if path is None:
                print(f"[skip] no cache for {mtag} / {dtag}")
                continue
            cache = torch.load(path, map_location="cpu")
            stats = collapse_cosines(cache)
            collected.append((mtag, dtag, path, stats))
            print(f"[ok]   {mtag:6s} {dtag:11s} L={stats['n_layers']:3d} "
                  f"P={stats['n_positions']:4d} H={stats['hidden']:5d}  "
                  f"cos(last,mean)={stats['last_mean'][0]:+.3f}  {path.parent.name}")

    if not collected:
        sys.exit("No steering caches found. Nothing to do.")

    summary_path = out_dir / "cosine_summary.csv"
    with open(summary_path, "w") as f:
        f.write("model,dataset,n_layers,n_positions,hidden,n_active,coverage,first_active,"
                "cos_last_mean,cos_last_mean_std,"
                "cos_last_pos_firstactive,cos_mean_pos_firstactive,cache_path\n")
        for mtag, dtag, path, s in collected:
            lm_mu, lm_sd = s["last_mean"]
            lp_mu, _ = s["last_pos"]
            mp_mu, _ = s["mean_pos"]
            fa = s["first_active"]
            f.write(f"{mtag},{dtag},{s['n_layers']},{s['n_positions']},{s['hidden']},"
                    f"{s['n_active']},{s['n_active'] / s['n_positions']:.4f},{fa},"
                    f"{lm_mu:.6f},{lm_sd:.6f},"
                    f"{lp_mu[fa].item():.6f},{mp_mu[fa].item():.6f},"
                    f"{path.relative_to(REPO_ROOT)}\n")

    positional_path = out_dir / "cosine_positional.csv"
    with open(positional_path, "w") as f:
        f.write("model,dataset,position,active,norm,cos_last_pos,cos_last_pos_std,"
                "cos_mean_pos,cos_mean_pos_std\n")
        for mtag, dtag, _, s in collected:
            lp_mu, lp_sd = s["last_pos"]
            mp_mu, mp_sd = s["mean_pos"]
            norms = s["norm_by_pos"]
            for p in range(s["n_positions"]):
                f.write(f"{mtag},{dtag},{p},{int(p >= s['first_active'])},"
                        f"{norms[p].item():.6f},"
                        f"{lp_mu[p].item():.6f},{lp_sd[p].item():.6f},"
                        f"{mp_mu[p].item():.6f},{mp_sd[p].item():.6f}\n")

    print(f"\nwrote {summary_path.relative_to(REPO_ROOT)}")
    print(f"wrote {positional_path.relative_to(REPO_ROOT)}")

    print("\nMODEL  DATASET        L    P  ACTIVE  COVERAGE  cos(last,mean)  cos(last,pos[first])")
    print("-----  -----------  ---  ---  ------  --------  --------------  --------------------")
    for mtag, dtag, _, s in collected:
        lp_mu, _ = s["last_pos"]
        fa = s["first_active"]
        print(f"{mtag:5s}  {dtag:11s}  {s['n_layers']:3d}  {s['n_positions']:3d}  "
              f"{s['n_active']:6d}  {s['n_active'] / s['n_positions']:8.2f}  "
              f"{s['last_mean'][0]:+14.3f}  {lp_mu[fa].item():+20.3f}")

    if args.no_plots:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present_models = [m for m in models if any(c[0] == m for c in collected)]
    present_datasets = [d for d in datasets if any(c[1] == d for c in collected)]
    lookup = {(m, d): s for m, d, _, s in collected}

    fig, axes = plt.subplots(len(present_models), len(present_datasets),
                             figsize=(4.6 * len(present_datasets), 3.2 * len(present_models)),
                             squeeze=False)
    for i, mtag in enumerate(present_models):
        for j, dtag in enumerate(present_datasets):
            ax = axes[i][j]
            s = lookup.get((mtag, dtag))
            if s is None:
                ax.axis("off")
                continue
            plot_one(ax, s, f"{mtag} / {dtag}")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    grid_path = out_dir / "positional_grid.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"wrote {grid_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
