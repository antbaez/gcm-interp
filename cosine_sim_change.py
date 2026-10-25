"""
Cosine-similarity *change* along the positional steering vector's position axis.

cosine_sim.py compares the three steering_type collapses (last/mean/positional)
against each other. This script instead asks how much the `positional` cache's
own direction rotates from one token position to the next:

    cos_adj[layer, p] = cos(cache[layer, p], cache[layer, p+1])

averaged across layers (std across layers reported alongside, for the same
sign-flip-vs-orthogonal reason as cosine_sim.py). cos_adj close to 1 means the
steering vector barely rotates between adjacent positions there — pooling that
region would lose little direction, only magnitude information. cos_adj
dropping toward 0 (or negative) means the direction is genuinely
position-specific — pooling would average out competing directions rather
than just denoising.

Reuses cosine_sim.py's MODEL_DIRS/DATASET_SPECS/find_cache so caches are
looked up identically to that script.

Outputs (default under analysis/cosine/):
    cosine_change_summary.csv     one row per model x dataset
    cosine_change_positional.csv  one row per model x dataset x adjacent pair
    adjacent_grid.png

Usage (from the repo root):
    python cosine_sim_change.py
    python cosine_sim_change.py --model llama,olmo --dataset harmful
    python cosine_sim_change.py --no-plots
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from cosine_sim import REPO_ROOT, DEFAULT_OUT, MODEL_DIRS, DATASET_SPECS, expand_tags, find_cache


def adjacent_cosines(cache):
    cache = cache.to(torch.float32)
    a, b = cache[:, :-1], cache[:, 1:]
    cos = F.cosine_similarity(a, b, dim=-1)  # [L, P-1], cos_adj[:, p] = cos(pos[p], pos[p+1])

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
        "cos_adj": (cos.mean(dim=0), cos.std(dim=0)),
    }


def plot_one(ax, stats, title):
    P = stats["n_positions"]
    start = stats["first_active"]
    # cos_adj[p] pairs position p with p+1; a pair is meaningful once at
    # least one side is a real (non-pad) token, i.e. from start-1 onward.
    pair_start = max(0, start - 1)
    mu, sd = stats["cos_adj"]
    pairs = range(pair_start, P - 1)
    mu_v = mu[pair_start:].numpy()
    sd_v = sd[pair_start:].numpy()

    if pair_start > 0:
        ax.axvspan(0, pair_start - 0.5, color="0.88", zorder=0)
        ax.text(pair_start / 2, -0.92, "pad\n(Δ = 0)", ha="center", va="bottom",
                fontsize=6.5, color="0.35")

    ax.plot(pairs, mu_v, color="tab:purple", linewidth=1.6, label="cos(pos[p], pos[p+1])")
    ax.fill_between(pairs, mu_v - sd_v, mu_v + sd_v, color="tab:purple", alpha=0.18, linewidth=0)
    ax.axhline(1.0, color="0.6", linewidth=0.8, zorder=0)
    ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=0)
    ax.set_xlim(0, max(P - 2, 0))
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"{title}   (P={P}, active={stats['n_active']})", fontsize=10)
    ax.set_xlabel("position p (pair = pos[p] vs pos[p+1]; left-padded)", fontsize=8)
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
            stats = adjacent_cosines(cache)
            collected.append((mtag, dtag, path, stats))
            mu, _ = stats["cos_adj"]
            pair_idx = max(0, stats["first_active"] - 1)
            print(f"[ok]   {mtag:6s} {dtag:11s} L={stats['n_layers']:3d} "
                  f"P={stats['n_positions']:4d} H={stats['hidden']:5d}  "
                  f"cos(pos[first],pos[first+1])={mu[pair_idx].item():+.3f}  {path.parent.name}")

    if not collected:
        sys.exit("No steering caches found. Nothing to do.")

    summary_path = out_dir / "cosine_change_summary.csv"
    with open(summary_path, "w") as f:
        f.write("model,dataset,n_layers,n_positions,hidden,n_active,coverage,first_active,"
                "cos_adj_mean,cos_adj_std,cos_adj_firstpair,cache_path\n")
        for mtag, dtag, path, s in collected:
            mu, sd = s["cos_adj"]
            pair_idx = max(0, s["first_active"] - 1)
            f.write(f"{mtag},{dtag},{s['n_layers']},{s['n_positions']},{s['hidden']},"
                    f"{s['n_active']},{s['n_active'] / s['n_positions']:.4f},{s['first_active']},"
                    f"{mu[pair_idx:].mean().item():.6f},{sd[pair_idx:].mean().item():.6f},"
                    f"{mu[pair_idx].item():.6f},"
                    f"{path.relative_to(REPO_ROOT)}\n")

    positional_path = out_dir / "cosine_change_positional.csv"
    with open(positional_path, "w") as f:
        f.write("model,dataset,position,active,cos_adj,cos_adj_std\n")
        for mtag, dtag, _, s in collected:
            mu, sd = s["cos_adj"]
            pair_start = max(0, s["first_active"] - 1)
            for p in range(s["n_positions"] - 1):
                f.write(f"{mtag},{dtag},{p},{int(p >= pair_start)},"
                        f"{mu[p].item():.6f},{sd[p].item():.6f}\n")

    print(f"\nwrote {summary_path.relative_to(REPO_ROOT)}")
    print(f"wrote {positional_path.relative_to(REPO_ROOT)}")

    print("\nMODEL  DATASET        L    P  ACTIVE  COVERAGE  mean cos(adj)  cos(adj)[first pair]")
    print("-----  -----------  ---  ---  ------  --------  -------------  --------------------")
    for mtag, dtag, _, s in collected:
        mu, _ = s["cos_adj"]
        pair_idx = max(0, s["first_active"] - 1)
        print(f"{mtag:5s}  {dtag:11s}  {s['n_layers']:3d}  {s['n_positions']:3d}  "
              f"{s['n_active']:6d}  {s['n_active'] / s['n_positions']:8.2f}  "
              f"{mu[pair_idx:].mean().item():+13.3f}  {mu[pair_idx].item():+20.3f}")

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
    grid_path = out_dir / "adjacent_grid.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"wrote {grid_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
