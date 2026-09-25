"""
3D PCA of the token activations captured by analysis/capture_activations.py,
with the three steering vectors drawn as arrows. One interactive HTML per layer.

Per layer, a 3-component PCA is fitted on every stored prompt + response token
of all four conditions (baseline, last, mean, positional) pooled, excluding BOS
tokens, whose massive activations would otherwise dominate the components. All
tokens are projected; up to --max-points per condition are drawn (random,
seeded) so the page stays responsive. Marker shape separates prompt tokens
(circle), the chat-template tokens after the user text (square: <|eot_id|> +
assistant header, always all drawn) and response tokens (diamond).

Steering vectors come from that layer's slice of the steering cache, collapsed
the same way generate_with_patches does it (last: position -1, mean: mean over
positions, positional: every position), L2-normalized per position when the
run was normalized, then scaled by the method's N. The arrow starts at the
baseline centroid and shows the displacement actually added:
    last / mean  -> N * unit vector
    positional   -> mean over positions of N * unit_p (the per-position
                    endpoints are drawn as faint points)
Solid arrows mark the layer a method steers, dashed ones its vector at a layer
it doesn't steer. The dropdown switches between true scale and equal-length
arrows (direction only). Each arrow's legend entry shows the share of its norm
that lies in the PC1-3 subspace; a low share means the arrow is mostly
orthogonal to the plotted space, and its drawn length understates it.

With --prompt-idx, each listed held-out prompt gets its own plot per layer
showing just that prompt + response under every condition, as a token
trajectory (points joined in token order). All conditions still share one PCA
space: by default it's fitted on that prompt's tokens from all four conditions
(--fit prompt); --fit all reuses the space fitted on every prompt.

--prompt-only drops the response tokens (keeping the user text and the
chat-template tokens after it) before the PCA is fitted and plotted.

Outputs, in --in-dir (or --out-dir):
    pca_layer<L>.html              interactive 3D plot (all prompts)
    pca_layer<L>_prompt<i>.html    single-prompt plot (--prompt-idx)
    ..._promptonly.html            same, prompt tokens only (--prompt-only)
    pca_summary[_prompts].json     explained variance, arrow in-subspace shares, centroids

Usage:
    python analysis/plot_activations_pca.py
    python analysis/plot_activations_pca.py --layers 20 --max-points 30000
    python analysis/plot_activations_pca.py --prompt-idx 0 7 42
    python analysis/plot_activations_pca.py --prompt-idx 0 --prompt-only
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "activations" / "Llama-3.1-8B-Instruct" / "sycophancy"

# Baseline is the neutral reference; the three methods take categorical slots
# 1-3, the only slots that stay distinguishable (incl. under CVD) in a scatter.
COLORS = {"baseline": "#8f8d87", "last": "#2a78d6", "mean": "#eb6834", "positional": "#1baf7a"}
LABELS = {"baseline": "Baseline (unsteered)", "last": "Last-token",
          "mean": "Mean-padding", "positional": "Positional"}
# Token segment -> (marker, opacity, label)
SEGMENTS = {"prompt": ("circle", 0.35, "prompt tokens"),
            "template": ("square", 0.8, "chat-template tokens after prompt"),
            "response": ("diamond", 0.55, "response tokens")}


def unit(v, eps=1e-12):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def method_displacements(cache_l, method, N, normalize):
    """(arrow displacement [H], per-position displacements [P, H] or None)."""
    if method == "last":
        v = cache_l[-1]
    elif method == "mean":
        v = cache_l.mean(0)
    else:
        per_pos = N * (unit(cache_l) if normalize else cache_l)
        return per_pos.mean(0), per_pos
    return N * (unit(v) if normalize else v), None


def load_condition(in_dir, cond, layer):
    names = (f"acts_L{layer}", "prompt_idx", "position", "is_response", "is_template", "is_bos")
    with safe_open(str(in_dir / f"{cond}.safetensors"), framework="pt") as f:
        t = {n: f.get_tensor(n) for n in names}
    tokens = json.loads((in_dir / f"{cond}_tokens.json").read_text())
    keep = ~t["is_bos"]
    segment = np.where(t["is_response"].numpy(), "response",
                       np.where(t["is_template"].numpy(), "template", "prompt"))
    return {
        "acts": t[f"acts_L{layer}"][keep].float().numpy(),
        "prompt_idx": t["prompt_idx"][keep].numpy(),
        "position": t["position"][keep].numpy(),
        "segment": segment[keep.numpy()],
        "tokens": np.array(tokens, dtype=object)[keep.numpy()],
    }


def select_prompt(d, prompt_idx):
    mask = d["prompt_idx"] == prompt_idx
    if not mask.any():
        raise SystemExit(f"prompt {prompt_idx} not in the captured activations")
    return {k: v[mask] for k, v in d.items()}


def arrow_traces(name, color, start, end, dash, visible, legend_group, head_frac=0.12):
    """Shaft line + cone head; hidden from the legend except the shaft."""
    vec = end - start
    length = float(np.linalg.norm(vec))
    shaft = go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode="lines", line=dict(color=color, width=7, dash=dash),
        name=name, legendgroup=legend_group, visible=visible, hoverinfo="name")
    if length == 0:
        return [shaft]
    head = vec * head_frac
    cone = go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]], u=[head[0]], v=[head[1]], w=[head[2]],
        anchor="tip", sizemode="absolute", sizeref=1.0,
        colorscale=[[0, color], [1, color]], showscale=False,
        legendgroup=legend_group, showlegend=False, visible=visible, hoverinfo="skip")
    return [shaft, cone]


def fit_pca(data):
    pca = PCA(n_components=3, svd_solver="covariance_eigh")
    return pca.fit(np.concatenate([d["acts"] for d in data.values()]))


def plot_layer(meta, layer, data, vectors, max_points, rng, pca=None, prompt_idx=None):
    """`data`: condition -> loaded tokens (one prompt's when prompt_idx is set).
    `pca`: a pre-fitted PCA to reuse; fitted on `data` when None."""
    methods = meta["methods"]
    single = prompt_idx is not None
    pca = fit_pca(data) if pca is None else pca
    comps = pca.components_  # [3, H], orthonormal rows
    evr = pca.explained_variance_ratio_

    traces = []
    for cond, d in data.items():
        coords = pca.transform(d["acts"])
        idx = np.arange(len(coords))
        if not single and len(idx) > max_points:
            idx = np.sort(rng.choice(idx, max_points, replace=False))
        if single:
            # Trajectory through the tokens in order (stored in sequence order)
            traces.append(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="lines",
                line=dict(color=COLORS[cond], width=2), opacity=0.45,
                legendgroup=cond, showlegend=False, hoverinfo="skip"))
        # Template tokens are few (5 per prompt), so draw all of them
        idx = np.union1d(idx, np.nonzero(d["segment"] == "template")[0])
        for seg, (symbol, opacity, part) in SEGMENTS.items():
            sel = idx[d["segment"][idx] == seg]
            if len(sel) == 0:
                continue
            traces.append(go.Scatter3d(
                x=coords[sel, 0], y=coords[sel, 1], z=coords[sel, 2], mode="markers",
                marker=dict(size=(6 if seg == "template" else 4) if single else
                                 (3.5 if seg == "template" else 2.5), color=COLORS[cond],
                            symbol=symbol, opacity=opacity, line=dict(width=0)),
                name=f"{LABELS[cond]} · {part}", legendgroup=cond,
                customdata=np.stack([d["prompt_idx"][sel], d["position"][sel],
                                     d["tokens"][sel]], axis=-1),
                hovertemplate=(f"<b>{LABELS[cond]}</b> · {part}<br>prompt %{{customdata[0]}}, "
                               "token %{customdata[1]}<br>%{customdata[2]}<extra></extra>")))

    base_centroid = data["baseline"]["acts"].mean(0) if "baseline" in data else \
        np.concatenate([d["acts"] for d in data.values()]).mean(0)
    start = pca.transform(base_centroid[None])[0]
    # Equal-length mode: every arrow gets the length of 2 SDs along PC1
    equal_len = 2 * float(np.sqrt(pca.explained_variance_[0]))

    n_point_traces = len(traces)
    true_idx, equal_idx = [], []
    arrows_summary = {}
    for method, cfg in methods.items():
        disp, per_pos = method_displacements(vectors[layer], method, cfg["N"], meta["normalize"])
        proj = comps @ disp
        share = float(np.linalg.norm(proj) / (np.linalg.norm(disp) + 1e-12))
        steers_here = cfg["layer"] == layer
        dash = "solid" if steers_here else "dash"
        where = f"steers L{cfg['layer']}" + ("" if steers_here else f", shown at L{layer}")
        name = f"{LABELS[method]} vector (N={cfg['N']:g}, {where}) · {share:.0%} in PC1–3"
        arrows_summary[method] = {"N": cfg["N"], "steer_layer": cfg["layer"],
                                  "norm": float(np.linalg.norm(disp)), "share_in_pc123": share}

        group = f"vec_{method}"
        for mode, scale, bucket in (("true", 1.0, true_idx),
                                    ("equal", equal_len / (np.linalg.norm(proj) + 1e-12), equal_idx)):
            visible = mode == "true"
            end = start + proj * scale
            ts = arrow_traces(name, COLORS[method], start, end, dash, visible, group)
            if per_pos is not None:
                pts = start + (per_pos @ comps.T) * scale
                ts.append(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                    marker=dict(size=3, color=COLORS[method], opacity=0.3, symbol="x"),
                    name=f"{LABELS[method]} per-position endpoints", legendgroup=group,
                    showlegend=False, visible=visible,
                    hovertemplate="positional vector endpoint<extra></extra>"))
            for tr in ts:
                bucket.append(len(traces))
                traces.append(tr)

    def visibility(show):
        vis = [True] * n_point_traces + [False] * (len(traces) - n_point_traces)
        for i in show:
            vis[i] = True
        return vis

    fig = go.Figure(traces)
    scope = (f"prompt {prompt_idx} only · PCA fitted on "
             + ("this prompt's tokens" if meta.get("_fit") == "prompt" else f"all {meta['n_prompts']} prompts")
             if single else f"{meta['n_prompts']} prompts")
    axis = lambda i: dict(title=f"PC{i + 1} ({evr[i]:.1%} var)", showbackground=False,
                          gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_layout(
        title=dict(text=(f"Layer {layer} residual stream: "
                         f"{'prompt' if meta.get('_prompt_only') else 'prompt + response'} tokens, 3D PCA<br>"
                         f"<sup>{meta['model'].split('/')[-1]} · {meta['test_dataset']} · "
                         f"{scope} · arrows start at the baseline centroid</sup>"),
                   x=0.01),
        scene=dict(xaxis=axis(0), yaxis=axis(1), zaxis=axis(2), aspectmode="data"),
        legend=dict(itemsizing="constant", font=dict(size=11), groupclick="togglegroup"),
        paper_bgcolor="#fcfcfb", font=dict(color="#0b0b0b"),
        margin=dict(l=0, r=0, t=70, b=0),
        updatemenus=[dict(
            type="dropdown", x=0.01, y=0.9, xanchor="left",
            buttons=[dict(label="Arrows: true scale (N × unit)", method="restyle",
                          args=[{"visible": visibility(true_idx)}]),
                     dict(label="Arrows: equal length (direction only)", method="restyle",
                          args=[{"visible": visibility(equal_idx)}])])])

    return fig, {
        "explained_variance_ratio": evr.tolist(),
        "n_tokens_fit": {c: int(len(d["acts"])) for c, d in data.items()},
        "baseline_centroid_pc": start.tolist(),
        "condition_centroids_pc": {c: pca.transform(d["acts"].mean(0)[None])[0].tolist()
                                   for c, d in data.items()},
        "arrows": arrows_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: --in-dir")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Default: every captured layer")
    parser.add_argument("--max-points", type=int, default=15000,
                        help="Max tokens drawn per condition (PCA is always fitted on all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-idx", type=int, nargs="+", default=None,
                        help="Plot only these held-out prompts (one file each) instead of all prompts")
    parser.add_argument("--fit", choices=["prompt", "all"], default="prompt",
                        help="With --prompt-idx: fit PCA on that prompt's tokens (default) or on all prompts")
    parser.add_argument("--prompt-only", action="store_true",
                        help="Drop response tokens (keeps user text + chat-template tokens) before fitting and plotting")
    args = parser.parse_args()

    meta = json.loads((args.in_dir / "meta.json").read_text())
    out_dir = args.out_dir or args.in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = args.layers or meta["layers"]
    conditions = [c for c in ["baseline", *meta["methods"]] if (args.in_dir / f"{c}.safetensors").exists()]
    missing = set(["baseline", *meta["methods"]]) - set(conditions)
    if missing:
        print(f"Warning: no captured activations for {sorted(missing)}")
    vectors = {k: v.numpy() for k, v in
               ((int(k.split("_L")[1]), v) for k, v in load_file(str(args.in_dir / "steering_vectors.safetensors")).items())}

    meta["_fit"] = args.fit
    meta["_prompt_only"] = args.prompt_only
    suffix = "_promptonly" if args.prompt_only else ""
    prompts = args.prompt_idx or [None]
    summary = {}
    for layer in layers:
        print(f"Layer {layer}: loading {', '.join(conditions)}...")
        data = {c: load_condition(args.in_dir, c, layer) for c in conditions}
        if args.prompt_only:
            data = {c: {k: v[d["segment"] != "response"] for k, v in d.items()} for c, d in data.items()}
        shared_pca = fit_pca(data) if args.prompt_idx is None or args.fit == "all" else None
        for p in prompts:
            sub = data if p is None else {c: select_prompt(d, p) for c, d in data.items()}
            key = f"L{layer}" if p is None else f"L{layer}_prompt{p}"
            fig, summary[key] = plot_layer(meta, layer, sub, vectors, args.max_points,
                                           np.random.default_rng(args.seed), shared_pca, p)
            out = out_dir / f"pca_layer{layer}{'' if p is None else f'_prompt{p}'}{suffix}.html"
            fig.write_html(str(out), include_plotlyjs=True)
            s = summary[key]
            print(f"  {key}: PC1-3 explain {sum(s['explained_variance_ratio']):.1%} of variance; "
                  + ", ".join(f"{m} {a['share_in_pc123']:.0%}" for m, a in s["arrows"].items())
                  + f" of vector norm in PC1-3 -> {out}")
        del data
    name = f"pca_summary{'' if args.prompt_idx is None else '_prompts'}{suffix}.json"
    (out_dir / name).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
