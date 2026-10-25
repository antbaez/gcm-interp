"""
Project unsteered prompt activations onto the mean and positional steering directions.

For the first --n-prompts sycophancy test prompts, runs one forward pass per model
and, at every prompt position p of each chosen layer l, projects the residual-stream
activation h[l, p] onto two unit directions:

    mean        m_hat[l]    = masked_mean cache[l] / ||.||          (one direction for all positions)
    positional  v_hat[l, p] = positional cache[l, p] / ||.||        (position-specific direction)

so proj = <h[l, p], direction>. Only steered positions are kept (each prompt's content
start onward; pre-prompt chat template and left padding are NaN), the same positions the
steering vector is written to at generation time. Prompts are left-padded to the
pipeline's gen_max_len, so the x-axis is that padded position index and lines up with
the cache's position axis.

Outputs (under --out-dir, default projection_results/):
    <model>/projections.npz   mean_proj / pos_proj [layers_saved, n_prompts, P] (NaN off-mask),
                              layers, steer_mask [n_prompts, P], token_id [n_prompts, P]
    <model>/tokens.json       decoded token per (prompt, position)
    <model>/layer_<l>.png     one panel per prompt, mean vs positional projection by position
    meta.json                 models, layers, split, prompt count

Steering caches are read from results/ if present (built by run_steering.sh); otherwise
they are built here from the add/sub steering prompts.

Usage (from the repo root, on a GPU node):
    python project_activations.py
    python project_activations.py --models llama --layers 14 20
    python project_activations.py --models llama --n-prompts 5 --split test
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "olmo": "allenai/OLMo-2-1124-13B-DPO",
    "qwen3": "Qwen/Qwen3-14B",
    "gemma4": "google/gemma-4-12B-it",
}
DEFAULT_MODELS = ("llama", "olmo", "qwen3")
SOURCE, BASE, SOURCE_DIR = "non-sycophantic-long", "sycophancy", "sycophancy-long"
EVAL_BATCH_SIZE, PATCHING_BATCH_SIZE = 50, 100
LAYER_FRAC = 0.6667

COLOR_MEAN, COLOR_POS = "#0072B2", "#E69F00"


def build_pipeline(model_id, device, test_dataset):
    from utils.config import Config
    from utils.model_handler import ModelHandler
    from utils.data_handler import DataHandler

    model_name = model_id.split("/")[-1]
    data = f"./data/{model_name}/{SOURCE_DIR}"
    argv, sys.argv = sys.argv, [
        sys.argv[0], "-d", device, "-model_id", model_id,
        "-batch_size", str(PATCHING_BATCH_SIZE), "-seed", "42", "-max_new_tokens", "256",
        "-source", SOURCE, "-base", BASE, "-source_dir", SOURCE_DIR,
        "-steering_add_path", f"{data}/{SOURCE}-desired-all.jsonl",
        "-steering_sub_path", f"{data}/{BASE}-desired-all.jsonl",
        "-eval_batch_size", str(EVAL_BATCH_SIZE), "-eval_model", "--steering",
    ]
    try:
        config = Config()
    finally:
        sys.argv = argv
    for key in ("source", "base", "source_dir", "steering_add_path", "steering_sub_path"):
        setattr(config.args, key, getattr(config.args, key)[0])
    config.args.eval_test = f"{data}/{test_dataset}.jsonl"
    config.args.test_dataset = test_dataset
    config.set_output_prefix()

    model_handler = ModelHandler(config)
    data_handler = DataHandler(config, model_handler)
    return config, model_handler, data_handler


def forward_layers(model, layers_mod, input_ids, attention_mask, layer_ids, is_gemma3):
    """{layer: [b, P, H] float32 on CPU} of layer.output for an unsteered forward pass."""
    import torch

    position_ids = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    saved = {}
    with torch.no_grad():
        with model.trace({"input_ids": input_ids, "attention_mask": attention_mask},
                         position_ids=position_ids):
            for idx in layer_ids:
                out = layers_mod[idx].output
                out = out[0] if is_gemma3 else out
                saved[idx] = out.detach().float().cpu().save()
    return {k: v for k, v in saved.items()}


def plot_layer(path, model_tag, layer, mean_proj, pos_proj, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = mean_proj.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.1 * n + 0.9), sharex=True, squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        x = np.arange(mean_proj.shape[1])
        ax.plot(x, mean_proj[i], color=COLOR_MEAN, lw=1.5, label="mean direction")
        ax.plot(x, pos_proj[i], color=COLOR_POS, lw=1.5, label="positional direction")
        ax.axhline(0, color="#888888", lw=0.6)
        ax.set_ylabel(f"prompt {i}\nprojection", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("token position (left-padded to gen_max_len)")
    fig.suptitle(f"{model_tag} layer {layer}: activation projected onto unit steering direction",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_model(tag, args):
    import numpy as np
    import torch
    from eval.activations import steering_reps_cache, _get_layers

    model_id = MODELS[tag]
    model_name = model_id.split("/")[-1]
    out_dir = args.out_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = f"{BASE}-{args.split}"
    config, model_handler, data_handler = build_pipeline(model_id, args.device, test_dataset)
    model, tokenizer = model_handler.model, model_handler.tokenizer
    layers_mod = _get_layers(model)
    num_layers = len(layers_mod)
    layer_ids = sorted(set(args.layers)) if args.layers else \
        [min(round(LAYER_FRAC * num_layers), num_layers - 1)]
    is_gemma3 = ("gemma" in model.config._name_or_path.lower()
                 and getattr(model.config, "model_type", "") != "gemma4_unified")

    toks = data_handler.base_qs_toks["test"]
    n = min(args.n_prompts, toks["input_ids"].shape[0])
    input_ids = toks["input_ids"][:n]
    attention_mask = toks["attention_mask"][:n]
    steer_mask = data_handler.test_steer_mask[:n].cpu()
    P = input_ids.shape[1]

    pos_cache = steering_reps_cache(model, data_handler)
    mean_cache = steering_reps_cache(model, data_handler, variant="masked_mean")
    print(f"[{tag}] layers={layer_ids} n={n} P={P} positional cache {tuple(pos_cache.shape)}")

    acts = forward_layers(model, layers_mod, input_ids.to(model.device),
                          attention_mask.to(model.device), layer_ids, is_gemma3)

    mean_proj, pos_proj = [], []
    for l in layer_ids:
        h = acts[l]
        m_hat = mean_cache[l].float().cpu()
        m_hat = m_hat / (m_hat.norm() + 1e-12)
        v = pos_cache[l].float().cpu()
        v_hat = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
        mp = torch.einsum("bph,h->bp", h, m_hat)
        pp = torch.einsum("bph,ph->bp", h, v_hat)
        nan = torch.full_like(mp, float("nan"))
        mean_proj.append(torch.where(steer_mask, mp, nan))
        pos_proj.append(torch.where(steer_mask, pp, nan))
    mean_proj = torch.stack(mean_proj).numpy()
    pos_proj = torch.stack(pos_proj).numpy()

    np.savez(out_dir / "projections.npz", mean_proj=mean_proj, pos_proj=pos_proj,
             layers=np.array(layer_ids), steer_mask=steer_mask.numpy(),
             token_id=input_ids.cpu().numpy())
    (out_dir / "tokens.json").write_text(json.dumps(
        [[tokenizer.decode([t]) for t in row] for row in input_ids.cpu().tolist()]))

    for k, l in enumerate(layer_ids):
        plot_layer(out_dir / f"layer_{l}.png", model_name, l, mean_proj[k], pos_proj[k],
                   steer_mask.numpy())
    print(f"[{tag}] wrote {out_dir}")

    del model_handler, data_handler, model, acts, pos_cache, mean_cache
    gc.collect()
    torch.cuda.empty_cache()
    return {"model": model_id, "layers": layer_ids, "P": P, "n_prompts": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help=f"comma-separated tags from: {', '.join(MODELS)}")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="layer indices (default: the layer 2/3 of the way through the model)")
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--split", choices=["valid", "test"], default="valid",
                        help="sycophancy-<split>.jsonl to take the first prompts from")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "projection_results")
    args = parser.parse_args()

    tags = [t.strip() for t in args.models.split(",")]
    for t in tags:
        if t not in MODELS:
            sys.exit(f"Unknown model tag '{t}'. Choose from: {', '.join(MODELS)}")

    os.chdir(REPO_ROOT)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"split": f"{BASE}-{args.split}", "models": {}}
    for t in tags:
        meta["models"][t] = run_model(t, args)
        (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
