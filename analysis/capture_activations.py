"""
Capture residual-stream activations for every prompt and response token of the
Llama sycophancy held-out split, unsteered and at each method's harmonic-floor
(N, layer) config.

Per condition (baseline, last, mean, positional) this regenerates the held-out
responses exactly as run.py does (same tokenization, left-padding to
gen_max_len, batch size, greedy decoding, prefill-only steering via
generate_with_patches), then runs one forward trace over prompt + response with
the steering vector added at the prompt positions only. That matches generation
(steering is written during prefill and reaches the response tokens through the
KV cache), so the stored activations are the ones the model actually produced.
position_ids follow generate()'s left-padding rule (cumsum(mask) - 1).

Regenerated text is checked against the pipeline's saved *_gen.json (run
scripts/run_steering.sh --split test first) and the match rate printed.

Outputs (under --out-dir):
    <condition>.safetensors   acts_L<l> [n_tokens, H] bf16 for each --layers value
                              (post-steering layer.output), plus per-token
                              prompt_idx, position (index among the row's real
                              tokens), is_response, is_template (chat-template
                              tokens after the user text: <|eot_id|> + assistant
                              header), is_bos, token_id
    <condition>_tokens.json   decoded string per stored token (for plot hovers)
    steering_vectors.safetensors  cache_L<l> [P, H] fp32: the stacked steering
                              cache slice every method's vector is derived from
    meta.json                 configs, layers, normalize flag, gen match rates

Kept tokens per row: every non-padding prompt token (incl. the chat-template
tokens after the user text), then response tokens up to and including the
first stop token (Llama pads finished rows with <|eot_id|>).

Usage:
    python analysis/capture_activations.py --device cuda:0
    python analysis/capture_activations.py --emit-run-configs /tmp/best.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = MODEL_ID.split("/")[-1]
SOURCE, BASE, SOURCE_DIR = "non-sycophantic-long", "sycophancy", "sycophancy-long"
TASK = f"from_{SOURCE}_to_{BASE}"
STREAM, SCOPE = "residuals", "local"
TEST_DATASET = f"{BASE}-heldout-test"
# Must match run_steering.sh (BATCH_SIZE / PATCHING_BATCH_SIZE) so greedy
# outputs and the steering cache are identical to the pipeline's.
EVAL_BATCH_SIZE, PATCHING_BATCH_SIZE = 50, 100

# best-configs method names -> run.py / generate_with_patches steering types
METHOD_ALIASES = {"last-token": "last", "mean-padding": "mean", "mean_padding": "mean",
                  "weighted-pos": "positional", "weighted_pos": "positional"}
METHODS = ("last", "mean", "positional")


def load_methods(best_configs_path):
    """{method: {"N": float, "layer": int}} for the sycophancy residual/local entry."""
    with open(best_configs_path) as f:
        best = json.load(f)
    entries = best[MODEL_NAME][TASK][STREAM][SCOPE]
    methods = {}
    for name, entry in entries.items():
        method = METHOD_ALIASES.get(name, name)
        if method in METHODS:
            methods[method] = {"N": float(entry["N"]), "layer": int(entry["layer"])}
    return {m: methods[m] for m in METHODS if m in methods}


def emit_run_configs(methods, path):
    """Write the model/task/stream/scope/method file run.py pins configs from."""
    best = {MODEL_NAME: {TASK: {STREAM: {SCOPE: {
        m: {**cfg, "val_pass_rate": float("nan")} for m, cfg in methods.items()}}}}}
    with open(path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Wrote run.py best-configs for {', '.join(methods)} to {path}")


def build_pipeline(device):
    """Config/ModelHandler/DataHandler set up exactly as run.py does for --split test."""
    from utils.config import Config
    from utils.model_handler import ModelHandler
    from utils.data_handler import DataHandler

    data = f"./data/{MODEL_NAME}/{SOURCE_DIR}"
    argv, sys.argv = sys.argv, [
        sys.argv[0], "-d", device, "-model_id", MODEL_ID,
        "-batch_size", str(PATCHING_BATCH_SIZE), "-seed", "42", "-max_new_tokens", "512",
        "-source", SOURCE, "-base", BASE, "-source_dir", SOURCE_DIR,
        "-steering_add_path", f"{data}/{SOURCE}-desired-all.jsonl",
        "-steering_sub_path", f"{data}/{BASE}-desired-all.jsonl",
        "-eval_batch_size", str(EVAL_BATCH_SIZE), "-eval_model", "--steering", "--resid",
    ]
    try:
        config = Config()
    finally:
        sys.argv = argv
    for key in ("source", "base", "source_dir", "steering_add_path", "steering_sub_path"):
        setattr(config.args, key, getattr(config.args, key)[0])
    config.args.eval_test = f"{data}/{TEST_DATASET}.jsonl"
    config.args.test_dataset = TEST_DATASET
    config.set_output_prefix()

    model_handler = ModelHandler(config)
    data_handler = DataHandler(config, model_handler)
    config.args.batch_size = config.args.eval_batch_size
    return config, model_handler, data_handler


def pipeline_gen_file(config, method, cfg):
    return (Path(config.get_output_prefix()) / "normalized" / STREAM / SCOPE / method /
            f"N={cfg['N']:g}_steer_layer={cfg['layer']}_{TEST_DATASET}_gen.json")


def decode_response(tokenizer, prompt_ids, full_ids):
    """Same decoding as eval/generation.py decode_responses."""
    query = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    return tokenizer.decode(full_ids, skip_special_tokens=True).split(query)[-1]


def capture_batch(model, layers_mod, seqs, prompt_mask, save_layers, steer):
    """Forward trace over prompt+response; returns {layer: [b, T, H]} on CPU.

    `steer` is None or (layer_idx, vec) with vec = N * sv, shape [H] or [P, H];
    it's added to the first P (prompt) positions only.
    """
    import torch

    b, T = seqs.shape
    P = prompt_mask.shape[1]
    attn = torch.cat([prompt_mask, torch.ones(b, T - P, dtype=prompt_mask.dtype,
                                              device=prompt_mask.device)], dim=1)
    position_ids = (attn.long().cumsum(-1) - 1).masked_fill(attn == 0, 0)

    delta = None
    if steer is not None:
        steer_layer, vec = steer
        delta = torch.zeros(T, vec.shape[-1], dtype=vec.dtype, device=vec.device)
        delta[:P] = vec

    hook_layers = sorted(set(save_layers) | ({steer[0]} if steer else set()))
    saved = {}
    with torch.no_grad():
        with model.trace({"input_ids": seqs, "attention_mask": attn}, position_ids=position_ids):
            for idx in hook_layers:
                layer = layers_mod[idx]
                if steer is not None and idx == steer[0]:
                    # Same result as generate_with_patches' in-place `+=`
                    out = layer.output
                    layer.output = (out + delta).to(out.dtype)
                if idx in save_layers:
                    saved[idx] = layer.output.detach().cpu().save()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best-configs", type=Path, default=REPO_ROOT / "best_configs_harmonic_floor.json")
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 15, 20])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "activations" / MODEL_NAME / "sycophancy")
    parser.add_argument("--force", action="store_true", help="Recapture conditions already on disk")
    parser.add_argument("--emit-run-configs", type=Path, default=None,
                        help="Only write the run.py best-configs file for these configs, then exit")
    args = parser.parse_args()

    methods = load_methods(args.best_configs)
    if args.emit_run_configs:
        emit_run_configs(methods, args.emit_run_configs)
        return

    import torch
    from safetensors.torch import save_file
    from eval.activations import steering_reps_cache, _get_layers
    from eval.eval_runner import generate_baseline
    from eval.generation import generate_with_patches, _prepare_steering_vector

    os.chdir(REPO_ROOT)  # the pipeline's data/results paths are repo-relative
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_layers = sorted(set(args.layers))

    config, model_handler, data_handler = build_pipeline(args.device)
    model, tokenizer = model_handler.model, model_handler.tokenizer
    layers_mod = _get_layers(model)
    toks = data_handler.base_qs_toks["test"]
    n_prompts, P = toks["input_ids"].shape
    bs = config.args.batch_size
    normalize = config.args.normalize

    # generate() stops on any generation_config eos id and pads with tokenizer eos
    hf_model = getattr(model, "_model", model)
    gen_eos = getattr(getattr(hf_model, "generation_config", None), "eos_token_id", None)
    gen_eos = gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos]
    stop_ids = {t for t in [tokenizer.eos_token_id, *gen_eos] if t is not None}
    print(f"[capture] {n_prompts} prompts, P={P}, layers={save_layers}, stop ids={sorted(stop_ids)}")

    # Built from (or loaded out of) the pipeline's steering cache: [num_layers, P, H]
    cache = steering_reps_cache(model, data_handler).to(model.device)
    save_file({f"cache_L{l}": cache[l].float().cpu().contiguous() for l in save_layers},
              str(args.out_dir / "steering_vectors.safetensors"))

    meta_path = args.out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"model": MODEL_ID, "task": TASK, "test_dataset": TEST_DATASET,
                 "layers": save_layers, "normalize": normalize, "gen_max_len": P,
                 "n_prompts": n_prompts, "methods": methods})
    meta.setdefault("gen_match", {})

    baseline_outputs = None
    conditions = ["baseline"] + list(methods)
    for cond in conditions:
        out_path = args.out_dir / f"{cond}.safetensors"
        if out_path.exists() and not args.force:
            print(f"[capture] {cond}: already captured ({out_path}), skipping")
            continue
        cfg = methods.get(cond)
        steer = None
        if cfg is not None:
            sv = _prepare_steering_vector(cache, cfg["layer"], cond, normalize, None)
            steer = (cfg["layer"], cfg["N"] * sv)
            print(f"\n[capture] {cond}: N={cfg['N']:g} layer={cfg['layer']} vector {tuple(sv.shape)}")
        else:
            print(f"\n[capture] baseline (unsteered)")
            if baseline_outputs is None:
                baseline_outputs = generate_baseline(config, data_handler, model_handler)

        # Pipeline text to check the regenerated responses against
        ref, ref_key = None, f"edit_{BASE}"
        ref_cfgs = [(cond, cfg)] if cfg else list(methods.items())
        for m, c in ref_cfgs:
            gen_path = pipeline_gen_file(config, m, c)
            if gen_path.exists():
                ref = json.loads(gen_path.read_text())
                ref_key = f"edit_{BASE}" if cfg else f"old_{BASE}"
                break

        acts = {l: [] for l in save_layers}
        prompt_idx, position, is_response, is_template, is_bos, token_id, token_str = [], [], [], [], [], [], []
        n_match = n_checked = 0
        t_start = time.time()
        for start in range(0, n_prompts, bs):
            t0 = time.time()
            gen_toks = {k: v[start:start + bs] for k, v in toks.items()}
            if cfg is None:
                seqs = torch.tensor(baseline_outputs[start:start + bs], device=model.device)
            else:
                seqs = generate_with_patches(
                    model, gen_toks, cache, [cfg["layer"]], cfg["N"],
                    max_new_tokens=config.args.max_new_tokens, normalize=normalize,
                    steering_type=cond, resid=True).to(model.device)

            saved = capture_batch(model, layers_mod, seqs, gen_toks["attention_mask"], save_layers, steer)

            seqs_cpu = seqs.cpu()
            pmask = gen_toks["attention_mask"].cpu()
            for r in range(seqs_cpu.shape[0]):
                i = start + r
                resp = seqs_cpu[r, P:]
                stops = [k for k, t in enumerate(resp.tolist()) if t in stop_ids]
                resp_len = stops[0] + 1 if stops else resp.shape[0]
                keep = torch.cat([torch.nonzero(pmask[r], as_tuple=True)[0],
                                  torch.arange(P, P + resp_len)])
                ids = seqs_cpu[r, keep]
                for l in save_layers:
                    acts[l].append(saved[l][r, keep].to(torch.bfloat16))
                n_keep = keep.shape[0]
                n_prompt_tok = n_keep - resp_len
                # Chat-template tokens after the user's text: from the <|eot_id|>
                # closing the user turn through the assistant header
                prompt_ids = ids[:n_prompt_tok].tolist()
                eot_at = max((k for k, t in enumerate(prompt_ids) if t == tokenizer.eos_token_id),
                             default=n_prompt_tok)
                template = torch.zeros(n_keep, dtype=torch.bool)
                template[eot_at:n_prompt_tok] = True
                is_template.append(template)
                prompt_idx.append(torch.full((n_keep,), i, dtype=torch.int32))
                position.append(torch.arange(n_keep, dtype=torch.int32))
                is_response.append(torch.arange(n_keep) >= n_prompt_tok)
                is_bos.append(ids == tokenizer.bos_token_id)
                token_id.append(ids.to(torch.int32))
                token_str.extend(tokenizer.decode([t]) for t in ids.tolist())

                if ref is not None and i < len(ref):
                    text = decode_response(tokenizer, gen_toks["input_ids"][r], seqs_cpu[r])
                    n_checked += 1
                    n_match += text == ref[i][ref_key]
            del saved
            torch.cuda.empty_cache()
            print(f"  batch {start // bs + 1}/{-(-n_prompts // bs)} done in {time.time() - t0:.1f}s")

        tensors = {f"acts_L{l}": torch.cat(acts[l]).contiguous() for l in save_layers}
        tensors.update({
            "prompt_idx": torch.cat(prompt_idx), "position": torch.cat(position),
            "is_response": torch.cat(is_response), "is_template": torch.cat(is_template),
            "is_bos": torch.cat(is_bos),
            "token_id": torch.cat(token_id),
        })
        tmp = out_path.with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata={"condition": cond, **({k: str(v) for k, v in cfg.items()} if cfg else {})})
        os.replace(tmp, out_path)
        (args.out_dir / f"{cond}_tokens.json").write_text(json.dumps(token_str))

        n_tok = tensors["token_id"].shape[0]
        n_resp = int(tensors["is_response"].sum())
        if ref is None:
            print(f"[capture] {cond}: no pipeline gen file found to compare against")
            meta["gen_match"][cond] = None
        else:
            print(f"[capture] {cond}: regenerated text matches pipeline for {n_match}/{n_checked} prompts")
            meta["gen_match"][cond] = {"matched": n_match, "checked": n_checked}
        print(f"[capture] {cond}: saved {n_tok} tokens ({n_tok - n_resp} prompt, {n_resp} response) "
              f"to {out_path} in {time.time() - t_start:.0f}s")
        meta_path.write_text(json.dumps(meta, indent=2))

    meta_path.write_text(json.dumps(meta, indent=2))
    print("\n[capture] done")


if __name__ == "__main__":
    main()
