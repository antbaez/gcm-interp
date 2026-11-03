"""
Prompt -> response Jacobian lenses from layer L to the target layer, fit on the base (prose)
and source (verse) training prompts + their greedy responses, and the steering directions they
imply.

Sequences: the first N_PROMPTS `<add>-train` (verse) and `<sub>-train` (prose) prompts, chat-templated
with the generation prompt and tokenized with add_special_tokens=False (as DataHandler does),
each followed by its greedy unsteered response, up to --max-new-tokens. Responses are reused from
activations/<model>/<task>/pt/<name>_response_layer=*.pt (checked against the prompt ids, cut to
--max-new-tokens) or, when that file is missing, generated greedily here. One teacher-forced forward pass over
prompt + response gives every position's activations; sequences with an empty response are
dropped.

Positions, for a sequence with prompt length P, content start cs (first token of the user's
text, content_token_positions()) and response length R:
    prompt positions    cs .. P-1, the prompt positions steering is added to
    response positions  P .. P+R-1, the response tokens (as saved by --hook_response_acts)

Per-sequence Jacobian, one per reduction over the prompt positions p:
    mean   J = mean over response positions t, mean over p, of d h_target[t] / d h_L[p]
           the average effect of a layer-L prompt activation on a target-layer response activation
    sum    J = mean over response positions t, sum over p, of d h_target[t] / d h_L[p]
           the first-order effect of adding v at every steered prompt position, as steering does
Both come from the same ceil(H / DIM_BATCH) backward passes per sequence: batch element b
carries a one-hot cotangent at output dimension d0+b on every response position (weighted 1/R),
and the gradient at layer L is summed over the prompt positions (sum) and divided by P - cs
(mean). Row d of J is d h_target[d] / d h_L, so J maps a layer-L shift to a target-layer shift.
Each is averaged over the sequences of one fit set, sub (prose) or add (verse). Every run fits
both sets from scratch; only a requeue of the same Slurm job resumes.

Variants (VARIANTS), each a reduction and the fit sets its J is averaged over:
    mean, sum             J_sub: linearized at the base, where steering starts
    both-mean, both-sum   (J_sub + J_add) / 2: the trapezoid estimate of the Jacobian averaged
                          along the path from base to source, for steering shifts too large to
                          be first-order at the base alone

Direction: Delta = mean over add sequences - mean over sub sequences of the target-layer
activations averaged over each sequence's response positions. Each variant's direction solves
J v ~= Delta, the layer-L shift whose first-order effect on the target layer best matches
add - sub. With the SVD J = sum_i s_i u_i w_i^T, every solver is v = sum_i f(s_i) (u_i . Delta) w_i
for a different filter f (SOLVERS):
    ridge   f = s / (s^2 + lambda),  lambda = RIDGE * trace(J^T J) / H
            i.e. v = (J^T J + lambda I)^-1 J^T Delta: smooth roll-off below s ~ sqrt(lambda)
    pinv    f = 1/s for s > H * eps * s_max (float64), else 0: the Moore-Penrose pseudoinverse,
            the lambda -> 0 limit of ridge, every numerically nonzero direction inverted
    trunc   f = 1/s for s > sqrt(lambda), else 0: the same cutoff ridge rolls off at, sharp
            instead of smooth

Outputs, in jacobian/<model>/<task>/layer=<L>/:
    acts/<name>_layer=<L>.pt, acts/<name>_layer=<T>.pt
                          per-sequence bf16 [R_i, H] activations at the response positions
                          (`acts`), plus `positions`, `input_ids`, `prompt_len`,
                          `content_start`; the layer-L file also holds per-prompt `prompt_last`
                          (last prompt token) and `prompt_mean` (mean over cs onward), bf16 [n, H]
    J_<reduction>_fit=<name>.pt
                          {"J": [H, H] fp32, "reduction", "layer", "target_layer", "n_sequences",
                          "fit_set"}, one per reduction and fit set (name: prose / verse)
    delta_target.pt
    direction_<variant>-<solver>.pt
                          {"direction": [H] fp32, plus solver params / residual / metadata},
                          steered with --type jacobian-<variant>-<solver>
    summary.json          per-direction norm, fit residual, cosines with the other directions and
                          the layer-L last-token / mean / response difference-in-means; per
                          variant, cond(J), its singular values, and a truncation curve (norm,
                          residual, cos with the ridge direction over ranks k);
                          ||J_add - J_sub|| / ||J_sub||
    checkpoint_fit=<name>.pt
                          resumable running sums (removed once that set's Jacobians are saved)

Usage:
    python jacobian/fit_lens.py --model llama --layer 21
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.data_handler import content_token_positions
from utils.dataset_registry import DATASETS, data_path, task_dir

MODEL_IDS = {
    "olmo": "allenai/OLMo-2-1124-13B-DPO",
    "qwen": "Qwen/Qwen1.5-14B-Chat",
    "qwen3": "Qwen/Qwen3-14B",
    "gemma": "google/gemma-3-12b-it",
    "gemma4": "google/gemma-4-12B-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


class Recorder:
    """Forward hooks storing each listed block's output (not detached). The block at `root`
    gets requires_grad_(True), so with frozen weights the autograd graph starts there."""

    def __init__(self, layers, at, root=None):
        self.layers, self.at, self.root = layers, sorted(set(at)), root
        self.acts, self._handles = {}, []

    def _hook(self, idx):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            if idx == self.root:
                t.requires_grad_(True)
            self.acts[idx] = t
        return hook

    def __enter__(self):
        self._handles = [self.layers[i].register_forward_hook(self._hook(i)) for i in self.at]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def text_stack(hf_model):
    """(decoder module run without the LM head, its layers)."""
    inner = hf_model.model
    text = inner.language_model if hasattr(inner, "language_model") else inner
    return text, text.layers


REDUCTIONS = ("mean", "sum")
VARIANTS = {"mean": ("mean", ("sub",)), "sum": ("sum", ("sub",)),
            "both-mean": ("mean", ("sub", "add")), "both-sum": ("sum", ("sub", "add"))}
SOLVERS = ("ridge", "pinv", "trunc")
N_PROMPTS = 100
DIM_BATCH = 8
RIDGE = 1e-3
CHECKPOINT_EVERY = 5
DEVICE = "cuda"


def response_positions(prompt_len, response_len):
    """Positions of the response tokens: P .. P+R-1."""
    return torch.arange(prompt_len, prompt_len + response_len)


def sequence_jacobians(text_model, layers, seq, layer, target, dim_batch):
    """{variant: [H, H] fp32} Jacobians of one sequence (see the module docstring)."""
    ids, P, cs, resp = seq["ids"], seq["prompt_len"], seq["content_start"], seq["steps"]
    S = len(ids)
    cot_w, src_w = torch.zeros(S), torch.zeros(S)
    cot_w[resp] = 1 / len(resp)
    src_w[cs:P] = 1
    with Recorder(layers, at=[layer, target], root=layer) as rec, torch.enable_grad():
        device = next(text_model.parameters()).device
        text_model(input_ids=ids.to(device)[None].expand(dim_batch, -1), use_cache=False)
        h_t, h_l = rec.acts[target], rec.acts[layer]
        H = h_t.shape[-1]
        J = torch.zeros(H, H, dtype=torch.float32, device=h_l.device)
        cot_w, src_w = cot_w.to(h_t.device, h_t.dtype), src_w.to(h_l.device)
        cot = torch.zeros_like(h_t)
        b = torch.arange(dim_batch, device=h_t.device)
        pos = torch.arange(S, device=h_t.device)
        n_passes = math.ceil(H / dim_batch)
        for k, d0 in enumerate(range(0, H, dim_batch)):
            n = min(dim_batch, H - d0)
            cot.zero_()
            cot[b[:n, None], pos[None, :], d0 + b[:n, None]] = cot_w[None, :].expand(n, S)
            (g,) = torch.autograd.grad(h_t, h_l, grad_outputs=cot, retain_graph=k < n_passes - 1)
            J[d0:d0 + n] = torch.einsum("bsh,s->bh", g[:n].float(), src_w)
            del g
    J = J.cpu()
    return {"mean": J / (P - cs), "sum": J}


def load_prompts(tok, path, n, template_kwargs):
    rows = [json.loads(line) for line in open(path) if line.strip()][:n]
    seqs = []
    for i, row in enumerate(rows):
        templated = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False,
                                            **template_kwargs)
        toks = tok(templated, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        cs = content_token_positions(tok, [templated], [row["prompt"][-1]["content"]], toks)["content-start"]
        seqs.append({"prompt_ids": toks["input_ids"][0], "content_start": int(cs[0])})
    return seqs


def stop_ids(tok, hf_model):
    ids = {tok.eos_token_id}
    eos = getattr(getattr(hf_model, "generation_config", None), "eos_token_id", None)
    if eos is not None:
        ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
    ids.discard(None)
    return ids


def attach_responses(seqs, name, acts_dir, tok, hf_model, max_new_tokens):
    files = sorted(acts_dir.glob(f"{name}_response_layer=*.pt"))
    if files:
        saved = torch.load(files[0], map_location="cpu", weights_only=False)
        if len(saved["input_ids"]) < len(seqs):
            raise SystemExit(f"{files[0]} has {len(saved['input_ids'])} responses, need {len(seqs)}")
        for i, s in enumerate(seqs):
            if not torch.equal(saved["prompt_ids"][i].long(), s["prompt_ids"].long()):
                raise SystemExit(f"{files[0]} prompt {i} doesn't match this tokenization")
            ids = saved["input_ids"][i].long()
            s["response_ids"] = ids[:max_new_tokens]
            s["finished"] = bool(saved["finished"][i]) and len(ids) <= max_new_tokens
        print(f"{name}: reusing {len(seqs)} responses from {files[0]}")
    else:
        stops = stop_ids(tok, hf_model)
        device = next(hf_model.parameters()).device
        print(f"{name}: no saved responses in {acts_dir}; generating {len(seqs)} greedily")
        for s in seqs:
            with torch.no_grad():
                out = hf_model.generate(s["prompt_ids"][None].to(device), do_sample=False, top_p=None,
                                        top_k=None, temperature=None, max_new_tokens=max_new_tokens,
                                        pad_token_id=tok.eos_token_id)
            gen = out[0, len(s["prompt_ids"]):].tolist()
            stop = next((k for k, t in enumerate(gen) if t in stops), None)
            s["response_ids"] = torch.tensor(gen[:stop], dtype=torch.long)
            s["finished"] = stop is not None
    empty = [i for i, s in enumerate(seqs) if len(s["response_ids"]) == 0]
    if empty:
        print(f"{name}: dropping {len(empty)} sequences with an empty response: {empty}")
    seqs[:] = [s for s in seqs if len(s["response_ids"]) > 0]
    for s in seqs:
        s["ids"] = torch.cat([s["prompt_ids"].long(), s["response_ids"]])
        s["prompt_len"] = len(s["prompt_ids"])
        s["steps"] = response_positions(s["prompt_len"], len(s["response_ids"]))


def capture_acts(text_model, layers, seqs, name, layer, target, out_dir):
    """Layer-L and target-layer activations at each sequence's response positions (cached)."""
    paths = {l: out_dir / f"{name}_layer={l}.pt" for l in (layer, target)}
    if all(p.exists() for p in paths.values()):
        cached = {l: torch.load(p, map_location="cpu", weights_only=False) for l, p in paths.items()}
        if all(len(c["positions"]) == len(seqs) and all(torch.equal(a, s["steps"]) for a, s in zip(c["positions"], seqs))
               for c in cached.values()):
            return cached
        print(f"{name}: cached activations in {out_dir} are at other positions; recomputing")
    device = next(text_model.parameters()).device
    acts = {layer: [], target: []}
    prompt_last, prompt_mean = [], []
    with Recorder(layers, at=[layer, target]) as rec, torch.no_grad():
        for s in seqs:
            text_model(input_ids=s["ids"].to(device)[None], use_cache=False)
            for l in (layer, target):
                acts[l].append(rec.acts[l][0, s["steps"]].to(torch.bfloat16).cpu())
            h = rec.acts[layer][0].float()
            prompt_last.append(h[s["prompt_len"] - 1].to(torch.bfloat16).cpu())
            prompt_mean.append(h[s["content_start"]:s["prompt_len"]].mean(0).to(torch.bfloat16).cpu())
    meta = {"positions": [s["steps"] for s in seqs], "input_ids": [s["ids"] for s in seqs],
            "prompt_len": torch.tensor([s["prompt_len"] for s in seqs]),
            "content_start": torch.tensor([s["content_start"] for s in seqs])}
    out = {}
    for l, p in paths.items():
        out[l] = {"acts": acts[l], "layer": l, **meta}
        if l == layer:
            out[l].update(prompt_last=torch.stack(prompt_last), prompt_mean=torch.stack(prompt_mean))
        atomic_save(out[l], p)
    print(f"{name}: saved activations at layers {layer} and {target} -> {out_dir}")
    return out


def atomic_save(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit(text_model, layers, seqs, layer, target, dim_batch, ckpt_path, checkpoint_every, fit_set):
    """Running mean over `seqs` of each reduction's per-sequence Jacobian."""
    H = text_model.config.hidden_size
    state = {"sums": {r: torch.zeros(H, H) for r in REDUCTIONS}, "n_done": 0, "next_idx": 0}
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if (saved["layer"], saved["target_layer"], saved["fit_set"]) != (layer, target, fit_set):
            raise SystemExit(f"{ckpt_path} was fit with a different layer/target/fit set; delete it")
        state = saved
        print(f"resuming from {ckpt_path}: {state['next_idx']}/{len(seqs)} sequences done")
    sqrt_h = math.sqrt(H)
    for i, s in enumerate(seqs):
        if i < state["next_idx"]:
            continue
        t0 = time.perf_counter()
        J = sequence_jacobians(text_model, layers, s, layer, target, dim_batch)
        for r in REDUCTIONS:
            state["sums"][r] += J[r]
        state["n_done"] += 1
        state["next_idx"] = i + 1
        print(f"  sequence {i + 1}/{len(seqs)}  len={len(s['ids'])} response={len(s['steps'])}  "
              f"{time.perf_counter() - t0:.0f}s  ||J||/sqrt(H): " +
              "  ".join(f"{r} {J[r].norm().item() / sqrt_h:.3f}" for r in REDUCTIONS), flush=True)
        if state["next_idx"] % checkpoint_every == 0:
            atomic_save({**state, "layer": layer, "target_layer": target, "fit_set": fit_set}, ckpt_path)
    return {r: state["sums"][r] / state["n_done"] for r in REDUCTIONS}, state["n_done"]


def mean_steps(acts):
    """Mean over sequences of each sequence's mean activation over its response positions."""
    return torch.stack([a.float().mean(0) for a in acts["acts"]]).mean(0)


def solve(J, delta, ridge, device):
    """{solver: (direction [H] fp32, stats)} for J v ~= delta, plus J's spectrum stats and
    truncation curve (see the module docstring)."""
    J64, d64 = J.double().to(device), delta.double().to(device)
    H = J64.shape[1]
    U, S, Wh = torch.linalg.svd(J64)
    coef = U.T @ d64
    lam = ridge * (S ** 2).sum() / H
    tau = lam.sqrt()
    pinv_cut = H * torch.finfo(torch.float64).eps * S.max()
    inv = torch.where(S > 0, 1 / S.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(S))
    keep = S > tau
    filters = {"ridge": S / (S ** 2 + lam), "pinv": torch.where(S > pinv_cut, inv, 0), "trunc": torch.where(keep, inv, 0)}
    params = {"ridge": {"lambda": float(lam), "ridge": ridge},
              "pinv": {"cutoff": float(pinv_cut), "rank": int((S > pinv_cut).sum())},
              "trunc": {"cutoff": float(tau), "rank": int(keep.sum())}}

    def fit_stats(v):
        pred = J64 @ v
        return {"norm": float(v.norm()), "residual": float((pred - d64).norm() / d64.norm()),
                "cos_Jv_delta": float(F.cosine_similarity(pred, d64, dim=0))}

    out = {}
    for name, f in filters.items():
        v = Wh.T @ (f * coef)
        out[name] = (v.float().cpu(), {**params[name], **fit_stats(v)})
    v_ridge = out["ridge"][0].double().to(device)
    curve = []
    for k in sorted({2 ** i for i in range(int(math.log2(H)) + 1)} | {H}):
        v = Wh[:k].T @ (inv[:k] * coef[:k])
        curve.append({"rank": k, "sigma_k": float(S[k - 1]), **fit_stats(v),
                      "cos_ridge": float(F.cosine_similarity(v, v_ridge, dim=0))})
    spectrum = {"J_cond": float(S.max() / S.min()), "J_sigma_max": float(S.max()), "J_sigma_min": float(S.min()),
                "singular_values": S.float().cpu().tolist(), "truncation_curve": curve,
                "frac_delta_in_top_k": {k: float(coef[:k].norm() / coef.norm()) for k in (16, 64, 256, 1024)}}
    return out, spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama", choices=sorted(MODEL_IDS))
    parser.add_argument("--dataset", default="verse", help="tag in utils/dataset_registry.py")
    parser.add_argument("--layer", type=int, default=None, help="default round(2/3 x num_layers)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="response length limit")
    args = parser.parse_args()

    from utils.model_handler import ModelHandler

    model_id = MODEL_IDS[args.model]
    model_name = model_id.split("/")[-1]
    d = DATASETS[args.dataset]
    task = task_dir(args.dataset)
    config = SimpleNamespace(args=SimpleNamespace(model_id=model_id, device=DEVICE, thinking=False, no_model_cache=True))
    mh = ModelHandler(config)
    hf_model, tok = mh.model._model, mh.tokenizer
    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    text_model, layers = text_stack(hf_model)
    n_layers = len(layers)
    layer = min(math.floor(n_layers * 2 / 3 + 0.5), n_layers - 1) if args.layer is None else args.layer
    target = n_layers - 1
    if not 0 <= layer < target < n_layers:
        raise SystemExit(f"need 0 <= layer ({layer}) < target layer ({target}) < {n_layers}")

    out_dir = REPO_ROOT / "jacobian" / model_name / task / f"layer={layer}"
    acts_out = out_dir / "acts"
    acts_out.mkdir(parents=True, exist_ok=True)
    if not int(os.environ.get("SLURM_RESTART_COUNT", 0)):
        for old in [*out_dir.glob("J_mean_fit=*.pt"), *out_dir.glob("J_sum_fit=*.pt"), *out_dir.glob("checkpoint_fit=*.pt")]:
            old.unlink()
    acts_dir = REPO_ROOT / "activations" / model_name / task / "pt"
    print(f"{model_name} · {task} · layer {layer} -> target {target} · out {out_dir}")

    seqs, acts = {}, {}
    for side in ("add", "sub"):
        name = d[side]
        seqs[side] = load_prompts(tok, data_path(REPO_ROOT / "data", model_name, args.dataset, side, "train"),
                                  N_PROMPTS, mh.template_kwargs)
        attach_responses(seqs[side], name, acts_dir, tok, hf_model, args.max_new_tokens)
        acts[side] = capture_acts(text_model, layers, seqs[side], name, layer, target, acts_out)

    for side in ("sub", "add"):
        J_paths = {r: out_dir / f"J_{r}_fit={d[side]}.pt" for r in REDUCTIONS}
        if all(p.exists() for p in J_paths.values()):
            print(f"{d[side]}: Jacobians already in {out_dir}")
            continue
        ckpt = out_dir / f"checkpoint_fit={d[side]}.pt"
        print(f"fitting on {len(seqs[side])} {d[side]} sequences "
              f"({math.ceil(text_model.config.hidden_size / DIM_BATCH)} backward passes each)")
        J, n_fit = fit(text_model, layers, seqs[side], layer, target, DIM_BATCH, ckpt,
                       CHECKPOINT_EVERY, side)
        for r, p in J_paths.items():
            torch.save({"J": J[r], "reduction": r, "layer": layer, "target_layer": target, "n_sequences": n_fit,
                        "fit_set": d[side]}, p)
        ckpt.unlink(missing_ok=True)

    fitted = {side: {r: torch.load(out_dir / f"J_{r}_fit={d[side]}.pt", map_location="cpu") for r in REDUCTIONS}
              for side in ("sub", "add")}

    delta = mean_steps(acts["add"][target]) - mean_steps(acts["sub"][target])
    torch.save({"delta": delta, "target_layer": target}, out_dir / "delta_target.pt")

    la, ls = acts["add"][layer], acts["sub"][layer]
    baselines = {"last-token diff": la["prompt_last"].float().mean(0) - ls["prompt_last"].float().mean(0),
                 "mean diff": la["prompt_mean"].float().mean(0) - ls["prompt_mean"].float().mean(0),
                 "response diff": mean_steps(la) - mean_steps(ls)}
    device = DEVICE if torch.cuda.is_available() else "cpu"
    summary = {"model": model_name, "task": task, "layer": layer, "target_layer": target,
               "n_fit": {d[side]: fitted[side]["mean"]["n_sequences"] for side in fitted}, "directions": {}}
    summary["J_rel_diff_add_vs_sub"] = {
        r: float((fitted["add"][r]["J"] - fitted["sub"][r]["J"]).norm() / fitted["sub"][r]["J"].norm())
        for r in REDUCTIONS}
    summary["spectra"] = {}
    directions = {}
    for v, (r, sides) in VARIANTS.items():
        J = torch.stack([fitted[side][r]["J"] for side in sides]).mean(0)
        solved, summary["spectra"][v] = solve(J, delta, RIDGE, device)
        for solver, (direction, stats) in solved.items():
            name = f"{v}-{solver}"
            directions[name] = direction
            torch.save({"direction": direction, "variant": v, "solver": solver, "reduction": r, "layer": layer,
                        "target_layer": target, "fit_sets": [d[side] for side in sides],
                        "n_sequences": {d[side]: fitted[side][r]["n_sequences"] for side in sides}, **stats},
                       out_dir / f"direction_{name}.pt")
            summary["directions"][name] = {"variant": v, "solver": solver, "reduction": r,
                                           "fit_sets": [d[side] for side in sides], **stats}
    named = {**{f"jacobian-{n}": u for n, u in directions.items()}, **baselines}
    summary["cosines"] = {f"jacobian-{n}": {b: float(F.cosine_similarity(directions[n], u, dim=0))
                                           for b, u in named.items()} for n in directions}
    summary["baseline_norms"] = {k: float(u.norm()) for k, u in baselines.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for v in VARIANTS:
        sp = summary["spectra"][v]
        print(f"{v}: cond(J)={sp['J_cond']:.1e}  sigma [{sp['J_sigma_min']:.2e}, {sp['J_sigma_max']:.2e}]  "
              "|Delta| in top k: " + "  ".join(f"{k} {x:.3f}" for k, x in sp["frac_delta_in_top_k"].items()))
        for solver in SOLVERS:
            n = f"{v}-{solver}"
            st = summary["directions"][n]
            print(f"  {solver:5s} rank={st.get('rank', '-')}  ||v||={st['norm']:.3g}  residual={st['residual']:.3f}  "
                  f"cos(Jv, delta)={st['cos_Jv_delta']:.3f}  " +
                  "  ".join(f"cos[{b}]={summary['cosines'][f'jacobian-{n}'][b]:.3f}" for b in baselines))
    print("||J_add - J_sub|| / ||J_sub||: " +
          "  ".join(f"{r} {x:.3f}" for r, x in summary["J_rel_diff_add_vs_sub"].items()))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
