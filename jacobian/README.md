# jacobian/

## TODO

- **Per-position steering vectors.** The direction is one vector `v` added at all steered prompt positions (`J = mean_p J_p`). Instead, keep the per-source-position blocks `J_p = mean_t ∂h_target[t] / ∂h_L[p]` and solve jointly for a separate `v_p` at each steered position, `Σ_p J_p v_p ≈ Δ` (ridge-regularized: underdetermined), the Jacobian counterpart of `positional` steering. It needs no extra backward passes: the same cotangents, with the layer-L gradient kept per position instead of averaged over `p` (`[P, H, H]`, ~1.4 GB fp32 for ~22 steered positions at H = 4096). Prompts differ in length, so the positions need aligning first (e.g. relative to the end of the prompt, as `positional` does).

## What's here

Prompt → response Jacobian lenses from a layer L to the final layer, and the steering directions they imply, for the verse/prose dataset. Full details are in the `jacobian/fit_lens.py` docstring.

| File | What it does |
|------|--------------|
| `fit_lens.py` | Fits the Jacobians on the prose and verse training prompts + their greedy responses, then turns the verse − prose shift in the final layer's response activations into a layer-L steering direction per variant |
| `run_fit_lens.sh` | sbatch wrapper (`mit_preemptable`, H100, `--requeue`); passes its arguments through to `fit_lens.py` and resumes from the checkpoint on requeue |

With `p` over the steered prompt positions (`content_start` to the end of the prompt) and `t` over the response tokens, two reductions from the same backward passes, each averaged over the sequences of a fit set (prose or verse; every run fits both):

| Reduction | Per-sequence `J` | Meaning |
|---|---|---|
| `mean` | `mean_t mean_p ∂h_target[t] / ∂h_L[p]` | average effect of a layer-L prompt activation on a final-layer response activation |
| `sum` | `mean_t Σ_p ∂h_target[t] / ∂h_L[p]` | first-order effect of adding `v` at every steered prompt position, as steering does |

| Variant | `J` | Meaning |
|---|---|---|
| `mean`, `sum` | `J_prose` | linearized at the base, where steering starts |
| `both-mean`, `both-sum` | `(J_prose + J_verse) / 2` | trapezoid estimate of the Jacobian along the path from prose to verse, for shifts too large to be first-order at prose |

Each direction solves `J v ≈ Δ`, where `Δ` is the verse − prose difference-in-means of the final-layer activations over the response tokens. With the SVD `J = Σ σ_i u_i w_iᵀ`, every solver is `v = Σ f(σ_i) (u_iᵀΔ) w_i`, all three written per variant:

| Solver | `f(σ)` | Meaning |
|---|---|---|
| `ridge` | `σ / (σ² + λ)`, `λ = RIDGE · tr(JᵀJ)/H` | `(JᵀJ + λI)⁻¹ JᵀΔ`: smooth roll-off below `σ ≈ √λ` |
| `pinv` | `1/σ` above float64 precision | Moore–Penrose pseudoinverse (`λ → 0`): every direction inverted, noise included |
| `trunc` | `1/σ` for `σ > √λ` | sharp cutoff at the same point ridge rolls off |

`summary.json` also holds each variant's singular values and a truncation curve (‖v‖, residual, cosine with the ridge direction over ranks k). Every run refits all variants from scratch and overwrites the previous outputs; only a requeue of the same Slurm job resumes.

## Running

From the repo root:

```bash
sbatch jacobian/run_fit_lens.sh --model llama --layer 21
```

Flags: `--model` (default `llama`), `--dataset` (`verse`), `--layer` (round(2/3 × layers)), `--max-new-tokens` (256, also cuts reused responses). The target is always the final layer; `N_PROMPTS`, `DIM_BATCH`, `RIDGE` and `CHECKPOINT_EVERY` are constants in `fit_lens.py`.

Outputs go to `jacobian/<model>/from_prose_to_verse/layer=<L>/`: `J_{mean,sum}_fit={prose,verse}.pt`, `direction_<variant>-<solver>.pt`, `delta_target.pt`, `summary.json` (with `‖J_verse − J_prose‖ / ‖J_prose‖`), and the layer-L / target-layer activations under `acts/`. Steer with them via `--type jacobian-<variant>-<solver>`: `jacobian/verse_llama_jacobian.yaml` runs them normalized and unnormalized; its `jacobian` block lists the `fit` (prose | both), `pooling` (mean | sum) and `solver` (ridge | pinv | trunc) to use, and `scripts/load_settings.py` runs every combination, one Slurm job per fit × pooling pair (covering its solvers). Fit before launching: the dispatchers refuse to submit while a listed direction is missing.
