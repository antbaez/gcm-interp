# gcm-interp

Mechanistic interpretability project that identifies and steers attention heads in LLMs to alter model behavior across four behavioral tasks: harmful refusal, sycophancy, poetry (verse), and paragraph style.

Supported models: OLMo-2-1124-13B-DPO, Qwen1.5-14B-Chat, SOLAR-10.7B-Instruct-v1.0.
Datasets: `harmful-long`, `sycophancy-long`, `verse-long`, `paragraph-long` (each paired with a "base" counterpart).

---

## run_steering.sh

Runs ATP head-patching and steered generation. Calls `run.py`.

**Steps:**

1. Parse `--model` / `--dataset` / `--device` flags; expand `all` to full lists.
2. Build a JSON `dataset_list` (source/base paths) and `steering_combos` (`last-token`, `mean`, `positional`) array.
3. Call `python run.py` once per model with all datasets and combos bundled.

**Inside `run.py → main()`:**

4. `Config()` — parses CLI args, computes output prefix under `results/`.
5. `ModelHandler(config)` — loads the HuggingFace model and tokenizer onto GPU.
6. For each dataset in `dataset_list`:
   - `DataHandler(config, model_handler)` — tokenizes paired desired/undesired JSONL files into `base_toks` and `source_qs_toks`.
   - **Patching phase** (`-patch_model` flag):
     - `Experiment(config, data_handler, model_handler, 'heads').run()`
     - Iterates batches via `BatchHandler`; each batch calls `Patching.apply_patching()`.
     - `apply_patching()` runs ATP (Attribution Patching): forward passes on desired and undesired inputs, computes gradient-weighted head-effect scores (`attn_desired_effects`, `attn_undesired_effects`), stacks them into a logit tensor, saves per-batch `.pt` files to `results/<model>/<task>/heads/heads_<idx>.pt`.
   - **Evaluation phase** (`-eval_model --steering` flags), inside `run_eval()` (`eval/eval_runner.py`):
     - `load_logits()` (`eval/logits_handler.py`) — aggregates all `.pt` batch files into a single logits tensor.
     - `load_patching_reps()` — calls `steering_reps_cache()` (`eval/activations.py`) to compute the mean steering vector (desired minus base activations) for each layer/head.
     - Generate unsteered baseline responses for all test prompts; cache to `eval/baseline_outputs.json`.
     - For each `steering_type` in combos × each `topk` × each `N`:
       - `get_top_k_layer_and_head()` (`eval/logits_handler.py`) — selects top-k% heads by ATP score; saves CSV to `eval/numerator_1_targeted_<topk>.csv`.
       - `generate_with_patches()` (`eval/generation.py`) — runs steered generation: hooks the selected heads at forward-pass time and adds the steering vector scaled by `N`.
       - `decode_responses()` — pairs each steered output with the baseline output.
       - `save_prompt_responses()` — writes `.txt` and `.json` files to `results/<model>/<task>/eval/<steering_type>/steer_<dataset>_N=<N>_k=<topk>.(txt|json)`.

---

## run_judging.sh

Scores already-generated outputs using a vLLM judge model. Calls `judge-evals/run_judge.py` then `judge-evals/summarize_results.py`.

**Steps:**

1. Parse `--model` / `--dataset` / `--device` flags.
2. For each model × dataset combination, run `python judge-evals/run_judge.py` with paths into `results/`, `data/`, and `judge-evals/`.
3. After all combinations, run `python judge-evals/summarize_results.py`.

**Inside `run_judge.py → main()`:**

**Phase 1 — Prepare** (`phase1_prepare()`):

4. `discover_gen_files()` (`merge_outputs.py`) — glob `results/` for `*_gen.json` files matching the model/source/base filters.
5. `extract_path_metadata()` — parses path components into a metadata dict (model, source, base, method, topk, N, steering type).
6. `gen_to_csv()` (`merge_outputs.py`) — converts each `_gen.json` to `eval_output.csv` inside a per-file workdir under `judge-evals/workdirs/`.
7. Load tokenizer once (`load_tokenizer()` from `build_prompts.py`).
8. For long-eval files, build prompt CSVs:
   - `build_fluency_prompts()` + `build_relevance_prompts()` → `relevance_fluency_prompts.csv`
   - `build_judge_prompts()` → `judge_prompts.csv`

**Phase 2 — Evaluate** (`phase2_evaluate()`):

9. Single-eval files: `compute_accuracy_for_file()` (`compute_single_accuracies.py`) — token matching, no GPU needed.
10. Long-eval files: load vLLM judge model once (`make_llm()` from `evaluator.py`).
11. `_evaluate_all_workdirs_batched()` — for each mode (`fluency`, `relevance`, `judge`):
    - Collects all prompts from all workdirs into one list.
    - Runs `generate_in_batches()` in a single batched inference pass.
    - Fans results back out, writes per-workdir `fluency_ratings.jsonl`, `relevance_ratings.jsonl`, `judge_ratings.jsonl`.

**Phase 3 — Accuracies** (`phase3_accuracies()`):

12. `compute_accuracy_for_workdir()` (`compute_accuracies.py`) — reads the three JSONL rating files, computes pass rates with and without fluency/relevance filtering, writes `*_accuracy_wo_rf.json` and `*_accuracy_w_rf.json` to `judge-evals/accuracy/`.

**Summarize** (`summarize_results.py`):

13. Walks `judge-evals/workdirs/` collecting `judge_ratings.jsonl` files.
14. Aggregates pass rates into a DataFrame, produces per-(model, dataset) heatmaps (axes: N × topk, subplots per steering type).

---

## run.sh

End-to-end pipeline: steering followed by judging in one script.

**Steps:**

1. Parse flags, expand model/dataset lists (same as `run_steering.sh`).
2. Run the steering phase: calls `python run.py` for each model (steps 4–last of `run_steering.sh` above).
3. Run the judging phase: calls `judge-evals/run_judge.py` for each model × dataset (steps 4–12 of `run_judging.sh` above).
4. Call `judge-evals/summarize_results.py` to aggregate and plot results.

The key difference from `run_steering.sh` is that `run.sh` uses `STEERING_N="1 2"` (two N values) while `run_steering.sh` uses `STEERING_N="1"`.

---

## Steering mechanism

Steering vectors are built from the difference between two sets of prompts — `add` (desired direction) and `sub` (undesired direction) — captured at each layer's attention output (`self_attn.o_proj.output`, dimension `H` over all heads). Three `steering_type` modes (`eval/activations.py → steering_reps_cache()`) differ only in how the sequence axis is collapsed before subtracting `add − sub`:

- **`last-token`** — uses only the last real token's activation (left-padding keeps it flush right), averaged over examples. Shape `[layers, H]`.
- **`mean`** — masked mean over each example's non-padding tokens, then averaged over examples. Shape `[layers, H]`.
- **`positional`** — mean over examples only, keeping the sequence axis: one vector per token position. Shape `[layers, P, H]`.

Vectors are cached per type to `steering_cache_<type>.pt`.

At generation time (`eval/generation.py → generate_with_patches()`), `topk_df` selects which `(layer, head)` pairs to steer; each head's slice of the vector is optionally L2-normalized and scaled by `N`. The intervention is written into `o_proj.output` during the prefill pass only — steered values propagate through decoding via the KV cache. It applies to all prompt positions (padding writes are inert under the attention mask):

- **`last-token` / `mean`** — the single per-layer vector is broadcast across every position.
- **`positional`** — aligned by length: if the prompt is ≤ `P` tokens, the rightmost `total_len` rows of the vector are used; if longer, the full vector applies to the last `P` positions and earlier tokens are left unsteered.

`ablation_type == 'steer'` adds the contribution to the activation; `'mean'` replaces it.

---

## Key files

| File | Role |
|------|------|
| `config.py` | Arg parsing; computes `results/<model>/<task>/` output prefix |
| `model_handler.py` | Loads HuggingFace model + tokenizer via `nnsight` |
| `data_handler.py` | Tokenizes desired/undesired JSONL pairs |
| `batch_handler.py` | Slices tokenized data into batches |
| `experiment.py` | Orchestrates patching loops; saves `.pt` logit files |
| `patching.py` | ATP / ACP head-patching algorithms |
| `patching_utils.py` | Low-level activation extraction and head-patching hooks |
| `eval/eval_runner.py` | `run_eval()` — baseline generation + steered generation loops |
| `eval/activations.py` | `steering_reps_cache()` — computes per-head steering vectors |
| `eval/generation.py` | `generate_with_patches()` — hooks heads at inference time |
| `eval/logits_handler.py` | Loads/aggregates `.pt` files; selects top-k heads |
| `judge-evals/run_judge.py` | Three-phase judge pipeline |
| `judge-evals/merge_outputs.py` | Discovers gen files; converts JSON → CSV |
| `judge-evals/build_prompts.py` | Builds fluency / relevance / behavioral judge prompts |
| `judge-evals/evaluator.py` | vLLM wrapper (`make_llm`, `generate_in_batches`) |
| `judge-evals/compute_accuracies.py` | Computes pass rates from rating JSONL files |
| `judge-evals/summarize_results.py` | Aggregates accuracy JSONs into CSV + heatmaps |

---

## Model-specific notes

**Qwen3**: Thinking is suppressed by passing `enable_thinking=False` to all `apply_chat_template()` calls (`data_handler.py`, `gen_data.py`), which bakes an empty `<think>\n\n</think>` block into the generation prompt end. As a side effect, JSONL assistant responses begin with `<think>\n\n</think>` because `gen_data.py` splits on `'\nassistant\n'`, which precedes that block in the decoded output.
