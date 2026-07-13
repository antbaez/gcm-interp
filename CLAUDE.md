# gcm-interp

Mechanistic interpretability project that identifies and steers the components of LLMs responsible for a behavior, to causally shift that behavior at generation time, across three behavioral tasks: harmful refusal, sycophancy, and poetry (verse). Supports two steering targets — individual attention heads (localized to the most implicated heads) and the residual stream (steered uniformly, no localization) — see `Steering mechanism` below.

Supported models: OLMo-2-1124-13B-DPO, Qwen1.5-14B-Chat, Qwen3-14B, gemma-3-12b-it, Llama-3.1-8B-Instruct.
Datasets: `harmful-long`, `sycophancy-long`, `verse-long` (each paired with a "base" counterpart).

---

## General instructions

You don't need to check that the code compiles (e.g. running `py_compile` or similar) after every edit — skip that step by default.

---

## run_normal.sh / run_preemptable.sh

The top-level dispatchers — run these with `bash`, not `sbatch` (they hold no GPU allocation themselves). They expand `--model`/`--dataset` tags (comma-separated or `all`) and `sbatch` one job per model × dataset combo via `run_normal_job.sh` / `run_preemptable_job.sh` respectively. `run_preemptable.sh` also accepts `--seed`, forwarded through to `run.py`. (`run_preemptable_job.sh` submits to the `mit_preemptable` partition with `--requeue`; `run_normal_job.sh` submits to `mit_normal_gpu`.)

Flags accepted by all four scripts (dispatchers and job scripts) and forwarded down the chain: `--model`, `--dataset`, `--type`, `--nocache`, `--unnormalized`, `--resid`, `--patch`, `--judging`.

**Important — patching is off by default.** ATP head localization/patching (`-patch_model` in `run.py`) only runs if `--patch` is passed explicitly. Without `--patch` (and without `--resid`), `run_steering.sh` just runs the eval/generation phase against whatever steering-cache and head-selection CSVs already exist on disk — it will error or no-op on a fresh `results/` dir with no prior patching run. Use `--patch` for a full ATP + attention-head-steering pipeline; use `--resid` for residual-stream steering (which never needs patching — see below); the two are mutually exclusive in effect (`--resid` forces `PATCH=false` in `run_steering.sh` even if `--patch` is also passed).

Each job script (worker, submitted via `sbatch`, one per model × dataset combo):

1. If `--judging` is not set: sources `setup/setup.sh` then calls `bash run_steering.sh --model $MODEL --dataset $DATASET [--patch] [--resid] [...]`.
2. Always: sources `setup/setup_judging.sh` then calls `bash run_judging.sh --model $MODEL --dataset $DATASET [--resid] [--normalized|--unnormalized] [--nocache]`.

---

## run_steering.sh

Runs (optionally) ATP head-patching and steered generation. Calls `run.py` once per model (not once per dataset).

**Steps:**

1. Parse `--model` / `--dataset` / `--device` / `--type` / `--nocache` / `--unnormalized` / `--patch` / `--resid` / `--seed` flags; expand `all` to full lists. `PATCH` defaults to `false`; `--resid` sets `RESID=true` and forces `PATCH=false` regardless of `--patch`.
2. For each model, build five parallel arrays (`SOURCES`, `BASES`, `DIRS`, `ADD_PATHS`, `SUB_PATHS`) — one entry per dataset — by looping over the dataset tags.
3. Call `python run.py` once per model, passing all arrays as space-separated `nargs='+'` args. KV caching is on by default; pass `--nocache` to disable. Steering types are set with `--type` (default runs `last mean positional`; accepts a space-separated subset, e.g. `--type "positional last-token"`). `--resid` adds `--resid` to `run.py`'s flags and forces `TOPK_VALS="1.0"` (residual steering has no notion of top-k heads — see below).

**Inside `run.py → main()`:**

4. `Config()` — parses CLI args. `-source`, `-base`, `-source_dir`, `-steering_add_path`, `-steering_sub_path` are all `nargs='+'` lists. `-steering_types` is a `nargs='+'` list (default `['last-token']`). `--resid` (`action='store_true'`) selects residual-stream steering. Output prefix is not computed at init; it is set per-dataset below.
5. `ModelHandler(config)` — loads the HuggingFace model and tokenizer onto GPU **once**, outside the dataset loop.
6. For each dataset (zipped from the five arrays):
   - Overwrite scalar fields on `config.args` (`source`, `base`, `source_dir`, `steering_add_path`, `steering_sub_path`).
   - Call `config.set_output_prefix()` → `results/<model>/from_<source>_to_<base>/`. Create the directory and save `config.yml`.
   - `DataHandler(config, model_handler)` — tokenizes paired desired/undesired JSONL files into `base_toks`/`source_qs_toks` (used for ATP patching) and `steering_qs_toks['add'/'sub']` (used for the steering-vector cache).
   - **Patching phase** (`-patch_model` flag, i.e. `--patch` was passed and `--resid` was not):
     - `Experiment(config, data_handler, model_handler, 'heads').run()`
     - Iterates batches via `BatchHandler`; each batch calls `Patching.apply_patching()`.
     - `apply_patching()` runs ATP (Attribution Patching): forward passes on desired and undesired inputs, computes gradient-weighted head-effect scores (`attn_desired_effects`, `attn_undesired_effects`), stacks them into a logit tensor, saves per-batch `.pt` files to `results/<model>/<task>/heads/heads_<idx>.pt`.
   - **Evaluation phase** (`-eval_model --steering` flags):
     - `load_patching_reps()` (`eval/eval_runner.py`) — calls `steering_reps_cache()` (`eval/activations.py`) to compute the mean steering vector (desired minus base activations) for each layer (and, in non-resid mode, each head). Result is cached to `results/<model>/from_<source>_to_<base>/<model>_steering_cache_<source>[_resid].pt` (the `_resid` suffix keeps the attention-head cache and residual-stream cache from colliding) and loaded from there on subsequent runs.
     - For each `steering_type` in `config.args.steering_types` (outer loop in `run.py`):
       - Set `config.args.steering_type` and call `run_eval()` (`eval/eval_runner.py`).
       - **Inside `run_eval()`:**
         - If not resid mode: `load_logits()` (`eval/logits_handler.py`) — aggregates all `.pt` batch files into a single logits tensor. In resid mode there are no head logits to load; `logits = None`.
         - Generate unsteered baseline responses for all test prompts; stored in memory as `original_outputs`.
         - For each `topk` × each `N`:
           - Non-resid: `get_top_k_layer_and_head()` — selects top-k% heads by ATP score; saves CSV to `results/<model>/<task>/numerator_1_<reps_type>_<topk>.csv`. Resid mode: `topk_df` is synthesized directly as one row per layer (`{'layer': range(num_layers)}`) — every layer is steered, there is no head selection.
           - `generate_with_patches()` (`eval/generation.py`) — runs steered generation: hooks the selected heads (or, in resid mode, whole layers) at forward-pass time and adds the steering vector scaled by `N`.
           - `decode_responses()` — pairs each steered output with the baseline output.
           - `save_prompt_responses()` — writes `.txt` and `.json` to `results/<model>/from_<source>_to_<base>/<norm_dir>/<stream_dir>/<cache_dir>/<steering_type>/N=<N>_<ablation>_topk=<topk>_<test_dataset>_gen.(txt|json)`, where `norm_dir` is `normalized`/`unnormalized`, `stream_dir` is `attention`/`residuals`, and `cache_dir` is `cache`/`no_cache`. (An older filename stem `<N>_<reps_type>_<ablation>_<topk>_<test_dataset>_gen` is still recognized when checking for already-completed runs.)

---

## run_judging.sh

Scores already-generated outputs using a vLLM judge model. Calls `judge-evals/run_judge.py` then `judge-evals/summarize_results.py`.

**Steps:**

1. Parse `--model` / `--dataset` / `--device` / `--normalized`|`--unnormalized` / `--cache`|`--nocache` / `--resid` flags. `--resid` sets `STREAM_MODE=residuals` and switches the accuracy output directory to `judge-evals/accuracy_residual` (default/attention mode writes to `judge-evals/accuracy`), keeping resid and attention-head judging results from colliding.
2. For each model × dataset combination, run `python judge-evals/run_judge.py` with paths into `results/`, `data/`, `judge-evals/<accuracy_subdir>`, and `judge-evals/workdirs`, passing through `--norm_mode`, `--cache_mode`, and `--stream_mode`.
3. After all combinations, run `python judge-evals/summarize_results.py --stream_mode $STREAM_MODE` against the same accuracy/workdirs directories.

**Inside `run_judge.py → main()`:**

**Phase 1 — Prepare** (`phase1_prepare()`):

4. `discover_gen_files()` (`merge_outputs.py`) — glob `results/` for `*_gen.json` files matching the model/source/base filters.
5. `extract_path_metadata()` — parses path components (`results/<model>/from_<source>_to_<base>/<norm_mode>/<stream_mode>/<cache_mode>/<steering_type>/<filename>`) into a metadata dict (model, source, base, norm_mode, stream_mode, cache_mode, steering_type, topk, N).
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

12. `compute_accuracy_for_workdir()` (`compute_accuracies.py`) — reads the three JSONL rating files, computes pass rates with and without fluency/relevance filtering, writes `*_accuracy_wo_rf.json` and `*_accuracy_w_rf.json` to `judge-evals/<accuracy_subdir>/` (`accuracy/` for attention-head runs, `accuracy_residual/` for `--resid` runs).

**Summarize** (`summarize_results.py`):

13. Walks `judge-evals/workdirs/` collecting `judge_ratings.jsonl` files, filtered to the given `--stream_mode` (`attention` or `residuals`) if passed.
14. Aggregates pass rates into a DataFrame, produces per-(model, dataset) heatmaps (axes: N × topk, subplots per steering type).

---

## Steering mechanism

Steering vectors are built from the difference between two sets of prompts — `add` (desired direction) and `sub` (undesired direction). Where they're captured depends on `--resid`:

- **Attention-head mode (default)** — captured at each layer's attention output (`self_attn.o_proj.output`, dimension `H` over all heads).
- **Residual-stream mode (`--resid`)** — captured at each layer's full output (`layer.output`, dimension `H` = model hidden size) instead of the attention output. Enabled via `Config`'s `--resid` flag, which `eval/activations.py → steering_reps_cache()` and `eval/generation.py → generate_with_patches()` both key off of (`resid=True`).

Three `steering_type` modes (`eval/activations.py → steering_reps_cache()`) differ only in how the sequence axis is collapsed before subtracting `add − sub`; the same three modes apply in both attention and residual-stream mode, just over a differently-shaped per-layer vector:

- **`last-token`** — uses only the last real token's activation (left-padding keeps it flush right), averaged over examples. Shape `[layers, H]`.
- **`mean`** — masked mean over each example's non-padding tokens, then averaged over examples. Shape `[layers, H]`.
- **`positional`** — mean over examples only, keeping the sequence axis: one vector per token position. Shape `[layers, P, H]`.

Vectors are cached to `results/<model>/from_<source>_to_<base>/<model>_steering_cache_<source>[_resid].pt` (one file per source dataset, shared across both `desired` and `undesired` keys since the computation is identical; the `_resid` suffix is appended only in residual-stream mode so the two caches never collide). On the first run the file is computed and saved; on subsequent runs it is loaded directly.

At generation time (`eval/generation.py → generate_with_patches()`), `topk_df` selects which `(layer, head)` pairs to steer in attention mode; each head's slice of the vector is optionally L2-normalized and scaled by `N`. In residual-stream mode there is no head slicing — `topk_df` is just the full list of layers, and the whole per-layer vector is L2-normalized and scaled by `N`. The intervention is written into `o_proj.output` (attention mode) or `layer.output` (resid mode) during the prefill pass only — steered values propagate through decoding via the KV cache (default). It applies to all prompt positions (padding writes are inert under the attention mask):

- **`last-token` / `mean`** — the single per-layer vector is broadcast across every position.
- **`positional`** — aligned by length: if the prompt is ≤ `P` tokens, the rightmost `total_len` rows of the vector are used; if longer, the full vector applies to the last `P` positions and earlier tokens are left unsteered.

`ablation_type == 'steer'` adds the contribution to the activation; `'mean'` replaces it. With `--nocache`, `generate_with_patches()` uses `model.all()` to reapply the intervention at every decoding step instead of prefill-only (both attention and resid mode).

---

## Key files

| File | Role |
|------|------|
| `config.py` | Arg parsing; `set_output_prefix()` computes `results/<model>/<task>/` and is called per-dataset in `run.py` |
| `model_handler.py` | Loads HuggingFace model + tokenizer via `nnsight` |
| `data_handler.py` | Tokenizes desired/undesired JSONL pairs |
| `batch_handler.py` | Slices tokenized data into batches |
| `experiment.py` | Orchestrates patching loops; saves `.pt` logit files |
| `patching.py` | ATP / ACP head-patching algorithms |
| `patching_utils.py` | Low-level activation extraction and head-patching hooks |
| `eval/eval_runner.py` | `run_eval()` — baseline generation + steered generation loops |
| `eval/activations.py` | `steering_reps_cache()` — computes and caches per-head (or, with `--resid`, per-layer residual-stream) steering vectors |
| `eval/generation.py` | `generate_with_patches()` — hooks heads (or, with `--resid`, whole layers) at inference time |
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

**gemma-3-12b-it**: Its decoder layer's `forward()` returns a tuple (not a bare tensor like the other models), so in residual-stream steering mode (`--resid`) `eval/activations.py` and `eval/generation.py` both special-case `'gemma' in model.config._name_or_path.lower()` to read/write `layer.output[0]` and wrap assignments back into a 1-tuple, instead of treating `layer.output` as the tensor directly.
