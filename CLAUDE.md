# gcm-interp

Mechanistic interpretability project that identifies and steers the LLM components responsible for a behavior, to causally shift it at generation time, across three tasks: harmful refusal, sycophancy, and poetry (verse). Two steering targets — individual attention heads (localized to top-implicated heads) or the residual stream (steered uniformly) — see `Steering mechanism`. Layers are steered as a single-layer sweep (default) or all at once via `--global`.

Supported models: OLMo-2-1124-13B-DPO, Qwen1.5-14B-Chat, Qwen3-14B, gemma-3-12b-it, gemma-4-12B-it, Llama-3.1-8B-Instruct.
Datasets: `harmful-long`, `sycophancy-long`, `verse-long` (each paired with a "base" counterpart).

---

## General instructions

Skip checking that code compiles (`py_compile` etc.) after every edit.

---

## run_normal.sh / run_preemptable.sh

Top-level dispatchers — run with `bash`, not `sbatch` (no GPU allocation themselves). Expand `--model`/`--dataset` tags (comma-separated, or `all` — `olmo`, `qwen`, `qwen3`, `gemma`, `gemma4`, `llama` × `harmful`, `sycophancy`, `verse`) and `sbatch` one job per combo via `scripts/run_normal_job.sh` / `scripts/run_preemptable_job.sh`. `run_preemptable.sh` also takes `--seed`. (`run_preemptable_job.sh` submits to `mit_preemptable` with `--requeue`; `run_normal_job.sh` to `mit_normal_gpu`.)

Flags forwarded down the chain: `--model`, `--dataset`, `--type`, `--nocache`, `--unnormalized`, `--attention`, `--global`, `--split val|test`, `--judging`, `--judge <openai-model>` (`--seed` too, for preemptable). Residual-stream steering is the default — there's no `--resid` flag, only `--attention` to disable it. `--global` steers every layer at once. `--split` picks which phase(s) run: omitted runs the full pipeline; `val` stops after judging the sweep; `test` (re)selects from existing sweep results and only regenerates the held-out split if the winner moved. `--judge <model>` (e.g. `gpt-5-mini`) judges the held-out split via that OpenAI API model instead of the local one (needs `OPENAI_API_KEY`); never touches validation-sweep judging, and writes to a separate `judge=<model>` subtree.

**Attention-head steering (without `--global`) needs pre-existing localization artifacts** — ATP head localization was removed from the codebase, so there's no way to compute them fresh. `--attention` just runs eval/generation against whatever `.pt`/CSV files already exist under `results/<model>/<task>/heads/`; it errors or no-ops on a fresh `results/`. Residual-stream and `--global` steering need no such artifacts.

Each job script `cd`s to the repo root, then runs up to two phases depending on `--split`:

1. **Validation sweep** (`--split` omitted or `val`): unless `--judging`, sources `setup/setup.sh` and runs `run_steering.sh` to sweep N × layer on `<base>-test.jsonl`; then always sources `setup/setup_judging.sh` and runs `run_judging.sh` to judge it.
2. **Selection + held-out split** (`--split` omitted or `test`): runs `select_best_config.py --pending-out <tmpfile>` to pick each method's best validation config and prune stale held-out runs. If anything's pending, regenerates + rejudges just the held-out split (`run_steering.sh --split test`, `run_judging.sh --split test`); otherwise skips. Finally runs `stats/collect_pass_rates.py --split test` for this combo. `run_mcnemar_test.py` is deliberately not run here — run it by hand once every combo is done.

---

## scripts/run_steering.sh

Runs steered generation. Calls `run.py` once per model (not per dataset).

**Steps:**

1. Parse `--model`/`--dataset`/`--device`/`--type`/`--nocache`/`--unnormalized`/`--attention`/`--global`/`--seed`/`--split val|test`/`--best-configs <path>`; expand `all`. `RESID` defaults `true`; `--attention` switches to attention-head steering. `--global` steers every layer instead of the default single-layer sweep over `LAYER_RANGE_START`..`LAYER_RANGE_END` (currently 0.333..0.6667). `--split val` (default) sweeps on `<base>-test.jsonl`; `--split test` needs `best_configs.json` and generates once on `<base>-heldout-test.jsonl`.
2. Per model, build five parallel arrays (`SOURCES`, `BASES`, `DIRS`, `ADD_PATHS`, `SUB_PATHS`), one entry per dataset. Steering-`N` sweep values come from `STEERING_N_LOCAL_BY_MODEL` or `STEERING_N_GLOBAL_BY_MODEL` (`--global`) — separate lists, since whole-model steering needs different magnitudes.
3. Call `python run.py` once per model with all arrays. KV caching on by default (`--nocache` disables). `--type` sets steering types (default `last mean positional`). `--resid` is added automatically unless `--attention`, and forces `TOPK_VALS="1.0"`. `--split test` passes `-split test -best_configs $BEST_CONFIGS`, which overrides the swept `-steering_n`.

**Inside `run.py → main()`:**

4. `Config()` parses CLI args (`-source`/`-base`/`-source_dir`/`-steering_add_path`/`-steering_sub_path` are `nargs='+'` lists; `-steering_types` default `['last-token']`). `--resid` selects residual-stream steering; `--global` (`dest='global_steer'`) steers every layer; `-split test` requires `-best_configs`, loaded once up front.
5. `ModelHandler(config)` loads the model/tokenizer onto GPU once, outside the dataset loop.
6. Per dataset:
   - Overwrite `config.args` scalars (`source`, `base`, `source_dir`, `steering_add_path`, `steering_sub_path`).
   - Pick the split: `-split test` points at `<base>-heldout-test.jsonl` (erroring if missing); otherwise `<base>-test`.
   - `config.set_output_prefix()` → `results/<model>/from_<source>_to_<base>/`; save `config.yml`.
   - `DataHandler(config, model_handler)` tokenizes into `base_qs_toks['test']` and `steering_qs_toks['add'/'sub']`.
   - **Evaluation** (`-eval_model --steering`):
     - `generate_baseline()` runs unsteered generation once per dataset; the result is reused across every `steering_type`.
     - `load_patching_reps()` → `steering_reps_cache()` computes the mean steering vector (add − sub) per layer (and, non-resid, per head), cached to `<model>_steering_cache_<source>[_resid].pt`.
     - Per `steering_type`: on `-split test`, pin `steering_n`/`layers` from `best_configs[model][task][steering_type]` (skip if absent — the only place `best_configs.json` is read). Call `run_eval()`.
       - **Inside `run_eval()`**, three modes decide the sweep axis and `topk_df`:
         - **`head_selection`** (attention, no `--global`) — loads head-effect scores from `results/<model>/<task>/heads/numerator_1_heads.pt` (a prior ATP artifact); sweeps `topk`, selecting top-k% heads per value via `get_top_k_layer_and_head()`.
         - **`layer_sweep`** (resid, no `--global`, default) — sweeps `layer` over `round(num_layers*layer_range_start)..round(num_layers*layer_range_end)` (or a pinned layer on `-split test`); `topk_df` is one row.
         - **`global_steer`** (`--global`) — sweep axis `layer`, single value `'all'`; `topk_df` covers every layer.
       - `generate_with_patches()` hooks the selected heads/layers and adds the steering vector × `N`. `decode_responses()` pairs steered/baseline output. `save_prompt_responses()` writes to `results/<model>/from_<source>_to_<base>/<norm_dir>/<stream_dir>/<cache_dir>/<scope_dir>/<steering_type>/N=<N>_<ablation>_<slot>_<test_dataset>_gen.(txt|json)` — `scope_dir` is `global`/`local`, `slot` is `layer=<idx>`/`layer=all` or `topk=<topk>`. (Older stems without `scope_dir`, or the pre-layer-sweep `<N>_<reps_type>_<ablation>_<topk>_<test_dataset>_gen` naming, are still recognized as already-completed.)

---

## scripts/run_judging.sh

Scores generated outputs with the local vLLM judge or an OpenAI API judge. Calls `judge-evals/run_judge.py` then `summarize_results.py`. Resolves `judge-evals/`/`results/`/`data/` relative to its own location, so it runs from anywhere.

**Steps:**

1. Parse `--model`/`--dataset`/`--device`/`--normalized`|`--unnormalized`/`--cache`|`--nocache`/`--attention`/`--split val|test`/`--force`/`--judge <openai-model>`. Residual judging is default; `--attention` → `STREAM_MODE=attention` (else `residuals`); both write to `judge-evals/accuracy`. `--split` picks `<base>-test` (val, default) or `<base>-heldout-test` (test), passed as `--test_file`. `--force` re-judges existing conditions. `--judge <model>` (default `API_CONCURRENCY=16`) uses the OpenAI API instead of the local judge (needs `OPENAI_API_KEY`), writing to a separate `judge=<model>` subtree.
2. Per model × dataset, run `judge-evals/run_judge.py` with paths into `results/`, `data/`, `judge-evals/accuracy`, `judge-evals/workdirs`, passing `--norm_mode`/`--cache_mode`/`--stream_mode`/`--test_file` and any `--force`/`--judge_model`/`--api_concurrency`.
3. After all combos, run `summarize_results.py --stream_mode $STREAM_MODE`.

**Inside `run_judge.py → main()`:**

**Phase 1 — Prepare** (`phase1_prepare()`):

4. `discover_gen_files()` globs `results/` for `*_gen.json` by model/source/base/`test_file`/`scope`.
5. `extract_path_metadata()` parses `results/<model>/from_<source>_to_<base>/<norm_mode>/<stream_mode>/<cache_mode>/<scope>/<steering_type>/<filename>` into metadata (incl. `scope`: `local`/`global`). Filenames follow the layer-sweep convention (`N=<N>_steer_layer=<L>_<test_file>_gen.json`, `layer=all` for `--global`) or the older topk-sweep one.
6. Skip already-judged files (`accuracy_exists()`) unless `--force`; with `--judge`, workdir/accuracy paths get a `__judge_<model>`/`judge=<model>` segment. `gen_to_csv()` converts each `_gen.json` to `eval_output.csv` under `judge-evals/workdirs/`.
7. Load tokenizer once (`build_prompts.py`).
8. For long-eval files, build `relevance_fluency_prompts.csv` and `judge_prompts.csv`.

**Phase 2 — Evaluate** (`phase2_evaluate()`):

9. Single-eval files: `compute_accuracy_for_file()` — token matching, no GPU.
10. Long-eval: default loads the local vLLM judge (`make_llm()`); `--judge_model` instead uses `api_evaluator.make_api_generate_fn()` — no GPU load.
11. `_evaluate_all_workdirs_batched()` batches prompts across all workdirs through a backend-agnostic `generate_fn`, writing per-workdir `fluency_ratings.jsonl`/`relevance_ratings.jsonl`/`judge_ratings.jsonl`.

**Phase 3 — Accuracies** (`phase3_accuracies()`):

12. `compute_accuracy_for_workdir()` reads the rating JSONLs, writes `*_accuracy_wo_rf.json`/`*_accuracy_w_rf.json` to `judge-evals/accuracy/` (held-out runs get an extra `test_file` segment, `--judge` an extra `judge=<model>` segment). `_delete_intermediate_csvs()` then clears the workdir's scratch CSVs — phase 1 regenerates them on a future `--force` rerun.

**Summarize** (`summarize_results.py`):

13. Walks `judge-evals/workdirs/` for `judge_ratings.jsonl`, filtered to `--stream_mode` if passed.
14. Aggregates pass rates into `judge-evals/accuracy/results_summary.csv`. No longer makes heatmaps itself — that's the standalone `judge-evals/create_heatmaps.py`.

---

## Steering mechanism

Steering vectors come from the difference between `add` (desired) and `sub` (undesired) prompt sets. Capture point depends on `--resid`:

- **Attention-head mode** (`Config`'s default if `--resid` is omitted, though wrapper scripts always pass `--resid` unless `--attention`) — `self_attn.o_proj.output`.
- **Residual-stream mode (`--resid`, the effective default)** — `layer.output` (full hidden size), via `resid=True` in `eval/activations.py`/`eval/generation.py`.

Three `steering_type` modes (`steering_reps_cache()`) differ only in how the sequence axis collapses before `add − sub`:

- **`last-token`** — last real token's activation, averaged over examples. `[layers, H]`.
- **`mean`** — masked mean over non-padding tokens, averaged over examples. `[layers, H]`.
- **`positional`** — mean over examples only, sequence axis kept. `[layers, P, H]`.

Cached as a single stacked tensor to `<model>_steering_cache_<source>[_resid].pt` (`_resid` suffix keeps the two caches from colliding) — no longer a `{'desired', 'undesired'}` dict; loaded directly on later runs.

At generation time, `topk_df`'s `layer` column picks which layers get steered — one swept layer (`layer_sweep`), every layer (`--global`), or every layer with a top-k head (`head_selection`; see above). There's no sub-layer head slicing — selected layers get their whole `o_proj.output`/`layer.output` steered uniformly. The vector is optionally L2-normalized, scaled by `N`, and written during prefill only (propagates through decoding via the KV cache by default), across all prompt positions:

- **`last-token`/`mean`** — broadcast across every position.
- **`positional`** — the cached `[P, H]` vector is added directly to `[batch, seq_len, H]`; there's no length cropping/alignment despite `P` being fixed at cache time, so a mismatched batch length raises `RuntimeError: ... tensor a (X) must match tensor b (Y)`. Known gap, not documented behavior — regenerating the cache only helps if the new `P` happens to match.

`ablation_type == 'steer'` adds; `'mean'` replaces. `--nocache` uses `model.all()` to reapply every decoding step instead of prefill-only.

---

## Statistical analysis (`stats/`, `judge-evals/select_best_config.py`, `judge-evals/selection_utils.py`)

Validation-selection + significance-testing pipeline, run after `run_judging.sh` has produced ratings. Restricted to `local` scope — `global` is never considered.

1. **`select_best_config.py`** — per model/dataset/method, picks the argmax w_rf pass-rate condition from the validation sweep (ties → smallest `N`, then layer closest to median). Writes `best_configs.json`, which `run_steering.sh --split test` pins from. By default prunes held-out runs left at a superseded config (`results/`, `workdirs/`, `accuracy/`); `--no-prune` disables, `--dry-run` previews. `--pending-out <path>` writes `<model_tag> <dataset_tag>` per combo whose selected config has no current held-out run (checked after pruning) — the file's always created, so callers test emptiness; used by the job scripts to skip unnecessary held-out regeneration. Writes are `flock`-protected and atomic.
2. **`selection_utils.py`** — shared by both scripts: parses condition dirs (`N=<N>_<ablation>_layer=<layer>_<test_file>`), reads the rating JSONLs, applies the pass rule (empty response → fail; pass = `judge_rating==5`, `w_rf` also needs `fluency==2`/`relevance==2`). Mirrors `compute_accuracies.py`'s `_compute_and_write`.
3. **`stats/collect_pass_rates.py`** — per model/dataset/norm/stream/cache combo, builds a per-prompt pass/fail CSV (one column per method) on `test` (default, unbiased, held-out) or `val` (selection-biased sweep argmax). Writes to `stats/pass_results/`. `--model`/`--dataset` accept short tags or full dir names. `--force` (default) regenerates existing CSVs; `--no-force` skips them.
4. **`stats/run_mcnemar_test.py`** — paired McNemar's test (`mean`/`positional` vs `last`) per model/dataset: one-sided exact binomial p-value (H1: mean/positional beats last) plus a continuity-corrected one-sided z-test. Defaults `--split test`, `--model olmo,qwen3,llama,gemma4` (`--model all` adds `qwen`). `--simulate-n` rescales counts to hypothetical sample sizes; `--output-dir` writes CSVs. Prints a per-dataset table (`last (rate) vs. mean/positional (rate)`, padded for alignment) plus, at the bottom, how many combos beat `last` and how often `mean`/`positional` have the single highest pass rate of the three.

Both `stats/*.py` scripts resolve paths relative to their own location, so they run from anywhere.

---

## Key files

| File | Role |
|------|------|
| `config.py` | Arg parsing; `set_output_prefix()` computes `results/<model>/<task>/` and is called per-dataset in `run.py` |
| `model_handler.py` | Loads HuggingFace model + tokenizer via `nnsight` |
| `data_handler.py` | Tokenizes the desired-only JSONL data: base/source prompts, eval test prompts, and the `add`/`sub` steering pairs |
| `batch_handler.py` | Slices tokenized data into batches |
| `eval/eval_runner.py` | `run_eval()` — baseline generation + steered generation loops |
| `eval/activations.py` | `steering_reps_cache()` — computes and caches per-head (or, with `--resid`, per-layer residual-stream) steering vectors |
| `eval/generation.py` | `generate_with_patches()` — hooks heads (or, with `--resid`, whole layers) at inference time |
| `eval/logits_handler.py` | Loads/aggregates `.pt` files; selects top-k heads |
| `judge-evals/run_judge.py` | Three-phase judge pipeline |
| `judge-evals/merge_outputs.py` | Discovers gen files; converts JSON → CSV |
| `judge-evals/build_prompts.py` | Builds fluency / relevance / behavioral judge prompts |
| `judge-evals/evaluator.py` | vLLM wrapper (`make_llm`, `generate_in_batches`) |
| `judge-evals/api_evaluator.py` | OpenAI API judge (`make_api_generate_fn`) — drop-in alternative to `evaluator.py`, used with `--judge`/`--judge_model` |
| `judge-evals/compute_accuracies.py` | Computes pass rates from rating JSONL files |
| `judge-evals/summarize_results.py` | Aggregates accuracy JSONs into `judge-evals/accuracy/results_summary.csv` |
| `judge-evals/create_heatmaps.py` | Standalone, manually-run: reads `results_summary.csv`, produces N × steering-type and layer-axis heatmaps |
| `judge-evals/select_best_config.py` | Picks best (N, layer) per model/dataset/method on the validation split; writes `best_configs.json`; prunes superseded held-out runs |
| `judge-evals/selection_utils.py` | Shared condition-scanning + pass-rule logic for `select_best_config.py` and `stats/collect_pass_rates.py` |
| `stats/collect_pass_rates.py` | Per-prompt pass/fail CSV across steering methods, per model/dataset combo (val or held-out test split) |
| `stats/run_mcnemar_test.py` | Paired McNemar's significance test between steering methods on `collect_pass_rates.py` output |

---

## Model-specific notes

**Qwen3**: `enable_thinking=False` on all `apply_chat_template()` calls (`data_handler.py`, `gen_data.py`) bakes an empty `<think>\n\n</think>` block into the generation prompt. As a side effect, JSONL assistant responses begin with `<think>\n\n</think>`, since `gen_data.py` splits on `'\nassistant\n'`, which precedes that block.

**gemma-3-12b-it**: Its decoder layer's `forward()` returns a tuple, not a bare tensor — so in residual-stream mode `eval/activations.py`/`eval/generation.py` special-case `'gemma' in model.config._name_or_path.lower()` to read/write `layer.output[0]` and rewrap into a 1-tuple.

**gemma-4-12B-it**: Multimodal (`AutoModelForImageTextToText`, `model_type == 'gemma4_unified'`) — `model_handler.py` loads it via `nnsight.VisionLanguageModel` (`ModelHandler.is_gemma4`), and `_get_layers()` reaches through `model.model.language_model.layers`. Its decoder layer's `forward()` returns a bare tensor (unlike gemma-3's tuple), so the gemma-3 special-case also checks `model.config.model_type != 'gemma4_unified'` to avoid misfiring (a plain `'gemma' in _name_or_path` match would otherwise catch both).
