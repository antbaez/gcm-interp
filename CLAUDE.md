# gcm-interp

Mechanistic interpretability project that identifies and steers the LLM components responsible for a behavior, to causally shift it at generation time, across three tasks: harmful refusal, sycophancy, and poetry (verse). Steering is added to the residual stream (`layer.output`); see `Steering mechanism`. It doesn't localize heads: whole layers are steered uniformly, as a single-layer sweep (default) or all at once via `--global`.

Supported models: OLMo-2-1124-13B-DPO, Qwen1.5-14B-Chat, Qwen3-14B, gemma-3-12b-it, gemma-4-12B-it, Llama-3.1-8B-Instruct.
Datasets: `harmful-long`, `sycophancy-long`, `verse-long` (each paired with a "base" counterpart). The steering vector is source − base, added to base prompts, so task dirs are named in steering direction, `from_<base>_to_<source>`: `from_harmless_to_harmful-long`, `from_sycophancy_to_non-sycophantic-long`, `from_prose_to_verse-long`.
Dataset variants (own data dirs under `data/<model>/`, so own steering vectors): `sycophancy-{more-templates,unaligned,diff-length,unaligned-diff-length}` (dir `sycophancy-long-<v>`, source `non-sycophantic-long-<v>`, base `sycophancy-<v>`) and `verse-{aligned,longer,varied}` (dir/source `verse-long-<v>`, base `prose-<v>`). `run_steering.sh`/`run_judging.sh` accept these tags via `--dataset` (`VARIANT_DATASETS`), but `all` still means only the three base datasets, and the sbatch/selection/stats pipeline doesn't cover them — they're run via `run_harmonic_tests.py --datasets`.

---

## General instructions

Skip checking that code compiles (`py_compile` etc.) after every edit.

---

## run_normal.sh / run_preemptable.sh

Top-level dispatchers — run with `bash`, not `sbatch` (no GPU allocation themselves). Their only flag is the required `--settings <yaml>` (e.g. `settings/default.yaml`; they exit with a usage message without it): a settings file names a `results_dir` (relative to the repo root) and explicitly lists `models` × `datasets` and every run setting (`steering_type`, `steering_n` (validation N sweep: `null` keeps `run_steering.sh`'s per-model tables, a list applies to every model, a `{model_tag: list|null}` mapping sets it per model; emitted as a `STEERING_N` assoc array and passed per model as `--steering-n "<N...>"` through the job scripts to `run_steering.sh`, where it replaces the table; it applies to the local scope), `global_steering_n` (the same for the global scope, replacing the `STEERING_N_GLOBAL_*` tables), `normalization`, `scope` local|global|[local, global], `local_layer` {`mode` single|sweep, `single` frac, `sweep` [start, end]}, `split`, `generation`, `judging`, `no_model_cache`, `metric` harmonic|pass_rate, `train_batch_size` (→ `--train-batch-size` → `run.py -batch_size`, kept as `args.train_batch_size`: the steering-vector activation pass), `generation_batch_size` (→ `--gen-batch-size` → `-eval_batch_size`: the validation sweep and the held-out split), plus preemptable-only `seed`/`hook_prompt_acts`/`hook_response_acts`/`hook_target_acts`/`hook_batch_size` — the hooks' own batch size, `--hook-batch-size` → `run.py -hook_batch_size`, required when any hook is on). `scripts/load_settings.py` validates it (every key required, unknown keys rejected) and emits the `MODELS`/`DATASETS`/`JOB_FLAGS` bash arrays, plus `SCOPES` with per-scope `SCOPE_FLAGS_<scope>` (`--global`, or the `local_layer` flags) and `STEERING_N_<scope>` arrays, and the dispatcher `sbatch`es one job per model × dataset × scope via `scripts/run_normal_job.sh` / `scripts/run_preemptable_job.sh` with those flags; with both scopes the local and global jobs run in parallel (shared files such as the steering-vector cache and the unsteered hook files may be written by both). `local_layer` becomes `--layer-frac` or `--layer-range` (both forwarded to `run_steering.sh`); `generation: false` → `--no-generation` (judge what's on disk), `judging: false` → `--no-judging` (also skips the held-out scores). The dispatcher copies the settings file to `<results_dir>/settings.yaml` (a differing earlier copy is kept as `settings.<mtime>.yaml`) and passes `--results-dir` down the chain, so a run is self-contained: generations at `<results_dir>/<model>/from_<base>_to_<source>/...` (`run.py -results_dir`; steering caches live there too, so each results dir builds its own), ratings/accuracies/`best_configs*.json` under `<results_dir>/judge-evals/`, held-out scores under `<results_dir>/stats/`. `run_judging.sh`/`run_steering.sh`/`select_best_config.py` take `--results-dir` and fall back to the repo-level `results/` + `judge-evals/` without it; `merge_outputs.extract_path_metadata()` parses gen paths from the end, so the results root can be any directory. (`run_preemptable_job.sh` submits to `mit_preemptable` with `--requeue`; `run_normal_job.sh` to `mit_normal_gpu`.)

Job-script flags: `--model`, `--dataset`, `--type`, `--unnormalized`, `--global`, `--layer-frac`/`--layer-range`, `--split val|test`, `--no-generation`, `--no-judging`, `--judge <openai-model>` (`--seed`, `--hook_prompt_acts`, `--hook_response_acts` and `--hook_target_acts` too, for preemptable). `--hook_prompt_acts` (through `run_steering.sh` → `run.py --hook_prompt_acts` → `eval/activations.save_prompt_activations()`) saves layer round(2/3 × num_layers)'s residual-stream activations at every non-padding token of the first 100 `verse-long`/`prose` training (`-desired-all`) prompts to `activations/<model>/from_prose_to_verse-long/pt/{verse-long,prose}_layer=<L>.pt` (dict of per-prompt bf16 `[T_i, H]` `acts`, `layer`, `input_ids`, `content_start`, `content_end`), right after `DataHandler` is built. `--hook_response_acts` (→ `save_response_activations()`) greedily generates an unsteered response (up to `max_new_tokens`) to each of those same 200 prompts, reruns prompt + response, and saves the same layer's activations at the response tokens only (up to, not including, the first stop token) to `{verse-long,prose}_response_layer=<L>.pt` in the same dir (per-response bf16 `[R_i, H]` `acts`, response `input_ids`, `prompt_ids`, decoded `text`, `finished`, `max_new_tokens`, `layer`). For both, other datasets are skipped, existing files are kept, and the steering pipeline then runs as usual. Both hooks also cover every prompt of the eval split the run generates on (`base_qs_toks['test']`, via `_hook_sets(split='eval')` / `DataHandler.test_token_positions()`): the validation split on sweep runs, the held-out split on `-split test` runs, saved in the same dir as `<test_dataset>_layer=<L>.pt` / `<test_dataset>_response_layer=<L>.pt` (`prose-valid_...`, `prose-test_...`, same keys); `run_preemptable_job.sh` runs the `--split test` pass even when the held-out generations are already current whenever a hook (any of the three) is on, so those files get written. With `hook_response_acts` on, `run_eval()` also saves the activations under steering for every (method, N, layer) condition it runs on the eval split (every validation N, and the pinned held-out config): `eval/activations.save_steered_activations()` reruns each padded prompt + steered response through `eval/generation.trace_with_patches()` with the same prefill steering and steer mask as `generate_with_patches()`, capturing the hook layer after any steering added there, to `<hook dir>/steered/<steering_type>/<test_dataset>_N=<N>_steer-layer=<L|all>_layer=<hook L>.pt` (per-prompt bf16 `prompt_acts` [T_i, H] and response `acts` [R_i, H] up to the first stop token, `prompt_ids`, response `input_ids`, `text`, `finished`, `content_start`/`content_end`, `N`, `steering_type`, `steer_layers`, `layer`). Conditions whose gen files already exist are regenerated just for the hook (gen files untouched); existing activation files are skipped. `--hook_target_acts` (→ `run.py --hook_target_acts`) does the same unsteered prompt + response hooking (`save_prompt_activations`/`save_response_activations` with `split='target'`) on the eval split's target-concept counterpart, `DataHandler.target_split()`: the same prompts with the base's instruction swapped for the source's (`TARGET_BASES` in `utils/data_handler.py`, `prose`→`verse`), read from `data/<model>/verse-long/verse-valid.jsonl` / `verse-test.jsonl` and saved as `verse-valid_[response_]layer=<L>.pt` / `verse-test_[response_]layer=<L>.pt` — the model's natural verse answer to each validation/held-out question, a per-prompt target for the steered responses. `--global` steers every layer at once. Selection and the final report use the harmonic score by default (mean over prompts of the per-prompt harmonic mean of concept 1-3→0/4→1/5→2, fluency and relevance, all 0-2): `select_best_config.py --metric harmonic` → `judge-evals/best_configs_harmonic.json`, which `run_steering.sh --split test --best-configs` pins from, then `calculate_harmonic_scores.py --best-configs` writes `stats/harmonic_scores/<model>_<dataset>_<scope>_test.csv`. `--pass-rate` switches back to the w_rf pass rate (`best_configs.json`, `collect_pass_rates.py`). Pruning in either mode spares held-out runs at the other mode's selection. `--split` picks which phase(s) run: omitted runs the full pipeline; `val` stops after judging the sweep; `test` (re)selects from existing sweep results and only regenerates the held-out split if the winner moved. `--judge <model>` (e.g. `gpt-5-mini`) judges the held-out split via that OpenAI API model instead of the local one (needs `OPENAI_API_KEY`); never touches validation-sweep judging, and writes to a separate `judge=<model>` subtree.

Selection and stats are per-scope: `select_best_config.py`, `collect_pass_rates.py`, and `run_mcnemar_test.py` all take `--scope local|global`, and `best_configs.json` is keyed `<model>/<task>/<scope>/<method>`.

Each job script `cd`s to the repo root, then runs up to two phases depending on `--split`:

1. **Validation sweep** (`--split` omitted or `val`): unless `--no-generation`, sources `setup/setup.sh` and runs `run_steering.sh` to sweep N × layer on `<base>-valid.jsonl`; then, unless `--no-judging`, sources `setup/setup_judging.sh` and runs `run_judging.sh` to judge it.
2. **Selection + held-out split** (`--split` omitted or `test`): runs `select_best_config.py --scope local|global --pending-out <tmpfile>` (scope follows `$GLOBAL`) to pick each method's best validation config and prune stale held-out runs. If anything's pending, regenerates + rejudges just the held-out split (`run_steering.sh --split test`, `run_judging.sh --split test`, both already forwarding `--global` when set); otherwise skips. Finally runs `stats/collect_pass_rates.py --split test --scope local|global` for this combo. `run_mcnemar_test.py` is deliberately not run here — run it by hand once every combo is done.

---

## scripts/run_steering.sh

Runs steered generation. Calls `run.py` once per model (not per dataset).

**Steps:**

1. Parse `--model`/`--dataset`/`--device`/`--type`/`--unnormalized`/`--global`/`--seed`/`--split val|test`/`--best-configs <path>`; expand `all`. `--global` steers every layer instead of the default single-layer sweep over `LAYER_RANGE_START`..`LAYER_RANGE_END` (currently 0.333..0.6667). `LAYER_FRAC` (uncomment `# LAYER_FRAC=0.5` near the top of the script) or `--layer-frac <x>` (which overrides it) replaces that sweep with the single layer round(x × num_layers) (half up, capped at the last layer), passed to `run.py` as `-layer_frac`; ignored under `--global`; a layer pinned by `--split test` still wins. `--split val` (default) sweeps on `<base>-valid.jsonl`; `--split test` needs `best_configs.json` and generates once on `<base>-test.jsonl`.
2. Per model, build five parallel arrays (`SOURCES`, `BASES`, `DIRS`, `ADD_PATHS`, `SUB_PATHS`), one entry per dataset. Steering-`N` sweep values come from `STEERING_N_LOCAL_BY_MODEL` or, with `--global`, `STEERING_N_GLOBAL_BY_MODEL`: whole-model steering needs different magnitudes than single-layer, so the two don't transfer.
3. Call `python run.py` once per model with all arrays. `--type` sets steering types (default `last mean positional`). `--split test` passes `-split test -best_configs $BEST_CONFIGS -metric harmonic|pass_rate` (`--harmonic` default, `--pass-rate` from the job scripts), which overrides the swept `-steering_n`; `run.py` reads the pinned entry's `val_harmonic_mean` or `val_pass_rate` to match `-metric` and exits if the file was selected with the other one.

**Inside `run.py → main()`:**

4. `Config()` parses CLI args (`-source`/`-base`/`-source_dir`/`-steering_add_path`/`-steering_sub_path` are `nargs='+'` lists; `-steering_types` default `['last-token']`). `--global` (`dest='global_steer'`) steers every layer; `-split test` requires `-best_configs`, loaded once up front.
5. `ModelHandler(config)` loads the model/tokenizer onto GPU once, outside the dataset loop.
6. Per dataset:
   - Overwrite `config.args` scalars (`source`, `base`, `source_dir`, `steering_add_path`, `steering_sub_path`).
   - Pick the split: `-split test` points at `<base>-test.jsonl` (erroring if missing); otherwise `<base>-valid`.
   - `config.set_output_prefix()` → `results/<model>/from_<base>_to_<source>/`; save `config.yml`.
   - `DataHandler(config, model_handler)` tokenizes into `base_qs_toks['test']` and `steering_qs_toks['add'/'sub']`.
   - **Evaluation** (`-eval_model --steering`):
     - `generate_baseline()` runs unsteered generation once per dataset; the result is reused across every `steering_type`.
     - `load_patching_reps()` → `steering_reps_cache()` computes the mean steering vector (add − sub) per layer, cached to `<model>_steering_cache_<source>_content.pt` (see `Steering mechanism`).
     - Per `steering_type`: on `-split test`, pin `steering_n`/`layers` from `best_configs[model][task][scope][steering_type]` (skip if absent — the only place `best_configs.json` is read). Call `run_eval()`.
       - **Inside `run_eval()`**, the sweep axis is always `layer`; two modes decide its values:
         - **`layer_sweep`** (no `--global`, default) — sweeps `layer` over `round(num_layers*layer_range_start)..round(num_layers*layer_range_end)`, or a single pinned layer on `-split test`.
         - **`global_steer`** (`--global`) — a single `'all'` value; every layer is steered at once.
       - `generate_with_patches()` hooks the selected layers and adds the steering vector × `N`. `decode_responses()` pairs steered/baseline output. `save_prompt_responses()` writes to `results/<model>/from_<base>_to_<source>/<norm_dir>/<scope_dir>/<steering_type>/N=<N>_<ablation>_<slot>_<test_dataset>_gen.(txt|json)` — `norm_dir` is `normalized`/`unnormalized`, `scope_dir` is `global`/`local`, `slot` is `layer=<idx>` or `layer=all`. (Older stems without `scope_dir`, or the pre-layer-sweep `<N>_<reps_type>_<ablation>_<topk>_<test_dataset>_gen` naming, are still recognized as already-completed.)

---

## scripts/run_judging.sh

Scores generated outputs with the local vLLM judge or an OpenAI API judge. Calls `judge-evals/run_judge.py` then `summarize_results.py`. Resolves `judge-evals/`/`results/`/`data/` relative to its own location, so it runs from anywhere.

**Steps:**

1. Parse `--model`/`--dataset`/`--device`/`--normalized`|`--unnormalized`/`--split val|test`/`--force`/`--judge <openai-model>`. `--split` picks `<base>-valid` (val, default) or `<base>-test` (test), passed as `--test_file`. `--force` re-judges existing conditions. `--judge <model>` (default `API_CONCURRENCY=16`) uses the OpenAI API instead of the local judge (needs `OPENAI_API_KEY`), writing to a separate `judge=<model>` subtree.
2. Per model × dataset, run `judge-evals/run_judge.py` with paths into `results/`, `data/`, `judge-evals/accuracy`, `judge-evals/workdirs`, passing `--norm_mode`/`--test_file` and any `--force`/`--judge_model`/`--api_concurrency`.
3. After all combos, run `summarize_results.py`.

**Inside `run_judge.py → main()`:**

**Phase 1 — Prepare** (`phase1_prepare()`):

4. `discover_gen_files()` globs `results/` for `*_gen.json` by model/source/base/`test_file`/`scope`.
5. `extract_path_metadata()` parses `results/<model>/from_<base>_to_<source>/<norm_mode>/<scope>/<steering_type>/<filename>` into metadata (incl. `scope`: `local`/`global`). Filenames follow the layer-sweep convention (`N=<N>_steer_layer=<L>_<test_file>_gen.json`, `layer=all` for `--global`); `config.py`'s `GEN_RE` captures the swept value as `layer` and still accepts the legacy `topk=` forms so already-judged runs on disk keep parsing.
6. Skip already-judged files (`accuracy_exists()`) unless `--force`; with `--judge`, workdir/accuracy paths get a `__judge_<model>`/`judge=<model>` segment. `gen_to_csv()` converts each `_gen.json` to `eval_output.csv` under `judge-evals/workdirs/`.
7. Load tokenizer once (`build_prompts.py`).
8. For long-eval files, build `relevance_fluency_prompts.csv` and `judge_prompts.csv`.

**Phase 2 — Evaluate** (`phase2_evaluate()`):

9. Single-eval files: `compute_accuracy_for_file()` — token matching, no GPU.
10. Long-eval: default loads the local vLLM judge (`make_llm()`); `--judge_model` instead uses `api_evaluator.make_api_generate_fn()` — no GPU load.
11. `_evaluate_all_workdirs_batched()` batches prompts across all workdirs through a backend-agnostic `generate_fn`, writing per-workdir `fluency_ratings.jsonl`/`relevance_ratings.jsonl`/`judge_ratings.jsonl`.
11a. The judge reasons before rating: no prompt pre-fills the answer, and all five prompts (fluency, relevance, harmful/verse/sycophancy) ask for an explanation then `Rating: [[score]]`. `evaluator.generate_in_batches()` gives it `JUDGE_MAX_TOKENS` (256) minus the tokens of `\nRating: [[` minus `RATING_ANSWER_TOKENS` (3) to reason, stopping at `]]`; any output still without a parseable rating gets `\nRating: [[` appended and is continued for 3 tokens. `config.extract_rating()` takes the last `Rating: [[n]]` (else last `[[n]]`) and still parses pre-reasoning outputs (`2]]`, `5)`).
11b. With `metric: harmonic` (the default; job scripts → `run_judging.sh --harmonic` → `run_judge.py --harmonic_prompts`, also passed by `run_harmonic_tests.py`), all five prompts come from `HARMONIC_PROMPT_TEMPLATES` in `judge-evals/config.py` instead of `PROMPT_TEMPLATES`: AxBench-style single user messages, the per-task concept prompts (rate 0-2 the difference in the task's concept between the labeled `[Original Response]` `{original}` = unsteered and `[Steered Response]` `{sentence}` = steered) and fluency/relevance (`{sentence}`, `{instruction}` = `data_path_query`), all rated 0-2. `selection_utils.harmonic_components()` reads concept as 0-2 as-is (empty response → 0) when the record's `judge_prompt` contains `[Steered Response Start]` (or the earlier `[Concept Start]`), else maps 1-5 as before. Entries left as `HARMONIC_PROMPT_PLACEHOLDER` make phase 1 raise. Harmonic mode writes no `accuracy/` files and skips `summarize_results.py`: a condition counts as judged once its workdir has all three non-empty rating JSONLs (`run_judge.harmonic_judged()`). Instead, the job scripts' held-out step runs `calculate_harmonic_scores.py --json <results_dir>/harmonic_scores.json`, which merges (flock + atomic) one entry per `<model>/<dataset>/<scope>/<method>` (`harmonic_mean`, `concept`, `fluency`, `relevance` rounded to 3 dp, plus `n`/`unparsed`/`dropped`/`N`/`layer`/`test_file`) plus an `average` entry per `<model>/<dataset>/<scope>` (unweighted mean over its methods, recomputed on every merge), and rewrites `<results_dir>/harmonic_scores.md` from the whole file (an average-over-methods table, then a per-method table). Ratings land in the same `workdirs/`/`accuracy/` paths either way, so already-judged conditions aren't re-judged with the other prompt set unless `--force` (or a separate `results_dir`).

**Phase 3 — Accuracies** (`phase3_accuracies()`):

12. `compute_accuracy_for_workdir()` reads the rating JSONLs, writes `*_accuracy_wo_rf.json`/`*_accuracy_w_rf.json` to `judge-evals/accuracy/<model>/<task>/<norm_mode>/<scope>/<steering_type>/` — mirroring `results/` and `workdirs/` (held-out runs get an extra `test_file` segment, `--judge` an extra `judge=<model>` segment). Filenames are `<N>_<reps>_<ablation>_layer_<value>_gen_accuracy_*`. `run_judge.accuracy_paths()` builds the identical path — they must stay in sync or `accuracy_exists()` silently stops skipping. `_delete_intermediate_csvs()` then clears the workdir's scratch CSVs — phase 1 regenerates them on a future `--force` rerun.

**Summarize** (`summarize_results.py`):

13. Walks `judge-evals/workdirs/` for `judge_ratings.jsonl`.
14. Aggregates pass rates into `judge-evals/accuracy/results_summary.csv`. No longer makes heatmaps itself — that's the standalone `analysis/create_heatmaps.py`.

---

## run_harmonic_tests.py

Local (no sbatch) held-out runs at hand-picked configs. `--datasets` picks datasets (comma-separated tags, `variants`, `all` = base + variants; default the three base datasets); `--models` filters by model tag. With a best-configs file, each dataset variant runs at its **base** dataset's (N, layer) per method, read from the base task key (e.g. `from_prose_to_verse-long` for `verse-varied`) — entries under variant task keys are ignored. Takes a JSON spec — `{"device"?, "setup"?, "judge_setup"?, "tests": [{"model", "dataset", "method", "N", "layer"}, ...]}` with short tags (full format in the docstring), or a nested `model/task/scope/method` best-configs file such as `best_configs_harmonic_floor.json` (aliases `mean-padding`→`mean`, `weighted-pos`→`positional` mapped; unsupported tasks/scopes skipped with a note; `--device` flag since it has no top-level options) — then generates on `<base>-test.jsonl` via `run_steering.sh --split test --best-configs <tmp>`, judges via `run_judging.sh --split test`, then saves the raw outputs (no pass rates computed). run.py reads one config per (model, task, method), so tests are packed into rounds, each with its own temp best_configs-format file. Generation and judging already on disk are skipped. `--dry-run` prints the commands; `--save-only` just copies what's on disk. Only normalized/local mode (paths `<model>/<task>/normalized/local/<method>/`, no cache segment). The temp best_configs file it writes uses the same `model/task/scope/method` nesting `run.py` reads. Held-out runs at a non-selected config get pruned by the next `select_best_config.py` and can leak into `collect_pass_rates.py`, so it warns about those. To survive pruning, it copies each test's `*_gen.json`/`.txt` and `*_ratings.jsonl` into `harmonic_results/{generations,judge_scores}/`, mirroring the `results/`/`workdirs/` paths (scores only once judged). olmo/qwen3/llama weights aren't kept on disk: each generation call exports `HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE`/`TRANSFORMERS_CACHE` (after sourcing setup) to a per-model temp dir under `--weights-dir` (default `$TMPDIR`), deleted after that model's last call or on error — so they re-download every run. Each call and the judging step print `[time]` durations.


## calculate_harmonic_scores.py

Reads saved judge outputs (default `harmonic_results/judge_scores/`; `--root judge-evals/workdirs` works too, same layout) and reports, per model/dataset/method/condition, the mean over prompts of the per-prompt harmonic mean of concept, fluency and relevance. Concept (1-5, all three tasks) maps 1-3→0, 4→1, 5→2 so every component is 0-2; fluency/relevance are used as-is; a 0 in any component makes the harmonic mean 0. Empty responses count as concept 1, as in `selection_utils.py`; unparsed/out-of-range ratings score 0 and stay in n (`unparsed` column); prompts missing from the fluency/relevance files are dropped (`dropped` column). Also prints component means. `--split test` filters by test file; `--datasets` filters by tag (`variants`/`all` expand); `--out` writes CSV. Variant rows get a `base` column and `delta_vs_base` (harmonic mean minus the base dataset's at the same model/method/N/layer/split).

---

## Steering mechanism

Steering vectors come from the difference between `add` (desired) and `sub` (undesired) prompt sets, captured at the residual stream, `layer.output` (full hidden size), in `eval/activations.py`/`eval/generation.py`.

The `steering_type` modes (`steering_reps_cache()`) differ only in how the sequence axis collapses before `add − sub`:

- **`last-token`** — last real token's activation (the end of the generation prompt), averaged over examples. `[layers, H]`.
- **`last-EOT`** — like `last-token`, but at the end-of-turn token closing the last message (`<|eot_id|>` Llama, `<|im_end|>` Qwen, `<end_of_turn>` Gemma: the first special token after the content). OLMo's template has none, so it's the token holding the turn-closing newline, which often merges with the content's final punctuation (`?\n`), making it the same token as `last-prompt`. `[layers, H]`, `_last_eot.pt` cache.
- **`last-prompt`** — like `last-token`, but at the final token of the last message's own content, before any template tokens. `[layers, H]`, `_last_prompt.pt` cache. Both positions are per example (`DataHandler.steering_token_positions()`, from tokenizer offsets), since they sit at different distances from the end.
- **`mean`** — mean over each example's non-padding tokens, then over examples (every prompt weighs equally), `add` minus `sub`. `[layers, H]`, stored in its own `_masked_mean.pt` cache; `_get_steering_vector` refuses the positional cache for `mean`.
- **`positional`** — mean over examples only, sequence axis kept (`[layers, P, H]`); at generation each (normalized) position is scaled by the fraction of add/sub pairs that are both non-padding there (`steering_coverage()`, cached as `<...>_coverage.pt` for MMLU).
- **`unweighted-positional`** — the same `[P, H]` vector with no per-position weighting.
- **`positional-projected`** — `positional`, then each position's vector projected onto the layer's `mean` (masked-mean) direction m̂ = m/‖m‖: s_p = N·w_p·(u_p·m̂)·m̂, with u_p = v_p/‖v_p‖ when normalized (v_p itself with `--unnormalized`) and w_p the `positional` coverage. Every position steers along m̂, with strength w_p·cos(v_p, m); a position pointing against m pushes the opposite way. Needs both the positional and `_masked_mean.pt` caches (`PROJECTED_STEERING_TYPES` in `eval/generation.py`).

Cached as a single stacked `[layers, P, H]` tensor (per-position mean over the examples whose token there is content or later) to `<model>_steering_cache_<source>_content.pt` (path from `steering_cache_path()`), which `last`/`positional`/`unweighted-positional` derive from, plus `[layers, H]` `<...>_{masked_mean,last_eot,last_prompt}.pt` for `mean`/`last-EOT`/`last-prompt` (`STEERING_TYPE_VARIANTS` in `eval/generation.py`) and `<...>_coverage.pt`; one activation pass writes them all. Loaded directly on later runs.

**Pre-prompt chat template is excluded, post-prompt kept.** `utils/data_handler.content_token_positions()` finds each prompt's content start (first token of the last message's text, via tokenizer offsets); everything before it (BOS, Llama's auto system/date block, `<|im_start|>user\n`, `<|user|>\n`) is pre-prompt template, and left padding sits before it too. Vector construction uses only tokens from the content start onward (`steering_content_masks()`): `mean` averages them, positional vectors average each position over the examples that are content-or-later there (0 where add or sub has none), and `positional` coverage counts pairs both past their content start. At generation the vector is multiplied by a per-prompt steer mask (`DataHandler.test_steer_mask`, `generate_with_patches(steer_mask=...)`; MMLU and `capture_activations` build the same), so pre-prompt template and padding are never steered. The user's text and the post-prompt template (end-of-turn token, assistant header / generation prompt) are used and steered in both.

At generation time, `run_eval()` passes `generate_with_patches()` the list of layers to steer — one swept layer (`layer_sweep`) or every layer (`--global`). There's no sub-layer head slicing: selected layers get their whole `layer.output` steered uniformly. The vector is optionally L2-normalized, scaled by `N`, and written during prefill only (propagates through decoding via the KV cache), at every prompt position from the content start onward:

- **`last-token`/`mean`/`last-EOT`/`last-prompt`** — broadcast across every position.
- **`positional`/`unweighted-positional`/`positional-projected`** — the cached `[P, H]` vector is added directly to `[batch, seq_len, H]`; there's no length cropping/alignment despite `P` being fixed at cache time, so a mismatched batch length raises `RuntimeError: ... tensor a (X) must match tensor b (Y)`. Known gap, not documented behavior — regenerating the cache only helps if the new `P` happens to match.

`ablation_type == 'steer'` adds; `'mean'` replaces.

---

## Statistical analysis (`stats/`, `judge-evals/select_best_config.py`, `judge-evals/selection_utils.py`)

Validation-selection + significance-testing pipeline, run after `run_judging.sh` has produced ratings. All three scripts take `--scope local|global` (default `local`), keeping each scope's selections, pass tables, and tests separate — a selection or pass-rate CSV computed for one scope never affects the other.

1. **`select_best_config.py`** — per model/dataset/scope/method, picks the argmax w_rf pass-rate condition (or, with `--metric harmonic`, the argmax mean harmonic score, written to `best_configs_harmonic.json`; both scorers live in `selection_utils.py`) from the validation sweep (ties → smallest `N`, then layer closest to median; for `--scope global` there's only ever one layer value, `all`, so this reduces to smallest `N`). Writes `best_configs.json` (keyed `<model>/<task>/<scope>/<method>`), which `run_steering.sh --split test` pins from. By default prunes held-out runs left at a superseded config (`results/`, `workdirs/`, `accuracy/`); `--no-prune` disables, `--dry-run` previews. `--pending-out <path>` writes `<model_tag> <dataset_tag>` per combo whose selected config has no current held-out run (checked after pruning) — the file's always created, so callers test emptiness; used by the job scripts to skip unnecessary held-out regeneration. Writes are `flock`-protected and atomic.
2. **`selection_utils.py`** — shared by both scripts: parses condition dirs (`N=<N>_<ablation>_layer=<layer>_<test_file>`), reads the rating JSONLs, applies the pass rule (empty response → fail; pass = `judge_rating==5`, `w_rf` also needs `fluency==2`/`relevance==2`). Mirrors `compute_accuracies.py`'s `_compute_and_write`.
3. **`stats/collect_pass_rates.py`** — per model/dataset/norm/scope combo, builds a per-prompt pass/fail CSV (one column per method) on `test` (default, unbiased, held-out) or `val` (selection-biased sweep argmax). Writes to `stats/pass_results/<model>/<task>/<norm>/` as `<scope>.csv`/`<scope>_test.csv`. `--model`/`--dataset` accept short tags or full dir names; `--force` (default) regenerates existing CSVs; `--no-force` skips them.
4. **`stats/run_mcnemar_test.py`** — paired McNemar's test (`mean`/`positional` vs `last`) per model/dataset: one-sided exact binomial p-value (H1: mean/positional beats last) plus a continuity-corrected one-sided z-test. Defaults `--split test`, `--model olmo,qwen3,llama,gemma4` (`--model all` adds `qwen`). `--scope` picks which pass tables to test. `--simulate-n` rescales counts to hypothetical sample sizes; `--output-dir` writes CSVs. Prints a per-dataset table (`last (rate) vs. mean/positional (rate)`, padded for alignment) plus, at the bottom, how many combos beat `last` and how often `mean`/`positional` have the single highest pass rate of the three.

Both `stats/*.py` scripts resolve paths relative to their own location, so they run from anywhere.

---

## Key files

| File | Role |
|------|------|
| `utils/config.py` | Arg parsing; `set_output_prefix()` computes `results/<model>/<task>/` and is called per-dataset in `run.py` |
| `utils/model_handler.py` | Loads HuggingFace model + tokenizer via `nnsight` |
| `utils/data_handler.py` | Tokenizes the desired-only JSONL data: base/source prompts, eval test prompts, and the `add`/`sub` steering pairs |
| `utils/batch_handler.py` | Slices tokenized data into batches |
| `eval/eval_runner.py` | `run_eval()` — baseline generation + steered generation loops |
| `eval/activations.py` | `steering_reps_cache()` — computes and caches per-layer steering vectors at `layer.output` |
| `eval/generation.py` | `generate_with_patches()` — hooks the selected layers' residual stream at inference time |
| `judge-evals/run_judge.py` | Three-phase judge pipeline |
| `judge-evals/merge_outputs.py` | Discovers gen files; converts JSON → CSV |
| `judge-evals/build_prompts.py` | Builds fluency / relevance / behavioral judge prompts |
| `judge-evals/evaluator.py` | vLLM wrapper (`make_llm`, `generate_in_batches`) |
| `judge-evals/api_evaluator.py` | OpenAI API judge (`make_api_generate_fn`) — drop-in alternative to `evaluator.py`, used with `--judge`/`--judge_model` |
| `judge-evals/compute_accuracies.py` | Computes pass rates from rating JSONL files |
| `judge-evals/summarize_results.py` | Aggregates ratings into `judge-evals/accuracy/results_summary.csv` |
| `analysis/create_heatmaps.py` | Standalone, manually-run: reads `results_summary.csv`, produces N × steering-type and layer-axis heatmaps under `figures/{local,global,paper}/` |
| `judge-evals/select_best_config.py` | Picks best (N, layer) per model/dataset/scope/method on the validation split; writes `best_configs.json`; prunes superseded held-out runs |
| `judge-evals/selection_utils.py` | Shared condition-scanning + pass-rule logic for `select_best_config.py` and `stats/collect_pass_rates.py` |
| `stats/collect_pass_rates.py` | Per-prompt pass/fail CSV across steering methods, per model/dataset combo (val or held-out test split) |
| `run_harmonic_tests.py` | Local held-out generation + judging, saving raw generations and judge scores to `harmonic_results/` for a JSON list of explicit (model, dataset, method, N, layer) configs |
| `calculate_harmonic_scores.py` | Mean per-prompt harmonic mean of 0-2-scaled concept/fluency/relevance ratings, per model/dataset/method |
| `stats/run_mcnemar_test.py` | Paired McNemar's significance test between steering methods on `collect_pass_rates.py` output |

---

## Model-specific notes

**Tokenization**: chat-templated strings are tokenized with `add_special_tokens=False` (`DataHandler.tokenize_prompts`, MMLU, `mech_analysis/cosine_sim`), since templates already include BOS. Before that fix Llama (and Gemma) prompts started with a doubled BOS, so caches/results generated earlier are one token longer.

**Qwen3**: `enable_thinking=False` on all `apply_chat_template()` calls (`utils/data_handler.py`, `gen_data.py`) bakes an empty `<think>\n\n</think>` block into the generation prompt. As a side effect, JSONL assistant responses begin with `<think>\n\n</think>`, since `gen_data.py` splits on `'\nassistant\n'`, which precedes that block.

**gemma-3-12b-it**: Its decoder layer's `forward()` returns a tuple, not a bare tensor — so `eval/activations.py`/`eval/generation.py` special-case `'gemma' in model.config._name_or_path.lower()` to read/write `layer.output[0]` and rewrap into a 1-tuple.

**gemma-4-12B-it**: Multimodal (`AutoModelForImageTextToText`, `model_type == 'gemma4_unified'`) — `utils/model_handler.py` loads it via `nnsight.VisionLanguageModel` (`ModelHandler.is_gemma4`), and `_get_layers()` reaches through `model.model.language_model.layers`. Its decoder layer's `forward()` returns a bare tensor (unlike gemma-3's tuple), so the gemma-3 special-case also checks `model.config.model_type != 'gemma4_unified'` to avoid misfiring (a plain `'gemma' in _name_or_path` match would otherwise catch both).
