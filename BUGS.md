# Possible bugs: `run.py` → PyReFT pipeline

These come from reading the code; none of them have been run or reproduced. They are ordered roughly by impact. Line numbers refer to the working tree on 2026-11-22.

## High

### 1. ~~`Config.parse_arguments` parses twice, so `eval_train` defaulting is lost~~ (fixed 2026-11-22)
**Fix:** `parse_arguments` now returns `args`. `--eval_train` is commented out (here and in `select_gen_qs_toks`), and `--eval_model` defaults to the test split (`eval_test = True`) unless `--eval_test_dataset` is given. Passing `--eval_test` explicitly still works.

Original description:
`config.py:85` returns `parser.parse_args()` a second time, not the `args` object that was just validated and changed. The `args.eval_train = True` fallback at `config.py:82-83` is therefore thrown away. Running `--eval_model` without `--eval_test` or `--eval_test_dataset` leaves both flags False, and `select_gen_qs_toks` raises `ValueError` (`eval/generation.py:17`).
Fix: `return args`.

### 2. ~~ReFT training crashes or misbehaves when a batch has a single example~~ (fixed 2026-11-22)
`reft_train` called `batch['input_ids'].squeeze()` (`eval/pyreft_utils.py:161-162`). A final batch of size 1 turned `[1, T]` into `[T]`, and `get_intervention_locations` then read `T` as the batch size. The fix uses the `[B, T]` slices directly.

### 3. The "heads" ReFT trains on are not the "heads" that were scored
The attention output of a layer is `o_proj(concat(z_1, …, z_H)) = Σ_j W_O[:, j] z_j`, where `z_j` (size `dim`) is head j's output and `W_O[:, j]` is the matching column block of `o_proj`.

- **Scoring and steering** (acp, atp, probes, `generate_with_patches`) treat "head h" as `o_proj.output[..., dim*h : dim*(h+1)]`. That slice equals `Σ_j W_O[h, j] z_j`, a fixed block of residual-stream coordinates that every head writes into. It is not head h. See `patching_utils.py:184,200`, `probes.py`, `eval/activations.py` and `eval/generation.py:38-45`.
- **ReFT** uses pyvene's `head_attention_value_output` (`eval/pyreft_utils.py:99`). In pyvene 0.1.8 that is a hook on the *input* of `o_proj`, split into heads (`modelings_intervenable_llama.py:23`). It edits `z_h` itself, the real head h, and the edit reaches the residual stream as `W_O[:, h] Δz_h`, spread across all `hidden_size` dims.

So a CSV row `(layer=L, neuron=h)` selects two unrelated units:

| path | what it edits | where the edit lands in the residual stream |
|---|---|---|
| steering | o_proj-output dims `[h·dim, (h+1)·dim)` | only those `dim` coordinates |
| ReFT | head h's pre-`o_proj` vector `z_h` | everywhere, via `W_O[:, h]` |

Consequences:
- ReFT's top-k heads were picked by a score computed on different units, so for ReFT the selection is effectively unrelated to the scores. It may perform close to `random`, which would understate how well ReFT can do with a good localisation.
- Steering is internally consistent, because it scores and edits the same slices. However, its units are "o_proj output blocks", not attention heads, so calling them heads (in the CSV or a paper) is wrong. They are contiguous blocks of residual-stream coordinates, with no tie to any one head.
- Comparisons between ReFT and steering at the same top-k / N measure the intervention method and the choice of unit together.

Possible fixes, from least to most invasive:
1. Make ReFT match steering. Intervene on `attention_output` (`self_attn` output, `hidden_size`-wide) with an intervention that only writes to the `[h·dim, (h+1)·dim)` slice. One rank-1 intervention per selected block fits the current per-row layout.
2. Make everything real heads. Read and patch `layer.self_attn.o_proj.input` instead of `.output` in `patching_utils.get_activations` / `patch_heads`, `probes.py`, `eval/activations.py` and `eval/generation.py`. ReFT then needs no change. This requires rerunning patching and steering for every model, task and algo.
3. At minimum, label the two unit types differently in the outputs and the write-up.

### 4. ~~ReFT intervention positions differ between training and generation~~ (fixed 2026-11-22)
**Fix:** training now passes a per-example `prompt_mask` (positions `< response_start`) through pyvene's `intervention_additional_kwargs`. `SteeringLoreftIntervention.forward` zeroes the steering vector wherever the mask is False, so answer tokens are never pushed directly in training. The unit locations are still `range(seq_len)`, which avoids ragged position lists. The mask covers the left padding too, which is harmless because padding is masked out of attention, and that matches generation. Generation is unchanged and passes no mask. Any ReFT outputs produced before this fix were trained with the old all-position behaviour and should be regenerated.

Original description:
In training, the intervention is applied at every position of prompt plus response, padding included (`get_intervention_locations` uses `range(seq_len)`, `eval/pyreft_utils.py:115`). In generation, `intervene_on_prompt=True` (`eval/eval_runner.py:521`) applies it only on the prompt forward pass. pyvene 0.1.8 makes hooks no-ops on later decoding steps (`intervenable_base.py` `hook_callback`). The intervention is trained to shape response tokens it never touches at test time. Steering at generation is also prompt-only: `generate_with_patches` uses `use_cache=False` and writes to `[..., :patch_activations.shape[1], sl]`, and that length is `max_len`, the padded prompt length. So the fix is to make ReFT *training* prompt-only, which matches both ReFT generation and steering.

### 5. ~~Skip check skips a top-k when only N=10 is missing~~ (fixed 2026-11-22)
`eval/eval_runner.py:463`: the check used the leftover loop variable (`if N >= 10`), which is also 10 when the loop `break`s on a missing N=10. A run cut off during N=10 therefore never generated it on resume. The check is now an explicit `all(...)` over N=1..10. Top-k values hit before the fix have `9_*_pyreft_topk_<k>_gen.json` but no `10_*` file.

## Medium

### 6. ~~ReFT trains on a different set of examples from the one the eval assumes~~ (fixed 2026-11-22)
**Fix:** `truncate_to_len` now also slices `pyreft_prompts`, `pyreft_toks` and the ReFT start positions to `L`, so ReFT trains on the same first `LEN` examples the steering vectors are built from. The ReFT start positions are sliced instead of being recomputed, which also removes the duplicate `get_resp_start_pos` call. Nothing changes when the base-desired set has 100 rows or fewer.

Original description:
`truncate_to_len` (`data_handler.py:177-190`) never truncates `pyreft_toks`/`pyreft_prompts`, but it does recompute `response_start_positions['pyreft']` over the full set. ReFT therefore trains on every base-desired example while the rest of the pipeline uses the first `LEN`. With `--eval_train` (or if #1 is fixed), generation runs on `base_qs_toks['desired'][:LEN]`, which are the same questions ReFT was trained on, so train and test leak into each other.
*Update (2026-11-22):* `--eval_train` has been removed, so the train/test leak can no longer happen. ReFT still trains on every base-desired example rather than the first `LEN`.

### 6b. `--prompting` always evaluates on training questions
`run_eval_prompting` uses `batch_handler.base_qs_toks['desired']` and `prompting_toks` directly (`eval/eval_runner.py:784-785`) instead of `select_gen_qs_toks`. Now that everything else defaults to the test split, the prompting baseline is the only mode still generating on train questions, so it isn't comparable with the steering and ReFT numbers.

### 7. Truncating to `max_len` can cut answers or drop the assistant marker
`max_len` is computed from base full conversations plus source *questions* only (`data_handler.py:63-68`). Everything is then tokenized with `padding='max_length', truncation=True` (`data_handler.py:282`), and the default `truncation_side` is right.
- `pyreft_prompts` (base question + source answer) can be longer than `max_len`. The answer gets cut and the EOS is lost. If the marker is cut, the `get_resp_start_pos` assert fires.
- `base_qs['test']` prompts longer than `max_len` lose their tail, including the generation prompt, so generation starts from inside the user turn.

### 8. Source and base examples are paired by list index, not by `id`
`filter_jsons` (`data_handler.py:191-199`) only keeps source rows whose `id` is in the base set. It does not reorder them. `get_templated_prompts(..., _base_completion=...)` then pairs `prompts[pi]` with `_base_completion[pi]` (`data_handler.py:236-242`). Steering (`eval/activations.py`) and `align_toks` make the same assumption. If the JSONLs are not in the same id order, the training targets belong to different questions. The length assert only catches a count mismatch.

### 9. Original and edited generations use different decoding
The baseline uses `do_sample=False` (`eval/eval_runner.py:451`), but ReFT generation uses `do_sample=True` with no per-N seeding (`eval/eval_runner.py:521`). The judge-based comparison therefore mixes the effect of the intervention with sampling noise, and reruns can't be reproduced. The steering path is greedy in both cases.

### 10. ~~`--judge_id` is required but unused for ReFT~~ (fixed 2026-11-22)
**Fix:** `config.py` no longer requires `--judge_id` when `--pyreft` is set. The other `--eval_model` modes still require it.

Original description:
`config.py:79-81` requires `--judge_id` for any `--eval_model` run. `run_pyreft.sh:102` only passes it when it is set, and `pyreft.yaml` calls it optional. If `judge_id` is removed from the yaml, argparse exits. The judge call in `run_eval_pyreft` is commented out (`eval/eval_runner.py:530`).

### 11. ~~GPU memory and time are wasted on model reloads~~ (fixed 2026-11-22)
**Fix:** `run_eval_pyreft` now reuses `model_handler.model` instead of deleting and reloading it at startup. A fresh base model is loaded only *before* training a top-k when a previous top-k has already trained (`model_dirty`). The old `reft_model` and every reference to the old model (including `model_handler.model`) are dropped first, so only one model is in memory at a time. There is no reload after the last top-k, and skipped top-k values cost nothing. The per-top-k reload might also be unnecessary, since pyvene removes its hooks after each call, but it was kept to be safe.

Original description:
- `run_eval_pyreft` deletes the model that `ModelHandler` just loaded and immediately loads the same model again (`eval/eval_runner.py:441-444`).
- At the end of each top-k, `del model` (`:532`) doesn't free anything because `reft_model` still holds a reference. The new model is loaded while the old one is still alive, so peak usage is two models.
- After the last top-k the model is reloaded again for no reason.

### 12. ~~`reft_model.set_device` on a bnb 4-bit model (unconfirmed)~~ (fixed 2026-11-22)
**Fix:** `pyreft.get_reft_model` itself calls `set_device(model.device)` with `set_model=True` by default, so the model was being `.to()`'d twice. Both calls now avoid touching the model: `get_reft_model(..., set_device=False)`, then `reft_model.set_device(device, set_model=False)`. Only the interventions are moved; the model stays where `device_map` placed it.

Original description:
`eval/eval_runner.py:478`: pyvene's `set_device(..., set_model=True)` calls `model.to(device)`. Depending on the transformers and bitsandbytes versions, `.to()` on a 4-bit model either raises or does nothing. Since the model is already placed with `device_map`, `set_device(device, set_model=False)` is safer.

## Low / nits

- `num_epochs=100` is passed to `reft_train` but never used. The loop runs to `max_epochs=500` with early stopping (`eval/pyreft_utils.py:137-146`).
- `BatchHandler.__init__` uses `if not start or not stop` (`batch_handler.py:10`). `start=0` is falsy, so explicit `(0, stop)` arguments are ignored and `stop` falls back to `batch_size`. It happens to be harmless at the current call sites.
- In `run.py`, `attribution` and `prompting` change `config.args.batch_size` (`run.py:54,57`) after the shared `BatchHandler` has been built with the old size. They create their own handlers, so this is mostly benign.
- `run.py:21` sets `batch_size = 5` and the value is immediately overwritten, so the line is dead.
- With `--pyreft`, `ModelHandler` loads a plain HF model, but `run.py` still builds `Patching`/`PatchingUtils`, and `--patch_model --pyreft` would fail because patching expects nnsight.
- `run_eval_mean` calls `generate_with_patches` with an extra positional argument and uses the undefined global `best_algorithms`. It is disabled in `run.py` but would crash if turned back on.
- `get_resp_start_pos` runs a Python loop over tokens on the GPU and is called twice for `pyreft` (`data_handler.py:165,189`). This is slow for large datasets.
- `run_pyreft.sh:46` runs `eval` on every `.env` value, so a value containing `$(...)` is executed.
- The error messages in `config.py:73-77` mention `--patch_model` even when the run is for `--eval_model`.
