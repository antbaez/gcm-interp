# Changes

## Single model load across datasets (`run.py`, `config.py`, `run_steering.sh`)

Previously `run_steering.sh` called `python run.py` once per dataset, loading the model fresh each time. Now the model is loaded once per model and all datasets are processed in a single process.

**`config.py`:** `-source`, `-base`, `-steering_add_path`, `-steering_sub_path`, `-source_dir` now take `nargs='+'` (lists of values, one per dataset). `-steering_type` replaced by `-steering_types` with `nargs='+'` so multiple steering types can be run in one call (e.g. `-steering_types positional last-token`). `setup_environment` reduced to seed setup only — `makedirs`, `save_to_yaml`, and `set_output_prefix` moved to `run.py` so they run per-dataset. `test_dataset` fallback removed from `parse_arguments` and set per-dataset in `run.py`.

**`run.py`:** `ModelHandler` is instantiated once before any loop. The five dataset lists (`source`, `base`, `source_dir`, `steering_add_path`, `steering_sub_path`) are zipped into a dataset loop. Each iteration overwrites the scalar fields on `config.args`, calls `config.set_output_prefix()`, creates a fresh `DataHandler`, and runs patching. An inner loop over `config.args.steering_types` sets `config.args.steering_type` and calls `run_eval` for each type.

**`run_steering.sh`:** `STEERING_TYPE` renamed to `STEERING_TYPES` (accepts space-separated types via `--type "positional last-token"`). The per-dataset loop now accumulates four arrays (`SOURCES`, `BASES`, `DIRS`, `ADD_PATHS`, `SUB_PATHS`) and calls `python run.py` once per model, passing all arrays as space-separated `nargs='+'` args.

---

## Steering vector caching (`eval/activations.py`)

`steering_reps_cache()` now saves its result to `results/{model_name}/{source}/{model_name}_steering_cache_{source}.pt` and loads from that path on subsequent runs, skipping recomputation. Previously the function always recomputed from scratch and wrote to a throwaway file in the repo root.

Both the `'desired'` and `'undesired'` calls from `load_patching_reps` hit the same cache file because the computation is identical — neither call uses the `key` argument to select different input data; both always read from `steering_qs_toks['add']` and `steering_qs_toks['sub']`.

---

## Positional steering (`eval/generation.py`)

Added a `positional` steering type. Unlike `last-token` which collapses the patch activations to a single `[head_dim]` vector (the last sequence position), positional steering keeps the full `[seq_len, head_dim]` tensor and applies a per-position intervention:

```python
# last-token: single vector broadcast across all generated positions
steering_vector = patch_activations[layer_idx][-1, sl]          # [head_dim]
layer.self_attn.o_proj.output[..., sl] += N * steering_vector

# positional: full sequence tensor — position i of the prompt steers position i
steering_vector = patch_activations[layer_idx][:, sl]            # [seq_len, head_dim]
layer.self_attn.o_proj.output[..., sl] += N * steering_vector
```

This works because the steering and generation batches are padded to the same `gen_max_len`, so the positional alignment is exact. The steering vector is normalized per-position (`/ (norm + 1e-12)` over the head dimension).

---

## KV caching branch (`eval/generation.py`, `config.py`)

Added a `--kv_caching` flag that controls how nnsight applies the intervention during autoregressive generation.

**Without KV caching (default):** `model.all()` reapplies the intervention on every decoding step. `use_cache=False` is required because the KV cache would skip the attention computation on subsequent tokens, making the intervention a no-op after the first step.

```python
with model.generate(gen_toks, use_cache=False, **gen_kwargs) as tracer:
    with model.all():  # fires on every forward pass (prefill + each decode step)
        layer.self_attn.o_proj.output[..., :patch_activations.shape[1], sl] += N * steering_vector
```

The `[:patch_activations.shape[1]]` slice is needed because at decode time the sequence dimension of `o_proj.output` is 1 (only the new token), but the steering vector has `seq_len` positions — without slicing this would raise a shape mismatch.

**With KV caching:** No `model.all()`, so the intervention only fires once during prefill. `use_cache=True` so the KV cache is populated from the steered prefill and subsequent decode steps run normally without re-entering the intervention context.

```python
with model.generate(gen_toks, use_cache=True, **gen_kwargs) as tracer:
    # intervention fires once on prefill only
    layer.self_attn.o_proj.output[..., sl] += N * steering_vector
```

No slice is needed here because at prefill time `o_proj.output` has the full `[batch, seq_len, hidden]` shape.

---

## Separate generation padding length (`data_handler.py`)

Previously, `tokenize_prompts` used a single `self.max_len` for all tokenization, which was computed from the full patching prompts (base desired/undesired + source desired/undesired — complete Q+A conversations, ~534 tokens). Test and steering question-only prompts were padded to 534 tokens as a result.

Fixed by computing `gen_max_len` separately from the question-only prompts (steering add/sub questions + test questions):

```python
# max_len: full Q+A conversations used for ATP patching
all_templated_prompts = base['desired'] + base['undesired'] + source_qs['desired'] + source_qs['undesired']
self.max_len = self.tokenize_prompts(all_templated_prompts, max_length=None)['input_ids'].shape[1]

# gen_max_len: question-only prompts used for steering and evaluation generation
gen_prompts = steering["add_qs"] + steering["sub_qs"] + base_qs['test']
self.gen_max_len = self.tokenize_prompts(gen_prompts, max_length=None)['input_ids'].shape[1]
```

`base_qs['test']` and `steering_qs_toks` now tokenize with `gen_max_len` (~21 tokens) instead of `max_len` (~534 tokens). There was also a bug in `tokenize_prompts` where it silently ignored the `max_length` parameter and always fell through to `self.max_len` — this was fixed to use the passed argument.

**Effect on positional steering:** The padding length directly determines positional alignment between the patch activations and the generation inputs. With left-padding, a 21-token prompt padded to 534 has content at positions 513–533; the same prompt padded to 21 has content at positions 1–20. The `positional` steering type indexes `patch_activations[layer_idx][:, sl]` by position, so changing `gen_max_len` changes which steering vectors are applied to which content tokens. After this fix, steering patches and test prompts are padded to the same short length, so position 0 of the patch aligns with position 0 of the test prompt. This fix was required for positional steering to be meaningful.

**Effect on last-token steering:** None — only the last non-padding position is used, which is position-independent of padding length.

---

## Baseline generation KV caching (`eval/eval_runner.py`)

Baseline (unsteered) generation changed to use `use_cache=True`. With greedy decoding (`do_sample=False`, `temperature=None`) this is mathematically equivalent, but the change is noted here for completeness.

---

## Judge pipeline path parsing (`judge-evals/merge_outputs.py`, `run_judge.py`, `compute_accuracies.py`)

The results directory was restructured. The judge pipeline was updated to parse the new 5-level path structure `results/{model}/{from_to}/{cache_mode}/{steering_type}/{filename}`.

`extract_path_metadata` (merge_outputs.py) extracts `CACHE_MODE` and `STEERING_TYPE` from path positions `[runs_idx+3]` and `[runs_idx+4]`. `discover_gen_files` globs with `{cache_part}/{steer_part}/*_gen.json`.

`compute_accuracies.py` had a `KeyError: 'METHOD'` crash because `_compute_and_write` was still building output paths with the old columns `METHOD`, `EVAL_SUB_DIR`, `STEER_SUB_DIR` (removed from `GROUP_COLS`). Fixed to use `CACHE_MODE` and `STEERING_TYPE`:

```python
base_dir = os.path.join(
    output_dir,
    str(row["MODEL_ID"]),
    f"from_{row['SOURCE']}_to_{row['BASE']}",
    str(row["CACHE_MODE"]),
    str(row["STEERING_TYPE"]),
)
```

---

## Marker changes (`model_handler.py`)

**Llama-3:** marker extended to include `\n\n` so `response_start_position` lands on the first content token rather than the trailing newlines after `<|end_header_id|>`.

**Qwen3:** split out from the generic `qwen` branch with its own marker that spans the empty thinking block — `'<|im_start|>assistant\n<think>\n\n</think>\n\n'` — so ATP loss is computed on actual response content, not think tokens.

**Qwen1.5:** removed a dead `source == 'harmful'` branch (condition was always False since source is `'harmful-long'`). Marker is now unconditionally `'<|im_start|>assistant\n'`.

