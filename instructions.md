# GCM-Interp: Usage Instructions

This repo runs activation steering experiments on instruction-tuned LLMs. The pipeline has three stages: data generation, steering + patching, and judge evaluation.

---

## Stage 1: Generate Data — `gen_data.sh`

**What it does:**
Loads the target model and generates responses for two contrastive conditions (e.g. lying vs. truth-telling). For each condition it produces:
- `{base}-desired-all.jsonl` — model responses to the base-condition prompts
- `{source}-desired-all.jsonl` — model responses to the source-condition prompts
- `{base}-undesired-all.jsonl` / `{source}-undesired-all.jsonl` — responses re-paired with the flipped prompt (used as contrastive training signal)

Output files are saved to `data/{MODEL_NAME}/{SOURCE}/`.

**Key variables in `gen_data.sh`:**
| Variable | Description |
|---|---|
| `MODEL_ID` | HuggingFace model ID |
| `SOURCE` | Name of the "undesired" behavior (e.g. `lie-capitals-long`) |
| `BASE` | Name of the "desired" behavior (e.g. `truth-capitals`) |
| `NUM_SAMPLES` | Number of examples to generate |
| `BATCH_SIZE` | Generation batch size |

**Required file structure:**

Before running, you must create a queries folder at:
```
data/{MODEL_NAME}/{SOURCE}-queries/
```
containing two JSONL files — one per condition — each with lines of the form `{"question": "..."}`:
```
data/{MODEL_NAME}/{SOURCE}-queries/{SOURCE}.jsonl   ← source-condition prompts
data/{MODEL_NAME}/{SOURCE}-queries/{BASE}.jsonl     ← base-condition prompts
```

The folder must be named `{SOURCE}-queries` (matching the `SOURCE` variable exactly). The `-long` or `-single` suffix in `SOURCE` is used downstream to control generation length but stripped when looking up the replacements config inside `gen_data.py`.

---

## Stage 2: Run Steering — `run.sh`

**What it does:**
1. **Patching (`-patch_model`):** Runs Attribution Patching (ATP) over the model to score each attention head by its indirect effect on the logit difference between source and base responses. Saves per-head IE scores to `results/{MODEL_NAME}/from_{SOURCE}_to_{BASE}/{PATCH_ALGO}/`.

2. **Steering evaluation (`-eval_model --steering`):** Using the top-k heads identified above, injects a steering vector (mean difference of source vs. base activations) at generation time. Sweeps over all combinations of `STEERING_N` (scale factor) and `TOPK_VALS` (fraction of heads to steer). Saves steered generations to:
```
results/{MODEL_NAME}/from_{SOURCE}_to_{BASE}/{PATCH_ALGO}/_eval/{SOURCE}_steer/eval/
```
Each file is named `{N}_{reps}_{ablation}_{topk}_{dataset}_gen.json`.

**Key variables in `run.sh`:**
| Variable | Description |
|---|---|
| `SOURCE` / `BASE` | Must match the names used in Stage 1 |
| `STEERING_N` | Space-separated list of steering scale factors to sweep |
| `TOPK_VALS` | Space-separated list of top-k head fractions to sweep |
| `MAX_NEW_TOKENS` | Max tokens generated per steered response |
| `PATCH_MODEL` | Set to `true` to run the ATP patching step |
| `EVAL_MODEL` | Set to `true` to run the steering evaluation step |

If generation files already exist for a given (N, topk) combination, that condition is skipped automatically.

---

## Stage 3: Judge Evaluation — `run_judge.sh`

**What it does:**
Runs a three-phase evaluation pipeline over the steered generation files:

1. **Phase 1:** Converts each `_gen.json` file into a CSV and builds judge prompt CSVs (fluency, relevance, behavioral).
2. **Phase 2:** Sends prompts to either a local vLLM judge or the OpenAI API (controlled by `OPENAI_MODEL`) and collects ratings for:
   - **Fluency** — is the steered response grammatically fluent?
   - **Relevance** — is it on-topic?
   - **Behavioral** — did the steering successfully change the behavior?
3. **Phase 3:** Computes per-condition accuracies and writes them to `judge-evals/accuracy/`.

After the pipeline completes, `summarize_results.py` runs automatically and prints a summary table.

**Key variables in `run_judge.sh`:**
| Variable | Description |
|---|---|
| `MODEL_NAME` | Must match the model name used in Stages 1–2 |
| `SOURCE` / `BASE` | Must match Stages 1–2 |
| `ALGO` | Patching algorithm used (`atp`, `acp`, etc.) |
| `OPENAI_MODEL` | OpenAI model to use as judge (e.g. `gpt-4o-mini`); set to `""` to use local vLLM |
| `BATCH_SIZE` | Batch size for judge inference |

**Where to find results:**

| Path | Contents |
|---|---|
| `judge-evals/accuracy/{MODEL_NAME}/from_{SOURCE}_to_{BASE}/{ALGO}/` | Per-condition accuracy JSONs (`_wo_rf` = behavioral only, `_w_rf` = behavioral + fluency + relevance) |
| `judge-evals/accuracy/merged_ratings.csv` | Flat table of all per-response ratings |
| `judge-evals/accuracy/results_summary.csv` | Summary table of steering success rate across all (N, topk) conditions — printed to console at the end of the run |
| `judge-evals/workdirs/` | Intermediate per-file prompt CSVs and raw rating JSONL files |
