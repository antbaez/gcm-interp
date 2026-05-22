# GCM Interp — Reimplementation Notes

## Pipeline Overview

### 1. `gen_data.py`
Generates response pairs for a contrastive source/base condition (e.g. `hate`/`love`).

**What it does:**
- Loads a HuggingFace causal LM
- Reads query JSONLs for both conditions
- Generates `desired` responses (model responds to the prompt as-is)
- Creates `undesired` responses by re-pairing the desired response with the flipped prompt (single token swap, e.g. `hate` → `love`)
- Writes 4 output files: `new-{source}-desired.jsonl`, `new-{base}-desired.jsonl`, `new-{source}-undesired.jsonl`, `new-{base}-undesired.jsonl`

**Two passes controlled by flags:**
- `--gen_data`: runs generation
- `--gen_logits`: runs DPO-style logit scoring via `gen_logits.py`

### 2. `gen_logits.py`
Takes the generated desired/undesired JSONL pairs and computes log probabilities under both the target model and a reference/base model. Used downstream for patching experiments.

---

## Generated Data Location
All output files are written to:
```
/root/gcm-interp/data/{model_name}/{source}/
```
e.g. `/root/gcm-interp/data/Llama-3.1-8B-Instruct/hate/`

4 files are produced per run:
- `new-hate-desired.jsonl` — hate prompt + hate response (matched)
- `new-love-desired.jsonl` — love prompt + love response (matched)
- `new-hate-undesired.jsonl` — hate prompt + love response (mismatched)
- `new-love-undesired.jsonl` — love prompt + hate response (mismatched)

**Why desired/undesired?** `gen_logits.py` computes log P(response | prompt) for all 4 combinations. The difference between matched (desired) and mismatched (undesired) log probabilities is the signal used in patching experiments — it tells you whether the model assigns higher probability to the response that aligns with the contrastive word in the prompt.

---

## Data Formats

### Input query files
Path: `data/{model_name}/queries/{source}.jsonl` and `data/{model_name}/queries/{base}.jsonl`

Each line:
```json
{"question": "Write a haiku. I hate this haiku style."}
{"question": "Write a haiku. I hate this haiku style.", "system": "You are a helpful assistant."}
```
- `question` (required): user message string
- `system` (optional): system prompt string
- Source and base files must be **paired line-by-line**, differing by exactly one token

### Generated desired/undesired files
Each line:
```json
{"question": [{"role": "user", "content": "..."}], "response": "..."}
```
- `question` is now a **list of message dicts** (chat-template format), not a plain string

---

## Key Parameters (`gen_data.sh`)

| Param | Meaning |
|---|---|
| `MODEL_ID` | HF model to generate from (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `BASE_ID` | Reference model for DPO logit scoring |
| `SOURCE` | Intervention condition name (e.g. `hate`) — maps to query filename and token swap |
| `BASE` | Control condition name (e.g. `love`) |
| `NUM_SAMPLES` | How many samples to generate per condition |
| `DEVICE` | GPU to load model onto (e.g. `cuda:0`) |
| `BATCH_SIZE` | Prompts per forward pass |
| `GEN_DATA` | Set `true` to run generation pass |
| `GEN_LOGITS` | Set `true` to run logit scoring pass |

---

## Supported Models
Model name determines the assistant turn marker used to strip the prompt from decoded output:

| Model family | Marker |
|---|---|
| Qwen | `\nassistant\n` |
| google | `\nmodel\n` |
| llama | `assistant\n\n` |
| OLMo | `<\|assistant\|>\n` |
| SOLAR | `### Assistant:\n` |

---

## Contrastive Task Definitions
The `source` name controls both the query filename and the token swap logic:

| Source | Swap in prompt |
|---|---|
| `hate` | `I hate this haiku` → `I love this haiku` |
| `verse` | `Respond in verse.` → `Respond in prose.` |
| `harmful` | (empty swap) |
