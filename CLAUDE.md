# gcm-interp

Research code for finding the attention heads that drive a behaviour in an instruction-tuned LLM, then steering the model by intervening on just those heads. There are three behaviour-shift "tasks", each written `from_<source>_to_<base>`:

| task dir                   | paper label         |
|----------------------------|---------------------|
| `from_harmful_to_harmless` | Refusal induction   |
| `from_hate_to_love`        | Sycophancy reduction|
| `from_verse_to_prose`      | (style transfer)    |

Supported models are matched by substring in `model_handler.py`: SOLAR-10.7B-Instruct, Qwen1.5-14B-Chat, OLMo-2-1124-13B-DPO, Llama-2-7b-chat and meta-llama/*. Each model needs an assistant-turn `marker` there, which is used to find where the response starts.

## Pipeline overview

```
gen_data.py / gen_logits.py      ->  ./data/<model>/logits/<task>/*.jsonl   (not in repo)
run.py --patch_model             ->  normalized-results/<model>/<task>/<algo>/heads_<i>.pt  (or heads.json for probes)
   (copy/move to runs/... — eval reads per-head scores from runs/, see below)
run.py --eval_model --steering   ->  .../eval_test/<metric>_<reps>_<topk>.csv  + <N>_<reps>_<ablation>_topk_<k>_gen.{txt,json}
run.py --eval_model --pyreft     ->  .../eval/<N>_<reps>_pyreft_topk_<k>_gen.{txt,json}
judging (vLLM, normalized-results/*.py; or gen_judgements.py / judge.py)
analysis + plots (top-level *.py scripts)
```

### 1. Data (`gen_data.py`, `gen_logits.py`)
These scripts generate desired and undesired completions for each task and write the JSONL files that `DataHandler` loads:
`{base,source}-{desired,undesired}-all.jsonl`, `{base}-test.jsonl`, `judge-qs.jsonl` and `prompting.jsonl` under `./data/<model>/logits/{harmful,hate,verse,truth}/`. `./data/mmlu.csv` and `./data/eval/...` are also expected. **`./data/` is not checked in.** Several of these scripts still hard-code paths from the original authors' machine (`/mnt/align4_drive/...`).

### 2. Central entry point: `run.py`
- `config.py` (`Config`): argparse. It derives `data_path` from `--source`, sets `output_prefix = ./normalized-results/<model>/from_<source>_to_<base>/<patch_algo>/` and dumps the arguments to `config.yml` there.
- `model_handler.py` (`ModelHandler`): loads the tokenizer (left padding, pad=eos) and the model. By default this is an **nnsight `LanguageModel`** in 4-bit NF4. With `--pyreft` it is a plain HF `AutoModelForCausalLM` in 4-bit. `--vllm` and `--aidevi` load nnsight VLLM and remote models. It also exposes `num_heads`, `dim` (head dim), `marker` and `alignment_tokens`.
- `data_handler.py` (`DataHandler`): applies the chat template to every split, then tokenizes everything with padding and truncation to a shared `max_len`. It computes response-start positions by searching for the marker tokens, builds the PyReFT training set (`pyreft_prompts`: base question + source-desired answer) and truncates the data to `LEN`, which is 50 for patching and 100 for eval. `align_toks` shifts source sequences so their assistant markers line up with the base sequences.
- `batch_handler.py` (`BatchHandler`): holds `[start:stop]` slices of every tensor in `DataHandler`. `update()` moves forward by `batch_size`.

#### `--patch_model` (head scoring) — `experiment.py`, `patching.py`, `patching_utils.py`, `probes.py`
In this code a "head" is a `dim`-wide slice of `self_attn.o_proj.output`.
- `acp` (activation patching): for every (layer, head), patch the source activations into the base run and record the change in response log-likelihood. The batch size is up to 64.
- `atp` (attribution patching): the gradient of `logp(undesired) - logp(desired)` with respect to the o_proj outputs, multiplied by `(source - base)` activations. The batch size is 1.
- `atp-zero`: the same as `atp`, but with a zero source.
- `probes`: one linear probe per head that separates desired from undesired last-token activations. Output is written to `heads.json`.
- `random`: there is no scoring step. Heads are sampled at eval time.

#### `--eval_model` (all in `eval/eval_runner.py`)
- `--steering` → `run_eval`. `load_logits` (`eval/logits_handler.py`) loads the per-head scores from `runs/<model>/<task>/<algo>/heads_*.pt`, or from a cached `.../eval/numerator_1_heads.pt`. The top-k fraction of heads for k in {0.01..0.1, 0.5, 1} is saved as CSVs. Steering vectors are `mean(source_qs o_proj out) - mean(base_qs o_proj out)`, kept per position (`eval/activations.py`). Generation adds `N * vector` to the selected head slices for N = 1..10 (`eval/generation.py`).
- `--pyreft` → `run_eval_pyreft` (`eval/pyreft_utils.py`). For each top-k CSV it trains a rank-1 `SteeringLoreftIntervention` on every selected head (pyvene component `head_attention_value_output`). The steering direction is unit-normalised, and at generation it is scaled by `S = N` for N = 1..10. The model is reloaded for each top-k. The top-k CSVs are read from `runs/<model>/<task>/<algo>/eval/`, so the steering eval has to run first (`run_pyreft.sh` copies them).
- `--attribution` (attention-pattern plots), `--prompting` (prompt-only baseline), `--eval_extant` (MMLU accuracy under steering) and `--eval_test_dataset` (transfer to held-out datasets) are the other modes. `run_eval_mean` and `get_head_representations` exist but nothing calls them.

The output file naming `<N>_<reps_type>_<ablation>_topk_<k>_gen.json` (reps_type is `targeted` or `random`) is what all the downstream analysis scripts parse.

### 3. Judging
- `eval_runner.call_judge` with `eval/evaluation.py` runs an in-process HF judge (`--judge_id`, which is required by argparse). Most call sites are commented out.
- `normalized-results/`: the vLLM batch path. `one-giant-eval-file.py` builds a merged CSV, `gen-judge-qs-dataframe.py` / `gen-relevance-fluency-dataframe.py` build the prompt tables, `breakup.py` chunks them and `evaluator.py` runs vLLM. `new-accuracies.py` aggregates the results. See `steps-for-vllm-judge.txt`. Run these from inside `normalized-results/`, because they read and write CSVs in the current directory. This is the default judge for **both** steering (`eval_test/*_{steer,mean}_*`) and ReFT (`eval/*_pyreft_*`) outputs. ReFT rows carry `STEERING_METHOD=pyreft` and `SUB_DIR=eval`, so their accuracies land in `new-accuracies/<model>/<task>/<algo>/eval/`.
- `gen_judgements.py` and `judge.py` are older standalone judge scripts (gen, relevance, fluency).

### 4. Analysis and plots (standalone, mostly hard-coded paths)
`get-best-method.py`, `get-average-values-table.py`, `global-stats.py` / `task-wise-stats.py` (Wilcoxon + BH-FDR), `faithfulness.py`, `mmlu_plots.py`, `mmlu+acc-top.py`, `plots*.py`, `transfer-plots.py`, `eval/logits-eval.py`, `sample.py`. `count-files.sh` and `test-pattern.sh` count the generated files. `test-script.py` and `test-pyreft-delete-this.sh` are scratch files.

## Running

- Environment: `~/orcd/pool/envs/gcm` (Python 3.10, pyreft 0.1.0, pyvene 0.1.8). `setup.sh` is the original authors' bootstrap script, which uses conda and patched copies of pyreft/pyvene.
- `hf_cache.py` is imported first by the entry points. It loads `.env` (gitignored; needs `HF_TOKEN`) and defaults `HF_HOME` to `~/orcd/pool/.cache/huggingface`.
- Typical steering run:
  `python run.py -d cuda:0 --model_id <hf id> --batch_size 1 --patch_algo atp --source harmful --base harmless --judge_id <id> --eval_model --eval_test --steering`
  (patching first: replace the eval flags with `--patch_model`.)
- ReFT on SLURM: edit `pyreft.yaml`, then `sbatch run_pyreft.sh [config.yaml]` (H100, 6h).
- `--batch_size` is required, but `run.py` overrides it for every mode.

## Conventions and gotchas
- There are two output roots. `normalized-results/` is where `run.py` writes. `runs/` is where `load_logits` and the PyReFT/attribution/test-data modes read per-head scores and top-k CSVs. These have to be synced by hand.
- Most stages skip work when their output file already exists, so delete the outputs to force a rerun.
- Don't add explanatory comments alongside code edits in this repo.
- Known and suspected bugs in the `run.py` → ReFT path are tracked in `BUGS.md`.
