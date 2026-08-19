#!/bin/bash
# Dispatcher — run this with `bash`, NOT `sbatch` (it holds no GPU allocation
# itself). Expands --model/--dataset tags and submits one sbatch job per
# model x dataset combo via scripts/run_preemptable_job.sh.
#
# Usage:
#   bash run_preemptable.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama|olmo,qwen,...|all> --dataset <harmful|sycophancy|verse|harmful,sycophancy,...|all> [--type "last mean positional"] [--unnormalized] [--attention] [--global] [--split val|test] [--judging] [--seed N]
#   Defaults: --model all --dataset all (uses steering types from scripts/run_steering.sh)
#   Note: Residual-stream steering runs by default. Pass --attention to steer the attention
#   output (self_attn.o_proj.output) instead; the two streams differ only in hook site.

set -e

MODEL="all"
DATASET="all"
TYPE_VAL=""
UNNORMALIZED=false
JUDGING_ONLY=false
RESID=true
GLOBAL=false
SEED=""
# Which phases to run. Omitted (the default "all") runs the validation sweep,
# selection, and the held-out split in one go. "val" stops after judging the
# sweep; "test" selects from whatever sweep results already exist and only
# regenerates the held-out split when the winning config moved.
SPLIT="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --type)         TYPE_VAL="$2";     shift 2 ;;
        --unnormalized) UNNORMALIZED=true; shift ;;
        --judging)      JUDGING_ONLY=true; shift ;;
        --attention)    RESID=false;       shift ;;
        --global)       GLOBAL=true;       shift ;;
        --seed)         SEED="$2";         shift 2 ;;
        --split)        SPLIT="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "$SPLIT" != "all" ] && [ "$SPLIT" != "val" ] && [ "$SPLIT" != "test" ]; then
    echo "Error: --split must be 'val' or 'test' (got '$SPLIT'); omit it to run both"; exit 1
fi

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "gemma4" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse")

if [ "$MODEL" = "all" ]; then MODELS=("${ALL_MODELS[@]}"); else IFS=',' read -ra MODELS <<< "$MODEL"; fi
if [ "$DATASET" = "all" ]; then DATASETS=("${ALL_DATASETS[@]}"); else IFS=',' read -ra DATASETS <<< "$DATASET"; fi

JOB_FLAGS=()
[ -n "$TYPE_VAL" ]         && JOB_FLAGS+=(--type "$TYPE_VAL")
[ "$UNNORMALIZED" = true ] && JOB_FLAGS+=(--unnormalized)
[ "$JUDGING_ONLY" = true ] && JOB_FLAGS+=(--judging)
[ "$RESID" = false ]       && JOB_FLAGS+=(--attention)
[ "$GLOBAL" = true ]       && JOB_FLAGS+=(--global)
[ -n "$SEED" ]             && JOB_FLAGS+=(--seed "$SEED")
[ "$SPLIT" != "all" ]      && JOB_FLAGS+=(--split "$SPLIT")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting $(( ${#MODELS[@]} * ${#DATASETS[@]} )) job(s) (one per model x dataset combo)..."
for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        sbatch "$SCRIPT_DIR/scripts/run_preemptable_job.sh" --model "$M" --dataset "$D" "${JOB_FLAGS[@]}"
    done
done
