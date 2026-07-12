#!/bin/bash
# Dispatcher — run this with `bash`, NOT `sbatch` (it holds no GPU allocation
# itself). Expands --model/--dataset tags and submits one sbatch job per
# model x dataset combo via run_normal_job.sh.
#
# Usage:
#   bash run_normal.sh --model <olmo|qwen|qwen3|gemma|llama|olmo,qwen,...|all> --dataset <harmful|sycophancy|verse|harmful,sycophancy,...|all> [--type "last mean positional"] [--nocache] [--unnormalized] [--resid] [--judging]
#   Defaults: --model all --dataset all (uses steering types from run_steering.sh)

set -e

MODEL="all"
DATASET="all"
TYPE_VAL=""
NOCACHE=false
UNNORMALIZED=false
JUDGING_ONLY=false
RESID=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --type)         TYPE_VAL="$2";     shift 2 ;;
        --nocache)      NOCACHE=true;      shift ;;
        --unnormalized) UNNORMALIZED=true; shift ;;
        --judging)      JUDGING_ONLY=true; shift ;;
        --resid)        RESID=true;        shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse" "sycophancy-haiku" "sycophancy-poem" "sycophancy-haiku-concise" "sycophancy-poem-concise")

if [ "$MODEL" = "all" ]; then MODELS=("${ALL_MODELS[@]}"); else IFS=',' read -ra MODELS <<< "$MODEL"; fi
if [ "$DATASET" = "all" ]; then DATASETS=("${ALL_DATASETS[@]}"); else IFS=',' read -ra DATASETS <<< "$DATASET"; fi

JOB_FLAGS=()
[ -n "$TYPE_VAL" ]         && JOB_FLAGS+=(--type "$TYPE_VAL")
[ "$NOCACHE" = true ]      && JOB_FLAGS+=(--nocache)
[ "$UNNORMALIZED" = true ] && JOB_FLAGS+=(--unnormalized)
[ "$JUDGING_ONLY" = true ] && JOB_FLAGS+=(--judging)
[ "$RESID" = true ]        && JOB_FLAGS+=(--resid)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting $(( ${#MODELS[@]} * ${#DATASETS[@]} )) job(s) (one per model x dataset combo)..."
for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        sbatch "$SCRIPT_DIR/run_normal_job.sh" --model "$M" --dataset "$D" "${JOB_FLAGS[@]}"
    done
done
