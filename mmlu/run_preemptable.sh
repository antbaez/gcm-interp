#!/bin/bash
# Dispatcher — run this with `bash`, NOT `sbatch` (it holds no GPU allocation
# itself). Expands --model/--dataset tags and submits one sbatch job per
# model x dataset combo via mmlu/scripts/run_preemptable_job.sh.
#
# Usage:
#   bash mmlu/run_preemptable.sh --model <olmo|qwen3|gemma4|llama|olmo,qwen3,...|all> --dataset <harmful|sycophancy|verse|harmful,sycophancy,...|all> [--fraction 0.1]
#   Defaults: --model all --dataset all --fraction 0.1
#   --model is restricted to the models with pinned held-out configs in
#   judge-evals/best_configs.json (qwen/gemma have none).
#   --fraction is the fraction of MMLU questions sampled per subject (stratified,
#   deterministic seed), forwarded all the way down to mmlu/run_mmlu.py.

set -e

MODEL="all"
DATASET="all"
FRACTION="0.1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2";    shift 2 ;;
        --dataset)  DATASET="$2";  shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_MODELS=("olmo" "qwen3" "gemma4" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse")

if [ "$MODEL" = "all" ]; then MODELS=("${ALL_MODELS[@]}"); else IFS=',' read -ra MODELS <<< "$MODEL"; fi
if [ "$DATASET" = "all" ]; then DATASETS=("${ALL_DATASETS[@]}"); else IFS=',' read -ra DATASETS <<< "$DATASET"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting $(( ${#MODELS[@]} * ${#DATASETS[@]} )) job(s) (one per model x dataset combo)..."
for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        sbatch "$SCRIPT_DIR/scripts/run_preemptable_job.sh" --model "$M" --dataset "$D" --fraction "$FRACTION"
    done
done
