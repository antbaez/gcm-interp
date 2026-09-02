#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Worker job: runs a single model x dataset combo. Submitted by
# mmlu/run_normal.sh (the dispatcher) — do not sbatch this directly with
# comma-separated or "all" model/dataset values, it will not fan out.
#
# Usage:
#   sbatch mmlu/scripts/run_normal_job.sh --model <olmo|qwen3|gemma4|llama> --dataset <harmful|sycophancy|verse|...> [--fraction 0.1] [--global] [--split val|test]

set -e

MODEL="all"
DATASET="all"
FRACTION="0.1"
GLOBAL=false
SPLIT="test"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2";    shift 2 ;;
        --dataset)  DATASET="$2";  shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --global)   GLOBAL=true;   shift ;;
        --split)    SPLIT="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Running: model=$MODEL  dataset=$DATASET  fraction=$FRACTION  global=$GLOBAL  split=$SPLIT"

cd ~/gcm-interp
source ~/gcm-interp/setup/setup.sh
EXTRA_FLAGS=()
[ "$GLOBAL" = true ] && EXTRA_FLAGS+=(--global)

bash mmlu/scripts/run_mmlu_eval.sh --model "$MODEL" --dataset "$DATASET" --fraction "$FRACTION" --split "$SPLIT" "${EXTRA_FLAGS[@]}"

echo ""
echo "Summarizing MMLU results..."
python mmlu/summarize_mmlu.py
