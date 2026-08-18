#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Worker job: runs a single model x dataset combo. Submitted by
# mmlu/run_preemptable.sh (the dispatcher) — do not sbatch this directly with
# comma-separated or "all" model/dataset values, it will not fan out.
#
# Usage:
#   sbatch mmlu/scripts/run_preemptable_job.sh --model <olmo|qwen3|gemma4|llama> --dataset <harmful|sycophancy|verse|...> [--fraction 0.1]

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

echo "Running: model=$MODEL  dataset=$DATASET  fraction=$FRACTION"

cd ~/gcm-interp
source ~/gcm-interp/setup/setup.sh
bash mmlu/scripts/run_mmlu_eval.sh --model "$MODEL" --dataset "$DATASET" --fraction "$FRACTION"

echo ""
echo "Summarizing MMLU results..."
python mmlu/summarize_mmlu.py
