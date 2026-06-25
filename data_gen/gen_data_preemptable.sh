#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acbaez@mit.edu

# Usage:
#   sbatch gen_data_preemptable.sh --model <olmo|qwen|qwen3|gemma|llama|all> --dataset <harmful|sycophancy|verse|all> [--max_tokens <N>]
#   Defaults: --model all --dataset all

set -e

MODEL="all"
DATASET="all"
MAX_TOKENS="512"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Running: model=$MODEL  dataset=$DATASET"

source ~/gcm-interp/setup.sh
cd ~/gcm-interp

bash data_gen/gen_data.sh --model "$MODEL" --dataset "$DATASET" ${MAX_TOKENS:+--max_tokens "$MAX_TOKENS"}
