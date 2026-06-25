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
#   sbatch run_jobs_preemptable.sh --model <olmo|qwen|qwen3|gemma|all> --dataset <harmful|sycophancy|verse|all> [--type last|positional] [--cache] [--judging]
#   Defaults: --model all --dataset all --type positional (no cache)

set -e

MODEL="all"
DATASET="all"
STEERING_TYPE="positional"
CACHE_FLAG=""
JUDGING_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";           shift 2 ;;
        --dataset) DATASET="$2";         shift 2 ;;
        --type)    STEERING_TYPE="$2";   shift 2 ;;
        --cache)   CACHE_FLAG="--cache"; shift ;;
        --judging) JUDGING_ONLY=true;    shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Running: model=$MODEL  dataset=$DATASET  type=$STEERING_TYPE  cache=${CACHE_FLAG:-none}  judging_only=$JUDGING_ONLY"

cd ~/gcm-interp

if [ "$JUDGING_ONLY" = false ]; then
    source ~/gcm-interp/setup/setup.sh
    bash run_steering.sh --model "$MODEL" --dataset "$DATASET" --type "$STEERING_TYPE" $CACHE_FLAG --patch
fi

source ~/gcm-interp/setup/setup_judging.sh
bash run_judging.sh --model "$MODEL" --dataset "$DATASET"
