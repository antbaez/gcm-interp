#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Usage:
#   sbatch run_normal.sh --model <olmo|qwen|qwen3|gemma|llama|olmo,qwen,...|all> --dataset <harmful|sycophancy|verse|harmful,sycophancy,...|all> [--type "last mean positional"] [--nocache] [--unnormalized] [--judging]
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

EXTRA_FLAGS=()
[ -n "$TYPE_VAL" ]         && EXTRA_FLAGS+=(--type "$TYPE_VAL")
[ "$NOCACHE" = true ]      && EXTRA_FLAGS+=(--nocache)
[ "$UNNORMALIZED" = true ] && EXTRA_FLAGS+=(--unnormalized)
[ "$RESID" = true ]        && EXTRA_FLAGS+=(--resid)
[ "$RESID" = false ]       && EXTRA_FLAGS+=(--patch)

JUDGE_FLAGS=()
[ "$NOCACHE" = true ]     && JUDGE_FLAGS+=(--nocache)
[ "$UNNORMALIZED" = true ] && JUDGE_FLAGS+=(--unnormalized)
[ "$UNNORMALIZED" = false ] && JUDGE_FLAGS+=(--normalized)

echo "Running: model=$MODEL  dataset=$DATASET  type=${TYPE_VAL:-default}  nocache=$NOCACHE  unnormalized=$UNNORMALIZED  judging_only=$JUDGING_ONLY"

cd ~/gcm-interp

if [ "$JUDGING_ONLY" = false ]; then
    source ~/gcm-interp/setup/setup.sh
    bash run_steering.sh --model "$MODEL" --dataset "$DATASET" "${EXTRA_FLAGS[@]}"
fi

source ~/gcm-interp/setup/setup_judging.sh
bash run_judging.sh --model "$MODEL" --dataset "$DATASET" "${JUDGE_FLAGS[@]}"
