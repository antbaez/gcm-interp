#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Worker job: runs a single model x dataset combo. Submitted by
# run_preemptable.sh (the dispatcher) — do not sbatch this directly with
# comma-separated or "all" model/dataset values, it will not fan out.
#
# Usage:
#   sbatch run_preemptable_job.sh --model <olmo|qwen|qwen3|gemma|llama> --dataset <harmful|sycophancy|verse|...> [--type "last mean positional"] [--nocache] [--unnormalized] [--resid] [--judging]

set -e

MODEL="all"
DATASET="all"
TYPE_VAL=""
NOCACHE=false
UNNORMALIZED=false
JUDGING_ONLY=false
RESID=false
SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --type)         TYPE_VAL="$2";     shift 2 ;;
        --nocache)      NOCACHE=true;      shift ;;
        --unnormalized) UNNORMALIZED=true; shift ;;
        --judging)      JUDGING_ONLY=true; shift ;;
        --resid)        RESID=true;        shift ;;
        --seed)         SEED="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXTRA_FLAGS=()
[ -n "$TYPE_VAL" ]         && EXTRA_FLAGS+=(--type "$TYPE_VAL")
[ "$NOCACHE" = true ]      && EXTRA_FLAGS+=(--nocache)
[ "$UNNORMALIZED" = true ] && EXTRA_FLAGS+=(--unnormalized)
[ "$RESID" = true ]        && EXTRA_FLAGS+=(--resid)
[ "$RESID" = false ]       && EXTRA_FLAGS+=(--patch)
[ -n "$SEED" ]             && EXTRA_FLAGS+=(--seed "$SEED")

JUDGE_FLAGS=()
[ "$NOCACHE" = true ]      && JUDGE_FLAGS+=(--nocache)
[ "$UNNORMALIZED" = true ] && JUDGE_FLAGS+=(--unnormalized)
[ "$UNNORMALIZED" = false ] && JUDGE_FLAGS+=(--normalized)
[ "$RESID" = true ]         && JUDGE_FLAGS+=(--resid)

echo "Running: model=$MODEL  dataset=$DATASET  type=${TYPE_VAL:-default}  nocache=$NOCACHE  unnormalized=$UNNORMALIZED  resid=$RESID  judging_only=$JUDGING_ONLY"

cd ~/gcm-interp

if [ "$JUDGING_ONLY" = false ]; then
    source ~/gcm-interp/setup/setup.sh
    bash run_steering.sh --model "$MODEL" --dataset "$DATASET" "${EXTRA_FLAGS[@]}"
fi

source ~/gcm-interp/setup/setup_judging.sh
bash run_judging.sh --model "$MODEL" --dataset "$DATASET" "${JUDGE_FLAGS[@]}"
