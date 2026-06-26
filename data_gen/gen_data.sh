#!/bin/bash
set -e

# Usage: ./gen_data.sh --model <olmo|qwen|qwen3|gemma|llama|olmo,qwen|all> --dataset <harmful|sycophancy|verse|sycophancy-haiku|sycophancy-poem|sycophancy-haiku-concise|sycophancy-poem-concise|all> [--device <cuda:0>] [--max_tokens <N>] [--batch_size <N>]
MODEL_TAG=""
DATASET_TAG=""
DEVICE="cuda:0"
MAX_TOKENS=512
BATCH_SIZE=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_TAG="$2";   shift 2 ;;
        --dataset)    DATASET_TAG="$2"; shift 2 ;;
        --device)     DEVICE="$2";      shift 2 ;;
        --max_tokens) MAX_TOKENS="$2";  shift 2 ;;
        --batch_size) BATCH_SIZE="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse" "sycophancy-haiku" "sycophancy-poem")

# Validate model tags
if [ "$MODEL_TAG" != "all" ]; then
    IFS=',' read -ra MODELS <<< "$MODEL_TAG"
    for M in "${MODELS[@]}"; do
        if [[ ! " ${ALL_MODELS[*]} " == *" $M "* ]]; then
            echo "Error: unknown model '$M'. Must be one of: ${ALL_MODELS[*]}, all"; exit 1
        fi
    done
fi

# Validate dataset tags
if [ "$DATASET_TAG" != "all" ]; then
    IFS=',' read -ra DATASETS <<< "$DATASET_TAG"
    for D in "${DATASETS[@]}"; do
        if [[ ! " ${ALL_DATASETS[*]} " == *" $D "* ]]; then
            echo "Error: unknown dataset '$D'. Must be one of: ${ALL_DATASETS[*]}, all"; exit 1
        fi
    done
fi

echo "Running: model=$MODEL_TAG  dataset=$DATASET_TAG"

python "$(dirname "$0")/gen_data.py" \
    --model      "$MODEL_TAG" \
    --dataset    "$DATASET_TAG" \
    --device     "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_tokens "$MAX_TOKENS"
