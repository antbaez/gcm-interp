#!/bin/bash
set -e

# Usage: ./gen_data.sh --model <olmo|qwen|qwen3|gemma|llama|olmo,qwen|all> --dataset <harmful|sycophancy|verse|all> [--device <cuda:0>] [--max_tokens <N>]
MODEL_TAG=""
DATASET_TAG=""
DEVICE="cuda:0"
MAX_TOKENS=512

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_TAG="$2";   shift 2 ;;
        --dataset)    DATASET_TAG="$2"; shift 2 ;;
        --device)     DEVICE="$2";      shift 2 ;;
        --max_tokens) MAX_TOKENS="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse")

# Expand model tag (supports comma-separated values, e.g. "olmo,qwen")
if [ "$MODEL_TAG" = "all" ]; then
    MODELS=("${ALL_MODELS[@]}")
else
    IFS=',' read -ra MODELS <<< "$MODEL_TAG"
    for M in "${MODELS[@]}"; do
        if [[ ! " ${ALL_MODELS[*]} " == *" $M "* ]]; then
            echo "Error: unknown model '$M'. Must be one of: olmo, qwen, qwen3, gemma, llama, all"; exit 1
        fi
    done
fi

# Expand dataset tag (supports comma-separated values, e.g. "harmful,verse")
if [ "$DATASET_TAG" = "all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    IFS=',' read -ra DATASETS <<< "$DATASET_TAG"
    for D in "${DATASETS[@]}"; do
        if [[ ! " ${ALL_DATASETS[*]} " == *" $D "* ]]; then
            echo "Error: unknown dataset '$D'. Must be one of: harmful, sycophancy, verse, all"; exit 1
        fi
    done
fi

BATCH_SIZE=50
NUM_SAMPLES=100

for M_TAG in "${MODELS[@]}"; do
    case "$M_TAG" in
        olmo)  MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_ID="Qwen/Qwen1.5-14B-Chat" ;;
        qwen3) MODEL_ID="Qwen/Qwen3-14B" ;;
        gemma) MODEL_ID="google/gemma-3-12b-it" ;;
        llama) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    esac

    echo "[${M_TAG}] Generating datasets: ${DATASET_TAG}..."
    python gen_data.py \
        --model_id    "$MODEL_ID" \
        --dataset     "$DATASET_TAG" \
        --num_samples "$NUM_SAMPLES" \
        --device      "$DEVICE" \
        --batch_size  "$BATCH_SIZE" \
        --max_tokens  "$MAX_TOKENS" \
        --gen_data
done

