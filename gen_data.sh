#!/bin/bash

# Usage: ./gen_data.sh --model <olmo|qwen|solar|all> [--max_tokens <N>]
MODEL_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_TAG="$2";  shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "$MODEL_TAG" = "all" ]; then
    for m in olmo qwen solar; do
        bash "$0" --model "$m" ${MAX_TOKENS:+--max_tokens "$MAX_TOKENS"} || exit 1
    done
    exit 0
fi

case "$MODEL_TAG" in
    olmo)  MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
    qwen)  MODEL_ID="Qwen/Qwen1.5-14B-Chat" ;;
    solar) MODEL_ID="upstage/SOLAR-10.7B-Instruct-v1.0" ;;
    *) echo "Error: --model must be one of: olmo, qwen, solar, all"; exit 1 ;;
esac

BASE_ID=""
SOURCE="non-sycophantic-long"
BASE="sycophancy-long"
NUM_SAMPLES=100
DEVICE="cuda:0"
BATCH_SIZE=50
GEN_DATA=true
GEN_LOGITS=false
MAX_TOKENS=512

# Build optional flags
FLAGS=""
if [ "$GEN_DATA" = true ]; then FLAGS="$FLAGS --gen_data"; fi

python gen_data.py \
    --model_id "$MODEL_ID" \
    --base_id "$BASE_ID" \
    --source "$SOURCE" \
    --base "$BASE" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_tokens "$MAX_TOKENS" \
    $FLAGS
