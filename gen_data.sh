#!/bin/bash

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
BASE_ID=""
SOURCE="hate"
BASE="love"
NUM_SAMPLES=100
DEVICE="cuda:0"
BATCH_SIZE=100
GEN_DATA=true
GEN_LOGITS=false

# Build optional flags
FLAGS=""
if [ "$GEN_DATA" = true ]; then FLAGS="$FLAGS --gen_data"; fi
if [ "$GEN_LOGITS" = true ]; then FLAGS="$FLAGS --gen_logits"; fi

python gen_data.py \
    --model_id "$MODEL_ID" \
    --base_id "$BASE_ID" \
    --source "$SOURCE" \
    --base "$BASE" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    $FLAGS
