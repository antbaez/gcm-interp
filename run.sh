#!/bin/bash

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3.1-8B-Instruct"
DEVICE="cuda:0"
BATCH_SIZE=100
SOURCE="lie-long"
BASE="truth"
PATCH_ALGO="atp"
SEED=42
STEERING_ADD="./data/${MODEL_NAME}/${SOURCE}/${SOURCE}-desired-all.jsonl"
STEERING_SUB="./data/${MODEL_NAME}/${SOURCE}/${BASE}-desired-all.jsonl"
STEERING_BATCH_SIZE=10
STEERING_N="1"
# [1, 2, 4, 5, 6, 8, 10]
TOPK_VALS="0.05"
# [1.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5]

PATCH_MODEL=true
EVAL_MODEL=true
STEERING=true
EVAL_TEST=false

FLAGS=""
if [ "$PATCH_MODEL" = true ]; then FLAGS="$FLAGS -patch_model"; fi
if [ "$EVAL_MODEL" = true ]; then FLAGS="$FLAGS -eval_model"; fi
if [ "$STEERING" = true ]; then FLAGS="$FLAGS --steering"; fi
if [ "$EVAL_TEST" = true ]; then FLAGS="$FLAGS --eval_test true"; else FLAGS="$FLAGS --eval_test false"; fi

python run.py \
    -d "$DEVICE" \
    -model_id "$MODEL_ID" \
    -batch_size "$BATCH_SIZE" \
    -source "$SOURCE" \
    -base "$BASE" \
    -patch_algo "$PATCH_ALGO" \
    -seed "$SEED" \
    -steering_add_path "$STEERING_ADD" \
    -steering_sub_path "$STEERING_SUB" \
    -steering_batch_size "$STEERING_BATCH_SIZE" \
    -steering_n $STEERING_N \
    -topk_vals $TOPK_VALS \
    $FLAGS
