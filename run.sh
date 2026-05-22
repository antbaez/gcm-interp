#!/bin/bash

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
DEVICE="cuda:0"
BATCH_SIZE=100
SOURCE="lie"
BASE="truth"
PATCH_ALGO="atp"
SEED=42

PATCH_MODEL=true
EVAL_MODEL=false
STEERING=false

FLAGS=""
if [ "$PATCH_MODEL" = true ]; then FLAGS="$FLAGS -patch_model"; fi
if [ "$EVAL_MODEL" = true ]; then FLAGS="$FLAGS -eval_model"; fi
if [ "$STEERING" = true ]; then FLAGS="$FLAGS --steering"; fi

python run.py \
    -d "$DEVICE" \
    -model_id "$MODEL_ID" \
    -batch_size "$BATCH_SIZE" \
    -source "$SOURCE" \
    -base "$BASE" \
    -patch_algo "$PATCH_ALGO" \
    -seed "$SEED" \
    $FLAGS
