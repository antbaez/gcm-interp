#!/bin/bash

# Usage: ./run_dynamic_steering.sh --model <olmo|qwen|solar|all> --dataset <harmful|sycophancy|verse|all>
MODEL_TAG=""
DATASET_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL_TAG="$2";   shift 2 ;;
        --dataset) DATASET_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_MODELS=("olmo" "qwen" "solar")
ALL_DATASETS=("harmful" "sycophancy" "verse")

# Expand model tag
if [ "$MODEL_TAG" = "all" ]; then
    MODELS=("${ALL_MODELS[@]}")
elif [[ " ${ALL_MODELS[*]} " == *" $MODEL_TAG "* ]]; then
    MODELS=("$MODEL_TAG")
else
    echo "Error: --model must be one of: olmo, qwen, solar, all"; exit 1
fi

# Expand dataset tag
if [ "$DATASET_TAG" = "all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
elif [[ " ${ALL_DATASETS[*]} " == *" $DATASET_TAG "* ]]; then
    DATASETS=("$DATASET_TAG")
else
    echo "Error: --dataset must be one of: harmful, sycophancy, verse, all"; exit 1
fi

DEVICE="cuda:0"
PATCHING_BATCH_SIZE=100 # number of prompts processed per forward pass during ATP patching
PATCH_ALGO="atp"
SEED=42
STEERING_PATCHING_BATCH_SIZE="10"  # number of prompts used per batch when computing the steering vector 
MAX_NEW_TOKENS=256      # max tokens the model generates per prompt during eval
MAX_NEW_TOKENS=16
STEERING_N="1 2 5 10"
STEERING_N="1"
TOPK_VALS="0.01 0.05 0.1 0.5"
TOPK_VALS="0.5"

PATCH_MODEL=true
EVAL_MODEL=true
STEERING=true
EVAL_TEST=false

COMBINATIONS=(
    "last_token last_token"
    "last_token all_tokens"
    "mean       last_token"
    "mean       all_tokens"
    "positional all_tokens"
)

EVAL_FLAGS=""
if [ "$EVAL_MODEL" = true ]; then EVAL_FLAGS="$EVAL_FLAGS -eval_model"; fi
if [ "$STEERING" = true ]; then EVAL_FLAGS="$EVAL_FLAGS --steering"; fi
if [ "$EVAL_TEST" = true ]; then EVAL_FLAGS="$EVAL_FLAGS --eval_test true"; else EVAL_FLAGS="$EVAL_FLAGS --eval_test false"; fi

run_experiment() {
    local M_TAG="$1"
    local D_TAG="$2"

    case "$M_TAG" in
        olmo)  MODEL_ID="allenai/OLMo-2-1124-13B-DPO";        MODEL_NAME="OLMo" ;;
        qwen)  MODEL_ID="Qwen/Qwen1.5-14B-Chat";              MODEL_NAME="Qwen" ;;
        solar) MODEL_ID="upstage/SOLAR-10.7B-Instruct-v1.0";  MODEL_NAME="Solar" ;;
    esac

    case "$D_TAG" in
        harmful)    SOURCE="harmful-long";    BASE="harmless" ;;
        sycophancy) SOURCE="sycophancy-long"; BASE="non-sycophantic" ;;
        verse)      SOURCE="verse-long";      BASE="prose" ;;
    esac

    STEERING_ADD="./dynamic-steering-data/${MODEL_ID##*/}/${SOURCE}/${SOURCE}-desired-all.jsonl"
    STEERING_SUB="./dynamic-steering-data/${MODEL_ID##*/}/${SOURCE}/${BASE}-desired-all.jsonl"

    BASE_ARGS=(
        -d "$DEVICE"
        -model_id "$MODEL_ID"
        -batch_size "$PATCHING_BATCH_SIZE"
        -source "$SOURCE"
        -base "$BASE"
        -patch_algo "$PATCH_ALGO"
        -seed "$SEED"
        -steering_add_path "$STEERING_ADD"
        -steering_sub_path "$STEERING_SUB"
        -steering_batch_size "$STEERING_PATCHING_BATCH_SIZE"
        -steering_n $STEERING_N
        -topk_vals $TOPK_VALS
        -max_new_tokens "$MAX_NEW_TOKENS"
    )

    python run.py "${BASE_ARGS[@]}" -patch_model $EVAL_FLAGS

    for COMBO in "${COMBINATIONS[@]}"; do
        ST=$(echo $COMBO | awk '{print $1}')
        SP=$(echo $COMBO | awk '{print $2}')
        echo "[$M_TAG/$D_TAG] Running eval: steering_type=$ST steering_pos=$SP"
        python run.py "${BASE_ARGS[@]}" -steering_type "$ST" -steering_pos "$SP" $EVAL_FLAGS
    done
}

for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        run_experiment "$M" "$D"
    done
done
