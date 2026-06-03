#!/bin/bash
set -e

# Usage: ./run_steering.sh --model <olmo|qwen|solar|all> --dataset <harmful|sycophancy|verse|all> [--device <cuda:0>]
MODEL_TAG=""
DATASET_TAG=""
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL_TAG="$2";   shift 2 ;;
        --dataset) DATASET_TAG="$2"; shift 2 ;;
        --device)  DEVICE="$2";      shift 2 ;;
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

PATCHING_BATCH_SIZE=100 # number of prompts processed per forward pass during ATP patching
PATCH_ALGO="atp"
SEED=42
STEERING_BATCH_SIZE="5"  # number of prompts used per batch when computing the steering vector
MAX_NEW_TOKENS=64      # max tokens the model generates per prompt during eval
STEERING_N="1 2 5 10"
STEERING_N="1 2"
TOPK_VALS="0.01 0.05 0.1 0.5"
TOPK_VALS="0.5 0.1"

EVAL_MODEL=true
STEERING=true
EVAL_TEST=false

COMBINATIONS=(
    "last-token last-token"
    "last-token all-tokens"
    "mean       last-token"
    "mean       all-tokens"
    "positional all-tokens"
)

EVAL_FLAGS=""
if [ "$EVAL_MODEL" = true ]; then EVAL_FLAGS="$EVAL_FLAGS -eval_model"; fi
if [ "$STEERING" = true ]; then EVAL_FLAGS="$EVAL_FLAGS --steering"; fi
if [ "$EVAL_TEST" = true ]; then EVAL_FLAGS="$EVAL_FLAGS --eval_test true"; else EVAL_FLAGS="$EVAL_FLAGS --eval_test false"; fi

run_experiments_for_model() {
    local M_TAG="$1"
    shift
    local D_TAGS=("$@")

    case "$M_TAG" in
        olmo)  MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_ID="Qwen/Qwen1.5-14B-Chat" ;;
        solar) MODEL_ID="upstage/SOLAR-10.7B-Instruct-v1.0" ;;
    esac

    # Build dataset_list JSON — all datasets for this model in one array
    local DS_JSON='['
    local FIRST=true
    for D_TAG in "${D_TAGS[@]}"; do
        case "$D_TAG" in
            harmful)    D_SOURCE="harmful-long";    D_BASE="harmless" ;;
            sycophancy) D_SOURCE="sycophancy-long"; D_BASE="non-sycophantic" ;;
            verse)      D_SOURCE="verse-long";      D_BASE="prose" ;;
        esac
        local SA="./data/${MODEL_ID##*/}/${D_SOURCE}/${D_SOURCE}-desired-all.jsonl"
        local SS="./data/${MODEL_ID##*/}/${D_SOURCE}/${D_BASE}-desired-all.jsonl"
        [ "$FIRST" = true ] && FIRST=false || DS_JSON="${DS_JSON},"
        DS_JSON="${DS_JSON}{\"source\":\"${D_SOURCE}\",\"base\":\"${D_BASE}\",\"steering_add\":\"${SA}\",\"steering_sub\":\"${SS}\"}"
    done
    DS_JSON="${DS_JSON}]"

    # Build steering combos JSON
    local COMBOS_JSON='['
    local CFIRST=true
    for COMBO in "${COMBINATIONS[@]}"; do
        local ST SP
        ST=$(echo $COMBO | awk '{print $1}')
        SP=$(echo $COMBO | awk '{print $2}')
        [ "$CFIRST" = true ] && CFIRST=false || COMBOS_JSON="${COMBOS_JSON},"
        COMBOS_JSON="${COMBOS_JSON}[\"${ST}\",\"${SP}\"]"
    done
    COMBOS_JSON="${COMBOS_JSON}]"

    BASE_ARGS=(
        -d "$DEVICE"
        -model_id "$MODEL_ID"
        -batch_size "$PATCHING_BATCH_SIZE"
        -patch_algo "$PATCH_ALGO"
        -seed "$SEED"
        -steering_batch_size "$STEERING_BATCH_SIZE"
        -steering_n $STEERING_N
        -topk_vals $TOPK_VALS
        -max_new_tokens "$MAX_NEW_TOKENS"
        -dataset_list "$DS_JSON"
    )

    echo ""
    echo "[$M_TAG / ${D_TAGS[*]}] Running ${#D_TAGS[@]} dataset(s) × ${#COMBINATIONS[@]} combos in single process"
    local START_TIME=$SECONDS
    python run.py "${BASE_ARGS[@]}" -steering_combos "$COMBOS_JSON" $EVAL_FLAGS
    local ELAPSED=$(( SECONDS - START_TIME ))
    echo "[$M_TAG] Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
}

for M in "${MODELS[@]}"; do
    run_experiments_for_model "$M" "${DATASETS[@]}"
done
