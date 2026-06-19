#!/bin/bash
set -e

# Usage: ./run_steering.sh --model <olmo|qwen|solar|all> --dataset <harmful|sycophancy|verse|paragraph|all> [--device <cuda:0>]
MODEL_TAG=""
DATASET_TAG=""
DEVICE="cuda:0"
PATCH=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL_TAG="$2";   shift 2 ;;
        --dataset) DATASET_TAG="$2"; shift 2 ;;
        --device)  DEVICE="$2";      shift 2 ;;
        --patch)   PATCH=true;       shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_MODELS=("olmo" "qwen" "solar")
ALL_DATASETS=("harmful" "sycophancy" "verse" "paragraph")

# Expand model tag
if [ "$MODEL_TAG" = "all" ]; then
    MODELS=("${ALL_MODELS[@]}")
elif [[ " ${ALL_MODELS[*]} " == *" $MODEL_TAG "* ]]; then
    MODELS=("$MODEL_TAG")
else
    echo "Error: --model must be one of: olmo, qwen, solar, all"; exit 1
fi

# Expand dataset tag (supports comma-separated values, e.g. "harmful,sycophancy")
if [ "$DATASET_TAG" = "all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    IFS=',' read -ra DATASETS <<< "$DATASET_TAG"
    for D in "${DATASETS[@]}"; do
        if [[ ! " ${ALL_DATASETS[*]} " == *" $D "* ]]; then
            echo "Error: unknown dataset '$D'. Must be one of: harmful, sycophancy, verse, paragraph, all"; exit 1
        fi
    done
fi

PATCHING_BATCH_SIZE=100 # number of prompts processed per forward pass during ATP patching
PATCH_ALGO="atp"
SEED=42
MAX_NEW_TOKENS=512      # max tokens the model generates per prompt during eval      

BATCH_SIZE_HARMFUL=50
BATCH_SIZE_SYCOPHANCY=50
BATCH_SIZE_VERSE=50
BATCH_SIZE_PARAGRAPH=10
STEERING_N="1 3 5 8 10"
TOPK_VALS="0.01 0.03 0.05 0.08 0.1 0.5 1.0"


EVAL_MODEL=true
STEERING=true
EVAL_TEST=true

COMBINATIONS=(
    "last-token"
    "mean"
    "positional"
)

EVAL_FLAGS=""
if [ "$PATCH" = true ];       then EVAL_FLAGS="$EVAL_FLAGS -patch_model"; fi
if [ "$EVAL_MODEL" = true ];  then EVAL_FLAGS="$EVAL_FLAGS -eval_model"; fi
if [ "$STEERING" = true ];    then EVAL_FLAGS="$EVAL_FLAGS --steering"; fi
if [ "$EVAL_TEST" = true ];   then EVAL_FLAGS="$EVAL_FLAGS --eval_test true"; else EVAL_FLAGS="$EVAL_FLAGS --eval_test false"; fi
if [ "$EVAL_TRAIN" = true ];  then EVAL_FLAGS="$EVAL_FLAGS -eval_train"; fi

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
            harmful)    D_SOURCE="harmful-long";         D_BASE="harmless";  D_DIR="harmful-long" ;;
            sycophancy) D_SOURCE="non-sycophantic-long"; D_BASE="sycophancy"; D_DIR="sycophancy-long" ;;
            verse)      D_SOURCE="verse-long";           D_BASE="prose";     D_DIR="verse-long" ;;
            paragraph)  D_SOURCE="paragraph-long";       D_BASE="sentence";  D_DIR="paragraph-long" ;;
        esac
        local SA="./data/${MODEL_ID##*/}/${D_DIR}/${D_SOURCE}-desired-all.jsonl"
        local SS="./data/${MODEL_ID##*/}/${D_DIR}/${D_BASE}-desired-all.jsonl"
        local D_UPPER="${D_TAG^^}"
        local BS_VAR="BATCH_SIZE_${D_UPPER}"
        local SBS="${!BS_VAR}"
        local VBS="${!BS_VAR}"
        [ "$FIRST" = true ] && FIRST=false || DS_JSON="${DS_JSON},"
        DS_JSON="${DS_JSON}{\"source\":\"${D_SOURCE}\",\"base\":\"${D_BASE}\",\"dir\":\"${D_DIR}\",\"tag\":\"${D_TAG}\",\"steering_add\":\"${SA}\",\"steering_sub\":\"${SS}\",\"steering_batch_size\":${SBS},\"vector_creation_batch_size\":${VBS}}"
    done
    DS_JSON="${DS_JSON}]"

    # Build steering combos JSON
    local COMBOS_JSON='['
    local CFIRST=true
    for COMBO in "${COMBINATIONS[@]}"; do
        [ "$CFIRST" = true ] && CFIRST=false || COMBOS_JSON="${COMBOS_JSON},"
        COMBOS_JSON="${COMBOS_JSON}\"${COMBO}\""
    done
    COMBOS_JSON="${COMBOS_JSON}]"

    BASE_ARGS=(
        -d "$DEVICE"
        -model_id "$MODEL_ID"
        -batch_size "$PATCHING_BATCH_SIZE"
        -patch_algo "$PATCH_ALGO"
        -seed "$SEED"
        -steering_n $STEERING_N
        -topk_vals $TOPK_VALS
        -max_new_tokens "$MAX_NEW_TOKENS"
        -dataset_list "$DS_JSON"
    )

    read -ra _N_ARR <<< "$STEERING_N"
    read -ra _K_ARR <<< "$TOPK_VALS"
    local N_COMBOS=$(( ${#_N_ARR[@]} * ${#_K_ARR[@]} ))
    echo ""
    echo "[$M_TAG / ${D_TAGS[*]}] Running ${#D_TAGS[@]} dataset(s) × ${N_COMBOS} combinations (N=[${STEERING_N}]  k=[${TOPK_VALS}]) in single process"
    local START_TIME=$SECONDS
    python -u run.py "${BASE_ARGS[@]}" -steering_combos "$COMBOS_JSON" $EVAL_FLAGS
    local ELAPSED=$(( SECONDS - START_TIME ))
    echo "[$M_TAG] Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
}

for M in "${MODELS[@]}"; do
    run_experiments_for_model "$M" "${DATASETS[@]}"
done
