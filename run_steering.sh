#!/bin/bash
set -e

# Usage: ./run_steering.sh --model <olmo|qwen|qwen3|gemma|llama|all> --dataset <harmful|sycophancy|verse|all> [--type last|positional|mean] [--nocache] [--device <cuda:0>]

PATCHING_BATCH_SIZE=100
PATCH_ALGO="atp"
SEED=42
MAX_NEW_TOKENS=512
BATCH_SIZE=50
STEERING_N="1 3 5 8 10"
TOPK_VALS="0.01 0.03 0.05 0.08 0.1 0.5 1.0"

EVAL_MODEL=true
STEERING=true

MODEL_TAG=""
DATASET_TAG=""
DEVICE="cuda:0"
PATCH=true
RESID=false
STEERING_TYPES=(
    last
    mean
    positional
)
KV_CACHING=true
NORMALIZE=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL_TAG="$2";      shift 2 ;;
        --dataset) DATASET_TAG="$2";    shift 2 ;;
        --device)  DEVICE="$2";         shift 2 ;;
        --type)    IFS=' ' read -ra STEERING_TYPES <<< "$2"; shift 2 ;;
        --nocache)      KV_CACHING=false;  shift ;;
        --unnormalized) NORMALIZE=false;   shift ;;
        --patch)   PATCH=true;          shift ;;
        --resid)   RESID=true; PATCH=false; shift ;;
        --seed)    SEED="$2";           shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse" "sycophancy-haiku" "sycophancy-poem" "sycophancy-haiku-concise" "sycophancy-poem-concise")

# Expand model tag (supports comma-separated values, e.g. "olmo,qwen,gemma")
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

# Expand dataset tag (supports comma-separated values, e.g. "harmful,sycophancy")
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

if [ "$RESID" = true ]; then TOPK_VALS="1.0"; fi

EVAL_FLAGS=""
if [ "$PATCH" = true ];        then EVAL_FLAGS="$EVAL_FLAGS -patch_model"; fi
if [ "$EVAL_MODEL" = true ];   then EVAL_FLAGS="$EVAL_FLAGS -eval_model"; fi
if [ "$STEERING" = true ];     then EVAL_FLAGS="$EVAL_FLAGS --steering"; fi
if [ "$KV_CACHING" = true ];   then EVAL_FLAGS="$EVAL_FLAGS --kv_caching"; fi
if [ "$NORMALIZE" = false ];   then EVAL_FLAGS="$EVAL_FLAGS --unnormalized"; fi
if [ "$RESID" = true ];        then EVAL_FLAGS="$EVAL_FLAGS --resid"; fi

run_experiments_for_model() {
    local M_TAG="$1"
    shift
    local D_TAGS=("$@")

    case "$M_TAG" in
        olmo)  MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_ID="Qwen/Qwen1.5-14B-Chat" ;;
        qwen3) MODEL_ID="Qwen/Qwen3-14B" ;;
        gemma) MODEL_ID="google/gemma-3-12b-it" ;;
        llama) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    esac

    local SOURCES=() BASES=() DIRS=() ADD_PATHS=() SUB_PATHS=()
    for D_TAG in "${D_TAGS[@]}"; do
        case "$D_TAG" in
            harmful)                D_SOURCE="harmful-long";              D_BASE="harmless";               D_DIR="harmful-long" ;;
            sycophancy)             D_SOURCE="non-sycophantic-long";      D_BASE="sycophancy";             D_DIR="sycophancy-long" ;;
            verse)                  D_SOURCE="verse-long";                D_BASE="prose";                  D_DIR="verse-long" ;;
            sycophancy-haiku)       D_SOURCE="non-sycophantic-haiku-long"; D_BASE="sycophancy-haiku";        D_DIR="sycophancy-haiku-long" ;;
            sycophancy-poem)        D_SOURCE="non-sycophantic-poem-long";  D_BASE="sycophancy-poem";         D_DIR="sycophancy-poem-long" ;;
            sycophancy-haiku-concise) D_SOURCE="non-sycophantic-haiku-concise-long"; D_BASE="sycophancy-haiku-concise"; D_DIR="sycophancy-haiku-concise-long" ;;
            sycophancy-poem-concise)  D_SOURCE="non-sycophantic-poem-concise-long";  D_BASE="sycophancy-poem-concise";  D_DIR="sycophancy-poem-concise-long" ;;
        esac
        SOURCES+=("$D_SOURCE")
        BASES+=("$D_BASE")
        DIRS+=("$D_DIR")
        ADD_PATHS+=("./data/${MODEL_ID##*/}/${D_DIR}/${D_SOURCE}-desired-all.jsonl")
        SUB_PATHS+=("./data/${MODEL_ID##*/}/${D_DIR}/${D_BASE}-desired-all.jsonl")
    done

    echo ""
    echo "[$M_TAG] datasets=[${D_TAGS[*]}]  steering_types=[${STEERING_TYPES[*]}]  N=[$STEERING_N]  k=[$TOPK_VALS]"
    local START_TIME=$SECONDS

    python -u run.py \
        -d "$DEVICE" \
        -model_id "$MODEL_ID" \
        -batch_size "$PATCHING_BATCH_SIZE" \
        -patch_algo "$PATCH_ALGO" \
        -seed "$SEED" \
        -max_new_tokens "$MAX_NEW_TOKENS" \
        -source  "${SOURCES[@]}" \
        -base    "${BASES[@]}" \
        -source_dir "${DIRS[@]}" \
        -steering_add_path "${ADD_PATHS[@]}" \
        -steering_sub_path "${SUB_PATHS[@]}" \
        -eval_batch_size "$BATCH_SIZE" \
        -steering_n $STEERING_N \
        -topk_vals $TOPK_VALS \
        -steering_types "${STEERING_TYPES[@]}" \
        $EVAL_FLAGS

    local ELAPSED=$(( SECONDS - START_TIME ))
    echo "[$M_TAG] Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
}

for M in "${MODELS[@]}"; do
    run_experiments_for_model "$M" "${DATASETS[@]}"
done
