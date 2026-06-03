#!/bin/bash

# Usage: ./run_patching.sh --model <olmo|qwen|solar|all> --dataset <harmful|sycophancy|verse|paragraph|all> [--device <cuda:0>]
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
ALL_DATASETS=("harmful" "sycophancy" "verse" "paragraph")

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
    echo "Error: --dataset must be one of: harmful, sycophancy, verse, paragraph, all"; exit 1
fi

PATCHING_BATCH_SIZE=100
PATCH_ALGO="atp"
SEED=42

run_patching() {
    local M_TAG="$1"
    local D_TAG="$2"

    case "$M_TAG" in
        olmo)  MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_ID="Qwen/Qwen1.5-14B-Chat" ;;
        solar) MODEL_ID="upstage/SOLAR-10.7B-Instruct-v1.0" ;;
    esac

    case "$D_TAG" in
        harmful)    SOURCE="harmful-long";    BASE="harmless" ;;
        sycophancy) SOURCE="sycophancy-long"; BASE="non-sycophantic" ;;
        verse)      SOURCE="verse-long";      BASE="prose" ;;
        paragraph)  SOURCE="paragraph-long";  BASE="sentence" ;;
    esac

    STEERING_ADD="./data/${MODEL_ID##*/}/${SOURCE}/${SOURCE}-desired-all.jsonl"
    STEERING_SUB="./data/${MODEL_ID##*/}/${SOURCE}/${BASE}-desired-all.jsonl"

    echo "[$M_TAG/$D_TAG] Patching model=$MODEL_ID source=$SOURCE base=$BASE"
    python run.py \
        -d "$DEVICE" \
        -model_id "$MODEL_ID" \
        -batch_size "$PATCHING_BATCH_SIZE" \
        -source "$SOURCE" \
        -base "$BASE" \
        -patch_algo "$PATCH_ALGO" \
        -seed "$SEED" \
        -steering_add_path "$STEERING_ADD" \
        -steering_sub_path "$STEERING_SUB" \
        -patch_model
}

for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        run_patching "$M" "$D"
    done
done
