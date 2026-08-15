#!/bin/bash
set -e

# Usage: ./scripts/run_steering.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama|all> [--dataset <harmful|sycophancy|verse|all>] [--type last|positional|mean] [--attention] [--global] [--split val|test] [--device <cuda:0>]
# --dataset defaults to "all" (harmful, sycophancy, verse).
# --split val (default) sweeps N x layer on the validation split; --split test pins the
# selection from best_configs.json and generates once on the held-out test split.
# Residual-stream steering runs by default. Pass --attention for attention-head steering
# (reads whatever head-selection artifacts already exist under results/.../heads/).
# By default steering is a single-layer sweep over LAYER_RANGE_START..LAYER_RANGE_END
# of layers (below); --global steers all layers at once instead.

PATCHING_BATCH_SIZE=100
SEED=42
MAX_NEW_TOKENS=512
BATCH_SIZE=50
TOPK_VALS="1.0"

# Fraction of total layers (0.0-1.0) the single-layer sweep ranges over; ignored
# in --global mode. Defaults to the first two-thirds of layers (0.0, 0.667) —
# set to (0.333, 0.667) for the old middle-third sweep.
LAYER_RANGE_START=0.333
LAYER_RANGE_END=0.6667

# Per-model steering N sweep (edit each model's list independently).
# Local (single-layer sweep over LAYER_RANGE_START..LAYER_RANGE_END) and global
# (all layers at once) get independent lists — steering every layer usually needs
# different magnitudes than steering a single one. --global selects the GLOBAL list.
declare -A STEERING_N_LOCAL_BY_MODEL=(
    [olmo]="1 10 20 30 40 50 60 70 80 90 100"
    [qwen3]="1 25 50 75 100 125 150 175 200 225 250"
    [llama]="1 10 20 30 40 50 60 70 80 90 100"
    [gemma4]="1 10 25 50 75 100 125 150 175 200 225 250"
    [gemma]="1 500 1000 2000 3000 4000 5000"
)
declare -A STEERING_N_GLOBAL_BY_MODEL=(
    [olmo]="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2"
    [qwen3]="0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 5"
    [llama]="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2"
    [gemma]="0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 5"
    [gemma4]="0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 5"
)

EVAL_MODEL=true
STEERING=true

MODEL_TAG=""
DATASET_TAG="all"
DEVICE="cuda:0"
RESID=true
GLOBAL=false
# val: sweep N x layer on <base>-test.jsonl (the validation split).
# test: pin the config select_best_config.py chose on validation and generate
# once on <base>-heldout-test.jsonl, which the sweep never touched.
SPLIT="val"
BEST_CONFIGS="judge-evals/best_configs.json"
STEERING_TYPES=(
    last
    mean
    positional
)
NORMALIZE=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL_TAG="$2";      shift 2 ;;
        --dataset) DATASET_TAG="$2";    shift 2 ;;
        --device)  DEVICE="$2";         shift 2 ;;
        --type)    IFS=' ' read -ra STEERING_TYPES <<< "$2"; shift 2 ;;
        --unnormalized) NORMALIZE=false;   shift ;;
        --attention) RESID=false;       shift ;;
        --global)  GLOBAL=true;         shift ;;
        --seed)    SEED="$2";           shift 2 ;;
        --split)   SPLIT="$2";          shift 2 ;;
        --best-configs) BEST_CONFIGS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "$SPLIT" != "val" ] && [ "$SPLIT" != "test" ]; then
    echo "Error: --split must be 'val' or 'test' (got '$SPLIT')"; exit 1
fi
if [ "$SPLIT" = "test" ] && [ ! -f "$BEST_CONFIGS" ]; then
    echo "Error: --split test needs '$BEST_CONFIGS'. Run: python judge-evals/select_best_config.py"; exit 1
fi

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "gemma4" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse")

# Expand model tag (supports comma-separated values, e.g. "olmo,qwen,gemma")
if [ "$MODEL_TAG" = "all" ]; then
    MODELS=("${ALL_MODELS[@]}")
else
    IFS=',' read -ra MODELS <<< "$MODEL_TAG"
    for M in "${MODELS[@]}"; do
        if [[ ! " ${ALL_MODELS[*]} " == *" $M "* ]]; then
            echo "Error: unknown model '$M'. Must be one of: olmo, qwen, qwen3, gemma, gemma4, llama, all"; exit 1
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
if [ "$EVAL_MODEL" = true ];   then EVAL_FLAGS="$EVAL_FLAGS -eval_model"; fi
if [ "$STEERING" = true ];     then EVAL_FLAGS="$EVAL_FLAGS --steering"; fi
if [ "$NORMALIZE" = false ];   then EVAL_FLAGS="$EVAL_FLAGS --unnormalized"; fi
if [ "$RESID" = true ];        then EVAL_FLAGS="$EVAL_FLAGS --resid"; fi
if [ "$GLOBAL" = true ];       then EVAL_FLAGS="$EVAL_FLAGS --global"; fi
# In test mode run.py reads N/layer per dataset+steering type out of best_configs.json,
# so the swept -steering_n list passed below is ignored.
if [ "$SPLIT" = "test" ];      then EVAL_FLAGS="$EVAL_FLAGS -split test -best_configs $BEST_CONFIGS"; fi

run_experiments_for_model() {
    local M_TAG="$1"
    shift
    local D_TAGS=("$@")

    case "$M_TAG" in
        olmo)  MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_ID="Qwen/Qwen1.5-14B-Chat" ;;
        qwen3) MODEL_ID="Qwen/Qwen3-14B" ;;
        gemma) MODEL_ID="google/gemma-3-12b-it" ;;
        gemma4) MODEL_ID="google/gemma-4-12B-it" ;;
        llama) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    esac

    local STEERING_N
    if [ "$GLOBAL" = true ]; then
        STEERING_N="${STEERING_N_GLOBAL_BY_MODEL[$M_TAG]}"
    else
        STEERING_N="${STEERING_N_LOCAL_BY_MODEL[$M_TAG]}"
    fi
    # Not every model has both lists filled in. Catch it here rather than letting
    # run.py fail on an empty -steering_n, which reports itself as an argparse error.
    if [ -z "$STEERING_N" ]; then
        local WHICH_LIST="STEERING_N_LOCAL_BY_MODEL"
        [ "$GLOBAL" = true ] && WHICH_LIST="STEERING_N_GLOBAL_BY_MODEL"
        echo "Error: no N sweep defined for model '$M_TAG' in $WHICH_LIST (run_steering.sh). Add one."
        exit 1
    fi

    local SOURCES=() BASES=() DIRS=() ADD_PATHS=() SUB_PATHS=()
    for D_TAG in "${D_TAGS[@]}"; do
        case "$D_TAG" in
            harmful)                D_SOURCE="harmful-long";              D_BASE="harmless";               D_DIR="harmful-long" ;;
            sycophancy)             D_SOURCE="non-sycophantic-long";      D_BASE="sycophancy";             D_DIR="sycophancy-long" ;;
            verse)                  D_SOURCE="verse-long";                D_BASE="prose";                  D_DIR="verse-long" ;;
        esac
        SOURCES+=("$D_SOURCE")
        BASES+=("$D_BASE")
        DIRS+=("$D_DIR")
        ADD_PATHS+=("./data/${MODEL_ID##*/}/${D_DIR}/${D_SOURCE}-desired-all.jsonl")
        SUB_PATHS+=("./data/${MODEL_ID##*/}/${D_DIR}/${D_BASE}-desired-all.jsonl")
    done

    echo ""
    echo "[$M_TAG] split=$SPLIT  datasets=[${D_TAGS[*]}]  steering_types=[${STEERING_TYPES[*]}]  N=[$STEERING_N]  k=[$TOPK_VALS]"
    local START_TIME=$SECONDS

    python -u run.py \
        -d "$DEVICE" \
        -model_id "$MODEL_ID" \
        -batch_size "$PATCHING_BATCH_SIZE" \
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
        -layer_range_start "$LAYER_RANGE_START" \
        -layer_range_end "$LAYER_RANGE_END" \
        $EVAL_FLAGS

    local ELAPSED=$(( SECONDS - START_TIME ))
    echo "[$M_TAG] Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
}

for M in "${MODELS[@]}"; do
    run_experiments_for_model "$M" "${DATASETS[@]}"
done
