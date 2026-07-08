#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGE_DIR="$SCRIPT_DIR/judge-evals"

BATCH_SIZE=128
EVAL_MODE=eval_test   # eval_train -> {base}-desired-all.jsonl, eval_test -> {base}-test.jsonl

# Usage: ./run_judging.sh --model <olmo|qwen|qwen3|gemma|llama|olmo,qwen|all> --dataset <harmful|sycophancy|verse|harmful,sycophancy|all> [--normalized|--unnormalized] [--cache|--nocache] [--device <cuda:0>]
MODEL_TAG=""
DATASET_TAG=""
DEVICE="cuda:0"
NORM_MODE=""
CACHE_MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL_TAG="$2";   shift 2 ;;
        --dataset)      DATASET_TAG="$2"; shift 2 ;;
        --device)       DEVICE="$2";      shift 2 ;;
        --normalized)   NORM_MODE="normalized";   shift ;;
        --unnormalized) NORM_MODE="unnormalized"; shift ;;
        --cache)        CACHE_MODE="cache";        shift ;;
        --nocache)      CACHE_MODE="no_cache";     shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_TAG" ] || [ -z "$DATASET_TAG" ]; then
    echo "Usage: ./run_judge.sh --model <olmo|qwen|qwen3|gemma|llama|olmo,qwen|all> --dataset <harmful|sycophancy|verse|harmful,sycophancy|all> [--device <cuda:0>]"
    exit 1
fi

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse" "sycophancy-haiku" "sycophancy-poem" "sycophancy-haiku-concise" "sycophancy-poem-concise")

# Expand model tag (supports comma-separated values, e.g. "olmo,qwen")
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

# Strip "cuda:" prefix for run_judge.py --device (expects an int)
DEVICE_IDX="${DEVICE#cuda:}"

for M_TAG in "${MODELS[@]}"; do
    case "$M_TAG" in
        olmo)  MODEL_NAME="OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_NAME="Qwen1.5-14B-Chat" ;;
        qwen3) MODEL_NAME="Qwen3-14B" ;;
        gemma) MODEL_NAME="gemma-3-12b-it" ;;
        llama) MODEL_NAME="Llama-3.1-8B-Instruct" ;;
    esac

    for D_TAG in "${DATASETS[@]}"; do
        case "$D_TAG" in
            harmful)                  SOURCE="harmful-long";               BASE="harmless";                DATA_SOURCE="$SOURCE"; DATA_BASE="$BASE" ;;
            sycophancy)               SOURCE="non-sycophantic-long";       BASE="sycophancy";              DATA_SOURCE="sycophancy-long"; DATA_BASE="sycophancy" ;;
            verse)                    SOURCE="verse-long";                 BASE="prose";                   DATA_SOURCE="$SOURCE"; DATA_BASE="$BASE" ;;
            sycophancy-haiku)         SOURCE="non-sycophantic-haiku-long"; BASE="sycophancy-haiku";        DATA_SOURCE="sycophancy-haiku-long"; DATA_BASE="$BASE" ;;
            sycophancy-poem)          SOURCE="non-sycophantic-poem-long";  BASE="sycophancy-poem";         DATA_SOURCE="sycophancy-poem-long"; DATA_BASE="$BASE" ;;
            sycophancy-haiku-concise) SOURCE="non-sycophantic-haiku-concise-long"; BASE="sycophancy-haiku-concise"; DATA_SOURCE="sycophancy-haiku-concise-long"; DATA_BASE="$BASE" ;;
            sycophancy-poem-concise)  SOURCE="non-sycophantic-poem-concise-long";  BASE="sycophancy-poem-concise";  DATA_SOURCE="sycophancy-poem-concise-long"; DATA_BASE="$BASE" ;;

        esac

        JUDGE_FLAGS=()
        [ -n "$NORM_MODE" ]  && JUDGE_FLAGS+=(--norm_mode  "$NORM_MODE")
        [ -n "$CACHE_MODE" ] && JUDGE_FLAGS+=(--cache_mode "$CACHE_MODE")

        echo ""
        echo "[$M_TAG / $D_TAG] Judging model=$MODEL_NAME  source=$SOURCE  base=$BASE  norm=${NORM_MODE:-any}  cache=${CACHE_MODE:-any}  device=$DEVICE"
        cd "$JUDGE_DIR" && python run_judge.py \
            --model_name "$MODEL_NAME" \
            --source "$SOURCE" \
            --base "$BASE" \
            --runs_dir "$SCRIPT_DIR/results" \
            --data_dir "$SCRIPT_DIR/data" \
            --accuracy_dir "$SCRIPT_DIR/judge-evals/accuracy" \
            --workdirs_root "$SCRIPT_DIR/judge-evals/workdirs" \
            --batch_size "$BATCH_SIZE" \
            --force \
            "${JUDGE_FLAGS[@]}"
        cd "$SCRIPT_DIR"
    done
done

echo ""
echo "Summarizing results..."
python "$JUDGE_DIR/summarize_results.py" --accuracy_dir "$SCRIPT_DIR/judge-evals/accuracy"
