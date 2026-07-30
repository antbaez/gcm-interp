#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGE_DIR="$SCRIPT_DIR/judge-evals"

BATCH_SIZE=128
EVAL_MODE=eval_test   # eval_train -> {base}-desired-all.jsonl, eval_test -> {base}-test.jsonl

# Usage: ./run_judging.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama|olmo,qwen|all> [--dataset <harmful|sycophancy|verse|harmful,sycophancy|all>] [--normalized|--unnormalized] [--cache|--nocache] [--attention] [--device <cuda:0>] [--force]
# --dataset defaults to "all" (harmful, sycophancy, verse).
# Residual-stream judging runs by default. Pass --attention to judge attention-head steering results instead.
MODEL_TAG=""
DATASET_TAG="all"
DEVICE="cuda:0"
NORM_MODE=""
CACHE_MODE=""
RESID=true
# Which split's generations to judge: val = <base>-test, test = <base>-heldout-test.
SPLIT="val"
# By default, already-judged conditions (accuracy files present) are skipped.
# Pass --force to re-judge everything found regardless of existing accuracy files.
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL_TAG="$2";   shift 2 ;;
        --dataset)      DATASET_TAG="$2"; shift 2 ;;
        --device)       DEVICE="$2";      shift 2 ;;
        --normalized)   NORM_MODE="normalized";   shift ;;
        --unnormalized) NORM_MODE="unnormalized"; shift ;;
        --cache)        CACHE_MODE="cache";        shift ;;
        --nocache)      CACHE_MODE="no_cache";     shift ;;
        --attention)    RESID=false;               shift ;;
        --split)        SPLIT="$2";                shift 2 ;;
        --force)        FORCE=true;                shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "$SPLIT" != "val" ] && [ "$SPLIT" != "test" ]; then
    echo "Error: --split must be 'val' or 'test' (got '$SPLIT')"; exit 1
fi

STREAM_MODE="residuals"
ACCURACY_SUBDIR="accuracy_residual"
if [ "$RESID" = false ]; then
    STREAM_MODE="attention"
    ACCURACY_SUBDIR="accuracy"
fi

if [ -z "$MODEL_TAG" ]; then
    echo "Usage: ./run_judge.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama|olmo,qwen|all> [--dataset <harmful|sycophancy|verse|harmful,sycophancy|all>] [--device <cuda:0>]"
    exit 1
fi

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "gemma4" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse")

# Expand model tag (supports comma-separated values, e.g. "olmo,qwen")
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

# Strip "cuda:" prefix for run_judge.py --device (expects an int)
DEVICE_IDX="${DEVICE#cuda:}"

for M_TAG in "${MODELS[@]}"; do
    case "$M_TAG" in
        olmo)  MODEL_NAME="OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_NAME="Qwen1.5-14B-Chat" ;;
        qwen3) MODEL_NAME="Qwen3-14B" ;;
        gemma) MODEL_NAME="gemma-3-12b-it" ;;
        gemma4) MODEL_NAME="gemma-4-12B-it" ;;
        llama) MODEL_NAME="Llama-3.1-8B-Instruct" ;;
    esac

    for D_TAG in "${DATASETS[@]}"; do
        case "$D_TAG" in
            harmful)                  SOURCE="harmful-long";               BASE="harmless";                DATA_SOURCE="$SOURCE"; DATA_BASE="$BASE" ;;
            sycophancy)               SOURCE="non-sycophantic-long";       BASE="sycophancy";              DATA_SOURCE="sycophancy-long"; DATA_BASE="sycophancy" ;;
            verse)                    SOURCE="verse-long";                 BASE="prose";                   DATA_SOURCE="$SOURCE"; DATA_BASE="$BASE" ;;
        esac

        # The split's test-file stem is what distinguishes validation-sweep
        # generations from held-out ones inside the same results/ tree.
        if [ "$SPLIT" = "test" ]; then TEST_FILE="${BASE}-heldout-test"; else TEST_FILE="${BASE}-test"; fi

        JUDGE_FLAGS=()
        [ -n "$NORM_MODE" ]  && JUDGE_FLAGS+=(--norm_mode  "$NORM_MODE")
        [ -n "$CACHE_MODE" ] && JUDGE_FLAGS+=(--cache_mode "$CACHE_MODE")
        JUDGE_FLAGS+=(--stream_mode "$STREAM_MODE")
        JUDGE_FLAGS+=(--test_file "$TEST_FILE")
        [ "$FORCE" = true ] && JUDGE_FLAGS+=(--force)

        echo ""
        echo "[$M_TAG / $D_TAG] Judging model=$MODEL_NAME  source=$SOURCE  base=$BASE  norm=${NORM_MODE:-any}  cache=${CACHE_MODE:-any}  stream=$STREAM_MODE  split=$SPLIT ($TEST_FILE)  device=$DEVICE"
        cd "$JUDGE_DIR" && python run_judge.py \
            --model_name "$MODEL_NAME" \
            --source "$SOURCE" \
            --base "$BASE" \
            --runs_dir "$SCRIPT_DIR/results" \
            --data_dir "$SCRIPT_DIR/data" \
            --accuracy_dir "$SCRIPT_DIR/judge-evals/$ACCURACY_SUBDIR" \
            --workdirs_root "$SCRIPT_DIR/judge-evals/workdirs" \
            --batch_size "$BATCH_SIZE" \
            "${JUDGE_FLAGS[@]}"
        cd "$SCRIPT_DIR"
    done
done

echo ""
echo "Summarizing results..."
python "$JUDGE_DIR/summarize_results.py" \
    --accuracy_dir "$SCRIPT_DIR/judge-evals/$ACCURACY_SUBDIR" \
    --workdirs_dir "$SCRIPT_DIR/judge-evals/workdirs" \
    --stream_mode "$STREAM_MODE"
