#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGE_DIR="$SCRIPT_DIR/judge-evals"

# Usage: ./run_judging.sh --model <olmo|qwen|solar|olmo,qwen|all> --dataset <harmful|sycophancy|verse|paragraph|harmful,sycophancy|all> [--device <cuda:0>]
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

if [ -z "$MODEL_TAG" ] || [ -z "$DATASET_TAG" ]; then
    echo "Usage: ./run_judge.sh --model <olmo|qwen|solar|olmo,qwen|all> --dataset <harmful|sycophancy|verse|paragraph|harmful,sycophancy|all> [--device <cuda:0>]"
    exit 1
fi

ALL_MODELS=("olmo" "qwen" "solar")
ALL_DATASETS=("harmful" "sycophancy" "verse" "paragraph")

# Expand model tag (supports comma-separated values, e.g. "olmo,qwen")
if [ "$MODEL_TAG" = "all" ]; then
    MODELS=("${ALL_MODELS[@]}")
else
    IFS=',' read -ra MODELS <<< "$MODEL_TAG"
    for M in "${MODELS[@]}"; do
        if [[ ! " ${ALL_MODELS[*]} " == *" $M "* ]]; then
            echo "Error: unknown model '$M'. Must be one of: olmo, qwen, solar, all"; exit 1
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
            echo "Error: unknown dataset '$D'. Must be one of: harmful, sycophancy, verse, paragraph, all"; exit 1
        fi
    done
fi

# Strip "cuda:" prefix for run_judge.py --device (expects an int)
DEVICE_IDX="${DEVICE#cuda:}"

BATCH_SIZE=64
EVAL_MODE=eval_test   # eval_train -> {base}-desired-all.jsonl, eval_test -> {base}-test.jsonl

for M_TAG in "${MODELS[@]}"; do
    case "$M_TAG" in
        olmo)  MODEL_NAME="OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_NAME="Qwen1.5-14B-Chat" ;;
        solar) MODEL_NAME="SOLAR-10.7B-Instruct-v1.0" ;;
    esac

    for D_TAG in "${DATASETS[@]}"; do
        case "$D_TAG" in
            harmful)    SOURCE="harmful-long";    BASE="harmless" ;;
            sycophancy) SOURCE="sycophancy-long"; BASE="non-sycophantic" ;;
            verse)      SOURCE="verse-long";      BASE="prose" ;;
            paragraph)  SOURCE="paragraph-long";  BASE="sentence" ;;
        esac

        echo ""
        echo "[$M_TAG / $D_TAG] model=$MODEL_NAME  source=$SOURCE  base=$BASE  device=$DEVICE"
        cd "$JUDGE_DIR" && python run_judge.py \
            --model_name "$MODEL_NAME" \
            --source "$SOURCE" \
            --base "$BASE" \
            --runs_dir "$SCRIPT_DIR/results" \
            --data_dir "$SCRIPT_DIR/data" \
            --accuracy_dir "$SCRIPT_DIR/judge-evals/accuracy" \
            --workdirs_root "$SCRIPT_DIR/judge-evals/workdirs" \
            --eval_mode "$EVAL_MODE" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE_IDX"
    done
done

echo ""
echo "Summarizing results..."
python "$JUDGE_DIR/summarize_results.py" --accuracy_dir "$SCRIPT_DIR/judge-evals/accuracy"
