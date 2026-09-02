#!/bin/bash
set -e

# Usage: bash mmlu/scripts/run_mmlu_eval.sh --model <olmo|qwen3|gemma4|llama|all> [--dataset <harmful|sycophancy|verse|all>] [--fraction 0.1] [--global] [--split val|test] [--device <cuda:0>]
# --dataset defaults to "all" (harmful, sycophancy, verse).
# --model is restricted to the models actually present in judge-evals/best_configs.json
# (qwen/gemma have no pinned held-out configs, so there is nothing to score there).
# Scores steered configs for the given model x dataset combo, against a stratified,
# deterministic MMLU subsample (--fraction of questions per subject, fixed seed),
# plus one unsteered baseline per model. Always normalized.
# --split test (default) scores just the single best (stream, steering_type) config
# pinned in best_configs.json; --split val scores every (N, layer) condition from
# the validation sweep instead.

# Controls MMLU eval batching (--mmlu_batch_size, what build_batches() actually
# uses); also forwarded as -batch_size to satisfy Config's required arg, which
# is otherwise unused on this code path.
BATCH_SIZE=32
FRACTION=0.1
GLOBAL=false
SPLIT="test"

MODEL_TAG=""
DATASET_TAG="all"
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL_TAG="$2";   shift 2 ;;
        --dataset)  DATASET_TAG="$2"; shift 2 ;;
        --device)   DEVICE="$2";      shift 2 ;;
        --fraction) FRACTION="$2";    shift 2 ;;
        --global)   GLOBAL=true;      shift ;;
        --split)    SPLIT="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

GLOBAL_FLAG=""
if [ "$GLOBAL" = true ]; then GLOBAL_FLAG="--global"; fi

ALL_MODELS=("olmo" "qwen3" "gemma4" "llama")
ALL_DATASETS=("harmful" "sycophancy" "verse")

if [ "$MODEL_TAG" = "all" ]; then
    MODELS=("${ALL_MODELS[@]}")
else
    IFS=',' read -ra MODELS <<< "$MODEL_TAG"
    for M in "${MODELS[@]}"; do
        if [[ ! " ${ALL_MODELS[*]} " == *" $M "* ]]; then
            echo "Error: unknown model '$M'. Must be one of: olmo, qwen3, gemma4, llama, all"; exit 1
        fi
    done
fi

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

run_mmlu_for_model() {
    local M_TAG="$1"
    local D_TAG="$2"

    case "$M_TAG" in
        olmo)   MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
        qwen3)  MODEL_ID="Qwen/Qwen3-14B" ;;
        gemma4) MODEL_ID="google/gemma-4-12B-it" ;;
        llama)  MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    esac

    case "$D_TAG" in
        harmful)    SOURCE="harmful-long";         BASE="harmless" ;;
        sycophancy) SOURCE="non-sycophantic-long"; BASE="sycophancy" ;;
        verse)      SOURCE="verse-long";           BASE="prose" ;;
    esac

    echo ""
    echo "[$M_TAG/$D_TAG] fraction=$FRACTION global=$GLOBAL split=$SPLIT"
    local START_TIME=$SECONDS

    python -u mmlu/run_mmlu.py \
        -d "$DEVICE" \
        -model_id "$MODEL_ID" \
        -batch_size "$BATCH_SIZE" \
        --mmlu_batch_size "$BATCH_SIZE" \
        -eval_model \
        -source "$SOURCE" \
        -base "$BASE" \
        --fraction "$FRACTION" \
        --mmlu_batch_size "$BATCH_SIZE" \
        $GLOBAL_FLAG \
        --split "$SPLIT" \
        --best_configs judge-evals/best_configs.json \
        --mmlu_data mmlu/data/mmlu_test.jsonl \
        --mmlu_results_dir mmlu/results

    local ELAPSED=$(( SECONDS - START_TIME ))
    echo "[$M_TAG/$D_TAG] Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
}

for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        run_mmlu_for_model "$M" "$D"
    done
done
