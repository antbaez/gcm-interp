#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUDGE_DIR="$SCRIPT_DIR/judge-evals"
RUNS_DIR="$SCRIPT_DIR/results"
JUDGE_OUT="$JUDGE_DIR"

if [ -f "$SCRIPT_DIR/.env" ]; then set -a; source "$SCRIPT_DIR/.env"; set +a; fi

BATCH_SIZE=128
EVAL_MODE=eval_test   # eval_train -> {base}-train.jsonl, eval_test -> {base}-valid.jsonl

# Usage: ./scripts/run_judging.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama|olmo,qwen|all> [--dataset <harmful|sycophancy|verse|harmful,sycophancy|all>] [--normalized|--unnormalized] [--results-dir <dir>] [--device <cuda:0>] [--force] [--harmonic] [--judge <openai-model>]
# --results-dir reads generations from <dir> and writes ratings/accuracies under <dir>/judge-evals/;
# omitted, it uses results/ and the repo's judge-evals/.
# --dataset defaults to "all" (harmful, sycophancy, verse). Dataset variants
# (VARIANT_DATASETS below, e.g. verse-varied) can be named explicitly but are
# never part of "all".
MODEL_TAG=""
DATASET_TAG="all"
DEVICE="cuda:0"
NORM_MODE=""
# Which split's generations to judge: val = <base>-valid, test = <base>-test.
SPLIT="val"
# By default, already-judged conditions (accuracy files present) are skipped.
# Pass --force to re-judge everything found regardless of existing accuracy files.
FORCE=false
# Pass --layer-range <start> <end> to judge only the single-layer sweep conditions at layers
# round(start * num_layers)..round(end * num_layers) - 1 (the same layers run_steering.sh --layer-range sweeps).
# Pass --harmonic to judge with HARMONIC_PROMPT_TEMPLATES (judge-evals/config.py).
HARMONIC=false
LAYER_RANGE=()
# Pass --type "<t1 t2 ...>" to judge only those steering types' generations (default: every type on disk).
TYPES=()
# Pass --judge <openai-model> to judge through the OpenAI API instead of the local judge (needs OPENAI_API_KEY);
# ratings and accuracies then go to <judge-evals>/judge=<openai-model>/, apart from the local judge's.
JUDGE_MODEL=""
API_CONCURRENCY=16
# Pass --layers "<l1 l2 ...>" to judge only those layers' single-layer conditions.
LAYERS=()
N_VALUES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL_TAG="$2";   shift 2 ;;
        --dataset)      DATASET_TAG="$2"; shift 2 ;;
        --device)       DEVICE="$2";      shift 2 ;;
        --normalized)   NORM_MODE="normalized";   shift ;;
        --unnormalized) NORM_MODE="unnormalized"; shift ;;
        --split)        SPLIT="$2";                shift 2 ;;
        --force)        FORCE=true;                shift ;;
        --harmonic)     HARMONIC=true;             shift ;;
        --results-dir)  RUNS_DIR="$(realpath -m "$2")"; JUDGE_OUT="$RUNS_DIR/judge-evals"; shift 2 ;;
        --layer-range)  LAYER_RANGE=("$2" "$3"); shift 3 ;;
        --type)         read -ra TYPES <<< "$2"; shift 2 ;;
        --judge)        JUDGE_MODEL="$2"; shift 2 ;;
        --layers)       read -ra LAYERS <<< "$2"; shift 2 ;;
        --n-values)     read -ra N_VALUES <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "$SPLIT" != "val" ] && [ "$SPLIT" != "test" ]; then
    echo "Error: --split must be 'val' or 'test' (got '$SPLIT')"; exit 1
fi

ACCURACY_SUBDIR="accuracy"
[ -n "$JUDGE_MODEL" ] && JUDGE_OUT="$JUDGE_OUT/judge=$JUDGE_MODEL"

if [ -z "$MODEL_TAG" ]; then
    echo "Usage: ./run_judge.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama|olmo,qwen|all> [--dataset <harmful|sycophancy|verse|harmful,sycophancy|all>] [--device <cuda:0>]"
    exit 1
fi

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma" "gemma4" "llama")
# Dataset tags come from utils/dataset_registry.py; extra ones are accepted by --dataset, excluded from "all".
read -ra ALL_DATASETS <<< "$(python3 "$SCRIPT_DIR/utils/dataset_registry.py" --base)"
read -ra VARIANT_DATASETS <<< "$(python3 "$SCRIPT_DIR/utils/dataset_registry.py" --extra)"

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
        if [[ ! " ${ALL_DATASETS[*]} ${VARIANT_DATASETS[*]} " == *" $D "* ]]; then
            echo "Error: unknown dataset '$D'. Must be one of: ${ALL_DATASETS[*]} ${VARIANT_DATASETS[*]} all"; exit 1
        fi
    done
fi

# Strip "cuda:" prefix for run_judge.py --device (expects an int)
DEVICE_IDX="${DEVICE#cuda:}"

for M_TAG in "${MODELS[@]}"; do
    case "$M_TAG" in
        olmo)  MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_ID="Qwen/Qwen1.5-14B-Chat" ;;
        qwen3) MODEL_ID="Qwen/Qwen3-14B" ;;
        gemma) MODEL_ID="google/gemma-3-12b-it" ;;
        gemma4) MODEL_ID="google/gemma-4-12B-it" ;;
        llama) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    esac
    MODEL_NAME="${MODEL_ID##*/}"

    for D_TAG in "${DATASETS[@]}"; do
        read -r SOURCE BASE _ <<< "$(python3 "$SCRIPT_DIR/utils/dataset_registry.py" "$D_TAG")"

        # The split's test-file stem is what distinguishes validation-sweep
        # generations from held-out ones inside the same results/ tree.
        if [ "$SPLIT" = "test" ]; then TEST_FILE="${BASE}-test"; else TEST_FILE="${BASE}-valid"; fi

        JUDGE_FLAGS=()
        [ -n "$NORM_MODE" ]  && JUDGE_FLAGS+=(--norm_mode  "$NORM_MODE")
        JUDGE_FLAGS+=(--test_file "$TEST_FILE")
        [ "$FORCE" = true ] && JUDGE_FLAGS+=(--force)
        [ "$HARMONIC" = true ] && JUDGE_FLAGS+=(--harmonic_prompts)
        [ ${#LAYER_RANGE[@]} -gt 0 ] && JUDGE_FLAGS+=(--layer_range "${LAYER_RANGE[@]}" --model_id "$MODEL_ID")
        [ ${#TYPES[@]} -gt 0 ] && JUDGE_FLAGS+=(--steering_type "${TYPES[@]}")
        [ ${#LAYERS[@]} -gt 0 ] && JUDGE_FLAGS+=(--layers "${LAYERS[@]}")
        [ ${#N_VALUES[@]} -gt 0 ] && JUDGE_FLAGS+=(--n_values "${N_VALUES[@]}")
        [ -n "$JUDGE_MODEL" ] && JUDGE_FLAGS+=(--judge_model "$JUDGE_MODEL" --api_concurrency "$API_CONCURRENCY")

        echo ""
        echo "[$M_TAG / $D_TAG] Judging model=$MODEL_NAME  source=$SOURCE  base=$BASE  norm=${NORM_MODE:-any}  split=$SPLIT ($TEST_FILE)  device=$DEVICE"
        cd "$JUDGE_DIR" && python run_judge.py \
            --model_name "$MODEL_NAME" \
            --source "$SOURCE" \
            --base "$BASE" \
            --runs_dir "$RUNS_DIR" \
            --data_dir "$SCRIPT_DIR/data" \
            --accuracy_dir "$JUDGE_OUT/$ACCURACY_SUBDIR" \
            --workdirs_root "$JUDGE_OUT/workdirs" \
            --batch_size "$BATCH_SIZE" \
            "${JUDGE_FLAGS[@]}"
        cd "$SCRIPT_DIR"
    done
done

if [ "$HARMONIC" = true ]; then
    exit 0
fi

echo ""
echo "Summarizing results..."
python "$JUDGE_DIR/summarize_results.py" \
    --accuracy_dir "$JUDGE_OUT/$ACCURACY_SUBDIR" \
    --workdirs_dir "$JUDGE_OUT/workdirs"
