#!/bin/bash
set -e

# Usage: bash mmlu/scripts/run_mmlu_seed_variance.sh --model <olmo|qwen3|gemma4|llama> \
#          [--dataset harmful] [--fractions 0.05,0.1,0.2,0.5] [--seeds 0,1,2,3,4] \
#          [--device cuda:0] [--batch_size 128] [--print_examples]
#
# Measures how much unsteered MMLU baseline accuracy varies purely from the
# random per-subject sampling in mmlu_scoring.sample_mmlu(), by running the
# baseline eval at each --fractions value with several different --mmlu_seed
# values. Steered configs are always skipped (an empty best_configs.json is
# passed) since only the baseline's sampling variance is in scope here.
#
# Each (fraction, seed) combo writes to its own results dir (so run_mmlu.py's
# own skip-if-exists baseline check can't collide across combos):
#   mmlu/results/seed_variance/frac=<f>/seed=<s>/<model_name>/baseline_mmlu_accuracy.json
# A skip check is duplicated here at the bash level too, so an already-done
# combo doesn't pay the cost of loading the model just to be told to skip.
#
# Afterwards, analyze_seed_variance.py aggregates all combos into a
# per-fraction mean/std/min/max table and a summary CSV.

MODEL_TAG=""
DATASET_TAG="harmful"
FRACTIONS="0.1"
SEEDS="0,1,2,3,4"
DEVICE="cuda:0"
# Controls MMLU eval batching (--mmlu_batch_size, what build_batches() actually
# uses); also forwarded as -batch_size to satisfy Config's required arg, which
# is otherwise unused on this code path.
BATCH_SIZE=2048
PRINT_EXAMPLES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_TAG="$2";   shift 2 ;;
        --dataset)    DATASET_TAG="$2"; shift 2 ;;
        --fractions)  FRACTIONS="$2";   shift 2 ;;
        --seeds)      SEEDS="$2";       shift 2 ;;
        --device)     DEVICE="$2";      shift 2 ;;
        --batch_size) BATCH_SIZE="$2";  shift 2 ;;
        --print_examples) PRINT_EXAMPLES=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PRINT_EXAMPLES_FLAG=()
if [ "$PRINT_EXAMPLES" = true ]; then
    PRINT_EXAMPLES_FLAG=(--print_examples)
fi

if [ -z "$MODEL_TAG" ]; then
    echo "Error: --model is required (olmo, qwen3, gemma4, llama)"; exit 1
fi

case "$MODEL_TAG" in
    olmo)   MODEL_ID="allenai/OLMo-2-1124-13B-DPO" ;;
    qwen3)  MODEL_ID="Qwen/Qwen3-14B" ;;
    gemma4) MODEL_ID="google/gemma-4-12B-it" ;;
    llama)  MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
    *) echo "Error: unknown model '$MODEL_TAG'. Must be one of: olmo, qwen3, gemma4, llama"; exit 1 ;;
esac

case "$DATASET_TAG" in
    harmful)    SOURCE="harmful-long";         BASE="harmless" ;;
    sycophancy) SOURCE="non-sycophantic-long"; BASE="sycophancy" ;;
    verse)      SOURCE="verse-long";           BASE="prose" ;;
    *) echo "Error: unknown dataset '$DATASET_TAG'. Must be one of: harmful, sycophancy, verse"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="$REPO_ROOT/mmlu/results/seed_variance"
MODEL_NAME="${MODEL_ID##*/}"

EMPTY_BEST_CONFIGS="$REPO_ROOT/mmlu/scripts/empty_best_configs.json"
echo '{}' > "$EMPTY_BEST_CONFIGS"

IFS=',' read -ra FRACTION_LIST <<< "$FRACTIONS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

for FRACTION in "${FRACTION_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
        OUT_DIR="$RESULTS_ROOT/frac=${FRACTION}/seed=${SEED}"
        BASELINE_FILE="$OUT_DIR/$MODEL_NAME/baseline_mmlu_accuracy.json"
        if [ -f "$BASELINE_FILE" ]; then
            echo "[frac=$FRACTION seed=$SEED] already exists, skipping."
            continue
        fi

        echo ""
        echo "[frac=$FRACTION seed=$SEED]"
        python -u "$REPO_ROOT/mmlu/run_mmlu.py" \
            -d "$DEVICE" \
            -model_id "$MODEL_ID" \
            -batch_size "$BATCH_SIZE" \
            -eval_model \
            -source "$SOURCE" \
            -base "$BASE" \
            --fraction "$FRACTION" \
            --mmlu_seed "$SEED" \
            --mmlu_batch_size "$BATCH_SIZE" \
            --best_configs "$EMPTY_BEST_CONFIGS" \
            --mmlu_data "$REPO_ROOT/mmlu/data/mmlu_test.jsonl" \
            --mmlu_results_dir "$OUT_DIR" \
            "${PRINT_EXAMPLES_FLAG[@]}"
    done
done

echo ""
echo "=== Variance across seeds, per fraction ==="
python3 "$REPO_ROOT/mmlu/analyze_seed_variance.py" \
    --results_root "$RESULTS_ROOT" \
    --model_id "$MODEL_ID" \
    --out_csv "$RESULTS_ROOT/${MODEL_NAME}_seed_variance_summary.csv"
