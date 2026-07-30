#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Worker job: runs a single model x dataset combo. Submitted by
# run_preemptable.sh (the dispatcher) — do not sbatch this directly with
# comma-separated or "all" model/dataset values, it will not fan out.
#
# Usage:
#   sbatch scripts/run_preemptable_job.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama> --dataset <harmful|sycophancy|verse|...> [--type "last mean positional"] [--nocache] [--unnormalized] [--attention] [--global] [--split val|test] [--judging]
#   Note: Residual-stream steering runs by default. Pass --attention for attention-head steering
#   (reads whatever head-selection artifacts already exist under results/.../heads/).
#   With no --split, runs the whole pipeline: validation sweep, judging, selection,
#   then the held-out split if the selection changed. --split val stops after
#   judging the sweep; --split test selects from existing sweep results and
#   regenerates the held-out split only when the winning config moved.

set -e

MODEL="all"
DATASET="all"
TYPE_VAL=""
NOCACHE=false
UNNORMALIZED=false
JUDGING_ONLY=false
RESID=true
GLOBAL=false
# Which phases to run: "all" (default) does the validation sweep, selection, and
# the held-out split; "val" stops after judging the sweep; "test" selects from
# existing sweep results and regenerates the held-out split only when the winner
# moved. Selection lives in the test phase on purpose — "val" must never delete a
# held-out run as a side effect.
SPLIT="all"
SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --type)         TYPE_VAL="$2";     shift 2 ;;
        --nocache)      NOCACHE=true;      shift ;;
        --unnormalized) UNNORMALIZED=true; shift ;;
        --judging)      JUDGING_ONLY=true; shift ;;
        --attention)    RESID=false;       shift ;;
        --global)       GLOBAL=true;       shift ;;
        --split)        SPLIT="$2";        shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXTRA_FLAGS=()
[ -n "$TYPE_VAL" ]         && EXTRA_FLAGS+=(--type "$TYPE_VAL")
[ "$NOCACHE" = true ]      && EXTRA_FLAGS+=(--nocache)
[ "$UNNORMALIZED" = true ] && EXTRA_FLAGS+=(--unnormalized)
[ "$RESID" = false ]       && EXTRA_FLAGS+=(--attention)
[ "$GLOBAL" = true ]       && EXTRA_FLAGS+=(--global)
[ -n "$SEED" ]             && EXTRA_FLAGS+=(--seed "$SEED")

JUDGE_FLAGS=()
[ "$NOCACHE" = true ]      && JUDGE_FLAGS+=(--nocache)
[ "$UNNORMALIZED" = true ] && JUDGE_FLAGS+=(--unnormalized)
[ "$UNNORMALIZED" = false ] && JUDGE_FLAGS+=(--normalized)
[ "$RESID" = false ]        && JUDGE_FLAGS+=(--attention)

echo "Running: model=$MODEL  dataset=$DATASET  type=${TYPE_VAL:-default}  nocache=$NOCACHE  unnormalized=$UNNORMALIZED  resid=$RESID  global=$GLOBAL  split=$SPLIT  judging_only=$JUDGING_ONLY"

cd ~/gcm-interp

# --- Validation sweep: generate, then judge. Both skip work already on disk. ---
if [ "$SPLIT" = "all" ] || [ "$SPLIT" = "val" ]; then
    if [ "$JUDGING_ONLY" = false ]; then
        source ~/gcm-interp/setup/setup.sh
        bash scripts/run_steering.sh --model "$MODEL" --dataset "$DATASET" "${EXTRA_FLAGS[@]}"
    fi

    source ~/gcm-interp/setup/setup_judging.sh
    bash scripts/run_judging.sh --model "$MODEL" --dataset "$DATASET" "${JUDGE_FLAGS[@]}"
fi

if [ "$SPLIT" = "all" ] || [ "$SPLIT" = "test" ]; then
    # --- Select on validation, and drop any held-out run left at a superseded
    # config. PENDING_FILE lists combinations still missing a held-out run; it is
    # always created, so an empty file means there is nothing to generate.
    PENDING_FILE="$(mktemp)"
    trap 'rm -f "$PENDING_FILE"' EXIT

    echo ""
    echo "Selecting best validation config..."
    python judge-evals/select_best_config.py \
        --model "$MODEL" --dataset "$DATASET" --pending-out "$PENDING_FILE"

    # --- Held-out split, only when the selection moved or was never run.
    # Generation would skip on its own, but not before loading the model and
    # regenerating the unsteered baseline, so the check is worth making up front.
    if [ -s "$PENDING_FILE" ]; then
        echo ""
        echo "Held-out run needed for:"
        cat "$PENDING_FILE"

        source ~/gcm-interp/setup/setup.sh
        bash scripts/run_steering.sh --model "$MODEL" --dataset "$DATASET" --split test "${EXTRA_FLAGS[@]}"

        source ~/gcm-interp/setup/setup_judging.sh
        bash scripts/run_judging.sh --model "$MODEL" --dataset "$DATASET" --split test "${JUDGE_FLAGS[@]}"
    else
        echo ""
        echo "Held-out run already current for the selected config — skipping."
    fi

    # --- Pass table for this combination only, so concurrent jobs never write
    # the same file. run_mcnemar_test.py is deliberately not run here: it is
    # meant to be run by hand once every combination has finished.
    echo ""
    echo "Collecting held-out pass rates..."
    python stats/collect_pass_rates.py --split test --model "$MODEL" --dataset "$DATASET"
fi
