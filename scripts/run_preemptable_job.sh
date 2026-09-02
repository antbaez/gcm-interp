#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Worker job: runs a single model x dataset combo. Submitted by
# run_preemptable.sh (the dispatcher) — do not sbatch this directly with
# comma-separated or "all" model/dataset values, it will not fan out.
#
# Usage:
#   sbatch scripts/run_preemptable_job.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama> --dataset <harmful|sycophancy|verse|...> [--type "last mean positional"] [--unnormalized] [--attention] [--global] [--split val|test] [--judging] [--no-model-cache]
#   Note: Residual-stream steering runs by default. Pass --attention to steer the attention
#   output (self_attn.o_proj.output) instead; the two streams differ only in hook site.
#   With no --split, runs the whole pipeline: validation sweep, judging, selection,
#   then the held-out split if the selection changed. --split val stops after
#   judging the sweep; --split test selects from existing sweep results and
#   regenerates the held-out split only when the winning config moved.

set -e

MODEL="all"
DATASET="all"
TYPE_VAL=""
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
NO_MODEL_CACHE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --type)         TYPE_VAL="$2";     shift 2 ;;
        --unnormalized) UNNORMALIZED=true; shift ;;
        --judging)      JUDGING_ONLY=true; shift ;;
        --attention)    RESID=false;       shift ;;
        --global)       GLOBAL=true;       shift ;;
        --split)        SPLIT="$2";        shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --no-model-cache) NO_MODEL_CACHE=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXTRA_FLAGS=()
[ -n "$TYPE_VAL" ]         && EXTRA_FLAGS+=(--type "$TYPE_VAL")
[ "$UNNORMALIZED" = true ] && EXTRA_FLAGS+=(--unnormalized)
[ "$RESID" = false ]       && EXTRA_FLAGS+=(--attention)
[ "$GLOBAL" = true ]       && EXTRA_FLAGS+=(--global)
[ -n "$SEED" ]             && EXTRA_FLAGS+=(--seed "$SEED")
[ "$NO_MODEL_CACHE" = true ] && EXTRA_FLAGS+=(--no-model-cache)

JUDGE_FLAGS=()
[ "$UNNORMALIZED" = true ] && JUDGE_FLAGS+=(--unnormalized)
[ "$UNNORMALIZED" = false ] && JUDGE_FLAGS+=(--normalized)
[ "$RESID" = false ]        && JUDGE_FLAGS+=(--attention)

# Selection and the pass tables are per-stream and per-scope: each has a
# separate workdir tree, so they must be selected and collected separately.
STREAM="residuals"
[ "$RESID" = false ] && STREAM="attention"
SCOPE="local"
[ "$GLOBAL" = true ] && SCOPE="global"

echo "Running: model=$MODEL  dataset=$DATASET  type=${TYPE_VAL:-default}  unnormalized=$UNNORMALIZED  resid=$RESID  global=$GLOBAL  split=$SPLIT  judging_only=$JUDGING_ONLY  no_model_cache=$NO_MODEL_CACHE"

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
        --model "$MODEL" --dataset "$DATASET" --stream "$STREAM" --scope "$SCOPE" --pending-out "$PENDING_FILE"

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
    python stats/collect_pass_rates.py --split test --model "$MODEL" --dataset "$DATASET" --stream "$STREAM" --scope "$SCOPE"
fi
