#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Worker job: runs a single model x dataset combo. Submitted by
# scripts/run_normal.sh (the dispatcher) — do not sbatch this directly with
# comma-separated or "all" model/dataset values, it will not fan out.
#
# Usage:
#   sbatch scripts/run_normal_job.sh --model <olmo|qwen|qwen3|gemma|gemma4|llama> --dataset <harmful|sycophancy|verse|...> [--type "last mean positional"] [--unnormalized] [--global] [--layer-frac <x> | --layer-range <start> <end>] [--split val|test] [--results-dir <dir>] [--no-generation] [--no-judging] [--no-model-cache] [--pass-rate] [--train-batch-size <n>] [--gen-batch-size <n>] [--steering-n "<N1 N2 ...>"]
#   With no --split, runs the whole pipeline: validation sweep, judging, selection,
#   then the held-out split if the selection changed. --split val stops after
#   judging the sweep; --split test selects from existing sweep results and
#   regenerates the held-out split only when the winning config moved.

set -e

MODEL="all"
DATASET="all"
TYPE_VAL=""
UNNORMALIZED=false
GENERATION=true
JUDGING=true
LAYER_FLAGS=()
# Settings results_dir: everything this job reads and writes lives under it.
# Omitted, the repo-level results/, judge-evals/ and stats/ trees are used.
RESULTS_DIR=""
GLOBAL=false
NO_MODEL_CACHE=false
HARMONIC=true
TRAIN_BATCH_SIZE=""
STEERING_N_VAL=""
GEN_BATCH_SIZE=""
# Which phases to run: "all" (default) does the validation sweep, selection, and
# the held-out split; "val" stops after judging the sweep; "test" selects from
# existing sweep results and regenerates the held-out split only when the winner
# moved. Selection lives in the test phase on purpose — "val" must never delete a
# held-out run as a side effect.
SPLIT="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --type)         TYPE_VAL="$2";     shift 2 ;;
        --unnormalized) UNNORMALIZED=true; shift ;;
        --no-generation) GENERATION=false; shift ;;
        --no-judging)   JUDGING=false;     shift ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --layer-frac)   LAYER_FLAGS=(--layer-frac "$2"); shift 2 ;;
        --layer-range)  LAYER_FLAGS=(--layer-range "$2" "$3"); shift 3 ;;
        --global)       GLOBAL=true;       shift ;;
        --split)        SPLIT="$2";        shift 2 ;;
        --no-model-cache) NO_MODEL_CACHE=true; shift ;;
        --harmonic)     HARMONIC=true;     shift ;;
        --pass-rate)    HARMONIC=false;    shift ;;
        --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --gen-batch-size)   GEN_BATCH_SIZE="$2";   shift 2 ;;
        --steering-n)       STEERING_N_VAL="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXTRA_FLAGS=()
[ -n "$TYPE_VAL" ]         && EXTRA_FLAGS+=(--type "$TYPE_VAL")
[ "$UNNORMALIZED" = true ] && EXTRA_FLAGS+=(--unnormalized)
[ "$GLOBAL" = true ]       && EXTRA_FLAGS+=(--global)
EXTRA_FLAGS+=("${LAYER_FLAGS[@]}")

RESULTS_FLAGS=()
JUDGE_OUT="judge-evals"
STATS_OUT="stats"
if [ -n "$RESULTS_DIR" ]; then
    RESULTS_FLAGS=(--results-dir "$RESULTS_DIR")
    JUDGE_OUT="$RESULTS_DIR/judge-evals"
    STATS_OUT="$RESULTS_DIR/stats"
fi
EXTRA_FLAGS+=("${RESULTS_FLAGS[@]}")
[ "$NO_MODEL_CACHE" = true ] && EXTRA_FLAGS+=(--no-model-cache)
[ "$HARMONIC" = false ]     && EXTRA_FLAGS+=(--pass-rate)
[ -n "$TRAIN_BATCH_SIZE" ]  && EXTRA_FLAGS+=(--train-batch-size "$TRAIN_BATCH_SIZE")
[ -n "$GEN_BATCH_SIZE" ]    && EXTRA_FLAGS+=(--gen-batch-size "$GEN_BATCH_SIZE")
[ -n "$STEERING_N_VAL" ]    && EXTRA_FLAGS+=(--steering-n "$STEERING_N_VAL")

JUDGE_FLAGS=("${RESULTS_FLAGS[@]}")
[ "$UNNORMALIZED" = true ] && JUDGE_FLAGS+=(--unnormalized)
[ "$UNNORMALIZED" = false ] && JUDGE_FLAGS+=(--normalized)
[ "$HARMONIC" = true ]     && JUDGE_FLAGS+=(--harmonic)

# Selection and the pass tables are per-scope: each has a separate workdir
# tree, so they must be selected and collected separately.
SCOPE="local"
[ "$GLOBAL" = true ] && SCOPE="global"

SELECT_FLAGS=("${RESULTS_FLAGS[@]}")
BEST_CONFIGS="$JUDGE_OUT/best_configs.json"
if [ "$HARMONIC" = true ]; then
    SELECT_FLAGS+=(--metric harmonic)
    BEST_CONFIGS="$JUDGE_OUT/best_configs_harmonic.json"
fi

echo "Running: model=$MODEL  dataset=$DATASET  results_dir=${RESULTS_DIR:-default}  type=${TYPE_VAL:-default}  unnormalized=$UNNORMALIZED  global=$GLOBAL  split=$SPLIT  layers=${LAYER_FLAGS[*]:-default}  generation=$GENERATION  judging=$JUDGING  no_model_cache=$NO_MODEL_CACHE  harmonic=$HARMONIC"

cd ~/gcm-interp

# --- Validation sweep: generate, then judge. Both skip work already on disk. ---
if [ "$SPLIT" = "all" ] || [ "$SPLIT" = "val" ]; then
    if [ "$GENERATION" = true ]; then
        source ~/gcm-interp/setup/setup.sh
        bash scripts/run_steering.sh --model "$MODEL" --dataset "$DATASET" "${EXTRA_FLAGS[@]}"
    fi

    if [ "$JUDGING" = true ]; then
        source ~/gcm-interp/setup/setup_judging.sh
        bash scripts/run_judging.sh --model "$MODEL" --dataset "$DATASET" "${JUDGE_FLAGS[@]}"
    fi
fi

if [ "$SPLIT" = "all" ] || [ "$SPLIT" = "test" ]; then
    # --- Select on validation, and drop any held-out run left at a superseded
    # config. PENDING_FILE lists combinations still missing a held-out run; it is
    # always created, so an empty file means there is nothing to generate.
    PENDING_FILE="$(mktemp)"
    trap 'rm -f "$PENDING_FILE"' EXIT

    echo ""
    source ~/gcm-interp/setup/setup_judging.sh
    echo "Selecting best validation config..."
    python judge-evals/select_best_config.py \
        --model "$MODEL" --dataset "$DATASET" --scope "$SCOPE" --pending-out "$PENDING_FILE" "${SELECT_FLAGS[@]}"

    # --- Held-out split, only when the selection moved or was never run.
    # Generation would skip on its own, but not before loading the model and
    # regenerating the unsteered baseline, so the check is worth making up front.
    if [ -s "$PENDING_FILE" ]; then
        echo ""
        echo "Held-out run needed for:"
        cat "$PENDING_FILE"

        if [ "$GENERATION" = true ]; then
            source ~/gcm-interp/setup/setup.sh
            bash scripts/run_steering.sh --model "$MODEL" --dataset "$DATASET" --split test --best-configs "$BEST_CONFIGS" "${EXTRA_FLAGS[@]}"
        fi

        if [ "$JUDGING" = true ]; then
            source ~/gcm-interp/setup/setup_judging.sh
            bash scripts/run_judging.sh --model "$MODEL" --dataset "$DATASET" --split test "${JUDGE_FLAGS[@]}"
        fi
    else
        echo ""
        echo "Held-out run already current for the selected config — skipping."
    fi

    # --- Pass table for this combination only, so concurrent jobs never write
    # the same file. run_mcnemar_test.py is deliberately not run here: it is
    # meant to be run by hand once every combination has finished.
    echo ""
    if [ "$JUDGING" = false ]; then
        echo "Judging disabled — skipping held-out scores."
    elif [ "$HARMONIC" = true ]; then
        echo "Computing held-out harmonic scores..."
        python calculate_harmonic_scores.py --root "$JUDGE_OUT/workdirs" --split test \
            --models "$MODEL" --datasets "$DATASET" --scope "$SCOPE" \
            --best-configs "$BEST_CONFIGS" \
            --out "$STATS_OUT/harmonic_scores/${MODEL}_${DATASET}_${SCOPE}_test.csv" \
            --json "${RESULTS_DIR:-results}/harmonic_scores.json"
    else
        echo "Collecting held-out pass rates..."
        python stats/collect_pass_rates.py --split test --model "$MODEL" --dataset "$DATASET" --scope "$SCOPE"
    fi
fi
