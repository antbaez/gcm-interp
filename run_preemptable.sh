#!/bin/bash
# Dispatcher — run this with `bash`, NOT `sbatch` (it holds no GPU allocation
# itself). Reads a settings/*.yaml file and submits one sbatch job per
# model x dataset x scope combo listed there via scripts/run_preemptable_job.sh.
#
# Usage:
#   bash run_preemptable.sh --settings settings/<name>.yaml
#   --settings is required (e.g. settings/default.yaml). See that file for every available setting;
#   scripts/load_settings.py turns it into the job script's flags. Every output
#   lands under the settings' results_dir, which also gets a copy of the settings
#   file as settings.yaml (earlier, different copies are kept as settings.<time>.yaml).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --settings) SETTINGS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$SETTINGS" ]; then
    echo "Usage: bash run_preemptable.sh --settings settings/<name>.yaml"; exit 1
fi
if [ ! -f "$SETTINGS" ]; then
    echo "Error: settings file '$SETTINGS' not found"; exit 1
fi

LOADED="$(python3 "$SCRIPT_DIR/scripts/load_settings.py" "$SETTINGS" preemptable)"
eval "$LOADED"

mkdir -p "$RESULTS_DIR"
if [ -f "$RESULTS_DIR/settings.yaml" ] && ! cmp -s "$SETTINGS" "$RESULTS_DIR/settings.yaml"; then
    PREV="$RESULTS_DIR/settings.$(date -r "$RESULTS_DIR/settings.yaml" +%Y%m%d-%H%M%S).yaml"
    mv "$RESULTS_DIR/settings.yaml" "$PREV"
    echo "Note: $RESULTS_DIR had different settings; kept them as $PREV"
fi
cp "$SETTINGS" "$RESULTS_DIR/settings.yaml"

echo "Settings: $SETTINGS"
echo "Results dir: $RESULTS_DIR"
echo "Job flags: ${JOB_FLAGS[*]}"
echo "Scopes: ${SCOPES[*]}"
echo "Submitting $(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#SCOPES[@]} )) job(s) (one per model x dataset x scope combo)..."
for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        for S in "${SCOPES[@]}"; do
            declare -n SCOPE_FLAGS="SCOPE_FLAGS_$S" SCOPE_N="STEERING_N_$S"
            N_FLAGS=()
            if [ -n "${SCOPE_N[$M]:-}" ]; then N_FLAGS=(--steering-n "${SCOPE_N[$M]}"); fi
            JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/scripts/run_preemptable_job.sh" \
                --model "$M" --dataset "$D" "${JOB_FLAGS[@]}" "${SCOPE_FLAGS[@]}" "${N_FLAGS[@]}")
            echo "  $M / $D / $S: job $JOB_ID"
            unset -n SCOPE_FLAGS SCOPE_N
        done
    done
done
