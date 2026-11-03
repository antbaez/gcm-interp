#!/bin/bash
# Dispatcher — run this with `bash`, NOT `sbatch` (it holds no GPU allocation
# itself). Reads a settings/*.yaml file and submits one sbatch job per
# model x dataset x scope combo listed there via scripts/run_normal_job.sh.
#
# Usage:
#   bash run_normal.sh --settings settings/<name>.yaml
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
    echo "Usage: bash run_normal.sh --settings settings/<name>.yaml"; exit 1
fi
if [ ! -f "$SETTINGS" ]; then
    echo "Error: settings file '$SETTINGS' not found"; exit 1
fi

if [ -f "$SCRIPT_DIR/.env" ]; then set -a; source "$SCRIPT_DIR/.env"; set +a; fi

LOADED="$(python3 "$SCRIPT_DIR/scripts/load_settings.py" "$SETTINGS" normal)"
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
echo "Normalizations: ${NORMS[*]}"
if [ ${#LAYER_CHUNKS[@]} -gt 0 ]; then
    echo "Local sweep covers every layer: split into ${#LAYER_CHUNKS[@]} validation jobs per combo, then one held-out job after them"
fi
echo "Submitting job(s) per model x dataset x scope x normalization x steering-type group..."
for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        for S in "${SCOPES[@]}"; do
            for NM in "${NORMS[@]}"; do
                declare -n SCOPE_FLAGS="SCOPE_FLAGS_$S" NORM_FLAGS="NORM_FLAGS_$NM" SCOPE_N="STEERING_N_${S}_$NM"
                declare -n TYPE_GROUPS="TYPE_GROUPS_$NM"
                for G in "${TYPE_GROUPS[@]}"; do
                    TYPE_FLAGS=()
                    if [ -n "$G" ]; then TYPE_FLAGS=(--type "$G"); fi
                    N_FLAGS=()
                    if [ -n "${SCOPE_N[$M]:-}" ]; then N_FLAGS=(--steering-n "${SCOPE_N[$M]}"); fi
                    if [ "$S" = local ] && [ ${#LAYER_CHUNKS[@]} -gt 0 ] && [ "$SPLIT" != test ]; then
                        VAL_IDS=()
                        for C in "${LAYER_CHUNKS[@]}"; do
                            read -r C_START C_END <<< "$C"
                            JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/scripts/run_normal_job.sh" \
                                --model "$M" --dataset "$D" "${JOB_FLAGS[@]}" "${NORM_FLAGS[@]}" "${TYPE_FLAGS[@]}" --layer-range "$C_START" "$C_END" "${N_FLAGS[@]}" --split val)
                            echo "  $M / $D / $S / $NM / ${G:-default types} layers [$C_START, $C_END): job $JOB_ID"
                            VAL_IDS+=("$JOB_ID")
                        done
                        if [ "$SPLIT" = all ]; then
                            JOB_ID=$(sbatch --parsable --dependency="afterok:$(IFS=:; echo "${VAL_IDS[*]}")" --kill-on-invalid-dep=yes \
                                "$SCRIPT_DIR/scripts/run_normal_job.sh" \
                                --model "$M" --dataset "$D" "${JOB_FLAGS[@]}" "${NORM_FLAGS[@]}" "${TYPE_FLAGS[@]}" "${SCOPE_FLAGS[@]}" "${N_FLAGS[@]}" --split test)
                            echo "  $M / $D / $S / $NM / ${G:-default types} held-out (after ${VAL_IDS[*]}): job $JOB_ID"
                        fi
                    else
                        JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/scripts/run_normal_job.sh" \
                            --model "$M" --dataset "$D" "${JOB_FLAGS[@]}" "${NORM_FLAGS[@]}" "${TYPE_FLAGS[@]}" "${SCOPE_FLAGS[@]}" "${N_FLAGS[@]}")
                        echo "  $M / $D / $S / $NM / ${G:-default types}: job $JOB_ID"
                    fi
                done
                unset -n SCOPE_FLAGS NORM_FLAGS SCOPE_N TYPE_GROUPS
            done
        done
    done
done
