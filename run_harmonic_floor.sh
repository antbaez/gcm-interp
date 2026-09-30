#!/bin/bash
# One-shot submitter for the harmonic-floor held-out runs — run with `bash`,
# not `sbatch`. Submits harmful/sycophancy/verse first, then the dataset
# variants with an afterok dependency on every one of those jobs, so the
# variants only start once all the parent jobs have finished successfully.
#
# Usage:
#   bash run_harmonic_floor.sh [--partition normal|preemptable|both] [--gpu <type>]
#   --partition defaults to both. --gpu (e.g. a100) overrides the job scripts'
#   GPU type (h200 on normal, h100 on preemptable).

set -e

PARTITION="both"
GPU=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --gpu)       GPU="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "$PARTITION" in
    normal)      DISPATCHERS=(run_normal.sh) ;;
    preemptable) DISPATCHERS=(run_preemptable.sh) ;;
    both)        DISPATCHERS=(run_normal.sh run_preemptable.sh) ;;
    *) echo "Error: --partition must be normal, preemptable, or both"; exit 1 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")"

# Held-out-only jobs take ~20-30 min and <10 GB RAM on ~1 busy core, far below
# the job scripts' full-pipeline defaults (12h / 100G / 8 CPUs); asking for less
# lets them backfill sooner.
COMMON=(--model olmo,qwen3,llama --split test --type "last mean-padding positional" --harmonic-floor
        --sbatch-extra "--time=04:00:00 --mem=32G -c 4${GPU:+ --gres=gpu:$GPU:1}")
PARENTS="harmful,sycophancy,verse"
VARIANTS="sycophancy-unaligned,sycophancy-more-templates,verse-aligned,verse-longer,verse-varied"

IDS_FILE="$(mktemp)"
trap 'rm -f "$IDS_FILE"' EXIT

for DISPATCHER in "${DISPATCHERS[@]}"; do
    bash "$DISPATCHER" --dataset "$PARENTS" "${COMMON[@]}" --jobid-file "$IDS_FILE"
done

if [ ! -s "$IDS_FILE" ]; then
    echo "Error: no parent jobs were submitted; not submitting the variants"; exit 1
fi
DEP="afterok:$(paste -sd: "$IDS_FILE")"
echo "Variants will wait on: $DEP"

for DISPATCHER in "${DISPATCHERS[@]}"; do
    bash "$DISPATCHER" --dataset "$VARIANTS" "${COMMON[@]}" --dependency "$DEP"
done
