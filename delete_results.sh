#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage: ./delete_results.sh --model <olmo|qwen|qwen3|gemma|olmo,qwen|all> --dataset <harmful|sycophancy|verse|harmful,sycophancy|all> [--yes]
# Deletes the corresponding folders in results/, judge-evals/accuracy/ and judge-evals/workdirs/.
MODEL_TAG=""
DATASET_TAG=""
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL_TAG="$2";   shift 2 ;;
        --dataset) DATASET_TAG="$2"; shift 2 ;;
        --yes|-y)  ASSUME_YES=1;     shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_TAG" ] || [ -z "$DATASET_TAG" ]; then
    echo "Usage: ./delete_results.sh --model <olmo|qwen|qwen3|gemma|olmo,qwen|all> --dataset <harmful|sycophancy|verse|harmful,sycophancy|all> [--yes]"
    exit 1
fi

ALL_MODELS=("olmo" "qwen" "qwen3" "gemma")
ALL_DATASETS=("harmful" "sycophancy" "verse")

# Expand model tag (supports comma-separated values, e.g. "olmo,qwen")
if [ "$MODEL_TAG" = "all" ]; then
    MODELS=("${ALL_MODELS[@]}")
else
    IFS=',' read -ra MODELS <<< "$MODEL_TAG"
    for M in "${MODELS[@]}"; do
        if [[ ! " ${ALL_MODELS[*]} " == *" $M "* ]]; then
            echo "Error: unknown model '$M'. Must be one of: olmo, qwen, qwen3, gemma, all"; exit 1
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
            echo "Error: unknown dataset '$D'. Must be one of: harmful, sycophancy, verse, all"; exit 1
        fi
    done
fi

ROOT_DIRS=(
    "$SCRIPT_DIR/results"
    "$SCRIPT_DIR/judge-evals/accuracy"
    "$SCRIPT_DIR/judge-evals/workdirs"
)

# Build the list of folders to delete.
TARGETS=()
for M_TAG in "${MODELS[@]}"; do
    case "$M_TAG" in
        olmo)  MODEL_NAME="OLMo-2-1124-13B-DPO" ;;
        qwen)  MODEL_NAME="Qwen1.5-14B-Chat" ;;
        qwen3) MODEL_NAME="Qwen3-14B" ;;
        gemma) MODEL_NAME="gemma-3-12b-it" ;;
    esac

    for D_TAG in "${DATASETS[@]}"; do
        case "$D_TAG" in
            harmful)    SOURCE="harmful-long";         BASE="harmless" ;;
            sycophancy) SOURCE="non-sycophantic-long"; BASE="sycophancy" ;;
            verse)      SOURCE="verse-long";           BASE="prose" ;;

        esac

        REL="$MODEL_NAME/from_${SOURCE}_to_${BASE}"
        for ROOT in "${ROOT_DIRS[@]}"; do
            TARGETS+=("$ROOT/$REL")
        done
    done
done

# Show what exists and will be deleted.
echo "The following folders will be deleted:"
FOUND=0
for T in "${TARGETS[@]}"; do
    if [ -d "$T" ]; then
        echo "  [exists]   $T"
        FOUND=1
    else
        echo "  [missing]  $T"
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "Nothing to delete."
    exit 0
fi

if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Delete the existing folders listed above? [y/N] " REPLY
    case "$REPLY" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

for T in "${TARGETS[@]}"; do
    if [ -d "$T" ]; then
        rm -rf "$T"
        echo "Deleted $T"
    fi
done

echo "Done."
