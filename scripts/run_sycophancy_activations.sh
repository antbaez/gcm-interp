#!/bin/bash
set -euo pipefail

# Llama sycophancy at the harmonic-floor (N, layer) configs: held-out steered
# generation, activation capture, then 3D PCA plots. No judging.
#
#   1. run_steering.sh --split test for last/mean/positional, pinned from
#      best_configs_harmonic_floor.json -> results/.../*_gen.{json,txt}
#   2. analysis/capture_activations.py -> activations/<model>/sycophancy/
#   3. deletes the downloaded model weights (also on error/interrupt)
#   4. analysis/plot_activations_pca.py -> pca_layer<L>.html
#
# Usage: bash scripts/run_sycophancy_activations.sh [--device cuda:0] [--weights-dir <parent>]
# The weights (~16 GB) go to a fresh temp dir under --weights-dir (default $TMPDIR or /tmp).
# HF_TOKEN is read from /root/.env (override with ENV_FILE=...).

cd "$(dirname "$0")/.."

DEVICE="cuda:0"
WEIGHTS_PARENT="${TMPDIR:-/tmp}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)      DEVICE="$2";         shift 2 ;;
        --weights-dir) WEIGHTS_PARENT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

BEST_CONFIGS="best_configs_harmonic_floor.json"
OUT_DIR="activations/Llama-3.1-8B-Instruct/sycophancy"
LAYERS="14 15 20"

set -a; source "${ENV_FILE:-/root/.env}"; set +a
: "${HF_TOKEN:?HF_TOKEN not set (expected in ${ENV_FILE:-/root/.env})}"

mkdir -p "$WEIGHTS_PARENT"
WEIGHTS=$(mktemp -d "$WEIGHTS_PARENT/hf_weights_llama_XXXXXX")
RUN_CONFIGS=$(mktemp --suffix=.json)
delete_weights() {
    if [ -d "$WEIGHTS" ]; then
        rm -rf "$WEIGHTS"
        echo "Deleted downloaded model weights ($WEIGHTS)"
    fi
}
trap 'delete_weights; rm -f "$RUN_CONFIGS"' EXIT
export HF_HUB_CACHE="$WEIGHTS" HUGGINGFACE_HUB_CACHE="$WEIGHTS" TRANSFORMERS_CACHE="$WEIGHTS"

echo "=== 1/4 Steered generation (held-out split) ==="
python analysis/capture_activations.py --best-configs "$BEST_CONFIGS" --emit-run-configs "$RUN_CONFIGS"
bash scripts/run_steering.sh --model llama --dataset sycophancy --split test \
    --best-configs "$RUN_CONFIGS" --type "last mean positional" --device "$DEVICE"

echo "=== 2/4 Activation capture (layers $LAYERS) ==="
python -u analysis/capture_activations.py --best-configs "$BEST_CONFIGS" \
    --layers $LAYERS --device "$DEVICE" --out-dir "$OUT_DIR"

echo "=== 3/4 Deleting model weights ==="
delete_weights

echo "=== 4/4 PCA plots ==="
python -u analysis/plot_activations_pca.py --in-dir "$OUT_DIR"
