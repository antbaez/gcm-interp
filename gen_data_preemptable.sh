#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acbaez@mit.edu

# Usage:
#   sbatch gen_data_preemptable.sh --model <olmo|qwen|solar|all> [--max_tokens <N>]
#   Defaults: --model all

set -e

MODEL="all"
MAX_TOKENS="512"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Running: model=$MODEL"

source ~/gcm-interp/setup.sh
cd ~/gcm-interp

bash gen_data.sh --model "$MODEL" ${MAX_TOKENS:+--max_tokens "$MAX_TOKENS"}
