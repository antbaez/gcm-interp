#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err


set -e

MODEL="all"
DATASET="all"
JUDGING_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";        shift 2 ;;
        --dataset) DATASET="$2";      shift 2 ;;
        --judging) JUDGING_ONLY=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Running: model=$MODEL  dataset=$DATASET  judging_only=$JUDGING_ONLY"

cd ~/gcm-interp

if [ "$JUDGING_ONLY" = false ]; then
    source ~/gcm-interp/setup/setup.sh
    bash run_steering.sh --model "$MODEL" --dataset "$DATASET" --patch
fi

source ~/gcm-interp/setup/setup_judging.sh
bash run_judging.sh --model "$MODEL" --dataset "$DATASET"