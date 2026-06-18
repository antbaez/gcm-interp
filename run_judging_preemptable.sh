#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acbaez@mit.edu


# Usage:
#   sbatch run_jobs_preemptable.sh --model <olmo|qwen|solar|all> --dataset <harmful|sycophancy|verse|paragraph|all>
#   Defaults: --model all --dataset all

set -e

MODEL="all"
DATASET="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";   shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Running: model=$MODEL  dataset=$DATASET"

source ~/gcm-interp/setup_judging.sh
cd ~/gcm-interp

bash run_judging.sh  --model "$MODEL" --dataset "$DATASET"
