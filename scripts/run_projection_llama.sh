#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Smoke test: llama only, one sycophancy prompt.
# Usage: sbatch scripts/run_projection_llama.sh

set -e
cd "$(dirname "$0")/.."
MODEL=llama
source setup/setup.sh

python project_activations.py --models llama --n-prompts 1 --out-dir projection_results/llama_1prompt
