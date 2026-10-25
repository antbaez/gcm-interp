#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# llama, olmo, qwen3 on the first 5 sycophancy prompts.
# Usage: sbatch scripts/run_projection_all.sh

set -e
cd "$(dirname "$0")/.."
MODEL=llama
source setup/setup.sh

python project_activations.py --models llama,olmo,qwen3 --n-prompts 5 --out-dir projection_results
