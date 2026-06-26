#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

set -e
cd ~/gcm-interp
source ~/gcm-interp/setup/setup.sh
python chat_template_experiments/verify_qwen3_tokenizer.py
