#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/out/%j.out
#SBATCH --error=logs/err/%j.err

# Fits every Jacobian variant and its steering directions (jacobian/fit_lens.py), from scratch on each
# submission; a requeued job resumes from its checkpoint.
#
# Usage (from the repo root):
#   sbatch jacobian/run_fit_lens.sh [--model llama] [--dataset verse] [--layer 21] [--max-new-tokens 256]

cd ~/gcm-interp
source ~/gcm-interp/setup/setup.sh
echo "Running: jacobian/fit_lens.py $*"
python jacobian/fit_lens.py "$@"
