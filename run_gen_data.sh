#!/bin/bash
# Usage: bash run_gen_data.sh --n|--p [--model <olmo|qwen|qwen3|gemma|llama|all>] [--dataset <harmful|sycophancy|verse|all>] [--max_tokens <N>]
# Defaults: --model all --dataset all --max_tokens 512

QUEUE=""
MODEL="all"
DATASET="all"
MAX_TOKENS="512"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)          QUEUE="normal";      shift ;;
        --p)          QUEUE="preemptable"; shift ;;
        --model)      MODEL="$2";          shift 2 ;;
        --dataset)    DATASET="$2";        shift 2 ;;
        --max_tokens) MAX_TOKENS="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$QUEUE" ]; then
    echo "Error: specify --n (normal) or --p (preemptable)"
    exit 1
fi

if [ "$QUEUE" = "normal" ]; then
    PARTITION="mit_normal_gpu"
    TIME="06:00:00"
    REQUEUE=""
else
    PARTITION="mit_preemptable"
    TIME="24:00:00"
    REQUEUE="#SBATCH --requeue"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=$TIME
$REQUEUE
#SBATCH --output=$HOME/gcm-interp/logs/out/%j.out
#SBATCH --error=$HOME/gcm-interp/logs/err/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acbaez@mit.edu

set -e
echo "Running: model=$MODEL  dataset=$DATASET  max_tokens=$MAX_TOKENS"
source $HOME/gcm-interp/setup.sh
cd $HOME/gcm-interp
bash gen_data.sh --model "$MODEL" --dataset "$DATASET" --max_tokens "$MAX_TOKENS"
EOF
