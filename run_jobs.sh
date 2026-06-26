#!/bin/bash
# Usage: bash run_jobs.sh --n|--p [--model <olmo|qwen|qwen3|gemma|all>] [--dataset <harmful|sycophancy|verse|all>] [--judging]
# Defaults: --model all --dataset all (runs both steering and judging)

QUEUE=""
MODEL="all"
DATASET="all"
JUDGING_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)       QUEUE="normal";        shift ;;
        --p)       QUEUE="preemptable";   shift ;;
        --model)   MODEL="$2";            shift 2 ;;
        --dataset) DATASET="$2";          shift 2 ;;
        --judging) JUDGING_ONLY=true;     shift ;;
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

echo "Submitting: partition=$PARTITION  time=$TIME  model=$MODEL  dataset=$DATASET  judging=$JUDGING_ONLY"
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
echo "Running: model=$MODEL  dataset=$DATASET  judging_only=$JUDGING_ONLY"
cd $HOME/gcm-interp

if [ "$JUDGING_ONLY" = false ]; then
    source $HOME/gcm-interp/setup.sh
    bash run_steering.sh --model "$MODEL" --dataset "$DATASET" --patch
fi

source $HOME/gcm-interp/setup_judging.sh
bash run_judging.sh --model "$MODEL" --dataset "$DATASET"
EOF
