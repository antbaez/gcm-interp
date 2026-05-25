#!/bin/bash

MODEL_NAME="Llama-3.1-8B-Instruct"
SOURCE="lie-long"
BASE="truth"
ALGO="atp"
BATCH_SIZE=16
OPENAI_MODEL="gpt-5-mini"  # set to empty string to use local vLLM instead
# OPENAI_MODEL=""  # set to empty string to use local vLLM instead

cd /root/gcm-interp/judge-evals && python run_judge.py \
    --model_name "$MODEL_NAME" \
    --source "$SOURCE" \
    --base "$BASE" \
    --algos "$ALGO" \
    --runs_dir "/root/gcm-interp/results" \
    --data_dir "/root/gcm-interp/data" \
    --batch_size "$BATCH_SIZE" \
    --openai_model "$OPENAI_MODEL"

echo ""
echo "Summarizing results..."
python /root/gcm-interp/judge-evals/accuracy/summarize_results.py
