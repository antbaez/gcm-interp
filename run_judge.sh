#!/bin/bash

BATCH_SIZE=64

cd /root/gcm-interp/judge-evals && python run_judge.py \
    --all \
    --runs_dir "/root/gcm-interp/results" \
    --data_dir "/root/gcm-interp/data" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "Summarizing results..."
python /root/gcm-interp/judge-evals/summarize_results.py
