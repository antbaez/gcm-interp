#!/bin/bash

BATCH_SIZE=64
EVAL_MODE=eval_train   # eval_train -> {base}-desired-all.jsonl, eval_test -> {base}-test.jsonl

cd /root/gcm-interp/judge-evals && python run_judge.py \
    --all \
    --runs_dir "/root/gcm-interp/results" \
    --data_dir "/root/gcm-interp/data" \
    --eval_mode "$EVAL_MODE" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "Summarizing results..."
python /root/gcm-interp/judge-evals/summarize_results.py
