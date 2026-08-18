"""One-time download of cais/mmlu -> mmlu/data/mmlu_test.jsonl.

Run this manually once, on a node with outbound network access (e.g. a login
node), before submitting any mmlu/run_normal.sh or mmlu/run_preemptable.sh
jobs — compute nodes may not have reliable internet access, and this must not
run inside a SLURM job.

Usage: python mmlu/data/prepare_mmlu.py
"""
import argparse
import json
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='all')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(__file__), 'mmlu_test.jsonl'))
    args = parser.parse_args()

    dataset = load_dataset('cais/mmlu', args.config, split=args.split)
    with open(args.out, 'w') as f:
        for row in dataset:
            f.write(json.dumps({
                'subject': row['subject'],
                'question': row['question'],
                'choices': row['choices'],
                'answer': row['answer'],
            }) + '\n')
    print(f"Wrote {len(dataset)} rows to {args.out}")


if __name__ == "__main__":
    main()
