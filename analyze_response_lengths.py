"""
Compute mean/min/max response token lengths per model and dataset from the data dir.
Used to investigate whether positional steering performance correlates with response length.

Usage:
    python analyze_response_lengths.py
"""

import ast
import json
import os
import numpy as np
from transformers import AutoTokenizer

DATA_ROOT = "./data"

MODELS = {
    "olmo": "allenai/OLMo-2-1124-13B-DPO",
    "qwen": "Qwen/Qwen1.5-14B-Chat",
}

# (dir, add file, sub file)
DATASETS = {
    "harmful":    ("harmful-long",    "harmful-long-desired-all.jsonl",         "harmless-desired-all.jsonl"),
    "sycophancy": ("sycophancy-long", "non-sycophantic-long-desired-all.jsonl", "sycophancy-desired-all.jsonl"),
    "verse":      ("verse-long",      "verse-long-desired-all.jsonl",           "prose-desired-all.jsonl"),
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(json.dumps(ast.literal_eval(line))))
            except Exception:
                pass
    return records


def get_responses(records):
    return [r["prompt"][-1]["content"] for r in records if r["prompt"] and r["prompt"][-1]["role"] == "assistant"]


def token_lengths(texts, tokenizer):
    enc = tokenizer(texts, padding=False, truncation=False)
    return np.array([len(ids) for ids in enc["input_ids"]])


def load_tokenizer(model_id):
    hf_token = os.environ.get("HF_TOKEN")
    if "qwen" in model_id.lower():
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, pad_token="<|pad|>", eos_token="<|endoftext|>")
        tok.add_special_tokens({"pad_token": "<|endoftext|>"})
    else:
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        tok.pad_token = tok.eos_token
    return tok


def main():
    for model_tag, model_id in MODELS.items():
        print(f"\n{'='*55}")
        print(f"Model: {model_tag}  ({model_id})")
        print(f"{'='*55}")
        tokenizer = load_tokenizer(model_id)

        for ds_name, (ds_dir, add_file, sub_file) in DATASETS.items():
            base = os.path.join(DATA_ROOT, model_id.split("/")[-1], ds_dir)
            add_path = os.path.join(base, add_file)
            sub_path = os.path.join(base, sub_file)

            if not os.path.exists(add_path) or not os.path.exists(sub_path):
                print(f"  [{ds_name}] MISSING")
                continue

            responses = get_responses(load_jsonl(add_path)) + get_responses(load_jsonl(sub_path))
            if not responses:
                print(f"  [{ds_name}] no assistant turns found")
                continue

            lens = token_lengths(responses, tokenizer)
            print(f"  [{ds_name}]  n={len(lens)}  mean={lens.mean():.1f}  min={lens.min()}  max={lens.max()}")


if __name__ == "__main__":
    main()
