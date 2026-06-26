"""
Analyze prompt token-length distributions for the steering datasets.

For a given fraction p (--pct), reports the token position T such that
fraction p of examples have at least T real tokens. p=1.0 => minimum
(every sequence reaches T); p=0.5 => median; etc.

Usage:
    python analyze_lengths.py --pct 0.5
"""

import argparse
import ast
import json
import os
import numpy as np
from transformers import AutoTokenizer

DATA_ROOT = "./data"

MODELS = {
    "olmo":  "allenai/OLMo-2-1124-13B-DPO",
    "qwen":  "Qwen/Qwen1.5-14B-Chat",
    # "solar": "upstage/SOLAR-10.7B-Instruct-v1.0",
}

# (source file, base file) relative to data/<model>/<dir>/
DATASETS = {
    "harmful":    ("harmful-long",    "harmful-long-desired-all.jsonl",          "harmless-desired-all.jsonl"),
    "sycophancy": ("sycophancy-long", "non-sycophantic-long-desired-all.jsonl",  "sycophancy-desired-all.jsonl"),
    "verse":      ("verse-long",      "verse-long-desired-all.jsonl",            "prose-desired-all.jsonl"),
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


def get_prompt_strings(records, tokenizer):
    """Return chat-formatted prompt strings (question only, with generation prompt)."""
    prompts = []
    for rec in records:
        turns = rec["prompt"]
        if turns and turns[-1]["role"] == "assistant":
            turns = turns[:-1]
        text = tokenizer.apply_chat_template(
            turns,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(text)
    return prompts


def get_response_strings(records):
    """Return raw assistant response text for each record."""
    responses = []
    for rec in records:
        turns = rec["prompt"]
        if turns and turns[-1]["role"] == "assistant":
            responses.append(turns[-1]["content"])
    return responses


def real_lengths(prompts, tokenizer):
    """Return array of real (non-padding) token counts for each prompt."""
    enc = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    return enc["attention_mask"].sum(dim=1).numpy()


def threshold_at_pct(lengths, p):
    """Token position T where fraction p of examples have >= T real tokens."""
    return int(np.percentile(lengths, (1.0 - p) * 100))


def load_tokenizer(model_id):
    hf_token = os.environ.get("HF_TOKEN")
    if "qwen" in model_id.lower():
        tok = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, pad_token="<|pad|>", eos_token="<|endoftext|>"
        )
        tok.add_special_tokens({"pad_token": "<|endoftext|>"})
    else:
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pct", type=float, default=0.5,
                        help="Fraction of examples that must reach the reported token position (default 0.5)")
    args = parser.parse_args()
    p = args.pct

    print(f"\nReporting token position reached by fraction p={p} of examples")
    print(f"  p=1.0 => min (current truncation point)")
    print(f"  p={p}  => target threshold\n")

    for model_tag, model_id in MODELS.items():
        print(f"{'='*60}")
        print(f"Model: {model_tag} ({model_id})")
        print(f"{'='*60}")
        print(f"  Loading tokenizer...")
        tokenizer = load_tokenizer(model_id)

        for ds_name, (ds_dir, add_file, sub_file) in DATASETS.items():
            data_dir = os.path.join(DATA_ROOT, model_id.split("/")[-1], ds_dir)
            add_path = os.path.join(data_dir, add_file)
            sub_path = os.path.join(data_dir, sub_file)

            if not os.path.exists(add_path) or not os.path.exists(sub_path):
                print(f"  [{ds_name}] MISSING — skipping")
                continue

            add_records = load_jsonl(add_path)
            sub_records = load_jsonl(sub_path)

            add_prompts = get_prompt_strings(add_records, tokenizer)
            sub_prompts = get_prompt_strings(sub_records, tokenizer)

            add_lens = real_lengths(add_prompts, tokenizer)
            sub_lens = real_lengths(sub_prompts, tokenizer)
            combined = np.concatenate([add_lens, sub_lens])

            # Response lengths (raw assistant content, not chat-formatted)
            add_resp = get_response_strings(add_records)
            sub_resp = get_response_strings(sub_records)
            all_resp = add_resp + sub_resp
            resp_lens = real_lengths(all_resp, tokenizer) if all_resp else np.array([])

            t_min    = threshold_at_pct(combined, 1.0)
            t_target = threshold_at_pct(combined, p)
            gain     = t_target - t_min

            print(f"\n  [{ds_name}]  n={len(combined)} prompts (add={len(add_lens)}, sub={len(sub_lens)})")
            print(f"    prompts  — mean: {combined.mean():.1f}  min: {combined.min()}  max: {combined.max()}")
            if len(resp_lens):
                print(f"    responses — mean: {resp_lens.mean():.1f}  min: {resp_lens.min()}  max: {resp_lens.max()}")
            print(f"    min length  (p=1.0): {t_min:>5} tokens  <- current positional P")
            print(f"    p={p} threshold:  {t_target:>5} tokens")
            print(f"    gain:              {gain:>5} tokens  (+{gain/max(t_min,1)*100:.1f}%)")

        print()


if __name__ == "__main__":
    main()
