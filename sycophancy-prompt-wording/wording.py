"""
Quick probe: generate responses to the sycophancy-test passages under several
self-assessment wordings ("terrible", "ok", "great", "brilliant"), for all
four models, and dump everything to a single JSON file for inspection.

This does not touch the steering pipeline - it's a standalone sanity check
for comparing how each model's natural (unsteered) response shifts across
wordings, before committing to a new contrastive-pair design.

Usage:
    python wording_probe.py
    python wording_probe.py --models gemma,qwen3 --n_prompts 5
    python wording_probe.py --wordings terrible,ok,great,brilliant,amazing
"""

import argparse
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_MAP = {
    'olmo': 'allenai/OLMo-2-1124-13B-DPO',
    'qwen3': 'Qwen/Qwen3-14B',
    'gemma': 'google/gemma-3-12b-it',
    'llama': 'meta-llama/Llama-3.1-8B-Instruct',
}

TEST_QUERIES_PATH = f'data/queries/sycophancy-long-queries/sycophancy-test.jsonl'
DEFAULT_WORDINGS = ['terrible', 'ok', 'great', 'brilliant']
SWAP_TARGET = 'terrible'


def get_marker(model_name):
    if 'Qwen' in model_name:
        return '\nassistant\n'
    if 'google' in model_name:
        return '\nmodel\n'
    if 'llama' in model_name.lower():
        return 'assistant\n\n'
    if 'OLMo' in model_name:
        return '<|assistant|>\n'
    raise ValueError(f"No assistant marker defined for model: {model_name}")


def load_model_and_tokenizer(model_name, device):
    hf_token = os.environ.get('HF_TOKEN')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, dtype=torch.bfloat16, token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def preprocess_input(prompts, tokenizer, model_name):
    if tokenizer.chat_template:
        extra_kwargs = {'enable_thinking': False} if 'Qwen3' in model_name else {}
        return tokenizer.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=False, **extra_kwargs
        )
    return [f"{p[0]['content']}\n" for p in prompts]


def generate_batch(prompts, model, tokenizer, model_name, marker, max_new_tokens):
    texts = preprocess_input(prompts, tokenizer, model_name)
    tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=False).to(model.device)
    outputs = model.generate(
        input_ids=tokens['input_ids'],
        attention_mask=tokens['attention_mask'],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [d.split(marker)[-1].strip() for d in decoded]


def load_base_examples(n_prompts):
    with open(TEST_QUERIES_PATH) as f:
        rows = [json.loads(line) for line in f][:n_prompts]
    return rows


def build_variants(rows, wordings):
    variants = {w: [] for w in wordings}
    for row in rows:
        base_content = row['prompt'][0]['content']
        for w in wordings:
            content = base_content.replace(SWAP_TARGET, w) if w != SWAP_TARGET else base_content
            variants[w].append({
                'id': row['id'],
                'prompt': [{'role': 'user', 'content': content}],
            })
    return variants


def run_model(model_tag, wordings, rows, max_new_tokens, device):
    model_name = MODEL_MAP[model_tag]
    print(f"\n=== {model_tag} ({model_name}) ===")
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    marker = get_marker(model_name)
    variants = build_variants(rows, wordings)

    results = {}
    for wording, examples in variants.items():
        print(f"  generating wording={wording!r} ({len(examples)} prompts)")
        responses = generate_batch(
            [ex['prompt'] for ex in examples], model, tokenizer, model_name, marker, max_new_tokens
        )
        results[wording] = [
            {'id': ex['id'], 'prompt': ex['prompt'][0]['content'], 'response': resp}
            for ex, resp in zip(examples, responses)
        ]

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='gemma,llama,olmo,qwen3')
    parser.add_argument('--n_prompts', type=int, default=10)
    parser.add_argument('--wordings', type=str, default=','.join(DEFAULT_WORDINGS))
    parser.add_argument('--max_tokens', type=int, default=400)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='sycophancy_wording_results.json')
    args = parser.parse_args()

    model_tags = [t.strip() for t in args.models.split(',')]
    for t in model_tags:
        if t not in MODEL_MAP:
            raise ValueError(f"Unknown model tag '{t}'. Must be one of: {', '.join(MODEL_MAP)}")
    wordings = [w.strip() for w in args.wordings.split(',')]

    rows = load_base_examples(args.n_prompts)

    all_results = {'wordings': wordings, 'n_prompts': args.n_prompts, 'models': {}}
    for tag in model_tags:
        all_results['models'][tag] = run_model(tag, wordings, rows, args.max_tokens, args.device)
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  saved progress to {args.output}")

    print(f"\nDone. Full results at {args.output}")


if __name__ == '__main__':
    main()
