from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import torch
import random
from tqdm import tqdm
import argparse
import gc

args = argparse.ArgumentParser()
args.add_argument('--model', type=str, required=True,
                  help='Short model tag or comma-separated tags: olmo, qwen, qwen3, gemma, llama, all')
args.add_argument('--dataset', type=str, default=None)  # comma-separated or 'all'
# --source/--base kept for direct invocation without --dataset
args.add_argument('--source', type=str, default=None)
args.add_argument('--base', type=str, default=None)
args.add_argument('--queries-dir', type=str, default=None,
                  help='Queries directory name under data/queries/ (required with --source/--base)')
args.add_argument('--output-dir', type=str, default=None,
                  help='Output directory name under data/<model>/ (defaults to source with --source/--base)')
args.add_argument('--device', type=str, default='cuda:0')
args.add_argument('--batch_size', type=int, default=50)
args.add_argument('--max_tokens', type=int, default=512)

config = args.parse_args()
print('####### CONFIG ', config)

MODEL_MAP = {
    'olmo':  'allenai/OLMo-2-1124-13B-DPO',
    'qwen':  'Qwen/Qwen1.5-14B-Chat',
    'qwen3': 'Qwen/Qwen3-14B',
    'gemma': 'google/gemma-3-12b-it',
    'llama': 'meta-llama/Llama-3.1-8B-Instruct',
}
ALL_MODELS = list(MODEL_MAP.keys())

if config.model == 'all':
    model_ids = list(MODEL_MAP.values())
else:
    tags = [t.strip() for t in config.model.split(',')]
    for t in tags:
        if t not in MODEL_MAP:
            raise ValueError(f"Unknown model '{t}'. Must be one of: {', '.join(ALL_MODELS)}, all")
    model_ids = [MODEL_MAP[t] for t in tags]

# Each entry: source/base file stems, queries_dir under data/queries/,
# output_dir under data/<model>/, and the ok/terrible snippet used for
# the contrastive word swap when building undesired files.
DATASET_MAP = {
    'harmful': {
        'source': 'harmful-long', 'base': 'harmless',
        'queries_dir': 'harmful-long-queries', 'output_dir': 'harmful-long',
        'ok_snippet': '', 'terrible_snippet': '',
    },
    'sycophancy': {
        'source': 'non-sycophantic-long', 'base': 'sycophancy-long',
        'queries_dir': 'sycophancy-long-queries', 'output_dir': 'sycophancy-long',
        'ok_snippet': 'I believe that it is ok', 'terrible_snippet': 'I believe that it is terrible',
    },
    'verse': {
        'source': 'verse-long', 'base': 'prose',
        'queries_dir': 'verse-long-queries', 'output_dir': 'verse-long',
        'ok_snippet': 'Respond in verse', 'terrible_snippet': 'Respond in prose',
    },
    'sycophancy-haiku': {
        'source': 'non-sycophantic-haiku-long', 'base': 'sycophancy-haiku',
        'queries_dir': 'sycophancy-haiku-long-queries', 'output_dir': 'sycophancy-haiku-long',
        'ok_snippet': 'my passage is ok', 'terrible_snippet': 'my passage is terrible',
    },
    'sycophancy-poem': {
        'source': 'non-sycophantic-poem-long', 'base': 'sycophancy-poem',
        'queries_dir': 'sycophancy-poems-long-queries', 'output_dir': 'sycophancy-poem-long',
        'ok_snippet': 'my passage is ok', 'terrible_snippet': 'my passage is terrible',
    },
    'sycophancy-haiku-concise': {
        'source': 'non-sycophantic-haiku-long', 'base': 'sycophancy-haiku',
        'queries_dir': 'sycophancy-haiku-concise-long-queries', 'output_dir': 'sycophancy-haiku-concise-long',
        'ok_snippet': 'my passage is ok', 'terrible_snippet': 'my passage is terrible',
    },
    'sycophancy-poem-concise': {
        'source': 'non-sycophantic-poem-long', 'base': 'sycophancy-poem',
        'queries_dir': 'sycophancy-poems-concise-long-queries', 'output_dir': 'sycophancy-poem-concise-long',
        'ok_snippet': 'my passage is ok', 'terrible_snippet': 'my passage is terrible',
    },
}
ALL_DATASETS = list(DATASET_MAP.keys())

if config.dataset:
    tags = ALL_DATASETS if config.dataset == 'all' else [t.strip() for t in config.dataset.split(',')]
    for t in tags:
        if t not in DATASET_MAP:
            raise ValueError(f"Unknown dataset '{t}'. Must be one of: {', '.join(ALL_DATASETS)}, all")
    datasets = [{'tag': t, **DATASET_MAP[t]} for t in tags]
elif config.source and config.base:
    if not config.queries_dir:
        raise ValueError("--queries-dir is required when using --source/--base directly")
    datasets = [{
        'tag': config.source,
        'source': config.source,
        'base': config.base,
        'queries_dir': config.queries_dir,
        'output_dir': config.output_dir or config.source,
        'ok_snippet': '', 'terrible_snippet': '',
    }]
else:
    raise ValueError("Either --dataset or both --source and --base must be provided")

_filename_overrides = {'sycophancy-long': 'sycophancy'}

def get_marker(model_name):
    if 'Qwen' in model_name:
        return '\nassistant\n'
    if 'google' in model_name:
        return '\nmodel\n'
    if 'llama' in model_name.lower():
        return 'assistant\n\n'
    if 'OLMo' in model_name:
        return '<|assistant|>\n'
    if 'SOLAR' in model_name:
        return '### Assistant:\n'
    raise ValueError(f"No assistant marker defined for model: {model_name}")

def preprocess_input(inputs, tokenizer, model_name):
    if tokenizer.chat_template:
        is_qwen3 = 'Qwen3' in model_name
        extra_kwargs = {'enable_thinking': False} if is_qwen3 else {}
        return tokenizer.apply_chat_template(
            inputs, add_generation_prompt=True, tokenize=False, **extra_kwargs
        )
    return [f"{i[0]['content']}\n" for i in inputs]

def generate_response(inputs, model, tokenizer, model_name, marker, max_new_tokens=512):
    inputs = preprocess_input(inputs, tokenizer, model_name)
    input_tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=False).to(model.device)
    outputs = model.generate(
        input_ids=input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if not hasattr(generate_response, '_printed_first') or not generate_response._printed_first:
        generate_response._printed_first = True
        print("\n=== FIRST DECODED OUTPUT (raw) ===")
        print(repr(decoded[0]))
        print("=== END ===\n")
    return [r.split(marker)[-1] for r in decoded]

def read_jsonl(data_type, queries_path):
    filename = _filename_overrides.get(data_type, data_type)
    file_name = f"{queries_path}/{filename}.jsonl"
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def extract_messages(d):
    if 'prompt' in d:
        return d['prompt']
    if 'system' in d:
        return [{"role": "system", "content": d['system']}, {"role": "user", "content": d['question']}]
    return [{"role": "user", "content": d['question']}]

def check_differing_tokens(source, base, queries_path, model, tokenizer):
    source_prompts, base_prompts = [], []
    for data_type in [source, base]:
        data = read_jsonl(data_type, queries_path)
        for d in data:
            msgs = extract_messages(d)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            (source_prompts if data_type == source else base_prompts).append(prompt)

    for src, bas in zip(source_prompts, base_prompts):
        src_tokens = tokenizer(src, return_tensors="pt", padding=True, truncation=False).to(model.device)['input_ids']
        bas_tokens = tokenizer(bas, return_tensors="pt", padding=True, truncation=False).to(model.device)['input_ids']
        try:
            assert (src_tokens != bas_tokens).sum().item() == 1
        except AssertionError:
            print(src); print(bas)
            print(tokenizer.convert_ids_to_tokens(src_tokens[0].tolist()))
            print(tokenizer.convert_ids_to_tokens(bas_tokens[0].tolist()))
            print((src_tokens != bas_tokens).sum().item())

def load_model_and_tokenizer(model_name: str, device: str):
    hf_token = os.environ.get('HF_TOKEN')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, dtype=torch.bfloat16, token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


for MODEL_NAME in model_ids:
    print(f"\n{'='*60}\nModel: {MODEL_NAME}\n{'='*60}")
    marker = get_marker(MODEL_NAME)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, config.device)
    generate_response._printed_first = False  # print first output for each model

    for ds in datasets:
        source = ds['source']
        base   = ds['base']
        queries_path = f"/home/acbaez/gcm-interp/data/queries/{ds['queries_dir']}"
        output_dir   = ds['output_dir']

        print(f"\n=== Generating dataset: source={source}  base={base} ===")

        base_name   = base.replace('-long', '').replace('-single', '')
        source_name = source

        for gen in [base, source]:
            other    = source if gen == base else base
            gen_name = base_name if gen == base else source_name
            op_file  = f"/home/acbaez/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{output_dir}/{gen_name}-desired-all.jsonl"
            os.makedirs(os.path.dirname(op_file), exist_ok=True)
            random.seed(42)

            if source != 'harmful-long':
                check_differing_tokens(source, base, queries_path, model, tokenizer)

            data     = read_jsonl(gen, queries_path)
            qs       = [extract_messages(d) for d in data]

            with open(op_file, 'w') as f:
                for i in tqdm(range(0, len(qs), config.batch_size)):
                    batch_qs = qs[i:i + config.batch_size]
                    rs = generate_response(
                        batch_qs, model, tokenizer, MODEL_NAME, marker,
                        max_new_tokens=config.max_tokens
                    )
                    for j, r in enumerate(rs):
                        q_index = i + j
                        f.write(json.dumps({
                            "id": q_index,
                            "prompt": qs[q_index] + [{"role": "assistant", "content": r}]
                        }) + '\n')
                    if i % (10 * config.batch_size) == 0:
                        f.flush()
            print(op_file)

            # Build the undesired file by swapping the contrastive word in the prompt.
            ok_snip      = ds['ok_snippet']
            bad_snip     = ds['terrible_snippet']
            swap_replace = ok_snip  if gen == source else bad_snip
            swap_with    = bad_snip if gen == source else ok_snip

            with open(op_file, 'r') as f:
                dataset = [json.loads(line) for line in f]

            other_file = (
                f"/home/acbaez/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{output_dir}/{base_name}-undesired-all.jsonl"
                if gen == source else
                f"/home/acbaez/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{output_dir}/{source_name}-undesired-all.jsonl"
            )
            with open(other_file, 'w') as f:
                for entry in dataset:
                    if swap_replace:
                        for msg in entry['prompt']:
                            msg['content'] = msg['content'].replace(swap_replace, swap_with)
                    f.write(json.dumps(entry) + '\n')

        # Write test file (prompts only, no responses)
        test_file      = f"/home/acbaez/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{output_dir}/{base_name}-test.jsonl"
        test_questions = read_jsonl(f"{base_name}-test", queries_path)
        with open(test_file, 'w') as f:
            for q_index, d in enumerate(test_questions):
                f.write(json.dumps({
                    "id": d.get('id', q_index),
                    "prompt": extract_messages(d)
                }) + '\n')
        print(test_file)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
