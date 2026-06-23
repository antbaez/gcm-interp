from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
import torch
import random
from tqdm import tqdm
import argparse
import gc
import ast
import re

args = argparse.ArgumentParser()
args.add_argument('--model_id', type=str, required=True)
args.add_argument('--base_id', type=str, default='')
args.add_argument('--dataset', type=str, default=None)  # comma-separated or 'all'
# --source/--base kept for direct invocation without --dataset
args.add_argument('--source', type=str, default=None)
args.add_argument('--base', type=str, default=None)
args.add_argument('--num_samples', type=int, default=500)
args.add_argument('--device', type=str, default='cuda:0')
args.add_argument('--gen_data', action='store_true', default=False)
args.add_argument('--batch_size', type=int, default=4)
args.add_argument('--max_tokens', type=int, default=128)

config = args.parse_args()
print('####### CONFIG ', config)

MODEL_NAME = config.model_id

DATASET_MAP = {
    'harmful':    {'source': 'harmful-long',         'base': 'harmless'},
    'sycophancy': {'source': 'non-sycophantic-long', 'base': 'sycophancy-long'},
    'verse':      {'source': 'verse-long',           'base': 'prose'},
}
ALL_DATASETS = list(DATASET_MAP.keys())

if config.dataset:
    tags = ALL_DATASETS if config.dataset == 'all' else [t.strip() for t in config.dataset.split(',')]
    for t in tags:
        if t not in DATASET_MAP:
            raise ValueError(f"Unknown dataset '{t}'. Must be one of: {', '.join(ALL_DATASETS)}, all")
    datasets = [{'tag': t, **DATASET_MAP[t]} for t in tags]
elif config.source and config.base:
    datasets = [{'tag': config.source, 'source': config.source, 'base': config.base}]
else:
    raise ValueError("Either --dataset or both --source and --base must be provided")

# Model-specific string marking the start of the assistant turn, used to strip the prompt from decoded output
MARKER = None
if 'Qwen' in MODEL_NAME:
    MARKER = '\nassistant\n'
elif 'google' in MODEL_NAME:
    MARKER = '\nmodel\n'
elif 'llama' in MODEL_NAME:
    MARKER = 'assistant\n\n'
elif 'OLMo' in MODEL_NAME:
    MARKER = '<|assistant|>\n'
if 'SOLAR' in MODEL_NAME:
    MARKER = '### Assistant:\n'

assert MARKER is not None, "Please set the MARKER variable for your model"

_filename_overrides = {'sycophancy-long': 'sycophancy'}

def preprocess_input(inputs, tokenizer):
    # Apply chat template if available, otherwise fall back to plain text formatting
    if tokenizer.chat_template:
        is_qwen3 = 'Qwen3' in MODEL_NAME
        extra_kwargs = {'enable_thinking': False} if is_qwen3 else {}
        templated_prompts = tokenizer.apply_chat_template(
                                inputs,
                                add_generation_prompt=True,
                                tokenize=False,
                                **extra_kwargs
                            )
    else:
        templated_prompts = []
        for i in inputs:
            templated_prompts.append(f"{i[0]['content']}\n")
    return templated_prompts

def generate_response(inputs, model, tokenizer, max_new_tokens=512):
    inputs = preprocess_input(inputs, tokenizer)
    input_tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=False).to(model.device)
    outputs = model.generate(
        input_ids=input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if not hasattr(generate_response, '_printed_first'):
        generate_response._printed_first = True
        print("\n=== FIRST DECODED OUTPUT (raw) ===")
        print(repr(decoded[0]))
        print("=== END ===\n")
    # Strip everything up to and including the assistant marker to isolate the response
    response = [r.split(MARKER)[-1] for idx, r in enumerate(decoded)]
    return response

def read_jsonl(data_type, queries_path):
    filename = _filename_overrides.get(data_type, data_type)
    file_name = f"{queries_path}/{filename}.jsonl"
    questions = []
    with open(file_name, 'r') as file:
        for line in file:
            data = json.loads(line)
            questions.append(data)
    return questions

def extract_messages(d):
    # New format: {"id": ..., "prompt": [{"role": ..., "content": ...}]}
    if 'prompt' in d:
        return d['prompt']
    # Old format: {"question": ..., "system": ...}
    if 'system' in d:
        return [{"role": "system", "content": d['system']}, {"role": "user", "content": d['question']}]
    return [{"role": "user", "content": d['question']}]

def check_differing_tokens(source, base, queries_path):
    # Validates that each source/base prompt pair differs by exactly one token (the contrastive word swap)
    source_prompts = []
    base_prompts = []
    for data_type in [source, base]:
        data = read_jsonl(data_type, queries_path)
        for d in data:
            msgs = extract_messages(d)
            if data_type == source:
                source_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            else:
                base_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    for src, bas in zip(source_prompts, base_prompts):
        src_tokens = tokenizer(src, return_tensors="pt", padding=True, truncation=False).to(model.device)['input_ids']
        bas_tokens = tokenizer(bas, return_tensors="pt", padding=True, truncation=False).to(model.device)['input_ids']
        try:
            assert (src_tokens != bas_tokens).sum().item() == 1
        except:
            print(src)
            print(bas)
            print(src_tokens)
            print(bas_tokens)
            print(tokenizer.convert_ids_to_tokens(src_tokens[0].tolist()))
            print(tokenizer.convert_ids_to_tokens(bas_tokens[0].tolist()))
            print((src_tokens != bas_tokens).sum().item())

def load_model_and_tokenizer(model_name: str, device: str):
    hf_token = os.environ.get('HF_TOKEN')
    # Large models (72B/27B) are loaded in 4-bit NF4 quantization to fit in GPU memory
    if '72B' in model_name or '27b' in model_name:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=nf4_config, token=hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, dtype=torch.bfloat16, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer
    return model, tokenizer

# Load once at startup; reused across all datasets
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, config.device)

if config.gen_data:
    for ds in datasets:
        source = ds['source']
        base = ds['base']
        _queries_dir = 'sycophancy-long-queries' if source == 'non-sycophantic-long' else f'{source}-queries'
        queries_path = f"/home/acbaez/gcm-interp/data/queries/{_queries_dir}"
        output_dir = 'sycophancy-long' if source == 'non-sycophantic-long' else source

        print(f"\n=== Generating dataset: source={source}  base={base} ===")

        base_name = base.replace('-long', '').replace('-single', '')
        source_name = source
        # Generate desired responses for both base and source conditions
        for gen in [base, source]:
            other = source if gen == base else base
            gen_name = base_name if gen == base else source_name
            other_name = source_name if gen == base else base_name
            op_file = f"/home/acbaez/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{output_dir}/{gen_name}-desired-all.jsonl"
            os.makedirs(os.path.dirname(op_file), exist_ok=True)
            file = open(op_file, 'w')
            random.seed(42)

            if source != 'harmful-long':
                check_differing_tokens(source, base, queries_path)

            responses = {}
            questions = {}
            originals = {}
            harmful_responses = {}
            response_types = ["desired"]

            # Build question list from the current condition's JSONL, preserving optional system prompts
            for data_type in [gen]:
                responses[data_type] = {r: [] for r in response_types}
                questions[data_type] = {}
                data = read_jsonl(data_type, queries_path)
                for d in data:
                    if "desired" not in questions[data_type]:
                        questions[data_type]["desired"] = []
                    questions[data_type]["desired"].append(extract_messages(d))
                originals[data_type] = {"desired": ""}

            qs = questions[gen]["desired"]
            batch_size = config.batch_size

            for i in tqdm(range(0, config.num_samples, batch_size)):
                batch_qs = qs[i:i + batch_size]
                rs = generate_response(batch_qs, model, tokenizer, max_new_tokens=config.max_tokens)

                for j, r in enumerate(rs):
                    q_index = i + j
                    responses[gen]["desired"].append(r)
                    file.write(json.dumps({
                        "id": q_index,
                        "prompt": qs[q_index] + [{"role": "assistant", "content": r}]
                    }))
                    file.write('\n')

                if i % (10 * batch_size) == 0:
                    file.flush()

            file.close()
            print(op_file)

            # Maps each task to the single-token string swap that flips source↔base in the prompt
            replacements = {
                'verse': {
                    'replace': f'Respond in {gen}.',
                    'with': f'Respond in {other}.'
                },
                'hate': {
                    'replace': f'I {gen} this haiku',
                    'with': f'I {other} this haiku'
                },
                'harmful': {
                    'replace': f'',
                    'with': f''
                },
                'lie': {
                    'replace': f'',
                    'with': f''
                },
                'non-sycophantic': {
                    'replace': f'I believe that it is {"ok" if "non-sycophantic" in gen else "terrible"}',
                    'with': f'I believe that it is {"terrible" if "non-sycophantic" in gen else "ok"}'
                }
            }

            with open(op_file, 'r') as file:
                dataset = [json.loads(line) for line in file]

            source_key = source.replace('-long', '').replace('-single', '')

            # Undesired responses are the desired responses re-paired with the flipped (opposite condition) prompt
            if gen == source:
                other_file = f"/home/acbaez/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{output_dir}/{base_name}-undesired-all.jsonl"
            else:
                other_file = f"/home/acbaez/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{output_dir}/{source_name}-undesired-all.jsonl"

            with open(other_file, 'w') as file:
                for data in dataset:
                    for msg in data['prompt']:
                        msg['content'] = msg['content'].replace(
                            replacements[source_key]['replace'], replacements[source_key]['with']
                        )
                    file.write(json.dumps(data))
                    file.write('\n')

            file.close()

        # Create test file from queries test file (questions only, no responses)
        test_file = f"/home/acbaez/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{output_dir}/{base_name}-test.jsonl"
        test_questions = read_jsonl(f"{base_name}-test", queries_path)
        test_qs = [extract_messages(d) for d in test_questions]
        with open(test_file, 'w') as f_out:
            for q_index, q in enumerate(test_qs):
                f_out.write(json.dumps({
                    "id": test_questions[q_index].get('id', q_index),
                    "prompt": q
                }) + '\n')
        print(test_file)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
