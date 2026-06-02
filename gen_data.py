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
args.add_argument('--base_id', type=str, required=True)
# source/base define the contrastive pair, e.g. hate/love, verse/prose
args.add_argument('--source', type=str, default='hate')
args.add_argument('--base', type=str, default='love')
args.add_argument('--num_samples', type=int, default=500)
args.add_argument('--device', type=str, default='cuda:0')
args.add_argument('--gen_data', action='store_true', default=False)
args.add_argument('--batch_size', type=int, default=4)
args.add_argument('--max_tokens', type=int, default=128)

config = args.parse_args()
print('####### CONFIG ', config)
# os.environ['HF_HOME'] = "/mnt/align4_drive2/data"

MODEL_NAME = config.model_id

# Model-specific string marking the start of the assistant turn, used to strip the prompt from decoded output
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
source = config.source
base = config.base
# Input queries must exist at this path as {source}.jsonl and {base}.jsonl
queries_path = f"/root/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{source}-queries"



def preprocess_input(inputs, tokenizer):
    # Apply chat template if available, otherwise fall back to plain text formatting
    if tokenizer.chat_template:
        templated_prompts = tokenizer.apply_chat_template(
                                inputs,
                                add_generation_prompt=True,
                                tokenize=False
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
    # Strip everything up to and including the assistant marker to isolate the response
    response = [r.split(MARKER)[-1] for idx, r in enumerate(decoded)]
    return response

def read_jsonl(data_type, queries_path=queries_path):
    file_name = f"{queries_path}/{data_type}.jsonl"
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

def check_differing_tokens():
    # Validates that each source/base prompt pair differs by exactly one token (the contrastive word swap)
    source_prompts = []
    base_prompts = []
    for data_type in [source, base]:
        data = read_jsonl(data_type)
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
    # Large models (72B/27B) are loaded in 4-bit NF4 quantization to fit in GPU memory
    if '72B' in model_name or '27b' in model_name:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=nf4_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.tokenizer = tokenizer
    return model, tokenizer

# Load once at startup; reused across gen_data and gen_logits passes
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, config.device)

if config.gen_data:
    # Generate desired responses for both base and source conditions
    for gen in [base, source]:
        other = source if gen == base else base
        op_file = f"/root/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{source}/{gen}-desired-all.jsonl"
        os.makedirs(os.path.dirname(op_file), exist_ok=True)
        file = open(op_file, 'w')
        random.seed(42)

        check_differing_tokens()

        responses = {}
        questions = {}
        originals = {}
        harmful_responses = {}
        response_types = ["desired"]

        # Build question list from the current condition's JSONL, preserving optional system prompts
        for data_type in [gen]:
            responses[data_type] = {r: [] for r in response_types}
            questions[data_type] = {}
            data = read_jsonl(data_type)
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
            'paragraph': {
                'replace': 'a long summary',
                'with': 'a short summary'
            },
            'harmful': {
                'replace': f'',
                'with': f''
            },
            'lie': {
                'replace': f'',
                'with': f''
            },
            'lie-capitals': {
                'replace': f'',
                'with': f''
            }

        }

        with open(op_file, 'r') as file:
            dataset = [json.loads(line) for line in file]

        source_key = source.replace('-long', '').replace('-single', '')

        # Undesired responses are the desired responses re-paired with the flipped (opposite condition) prompt
        if gen == source:
            other_file = f"/root/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{source}/{base}-undesired-all.jsonl"
        else:
            other_file = f"/root/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{source}/{source}-undesired-all.jsonl"

        with open(other_file, 'w') as file:
            for data in dataset:
                for msg in data['prompt']:
                    msg['content'] = msg['content'].replace(
                        replacements[source_key]['replace'], replacements[source_key]['with']
                    )
                file.write(json.dumps(data))
                file.write('\n')

        file.close()

    # Create test file by generating responses for prompts in the queries test file
    test_file = f"/root/gcm-interp/data/{MODEL_NAME.split('/')[-1]}/{source}/{base}-test.jsonl"
    test_questions = read_jsonl(f"{base}-test")
    test_qs = [extract_messages(d) for d in test_questions]
    with open(test_file, 'w') as f_out:
        for i in tqdm(range(0, len(test_qs), batch_size)):
            batch_qs = test_qs[i:i + batch_size]
            rs = generate_response(batch_qs, model, tokenizer, max_new_tokens=config.max_tokens)
            for j, r in enumerate(rs):
                q_index = i + j
                f_out.write(json.dumps({
                    "id": test_questions[q_index].get('id', q_index),
                    "prompt": test_qs[q_index] + [{"role": "assistant", "content": r}]
                }) + '\n')
    print(test_file)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

def get_gen_accuracy_from_judge(self, prompt_responses, logit_metric, reps_type, topk):
    # print('GEN accuracy from judge ', prompt_responses)
    judge = self.model_handler.judge_model
    judge_items = self.data_handler.judge_qs
    instructions = [j['instruction'] for j in judge_items]
    questions = [j['qid'] for j in judge_items]
    expected_answers = [j['exp_answer'] for j in judge_items]
    tokenizer = self.model_handler.judge_model.tokenizer
    tokenizer.pad_token=tokenizer.eos_token
    conversations = []
    accuracy = { qid: 0 for qid in questions }
    # For binary tasks include both responses so the judge can compare; for others score the edit alone
    for i in range(len(prompt_responses)):
        prompt = prompt_responses[i]
        if self.config.args.source in ['hate', 'harmful']:
            conversation = f"{prompt['query']}\nResponse (1): {prompt[f'edit_{self.config.args.base}_q_response']}\nResponse (2): {prompt[f'old_{self.config.args.base}_q_response']}"
        else:
            conversation = f"{prompt['query']}\nResponse: {prompt[f'edit_{self.config.args.base}_q_response']}"
        conversations.append(conversation)

    # Run each judge instruction over all conversations and extract the numeric verdict from the last 10 chars
    for i, instruction in enumerate(instructions):
        prompts = [instruction.replace("{conversation}", conv) for conv in conversations]
        prompts = [[{'role': 'user', 'content': p}] for p in prompts]
        tp = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
        tokens = tokenizer(tp, return_tensors='pt', padding=True, return_attention_mask=True).to(self.model_handler.judge_model.device)
        op = judge.generate(
            **tokens,
            max_new_tokens=3,
            use_cache=False,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None
        )
        op = tokenizer.batch_decode(op, skip_special_tokens=True)
        for j, item in enumerate(op):
            # print('#### ITEM ', item)
            item = re.search(r'\d+', item[-10:])
            if item is not None:
                item = int(item.group())
                # print(item, expected_answers[i], type(item), str(item) in expected_answers[i])
                if str(item) in expected_answers[i]:
                    accuracy[questions[i]] += 1

    # Normalize raw counts to per-sample accuracy
    for q in accuracy:
        accuracy[q] = accuracy[q] / len(prompt_responses)

    os.makedirs(f'{self.config.get_output_prefix()}/eval/{self.config.args.eval_type}/', exist_ok=True)
    with open(f"{self.config.get_output_prefix()}/eval/{self.config.args.eval_type}/{logit_metric}/{reps_type}_{self.config.args.ablation_type}_{logit_metric}_{topk}_gen_accuracy.json", 'w') as f:
        json.dump(accuracy, f)
    return accuracy