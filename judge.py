import hf_cache
import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import argparse
import json
from math import ceil
import re
import ast
from tqdm import tqdm
args = argparse.ArgumentParser()
args.add_argument('--judge_id', type=str, default='meta-llama/Llama-3.1-70B-Instruct')
args.add_argument('--source', type=str, default='hate')
args.add_argument('--base', type=str, default='love')
args.add_argument('--ablation', type=str, default='steer')
args.add_argument('--model_id', type=str, default='Qwen/Qwen1.5-14B-Chat')
args = args.parse_args()
RM_INTERP_REPO = os.environ['RM_INTERP_REPO']
def load_model_and_tokenizer(model_id):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], quantization_config=nf4_config)
    return model, tokenizer

def compute_judge_accuracy(judge_model, tokenizer, prompts, judge_items, base, source):
    print('Starting judge accuracy computation...')
    tokenizer.pad_token = tokenizer.eos_token
    questions = [j['qid'] for j in judge_items]
    expected_answers = [j['exp_answer'] for j in judge_items]
    accuracy = {qid: 0 for qid in questions}
    responses = []

    conversations = []
    assert len(prompts) > 0, "No prompts provided for evaluation."
    for p in prompts:
        q = p['query']
        r = p[f'edit_{base}']
        if source in ['hate', 'harmful']:
            comparison = f"{q}\nResponse (1): {r}\nResponse (2): {p[f'old_{base}']}"
        else:
            comparison = f"{q}\nResponse: {r}"
        conversations.append(comparison)
    assert len(conversations) > 0, "No conversations to evaluate."
    batch_size = 32  # adjust for your GPU memory

    batched_inputs = []
    metadata = []  # keep track of (idx, conversation)

    # Step 1: collect all prompts + metadata
    for idx, item in enumerate(judge_items):
        for c in conversations:
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': item['instruction'].replace("{conversation}", c)}],
                add_generation_prompt=True,
                tokenize=False
            )
            batched_inputs.append(prompt)
            metadata.append((idx, c))

    print(f"Judging {len(batched_inputs)} conversations total")

    # Step 2: process in batches
    num_batches = ceil(len(batched_inputs) / batch_size)

    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        input_chunk = batched_inputs[start:end]
        meta_chunk = metadata[start:end]

        # Tokenize
        tokens = tokenizer(
            input_chunk,
            return_tensors='pt',
            padding=True,
            return_attention_mask=True
        ).to(judge_model.device)

        # Generate
        outputs = judge_model.generate(
            **tokens,
            max_new_tokens=3,
            use_cache=False,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None
        )

        # Decode + evaluate
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for (idx, _), d in zip(meta_chunk, decoded):
            match = re.search(r'\d+', d[-10:])
            if match and match.group() in expected_answers[idx]:
                accuracy[questions[idx]] += 1
            responses.append(d)


    for k in accuracy:
        accuracy[k] /= len(prompts)
    return accuracy, responses

def save_judge_accuracy(acc, path):
    with open(path, 'w') as f:
        json.dump(acc, f)
    print(f"Saved judge accuracy to {path}")

def load_from_jsonl(file_name):
    def load_json_line(line: str, i: int, file_name: str):
        try:
            return json.loads(json.dumps(ast.literal_eval(line)))
        except Exception as e:
            return None
    
    with open(file_name, "r") as f:
        data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
        data = [d for d in data if d is not None]
    return data

model, tokenizer = load_model_and_tokenizer(args.judge_id)
for model_id in [args.model_id]:
    judge_qs = load_from_jsonl(f'{RM_INTERP_REPO}/data/{model_id}/logits/{args.source}/judge-qs.jsonl')
    print('Judge questions are....')
    print(judge_qs)
    for patch_algo in ['acp', 'atp', 'atp-zero', 'probes', 'random']:
        patch = 'random' if patch_algo in ['random'] else 'targeted'
        for N in range(10, -1, -1):
            for topk in tqdm([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1], desc=f'topk for {N}'):
                if args.ablation in ['steer', 'mean']:
                    gen_path = f'{RM_INTERP_REPO}/runs-new/{model_id}/from_{args.source}_to_{args.base}/{patch_algo}/eval_test/{N}_{patch}_{args.ablation}_topk_{topk}_gen.json'
                    acc_path = f'{RM_INTERP_REPO}/runs-new/{model_id}/from_{args.source}_to_{args.base}/{patch_algo}/eval_test/{N}_{patch}_{args.ablation}_topk_{topk}_gen_accuracy.json'
                    acc_resp_path = f'{RM_INTERP_REPO}/runs-new/{model_id}/from_{args.source}_to_{args.base}/{patch_algo}/eval_test/{N}_{patch}_{args.ablation}_topk_{topk}_gen_accuracy_responses.json'
                elif args.ablation == 'pyreft':
                    gen_path = f'{RM_INTERP_REPO}/runs-new/{model_id}/from_{args.source}_to_{args.base}/{patch_algo}/eval/{N}_{patch}_{args.ablation}_topk_{topk}_gen.json'
                    acc_path = f'{RM_INTERP_REPO}/runs-new/{model_id}/from_{args.source}_to_{args.base}/{patch_algo}/eval/{N}_{patch}_{args.ablation}_topk_{topk}_gen_accuracy.json'
                    acc_resp_path = f'{RM_INTERP_REPO}/runs-new/{model_id}/from_{args.source}_to_{args.base}/{patch_algo}/eval/{N}_{patch}_{args.ablation}_topk_{topk}_gen_accuracy_responses.json'

                if os.path.exists(acc_path) and os.path.exists(acc_resp_path):
                    print(f"Skipping existing eval for Ablation: {args.ablation}, Reps: {patch}, TopK: {topk}, N: {N}")
                    continue
                try:
                    with open(gen_path, 'r') as f:
                        decoded_responses = json.load(f)
                        # print(decoded_responses)
                    print(f"Eval [[GEN]] → Ablation: {args.ablation}, Reps: {patch}, TopK: {topk}")
                    print(acc_path)
                    assert len(decoded_responses) > 0, "No responses to evaluate."
                    acc, responses = compute_judge_accuracy(model, tokenizer, decoded_responses, judge_qs, args.base, args.source)
                    print('acc ', acc)
                    save_judge_accuracy(acc, acc_path)
                    save_judge_accuracy(responses, acc_path.replace("_accuracy.json", "_accuracy_responses.json"))
                except Exception as e:
                    with open(f'{RM_INTERP_REPO}/from_{args.source}_to_{args.base}_{patch_algo}_errors.txt', 'a') as f:
                        f.write(f'Errored model {model_id} patch_algo {patch_algo} patch {patch} N {N} topk {topk}, Exception {e}')
                    continue
