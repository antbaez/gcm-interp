# for model in Qwen1.5-14B-Chat OLMo-2-1124-13B-DPO SOLAR-10.7B-Instruct-v1.0; do     for task in gen relevance fluency; do          python gen_judgements.py --model_id ${model} --source hate --base love --device cuda:0 --which_task ${task};      done; done
import hf_cache
import os
import json
import argparse
import re
import ast
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import gc
args = argparse.ArgumentParser()
args.add_argument("--model_id", type=str, default="Qwen1.5-14B-Chat")
args.add_argument("--source", type=str, default="harmful")
args.add_argument("--base", type=str, default="harmless")
args.add_argument("--device", type=str, default="cuda:0")
args.add_argument("--which_task", type=str, default="data")

config = args.parse_args()
model_id = config.model_id
device = config.device
base = config.base
source = config.source
which_task = config.which_task

judge_id = "meta-llama/Llama-3.1-70B-Instruct"

nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
judge = AutoModelForCausalLM.from_pretrained(judge_id, device_map=device, torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], quantization_config=nf4_config)
tokenizer = AutoTokenizer.from_pretrained(judge_id, token=os.environ['HF_TOKEN'])
tokenizer.pad_token=tokenizer.eos_token
def load_from_jsonl(file_name):
    def load_json_line(line: str, i: int, file_name: str):
        return json.loads(json.dumps(ast.literal_eval(line)))
    
    with open(file_name, "r") as f:
        data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
    return data

def get_gen_accuracy_from_judge(prompt_responses, source):
    # print('GEN accuracy from judge ', prompt_responses)
    judge_qs_path = f"./data/{model_id}/logits/{source}/judge-qs.jsonl"
    judge_items = load_from_jsonl(judge_qs_path)
    
    instructions = [j['instruction'] for j in judge_items]
    questions = [j['qid'] for j in judge_items]
    expected_answers = [j['exp_answer'] for j in judge_items]
    conversations = []
    accuracy = { qid: 0 for qid in questions }
    items = []
    # print('############## LEN PROMPT RESPONSES ', len(prompt_responses))
    for i in range(len(prompt_responses)):
        prompt = prompt_responses[i]
        if source in ['hate', 'harmful']:
            conversation = f"{prompt['query']}\nResponse (1): {prompt[f'edit_{base}']}\nResponse (2): {prompt[f'old_{base}']}"
        else:
            conversation = f"{prompt['query']}\nResponse: {prompt[f'edit_{base}']}"
        conversations.append(conversation)

    for i, instruction in enumerate(instructions):
        prompts = [instruction.replace("{conversation}", conv) for conv in conversations]
        prompts = [[{'role': 'user', 'content': p}] for p in prompts]
        tp = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
        tokens = tokenizer(tp, return_tensors='pt', padding=True, return_attention_mask=True).to(device)
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
            to_add = {"conversation": item, "ans": None}
            # print('#### ITEM ', item)
            item = re.search(r'\d+', item[-10:])
            if item is not None:
                item = int(item.group())
                to_add["ans"] = item
                items.append(to_add)
                # print(item, expected_answers[i], type(item), str(item) in expected_answers[i])
                if str(item) in expected_answers[i]:
                    accuracy[questions[i]] += int(item) * 0.2

    for q in accuracy:
        accuracy[q] = accuracy[q] / len(prompt_responses)

    return accuracy, items

def get_fluency_accuracy_from_judge(prompt_responses):
    # print('GEN accuracy from judge ', prompt_responses)
    judge_qs_path = f"./data/fluency-qs.jsonl"
    judge_items = load_from_jsonl(judge_qs_path)
    
    instructions = [j['instruction'] for j in judge_items]
    questions = [j['qid'] for j in judge_items]
    expected_answers = [j['exp_answer'] for j in judge_items]
    conversations = []
    accuracy = { qid: 0 for qid in questions }
    items = []
    # print('############## LEN PROMPT RESPONSES ', len(prompt_responses))
    for i in range(len(prompt_responses)):
        prompt = prompt_responses[i]
        conversation = prompt[f'edit_{base}']
        conversations.append(conversation)

    for i, instruction in enumerate(instructions):
        prompts = [instruction.replace("[Sentence goes here]", conv) for conv in conversations]
        prompts = [[{'role': 'user', 'content': p}] for p in prompts]
        tp = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
        tokens = tokenizer(tp, return_tensors='pt', padding=True, return_attention_mask=True).to(device)
        op = judge.generate(
            **tokens,
            max_new_tokens=8,
            use_cache=False,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None
        )
        # print(op)
        op = tokenizer.batch_decode(op, skip_special_tokens=True)
        # print('OP ', op)
        for j, item in enumerate(op):
            # print('#### ITEM ', item)
            to_add = {"conversation": item, "ans": None}
            item = re.search(r'Rating:.*?(\d)', item[-30:])
            # print(item)
            if item is not None:
                item = int(item.group(1))
                to_add["ans"] = item
                items.append(to_add)
                # print(item, expected_answers[i], type(item), str(item) in expected_answers[i])
                if str(item) in expected_answers[i]:
                    accuracy[questions[i]] += 1

    for q in accuracy:
        accuracy[q] = accuracy[q] / len(prompt_responses)

    return accuracy, items

def get_relevance_accuracy_from_judge(prompt_responses):
    # print('GEN accuracy from judge ', prompt_responses)
    judge_qs_path = f"./data/relevance-qs.jsonl"
    judge_items = load_from_jsonl(judge_qs_path)
    
    instructions = [j['instruction'] for j in judge_items]
    questions = [j['qid'] for j in judge_items]
    expected_answers = [j['exp_answer'] for j in judge_items]
    conversations = []
    queries = []
    accuracy = { qid: 0 for qid in questions }
    items = []
    # print('############## LEN PROMPT RESPONSES ', len(prompt_responses))
    for i in range(len(prompt_responses)):
        prompt = prompt_responses[i]
        conversation = prompt[f'edit_{base}']
        query = prompt['query']
        conversations.append(conversation)
        queries.append(query)

    for i, instruction in enumerate(instructions):
        prompts = [instruction.replace("[Sentence goes here]", conv) for conv in conversations]
        prompts = [p.replace("[Query goes here]", query) for p, query in zip(prompts, queries)]
        prompts = [[{'role': 'user', 'content': p}] for p in prompts]
        tp = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
        tokens = tokenizer(tp, return_tensors='pt', padding=True, return_attention_mask=True).to(device)
        op = judge.generate(
            **tokens,
            max_new_tokens=8,
            use_cache=False,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None
        )
        # print(op)
        op = tokenizer.batch_decode(op, skip_special_tokens=True)
        # print('OP ', op)
        for j, item in enumerate(op):
            to_add = {"conversation": item, "ans": None}
            # print('#### ITEM ', item)
            item = re.search(r'Rating:.*?(\d)', item[-30:])
            # print(item)
            if item is not None:
                item = int(item.group(1))
                to_add["ans"] = item
                items.append(to_add)
                # print(item, expected_answers[i], type(item), str(item) in expected_answers[i])
                if str(item) in expected_answers[i]:
                    accuracy[questions[i]] += 1

    for q in accuracy:
        accuracy[q] = accuracy[q] / len(prompt_responses)

    return accuracy, items

if config.which_task == 'data':
    for model_id in ['OLMo-2-1124-13B-DPO', 'Qwen1.5-14B-Chat', 'SOLAR-10.7B-Instruct-v1.0']:
        for base, source in tqdm([('prose', 'verse'), ('love', 'hate'), ('harmless', 'harmful')]):
            data_path = f"./data/{model_id}/{source}/"
            if os.path.exists(f"{data_path}/judge-accuracy-granular"):
                print(f"Judge accuracy already exists for {model_id} from {source} to {base}. Skipping...")
                continue
            desired_path = f"{data_path}/{base}-desired.jsonl"
            undesired_path = f"{data_path}/{base}-undesired.jsonl"

            desired_data = load_from_jsonl(desired_path)
            undesired_data = load_from_jsonl(undesired_path)

            prompt_responses = []
            for idx, d in enumerate(desired_data[:50]):
                prompt_responses.append({
                    "query": d["question"][-1]['content'],
                    f"edit_{base}": d["response"],
                    f"old_{base}": undesired_data[idx]["response"]
                })
            baseline_accuracy = get_gen_accuracy_from_judge(prompt_responses, source)

            print('############## DESIRED ACCURACY ', baseline_accuracy)

            prompt_responses = []
            for idx, d in enumerate(desired_data[:50]):
                prompt_responses.append({
                    "query": d["question"][-1]['content'],
                    f"old_{base}": d["response"],
                    f"edit_{base}": undesired_data[idx]["response"]
                })
            target_accuracy, target_items = get_gen_accuracy_from_judge(prompt_responses, source)

            print('############## UNDESIRED ACCURACY ', target_accuracy)

            with open(f"{data_path}/judge-accuracy-granular.json", 'w') as f:
                json.dump({
                    "baseline_accuracy": baseline_accuracy,
                    "target_accuracy": target_accuracy
                }, f)
            with open(f"{data_path}/judge-accuracy-granular_eval.json", 'w') as f:
                json.dump(target_items, f)
    del judge
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
elif config.which_task == 'gen':
    for top_k in tqdm([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1]):
        for method in tqdm(['atp', 'atp-zero', 'random', 'probes', 'acp'], desc='Methods'):
            if method == 'atp':
                ablation_types = ['mean', 'steer']
                reps_types = ['targeted']
            else:
                ablation_types = ['steer']
                reps_types = ['random'] if method == 'random' else ['targeted']
            for ablation_type in tqdm(ablation_types, desc='Ablation Types'):
                for reps_type in tqdm(reps_types, desc='Reps Types'):
                    for steering_factor in list(range(1, 11)):
                        if os.path.exists(f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_gen_accuracy_new.json") and \
                            os.path.exists(f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_gen_granular.json"):
                            print(f"Gen accuracy already exists for {model_id} from {source} to {base} with method {method}, steering factor {steering_factor}, reps type {reps_type}, and ablation type {ablation_type}. Skipping...")
                            continue
                        try:
                            data_path = f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/"
                            file_path = f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_gen.json"
                            data = load_from_jsonl(file_path)[0]
                            # print(data_path, file_path)
                            prompt_responses = []
                            # print(data)
                            for idx, d in enumerate(data):
                                prompt_responses.append({
                                    "query": d["query"],
                                    f"edit_{base}": d[f"edit_{base}"],
                                    f"old_{base}": d[f"old_{base}"]
                                })
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            continue
                        gen_accuracy, gen_items = get_gen_accuracy_from_judge(prompt_responses, source)
                        with open(f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_gen_accuracy_new.json", 'w') as f:
                            json.dump({
                                "gen": gen_accuracy
                            }, f)
                        with open(f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_gen_granular.json", 'w') as f:
                            json.dump(gen_items, f)
                        print(f'############## gen ACCURACY {config.model_id}', gen_accuracy)


elif config.which_task == 'fluency':
    for top_k in tqdm([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1]):
        for method in tqdm(['atp', 'atp-zero', 'random', 'probes', 'acp'], desc='Methods'):
            if method == 'atp':
                ablation_types = ['mean', 'steer']
                reps_types = ['targeted']
            else:
                ablation_types = ['steer']
                reps_types = ['random'] if method == 'random' else ['targeted']
            for ablation_type in tqdm(ablation_types, desc='Ablation Types'):
                for reps_type in tqdm(reps_types, desc='Reps Types'):
                    for steering_factor in list(range(1, 11)):
                        if os.path.exists(f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_fluency_accuracy.json") and \
                            os.path.exists(f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_fluency_granular.json"):
                            print(f"Fluency accuracy already exists for {model_id} from {source} to {base} with method {method}, steering factor {steering_factor}, reps type {reps_type}, and ablation type {ablation_type}. Skipping...")
                            continue
                        try:
                            data_path = f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/"
                            file_path = f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_gen.json"
                            data = load_from_jsonl(file_path)[0]
                            # print(data_path, file_path)
                            prompt_responses = []
                            # print(data)
                            for idx, d in enumerate(data):
                                prompt_responses.append({
                                    "query": d["query"],
                                    f"edit_{base}": d[f"edit_{base}"],
                                    f"old_{base}": d[f"old_{base}"]
                                })
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            continue
                        fluency_accuracy, fluency_items = get_fluency_accuracy_from_judge(prompt_responses)
                        with open(f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_fluency_accuracy.json", 'w') as f:
                            json.dump({
                                "fluency": fluency_accuracy
                            }, f)
                        with open(f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_fluency_granular.json", 'w') as f:
                            json.dump(fluency_items, f)
                        print(f'############## fluency ACCURACY {config.model_id}', fluency_accuracy)

elif config.which_task == 'relevance':
    for top_k in tqdm([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1]):
        for method in tqdm(['atp', 'atp-zero', 'random', 'probes', 'acp'], desc='Methods'):
            if method == 'atp':
                ablation_types = ['mean', 'steer']
                reps_types = ['targeted']
            else:
                ablation_types = ['steer']
                reps_types = ['random'] if method == 'random' else ['targeted']
            for ablation_type in tqdm(ablation_types, desc='Ablation Types'):
                for reps_type in tqdm(reps_types, desc='Reps Types'):
                    for steering_factor in list(range(1, 11)):
                        if os.path.exists(f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_relevance_accuracy.json") and \
                            os.path.exists(f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_relevance_granular.json"):
                            print(f"Relevance accuracy already exists for {model_id} from {source} to {base} with method {method}, steering factor {steering_factor}, reps type {reps_type}, and ablation type {ablation_type}. Skipping...")
                            continue
                        try:
                            data_path = f"./runs/{model_id}/from_{source}_to_{base}/{method}/eval/"
                            file_path = f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_gen.json"
                            data = load_from_jsonl(file_path)[0]
                            # print(data_path, file_path)
                            prompt_responses = []
                            # print(data)
                            for idx, d in enumerate(data):
                                prompt_responses.append({
                                    "query": d["query"],
                                    f"edit_{base}": d[f"edit_{base}"],
                                    f"old_{base}": d[f"old_{base}"]
                                })
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            continue
                        relevance_accuracy, relevance_items = get_relevance_accuracy_from_judge(prompt_responses)
                        with open(f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_relevance_accuracy.json", 'w') as f:
                            json.dump({
                                "relevance": relevance_accuracy
                            }, f)
                        with open(f"{data_path}/{steering_factor}_{reps_type}_{ablation_type}_topk_{top_k}_relevance_granular.json", 'w') as f:
                            json.dump(relevance_items, f)
                        print(f'############## relevance ACCURACY {config.model_id}', relevance_accuracy)

