import torch
import random
import torch.nn.functional as F
import json
import ast
import re
from tqdm import tqdm
random.seed(42)
import pandas as pd
class DataHandler:
    def __init__(self, config, model_handler):
        self.config = config
        self.model_handler = model_handler
        self.device = self.config.args.device
        self.no_generation_prompt_for_eval_test = False
        
        file_paths = {
            'base_desired': f"{self.config.args.data_path}/{self.config.args.base}-desired-all.jsonl",
            'prompting_base_desired': f"{self.config.args.data_path}/prompting.jsonl",
            'base_undesired': f"{self.config.args.data_path}/{self.config.args.base}-undesired-all.jsonl",
            'source_desired': f"{self.config.args.data_path}/{self.config.args.source}-desired-all.jsonl",
            'source_undesired': f"{self.config.args.data_path}/{self.config.args.source}-undesired-all.jsonl",
            'base_test': f"{self.config.args.data_path}/{self.config.args.base}-test.jsonl" if self.config.args.eval_test else None,
            'eval_test': f"./data/eval/from_{self.config.args.source}_to_{self.config.args.base}/{self.config.args.eval_test_dataset}.jsonl" if self.config.args.eval_test_dataset else None,
            'mmlu': f"./data/mmlu.csv"
        }

        jsons = {
            'base_desired': self.load_from_jsonl(file_paths['base_desired']),
            'prompting_base_desired': self.load_from_jsonl(file_paths['prompting_base_desired']),
            'base_undesired': self.load_from_jsonl(file_paths['base_undesired']),
            'source_desired': self.load_from_jsonl(file_paths['source_desired']),
            'source_undesired': self.load_from_jsonl(file_paths['source_undesired']),
            'base_test': self.load_from_jsonl(file_paths['base_test']) if self.config.args.eval_test else None,
            'eval_test': self.load_from_jsonl(file_paths['eval_test']) if self.config.args.eval_test_dataset else None
        }

        mmlu = pd.read_csv(file_paths['mmlu'])
        jsons = self.filter_jsons(jsons)

        print('Making base templated prompts...')
        base = {
            'desired': self.get_templated_prompts(jsons['base_desired']),
            'undesired': self.get_templated_prompts(jsons['base_undesired'])
        }

        print('Making base_qs templated prompts...')
        base_qs = {
            'desired': self.get_templated_prompts(jsons['base_desired'], only_q=True, add_generation_prompt=True),
            'undesired': self.get_templated_prompts(jsons['base_undesired'], only_q=True, add_generation_prompt=True),
        }

        if self.config.args.eval_test:
            print('Making base_qs test templated prompts...')
            base_qs['test'] = self.get_templated_prompts(jsons['base_test'], only_q=True, add_generation_prompt=True)

        print('Making source qs templated prompts...')
        source_qs = {
            'desired': self.get_templated_prompts(jsons['source_desired'], only_q=True, add_generation_prompt=True),
            'undesired': self.get_templated_prompts(jsons['source_undesired'], only_q=True, add_generation_prompt=True)
        }

        if 'len' in jsons['base_desired'][0]:
            self.max_len = max(jsons['base_desired'][0]['len'], jsons['base_undesired'][0]['len'], jsons['source_desired'][0]['len'], jsons['source_undesired'][0]['len'])
        else:
            all_templated_prompts = base['desired'] + base['undesired'] + source_qs['desired'] + source_qs['undesired']
            all_tokenized_prompts = self.tokenize_prompts(all_templated_prompts, max_length=None)
            self.max_len = all_tokenized_prompts['input_ids'].shape[1]

        if self.config.args.eval_model:
            orig_template = self.model_handler.tokenizer.chat_template
            # model_handler.model.tokenizer.chat_template = 
            # print(mmlu['prompt'].tolist()[:5])
            self.mmlu_prompts = self.tokenize_prompts(mmlu['prompt'].tolist(), max_length=self.max_len)
            self.mmlu_answers = mmlu['answer_key'].tolist()
            if self.config.args.judge_answer_match:
                self.no_generation_prompt_for_eval_test = True
                self.judge_qs = self.load_from_jsonl(f"./data/eval/from_{self.config.args.source}_to_{self.config.args.base}/{self.config.args.eval_test_dataset}-judge-qs.jsonl")
            else:
                self.judge_qs = self.load_from_jsonl(f"{self.config.args.data_path}/judge-qs.jsonl")
            if self.config.args.eval_test_dataset:
                # Removing the system prompt for eval_test dataset
                if config.args.model_id.split('/')[1] == 'Qwen1.5-14B-Chat':
                    self.model_handler.tokenizer.chat_template = "{% for message in messages %}\n" \
                    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}\n" \
                    "{% endfor %}\n" \
                    "{% if add_generation_prompt %}\n" \
                    "{{ '<|im_start|>assistant\n' }}" \
                    "{% endif %}"

                elif config.args.model_id.split('/')[1] == 'OLMo-2-1124-13B-DPO':
                    self.model_handler.tokenizer.chat_template = "{{ bos_token }}\n" \
                    "{% for message in messages %}\n" \
                    "{% if message['role'] == 'user' -%}\n" \
                    "{{ '<|user|>\\n' + message['content'] + '\\n' }}\n" \
                    "{%- elif message['role'] == 'assistant' -%}\n" \
                    "{%- if not loop.last -%}\n" \
                    "{{ '<|assistant|>\\n' + message['content'] + eos_token + '\\n' }}\n" \
                    "{%- else -%}\n" \
                    "{{ '<|assistant|>\\n' + message['content'] + eos_token }}\n" \
                    "{%- endif -%}\n" \
                    "{%- endif %}\n" \
                    "{% endfor -%}\n" \
                    "{% if add_generation_prompt -%}\n" \
                    "{{ '\\n<|assistant|>\\n' }}\n" \
                    "{%- endif %}"

                # Step 1: Get all starting indices where item_index % 5 == 0
                start_indices = [i for i in range(len(jsons['eval_test'])) if i % 5 == 0]

                # Step 2: Randomly sample 40 starting indices
                selected_starts = random.sample(start_indices, 40)

                # Step 3: For each start, take that index and the next 4
                random_indices = []
                for start in selected_starts:
                    block = [start + offset for offset in range(5)]
                    random_indices.extend(block)
                # random_indices = list(range(len(jsons['eval_test'])))
                print('###### Making eval_test dataset templated prompts...')
                self.eval_test_dataset = {
                    "queries": self.tokenize_prompts(random.sample(self.get_templated_prompts([jsons['eval_test'][j] for j in random_indices], only_q=True, add_generation_prompt=not(config.args.judge_answer_match)), k=200), max_length=None),
                    "answers": [p['base']['correct_answer'] for p in jsons['eval_test']] if self.config.args.judge_answer_match else None,
                }
                if self.eval_test_dataset['queries']['input_ids'].shape[1] < self.max_len:
                    # print('######### OLD MAX LEN ', self.max_len)
                    # print('######### NEW MAX LEN ', self.max_len)
                    self.eval_test_dataset = {
                        "queries": self.tokenize_prompts(random.sample(self.get_templated_prompts([jsons['eval_test'][j] for j in random_indices], only_q=True, add_generation_prompt=not(config.args.judge_answer_match)), k=200), max_length=self.max_len),
                        "answers": [p['base']['correct_answer'] for p in jsons['eval_test']] if self.config.args.judge_answer_match else None,
                    }
                else:
                    # print('######### OLD MAX LEN ', self.max_len)
                    self.max_len = self.eval_test_dataset['queries']['input_ids'].shape[1]
                    # print('######### NEW MAX LEN ', self.max_len)
                # print('########## EVAL TEST DATASET EXAMPLE ########### ')
                # print(self.model_handler.tokenizer.convert_ids_to_tokens(self.eval_test_dataset['queries']['input_ids'][0]))
                
                self.no_generation_prompt_for_eval_test = False
                self.model_handler.tokenizer.chat_template = orig_template

        self.base_toks = {
            key: self.tokenize_prompts(base[key], max_length=self.max_len) for key in base
        }

        self.base_qs_toks = {
            key: self.tokenize_prompts(base_qs[key], max_length=self.max_len) for key in base_qs
        }

        self.source_qs_toks = {
            key: self.tokenize_prompts(source_qs[key], max_length=self.max_len) for key in source_qs
        }

        self.pyreft_prompts = self.get_templated_prompts(jsons['base_desired'], _base_completion=jsons['source_desired'], add_generation_prompt=False)
        self.pyreft_toks = self.tokenize_prompts(self.pyreft_prompts, max_length=self.max_len)
                 
        
        self.prompting_prompts = self.get_templated_prompts(jsons['prompting_base_desired'], add_generation_prompt=True)
        self.prompting_toks = self.tokenize_prompts(self.prompting_prompts, max_length=self.max_len)

        self.response_start_positions = {
            "base": {
                key: self.get_resp_start_pos(self.base_toks[key], self.model_handler.marker, self.model_handler.tokenizer) for key in self.base_toks
            },
            "pyreft": self.get_resp_start_pos(self.pyreft_toks, self.model_handler.marker, self.model_handler.tokenizer)
        }

        if self.config.args.eval_test_dataset:
            self.LEN = self.eval_test_dataset['queries']['input_ids'].shape[0]
        else:
            if self.config.args.eval_model:
                self.LEN = min(len(base['desired']), 100)
            else:
                self.LEN = min(len(base['desired']), 50)

        self.truncate_to_len(self.LEN)
    def truncate_to_len(self, L):
        self.LEN = L
        for key in self.base_toks:
            self.base_toks[key] = {k: v[:L] for k, v in self.base_toks[key].items()}
        for key in self.base_qs_toks:
            self.base_qs_toks[key] = {k: v[:L] for k, v in self.base_qs_toks[key].items()}
        for key in self.source_qs_toks:
            self.source_qs_toks[key] = {k: v[:L] for k, v in self.source_qs_toks[key].items()}
        self.pyreft_prompts = self.pyreft_prompts[:L]
        self.pyreft_toks = {k: v[:L] for k, v in self.pyreft_toks.items()}
        self.response_start_positions = {
            "base": {
                key: self.response_start_positions["base"][key][:L] for key in self.response_start_positions["base"]
            },
            "pyreft": self.response_start_positions["pyreft"][:L]
        }
    def filter_jsons(self, jsons):
        assert len(jsons['base_desired']) == len(jsons['base_undesired'])
        assert len(jsons['source_desired']) == len(jsons['source_undesired'])
        
        ids = [p['id'] for p in jsons['base_desired']]
        jsons['source_desired'] = [p for p in jsons['source_desired'] if p['id'] in ids]
        jsons['source_undesired'] = [p for p in jsons['source_undesired'] if p['id'] in ids]

        return jsons

    def get_templated_prompts(self, prompts, _base_completion=None, only_q=False, add_generation_prompt=False):
        if only_q:
            prompt_lengths = None
            if self.no_generation_prompt_for_eval_test:
                print('Explicitly setting add_generation_prompt to False for eval_test dataset.')
                add_generation_prompt = False
                # prompt_lengths = [len(p['prompt']) for p in prompts]
                # return [
                #     self.model_handler.tokenizer.apply_chat_template(
                #         [p['prompt'][i] for i in range(prompt_lengths[pdx])],
                #         add_generation_prompt=add_generation_prompt,
                #         tokenize=False
                #     ) for pdx, p in enumerate(prompts)]
                # return [
                #     re.sub(r"(The answer is \().*", r"\1", self.model_handler.tokenizer.apply_chat_template(
                #         [p['prompt'][i] for i in range(prompt_lengths[pdx])],
                #         add_generation_prompt=add_generation_prompt,
                #         tokenize=False
                #     )).strip() for pdx, p in enumerate(prompts)]
            # else:
                # print('Adding generation_prompt to False for eval_test dataset.')
            print(prompts[0]['prompt'], [p['role'] == 'assistant' for p in prompts[0]['prompt']])
            assistant_exists = any([p['role'] == 'assistant' for p in prompts[0]['prompt']])
            if assistant_exists:
                print('assistant exists in prompt...')
                prompt_lengths = [len(p['prompt']) - 1 for p in prompts]
            else:
                print('No assistant exists in prompt...')
                prompt_lengths = [max(len(p['prompt']), 1) for p in prompts]
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    [p['prompt'][i] for i in range(prompt_lengths[pdx])],
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False
                ) for pdx, p in enumerate(prompts)]
        elif _base_completion is not None:
            assert len(prompts) == len(_base_completion), f"Length of prompts and base completion do not match: {len(prompts)} vs {len(_base_completion)}"
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    [p['prompt'][i] for i in range(len(p['prompt']) - 1)] + [_base_completion[pi]['prompt'][-1]],
                    tokenize=False
                ) for pi, p in enumerate(prompts)]
        else:
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    p['prompt'], 
                    add_generation_prompt=False,
                    tokenize=False
                ) for p in prompts]
        
    def get_resp_start_pos(self, tokens, marker, tokenizer):
        print_now = True
        response_start_positions = []
        # print('###### Marker Tokens:', self.model_handler.alignment_tokens.tolist(), '######\n')
        # print('###### Marker String:', marker, '######\n')
        for _, tok in tqdm(enumerate(tokens['input_ids']), desc="Finding response start positions..."):
            tok = tok.to(self.device)
            response_start_position = None
            marker_tokens = self.model_handler.alignment_tokens.to(self.device)
            marker_len = marker_tokens.size(0)
            for j in range(tok.size(0) - marker_len + 1):
                if torch.equal(tok[j : j + marker_len], marker_tokens):
                    response_start_position = j + marker_len
                    break
            assert response_start_position is not None, f"Marker {repr(marker)} not found in input {repr(tokenizer.decode(tok))}.\nTOKENS: {repr(tok)}\nMARKER: {repr(marker_tokens)}"
            response_start_positions.append(response_start_position)
        #     if print_now:
        #         print('#################### RESPONSE START POSITIONS #######################')
        #         print('--------------------------------------------------------------------------------')
        #         print(f'"{repr(self.model_handler.tokenizer.decode(tok[:response_start_position], skip_special_tokens=False))}"')
        #         print(f'"{repr(self.model_handler.tokenizer.decode(tok[response_start_position:], skip_special_tokens=False))}"')
        #         print_now = False
        #         print('--------------------------------------------------------------------------------')
        # print(f"Found response start positions {response_start_positions} prompts.")
        return response_start_positions
    
    def tokenize_prompts(self, p, max_length=None):
        if max_length is None:
            tokens = self.model_handler.tokenizer(p, padding=True, truncation=False, return_tensors="pt")
        else:
            # print('Using max length:', max_length)
            tokens =  self.model_handler.tokenizer(p, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")
        return {"input_ids": tokens["input_ids"].to(self.device), "attention_mask": tokens["attention_mask"].to(self.device)}
    
    def decode_prompts(self, p):
        return self.model_handler.tokenizer.decode(p, skip_special_tokens=True)

    def align_toks(self, source_toks, base_toks):
        # print(source_toks['input_ids'], base_toks['input_ids'])
        """
        Align source_toks to base_toks at the assistant alignment token subsequence.
        """

        def find_subseq_start(seq, subseq):
            """Find first index where subseq occurs in seq, or -1 if not found."""
            n, m = len(seq), len(subseq)
            for i in range(n - m + 1):
                if seq[i:i+m] == subseq:
                    return i
            return -1

        toks_mod = {"input_ids": [], "attention_mask": []}
        align_seq = self.model_handler.alignment_tokens.tolist()
        pad_token_id = self.model_handler.tokenizer.pad_token_id

        # print(source_toks["input_ids"].shape, base_toks["input_ids"].shape)
        for i in range(len(base_toks["input_ids"])):
            src_ids = source_toks["input_ids"][i]

            src_mask = source_toks["attention_mask"][i]
            base_ids = base_toks["input_ids"][i]
            base_mask = base_toks["attention_mask"][i]

            src_list = src_ids.tolist()
            base_list = base_ids.tolist()

            # locate assistant token subsequence
            src_start = find_subseq_start(src_list, align_seq)
            base_start = find_subseq_start(base_list, align_seq)

            if src_start == -1 or base_start == -1:
                # print("Base decode:", self.model_handler.tokenizer.convert_ids_to_tokens(base_ids.tolist()))
                # print("Source decode:", self.model_handler.tokenizer.convert_ids_to_tokens(src_ids.tolist()))
                raise AssertionError(f"Assistant alignment sequence {align_seq} {self.model_handler.tokenizer.convert_ids_to_tokens(align_seq)} not found in example {i}. Source start: {src_start}, Base start: {base_start}")

            offset = src_start - base_start
            L = src_ids.size(0)

            if offset > 0:
                # shift left by offset
                pad_ids = src_ids.new_full((offset,), pad_token_id)
                pad_mask = src_mask.new_full((offset,), 0)
                aligned_ids = torch.cat([src_ids[offset:], pad_ids], dim=0)
                aligned_mask = torch.cat([src_mask[offset:], pad_mask], dim=0)
            elif offset < 0:
                # shift right by -offset
                k = -offset
                pad_ids = src_ids.new_full((k,), pad_token_id)
                pad_mask = src_mask.new_full((k,), 0)
                aligned_ids = torch.cat([pad_ids, src_ids[:L-k]], dim=0)
                aligned_mask = torch.cat([pad_mask, src_mask[:L-k]], dim=0)
            else:
                aligned_ids = src_ids
                aligned_mask = src_mask

            toks_mod["input_ids"].append(aligned_ids)
            toks_mod["attention_mask"].append(aligned_mask)

        toks_mod["input_ids"] = torch.stack(toks_mod["input_ids"])
        toks_mod["attention_mask"] = torch.stack(toks_mod["attention_mask"])
        return toks_mod


    @staticmethod
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
