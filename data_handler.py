
import os
import torch
import torch.nn.functional as F
import json
import ast
import re
from tqdm import tqdm
import pandas as pd
class DataHandler:
    def __init__(self, config, model_handler):
        self.config = config
        self.model_handler = model_handler
        self.device = self.config.args.device

        file_paths = {
            'base_desired': f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.base}-desired-all.jsonl",
            'base_undesired': f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.base}-undesired-all.jsonl",
            'source_desired': f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.source}-desired-all.jsonl",
            'source_undesired': f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.source}-undesired-all.jsonl",
            'base_test': (
                f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.base}-test.jsonl"
                if isinstance(self.config.args.eval_test, bool) and self.config.args.eval_test
                else f"{self.config.args.eval_test}"
                if isinstance(self.config.args.eval_test, str) and self.config.args.eval_test
                else None
            ),
            'steering_add': self.config.args.steering_add_path,
            'steering_sub': self.config.args.steering_sub_path
        }

        jsons = {
            'base_desired': self.load_from_jsonl(file_paths['base_desired']),
            'base_undesired': self.load_from_jsonl(file_paths['base_undesired']),
            'source_desired': self.load_from_jsonl(file_paths['source_desired']),
            'source_undesired': self.load_from_jsonl(file_paths['source_undesired']),
            'base_test': self.load_from_jsonl(file_paths['base_test']) if self.config.args.eval_test else None,
            'steering_add': self.load_from_jsonl(file_paths['steering_add']) if self.config.args.steering_add_path else None,
            'steering_sub': self.load_from_jsonl(file_paths['steering_sub']) if self.config.args.steering_sub_path else None,
        }

        base = {
            'desired': self.get_templated_prompts(jsons['base_desired']),
            'undesired': self.get_templated_prompts(jsons['base_undesired'])
        }

        base_qs = {
            'desired': self.get_templated_prompts(jsons['base_desired'], only_q=True, add_generation_prompt=True),
            'undesired': self.get_templated_prompts(jsons['base_undesired'], only_q=True, add_generation_prompt=True),
        }

        if self.config.args.eval_test:
            base_qs['test'] = self.get_templated_prompts(jsons['base_test'], only_q=True, add_generation_prompt=True)

        source_qs = {
            'desired': self.get_templated_prompts(jsons['source_desired'], only_q=True, add_generation_prompt=True),
            'undesired': self.get_templated_prompts(jsons['source_undesired'], only_q=True, add_generation_prompt=True)
        }
        
        steering = {
            "add_qs": self.get_templated_prompts(jsons['steering_add'], only_q=True, add_generation_prompt=True) if jsons['steering_add'] else None,
            "sub_qs": self.get_templated_prompts(jsons['steering_sub'], only_q=True, add_generation_prompt=True) if jsons['steering_sub'] else None
        }

        all_templated_prompts = base['desired'] + base['undesired'] + source_qs['desired'] + source_qs['undesired']
        all_tokenized_prompts = self.tokenize_prompts(all_templated_prompts, max_length=None)
        self.max_len = all_tokenized_prompts['input_ids'].shape[1]
        print(f"[DataHandler] Patching max_len: {self.max_len} (base + source full conversations)")

        gen_prompts = []
        if steering["add_qs"]: gen_prompts += steering["add_qs"]
        if steering["sub_qs"]: gen_prompts += steering["sub_qs"]
        # Pull in both eval splits (not just whichever is active via --split),
        # so gen_max_len — and therefore the padding baked into the steering
        # vector cache, which is shared across val/test runs by source alone —
        # stays identical regardless of which split built the cache.
        for suffix in ('-test.jsonl', '-heldout-test.jsonl'):
            split_path = f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.base}{suffix}"
            if os.path.exists(split_path):
                gen_prompts += self.get_templated_prompts(self.load_from_jsonl(split_path), only_q=True, add_generation_prompt=True)
        if gen_prompts:
            self.gen_max_len = self.tokenize_prompts(gen_prompts, max_length=None)['input_ids'].shape[1]
        else:
            self.gen_max_len = self.max_len
        print(f"[DataHandler] Generation max_len: {self.gen_max_len} (steering + val/heldout test prompts)")
        if self.config.args.eval_test:
            print(f"[DataHandler] Evaluating on {len(base_qs['test'])} prompts from {file_paths['base_test']}")

        if self.config.args.eval_model:
            if self.config.args.eval_test:
                self.base_qs_toks = {
                    'test': self.tokenize_prompts(base_qs['test'], max_length=self.gen_max_len)
                }

        if steering["add_qs"] and steering["sub_qs"]:
            self.steering_qs_toks = {
                "add": self.tokenize_prompts(steering["add_qs"], max_length=self.gen_max_len),
                "sub": self.tokenize_prompts(steering["sub_qs"], max_length=self.gen_max_len)
            }

        self.LEN = len(base['desired'])

    def get_templated_prompts(self, prompts, _base_completion=None, only_q=False, add_generation_prompt=False):
        extra_kwargs = self.model_handler.template_kwargs
        if only_q:
            prompt_lengths = None
            assistant_exists = any([p['role'] == 'assistant' for p in prompts[0]['prompt']])
            if assistant_exists:
                prompt_lengths = [len(p['prompt']) - 1 for p in prompts]
            else:
                prompt_lengths = [max(len(p['prompt']), 1) for p in prompts]
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    [p['prompt'][i] for i in range(prompt_lengths[pdx])],
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                    **extra_kwargs
                ) for pdx, p in enumerate(prompts)]
        elif _base_completion is not None:
            assert len(prompts) == len(_base_completion), f"Length of prompts and base completion do not match: {len(prompts)} vs {len(_base_completion)}"
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    [p['prompt'][i] for i in range(len(p['prompt']) - 1)] + [_base_completion[pi]['prompt'][-1]],
                    tokenize=False,
                    **extra_kwargs
                ) for pi, p in enumerate(prompts)]
        else:
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    p['prompt'],
                    add_generation_prompt=False,
                    tokenize=False,
                    **extra_kwargs
                ) for p in prompts]
        
    def tokenize_prompts(self, p, max_length=None):
        if max_length is None:
            tokens = self.model_handler.tokenizer(p, padding=True, truncation=False, return_tensors="pt")
        else:
            tokens = self.model_handler.tokenizer(p, padding='max_length', max_length=max_length, truncation=False, return_tensors="pt")
        return {"input_ids": tokens["input_ids"].to(self.device), "attention_mask": tokens["attention_mask"].to(self.device)}
    
    def decode_prompts(self, p):
        return self.model_handler.tokenizer.decode(p, skip_special_tokens=True)

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
