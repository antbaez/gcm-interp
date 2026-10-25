
import os
import torch
import torch.nn.functional as F
import json
import ast
import re
from tqdm import tqdm
import pandas as pd

TARGET_BASES = {'prose': 'verse'}


def content_token_positions(tokenizer, templated, contents, toks, label="prompt"):
    """Per-row token indices locating each prompt's own content inside its chat template.

    `toks` is the tokenized `templated` batch with `offset_mapping`. Returns tensors of:
      content-start  first token of the content; everything before it is the pre-prompt
                     chat template (or left padding)
      last-prompt    final token of the content
      last-EOT       first special token after the content (e.g. <|eot_id|> / <|im_end|>),
                     or the token holding the character right after the content when the
                     template closes the turn without one, as OLMo does
    """
    special_ids = {i for i, t in tokenizer.added_tokens_decoder.items() if t.special}
    ids = toks['input_ids'].tolist()
    mask = toks['attention_mask'].tolist()
    offsets = toks['offset_mapping'].tolist()
    content_start, last_prompt, last_eot = [], [], []
    for i, (s, raw) in enumerate(zip(templated, contents)):
        content = raw.strip()
        start = s.rfind(content)
        if not content or start < 0:
            raise ValueError(f"{label} {i}: content not found in its templated prompt.")
        end = start + len(content)
        real = [j for j in range(len(ids[i])) if mask[i][j]]
        cs = next(j for j in real if offsets[i][j][1] > start)
        lp = next(j for j in real if offsets[i][j][0] <= end - 1 < offsets[i][j][1])
        eot = next((j for j in real if j > lp and ids[i][j] in special_ids), None)
        if eot is None:
            eot = next(j for j in real if offsets[i][j][0] <= end < offsets[i][j][1])
        content_start.append(cs)
        last_prompt.append(lp)
        last_eot.append(eot)
    return {'content-start': torch.tensor(content_start), 'last-prompt': torch.tensor(last_prompt),
            'last-EOT': torch.tensor(last_eot)}


def steer_mask_from_starts(content_start, seq_len):
    """[n, seq_len] bool: True from each row's content start onward (never the pre-prompt template or padding)."""
    return torch.arange(seq_len)[None, :] >= content_start[:, None]


class DataHandler:
    def __init__(self, config, model_handler):
        self.config = config
        self.model_handler = model_handler
        self.device = self.config.args.device

        file_paths = {
            'base_desired': f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.base}-desired-all.jsonl",
            'source_desired': f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.source}-desired-all.jsonl",
            'base_test': (
                f"{self.config.args.data_path}/{self.config.args.source_dir}/{self.config.args.base}-valid.jsonl"
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
            'source_desired': self.load_from_jsonl(file_paths['source_desired']),
            'base_test': self.load_from_jsonl(file_paths['base_test']) if self.config.args.eval_test else None,
            'steering_add': self.load_from_jsonl(file_paths['steering_add']) if self.config.args.steering_add_path else None,
            'steering_sub': self.load_from_jsonl(file_paths['steering_sub']) if self.config.args.steering_sub_path else None,
        }

        base = {
            'desired': self.get_templated_prompts(jsons['base_desired']),
        }

        base_qs = {
            'desired': self.get_templated_prompts(jsons['base_desired'], only_q=True, add_generation_prompt=True),
        }

        if self.config.args.eval_test:
            base_qs['test'] = self.get_templated_prompts(jsons['base_test'], only_q=True, add_generation_prompt=True)

        source_qs = {
            'desired': self.get_templated_prompts(jsons['source_desired'], only_q=True, add_generation_prompt=True),
        }

        steering = {
            "add_qs": self.get_templated_prompts(jsons['steering_add'], only_q=True, add_generation_prompt=True) if jsons['steering_add'] else None,
            "sub_qs": self.get_templated_prompts(jsons['steering_sub'], only_q=True, add_generation_prompt=True) if jsons['steering_sub'] else None
        }

        all_templated_prompts = base['desired'] + source_qs['desired']
        all_tokenized_prompts = self.tokenize_prompts(all_templated_prompts, max_length=None)
        self.max_len = all_tokenized_prompts['input_ids'].shape[1]
        print(f"[DataHandler] max_len: {self.max_len} (base + source full conversations)")

        gen_prompts = []
        if steering["add_qs"]: gen_prompts += steering["add_qs"]
        if steering["sub_qs"]: gen_prompts += steering["sub_qs"]
        # Pull in both eval splits (not just whichever is active via --split),
        # so gen_max_len — and therefore the padding baked into the steering
        # vector cache, which is shared across val/test runs by source alone —
        # stays identical regardless of which split built the cache.
        for suffix in ('-valid.jsonl', '-test.jsonl'):
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
                test_starts = self._content_positions(jsons['base_test'], self.base_qs_toks['test'], "test")['content-start']
                self.test_steer_mask = steer_mask_from_starts(test_starts, self.gen_max_len).to(self.device)

        self._test_json = jsons['base_test']
        self._steering_jsons = {'add': jsons['steering_add'], 'sub': jsons['steering_sub']}
        self._steering_positions = None
        if steering["add_qs"] and steering["sub_qs"]:
            self.steering_qs_toks = {
                "add": self.tokenize_prompts(steering["add_qs"], max_length=self.gen_max_len),
                "sub": self.tokenize_prompts(steering["sub_qs"], max_length=self.gen_max_len)
            }

        self.LEN = len(base['desired'])

    def get_templated_prompts(self, prompts, _base_completion=None, only_q=False, add_generation_prompt=False):
        extra_kwargs = self.model_handler.template_kwargs
        if only_q:
            return [
                self.model_handler.tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                    **extra_kwargs
                ) for msgs in self._query_messages(prompts)]
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
        
    def tokenize_prompts(self, p, max_length=None, return_offsets_mapping=False):
        kwargs = dict(truncation=False, return_tensors="pt", add_special_tokens=False,
                      return_offsets_mapping=return_offsets_mapping)
        if max_length is None:
            tokens = self.model_handler.tokenizer(p, padding=True, **kwargs)
        else:
            tokens = self.model_handler.tokenizer(p, padding='max_length', max_length=max_length, **kwargs)
        out = {"input_ids": tokens["input_ids"].to(self.device), "attention_mask": tokens["attention_mask"].to(self.device)}
        if return_offsets_mapping:
            out["offset_mapping"] = tokens["offset_mapping"]
        return out

    def _query_messages(self, prompts):
        """The messages each only_q prompt is templated from (see get_templated_prompts)."""
        assistant_exists = any([p['role'] == 'assistant' for p in prompts[0]['prompt']])
        if assistant_exists:
            prompt_lengths = [len(p['prompt']) - 1 for p in prompts]
        else:
            prompt_lengths = [max(len(p['prompt']), 1) for p in prompts]
        return [p['prompt'][:prompt_lengths[pdx]] for pdx, p in enumerate(prompts)]

    def steering_token_positions(self):
        """content_token_positions() for steering_qs_toks['add'/'sub'], keyed by side."""
        if self._steering_positions is None:
            positions = {}
            for side, rows in (('add', self._steering_jsons['add']), ('sub', self._steering_jsons['sub'])):
                positions[side] = self._content_positions(rows, self.steering_qs_toks[side], f"steering {side}")
            self._steering_positions = positions
        return self._steering_positions

    def test_token_positions(self):
        """content_token_positions() for base_qs_toks['test'] (the active eval split)."""
        return self._content_positions(self._test_json, self.base_qs_toks['test'], "test")

    def target_split(self):
        """(name, tokenized prompts, content-positions thunk) for the active eval split's
        target-concept counterpart: the same prompts with the base instruction swapped for the
        source's (prose-valid -> verse-valid, prose-test -> verse-test)."""
        args = self.config.args
        target = TARGET_BASES[args.base]
        name = target + args.test_dataset[len(args.base):]
        path = f"{args.data_path}/{args.source_dir}/{name}.jsonl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"--hook_target_acts needs {path}")
        rows = self.load_from_jsonl(path)
        toks = self.tokenize_prompts(self.get_templated_prompts(rows, only_q=True, add_generation_prompt=True),
                                     max_length=self.gen_max_len)
        return name, toks, lambda: self._content_positions(rows, toks, name)

    def steering_content_masks(self):
        """[n, P] bool per side: True from each steering prompt's content start onward."""
        positions = self.steering_token_positions()
        return {side: steer_mask_from_starts(positions[side]['content-start'], self.gen_max_len)
                for side in ('add', 'sub')}

    def _content_positions(self, rows, expected_toks, label):
        templated = self.get_templated_prompts(rows, only_q=True, add_generation_prompt=True)
        toks = self.tokenize_prompts(templated, max_length=self.gen_max_len, return_offsets_mapping=True)
        if not torch.equal(toks['input_ids'].cpu(), expected_toks['input_ids'].cpu()):
            raise RuntimeError(f"{label} prompts re-tokenized differently; positions would not line up.")
        contents = [msgs[-1]['content'] for msgs in self._query_messages(rows)]
        return content_token_positions(self.model_handler.tokenizer, templated, contents, toks, label)
    
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
