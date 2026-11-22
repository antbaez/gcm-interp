from ast import parse
import torch
import os
import random
import datetime
import time
import argparse
import sys
import yaml
from dataclasses import dataclass
import json
class Config:
    def __init__(self):
        self.args = self.parse_arguments()

        if self.args.source in ('harmful', 'harmless'):
            self.args.data_path = f"./data/{self.args.model_id.split('/')[-1]}/logits/harmful"
        elif self.args.source in ('hate', 'love'):
            self.args.data_path = f"./data/{self.args.model_id.split('/')[-1]}/logits/hate"
        elif self.args.source in ('verse', 'prose'):
            self.args.data_path = f"./data/{self.args.model_id.split('/')[-1]}/logits/verse"
        elif self.args.source in ('truth', 'lie'):
            self.args.data_path = f"./data/{self.args.model_id.split('/')[-1]}/logits/truth"

        #ISHA: Add a condition here for jailbreak
        if self.args.patch_algo == None:
            self.args.patch_algo = 'atp'
        self.setup_environment(seed=self.args.seed)

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Patching')
        parser.add_argument('-d', '--device', type=str, default='cuda:1', required=True, help='Device to run the model on')
        parser.add_argument('-model_id', '--model_id', type=str, required=True, help='Model ID for the model')
        parser.add_argument('-batch_size', '--batch_size', type=int, default=8, required=True, help='Batch size for patching')
        parser.add_argument('--batch_start', '-batch-start', type=int, default=0, help='Batch start idx')
        parser.add_argument('--N', '-N', type=int, default=10, help='Ablation Strength')
        parser.add_argument('-skip_existing', '--skip_existing', action='store_true', help='Skip existing patches')
        parser.add_argument('-vllm', '--vllm', action='store_true', help='Use vllm for inference')
        parser.add_argument('-aidevi', '--aidevi', action='store_true', help='AIDEVI nnsight remote')
        parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('-eval_extant', '--eval_extant', action='store_true', help='Evaluate on extant dataset')
        parser.add_argument('-ablation', '--ablation', type=str, default='steer', help='Apply steering ablation')
        # One or more of the following operations are possible:
        # ISHA: Use one of patch_model or eval_model
        parser.add_argument('-patch_model', '--patch_model', action='store_true', help='Patch the model')
        parser.add_argument('-eval_model', '--eval_model', action='store_true', help='Evaluate the model')
        parser.add_argument('-eval_test', '--eval_test', action='store_true', help='Evaluate the model on test set')
        # parser.add_argument('-eval_train', '--eval_train', action='store_true', help='Evaluate the model on train set')
        parser.add_argument('-eval_test_dataset', '--eval_test_dataset', type=str, help='Path to the test dataset for evaluation')
        parser.add_argument('--steering', action='store_true', help='Steering Eval mode')
        parser.add_argument('--pyreft', action='store_true', help='Use PyReFT Eval Mode')
        parser.add_argument('-attribution', '--attribution', action='store_true', help='get attn head attributions')
        parser.add_argument('-prompting', '--prompting', action='store_true', help='Use prompting eval mode')
        parser.add_argument('-get_head_reps_for_amir', '--get_head_reps_for_amir', action='store_true', help='Get head representations for Amir')
        parser.add_argument('-judge_answer_match', '--judge_answer_match', action='store_true', help='Judge answer match for eval')
        parser.add_argument('-max_new_tokens', '--max_new_tokens', type=int, default=256, help='Max new tokens to generate during eval')

        # required for patch model
        # ISHA: Use acp
        parser.add_argument('-patch_algo', '--patch_algo', type=str, help='acp/atp? acp for activation patching, atp for attribution patching')
        parser.add_argument('-source', '--source', type=str, help='Patch from source')
        parser.add_argument('-base', '--base', type=str, help='Patch to base')
        
        # required for eval model
        parser.add_argument('-judge_id', '--judge_id', type=str, help='Model ID for the judge model')
        parser.add_argument('-consistent_ablate', '--consistent_ablate', action='store_true', help='Consistently ablate tokens')

        args = parser.parse_args()
        if not (args.patch_model or args.eval_model):
            parser.error("At least one of -patch_model, -eval_model is required")
        if args.patch_model or args.eval_model:
            if not args.patch_algo:
                parser.error("-patch_algo argument is required when --patch_model is set")
            if not args.source:
                parser.error("-source argument is required when --patch_model is set")
            if not args.base:
                parser.error("-base argument is required when --patch_model is set")

        if args.eval_model:
            if not args.judge_id and not args.pyreft:
                parser.error("--judge_id argument is required when --eval_model is set")
            if not args.eval_test_dataset:
                args.eval_test = True

        return args

    def save_to_yaml(self, file_path, args):
        args_dict = vars(args)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(args_dict, yaml_file, default_flow_style=False)
            
    def setup_environment(self, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.makedirs(f'{self.set_output_prefix()}', exist_ok=True)
        self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
        print(f'Saved config to file {self.get_output_prefix()}/config.yml')

    def get_output_prefix(self):
        return self.output_prefix
    
    def set_output_prefix(self):
        model = self.args.model_id.split('/')[-1]
        if self.args.patch_model or self.args.eval_model:
            self.output_prefix = f"./normalized-results/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}/"
        print("op prefix ", self.output_prefix)
        return self.output_prefix
    
    def update_config(self, key, value):
        setattr(self.args, key, value)
        self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
