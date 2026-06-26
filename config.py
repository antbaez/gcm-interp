from ast import parse
import torch
import os
import random
import numpy as np
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


        self.args.data_path = f"./data/{self.args.model_id.split('/')[-1]}/"

        if self.args.patch_algo == None:
            self.args.patch_algo = 'atp'
        self.setup_environment(seed=self.args.seed)

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Patching')
        parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device to run the model on')
        parser.add_argument('-model_id', '--model_id', type=str, required=True, help='Model ID for the model')
        parser.add_argument('-batch_size', '--batch_size', type=int, default=8, required=True, help='Batch size for patching')
        parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('-ablation', '--ablation', type=str, default='steer', help='Apply steering ablation')
        parser.add_argument('-patch_model', '--patch_model', action='store_true', help='Patch the model')
        parser.add_argument('-eval_model', '--eval_model', action='store_true', help='Evaluate the model')
        parser.add_argument("--eval_test", nargs="?", const=True, default=None, help="Evaluate on test set. Optionally provide a test set name/path.")
        parser.add_argument('-eval_train', '--eval_train', action='store_true', help='Evaluate the model on train set')
        parser.add_argument('-eval_transfer', '--eval_transfer', type=str, help='Path to the test dataset for evaluation')
        parser.add_argument('--steering', action='store_true', help='Steering Eval mode')
        parser.add_argument('--pyreft', action='store_true', help='Use PyReFT Eval Mode')
        parser.add_argument('-max_new_tokens', '--max_new_tokens', type=int, default=256, help='Max new tokens to generate during eval')
        parser.add_argument('-patch_algo', '--patch_algo', type=str, help='acp/atp? acp for activation patching, atp for attribution patching')
        parser.add_argument('-source', '--source', type=str, help='Patch from source')
        parser.add_argument('-base', '--base', type=str, help='Patch to base')
        parser.add_argument('-steering_add_path', '--steering_add_path', type=str, help='steering reps to add')
        parser.add_argument('-steering_sub_path', '--steering_sub_path', type=str, help='steering reps to subtract')
        parser.add_argument('-dataset_list', '--dataset_list', type=str, default=None,
            help='JSON array of dataset specs: [{"source":..,"base":..,"steering_add":..,"steering_sub":..}]')
        parser.add_argument('-vector_creation_batch_size', '--vector_creation_batch_size', type=int, default=5, help='batch size for computing the steering vector')
        parser.add_argument('-steering_batch_size', '--steering_batch_size', type=int, default=9, help='batch size for steered generation')
        parser.add_argument('-steering_n', '--steering_n', type=int, nargs='+', default=[1, 2, 4, 5, 6, 8, 10], help='steering strength multipliers to sweep')
        parser.add_argument('-topk_vals', '--topk_vals', type=float, nargs='+', default=[1.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5], help='top-k fractions of heads to sweep')
        parser.add_argument('-steering_type', '--steering_type', type=str, default='mean', choices=['last-token', 'mean', 'positional'], help='how to compute the steering vector from patch_activations')
        parser.add_argument('-steering_combos', '--steering_combos', type=str, default=None, help='JSON array of [steering_type, ...] strings to run sequentially in one process')
        parser.add_argument('--normalize', action='store_true', default=False, help='Normalize steering vectors to unit norm before scaling by N')

        args = parser.parse_args()
        if isinstance(args.eval_test, str) and args.eval_test.lower() == 'false':
            args.eval_test = False
        elif isinstance(args.eval_test, str) and args.eval_test.lower() == 'true':
            args.eval_test = True
        if not (args.patch_model or args.eval_model):
            parser.error("At least one of -patch_model, -eval_model is required")
        if args.patch_model or args.eval_model:
            if not args.patch_algo:
                parser.error("-patch_algo argument is required when --patch_model is set")
            if not args.dataset_list:
                if not args.source:
                    parser.error("-source is required when -dataset_list is not provided")
                if not args.base:
                    parser.error("-base is required when -dataset_list is not provided")

        if args.eval_model:

            if not args.eval_test:
                args.eval_train = True
            if isinstance(args.eval_test, str) and not os.path.exists(args.eval_test):
                parser.error(f"The provided eval_test path '{args.eval_test}' does not exist.")
            if isinstance(args.eval_test, str):
                args.test_dataset = args.eval_test.split('/')[-2]
                print(f"Steering dataset set to: {args.test_dataset}")
            elif isinstance(args.eval_test, bool) and args.steering:
                args.test_dataset = args.source  # may be None in dataset_list mode; set per-dataset in run.py

            if getattr(args, 'test_dataset', None) and 'single' in args.test_dataset and args.max_new_tokens == 256:
                args.max_new_tokens = 3
            

        return args

    def save_to_yaml(self, file_path, args):
        args_dict = vars(args)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(args_dict, yaml_file, default_flow_style=False)
            
    def setup_environment(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True, warn_only=True)

        if not self.args.dataset_list:
            os.makedirs(f'{self.set_output_prefix()}', exist_ok=True)
            self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
            print('Saved config')

    def get_output_prefix(self):
        return self.output_prefix
    
    def set_output_prefix(self):
        model = self.args.model_id.split('/')[-1]
        self.output_prefix = f"./results/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}"
        print(f"Saving to: {self.output_prefix}")
        return self.output_prefix
    
    def update_config(self, key, value):
        setattr(self.args, key, value)
        self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
