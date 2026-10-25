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


        self.args.train_batch_size = self.args.batch_size
        self.args.data_path = f"./data/{self.args.model_id.split('/')[-1]}/"

        self.setup_environment(seed=self.args.seed)

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Steering')
        parser.add_argument('-d', '--device', type=str, default='cuda:1', required=True, help='Device to run the model on')
        parser.add_argument('-model_id', '--model_id', type=str, required=True, help='Model ID for the model')
        parser.add_argument('-batch_size', '--batch_size', type=int, default=8, required=True, help='Batch size for the steering-vector activation pass')
        parser.add_argument('-eval_batch_size', '--eval_batch_size', type=int, default=16, help='Batch size for generation during eval')
        parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('-ablation', '--ablation', type=str, default='steer', choices=['steer'], help='Apply steering ablation')
        parser.add_argument('-eval_model', '--eval_model', action='store_true', help='Evaluate the model')
        parser.add_argument("--eval_test", nargs="?", const=True, default=None, help="Evaluate on test set. Optionally provide a test set name/path.")
        parser.add_argument('--steering', action='store_true', help='Steering Eval mode')
        parser.add_argument('-max_new_tokens', '--max_new_tokens', type=int, default=256, help='Max new tokens to generate during eval')
        parser.add_argument('-source', '--source', nargs='+', help='Source dataset (one or more)')
        parser.add_argument('-base', '--base', nargs='+', help='Base dataset (one or more)')
        parser.add_argument('-steering_add_path', '--steering_add_path', nargs='+', help='steering reps to add (one per dataset)')
        parser.add_argument('-steering_sub_path', '--steering_sub_path', nargs='+', help='steering reps to subtract (one per dataset)')
        parser.add_argument('-source_dir', '--source_dir', nargs='+', default=None, help='Data subdirectory when it differs from source name (one per dataset)')
        parser.add_argument('-steering_n', '--steering_n', type=float, nargs='+', default=[1, 2, 4, 5, 6, 8, 10], help='Steering scale factors to sweep')
        parser.add_argument('-steering_types', '--steering_types', nargs='+', default=['last-token'], help='Steering vector aggregation types to run')
        parser.add_argument('-layers', '--layers', type=int, nargs='+', default=None, help='Explicit layers to steer, overriding the default layer-range sweep (used to pin a validation-selected layer)')
        parser.add_argument('-layer_range_start', '--layer_range_start', type=float, default=0.0, help='Start of the single-layer sweep range, as a fraction of total layers (default 0.0, i.e. from the first layer)')
        parser.add_argument('-layer_range_end', '--layer_range_end', type=float, default=2/3, help='End of the single-layer sweep range, as a fraction of total layers (default 2/3)')
        parser.add_argument('-layer_frac', '--layer_frac', type=float, default=None, help='Steer only layer round(layer_frac * num_layers) (half rounds up) instead of sweeping -layer_range_start/-layer_range_end; ignored when -split test pins a layer')
        parser.add_argument('-split', '--split', type=str, default='val', choices=['val', 'test'], help="Which eval split to generate on: 'val' sweeps N/layer on <base>-valid.jsonl; 'test' pins the validation-selected config and generates once on <base>-test.jsonl")
        parser.add_argument('-best_configs', '--best_configs', type=str, default=None, help='Path to best_configs.json from select_best_config.py (required for --split test)')
        parser.add_argument('-results_dir', '--results_dir', type=str, default='./results', help='Root generations are written under (<results_dir>/<model>/from_<base>_to_<source>/)')
        parser.add_argument('-metric', '--metric', choices=['harmonic', 'pass_rate'], default='harmonic', help='Validation score -best_configs was selected by (select_best_config.py --metric); picks val_harmonic_mean or val_pass_rate')
        parser.add_argument('--normalize', action='store_true', default=True, help='L2-normalize steering vectors before applying')
        parser.add_argument('--unnormalized', dest='normalize', action='store_false', help='Disable L2 normalization of steering vectors')
        parser.add_argument('--global', dest='global_steer', action='store_true', help='Steer all layers simultaneously (default is a single-layer sweep over -layer_range_start/-layer_range_end)')
        parser.add_argument('--hook_prompt_acts', action='store_true', help='Save layer round(2/3 * num_layers)\'s activations at all prompt tokens of the first 100 verse-long and prose training prompts to ./activations/')
        parser.add_argument('-hook_batch_size', '--hook_batch_size', type=int, default=10, help='Batch size for --hook_prompt_acts, --hook_response_acts and --hook_target_acts')
        parser.add_argument('--hook_target_acts', action='store_true', help='Greedily generate unsteered responses to the eval split\'s target-concept counterpart (verse-valid / verse-test: the same prompts asking for verse) and save layer round(2/3 * num_layers)\'s prompt and response activations to ./activations/')
        parser.add_argument('--hook_response_acts', action='store_true', help='Greedily generate unsteered responses to the first 100 verse-long and prose training prompts and save layer round(2/3 * num_layers)\'s activations at every response token to ./activations/')
        parser.add_argument('--thinking', action='store_true', help='Enable the reasoning block for chat templates that gate one (Qwen3, Gemma 4); off by default so completions are answer-only')
        parser.add_argument('--no-model-cache', dest='no_model_cache', action='store_true', help='Download the model/tokenizer to a temporary directory and delete it once loaded into memory (the default)')
        parser.add_argument('--model-cache', dest='no_model_cache', action='store_false', help='Keep the model/tokenizer in the persistent Hugging Face cache instead of deleting it after loading')
        parser.set_defaults(no_model_cache=True)

        args = parser.parse_args()
        if not args.eval_model:
            parser.error("-eval_model is required")
        if not args.source:
            parser.error("-source argument is required")
        if not args.base:
            parser.error("-base argument is required")

        # Only fall back to the default test file when no explicit path was
        # given; overwriting unconditionally would discard --eval_test and
        # make the isinstance(str) handling below unreachable.
        if not isinstance(args.eval_test, str):
            args.eval_test = True
        if isinstance(args.eval_test, str) and not os.path.exists(args.eval_test):
            parser.error(f"The provided eval_test path '{args.eval_test}' does not exist.")
        if isinstance(args.eval_test, str):
            # Name the split after the file itself, not its parent directory:
            # validation (`<base>-valid.jsonl`) and held-out test
            # (`<base>-test.jsonl`) live in the same dataset dir, so
            # keying off the directory would give both the same name and make
            # their generation outputs collide.
            args.test_dataset = os.path.splitext(os.path.basename(args.eval_test))[0]
            print(f"Steering dataset set to: {args.test_dataset}")

        args.max_new_tokens = 512

        return args

    def save_to_yaml(self, file_path, args):
        args_dict = vars(args)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(args_dict, yaml_file, default_flow_style=False)
            
    def setup_environment(self, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_output_prefix(self):
        return self.output_prefix
    
    def set_output_prefix(self):
        model = self.args.model_id.split('/')[-1]
        self.output_prefix = f"{self.args.results_dir.rstrip('/')}/{model}/from_{self.args.base}_to_{self.args.source}/"
        return self.output_prefix
    
    def update_config(self, key, value):
        setattr(self.args, key, value)
        self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
