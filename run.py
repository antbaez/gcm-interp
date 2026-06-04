import sys
import json
import gc
import torch
from config import Config
from model_handler import ModelHandler
from data_handler import DataHandler
from experiment import Experiment
from patching_utils import PatchingUtils
from patching import Patching
import os
from eval.eval_runner import *
import logging
logging.basicConfig(level=logging.WARNING)
from batch_handler import BatchHandler
from patching import Patching

def combo_outputs_exist(config, topk_vals, n_vals):
    reps_type = 'random' if config.args.patch_algo == 'random' else 'targeted'
    ablation = config.args.ablation
    prefix = config.get_output_prefix()
    for N in n_vals:
        for topk in topk_vals:
            gen_file = (f"{prefix}/eval/{N}_{reps_type}_{ablation}_{topk}_"
                        f"{config.args.test_dataset}_{config.args.steering_type}_gen.txt")
            if not os.path.exists(gen_file) or not os.path.exists(gen_file.replace('.txt', '.json')):
                return False
    return True

def run_dataset(config, model_handler):
    config.args.batch_size = 5
    data_handler = DataHandler(config, model_handler)
    if config.args.patch_algo == 'acp':
        config.args.batch_size = max(x for x in range(64, 0, -1) if data_handler.LEN % x != 1)
    else:
        config.args.batch_size = 1

    if config.args.patch_model:
        if config.args.patch_algo == 'probes':
            probes_experiment = Experiment(config, data_handler, model_handler, 'heads')
            probes_experiment.run_probes()
        else:
            print('Running patching on heads...')
            heads_experiment = Experiment(config, data_handler, model_handler, 'heads')
            heads_experiment.run()

    if config.args.eval_model:
        config.args.batch_size = max(x for x in range(16, 0, -1) if data_handler.LEN % x != 1)
        if config.args.model_id == 'allenai/OLMo-2-1124-13B-DPO':
            if config.args.source == 'hate':
                config.args.batch_size = max(x for x in range(8, 0, -1) if data_handler.LEN % x != 1)
        batch_size = config.args.batch_size
        batch_handler = BatchHandler(config, data_handler, 0, min(batch_size, data_handler.LEN))
        patching = Patching(model_handler, batch_handler, config)
        patching_utils = PatchingUtils(patching)
        if config.args.pyreft:
            run_eval_pyreft(config, data_handler, model_handler, batch_handler)
        elif config.args.steering:
            if config.args.steering_combos:
                combos = json.loads(config.args.steering_combos)
                for steering_type in combos:
                    config.args.steering_type = steering_type
                    if combo_outputs_exist(config, config.args.topk_vals, config.args.steering_n):
                        print(f"[skip] steering_type={steering_type} — all outputs cached")
                        continue
                    print(f"\nRunning combo: steering_type={steering_type}")
                    run_eval(config, data_handler, model_handler, batch_handler, patching_utils, 'heads',
                             topk_vals=config.args.topk_vals, N=config.args.steering_n)
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                run_eval(config, data_handler, model_handler, batch_handler, patching_utils, 'heads',
                         topk_vals=config.args.topk_vals, N=config.args.steering_n)
        elif config.args.eval_transfer:
            data_handler.LEN = min(data_handler.LEN, 100)
            run_eval_transfer(config, data_handler, model_handler, batch_handler, patching_utils)

def main():
    print('Parsing config...')
    config = Config()
    model_handler = ModelHandler(config)

    if config.args.dataset_list:
        datasets = json.loads(config.args.dataset_list)
        for ds in datasets:
            print(f"\n=== Dataset: {ds['source']} -> {ds['base']} ===")
            config.args.source = ds['source']
            config.args.base = ds['base']
            config.args.steering_add_path = ds['steering_add']
            config.args.steering_sub_path = ds['steering_sub']
            config.args.test_dataset = ds['source']
            config.set_output_prefix()
            os.makedirs(config.get_output_prefix(), exist_ok=True)
            config.save_to_yaml(f"{config.get_output_prefix()}/config.yml", config.args)
            run_dataset(config, model_handler)
    else:
        run_dataset(config, model_handler)

if __name__ == "__main__":
    main()
