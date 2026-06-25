from shutil import which
import torch
from tqdm import tqdm
import sys
from batch_handler import BatchHandler
from patching import Patching
from probes import LinearProbes
import einops
import logging
import os
import gc
import json
# from eval import Eval
# Set the logging level to WARNING to suppress DEBUG and INFO
logging.basicConfig(level=logging.WARNING)
# sys.stderr = open(os.devnull, 'w')
class Experiment:
    def __init__(self, config, data_handler, model_handler, which_patch):
        self.config = config
        self.data_handler = data_handler
        self.model_handler = model_handler
        self.batch_size = self.config.args.batch_size
        self.which_patch = which_patch
        self.batch_handler = BatchHandler(self.config, self.data_handler, 0, min(self.batch_size, self.data_handler.LEN))
        self.patching = Patching(self.model_handler, self.batch_handler, self.config)
        self.probes = LinearProbes(self.model_handler, self.batch_handler, self.config)
        self.patch_algo = self.config.args.patch_algo
        self.patching_logits = []

    def run(self):
        heads_dir = f'{self.config.get_output_prefix()}/heads'
        os.makedirs(heads_dir, exist_ok=True)
        for idx in tqdm(range(0, self.data_handler.LEN, self.batch_size)):
            if os.path.exists(f'{heads_dir}/{self.which_patch}_{idx}.pt'):
                continue
            start = idx
            stop = min(idx + self.batch_size, self.data_handler.LEN)
            self.batch_handler.update(start, stop)
            self.patching_logits = self.patching.apply_patching()
            self.save_logits(self.patching_logits, idx)

    def save_logits(self, logits, idx):
        heads_dir = f'{self.config.get_output_prefix()}/heads'
        torch.save(logits, f'{heads_dir}/{self.which_patch}_{idx}.pt')

    def run_probes(self):
        heads_dir = f'{self.config.get_output_prefix()}/heads'
        if os.path.exists(f'{heads_dir}/{self.which_patch}.json'):
            return
        print('Running probes...')
        self.accuracy = self.probes.get_accuracy()
        with open(f'{heads_dir}/{self.which_patch}.json', 'w') as f:
            json.dump(self.accuracy, f)