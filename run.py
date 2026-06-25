import sys
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

def main():
    print('Parsing config...')
    config = Config()
    print('Loading model...')
    model_handler = ModelHandler(config)

    source_dirs = config.args.source_dir if config.args.source_dir is not None else config.args.source

    datasets = list(zip(
        config.args.source,
        config.args.base,
        source_dirs,
        config.args.steering_add_path,
        config.args.steering_sub_path,
    ))

    for source, base, source_dir, add_path, sub_path in datasets:
        config.args.source = source
        config.args.base = base
        config.args.source_dir = source_dir
        config.args.steering_add_path = add_path
        config.args.steering_sub_path = sub_path

        config.set_output_prefix()
        os.makedirs(config.get_output_prefix(), exist_ok=True)
        config.save_to_yaml(f"{config.output_prefix}/config.yml", config.args)
        print(f'Saved config to file {config.get_output_prefix()}/config.yml')

        print(f'\n=== Dataset: {source} -> {base} ===')
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
            data_handler.LEN = min(data_handler.LEN, 50)
            config.args.batch_size = config.args.eval_batch_size
            if not hasattr(config.args, 'test_dataset') or not isinstance(config.args.test_dataset, str):
                config.args.test_dataset = f"{base}-test"
            batch_size = config.args.batch_size
            batch_handler = BatchHandler(config, data_handler, 0, min(batch_size, data_handler.LEN))
            patching = Patching(model_handler, batch_handler, config)
            patching_utils = PatchingUtils(patching)

            if config.args.pyreft:
                run_eval_pyreft(config, data_handler, model_handler, batch_handler)
            elif config.args.steering:
                for steering_type in config.args.steering_types:
                    config.args.steering_type = steering_type
                    print(f'\n--- Steering type: {steering_type} ---')
                    run_eval(config, data_handler, model_handler, batch_handler, patching_utils, 'heads')
            elif config.args.eval_transfer:
                data_handler.LEN = min(data_handler.LEN, 100)
                run_eval_transfer(config, data_handler, model_handler, batch_handler, patching_utils)

if __name__ == "__main__":
    main()
