import sys
import json
from config import Config
from model_handler import ModelHandler
from data_handler import DataHandler
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

    model_name = config.args.model_id.split('/')[-1]
    # An explicitly passed --eval_test path wins over the per-split defaults below.
    explicit_eval_test = config.args.eval_test if isinstance(config.args.eval_test, str) else None

    best_configs = None
    if config.args.split == 'test':
        if not config.args.best_configs:
            raise SystemExit("--split test requires -best_configs <path>; run judge-evals/select_best_config.py first")
        with open(config.args.best_configs) as f:
            best_configs = json.load(f)

    for source, base, source_dir, add_path, sub_path in datasets:
        config.args.source = source
        config.args.base = base
        config.args.source_dir = source_dir
        config.args.steering_add_path = add_path
        config.args.steering_sub_path = sub_path

        # Select the eval split before DataHandler tokenizes it. This is also set
        # per dataset rather than once: the previous guard only assigned
        # test_dataset when it was unset, so every dataset after the first in a
        # multi-dataset run inherited the first one's split name.
        if config.args.split == 'test':
            heldout = f"./data/{model_name}/{source_dir}/{base}-heldout-test.jsonl"
            if not os.path.exists(heldout):
                raise SystemExit(
                    f"--split test needs {heldout}. Author the prompts under "
                    f"data/queries/ and run data_gen/install_heldout.py."
                )
            config.args.eval_test = heldout
            config.args.test_dataset = f"{base}-heldout-test"
        elif explicit_eval_test is None:
            config.args.test_dataset = f"{base}-test"

        config.set_output_prefix()
        os.makedirs(config.get_output_prefix(), exist_ok=True)
        config.save_to_yaml(f"{config.output_prefix}/config.yml", config.args)

        print(f'\n=== Dataset: {source} -> {base} ===')
        data_handler = DataHandler(config, model_handler)

        if config.args.eval_model:
            data_handler.LEN = min(data_handler.LEN, 200)
            config.args.batch_size = config.args.eval_batch_size
            batch_size = config.args.batch_size
            batch_handler = BatchHandler(config, data_handler, 0, min(batch_size, data_handler.LEN))

            if config.args.steering:
                # Baseline generation doesn't depend on steering_type — compute it once
                # per dataset instead of once per steering_type in the loop below.
                original_outputs = generate_baseline(config, data_handler, model_handler)
                for steering_type in config.args.steering_types:
                    config.args.steering_type = steering_type
                    print(f'\n--- Steering type: {steering_type} ---')

                    if best_configs is not None:
                        task = f"from_{source}_to_{base}"
                        entry = best_configs.get(model_name, {}).get(task, {}).get(steering_type)
                        if entry is None:
                            print(f'No validation-selected config for {model_name}/{task}/{steering_type}, skipping')
                            continue
                        # Collapse the sweep to the single config chosen on validation,
                        # so the held-out split is never used to pick anything.
                        config.args.steering_n = [entry['N']]
                        if isinstance(entry['layer'], int):
                            config.args.layers = [entry['layer']]
                        print(f"Pinned from validation: N={entry['N']:g} layer={entry['layer']} "
                              f"(val w_rf={entry['val_pass_rate']:.3f})")

                    run_eval(config, data_handler, model_handler, batch_handler, 'heads', original_outputs=original_outputs)

if __name__ == "__main__":
    main()
