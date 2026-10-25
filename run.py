import sys
import json
from utils.config import Config
from utils.model_handler import ModelHandler
from utils.data_handler import DataHandler
import os
from eval.eval_runner import *
from eval.activations import save_prompt_activations, save_response_activations
import logging
logging.basicConfig(level=logging.WARNING)
from utils.batch_handler import BatchHandler

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
            heldout = f"./data/{model_name}/{source_dir}/{base}-test.jsonl"
            if not os.path.exists(heldout):
                raise SystemExit(
                    f"--split test needs {heldout}. Author the prompts under "
                    f"data/queries/ and run data_gen/install_heldout.py."
                )
            config.args.eval_test = heldout
            config.args.test_dataset = f"{base}-test"
        elif explicit_eval_test is None:
            config.args.test_dataset = f"{base}-valid"

        config.set_output_prefix()
        os.makedirs(config.get_output_prefix(), exist_ok=True)
        config.save_to_yaml(f"{config.output_prefix}/config.yml", config.args)

        print(f'\n=== Dataset: {source} -> {base} ===')
        data_handler = DataHandler(config, model_handler)

        if config.args.hook_prompt_acts or config.args.hook_response_acts or config.args.hook_target_acts:
            if source != 'verse-long':
                print(f'[hook_acts] only saves verse-long/prose activations, skipping {source}')
            else:
                # Training prompts (first 100 per side), then every prompt of the eval split
                # this run generates on (validation, or held-out under -split test).
                for hook_split, n in (('train', 100), ('eval', None)):
                    if config.args.hook_prompt_acts:
                        save_prompt_activations(model_handler.model, data_handler, n=n,
                                                batch_size=config.args.hook_batch_size, split=hook_split)
                    if config.args.hook_response_acts:
                        save_response_activations(model_handler.model, data_handler, n=n,
                                                  batch_size=config.args.hook_batch_size,
                                                  max_new_tokens=config.args.max_new_tokens, split=hook_split)
                if config.args.hook_target_acts:
                    save_prompt_activations(model_handler.model, data_handler, n=None,
                                            batch_size=config.args.hook_batch_size, split='target')
                    save_response_activations(model_handler.model, data_handler, n=None,
                                              batch_size=config.args.hook_batch_size,
                                              max_new_tokens=config.args.max_new_tokens, split='target')

        if config.args.eval_model:
            data_handler.LEN = min(data_handler.LEN, 200)
            config.args.batch_size = config.args.eval_batch_size
            batch_size = config.args.batch_size
            batch_handler = BatchHandler(config, data_handler, 0, min(batch_size, data_handler.LEN))

            if config.args.steering:
                # Baseline generation doesn't depend on steering_type — produce it at
                # most once per dataset, and only when some condition in the loop
                # below still needs generating. generate_baseline() itself reloads
                # it from disk when a matching saved baseline exists.
                baseline = []
                def get_baseline():
                    if not baseline:
                        baseline.append(generate_baseline(config, data_handler, model_handler))
                    return baseline[0]
                for steering_type in config.args.steering_types:
                    config.args.steering_type = steering_type
                    print(f'\n--- Steering type: {steering_type} ---')

                    if best_configs is not None:
                        task = f"from_{base}_to_{source}"
                        scope = 'global' if config.args.global_steer else 'local'
                        entry = (best_configs.get(model_name, {}).get(task, {})
                                 .get(scope, {}).get(steering_type))
                        if entry is None:
                            print(f'No validation-selected config for {model_name}/{task}/{scope}/{steering_type}, skipping')
                            continue
                        # Collapse the sweep to the single config chosen on validation,
                        # so the held-out split is never used to pick anything.
                        config.args.steering_n = [entry['N']]
                        if isinstance(entry['layer'], int):
                            config.args.layers = [entry['layer']]
                        score_key, score_label = (('val_harmonic_mean', 'harmonic') if config.args.metric == 'harmonic'
                                                  else ('val_pass_rate', 'w_rf'))
                        if score_key not in entry:
                            raise SystemExit(f"{config.args.best_configs} has no '{score_key}' for "
                                             f"{model_name}/{task}/{scope}/{steering_type}; "
                                             f"it was selected with a different -metric than '{config.args.metric}'")
                        print(f"Pinned from validation: N={entry['N']:g} layer={entry['layer']} "
                              f"(val {score_label}={entry[score_key]:.3f})")

                    run_eval(config, data_handler, model_handler, batch_handler, baseline_fn=get_baseline)

if __name__ == "__main__":
    main()
