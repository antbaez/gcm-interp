from asyncio import log
# from setup import set_seed
from eval.logits_handler import load_logits, get_top_k_layer_and_head, retrieve_random_k
from eval.activations import mean_ablations_cache, steering_reps_cache
from eval.generation import select_gen_qs_toks, generate_with_patches, decode_responses
from eval.logits import compute_logit_scores, get_logits_before_patch, get_heads, patch_heads_and_get_logit
from eval.evaluation import plot_violin_comparison, compute_judge_accuracy, compute_judge_accuracy_test
from eval.pyreft_utils import get_reft_layers_config, reft_train, get_intervention_locations
import random
import os
import gc
import ast
import json
import pandas as pd
from tqdm import tqdm
import sys
import torch
from eval.attn_attributions import compute_attention_weights, get_attn_tensors, clean_token, plot_layer_heads
sys.path.append('../')  # Adjust path to import modules correctly
from batch_handler import BatchHandler
from eval.eval_extant import ExtantDatasetEvaluator
import pyreft
from model_handler import ModelHandler
best_mmlu_topk_combined = {
    'Qwen1.5-14B-Chat': {
        'from_harmful_to_harmless': [('random', 8, 0.08), ('random', 2, 0.5)],
        'from_hate_to_love': [('atp', 2, 0.07), ('atp', 2, 0.09)]
    },
    'OLMo-2-1124-13B-DPO': {
        'from_harmful_to_harmless': [('probes', 5, 0.04), ('probes', 9, 0.04)],
        'from_hate_to_love': [('acp', 4, 0.07), ('acp', 6, 0.06)],
        'from_verse_to_prose': [('atp', 2, 0.5), ('acp', 3, 0.5)]
    },
    'SOLAR-10.7B-Instruct-v1.0': {
        'from_harmful_to_harmless': [('random', 4, 0.5), ('atp', 6, 0.06)],
        'from_hate_to_love': [('probes', 5, 0.07), ('probes', 5, 0.09)]
    }
}
def load_patching_reps(data_handler, model_handler, mean=True):
    model = model_handler.model
    patching_reps = {}
    for ablation in [data_handler.config.args.ablation]:
        patching_reps[ablation] = {}
        for key in ['desired', 'undesired']:
            print(f"Loading patching reps for {ablation} - {key}")
            patching_reps[ablation][key] = get_patch_activations(model, data_handler, ablation, key=key, mean=mean)
    print('Returning patching reps')
    return patching_reps

def get_patch_activations(model, data_handler, ablation_type, key='desired', mean=True):
    if ablation_type == 'mean':
        return mean_ablations_cache(model, data_handler, key=key)
    elif ablation_type == 'steer':
        return steering_reps_cache(model, data_handler, key=key, mean=mean)
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

def save_prompt_responses(responses, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for entry in responses:
            for k, v in entry.items():
                f.write(f"{v}\n")
            f.write('-' * 40 + '\n')
    with open(path.replace('.txt', '.json'), 'w') as jf:
        json.dump(responses, jf)
    print(f"Saved responses to {path} and {path.replace('.txt', '.json')}")

def save_judge_accuracy(acc, path):
    with open(path, 'w') as f:
        json.dump(acc, f)
    print(f"Saved judge accuracy to {path}")

def call_judge(model_handler, config, data_handler, ablations, reps_types, topk_vals, gen_file=None):
    judge_model = model_handler.load_model(config.args.judge_id, config.args.device,model_type='judge')
    tokenizer = model_handler.load_tokenizer(config.args.judge_id)
    for ablation in tqdm(ablations):
        for reps_type in tqdm(reps_types, desc="Reps Types"):
            for topk in tqdm(topk_vals, desc="TopK Values"):
                if gen_file is None:
                    gen_file = f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_gen.json"
                    acc_path = f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_gen_accuracy.json"
                else:
                    acc_path = f"{gen_file.replace('.json', '_accuracy.json')}"
                print('gen file ', gen_file)
                print('acc path ', acc_path)
                with open(gen_file, 'r') as f:
                    decoded_responses = json.load(f)
                    # print(decoded_responses)
                print(f"Eval [[GEN]] → Ablation: {ablation}, Reps: {reps_type}, TopK: {topk}")
                print(acc_path)
                if not os.path.exists(acc_path):
                    assert len(decoded_responses) > 0, "No responses to evaluate."
                    if config.args.judge_answer_match:
                        acc, responses = compute_judge_accuracy_test(judge_model, tokenizer, decoded_responses, data_handler.judge_qs, config.args.base, config.args.source)
                        save_judge_accuracy(acc, acc_path)
                        save_judge_accuracy(responses, f"{config.get_output_prefix()}/eval_test/{config.args.N}_{config.args.eval_test_dataset}_{reps_type}_{ablation}_topk_{topk}_gen_accuracy_responses.json")
                    else:
                        acc, responses = compute_judge_accuracy(judge_model, tokenizer, decoded_responses, data_handler.judge_qs, config.args.base, config.args.source)
                        print('acc ', acc)
                        save_judge_accuracy(acc, acc_path)
                        save_judge_accuracy(responses, acc_path.replace("_accuracy.json", "_accuracy_responses.json"))
                gen_file = None
                acc_path = None

    del judge_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def save_top_k(reps_type, config, model, topk, logits, logit_metric):
    if reps_type == 'random':
        topk_df = retrieve_random_k(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            topk
        )
    else:
        topk_df = get_top_k_layer_and_head(logits, topk, config.args.patch_algo)

    print("Saving topk to CSV at ", f"{config.get_output_prefix()}/eval_test/{logit_metric}_{reps_type}_{topk}.csv")
    os.makedirs(f"{config.get_output_prefix()}/eval_test/", exist_ok=True)
    topk_df.to_csv(f"{config.get_output_prefix()}/eval_test/{logit_metric}_{reps_type}_{topk}.csv", index=False)
    return topk_df

def run_eval(config, data_handler, model_handler, batch_handler, patching_utils, which_patch, topk_vals=None, N=None):
    # set_seed()
    print("Starting evaluation...")

    model = model_handler.model
    if not config.args.patch_algo == 'random':
        if os.path.exists(f"{config.get_output_prefix()}/eval_test/numerator_1_{which_patch}.pt"):
            logits = torch.load(f"{config.get_output_prefix()}/eval_test/numerator_1_{which_patch}.pt")
        else:
            logits = load_logits(config, data_handler, which_patch, model_handler)
    else:
        logits = None
    patching_reps = load_patching_reps(data_handler, model_handler)
    ablations = [data_handler.config.args.ablation]
    reps_types = ['random'] if config.args.patch_algo == 'random' else ['targeted']

    if topk_vals is None:
        topk_vals = [0.03, 0.05, 0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1]
    if N is not None:
        config.args.N = N
    if config.args.patch_algo == 'probes':
        logit_metric = 'probes'
    elif config.args.patch_algo == 'random':
        logit_metric = 'random'
    else:
        logit_metric = 'numerator_1'

    print('topk_vals ', topk_vals)
    print('config.args.N ', config.args.N)
    decoded_responses = {}
    batch_handler = BatchHandler(config, data_handler)
    len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
    original_outputs = []
    pre_patch_logits = None
    for idx in tqdm(range(0, min(data_handler.LEN, len_gen_qs), config.args.batch_size)):
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        with model.generate(gen_qs_toks, do_sample=False, max_new_tokens=256) as _:
            op = model.generator.output.save()
        original_outputs += op.cpu().numpy().tolist()
        if pre_patch_logits is None:
            pre_patch_logits = get_logits_before_patch(
                model,
                batch_handler,
                patching_utils.get_response_logits)
        else:
            pre_patch_logits = torch.cat([pre_patch_logits, get_logits_before_patch(
                model,
                batch_handler,
                patching_utils.get_response_logits)], dim=-1)
        batch_handler.update()
    print('Starting for loop')
    for N in range(1, 11):
        config.args.N = N
        for ablation in tqdm(ablations, desc="Ablations"):
            decoded_responses[ablation] = {}
            for reps_type in tqdm(reps_types, desc="Reps Types"):
                decoded_responses[ablation][reps_type] = {}
                for topk in tqdm(topk_vals, desc="TopK Values"):
                    if os.path.exists(f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_gen.txt") and os.path.exists(f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_gen.json"):
                        print(f"Skipping evaluation for {ablation}, {reps_type}, {topk} {config.args.N} as gen files already exist.")
                        continue
                    patch_logits = None
                    decoded_responses[ablation][reps_type][topk] = []
                    gen_file = f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_gen.txt"
                    print(f"Eval [[LOGITS]] → Ablation: {ablation}, Reps: {reps_type}, TopK: {topk}, N: {config.args.N}, algo: {config.args.patch_algo}, task: {config.args.source} -> {config.args.base}")

                    if os.path.exists(gen_file) and os.path.exists(gen_file.replace('.txt', '.json')) and os.path.exists(f"{config.get_output_prefix()}/eval_test/{logit_metric}_{reps_type}_{topk}.csv"):
                    # and os.path.exists(f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_gen_accuracy.json"):
                        print(f"Skipping generation as all relevant files exist.")
                        continue
                    if not os.path.exists(f"{config.get_output_prefix()}/eval_test/{logit_metric}_{reps_type}_{topk}.csv"):
                        topk_df = save_top_k(reps_type, config, model, topk, logits, logit_metric)
                    else:
                        topk_df = pd.read_csv(f"./normalized-results/{config.args.model_id.split('/')[-1]}/from_{config.args.source}_to_{config.args.base}/{config.args.patch_algo}/eval_test/{logit_metric}_{reps_type}_{topk}.csv")

                    batch_handler = BatchHandler(config, data_handler)
                    len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
                    for idx in tqdm(range(0, min(data_handler.LEN, len_gen_qs), config.args.batch_size)):
                        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
                        # Compute logit scores after edit
                        post_patch_logit_scores = compute_logit_scores(
                            batch_handler,
                            topk_df,
                            patching_reps[ablation],
                            model_handler,
                            ablation,
                            patching_utils.get_response_logits,
                            config.args.N
                        )
                        edited_outputs = generate_with_patches(model, gen_qs_toks, patching_reps[ablation], topk_df, config.args.N, ablation, model_handler.dim, max_new_tokens=256, normalize=False)
                        decoded = decode_responses(model, gen_qs_toks, original_outputs[idx:idx+config.args.batch_size], edited_outputs, config.args.base)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if len(decoded_responses[ablation][reps_type][topk]) == 0:
                            patch_logits = post_patch_logit_scores
                            decoded_responses[ablation][reps_type][topk] = decoded
                        else:
                            patch_logits = torch.cat([patch_logits, post_patch_logit_scores], dim=-1)
                            decoded_responses[ablation][reps_type][topk] += decoded
                        batch_handler.update()
                    
                    os.makedirs(f"{config.get_output_prefix()}/eval_test/", exist_ok=True)
                    # torch.save(patch_logits, f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_patch_logits.pt")
                    # torch.save(pre_patch_logits, f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_pre_patch_logits.pt")
                    # plot_violin_comparison(patch_logits, topk, reps_type, config, logit_metric, ablation)
                    save_prompt_responses(decoded_responses[ablation][reps_type][topk], gen_file)

    # del model_handler.model
    # del model_handler.tokenizer
    # gc.collect()
    # torch.cuda.empty_cache()
    # call_judge(model_handler, config, data_handler, ablations, reps_types, topk_vals)

    print("Evaluation complete.")

def run_eval_extant(config, data_handler, model_handler, batch_handler, which_patch='heads'):
    print('Starting MMLU eval')

    if not config.args.patch_algo == 'random':
        logits = load_logits(config, data_handler, which_patch, model_handler)
    else:
        logits = None

    patching_reps = load_patching_reps(data_handler, model_handler)
    ablations = ['steer']
    reps_types = ['random' if config.args.patch_algo == 'random' else 'targeted']

    topk_vals = [0, 1, 0.5, 0.03, 0.05, 0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.1]
    if config.args.patch_algo == 'probes':
        logit_metric = 'probes' 
    elif config.args.patch_algo == 'random':
        logit_metric = 'random'
    else:
        logit_metric = 'numerator_1'

    decoded_responses = {}
    old_batch_size = config.args.batch_size
    for ablation in tqdm(ablations, desc='ablations'):
        decoded_responses[ablation] = {}
        for reps_type in tqdm(reps_types, desc='Reps Types'):
            decoded_responses[ablation][reps_type] = {}
            acc_file = f"{config.get_output_prefix()}/eval_test/0_extant_mmlu_{config.args.N}_{reps_type}_{ablation}.json"
            if os.path.exists(acc_file):
                print(f"Skipping generation as all relevant files exist.")
                continue
            for topk in tqdm(topk_vals, desc='TopK Values'):
                num_equal = 0
                print(f"Eval [[LOGITS]] → Ablation: {ablation}, Reps: {reps_type}, TopK: {topk}, N: {config.args.N}, algo: {config.args.patch_algo}, task: {config.args.source} -> {config.args.base}")
                decoded_responses[ablation][reps_type][topk] = []
                if not os.path.exists(f"{config.get_output_prefix()}/eval_test/{logit_metric}_{reps_type}_{topk}.csv"):
                    topk_df = save_top_k(reps_type, config, model_handler.model, topk, logits, logit_metric)
                else:
                    topk_df = pd.read_csv(f"./normalized-results/{config.args.model_id.split('/')[-1]}/from_{config.args.source}_to_{config.args.base}/{config.args.patch_algo}/eval_test/{logit_metric}_{reps_type}_{topk}.csv")

                if 'OLMo' in config.args.model_id:
                    config.args.batch_size = 16
                else:
                    config.args.batch_size = 32
                batch_handler = BatchHandler(config, data_handler)
                print('MMLU LEN ', data_handler.mmlu_prompts['input_ids'].shape[0])
                for idx in tqdm(range(0, data_handler.mmlu_prompts['input_ids'].shape[0], config.args.batch_size)):
                    top_tokens = patch_heads_and_get_logit(
                        model_handler.model,
                        model_handler.dim,
                        patching_reps[ablation]['desired'],
                        topk_df,
                        batch_handler.mmlu_toks,
                        None,
                        config.args.N,
                        ablation,
                        None,
                        top_tokens=True
                    )
                    top_decoded = model_handler.tokenizer.batch_decode(top_tokens, skip_special_tokens=True)
                    print('top decoded ', top_decoded)
                    print('answers ', batch_handler.mmlu_answers)
                    num_equal += sum(d == a for d, a in zip(top_decoded, batch_handler.mmlu_answers))
                    torch.cuda.empty_cache()
                    gc.collect()
                    batch_handler.update()
                with open(acc_file, 'a') as f:
                    f.write(f"{{'num-correct': {num_equal}, 'topk': {topk}, 'N': {config.args.N}}}\n")
    config.args.batch_size = old_batch_size
    print('Finished MMLU eval')

def run_eval_mean(config, data_handler, model_handler, batch_handler, patching_utils, which_patch='heads'):
    global best_algorithms
    if config.args.patch_algo != best_algorithms.get(config.args.model_id.split('/')[-1], {}).get(f"from_{config.args.source}_to_{config.args.base}", config.args.patch_algo):
        print(f"The chosen patching algorithm {config.args.patch_algo} is not the best known for the model {config.args.model_id.split('/')[-1]} and task from {config.args.source} to {config.args.base}. The best known algorithm is {best_algorithms.get(config.args.model_id.split('/')[-1], {}).get(f'from_{config.args.source}_to_{config.args.base}', 'N/A')}. Returning")
        return
    
    model = model_handler.model

    with open(f"{config.get_output_prefix().replace('normalized-results', 'runs')}/eval/gen_best_config_summary.json", 'r') as f:
        decoded_responses = json.load(f)
    try:
        decoded_responses = [d for d in decoded_responses if d["model"] == config.args.model_id.split('/')[-1]][0]
    except Exception as e:
        print(decoded_responses)
        raise ValueError(f"No responses found for model {config.args.model_id.split('/')[-1]} in the provided JSON file.")

    decoded_responses = decoded_responses["best"][f"from_{config.args.source}_to_{config.args.base}"][f"{config.args.patch_algo}"]
    topk_vals = [decoded_responses["topk"]]
    config.args.N = decoded_responses["sf"]
    if not config.args.patch_algo == 'random':
        if os.path.exists(f"{config.get_output_prefix()}/eval_test/numerator_1_{which_patch}.pt"):
            logits = torch.load(f"{config.get_output_prefix()}/eval_test/numerator_1_{which_patch}.pt")
        else:
            logits = load_logits(config, data_handler, which_patch, model_handler)
    else:
        logits = None
    patching_reps = load_patching_reps(data_handler, model_handler)
    ablations = ['mean']
    reps_types = ['random'] if config.args.patch_algo == 'random' else ['targeted']

    # topk_vals = [0.03, 0.05, 0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1]
    
    if config.args.patch_algo == 'probes':
        logit_metric = 'probes'
    elif config.args.patch_algo == 'random':
        logit_metric = 'random'
    else:
        logit_metric = 'numerator_1'

    decoded_responses = {}
    print('Starting for loop')
    for ablation in tqdm(ablations, desc="Ablations"):
        decoded_responses[ablation] = {}
        for reps_type in tqdm(reps_types, desc="Reps Types"):
            decoded_responses[ablation][reps_type] = {}
            for topk in tqdm(topk_vals, desc="TopK Values"):
                if os.path.exists(f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_patch_logits.pt") and os.path.exists(f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_pre_patch_logits.pt"):
                    print(f"Skipping evaluation for {ablation}, {reps_type}, {topk} {config.args.N} as patch logits already exist.")
                    continue
                patch_logits = None
                pre_patch_logits = None
                decoded_responses[ablation][reps_type][topk] = []
                gen_file = f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_gen.txt"
                print(f"Eval [[LOGITS]] → Ablation: {ablation}, Reps: {reps_type}, TopK: {topk}, N: {config.args.N}, algo: {config.args.patch_algo}, task: {config.args.source} -> {config.args.base}")

                if os.path.exists(gen_file) and os.path.exists(gen_file.replace('.txt', '.json')) and os.path.exists(f"{config.get_output_prefix()}/eval_test/{logit_metric}_{reps_type}_{topk}.csv") and os.path.exists(f"{config.get_output_prefix()}/eval_test/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_gen_accuracy.json"):
                    print(f"Skipping generation as all relevant files exist.")
                    continue
                if not os.path.exists(f"{config.get_output_prefix()}/eval_test/{logit_metric}_{reps_type}_{topk}.csv"):
                    topk_df = save_top_k(reps_type, config, model, topk, logits, logit_metric)
                else:
                    topk_df = pd.read_csv(f"./normalized-results/{config.args.model_id.split('/')[-1]}/from_{config.args.source}_to_{config.args.base}/{config.args.patch_algo}/eval_test/{logit_metric}_{reps_type}_{topk}.csv")

                batch_handler = BatchHandler(config, data_handler)
                len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
                for idx in tqdm(range(0, min(data_handler.LEN, len_gen_qs), config.args.batch_size)):
                    gen_qs_toks = select_gen_qs_toks(config, batch_handler)
                    # Compute logit scores after edit
                    # post_patch_logit_scores = compute_logit_scores(
                    #     batch_handler,
                    #     topk_df,
                    #     patching_reps[ablation],
                    #     model_handler,
                    #     ablation,
                    #     patching_utils.get_response_logits,
                    #     config.args.N
                    # )
                    # pre_patch_logit_scores = get_logits_before_patch(
                    #     model,
                    #     batch_handler,
                    #     patching_utils.get_response_logits)
                    
                    # Compute generations after edit
                    edited_outputs = generate_with_patches(model, gen_qs_toks, patching_reps[ablation], batch_handler.response_start_positions['base']['desired'], topk_df, config.args.N, ablation, model_handler.dim, max_new_tokens=128)
                    with model.generate(gen_qs_toks, do_sample=False, max_new_tokens=128) as _:
                        original_outputs = model.generator.output.save()
                    decoded = decode_responses(model, gen_qs_toks, original_outputs, edited_outputs, config.args.base)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if len(decoded_responses[ablation][reps_type][topk]) == 0:
                        # patch_logits = post_patch_logit_scores
                        # pre_patch_logits = pre_patch_logit_scores
                        decoded_responses[ablation][reps_type][topk] = decoded
                    else:
                        # patch_logits = torch.cat([patch_logits, post_patch_logit_scores], dim=-1)
                        # pre_patch_logits = torch.cat([pre_patch_logits, pre_patch_logit_scores], dim=-1)
                        decoded_responses[ablation][reps_type][topk] += decoded
                    batch_handler.update()
                
                os.makedirs(f"{config.get_output_prefix()}/eval_test/", exist_ok=True)
                # torch.save(patch_logits, f"{config.get_output_prefix()}/eval_test/mean_{config.args.N}_{reps_type}_{ablation}_topk_{topk}_patch_logits.pt")
                # torch.save(pre_patch_logits, f"{config.get_output_prefix()}/eval_test/mean_{config.args.N}_{reps_type}_{ablation}_topk_{topk}_pre_patch_logits.pt")
                # plot_violin_comparison(patch_logits, topk, reps_type, config, logit_metric, ablation)
                save_prompt_responses(decoded_responses[ablation][reps_type][topk], gen_file)
                break

    del model_handler.model
    del model_handler.tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    call_judge(model_handler, config, data_handler, ablations, reps_types, topk_vals, gen_file=gen_file.replace('.txt', '.json'))

    print("Evaluation complete.")

def run_eval_pyreft(config, data_handler, model_handler, batch_handler):
    global best_algorithms
    # if config.args.patch_algo != best_algorithms.get(config.args.model_id.split('/')[-1], {}).get(f"from_{config.args.source}_to_{config.args.base}", config.args.patch_algo):
    #     print(f"The chosen patching algorithm {config.args.patch_algo} is not the best known for the model {config.args.model_id.split('/')[-1]} and task from {config.args.source} to {config.args.base}. The best known algorithm is {best_algorithms.get(config.args.model_id.split('/')[-1], {}).get(f'from_{config.args.source}_to_{config.args.base}', 'N/A')}. Returning")
    #     return
    # with open(f"{config.get_output_prefix().replace('normalized-results', 'runs')}/eval/gen_best_config_summary.json", 'r') as f:
    #     decoded_responses = json.load(f)
    # try:
    #     decoded_responses = [d for d in decoded_responses if d["model"] == config.args.model_id.split('/')[-1]][0]
    # except Exception as e:
    #     print(decoded_responses)
    #     raise ValueError(f"No responses found for model {config.args.model_id.split('/')[-1]} in the provided JSON file.")

    # decoded_responses = decoded_responses["best"][f"from_{config.args.source}_to_{config.args.base}"][f"{config.args.patch_algo}"]
    # topk_vals = [decoded_responses["topk"]]
    topk_vals = [0.03, 0.05, 0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1]
    model = model_handler.model
    model_dirty = False
    print(f"Running PyReFT evaluation with topk values: {topk_vals} and patching algorithm: {config.args.patch_algo}")
    original_outputs = []
    len_gen_qs = select_gen_qs_toks(config, data_handler)['input_ids'].shape[0]
    for idx in tqdm(range(0, min(len_gen_qs, data_handler.LEN), config.args.batch_size)):
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        op = model.generate(**gen_qs_toks, do_sample=False, max_new_tokens=256)
        original_outputs += op.cpu().numpy().tolist()
        batch_handler.update()
    print('Original inputs length ', len(original_outputs))
    for topk in tqdm(topk_vals, desc="TopK Values"):
        if config.args.patch_algo == 'random':
            topk_df = pd.read_csv(f"{config.get_output_prefix().replace('normalized-results', 'runs')}/eval/random_random_{topk}.csv")
        elif config.args.patch_algo == 'probes':
            topk_df = pd.read_csv(f"{config.get_output_prefix().replace('normalized-results', 'runs')}/eval/probes_targeted_{topk}.csv")    
        else:
            topk_df = pd.read_csv(f"{config.get_output_prefix().replace('normalized-results', 'runs')}/eval/numerator_1_targeted_{topk}.csv")
        reps = "targeted" if config.args.patch_algo != 'random' else "random"
        gen_files = [f"{config.get_output_prefix()}/eval/{N}_{reps}_pyreft_topk_{topk}_gen.txt" for N in range(1, 11)]
        if all(os.path.exists(f) and os.path.exists(f.replace('.txt', '.json')) for f in gen_files):
            print(f"All generations for PyReFT with topk {topk} exist. Skipping to next topk.")
            continue
        if model_dirty:
            del reft_model
            model_handler.model = None
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = model_handler.load_model(config.args.model_id, config.args.device)
            model.tokenizer = model_handler.tokenizer
            model_handler.model = model
        reft_layers_config = get_reft_layers_config(topk_df, model)

        reft_model = pyreft.get_reft_model(model, reft_layers_config, set_device=False)
        reft_model.set_device(config.args.device, set_model=False)
        reft_model.print_trainable_parameters()
        data = data_handler.pyreft_prompts
        # print('pyreft_prompts ', data[:5])
        cf_data = {
            'input_ids': data_handler.pyreft_toks['input_ids'].clone(),
            'attention_mask': data_handler.pyreft_toks['attention_mask'].clone()
        }

        # print(cf_data['input_ids'][0], cf_data['attention_mask'][0],)
        # print('pyreft_prompts ', model_handler.tokenizer.batch_decode(cf_data['input_ids'][:5], skip_special_tokens=True))
        labels = cf_data['input_ids'].clone()
        for i in range(labels.shape[0]):
            start_pos = data_handler.response_start_positions['pyreft'][i]
            labels[i, :start_pos] = -100  # Ignore tokens before the response start position
        labels[cf_data['attention_mask'] == 0] = -100
        response_starts = torch.tensor(data_handler.response_start_positions['pyreft'], device=labels.device)
        prompt_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < response_starts[:, None]
        reft_model = reft_train(topk_df, reft_model, cf_data, labels, prompt_mask, batch_size=10, lr=4e-3, num_epochs=100, device=config.args.device, display_bar=True)
        # reft_model = train_params['model']
        for N in range(1, 11):
            batch_handler = BatchHandler(config, data_handler)
            decoded_responses = {
                "pyreft": {
                    reps: {
                        str(topk): []
                    }
                }
            }
            gen_file = f"{config.get_output_prefix()}/eval/{N}_{reps}_pyreft_topk_{topk}_gen.txt"
            print('Entering generation loop for PyReFT...')
            if os.path.exists(gen_file) and os.path.exists(gen_file.replace('.txt', '.json')):
            # and os.path.exists(f"{config.get_output_prefix()}/eval/{N}_{reps}_pyreft_topk_{topk}_gen_accuracy.json"):
                print(f"Skipping generation as all relevant files exist.")
                continue
            for idx in tqdm(range(0, min(len_gen_qs, data_handler.LEN), config.args.batch_size)):
                gen_qs_toks = select_gen_qs_toks(config, batch_handler)
                topk_heads = get_intervention_locations(topk_df, gen_qs_toks["input_ids"])
                _, edited_outputs = reft_model.generate(
                    gen_qs_toks, unit_locations={
                        'sources->base': (
                            None,  # copy from
                            topk_heads  # paste to
                        )
                    },
                    intervene_on_prompt=True, max_new_tokens=256, do_sample=False,
                    eos_token_id=model_handler.tokenizer.eos_token_id, early_stopping=True,
                    intervention_additional_kwargs={'S': N}
                )
                decoded = decode_responses(model, gen_qs_toks, original_outputs[idx:idx + config.args.batch_size], edited_outputs, config.args.base)
                decoded_responses["pyreft"][reps][str(topk)] += decoded

                batch_handler.update()
            save_prompt_responses(decoded_responses["pyreft"][reps][str(topk)], gen_file)
            # call_judge(model_handler, config, data_handler, ["pyreft"], [reps], [topk], gen_file=gen_file.replace('.txt', '.json'))

        model_dirty = True

def run_eval_attributions(config, data_handler, model_handler, batch_handler):
    model = model_handler.model

    patching_reps = load_patching_reps(data_handler, model_handler)
    ablation = 'steer'
    reps_type = 'random' if config.args.patch_algo == 'random' else 'targeted'
    with open(f"{config.get_output_prefix().replace('normalized-results', 'runs')}/eval/gen_best_config_summary.json", 'r') as f:
        decoded_responses = json.load(f)
    try:
        decoded_responses = [d for d in decoded_responses if d["model"] == config.args.model_id.split('/')[-1]][0]
    except Exception as e:
        print(decoded_responses)
        raise ValueError(f"No responses found for model {config.args.model_id.split('/')[-1]} in the provided JSON file.")

    decoded_responses = decoded_responses["best"][f"from_{config.args.source}_to_{config.args.base}"][f"{config.args.patch_algo}"]
    topk = decoded_responses["topk"]
    sf = decoded_responses["sf"]
    config.args.N = sf

    if config.args.patch_algo == 'probes':
        logit_metric = 'probes'
    elif config.args.patch_algo == 'random':
        logit_metric = 'random'
    else:
        logit_metric = 'numerator_1'
    topk_df = pd.read_csv(f"{config.get_output_prefix().replace('normalized-results', 'runs')}/eval/{logit_metric}_{reps_type}_{topk}.csv")
    batch_handler = BatchHandler(config, data_handler)
    edited_outputs = []
    original_outputs = []
    data_handler.LEN = 50
    attn_pres = []
    attn_posts = []
    print('Batch size: ', config.args.batch_size)
    for idx in tqdm(range(0, data_handler.LEN, config.args.batch_size)):
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        edited_output, post_q_tensor, post_k_tensor = get_attn_tensors(model, gen_qs_toks, patching_reps[ablation], topk_df, config.args.N, ablation, model_handler.dim, edit=True)
        original_output, pre_q_tensor, pre_k_tensor = get_attn_tensors(model, gen_qs_toks, patching_reps[ablation], topk_df, config.args.N, ablation, model_handler.dim, edit=False)

        if not 'attn_pre' in locals():
            attn_pres.append(compute_attention_weights(pre_q_tensor, pre_k_tensor).squeeze(2))
            attn_posts.append(compute_attention_weights(post_q_tensor, post_k_tensor).squeeze(2))
        else:
            attn_pres.append(compute_attention_weights(pre_q_tensor, pre_k_tensor).squeeze(2))
            attn_posts.append(compute_attention_weights(post_q_tensor, post_k_tensor).squeeze(2))
            # attn_pre = torch.cat([attn_pre, compute_attention_weights(pre_q_tensor, pre_k_tensor)], dim=2)
            # attn_post = torch.cat([attn_post, compute_attention_weights(post_q_tensor, post_k_tensor)], dim=2)
        edited_outputs.append(edited_output.squeeze(0))
        original_outputs.append(original_output.squeeze(0))
        batch_handler.update()
        gc.collect()
        torch.cuda.empty_cache()
    layer_head_list = []
    for _, row in topk_df.iterrows():
        layer = row['layer']
        head = row['neuron']
        layer_head_list.append((layer, head))
    for idx in range(data_handler.LEN):
        os.makedirs(f"{config.get_output_prefix()}/eval/attributions/", exist_ok=True)
        attn_pre_idxed = attn_pres[idx]
        attn_post_idxed = attn_posts[idx]
        plot_layer_heads(attn_pre_idxed, attn_post_idxed, layer_head_list, gen_qs_toks['input_ids'].shape[-1], model_handler.tokenizer.convert_ids_to_tokens(original_outputs[idx]), model_handler.tokenizer.convert_ids_to_tokens(edited_outputs[idx]), f"{config.get_output_prefix()}/eval/attributions/attn_heads_{reps_type}_{ablation}_topk_{topk}_idx_{idx}.png")

    # attn_pre_avg = torch.mean(torch.stack(attn_pres), dim=0)
    # attn_post_avg = torch.mean(torch.stack(attn_posts), dim=0)
    # plot_layer_heads(attn_pre_avg, attn_post_avg, layer_head_list, gen_qs_toks['input_ids'].shape[-1], model_handler.tokenizer.convert_ids_to_tokens(original_outputs[0]), model_handler.tokenizer.convert_ids_to_tokens(edited_outputs[0]), f"{config.get_output_prefix()}/eval/attributions/attn_heads_{reps_type}_{ablation}_topk_{topk}_avg.png", avg=True)

def run_eval_test_data(config, data_handler, model_handler, batch_handler, patching_utils):
    
    # Needs N= 1, topk=1 for transfer to be done explicitly. Not baked into the code rn.
    best_algorithms = pd.read_csv('/mnt/align4_drive/arunas/rm-interp-minimal/normalized-results/new-accuracies/plots/best_topk_N_per_method_per_ablation.csv')
    best_algorithms = best_algorithms[best_algorithms['ablation'] == config.args.ablation]


    best_method = ast.literal_eval(best_algorithms[(best_algorithms['model'] == config.args.model_id.split('/')[-1]) & (best_algorithms['task'] == f"from_{config.args.source}_to_{config.args.base}") & (best_algorithms['ablation'] == config.args.ablation)]['pairs'].item())

    if len(best_method) > 1:
        best_method = random.Random(42).choice(best_method)
    else:
        best_method = best_method[0]

    model = model_handler.model
    patching_reps = load_patching_reps(data_handler, model_handler)
    ablation = data_handler.config.args.ablation
    reps_type = 'random' if config.args.patch_algo == 'random' else 'targeted'

    print(best_method)
    config.args.N = int(best_method['steering_factor'])
    topk = float(best_algorithms[(best_algorithms['model'] == config.args.model_id.split('/')[-1]) & (best_algorithms['task'] == f"from_{config.args.source}_to_{config.args.base}") & (best_algorithms['ablation'] == config.args.ablation)]['best_topk'].item())
    config.args.patch_algo = best_method['method']
    test_accuracy = float(best_method['accuracy'])
    print(f"Using topk: {topk}, N: {config.args.N}, algo: {config.args.patch_algo} for evaluation on transfer data (Test accuracy was {test_accuracy}).")
    ablation = data_handler.config.args.ablation
    
    config.set_output_prefix()
    os.makedirs(f"{config.get_output_prefix()}/eval_test/", exist_ok=True)
    if config.args.patch_algo == 'probes':
        logit_metric = 'probes'
    elif config.args.patch_algo == 'random':
        logit_metric = 'random'
    else:
        logit_metric = 'numerator_1'

    topk_df = pd.read_csv(f"{config.get_output_prefix().replace('normalized-results', 'runs')}/eval/{logit_metric}_{reps_type}_{topk}.csv")
    edited_outputs = []
    original_outputs = []

    decoded_responses = {
        ablation: {
            reps_type: {
                topk: []
            }
        }
    }
    batch_handler = BatchHandler(config, data_handler)
    print('Best method: ', config.args.patch_algo, ' N: ', config.args.N, ' topk: ', topk, ' test accuracy ', )
    for idx in tqdm(range(0, data_handler.LEN, config.args.batch_size), desc="Processing samples"):
        gen_file = f"{config.get_output_prefix()}/eval_test/{config.args.N}_{config.args.eval_test_dataset}_{reps_type}_{ablation}_topk_{topk}_gen_{config.args.seed}.txt"
        if os.path.exists(gen_file.replace('.txt', '_accuracy_responses.json')):
            print(f"Skipping generation as all relevant files exist.")
            return
        gen_qs_toks = select_gen_qs_toks(config, batch_handler)
        edited_outputs = generate_with_patches(model, gen_qs_toks, patching_reps[ablation], topk_df, config.args.N, ablation, model_handler.dim, max_new_tokens=256, normalize=False)
        with model.generate(gen_qs_toks, do_sample=False, max_new_tokens=256) as _:
            original_outputs = model.generator.output.save()
        if config.args.eval_test_dataset:
            answers = batch_handler.eval_test_dataset['answers']
        else:
            answers = None
        decoded = decode_responses(model, gen_qs_toks, original_outputs, edited_outputs, config.args.base, answers=answers)

        if len(decoded_responses[ablation][reps_type][topk]) == 0:
            decoded_responses[ablation][reps_type][topk] = decoded
        else:
            decoded_responses[ablation][reps_type][topk] += decoded
        batch_handler.update()

        os.makedirs(f"{config.get_output_prefix()}/eval_test/", exist_ok=True)
        save_prompt_responses(decoded_responses[ablation][reps_type][topk], gen_file)

    del model_handler.model
    del model_handler.tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    call_judge(model_handler, config, data_handler, [ablation], [reps_type], [topk], gen_file=gen_file.replace('.txt', '.json'))
  
def get_head_representations(config, data_handler, model_handler, batch_handler):
    # set_seed()
    print("Starting evaluation...")

    model = model_handler.model
    patching_reps = load_patching_reps(data_handler, model_handler, mean=False)
    ablations = ['steer']
    reps_types = ['random'] if config.args.patch_algo == 'random' else ['targeted']

    topk_vals = [1]
    if config.args.patch_algo == 'probes':
        logit_metric = 'probes'
    elif config.args.patch_algo == 'random':
        logit_metric = 'random'
    else:
        logit_metric = 'numerator_1'

    decoded_responses = {}
    pre_patch_head_representations_base = {}
    patch_head_representations_steer = {}
    pre_patch_head_representations_source = {}
    for ablation in tqdm(ablations, desc="Ablations"):
        decoded_responses[ablation] = {}
        for reps_type in tqdm(reps_types, desc="Reps Types"):
            decoded_responses[ablation][reps_type] = {}
            for topk in tqdm(topk_vals, desc="TopK Values"):
                decoded_responses[ablation][reps_type][topk] = []
                print(f"Eval [[head representations]] → Ablation: {ablation}, Reps: {reps_type}, TopK: {topk}, N: {config.args.N}, algo: {config.args.patch_algo}, task: {config.args.source} -> {config.args.base}")
                batch_handler = BatchHandler(config, data_handler)
                for idx in tqdm(range(0, data_handler.LEN, config.args.batch_size)):
                    # Compute logit scores after edit
                    for key in ['desired', 'undesired']:
                        if not key in pre_patch_head_representations_base:
                            pre_patch_head_representations_base[key] = get_heads(
                                model_handler.model,
                                model_handler.dim,
                                patching_reps[ablation],
                                batch_handler.base_qs_toks[key],
                                config.args.N,
                                ablation,
                                patch=False
                            )
                            patch_head_representations_steer[key] = patching_reps[ablation][key].clone().view(patching_reps[ablation][key].shape[0], patching_reps[ablation][key].shape[1], patching_reps[ablation][key].shape[2], model_handler.num_heads, model_handler.dim).permute(0, 1, 3, 2, 4)
                            pre_patch_head_representations_source[key] = get_heads(
                                model_handler.model,
                                model_handler.dim,
                                patching_reps[ablation],
                                batch_handler.source_qs_toks[key],
                                config.args.N,
                                ablation,
                                patch=False
                            )
                            print(" 1 ", pre_patch_head_representations_base[key].shape, patch_head_representations_steer[key].shape, pre_patch_head_representations_source[key].shape)
                        else:
                            pre_patch_head_representations_base[key] = torch.cat([
                                pre_patch_head_representations_base[key],
                                get_heads(
                                    model_handler.model,
                                    model_handler.dim,
                                    patching_reps[ablation],
                                    batch_handler.base_qs_toks[key],
                                    config.args.N,
                                    ablation,
                                    patch=False
                                )
                            ], dim=1)
                            pre_patch_head_representations_source[key] = torch.cat([
                                pre_patch_head_representations_source[key],
                                get_heads(
                                    model_handler.model,
                                    model_handler.dim,
                                    patching_reps[ablation],
                                    batch_handler.source_qs_toks[key],
                                    config.args.N,
                                    ablation,
                                    patch=False
                                )
                            ], dim=1)
                            print(" 2 ", pre_patch_head_representations_base[key].shape, patch_head_representations_steer[key].shape, pre_patch_head_representations_source[key].shape)
                    batch_handler.update()
                print("FINAL ", pre_patch_head_representations_base['desired'].shape, patch_head_representations_steer['desired'].shape, pre_patch_head_representations_source['desired'].shape)
                os.makedirs(f"{config.get_output_prefix()}/heads/", exist_ok=True)
                torch.save(pre_patch_head_representations_base, f"{config.get_output_prefix()}/heads/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_pre_patch.pt")
                torch.save(patch_head_representations_steer, f"{config.get_output_prefix()}/heads/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_patched_steer.pt")
                torch.save(pre_patch_head_representations_source, f"{config.get_output_prefix()}/heads/{config.args.N}_{reps_type}_{ablation}_topk_{topk}_patched_source.pt")

    del model_handler.model
    del model_handler.tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def run_eval_prompting(config, data_handler, model_handler, batch_handler):
    model = model_handler.model
    decoded_responses = {
        'prompting': {
            'prompting': {
                '0': []
            }
        }
    }
    batch_handler = BatchHandler(config, data_handler)
    for idx in tqdm(range(0, data_handler.LEN, config.args.batch_size)):
        edit_gen_qs_toks = batch_handler.prompting_toks
        gen_qs_toks = batch_handler.base_qs_toks['desired']
        with model.generate(edit_gen_qs_toks, do_sample=False, max_new_tokens=256) as _:
            edited_outputs = model.generator.output.save()
        with model.generate(gen_qs_toks, do_sample=False, max_new_tokens=256) as _:
            original_outputs = model.generator.output.save()
        decoded = decode_responses(model, gen_qs_toks, original_outputs, edited_outputs, config.args.base)
        decoded_responses['prompting']['prompting']['0'] += decoded
        batch_handler.update()
    save_prompt_responses(decoded_responses['prompting']['prompting']['0'], f"{config.get_output_prefix()}/eval/0_prompting_prompting_topk_0_gen.txt")
    call_judge(model_handler, config, data_handler, ['prompting'], ['prompting'], ['0'])
