import os
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from string import punctuation
import datasets
import csv
import ast
import gc
class Config:
    def __init__(self, args=None):
        self.args = args
        if not self.args:
            self.parse_arguments()
        self.op_file = f'./data/{self.args.model_id.split("/")[-1]}/logits/{self.args.source}/{self.args.desired_file.split("-")[-2].split("/")[-1]}.jsonl'
        print('Making directory for ', self.op_file, ' at ', os.path.dirname(self.op_file))
        os.makedirs(os.path.dirname(self.op_file), exist_ok=True)
        self.initialize_environment()
        
    def parse_arguments(self):
        import argparse
        parser = argparse.ArgumentParser(description='Generate prompts for classification setting')
        parser.add_argument('-model_id', '--model_id', type=str, required=True, help='HF model id')
        parser.add_argument('-desired_file', '--desired_file', type=str, required=True, help='Path to desired data file')
        parser.add_argument('-undesired_file', '--undesired_file', type=str, required=True, help='Path to undesired data file')
        parser.add_argument('-d', '--device', type=str, required=True, help='Device for model execution')
        parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for logit calculation')
        parser.add_argument('-max_length', '--max_length', action='store_true', help='Use max length for tokenization')
        parser.add_argument('-source', '--source', type=str, default='hate', help='Generate source file. If not specified, generate base file')
        self.args = parser.parse_args()
        
    def initialize_environment(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MakeModel:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.args.model_id, token=os.environ['HF_TOKEN'], padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.logit_model = self.load_logit_model()
        self.logit_model.tokenizer = self.tokenizer
        assert self.logit_model is not None, "logit model not loaded"

    def load_logit_model(self):
        kwargs = {
            "device_map": self.config.args.device,
            "token": os.environ['HF_TOKEN'],
            "torch_dtype": torch.bfloat16,
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        }
        return AutoModelForCausalLM.from_pretrained(self.config.args.model_id, **kwargs)

class PromptProcessor:
    def __init__(self, config, model_handler):
        self.config = config
        self.model_handler = model_handler
        des_dataset = self.load_dataset(config.args.desired_file)
        undes_dataset = self.load_dataset(config.args.undesired_file)
        des_dataset = des_dataset.rename(columns={"response": "desired"})
        undes_dataset = undes_dataset.rename(columns={"response": "undesired"})

        if des_dataset['question'].dtype == list:
            des_dataset['question_str'] = des_dataset['question'].apply(lambda x: self.model_handler.tokenizer.apply_chat_template(x, add_generation_prompt=False, tokenize=False))
            undes_dataset['question_str'] = undes_dataset['question'].apply(lambda x: self.model_handler.tokenizer.apply_chat_template(x, add_generation_prompt=False, tokenize=False))
            self.dataset = pd.merge(des_dataset, undes_dataset, on='question_str', how='inner')
        else:
            self.dataset = pd.merge(des_dataset, undes_dataset, on='question', how='inner')
        self.dataset = datasets.Dataset.from_pandas(self.dataset)
        
    def load_from_jsonl(self, file_name):
        def load_json_line(line: str, i: int, file_name: str):
            try:
                return json.loads(json.dumps(ast.literal_eval(line)))
            except Exception as e:
                raise ValueError(f"Error in line {i+1}\n{line} of {file_name} {e}")
        with open(file_name, "r") as f:
            data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
        return data
    
    def load_dataset(self, ip_file):
        data = self.load_from_jsonl(ip_file)
        dataframe = pd.DataFrame(data)
        return dataframe
        
    def generate_prompts(self):
        prompts = {'prompt': [], 'type': [], 'id': []}
        ctr = 0
        # print(self.dataset)
        for data in tqdm(self.dataset):
            if 'question_str' in self.dataset.features:
                prompt = data['question_x']
            else:
                prompt = data['question']
            for response_type in ['desired', 'undesired']:
                if type(prompt) == str:
                    crafted_prompt = [
                        {"role": "user", "content": f"{prompt}"},
                        {"role": "assistant", "content": data[response_type]}
                    ]
                elif type(prompt) == list:
                    if len(prompt) == 1:
                        crafted_prompt = [
                            prompt[0],
                            # prompt[1],
                            {"role": "assistant", "content": data[response_type]}
                        ]
                    elif len(prompt) == 2:
                        crafted_prompt = [
                            prompt[0],
                            prompt[1],
                            {"role": "assistant", "content": data[response_type]}
                        ]
                else:
                    assert False, "Prompt type not supported"
                prompts['prompt'].append(crafted_prompt)
                prompts['type'].append(response_type)
                prompts['id'].append(ctr)
            ctr += 1
        return prompts

class logitCalculator:
    def __init__(self, config, model_handler):
        self.config = config
        self.model_handler = model_handler
        self.responses = []
        self.dpo = config.args.dpo
        self.batch_size = config.args.batch_size
        self.tokenizer = model_handler.tokenizer
        self.device = config.args.device
        self.important = []
        self.marker = None

    def get_response_positions_and_sizes(self, tokenized_text, device):
        response_start_positions = []
        response_token_sizes = []
        marker_tokens = None
        if 'solar' in self.config.args.model_id.lower():
            self.marker = '### Assistant'
            marker_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][2:].to(device)
        elif 'qwen' in self.config.args.model_id.lower():
            self.marker = "<|im_start|>assistant"
            marker_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0].to(device)
        elif 'zephyr-7b-alpha' in self.config.args.model_id.lower():
            self.marker = '<|assistant|>'
            marker_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][2:].to(device)
        elif 'zephyr-7b-gemma' in self.config.args.model_id.lower():
            self.marker = '<|im_start|>assistant'
            marker_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][2:].to(device)
        elif 'tulu-2' in self.config.args.model_id.lower():
            print('#### Tulu-2')
            self.marker = "<|assistant|>"
            marker_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][:-1].to(device)
        elif 'meta-llama' in self.config.args.model_id.lower():
            self.marker = '<|start_header_id|>assistant<|end_header_id|>'
            marker_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:].to(device)
        elif 'google' in self.config.args.model_id.lower():
            self.marker = "<start_of_turn>model"
            marker_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:].to(device)
        elif 'olmo' in self.config.args.model_id.lower():
            self.marker = '<|assistant|>\n'
            marker_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0].to(device)

        assert marker_tokens is not None, f"Marker tokens not found for model {self.config.args.model_id}"
        assert self.marker is not None, f"Marker not found for model {self.config.args.model_id}"
        # print(marker_tokens)
        # print(tokenized_text)
        for i, text in enumerate(tokenized_text):
            response_start_position = None
            marker_len = marker_tokens.size(0)
            for j in range(text.size(0) - marker_len + 1):
                if torch.equal(text[j : j + marker_len], marker_tokens):
                    response_start_position = j + marker_len + 1
                    break
            # print(marker_tokens, text)
            assert response_start_position is not None, f"Marker {self.marker} not found in input {self.tokenizer.decode(text)}.\nTOKENS: {text}\nMARKER: {marker_tokens}"
            response_start_positions.append(response_start_position)
            response_token_size = tokenized_text.size(0) - response_start_position
            response_token_sizes.append(response_token_size)
        # print(self.tokenizer.decode(tokenized_text[0][:response_start_positions[0]], skip_special_tokens=True))
        return response_start_positions, response_token_sizes

    def calculate_logits(self, prompts):
        assert len(prompts['prompt']) > 0, 'Prompts cannot be empty!'
        tokenizer = self.model_handler.tokenizer
        logit_model = self.model_handler.logit_model
        if tokenizer.chat_template is None:
            templated_prompts = [ f"User: {p[1]['content']} Assistant: {p[2]['content']}" for p in prompts['prompt'] ]
        else:
            templated_prompts = [
                logit_model.tokenizer.apply_chat_template(
                prompts['prompt'][idx], 
                add_generation_prompt=False,
                tokenize=False) for idx in range(len(prompts['prompt']))
            ]
        if self.config.args.max_length:
            model_inputs = tokenizer(templated_prompts, return_tensors='pt', padding=True)['input_ids'].to(self.model_handler.logit_model.device)


        BATCH_SIZE = self.batch_size
        length = model_inputs.shape[1]
        for idx in tqdm(range(0, len(prompts['prompt']), BATCH_SIZE)):
            batch_inputs = model_inputs[idx: min(idx + BATCH_SIZE, len(prompts['prompt']))]
            
            # assert self.model_handler.base_model is not None, "Base model is required for DPO"
            response_start_positions, _ = self.get_response_positions_and_sizes(batch_inputs, self.model_handler.logit_model.device)
            with torch.no_grad():
                chat_model_outputs = self.model_handler.logit_model(batch_inputs)
                chat_model_logits = chat_model_outputs.logits
                chat_model_log_probs = torch.log_softmax(chat_model_logits, dim=-1)
                
                chat_model_log_likelihoods = torch.stack([
                    chat_model_log_probs[i, response_start_position:-1, :].gather(-1, batch_inputs[i, 1+response_start_position:].unsqueeze(-1)).squeeze(-1).sum()
                    for i, response_start_position in enumerate(response_start_positions)
                ]).detach().cpu()
                
            # print('logitS LENGTH ', len(logits))
            for i, logit in enumerate(chat_model_log_likelihoods):
                self.responses.append({
                    "prompt": prompts['prompt'][idx + i],
                    "logit": logit.item(),
                    "chat_model_logit": chat_model_log_likelihoods[i].item() if self.dpo else None,
                    "response": prompts['prompt'][idx + i][-1]['content'],
                    "type": prompts['type'][idx + i],
                    "id": prompts['id'][idx + i],
                    "len": length
                })


    def save_results(self):
        op_file = self.config.op_file
        with open(op_file, 'a') as f:
            for response in self.responses[:50]:
                f.write(json.dumps(response) + "\n")
                
    def get_important(self):
        errors = 0
        assert len(self.responses) != 0 and len(self.responses) % 2 == 0, "Responses not calculated"
        for idx in range(0, len(self.responses), 2):
            res_0 = self.responses[idx]
            res_1 = self.responses[idx+1]
            # print(res_0, res_1)
            assert res_0['id'] == res_1['id'], f"ID mismatch: {res_0['id']} != {res_1['id']}"
            assert res_0['type'] != res_1['type'], f"Type mismatch: {res_0['type']} == {res_1['type']}"
            assert res_0['response'] != res_1['response'], f"Response match {res_0['response']} == {res_1['response']}"
            assert res_0['len'] == res_1['len'], f"Max len mismatch: {res_0['len']} != {res_1['len']}"
            
            if (float(res_0['logit']) > float(res_1['logit']) + 10):
                self.important.append({
                    "prompt": res_0['prompt'],
                    "logit": res_0['logit'],
                    "response": res_0['response'],
                    "type": res_0['type'],
                    "id": res_0['id'],
                    "len": res_0['len']
                })
            else:
                errors += 1
                # print(f"{res_1['prompt'][0]['content']}, ({res_1['response']}#{res_1['logit']}), ({res_0['response']}#{res_0['logit']})")
                self.important.append({
                    "prompt": res_1['prompt'],
                    "logit": res_1['logit'],
                    "response": res_1['response'],
                    "type": res_1['type'],
                    "id": res_1['id'],
                    "len": res_1['len']
                })
        print(f"Errors: {errors}")
        return self.important

    def save_important(self):
        self.important = self.get_important()
        op_file = f"{self.config.op_file.split('.jsonl')[0]}-important.jsonl"
        with open(op_file, "a") as f:
            for response in self.important[:50]:
                f.write(json.dumps(response) + "\n")
                
    def save_desired(self):
        if len(self.important) == 0:
            self.important = self.get_important()
        op_file = f"{self.config.op_file.split('.jsonl')[0]}-desired-all.jsonl"
        desired = [response for response in self.responses if response['type'] == 'desired']
        with open(op_file, 'a') as f:
            for response in desired[:50]:
                f.write(json.dumps(response) + "\n")

        important_ids = set([response['id'] for response in self.important if response['type'] == 'desired'])
        # print('IMPORTANT DES', len(important_ids))
        op_file = f"{self.config.op_file.split('.jsonl')[0]}-desired.jsonl"
        desired = [response for response in self.responses if response['type'] == 'desired' and response['id'] in important_ids]
        with open(op_file, 'a') as f:
            for response in desired:
                f.write(json.dumps(response) + "\n")

    
    def save_undesired_for_desired_responses(self):
        if len(self.important) == 0:
            self.important = self.get_important()
        op_file = f"{self.config.op_file.split('.jsonl')[0]}-undesired-all.jsonl"
        undesired = [response for response in self.responses if response['type'] == 'undesired']
        with open(op_file, 'a') as f:
            for response in undesired[:50]:
                f.write(json.dumps(response) + "\n")

        important_ids = set([response['id'] for response in self.important if response['type'] == 'desired'])
        # print('IMPORTANT UNDES', len(important_ids))
        op_file = f"{self.config.op_file.split('.jsonl')[0]}-undesired.jsonl"
        undesired = [response for response in self.responses if response['type'] == 'undesired' and response['id'] in important_ids]
        with open(op_file, 'a') as f:
            for response in undesired:
                f.write(json.dumps(response) + "\n")


import matplotlib.pyplot as plt

class Visualization:
    @staticmethod
    def plot_logits(responses, op_file):
        # Extract logits for desired and undesired types
        desired_logits = [float(response['logit']) for response in responses if response['type'] == 'desired']
        undesired_logits = [float(response['logit']) for response in responses if response['type'] == 'undesired']

        # Ensure both lists are of the same length by truncating the longer one
        # min_length = min(len(desired_logits), len(undesired_logits))
        # desired_logits = desired_logits[:min_length]
        # undesired_logits = undesired_logits[:min_length]

        # print(len(desired_logits), len(undesired_logits))  # Debugging line

        # Determine plot limits
        min_val = min(min(desired_logits), min(undesired_logits))
        max_val = max(max(desired_logits), max(undesired_logits))

        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(desired_logits, undesired_logits, alpha=0.7, edgecolor='k')
        
        # Add diagonal line (y = x)
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

        # Set plot limits
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        # Add labels, title, and legend
        plt.xlabel('Desired logits')
        plt.ylabel('Undesired logits')
        plt.title('Pairwise Scatterplot of logits with Diagonal')
        plt.legend()
        plt.grid(True)

        # Save and show the plot
        plt.savefig(f"{op_file.split('.jsonl')[0]}-scatterplot.png")
        plt.close()

    @staticmethod
    def plot_and_save_chat_model_and_ref_logits(responses, op_file):
        # Extract DPO and reference logits
        chat_model_logits_desired = [float(response['chat_model_logit']) for response in responses if response['chat_model_logit'] is not None and response["type"] == 'desired']
        chat_model_logits_undesired = [float(response['chat_model_logit']) for response in responses if response['chat_model_logit'] is not None and response["type"] == 'undesired']
        ref_logits_desired = [float(response['ref_logit']) for response in responses if response['ref_logit'] is not None and response["type"] == 'desired']
        ref_logits_undesired = [float(response['ref_logit']) for response in responses if response['ref_logit'] is not None and response["type"] == 'undesired']
        
        # Sanity check to ensure lengths match
        min_len = min(len(chat_model_logits_desired), len(chat_model_logits_undesired), len(ref_logits_desired), len(ref_logits_undesired))
        chat_model_logits_desired = chat_model_logits_desired[:min_len]
        chat_model_logits_undesired = chat_model_logits_undesired[:min_len]
        ref_logits_desired = ref_logits_desired[:min_len]
        ref_logits_undesired = ref_logits_undesired[:min_len]
        
        # Compute differences
        chat_model_logit_differences = [chat_model_logits_desired[i] - chat_model_logits_undesired[i] for i in range(min_len)]
        ref_logit_differences = [ref_logits_desired[i] - ref_logits_undesired[i] for i in range(min_len)]

        # Create the scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(range(min_len), chat_model_logits_desired, label='DPO logit (Desired)', alpha=0.7)
        plt.scatter(range(min_len), chat_model_logits_undesired, label='DPO logit (Undesired)', alpha=0.7)
        plt.scatter(range(min_len), chat_model_logit_differences, label='DPO logit Difference', alpha=0.7)
        plt.scatter(range(min_len), ref_logits_desired, label='Ref logit (Desired)', alpha=0.7)
        plt.scatter(range(min_len), ref_logits_undesired, label='Ref logit (Undesired)', alpha=0.7)
        plt.scatter(range(min_len), ref_logit_differences, label='Ref logit Difference', alpha=0.7)

        plt.xlabel('Sample Index')
        plt.ylabel('logit Value')
        plt.title('DPO and Reference logits')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{op_file.split('.jsonl')[0]}-dpo-ref-scatterplot.png")
        plt.close()

        csv_file = f"{op_file.split('.jsonl')[0]}-dpo-ref-logits.csv"
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow([
                'DPO logit (Desired)', 
                'DPO logit (Undesired)', 
                'DPO logit Difference',
                'Ref logit (Desired)', 
                'Ref logit (Undesired)', 
                'Ref logit Difference'
            ])
            # Write rows
            for i in range(min_len):
                writer.writerow([
                    chat_model_logits_desired[i],
                    chat_model_logits_undesired[i],
                    chat_model_logit_differences[i],
                    ref_logits_desired[i],
                    ref_logits_undesired[i],
                    ref_logit_differences[i]
                ])


    @staticmethod
    def plot_and_save_boxplots_of_logits(responses, op_file):
        # Extract DPO and reference logits
        chat_model_logits_desired = [float(response['chat_model_logit']) for response in responses if response['chat_model_logit'] is not None and response["type"] == 'desired']
        chat_model_logits_undesired = [float(response['chat_model_logit']) for response in responses if response['chat_model_logit'] is not None and response["type"] == 'undesired']
        ref_logits_desired = [float(response['ref_logit']) for response in responses if response['ref_logit'] is not None and response["type"] == 'desired']
        ref_logits_undesired = [float(response['ref_logit']) for response in responses if response['ref_logit'] is not None and response["type"] == 'undesired']
        
        # Sanity check to match lengths
        min_len = min(len(chat_model_logits_desired), len(chat_model_logits_undesired), len(ref_logits_desired), len(ref_logits_undesired))
        chat_model_logits_desired = chat_model_logits_desired[:min_len]
        chat_model_logits_undesired = chat_model_logits_undesired[:min_len]
        ref_logits_desired = ref_logits_desired[:min_len]
        ref_logits_undesired = ref_logits_undesired[:min_len]
        
        # Compute differences
        chat_model_logit_differences = [chat_model_logits_desired[i] - chat_model_logits_undesired[i] for i in range(min_len)]
        ref_logit_differences = [ref_logits_desired[i] - ref_logits_undesired[i] for i in range(min_len)]

        # Combine all metrics for boxplot
        data = [
            chat_model_logits_desired,
            chat_model_logits_undesired,
            chat_model_logit_differences,
            ref_logits_desired,
            ref_logits_undesired,
            ref_logit_differences
        ]
        labels = [
            'DPO logit (Desired)',
            'DPO logit (Undesired)',
            'DPO logit Difference',
            'Ref logit (Desired)',
            'Ref logit (Undesired)',
            'Ref logit Difference'
        ]

        # Create the boxplot
        plt.figure(figsize=(12, 8))
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.ylabel('logit Value')
        plt.title('Boxplots of DPO and Reference logits')
        plt.xticks(rotation=20)
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{op_file.split('.jsonl')[0]}-dpo-ref-boxplots.png")
        plt.close()

    @staticmethod
    def plot_logits_boxplot(responses, op_file):
        # Extract logits for desired and undesired types
        desired_logits = [float(response['logit']) for response in responses if response['type'] == 'desired']
        undesired_logits = [float(response['logit']) for response in responses if response['type'] == 'undesired']

        # Create a box plot
        plt.figure(figsize=(8, 6))
        plt.boxplot([desired_logits, undesired_logits], labels=['Desired', 'Undesired'])

        # Add labels and title
        plt.ylabel('logits')
        plt.title('Box Plot of Desired vs. Undesired logits')

        # Save and show the plot
        plt.savefig(f"{op_file.split('.jsonl')[0]}-boxplot.png")
        plt.close()

def main(config=None):
    # Standalone execution
    if config is None:
        config = Config()
    model_handler = MakeModel(config)
    prompt_processor = PromptProcessor(config, model_handler)
    logit_calculator = logitCalculator(config, model_handler)
    
    prompts = prompt_processor.generate_prompts()
    assert len(prompts['prompt']) > 0, 'Prompts cannot be empty!'
    logit_calculator.calculate_logits(prompts)
    # os.makedirs("/".join(config.op_file.split('/')[:-1]), exist_ok=True)
    logit_calculator.save_important()
    logit_calculator.save_results()
    logit_calculator.save_desired()
    logit_calculator.save_undesired_for_desired_responses()
    # Visualization.plot_logits(logit_calculator.responses, config.op_file)
    # Visualization.plot_logits_boxplot(logit_calculator.responses, config.op_file)
    # Visualization.plot_and_save_chat_model_and_ref_logits(logit_calculator.responses, config.op_file)
    # Visualization.plot_and_save_boxplots_of_logits(logit_calculator.responses, config.op_file)
    del model_handler
    gc.collect()
    torch.cuda.empty_cache()

# Allows the script to be both imported as a module and run as standalone
if __name__ == "__main__":
    main()