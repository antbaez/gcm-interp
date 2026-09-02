import re
import shutil
import tempfile
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForSequenceClassification
import os
from nnsight import NNsight, LanguageModel
class ModelHandler:
    def __init__(self, config):
        self.config = config
        model_id = config.args.model_id
        self.device = config.args.device
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.is_qwen3 = 'qwen3' in model_id.lower()
        self.is_gemma4 = 'gemma-4' in model_id.lower()
        self.thinking = getattr(config.args, 'thinking', False)
        self.template_kwargs = (
            {'enable_thinking': self.thinking} if self.is_qwen3 or self.is_gemma4 else {}
        )
        # gemma4 already routes through its own persistent cache_dir below, so
        # --no-model-cache only applies to models that would otherwise land in
        # the default $HF_HOME cache.
        self.no_model_cache = getattr(config.args, 'no_model_cache', False) and not self.is_gemma4
        self._temp_cache_dir = tempfile.mkdtemp(prefix='hf-nocache-') if self.no_model_cache else None
        self.tokenizer = self.load_tokenizer(model_id)
        self.model = self.load_model(model_id, self.device)
        self.model.tokenizer = self.tokenizer
        if self._temp_cache_dir:
            # Weights are already copied onto self.device by dispatch=True, so the
            # on-disk snapshot is safe to remove — it just frees quota-limited disk.
            shutil.rmtree(self._temp_cache_dir, ignore_errors=True)
        model_config = self.model.config.to_dict()
        tc = model_config.get('text_config', model_config)
        hidden_size = tc['hidden_size']
        self.num_heads = tc['num_attention_heads']
        self.num_layers = tc['num_hidden_layers']
        self.dim = hidden_size // self.num_heads

        if 'solar' in model_id.lower():
            self.marker = '### Assistant'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][2:]
        elif 'qwen3' in model_id.lower():
            self.marker = "<|im_start|>assistant\n" if self.thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
        elif 'qwen' in model_id.lower():
            self.marker = "<|im_start|>assistant\n"
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
        elif 'llama-2-7b-chat-hf' in model_id.lower():
            self.marker = "[/INST] "
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:-1]
        elif 'meta-llama' in model_id.lower():
            self.marker = '<|start_header_id|>assistant<|end_header_id|>\n\n'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:]
        elif 'olmo' in model_id.lower():
            self.marker = '<|assistant|>\n'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
        elif self.is_gemma4:
            self.marker = '<|turn>model\n' if self.thinking else '<|turn>model\n<|channel>thought\n<channel|>'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0]
        elif 'google' in model_id.lower():
            self.marker = '<start_of_turn>model\n'
            self.alignment_tokens = self.tokenizer(self.marker, return_tensors="pt")["input_ids"][0][1:]

    def load_tokenizer(self, model_id):
        if 'qwen'  in model_id.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], pad_token='<|pad|>', eos_token='<|endoftext|>', cache_dir=self._temp_cache_dir)
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
            tokenizer.padding_side = 'left'
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], cache_dir=self._temp_cache_dir)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        print(f"Loading model {model_id} (padding: {tokenizer.padding_side})...")
        return tokenizer

    def load_model(self, model_id, device, model_type="causal"):
        if self.is_gemma4:
            from nnsight import VisionLanguageModel
            gemma4_cache_dir = os.path.expanduser("~/orcd/pool/huggingface")
            os.makedirs(gemma4_cache_dir, exist_ok=True)
            return VisionLanguageModel(model_id, device_map=device, tokenizer=self.tokenizer, dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], quantization_config=self.nf4_config, dispatch=True, cache_dir=gemma4_cache_dir)
        return LanguageModel(model_id, device_map=device, tokenizer=self.tokenizer, dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], quantization_config=self.nf4_config, dispatch=True, cache_dir=self._temp_cache_dir)
    