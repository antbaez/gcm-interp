#!/usr/bin/env python3
"""
Load each model, run inference on 5 harmful-test examples, and save decoded
outputs with special tokens so the chat template and markers can be verified.
"""
import gc
import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_DIR = Path(__file__).parent.parent
SCRIPT_DIR = Path(__file__).parent

MODELS = [
    ("olmo",   "allenai/OLMo-2-1124-13B-DPO"),
    ("qwen",   "Qwen/Qwen1.5-14B-Chat"),
    ("qwen3",  "Qwen/Qwen3-14B"),
    ("gemma",  "google/gemma-3-12b-it"),
    ("llama",  "meta-llama/Llama-3.1-8B-Instruct")
]

# Markers as set in model_handler.py
MARKERS = {
    "olmo":   "<|assistant|>\n",
    "qwen":   "<|im_start|>assistant\n",
    "qwen3":  "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "gemma":  "<start_of_turn>model\n",
    "llama":  "<|start_header_id|>assistant<|end_header_id|>\n\n",
}

TEST_JSONL     = str(BASE_DIR / "data/queries/harmful-long-queries/harmless-test.jsonl")
N_EXAMPLES     = 3
MAX_NEW_TOKENS = 32
DEVICE       = "cuda:0"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_examples(path: str, n: int) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line)["prompt"])
            if len(examples) >= n:
                break
    return examples


def get_tokenizer(tag: str, model_id: str):
    if "qwen" in model_id.lower():
        tok = AutoTokenizer.from_pretrained(
            model_id, token=os.environ["HF_TOKEN"],
            pad_token="<|endoftext|>", eos_token="<|endoftext|>",
        )
        tok.add_special_tokens({"pad_token": "<|endoftext|>"})
    else:
        tok = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def apply_template(tok, tag: str, prompt: list) -> str:
    kwargs = dict(add_generation_prompt=True, tokenize=False)
    if tag == "qwen3":
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(prompt, **kwargs)


def run_model(tag: str, model_id: str, examples: list, out_f):
    out_f.write(f"MODEL: {model_id}  [{tag}]\n")
    out_f.write(f"{'='*80}\n\n")

    tok = get_tokenizer(tag, model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
        quantization_config=nf4_config,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    marker = MARKERS[tag]

    for i, prompt in enumerate(examples):
        templated = apply_template(tok, tag, prompt)
        inputs = tok(templated, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        prompt_decoded   = tok.decode(inputs["input_ids"][0], skip_special_tokens=False)
        response_decoded = tok.decode(out[0][prompt_len:],    skip_special_tokens=False)
        full_decoded     = tok.decode(out[0],                 skip_special_tokens=False)

        marker_char_pos  = prompt_decoded.find(marker)
        marker_tok_pos   = prompt_decoded[:marker_char_pos + len(marker)].count(" ") if marker_char_pos >= 0 else -1

        out_f.write(f"--- Example {i+1} ---\n")
        out_f.write(f"PROMPT (with special tokens):\n{repr(prompt_decoded)}\n\n")
        out_f.write(f"MARKER: {repr(marker)}\n")
        out_f.write(f"  Found at char {marker_char_pos} in decoded prompt "
                    f"({'NOT FOUND' if marker_char_pos < 0 else 'OK'})\n\n")
        out_f.write(f"RESPONSE (new tokens only, with special tokens):\n{repr(response_decoded)}\n\n")
        out_f.write(f"FULL OUTPUT:\n{repr(full_decoded)}\n")
        out_f.write("-" * 60 + "\n\n")

    del model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    examples = load_examples(TEST_JSONL, N_EXAMPLES)
    print(f"Loaded {len(examples)} examples from {TEST_JSONL}")

    for tag, model_id in MODELS:
        output_file = str(SCRIPT_DIR / f"chat_template_{tag}.txt")
        print(f"\n[{tag}] Loading {model_id} ...")
        with open(output_file, "w") as f:
            try:
                run_model(tag, model_id, examples, f)
                print(f"[{tag}] Done. Saved to {output_file}")
            except Exception as e:
                msg = f"\nERROR running {model_id}: {e}\n"
                f.write(msg)
                print(msg)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
