#!/usr/bin/env python3
"""
Load Qwen3 with the current model_handler tokenizer setup and verify the
padding scheme by writing every token in a small left-padded batch to a txt file.
"""
import os
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID   = "Qwen/Qwen3-14B"
SCRIPT_DIR = Path(__file__).parent
OUTPUT     = SCRIPT_DIR / "qwen3_tokenizer_check.txt"

PROMPTS = [
    [{"role": "user", "content": "Write a short poem about the ocean."}],
    [{"role": "user", "content": "What is 2 + 2?"}],
    [{"role": "user", "content": "Explain gravity in one sentence."}],
]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=os.environ["HF_TOKEN"],
    pad_token="<|pad|>",
    eos_token="<|endoftext|>",
)
tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
tokenizer.padding_side = "left"

templated = [
    tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    for p in PROMPTS
]

batch = tokenizer(templated, padding=True, truncation=False, return_tensors="pt")
input_ids     = batch["input_ids"]
attention_mask = batch["attention_mask"]

with open(OUTPUT, "w") as f:
    f.write(f"MODEL: {MODEL_ID}\n")
    f.write(f"pad_token: {repr(tokenizer.pad_token)}  id={tokenizer.pad_token_id}\n")
    f.write(f"eos_token: {repr(tokenizer.eos_token)}  id={tokenizer.eos_token_id}\n")
    f.write(f"padding_side: {tokenizer.padding_side}\n")
    f.write(f"batch shape: {list(input_ids.shape)}\n\n")

    for i, prompt in enumerate(PROMPTS):
        f.write(f"{'='*70}\n")
        f.write(f"Example {i+1}: {prompt[0]['content']}\n")
        f.write(f"Templated (repr): {repr(templated[i])}\n\n")
        f.write(f"{'idx':>5}  {'token_id':>10}  {'mask':>4}  token\n")
        f.write(f"{'-'*50}\n")
        for j, (tok_id, mask) in enumerate(zip(input_ids[i].tolist(), attention_mask[i].tolist())):
            tok_str = repr(tokenizer.convert_ids_to_tokens(tok_id))
            f.write(f"{j:>5}  {tok_id:>10}  {mask:>4}  {tok_str}\n")
        f.write("\n")

print(f"Saved to {OUTPUT}")
