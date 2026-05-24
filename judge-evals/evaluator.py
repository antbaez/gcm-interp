"""
vLLM judge model — shared inference utilities for run_judge.py.
"""

import json
from pathlib import Path

import pandas as pd
import torch

from config import JUDGE_MODEL_NAME, PASSTHROUGH_COLS, extract_rating
from compute_accuracies import extract_first_int


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

SEED = 42


def make_llm(model_name: str = JUDGE_MODEL_NAME):
    from vllm import LLM
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")
    print(f"Detected {num_gpus} GPU(s). Loading judge model: {model_name}")

    return LLM(
        model=model_name,
        quantization="bitsandbytes",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="auto",
        max_num_seqs=64,
        max_model_len=4096,
        seed=SEED,
    )


def get_sampling_params():
    from vllm import SamplingParams
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=3,
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_in_batches(llm, prompts, sampling_params, batch_size):
    """Yield decoded output strings batch by batch."""
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        results = llm.generate(batch, sampling_params)
        yield [r.outputs[0].text for r in results]


# ---------------------------------------------------------------------------
# OpenAI API judge
# ---------------------------------------------------------------------------

import re


_LLAMA_DATE_HEADER_RE = re.compile(
    r'^Cutting Knowledge Date:.*?(?=\n\n|\Z)', re.DOTALL
)


def _strip_llama_date_header(text: str) -> str:
    """Remove the Llama tokenizer's auto-injected date header from a system message."""
    return _LLAMA_DATE_HEADER_RE.sub("", text).strip()


def _parse_llama_template(formatted_prompt: str) -> dict:
    """Parse a Llama chat-template string into system, user, and assistant-prefix parts."""
    system_match = re.search(
        r'<\|start_header_id\|>system<\|end_header_id\|>\s*(.*?)<\|eot_id\|>',
        formatted_prompt, re.DOTALL,
    )
    user_match = re.search(
        r'<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)<\|eot_id\|>',
        formatted_prompt, re.DOTALL,
    )
    asst_match = re.search(
        r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)$',
        formatted_prompt, re.DOTALL,
    )
    system_raw = system_match.group(1).strip() if system_match else None
    return {
        "system": _strip_llama_date_header(system_raw) if system_raw else None,
        "user": user_match.group(1).strip() if user_match else formatted_prompt,
        "asst_prefix": asst_match.group(1).strip() if asst_match else None,
    }


def generate_in_batches_openai(client, model: str, prompts: list, batch_size: int):
    """Yield decoded output strings batch by batch using OpenAI API (single-threaded)."""
    from tqdm import tqdm
    pbar = tqdm(total=len(prompts), desc="OpenAI judge", unit="prompt")
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        results = []
        for prompt in batch:
            parsed = _parse_llama_template(prompt)
            input_messages = []
            if parsed["system"]:
                input_messages.append({
                    "role": "developer",
                    "content": [{"type": "input_text", "text": parsed["system"]}],
                })
            input_messages.append({
                "role": "user",
                "content": [{"type": "input_text", "text": parsed["user"]}],
            })
            # print("\n" + "="*60)
            # print("[INPUT MESSAGES]")
            # for msg in input_messages:
            #     print(f"  role: {msg['role']}")
            #     for block in msg['content']:
            #         print(f"  {block['text']}")
            response = client.responses.create(
                model=model,
                input=input_messages,
                reasoning={"effort": "minimal"},
                max_output_tokens=1024,
            )
            # print(f"[RESPONSE] {response.output_text!r}")
            # print("="*60)
            results.append(response.output_text)
            pbar.update(1)
        yield results
    pbar.close()


