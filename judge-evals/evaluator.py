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


def make_llm(model_name: str = JUDGE_MODEL_NAME, max_num_seqs: int = 64):
    import os
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
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
        max_num_seqs=max_num_seqs,
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
    from tqdm import tqdm
    completed = 0
    with tqdm(total=len(prompts), desc="vLLM judge", unit="prompt") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            results = llm.generate(batch, sampling_params, use_tqdm=False)
            completed += len(batch)
            pbar.update(len(batch))
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


_OPENAI_TIMEOUT_SECS = 30
_OPENAI_MAX_RETRIES = 10


def generate_in_batches_openai(client, model: str, prompts: list, batch_size: int, num_workers: int = 20):
    """Yield decoded output strings batch by batch using OpenAI API (multi-threaded)."""
    import openai
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    def call_api(args):
        idx, prompt = args
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
        for attempt in range(1, _OPENAI_MAX_RETRIES + 1):
            try:
                response = client.responses.create(
                    model=model,
                    input=input_messages,
                    reasoning={"effort": "minimal"},
                    max_output_tokens=1024,
                    timeout=_OPENAI_TIMEOUT_SECS,
                )
                return response.output_text
            except openai.APITimeoutError:
                print(f"  [timeout] prompt {idx} timed out (attempt {attempt}/{_OPENAI_MAX_RETRIES})")
                if attempt == _OPENAI_MAX_RETRIES:
                    raise RuntimeError(
                        f"Prompt {idx} failed after {_OPENAI_MAX_RETRIES} retries — aborting generation."
                    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(call_api, enumerate(prompts)),
            total=len(prompts),
            desc="OpenAI judge",
            unit="prompt",
        ))

    for i in range(0, len(results), batch_size):
        yield results[i : i + batch_size]


