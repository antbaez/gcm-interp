"""
vLLM judge model — shared inference utilities for run_judge.py.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams

from config import JUDGE_MODEL_NAME, PASSTHROUGH_COLS, extract_rating
from compute_accuracies import extract_first_int


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

SEED = 42


def make_llm(model_name: str = JUDGE_MODEL_NAME) -> LLM:
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


def get_sampling_params() -> SamplingParams:
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


