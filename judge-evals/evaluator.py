"""
vLLM judge model — shared inference utilities for run_judge.py.
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch

# Suppress vLLM's verbose startup and inference logs before importing
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
os.environ.setdefault("GLOO_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.WARNING)

from vllm import LLM, SamplingParams

from config import (JUDGE_MODEL_NAME, JUDGE_MAX_TOKENS, PASSTHROUGH_COLS, RATING_ANSWER_TOKENS,
                    RATING_PREFIX, extract_rating)


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
        max_model_len=8192,
        seed=SEED,
    )


FORCED_RATING_SUFFIX = "\n" + RATING_PREFIX


def get_sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens,
        stop=["]]"],
        include_stop_str_in_output=True,
        seed=SEED,
    )


def reasoning_budget(llm) -> int:
    """Tokens the judge may reason for, leaving room within JUDGE_MAX_TOKENS to force a rating."""
    suffix_tokens = len(llm.get_tokenizer().encode(FORCED_RATING_SUFFIX, add_special_tokens=False))
    return JUDGE_MAX_TOKENS - suffix_tokens - RATING_ANSWER_TOKENS


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_in_batches(llm, prompts, batch_size):
    """Yield the judge's decoded outputs batch by batch.

    The judge reasons freely for up to reasoning_budget() tokens, stopping at the
    first `]]`. Any output that still has no parseable rating (reasoning ran out,
    or it ended without the format) gets FORCED_RATING_SUFFIX appended and is
    continued for RATING_ANSWER_TOKENS more, so no output exceeds JUDGE_MAX_TOKENS.
    The returned text includes the forced suffix and continuation.
    """
    reason_sp = get_sampling_params(reasoning_budget(llm))
    answer_sp = get_sampling_params(RATING_ANSWER_TOKENS)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [r.outputs[0].text for r in llm.generate(batch, reason_sp)]
        unrated = [j for j, t in enumerate(texts) if extract_rating(t) == -1]
        if unrated:
            forced = [batch[j] + texts[j] + FORCED_RATING_SUFFIX for j in unrated]
            for j, r in zip(unrated, llm.generate(forced, answer_sp)):
                texts[j] = texts[j] + FORCED_RATING_SUFFIX + r.outputs[0].text
        yield texts


