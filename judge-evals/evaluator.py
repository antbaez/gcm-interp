"""
vLLM judge model — shared inference utilities for run_judge.py.
"""

import contextlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

from config import JUDGE_MODEL_NAME, PASSTHROUGH_COLS, extract_rating
from compute_accuracies import extract_first_int


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

SEED = 42


@contextlib.contextmanager
def _suppress_fd_output():
    """Silence stdout/stderr at the OS file-descriptor level.

    vLLM's EngineCore runs in a subprocess and writes the "Loading safetensors
    checkpoint shards", "Capturing CUDA graphs", and "Done." lines straight to
    the raw fds, so VLLM_LOGGING_LEVEL can't suppress them. Redirecting fds 1/2
    (which the subprocess inherits) does. Exceptions still propagate, so a real
    load failure is not hidden.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def compute_max_prompt_len(long_items, model_name: str = JUDGE_MODEL_NAME) -> int:
    """Scan all prompt CSVs and return the max token length across all prompts."""
    from transformers import AutoTokenizer
    print("Scanning prompt CSVs to compute max token length...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_cols = {
        "relevance_fluency_prompts.csv": ["fluency_prompt", "relevance_prompt"],
        "judge_prompts.csv": ["judge_prompt"],
    }
    max_len = 0
    for wd, _meta, _gp in long_items:
        for csv_name, cols in prompt_cols.items():
            csv_path = wd / csv_name
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, keep_default_na=False)
            for col in cols:
                if col not in df.columns:
                    continue
                lengths = [len(tokenizer.encode(p)) for p in df[col].tolist()]
                if lengths:
                    max_len = max(max_len, max(lengths))
    print(f"Max prompt token length: {max_len}")
    return max_len


def make_llm(model_name: str = JUDGE_MODEL_NAME, max_num_seqs: int = 64, device: int | None = None, max_model_len: int | None = None):
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    from vllm import LLM
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")
    gpu_info = f"GPU {device}" if device is not None else f"{num_gpus} GPU(s)"
    print(f"Using {gpu_info}. Loading judge model: {model_name}")

    try:
        with _suppress_fd_output():
            llm = LLM(
                model=model_name,
                quantization="bitsandbytes",
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                dtype="auto",
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                seed=SEED,
            )
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    print(f"Judge model loaded.")
    return llm


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



