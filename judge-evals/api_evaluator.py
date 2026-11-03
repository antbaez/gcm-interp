"""
OpenAI API judge: the API counterpart of evaluator.py for run_judge.py.

Finished calls are cached under <judge dir>/api_cache/, so a preempted or rerun job doesn't pay for them twice.
"""

import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import extract_rating

API_MAX_OUTPUT_TOKENS = 4096
API_MAX_RETRIES = 8
API_TIMEOUT_SECONDS = 300


class ChatJSONTokenizer:
    """Stands in for the tokenizer in build_prompts.py: a prompt stays as its JSON chat messages."""

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        return json.dumps(chat, ensure_ascii=False)


def load_cache(cache_dir: Path) -> dict:
    cache = {}
    for path in sorted(cache_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[entry["key"]] = entry["output"]
    return cache


def make_api_generate_fn(model: str, cache_dir: Path, concurrency: int = 16):
    """A generate(prompts, batch_size) that yields each batch's outputs, as evaluator.generate_in_batches does."""
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(max_retries=API_MAX_RETRIES, timeout=API_TIMEOUT_SECONDS)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_dir)
    cache_file = cache_dir / f"{os.environ.get('SLURM_JOB_ID', os.getpid())}.jsonl"
    lock = threading.Lock()
    print(f"API judge {model}: {len(cache)} cached outputs in {cache_dir}")

    def complete(prompt: str) -> str:
        key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
        if key in cache:
            return cache[key]
        response = client.chat.completions.create(
            model=model,
            messages=json.loads(prompt),
            max_completion_tokens=API_MAX_OUTPUT_TOKENS,
        )
        output = response.choices[0].message.content or ""
        if extract_rating(output) != -1:
            with lock:
                cache[key] = output
                with open(cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "output": output}, ensure_ascii=False) + "\n")
        return output

    def generate(prompts, batch_size):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for i in range(0, len(prompts), batch_size):
                yield list(pool.map(complete, prompts[i : i + batch_size]))

    return generate
