import json
import glob
import os
import math
from transformers import AutoTokenizer

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

files = glob.glob(os.path.join(DIR, "*desired*.jsonl"))

for path in sorted(files):
    lengths = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            for msg in entry["prompt"]:
                if msg["role"] == "assistant":
                    lengths.append(len(tokenizer.encode(msg["content"], add_special_tokens=False)))

    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((x - mean) ** 2 for x in lengths) / (n - 1)
    se = math.sqrt(variance / n)
    ci = 1.96 * se

    print(f"{os.path.basename(path)}")
    print(f"  n={n}, mean={mean:.1f}, 95% CI=[{mean-ci:.1f}, {mean+ci:.1f}]")
