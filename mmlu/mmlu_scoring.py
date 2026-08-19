import json
import os
import random
import time

import torch
from tqdm import tqdm

from eval.generation import _get_layers, _prepare_steering_vector

LETTERS = ["A", "B", "C", "D"]


def load_mmlu(path):
    rows_by_subject = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_subject.setdefault(row["subject"], []).append(row)
    return rows_by_subject


def sample_mmlu(rows_by_subject, fraction):
    # Relies on the global random seed being set once by the caller (run_mmlu.py)
    # rather than reseeding here, so the RNG state advances across subjects
    # instead of every subject drawing from an identically-reset stream.
    sampled = []
    for subject in sorted(rows_by_subject.keys()):
        rows = rows_by_subject[subject]
        k = min(len(rows), max(1, round(len(rows) * fraction)))
        sampled.extend(random.sample(rows, k))
    return sampled


def build_prompt_text(question, choices):
    lines = [question, ""]
    lines += [f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices)]
    lines += ["", "Answer with only the letter of the correct answer (A, B, C, or D)."]
    return "\n".join(lines)


def build_batches(tokenizer, template_kwargs, device, rows, batch_size):
    """Tokenizes every MMLU row once; reused unchanged across every steering
    config scored in a job, since the question set and tokenizer don't change."""
    batches = []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt_text(r["question"], r["choices"])}],
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            for r in batch_rows
        ]
        tokens = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        prompt_toks = {
            "input_ids": tokens["input_ids"].to(device),
            "attention_mask": tokens["attention_mask"].to(device),
        }
        batches.append((batch_rows, prompt_toks))
    return batches


def resolve_answer_token_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Tokenizer produced no tokens for letter {letter!r}")
        ids.append(encoded[-1])
    if len(set(ids)) != len(LETTERS):
        raise ValueError(f"Answer letter token ids are not distinct: {dict(zip(LETTERS, ids))}")
    return ids


def _align_positional(sv, target_seq_len):
    # Positional vectors are cached at the original task's prompt length, not
    # MMLU's, and both are left-padded (position -1 is always the real,
    # generation-adjacent token) — so alignment must anchor on the tail, not
    # the front: slice the front off if longer, pad the front if shorter.
    p = sv.shape[0]
    if p >= target_seq_len:
        return sv[-target_seq_len:]
    pad = torch.zeros(target_seq_len - p, sv.shape[1], dtype=sv.dtype, device=sv.device)
    return torch.cat([pad, sv], dim=0)


def score_batch_with_patches(model, prompt_toks, answer_token_ids, patch_activations=None,
                              layer_ids=None, N=None, steering_type=None, resid=False, normalize=True):
    is_gemma_tuple = (
        resid and 'gemma' in model.config._name_or_path.lower()
        and getattr(model.config, 'model_type', '') != 'gemma4_unified'
    )
    seq_len = prompt_toks['input_ids'].shape[1]

    # Same call path as generate_with_patches() (eval/generation.py) — gets
    # transformers' implicit torch.no_grad() and applies interventions to
    # prefill only, matching the rest of the pipeline. max_new_tokens=1 since
    # we only need the forced-choice logits at the first generated position.
    gen_kwargs = dict(pad_token_id=model.tokenizer.eos_token_id, use_cache=True,
                       do_sample=False, top_p=None, top_k=None, temperature=None,
                       max_new_tokens=1)
    with model.generate(prompt_toks, **gen_kwargs) as tracer:
        if patch_activations is not None and layer_ids:
            patch_activations_dev = patch_activations.to(model.device)
            for layer_idx in layer_ids:
                layer = _get_layers(model)[layer_idx]
                sv = _prepare_steering_vector(patch_activations_dev, layer_idx, steering_type, normalize, None)
                if steering_type in ('positional', 'weighted-pos'):
                    sv = _align_positional(sv, seq_len)
                if resid:
                    if is_gemma_tuple:
                        layer.output = (layer.output[0] + N * sv,)
                    else:
                        layer.output += N * sv
                else:
                    layer.self_attn.o_proj.output += N * sv
        logits = model.output.logits[:, -1, :].save()

    letter_logits = logits[:, answer_token_ids]
    return letter_logits.argmax(dim=-1).cpu().tolist()


def evaluate_mmlu(model, batches, answer_token_ids, patch_activations=None, layer_ids=None,
                   N=None, steering_type=None, resid=False, normalize=True, print_examples=False):
    n_correct = 0
    n_total = 0
    per_subject = {}
    examples = []
    n_batches = len(batches)
    print(f"Scoring {n_batches} batches...")
    total_time = 0.0
    allocated_sum = 0.0
    reserved_sum = 0.0
    torch.cuda.reset_peak_memory_stats(model.device)
    pbar = tqdm(batches, total=n_batches, desc="MMLU batches")
    for i, (batch_rows, prompt_toks) in enumerate(pbar, start=1):
        t0 = time.time()
        preds = score_batch_with_patches(
            model, prompt_toks, answer_token_ids,
            patch_activations=patch_activations, layer_ids=layer_ids, N=N,
            steering_type=steering_type, resid=resid, normalize=normalize,
        )
        elapsed = time.time() - t0
        total_time += elapsed
        pbar.set_postfix(avg_s=f"{total_time / i:.2f}")

        allocated_sum += torch.cuda.memory_allocated(model.device) / 1024**3
        reserved_sum += torch.cuda.memory_reserved(model.device) / 1024**3

        for row, pred in zip(batch_rows, preds):
            correct = int(pred == row["answer"])
            n_correct += correct
            n_total += 1
            subj = per_subject.setdefault(row["subject"], {"n": 0, "correct": 0})
            subj["n"] += 1
            subj["correct"] += correct
            if print_examples:
                examples.append({
                    "subject": row["subject"], "question": row["question"], "choices": row["choices"],
                    "correct_letter": LETTERS[row["answer"]], "predicted_letter": LETTERS[pred],
                    "correct": bool(correct),
                })
    avg_time = total_time / n_batches if n_batches else 0.0
    print(f"Average time per batch: {avg_time:.2f}s")
    if n_batches:
        max_allocated = torch.cuda.max_memory_allocated(model.device) / 1024**3
        max_reserved = torch.cuda.max_memory_reserved(model.device) / 1024**3
        print(f"VRAM allocated: avg={allocated_sum / n_batches:.2f} GiB max={max_allocated:.2f} GiB")
        print(f"VRAM reserved:  avg={reserved_sum / n_batches:.2f} GiB max={max_reserved:.2f} GiB")

    n_show = min(5, len(examples)) if print_examples else 0
    if n_show:
        print(f"\n--- {n_show} random example predictions ---")
        for ex in random.sample(examples, n_show):
            print(f"[{ex['subject']}] {ex['question']}")
            for letter, choice in zip(LETTERS, ex['choices']):
                print(f"  {letter}. {choice}")
            status = "correct" if ex['correct'] else "WRONG"
            print(f"  -> predicted: {ex['predicted_letter']}  correct: {ex['correct_letter']}  [{status}]")
            print()

    return {
        "n_samples": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "per_subject": per_subject,
    }


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
