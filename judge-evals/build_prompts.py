"""
Build judge-ready prompt columns and write them to CSV.

Supports three prompt types via subcommands:
  - judge:     Behavioral judge prompts (harmful, hate, verse, sycophancy)
  - fluency:   Fluency evaluation prompts
  - relevance: Relevance evaluation prompts
  - all:       Build all three at once

Called directly by run_judge.py (phase 1). Can also be run standalone:
    python build_prompts.py judge     --input eval_output.csv --output judge_prompts.csv
    python build_prompts.py fluency   --input eval_output.csv --output relevance_fluency_prompts.csv
    python build_prompts.py relevance --input eval_output.csv --output relevance_fluency_prompts.csv
    python build_prompts.py all       --input eval_output.csv
"""

import argparse

import pandas as pd
from transformers import AutoTokenizer

from config import (
    TOKENIZER_MODEL_NAME,
    PROMPT_TEMPLATES,
    SOURCE_TO_TEMPLATE,
    PAIRED_TEMPLATES,
    SINGLE_TEMPLATES,
)


def load_tokenizer(model_name: str = TOKENIZER_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# Behavioral judge prompts
# ---------------------------------------------------------------------------

def _build_paired_judge_prompt(tokenizer, template: str, new_response: str,
                                old_response: str, query: str) -> str:
    """Build a judge prompt that compares two responses."""
    assert isinstance(new_response, str), "new_response must be a string"
    assert isinstance(old_response, str), "old_response must be a string"
    assert isinstance(query, str), "query must be a string"

    user_msg = f"{query}\nResponse (1): {new_response}\nResponse (2): {old_response}"
    chat = [
        {"role": "user", "content": template.format(conversation=user_msg)},
        {"role": "assistant", "content": "("},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    eot = "<|eot_id|>"
    if prompt.endswith(eot):
        prompt = prompt[: -len(eot)]
    return prompt


def _build_single_judge_prompt(tokenizer, template: str,
                                new_response: str, query: str) -> str:
    """Build a judge prompt that evaluates a single response."""
    assert isinstance(new_response, str), "new_response must be a string"
    assert isinstance(query, str), "query must be a string"

    user_msg = f"{query}\nResponse: {new_response}"
    chat = [{"role": "user", "content": template.format(conversation=user_msg)}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def build_judge_prompts(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """Add a 'judge_prompt' column based on each row's SOURCE."""
    prompts = []
    for idx, row in df.iterrows():
        source = row["SOURCE"]
        template_key = SOURCE_TO_TEMPLATE.get(source)

        if template_key is None:
            raise ValueError(f"Row {idx}: unknown SOURCE '{source}'")

        template = PROMPT_TEMPLATES[template_key]

        try:
            if template_key in PAIRED_TEMPLATES:
                prompt = _build_paired_judge_prompt(
                    tokenizer, template,
                    row["post-intervention-response"],
                    row["original-response"],
                    row["query"],
                )
            elif template_key in SINGLE_TEMPLATES:
                prompt = _build_single_judge_prompt(
                    tokenizer, template,
                    row["post-intervention-response"],
                    row["query"],
                )
            else:
                raise ValueError(f"Template '{template_key}' is neither paired nor single")
        except AssertionError as e:
            raise ValueError(f"Row {idx}: {e}") from e

        prompts.append(prompt)

    df["judge_prompt"] = prompts
    return df


# ---------------------------------------------------------------------------
# Fluency / relevance prompts
# ---------------------------------------------------------------------------

def _build_fluency_prompt(tokenizer, sentence: str) -> str:
    user_msg = (
        f"[Sentence Fragment Start]\n{sentence}\n[Sentence Fragment End]"
    )
    chat = [
        {"role": "system", "content": PROMPT_TEMPLATES["fluency"]},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "Rating: [["},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )
    # Strip the trailing end-of-turn token so the model continues from "[["
    eot = "<|eot_id|>"
    if prompt.endswith(eot):
        prompt = prompt[: -len(eot)]
    return prompt


def _build_relevance_prompt(tokenizer, instruction: str, sentence: str) -> str:
    user_msg = (
        f"[Instruction Start]\n{instruction}\n[Instruction End]\n"
        f"[Sentence Fragment Start]\n{sentence}\n[Sentence Fragment End]"
    )
    chat = [
        {"role": "system", "content": PROMPT_TEMPLATES["relevance"]},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "Rating: [["},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )
    eot = "<|eot_id|>"
    if prompt.endswith(eot):
        prompt = prompt[: -len(eot)]
    return prompt


def build_fluency_prompts(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    df["fluency_prompt"] = df["post-intervention-response"].apply(
        lambda x: _build_fluency_prompt(tokenizer, x)
    )
    return df


def build_relevance_prompts(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    df["relevance_prompt"] = df.apply(
        lambda row: _build_relevance_prompt(
            tokenizer, row["data_path_query"], row["post-intervention-response"]
        ),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_no_nans(df: pd.DataFrame):
    if not df.isna().any().any():
        return
    bad = df[df.isna().any(axis=1)]
    lines = []
    for idx, row in bad.iterrows():
        nan_cols = row.index[row.isna()].tolist()
        lines.append(f"  [row {idx}] NaN cols={nan_cols}")
    raise ValueError("NaNs detected:\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build judge prompt CSVs")
    sub = parser.add_subparsers(dest="mode", required=True)

    for mode in ("judge", "fluency", "relevance", "all"):
        sp = sub.add_parser(mode)
        sp.add_argument("--input", default="merged_eval_outputs.csv")
        sp.add_argument("--output", default=None,
                        help="Output CSV path (defaults depend on mode)")
        sp.add_argument("--tokenizer", default=TOKENIZER_MODEL_NAME)

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"Reading {args.input}")
    df = pd.read_csv(args.input, keep_default_na=False,
                     dtype={"post-intervention-response": str,
                            "original-response": str,
                            "query": str,
                            "data_path_query": str})

    if args.mode in ("judge", "all"):
        print("Building judge prompts...")
        df = build_judge_prompts(df, tokenizer)
        out = args.output or "judge_prompts.csv"
        if args.mode == "judge":
            check_no_nans(df)
            df.to_csv(out, index=False)
            print(f"Saved {out}")

    if args.mode in ("fluency", "relevance", "all"):
        if args.mode in ("fluency", "all"):
            print("Building fluency prompts...")
            df = build_fluency_prompts(df, tokenizer)
        if args.mode in ("relevance", "all"):
            print("Building relevance prompts...")
            df = build_relevance_prompts(df, tokenizer)
        out = args.output or "relevance_fluency_prompts.csv"
        if args.mode != "all":
            check_no_nans(df)
            df.to_csv(out, index=False)
            print(f"Saved {out}")

    if args.mode == "all":
        check_no_nans(df)
        df.to_csv(args.output or "judge_prompts.csv", index=False)
        # Also save the relevance/fluency version separately
        rf_out = "relevance_fluency_prompts.csv"
        df.to_csv(rf_out, index=False)
        print(f"Saved judge_prompts.csv and {rf_out}")


if __name__ == "__main__":
    main()