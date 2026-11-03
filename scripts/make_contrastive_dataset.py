import argparse
import json
import re
import sys
from pathlib import Path

SPLITS = ('train', 'valid', 'test')
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
sys.path.insert(0, str(REPO_ROOT))
from utils.dataset_registry import DATASETS


def match_case(src, repl):
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def swap(text, pattern, repl):
    return pattern.sub(lambda m: match_case(m.group(0), repl), text)


def contrast(path, pattern, args):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    orig_rows, swapped_rows = [], []
    for row in rows:
        content = row['prompt'][-1]['content']
        if not pattern.search(content):
            if args.allow_missing:
                continue
            sys.exit(f"{path.name} id {row['id']} has no '{args.word}': {content!r} (use --allow-missing to drop it)")
        swapped = json.loads(json.dumps(row))
        swapped['prompt'][-1]['content'] = swap(content, pattern, args.to)
        orig_rows.append(row)
        swapped_rows.append(swapped)
    return orig_rows, swapped_rows


def main():
    p = argparse.ArgumentParser(description='Clone a <name>-train.jsonl dataset (plus its -valid/-test splits, if present) into a contrastive pair differing by one word.')
    p.add_argument('--input', default=str(DATA_DIR / 'gemma-4-12B-it/helping/helping-train.jsonl'))
    p.add_argument('--word', default='help', help='word to change (whole-word, case-insensitive)')
    p.add_argument('--to', required=True, help='replacement word')
    p.add_argument('--prefix', default='', help='prefix for the output dir and files, e.g. refusal -> refusal-<word>-<to>/refusal-<word>-train.jsonl')
    p.add_argument('--models', default=None,
                   help="comma-separated model dirs under data/, or 'all' (default: the input's model dir)")
    p.add_argument('--allow-missing', action='store_true', help='drop prompts that lack the word instead of erroring')
    p.add_argument('--force', action='store_true', help='overwrite existing output files')
    args = p.parse_args()

    src_path = Path(args.input)
    stem = src_path.name.removesuffix('.jsonl').removesuffix('-train')
    prefix = f'{args.prefix}-' if args.prefix else ''
    pattern = re.compile(rf'\b{re.escape(args.word)}\b', re.IGNORECASE)

    splits = {}
    for suffix in SPLITS:
        path = src_path.with_name(f'{stem}-{suffix}.jsonl')
        if path.exists():
            splits[suffix] = contrast(path, pattern, args)

    if args.models == 'all':
        models = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    elif args.models:
        models = args.models.split(',')
    else:
        models = [src_path.resolve().parent.parent.name]

    word, to = args.word.lower(), args.to.lower()
    for model in models:
        out_dir = DATA_DIR / model / f'{prefix}{word}-{to}'
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix, (orig_rows, swapped_rows) in splits.items():
            for tag, data in ((word, orig_rows), (to, swapped_rows)):
                out = out_dir / f'{prefix}{tag}-{suffix}.jsonl'
                if out.exists() and not args.force:
                    print(f'skip (exists): {out}')
                    continue
                out.write_text(''.join(json.dumps(r) + '\n' for r in data))
                print(f'wrote {len(data)} prompts: {out}')

    tag = f'{prefix}{word}-{to}'
    if tag not in DATASETS:
        print(f"add to utils/dataset_registry.py DATASETS: '{tag}': {{'dir': '{tag}', 'add': '{prefix}{to}', 'sub': '{prefix}{word}', 'concept': ...}}")


if __name__ == '__main__':
    main()
