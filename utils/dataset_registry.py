"""Dataset tag -> data dir and the add (source) / sub (base) prompt sets.

The steering vector is add - sub, added to sub prompts, so a dataset's task dir is
`from_<sub>_to_<add>`. Files live at `data/<model>/<dir>/<add|sub>-<split>.jsonl`
(split: train, valid, test; only sub needs valid/test). `concept` picks the judge template.

CLI (used by the bash scripts):
    python utils/dataset_registry.py <tag>    -> "<add> <sub> <dir>"
    python utils/dataset_registry.py --base   -> base tags ("all")
    python utils/dataset_registry.py --extra  -> every other tag
"""
import sys

SYCOPHANCY_VARIANTS = ('more-templates', 'unaligned', 'diff-length', 'unaligned-diff-length')
VERSE_VARIANTS = ('aligned', 'longer', 'varied')

DATASETS = {
    'harmful':    {'dir': 'harmful',    'add': 'harmful',         'sub': 'harmless',   'concept': 'harmful'},
    'sycophancy': {'dir': 'sycophancy', 'add': 'non-sycophantic', 'sub': 'sycophancy', 'concept': 'sycophancy'},
    'verse':      {'dir': 'verse',      'add': 'verse',           'sub': 'prose',      'concept': 'verse'},
    **{f'sycophancy-{v}': {'dir': f'sycophancy-{v}', 'add': f'non-sycophantic-{v}', 'sub': f'sycophancy-{v}',
                           'concept': 'sycophancy', 'variant_of': 'sycophancy'} for v in SYCOPHANCY_VARIANTS},
    **{f'verse-{v}': {'dir': f'verse-{v}', 'add': f'verse-{v}', 'sub': f'prose-{v}',
                      'concept': 'verse', 'variant_of': 'verse'} for v in VERSE_VARIANTS},
    'verse-one-token': {'dir': 'verse-one-token', 'add': 'verse', 'sub': 'prose', 'concept': 'verse'},
    'refusal-help-hurt': {'dir': 'refusal-help-hurt', 'add': 'refusal-hurt', 'sub': 'refusal-help', 'concept': 'harmful'},
    'refusal-buy-steal': {'dir': 'refusal-buy-steal', 'add': 'refusal-steal', 'sub': 'refusal-buy', 'concept': 'harmful'},
    'refusal-legally-illegally': {'dir': 'refusal-legally-illegally', 'add': 'refusal-illegally',
                                  'sub': 'refusal-legally', 'concept': 'harmful'},
}

BASE_DATASETS = ('harmful', 'sycophancy', 'verse')
VARIANT_DATASETS = tuple(t for t, d in DATASETS.items() if 'variant_of' in d)
EXTRA_DATASETS = tuple(t for t in DATASETS if t not in BASE_DATASETS)
# Tags with their own task dir (verse-one-token shares verse's from_prose_to_verse).
TASK_DATASETS = tuple(t for t in DATASETS if t != 'verse-one-token')


def task_dir(tag):
    d = DATASETS[tag]
    return f"from_{d['sub']}_to_{d['add']}"


def by_task(task):
    """(tag, entry) for a `from_<sub>_to_<add>` task dir name."""
    for tag in TASK_DATASETS:
        if task_dir(tag) == task:
            return tag, DATASETS[tag]
    raise KeyError(f"no dataset in utils/dataset_registry.py has task dir {task!r}")


def data_path(data_root, model_name, tag, side, split):
    d = DATASETS[tag]
    return f"{data_root}/{model_name}/{d['dir']}/{d[side]}-{split}.jsonl"


def lookup(source, base=None):
    """The registry entry whose add is `source` (and sub is `base`, when given); None if absent."""
    for tag, d in DATASETS.items():
        if d['add'] == source and (base is None or d['sub'] == base):
            return tag, d
    return None


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else ''
    if arg == '--base':
        print(' '.join(BASE_DATASETS))
    elif arg == '--extra':
        print(' '.join(EXTRA_DATASETS))
    elif arg in DATASETS:
        d = DATASETS[arg]
        print(d['add'], d['sub'], d['dir'])
    else:
        sys.exit(f"unknown dataset '{arg}'. Must be one of: {' '.join(DATASETS)}")
