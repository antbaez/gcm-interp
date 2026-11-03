#!/usr/bin/env python3
"""Turns a settings/*.yaml file into the bash arrays run_normal.sh /
run_preemptable.sh submit with: MODELS, DATASETS, RESULTS_DIR (absolute; relative
paths resolve against the repo root), SCOPES / SCOPE_FLAGS_<scope>, NORMS (normalized
and/or unnormalized) / NORM_FLAGS_<norm> (--unnormalized) / TYPE_GROUPS_<norm> (one job each:
a space-separated --type list, or '' for run_steering.sh's default), STEERING_N_<scope>_<norm>
(model tag -> space-separated N sweep, only for models the settings give one), and JOB_FLAGS
(the flags shared by every job, forwarded to scripts/run_<normal|preemptable>_job.sh).

normalization is true, false, or [true, false] (one job per normalization). steering_type and
steering_n may then be keyed by normalization ({normalized: ..., unnormalized: ...}, one entry
per normalization run) to give each its own methods / N sweep.

jacobian (null for none, or keyed by normalization too) lists the Jacobian-lens steering
aspects, each a list: fit (prose: J fit on the base set; both: (J_base + J_source) / 2),
pooling (mean | sum over the prompt positions) and solver (ridge | pinv | trunc). Every
combination runs as steering type jacobian-[both-]<pooling>-<solver>; each fit x pooling pair is
its own job (over its solvers), and the steering_type methods another.

Usage: eval "$(python3 scripts/load_settings.py <settings.yaml> normal|preemptable)"

Every key must be present and valid; unknown keys are rejected so typos fail loudly.
"""
import glob
import os
import re
import shlex
import sys
from fractions import Fraction

import yaml

ALL_MODELS = ['olmo', 'qwen', 'qwen3', 'gemma', 'gemma4', 'llama']
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from utils.dataset_registry import DATASETS, task_dir

ALL_DATASETS = list(DATASETS)
MODEL_DIRS = {'olmo': 'OLMo-2-1124-13B-DPO', 'qwen': 'Qwen1.5-14B-Chat', 'qwen3': 'Qwen3-14B',
              'gemma': 'gemma-3-12b-it', 'gemma4': 'gemma-4-12B-it', 'llama': 'Llama-3.1-8B-Instruct'}
PREEMPTABLE_ONLY = {'seed': None, 'hook_prompt_acts': False, 'hook_response_acts': False, 'hook_target_acts': False,
                    'hook_batch_size': None}
KEYS = ['results_dir', 'models', 'datasets', 'steering_type', 'jacobian', 'steering_n', 'global_steering_n', 'normalization', 'scope',
        'local_layer', 'split', 'generation', 'judging', 'judge', 'api_judge_layers', 'api_judge_n', 'no_model_cache', 'metric',
        'train_batch_size', 'generation_batch_size',
        'seed', 'hook_prompt_acts', 'hook_response_acts', 'hook_target_acts', 'hook_batch_size']


def fail(msg):
    sys.exit('Error in settings: ' + msg)


def choice(settings, key, options):
    value = settings[key]
    if value not in options:
        fail('{} must be one of {} (got {!r})'.format(key, options, value))
    return value


def boolean(settings, key):
    value = settings[key]
    if not isinstance(value, bool):
        fail('{} must be true or false (got {!r})'.format(key, value))
    return value


def tag_list(settings, key, options):
    value = settings[key]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        fail('{} must be a non-empty list'.format(key))
    for v in value:
        if v not in options:
            fail('{}: unknown tag {!r} (expected some of {})'.format(key, v, options))
    return value


def fraction(value, what):
    try:
        x = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        fail('{} must be a fraction like 2/3 or 0.6667 (got {!r})'.format(what, value))
    if not 0 <= x <= 1:
        fail('{} must be between 0 and 1 (got {!r})'.format(what, value))
    return repr(x)


def layer_flags(local_layer):
    if not isinstance(local_layer, dict):
        fail('local_layer must be a mapping with mode, single and sweep')
    mode = local_layer.get('mode')
    if mode == 'single':
        return ['--layer-frac', fraction(local_layer.get('single'), 'local_layer.single')]
    if mode == 'sweep':
        sweep = local_layer.get('sweep')
        if not isinstance(sweep, list) or len(sweep) != 2:
            fail('local_layer.sweep must be [start, end] (got {!r})'.format(sweep))
        start = fraction(sweep[0], 'local_layer.sweep start')
        end = fraction(sweep[1], 'local_layer.sweep end')
        if float(start) > float(end):
            fail('local_layer.sweep start must not exceed its end (got {!r})'.format(sweep))
        return ['--layer-range', start, end]
    fail("local_layer.mode must be 'single' or 'sweep' (got {!r})".format(mode))


def layer_chunks(settings):
    """'start end' layer-range thirds when the local sweep covers every layer, else []."""
    local_layer = settings['local_layer']
    if local_layer['mode'] != 'sweep':
        return []
    start, end = (float(Fraction(str(v))) for v in local_layer['sweep'])
    if (start, end) != (0.0, 1.0):
        return []
    bounds = [repr(float(Fraction(i, 3))) for i in range(4)]
    return ['{} {}'.format(bounds[i], bounds[i + 1]) for i in range(3)]


NORM_KEYS = ('normalized', 'unnormalized')


def normalizations(settings):
    """Run normalizations, normalized first: normalization is true, false, or [true, false]."""
    value = settings['normalization']
    if isinstance(value, bool):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(v, bool) for v in value) \
            or len(set(value)) != len(value):
        fail('normalization must be true, false, or [true, false] (got {!r})'.format(settings['normalization']))
    return [n for n, flag in zip(NORM_KEYS, (True, False)) if flag in value]


def per_norm(settings, key, norm, norms):
    """settings[key] for normalization `norm`: the value itself, or its `norm` entry when keyed
    by normalization (which must then cover every run normalization)."""
    value = settings[key]
    if isinstance(value, dict) and value and set(value) <= set(NORM_KEYS):
        missing = [n for n in norms if n not in value]
        if missing:
            fail('{} is keyed by normalization but has no entry for {}'.format(key, ', '.join(missing)))
        return value[norm]
    return value


JACOBIAN_ASPECTS = {'fit': {'prose': '', 'both': 'both-'}, 'pooling': ['mean', 'sum'],
                    'solver': ['ridge', 'pinv', 'trunc']}


def jacobian_groups(settings, norm, norms):
    """[[jacobian-[both-]<pooling>-<solver> over the solvers] for each fit x pooling pair]."""
    value = per_norm(settings, 'jacobian', norm, norms)
    if value is None:
        return []
    if not isinstance(value, dict) or set(value) != set(JACOBIAN_ASPECTS):
        fail('jacobian must be null or a mapping with {} (got {!r})'.format(', '.join(JACOBIAN_ASPECTS), value))
    for aspect, options in JACOBIAN_ASPECTS.items():
        chosen = value[aspect]
        if not isinstance(chosen, list) or not chosen or any(c not in options for c in chosen) \
                or len(set(chosen)) != len(chosen):
            fail('jacobian.{} must be a non-empty list of {} (got {!r})'.format(aspect, list(options), chosen))
    return [['jacobian-{}{}-{}'.format(JACOBIAN_ASPECTS['fit'][f], p, s) for s in value['solver']]
            for f in value['fit'] for p in value['pooling']]


def type_groups(settings, norm, norms):
    """The --type list of each job: the steering_type methods, then one per Jacobian fit x pooling
    pair; [''] (run_steering.sh's default types) when neither is set."""
    steering_type = per_norm(settings, 'steering_type', norm, norms)
    if steering_type is not None:
        if isinstance(steering_type, str):
            steering_type = re.split(r'[,\s]+', steering_type.strip())
        if not isinstance(steering_type, list) or not all(isinstance(t, str) and t for t in steering_type):
            fail('steering_type must be null or a list like [last, mean] (got {!r})'.format(steering_type))
    jacobian = [[t for t in g if t not in (steering_type or [])] for g in jacobian_groups(settings, norm, norms)]
    groups = ([steering_type] if steering_type else []) + [g for g in jacobian if g]
    if not groups:
        if steering_type is not None:
            fail('steering_type for {} is empty and jacobian adds nothing'.format(norm))
        return ['']
    return [' '.join(g) for g in groups]


def check_jacobian_fits(settings, models, datasets, norms):
    """Refuses to submit steering jobs whose Jacobian directions have not been fitted."""
    if settings['generation'] is not True:
        return
    types = sorted({t for norm in norms for g in type_groups(settings, norm, norms) for t in g.split()
                    if t.startswith('jacobian-')})
    for model in models:
        for dataset in datasets:
            task = task_dir(dataset)
            root = os.path.join(REPO_ROOT, 'jacobian', MODEL_DIRS[model], task)
            missing = [t for t in types if not glob.glob(os.path.join(
                glob.escape(root), 'layer=*', 'direction_{}.pt'.format(t[len('jacobian-'):])))]
            if missing:
                fail('no fitted direction in jacobian/{}/{}/ for: {}\nFit them first, then launch this again once '
                     'that job has finished:\n  sbatch jacobian/run_fit_lens.sh --model {} --dataset {} --layer <steered layer>'.format(
                         MODEL_DIRS[model], task, ' '.join(missing), model, dataset))


def judges(settings):
    """The judges to run, the local one first: 'local', or an OpenAI model name for the API."""
    value = settings['judge']
    if not isinstance(value, list) or not value or not all(
            isinstance(j, str) and re.fullmatch(r'[A-Za-z0-9._:-]+', j) for j in value):
        fail("judge must be a non-empty list of 'local' and/or OpenAI model names (got {!r})".format(value))
    if settings['split'] != 'val' and 'local' not in value:
        fail("judge must include 'local' unless split is val: selection reads the local judge's ratings")
    if settings['judging'] is True and set(value) != {'local'} and not os.environ.get('OPENAI_API_KEY'):
        fail('judge lists an OpenAI model, so OPENAI_API_KEY must be set in .env or this shell')
    return sorted(set(value), key=lambda j: (j != 'local', value.index(j)))


def job_flags(settings, dispatcher):
    flags = []

    split = choice(settings, 'split', ['all', 'val', 'test'])
    if split != 'all':
        flags += ['--split', split]
    if not boolean(settings, 'generation'):
        flags.append('--no-generation')
    if not boolean(settings, 'judging'):
        flags.append('--no-judging')
    run_judges = judges(settings)
    if run_judges != ['local']:
        flags += ['--judge', ' '.join(run_judges)]
    api_layers = settings['api_judge_layers']
    if api_layers is not None:
        if not isinstance(api_layers, list) or not api_layers or not all(
                isinstance(l, int) and not isinstance(l, bool) and l >= 0 for l in api_layers):
            fail('api_judge_layers must be null or a non-empty list of layer indices (got {!r})'.format(api_layers))
        if run_judges == ['local']:
            fail('api_judge_layers is set, but judge lists no OpenAI model')
        flags += ['--api-judge-layers', ' '.join(str(l) for l in api_layers)]
    api_n = settings['api_judge_n']
    if api_n is not None:
        if not isinstance(api_n, list) or not api_n or not all(
                isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0 for n in api_n):
            fail('api_judge_n must be null or a non-empty list of positive N values (got {!r})'.format(api_n))
        if run_judges == ['local']:
            fail('api_judge_n is set, but judge lists no OpenAI model')
        flags += ['--api-judge-n', ' '.join(str(n) for n in api_n)]
    if boolean(settings, 'no_model_cache'):
        flags.append('--no-model-cache')
    if choice(settings, 'metric', ['harmonic', 'pass_rate']) == 'pass_rate':
        flags.append('--pass-rate')
    for key, flag in (('train_batch_size', '--train-batch-size'), ('generation_batch_size', '--gen-batch-size')):
        batch_size = settings[key]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            fail('{} must be a positive integer (got {!r})'.format(key, batch_size))
        flags += [flag, str(batch_size)]

    if dispatcher == 'normal':
        for key, default in PREEMPTABLE_ONLY.items():
            if settings[key] != default:
                fail('{} is only supported by run_preemptable.sh; set it to {} for run_normal.sh'.format(
                    key, 'null' if default is None else str(default).lower()))
    else:
        seed = settings['seed']
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                fail('seed must be an integer or null (got {!r})'.format(seed))
            flags += ['--seed', str(seed)]
        hooks = [flag for flag in ('hook_prompt_acts', 'hook_response_acts', 'hook_target_acts') if boolean(settings, flag)]
        hook_batch_size = settings['hook_batch_size']
        if hooks:
            if isinstance(hook_batch_size, bool) or not isinstance(hook_batch_size, int) or hook_batch_size < 1:
                fail('hook_batch_size must be a positive integer when {} is on (got {!r})'.format(
                    ' / '.join(hooks), hook_batch_size))
            flags += ['--' + flag for flag in hooks] + ['--hook-batch-size', str(hook_batch_size)]

    return flags


def scopes(settings):
    """[(scope, flags)] in run order: scope is local, global, or a list of both (local first)."""
    value = settings['scope']
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or any(v not in ('local', 'global') for v in value) \
            or len(set(value)) != len(value):
        fail('scope must be local, global, or [local, global] (got {!r})'.format(settings['scope']))
    return [(s, ['--global'] if s == 'global' else layer_flags(settings['local_layer']))
            for s in ('local', 'global') if s in value]


def steering_n(settings, models, key='steering_n', norm='normalized', norms=('normalized',)):
    """{model tag: 'N1 N2 ...'} for the models the settings set a sweep for. steering_n (local
    scope) / global_steering_n (global scope) is null (run_steering.sh's per-model tables for
    every model), a list (every model), or a mapping of model tag -> list or null (null /
    unlisted models keep the tables); any of these may be keyed by normalization."""
    value = per_norm(settings, key, norm, norms)
    if value is None:
        return {}
    if isinstance(value, list):
        value = {m: value for m in models}
    if not isinstance(value, dict):
        fail('{} must be null, a list like [1, 10, 20], or a mapping like {{llama: [1, 10, 20]}} (got {!r})'.format(key, value))
    out = {}
    for model, ns in value.items():
        if model not in ALL_MODELS:
            fail('{}: unknown model tag {!r} (expected some of {})'.format(key, model, ALL_MODELS))
        if ns is None or model not in models:
            continue
        if (not isinstance(ns, list) or not ns
                or not all(isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0 for n in ns)):
            fail('{} for {} must be a non-empty list of positive numbers (got {!r})'.format(key, model, ns))
        out[model] = ' '.join('{:g}'.format(n) for n in ns)
    return out


def results_dir(settings):
    value = settings['results_dir']
    if not isinstance(value, str) or not value.strip():
        fail('results_dir must be a directory path (got {!r})'.format(value))
    return os.path.normpath(os.path.join(REPO_ROOT, value))


def bash_array(name, values):
    return '{}=({})'.format(name, ' '.join(shlex.quote(v) for v in values))


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ('normal', 'preemptable'):
        sys.exit('Usage: load_settings.py <settings.yaml> normal|preemptable')
    path, dispatcher = sys.argv[1], sys.argv[2]

    with open(path) as f:
        settings = yaml.safe_load(f) or {}
    if not isinstance(settings, dict):
        fail('{} must be a YAML mapping'.format(path))
    unknown = sorted(set(settings) - set(KEYS))
    missing = [k for k in KEYS if k not in settings]
    if unknown:
        fail('unknown key(s): {}'.format(', '.join(unknown)))
    if missing:
        fail('missing key(s): {}'.format(', '.join(missing)))

    models = tag_list(settings, 'models', ALL_MODELS)
    run_datasets = tag_list(settings, 'datasets', ALL_DATASETS)
    run_scopes = scopes(settings)
    norms = normalizations(settings)
    check_jacobian_fits(settings, models, run_datasets, norms)
    print(bash_array('MODELS', models))
    print(bash_array('DATASETS', run_datasets))
    print(bash_array('SCOPES', [s for s, _ in run_scopes]))
    print(bash_array('NORMS', norms))
    for norm in norms:
        print(bash_array('NORM_FLAGS_' + norm, ['--unnormalized'] if norm == 'unnormalized' else []))
        print(bash_array('TYPE_GROUPS_' + norm, type_groups(settings, norm, norms)))
    for scope, flags in run_scopes:
        print(bash_array('SCOPE_FLAGS_' + scope, flags))
        key = 'steering_n' if scope == 'local' else 'global_steering_n'
        for norm in norms:
            print('declare -A STEERING_N_{}_{}=({})'.format(scope, norm, ' '.join(
                '[{}]={}'.format(m, shlex.quote(ns)) for m, ns in steering_n(settings, models, key, norm, norms).items())))
    out_dir = results_dir(settings)
    print('RESULTS_DIR={}'.format(shlex.quote(out_dir)))
    print(bash_array('LAYER_CHUNKS', layer_chunks(settings)))
    print('SPLIT={}'.format(shlex.quote(settings['split'])))
    print(bash_array('JOB_FLAGS', job_flags(settings, dispatcher) + ['--results-dir', out_dir]))


if __name__ == '__main__':
    main()
