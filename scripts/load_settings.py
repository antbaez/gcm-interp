#!/usr/bin/env python3
"""Turns a settings/*.yaml file into the bash arrays run_normal.sh /
run_preemptable.sh submit with: MODELS, DATASETS, RESULTS_DIR (absolute; relative
paths resolve against the repo root), STEERING_N (model tag -> space-separated N
sweep, only for models the settings give one), and JOB_FLAGS (the flags forwarded to
scripts/run_<normal|preemptable>_job.sh).

Usage: eval "$(python3 scripts/load_settings.py <settings.yaml> normal|preemptable)"

Every key must be present and valid; unknown keys are rejected so typos fail loudly.
"""
import os
import re
import shlex
import sys
from fractions import Fraction

import yaml

ALL_MODELS = ['olmo', 'qwen', 'qwen3', 'gemma', 'gemma4', 'llama']
ALL_DATASETS = ['harmful', 'sycophancy', 'verse']
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREEMPTABLE_ONLY = {'seed': None, 'hook_prompt_acts': False, 'hook_response_acts': False, 'hook_target_acts': False,
                    'hook_batch_size': None}
KEYS = ['results_dir', 'models', 'datasets', 'steering_type', 'steering_n', 'global_steering_n', 'normalization', 'scope',
        'local_layer', 'split', 'generation', 'judging', 'no_model_cache', 'metric',
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


def job_flags(settings, dispatcher):
    flags = []

    steering_type = settings['steering_type']
    if steering_type is not None:
        if isinstance(steering_type, str):
            steering_type = re.split(r'[,\s]+', steering_type.strip())
        if not isinstance(steering_type, list) or not all(isinstance(t, str) and t for t in steering_type):
            fail('steering_type must be null or a list like [last, mean] (got {!r})'.format(steering_type))
        flags += ['--type', ' '.join(steering_type)]

    if not boolean(settings, 'normalization'):
        flags.append('--unnormalized')
    split = choice(settings, 'split', ['all', 'val', 'test'])
    if split != 'all':
        flags += ['--split', split]
    if not boolean(settings, 'generation'):
        flags.append('--no-generation')
    if not boolean(settings, 'judging'):
        flags.append('--no-judging')
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


def steering_n(settings, models, key='steering_n'):
    """{model tag: 'N1 N2 ...'} for the models the settings set a sweep for. steering_n (local
    scope) / global_steering_n (global scope) is null (run_steering.sh's per-model tables for
    every model), a list (every model), or a mapping of model tag -> list or null (null /
    unlisted models keep the tables)."""
    value = settings[key]
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

    print(bash_array('MODELS', tag_list(settings, 'models', ALL_MODELS)))
    models = tag_list(settings, 'models', ALL_MODELS)
    print(bash_array('DATASETS', tag_list(settings, 'datasets', ALL_DATASETS)))
    run_scopes = scopes(settings)
    print(bash_array('SCOPES', [s for s, _ in run_scopes]))
    for scope, flags in run_scopes:
        print(bash_array('SCOPE_FLAGS_' + scope, flags))
        key = 'steering_n' if scope == 'local' else 'global_steering_n'
        print('declare -A STEERING_N_{}=({})'.format(scope, ' '.join(
            '[{}]={}'.format(m, shlex.quote(ns)) for m, ns in steering_n(settings, models, key).items())))
    out_dir = results_dir(settings)
    print('RESULTS_DIR={}'.format(shlex.quote(out_dir)))
    print(bash_array('JOB_FLAGS', job_flags(settings, dispatcher) + ['--results-dir', out_dir]))


if __name__ == '__main__':
    main()
