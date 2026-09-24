"""
Generate and judge held-out test splits locally (no sbatch) for an explicit
list of steering configs, saving the raw generations and judge scores.

The job scripts only ever run the held-out split at the config
select_best_config.py picked. This runs it at whatever (N, layer) you list,
e.g. to try a config by hand on an interactive GPU node.

Spec format (JSON):

    {
      "device": "cuda:0",                       # optional, default cuda:0
      "setup": "setup/setup.sh",                # optional, sourced before generation
      "judge_setup": "setup/setup_judging.sh",  # optional, sourced before judging
      "tests": [
        {"model": "llama", "dataset": "harmful", "method": "last", "N": 10, "layer": 12},
        {"model": "llama", "dataset": "harmful", "method": "last", "N": 20, "layer": 12},
        {"model": "qwen3", "dataset": "verse",   "method": "mean", "N": 50, "layer": 18}
      ]
    }

`model` is a run_steering.sh tag (olmo|qwen|qwen3|gemma|gemma4|llama), `dataset`
one of harmful|sycophancy|verse, `method` one of last|mean|positional. The setup
paths are relative to the repo root; if omitted they are sourced only when they
exist. Only the default pipeline mode is covered: normalized residual-stream,
KV-cached, local (single-layer) steering.

A nested best-configs file is also accepted in place of a spec, one test per entry:

    {"<model dir>": {"from_<source>_to_<base>": {"<stream>": {"<scope>": {
        "<method>": {"N": 30.0, "layer": 19, ...}}}}}}

e.g. best_configs_harmonic_floor.json. Method aliases (mean-padding -> mean, weighted-pos -> positional) are
mapped; entries for tasks, streams or scopes this script can't run (e.g. dataset
variants like sycophancy-more-templates, or non-residuals/non-local) are skipped
with a note. Such a file has no top-level options, so use --device.

run.py reads one config per (model, task, method) from its best_configs file, so
tests are packed into rounds with at most one config per key. Each round gets a
temporary best_configs-format file and one run_steering.sh call per model;
judging runs once at the end. Generation already on disk is skipped.

Held-out runs at a non-selected config share results/ and workdirs/ with the
pipeline's own: the next select_best_config.py call (without --no-prune) deletes
them, and until then collect_pass_rates.py may report them as the held-out
result. A warning is printed for any such test.

To survive that pruning, each test's generated responses (`*_gen.json`/`.txt`)
and judge scores (`{fluency,relevance,judge}_ratings.jsonl`) are copied into
harmonic_results/ at the repo root, mirroring their results/ and workdirs/
paths:

    harmonic_results/generations/<model>/<task>/normalized/residuals/cache/local/<method>/N=..._gen.{json,txt}
    harmonic_results/judge_scores/<model>/<task>/normalized/residuals/cache/local/<method>/<condition>/*_ratings.jsonl

Judge scores are only copied once a test is fully judged, and each run
refreshes the copies from the current source files.

olmo, qwen3 and llama weights are not kept on disk: each model is downloaded
into its own temp HF cache (under --weights-dir, default $TMPDIR or /tmp) for
generation, and that cache is deleted after the model's last generation call,
or on error. They are re-downloaded every run. Weights already in the normal
HF cache are left alone, and the judge model is unaffected.

Usage:
    python run_harmonic_tests.py tests.json
    python run_harmonic_tests.py tests.json --dry-run
    python run_harmonic_tests.py tests.json --save-only
    python run_harmonic_tests.py ../best_configs_harmonic_floor.json --device cuda:1
"""

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "judge-evals"))

from selection_utils import parse_condition_dir

RESULTS_ROOT = REPO_ROOT / "results"
WORKDIRS_ROOT = REPO_ROOT / "judge-evals" / "workdirs"
BEST_CONFIGS = REPO_ROOT / "judge-evals" / "best_configs.json"
HARMONIC_ROOT = REPO_ROOT / "harmonic_results"
RATING_FILES = ("fluency_ratings.jsonl", "relevance_ratings.jsonl", "judge_ratings.jsonl")

# Model tag -> results/ dir name (mirrors run_steering.sh's MODEL_ID case).
MODEL_DIRS = {
    "olmo": "OLMo-2-1124-13B-DPO",
    "qwen": "Qwen1.5-14B-Chat",
    "qwen3": "Qwen3-14B",
    "gemma": "gemma-3-12b-it",
    "gemma4": "gemma-4-12B-it",
    "llama": "Llama-3.1-8B-Instruct",
}

# Dataset tag -> (task dir, base name).
DATASET_TASKS = {
    "harmful":    ("from_harmful-long_to_harmless",           "harmless"),
    "sycophancy": ("from_non-sycophantic-long_to_sycophancy", "sycophancy"),
    "verse":      ("from_verse-long_to_prose",                "prose"),
}

METHODS = ("last", "mean", "positional")

# Method names other selection outputs use -> run_steering.sh --type names.
METHOD_ALIASES = {
    "last-token": "last",
    "mean-padding": "mean", "mean_padding": "mean",
    "weighted-pos": "positional", "weighted_pos": "positional",
}

# Models whose weights are downloaded into a throwaway HF cache for generation
# and deleted afterwards, instead of persisting in the shared HF cache.
UNCACHED_MODELS = {"olmo", "qwen3", "llama"}
# Every env var transformers / huggingface_hub may read the hub cache location
# from; all are exported, after the setup script, so none can override it.
HF_CACHE_VARS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")

# The only mode covered; matches select_best_config.py's constants.
MODE_REL = Path("normalized") / "residuals" / "cache" / "local"


def tests_from_best_configs(best: dict, path: Path) -> list[dict]:
    """Flatten a nested model/task/stream/scope/method best-configs file into tests."""
    model_tags = {d: tag for tag, d in MODEL_DIRS.items()}
    dataset_tags = {task: tag for tag, (task, _) in DATASET_TASKS.items()}
    tests, skipped = [], []
    for model_dir, tasks in best.items():
        for task, streams in tasks.items():
            for stream, scopes in streams.items():
                for scope, methods in scopes.items():
                    for method, entry in methods.items():
                        where = f"{model_dir}/{task}/{stream}/{scope}/{method}"
                        model = model_tags.get(model_dir, entry.get("model_tag"))
                        if model not in MODEL_DIRS:
                            skipped.append(f"{where} (unknown model)")
                        elif task not in dataset_tags:
                            skipped.append(f"{where} (unsupported task)")
                        elif (stream, scope) != ("residuals", "local"):
                            skipped.append(f"{where} (only residuals/local is supported)")
                        else:
                            tests.append({
                                "model": model, "dataset": dataset_tags[task],
                                "method": METHOD_ALIASES.get(method, method),
                                "N": entry["N"], "layer": entry["layer"],
                            })
    for s in skipped:
        print(f"{path}: skipping {s}")
    return tests


def load_spec(path: Path) -> dict:
    with open(path) as f:
        spec = json.load(f)
    if "tests" not in spec:
        spec = {"tests": tests_from_best_configs(spec, path)}
    tests = spec.get("tests")
    if not isinstance(tests, list) or not tests:
        raise SystemExit(f"{path}: 'tests' must be a non-empty list")

    seen = set()
    for i, t in enumerate(tests):
        where = f"{path}: tests[{i}]"
        missing = {"model", "dataset", "method", "N", "layer"} - t.keys()
        if missing:
            raise SystemExit(f"{where}: missing {sorted(missing)}")
        if t["model"] not in MODEL_DIRS:
            raise SystemExit(f"{where}: unknown model '{t['model']}' (one of {', '.join(MODEL_DIRS)})")
        if t["dataset"] not in DATASET_TASKS:
            raise SystemExit(f"{where}: unknown dataset '{t['dataset']}' (one of {', '.join(DATASET_TASKS)})")
        if t["method"] not in METHODS:
            raise SystemExit(f"{where}: unknown method '{t['method']}' (one of {', '.join(METHODS)})")
        if isinstance(t["N"], bool) or not isinstance(t["N"], (int, float)) or t["N"] <= 0:
            raise SystemExit(f"{where}: N must be a positive number")
        # run.py only pins the layer when it's an int; anything else silently
        # falls back to sweeping the whole layer range.
        if isinstance(t["layer"], bool) or not isinstance(t["layer"], int):
            raise SystemExit(f"{where}: layer must be an integer")
        t["N"] = float(t["N"])
        key = (t["model"], t["dataset"], t["method"], t["N"], t["layer"])
        if key in seen:
            raise SystemExit(f"{where}: duplicate test {key}")
        seen.add(key)
    return spec


def test_paths(t: dict):
    task, base = DATASET_TASKS[t["dataset"]]
    rel = Path(MODEL_DIRS[t["model"]]) / task / MODE_REL / t["method"]
    return rel, task, f"{base}-heldout-test"


def label(t: dict) -> str:
    return f"{t['model']}/{t['dataset']}/{t['method']} N={t['N']:g} layer={t['layer']}"


def generated_file(t: dict) -> Path | None:
    rel, _, test_file = test_paths(t)
    for gen_json in (RESULTS_ROOT / rel).glob(f"*_{test_file}_gen.json"):
        meta = parse_condition_dir(gen_json.name[: -len("_gen.json")])
        if meta and meta["N"] == t["N"] and meta["value"] == str(t["layer"]):
            return gen_json
    return None


def is_generated(t: dict) -> bool:
    return generated_file(t) is not None


def judged_dir(t: dict) -> Path | None:
    """The test's judging workdir, if its judge_ratings.jsonl has been written."""
    rel, _, test_file = test_paths(t)
    for cond_dir in (WORKDIRS_ROOT / rel).glob(f"*_{test_file}"):
        meta = parse_condition_dir(cond_dir.name)
        ratings = cond_dir / "judge_ratings.jsonl"
        if (meta and meta["N"] == t["N"] and meta["value"] == str(t["layer"])
                and ratings.exists() and ratings.stat().st_size > 0):
            return cond_dir
    return None


def warn_if_unselected(tests: list[dict]):
    """Flag tests that select_best_config.py's pruning would delete."""
    if not BEST_CONFIGS.exists():
        return
    with open(BEST_CONFIGS) as f:
        best = json.load(f)
    for t in tests:
        task, _ = DATASET_TASKS[t["dataset"]]
        entry = best.get(MODEL_DIRS[t["model"]], {}).get(task, {}).get(t["method"])
        if entry and not (float(entry["N"]) == t["N"] and entry["layer"] == t["layer"]):
            print(f"WARNING: {label(t)} is not the validation-selected config "
                  f"(N={entry['N']:g} layer={entry['layer']}); select_best_config.py will prune it")


def pack_rounds(tests: list[dict]) -> list[list[dict]]:
    """Split tests into rounds holding at most one config per (model, dataset, method)."""
    rounds: list[list[dict]] = []
    for t in tests:
        key = (t["model"], t["dataset"], t["method"])
        for r in rounds:
            if all((o["model"], o["dataset"], o["method"]) != key for o in r):
                r.append(t)
                break
        else:
            rounds.append([t])
    return rounds


def resolve_setup(spec: dict, key: str, default: str) -> Path | None:
    if key in spec:
        path = REPO_ROOT / spec[key]
        if not path.exists():
            raise SystemExit(f"'{key}' file not found: {path}")
        return path
    path = REPO_ROOT / default
    return path if path.exists() else None


def format_duration(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def run_script(setup: Path | None, script: str, args: list[str], dry_run: bool, what: str,
               env: dict[str, str] | None = None) -> float:
    """Run a pipeline script, printing and returning how long it took (0 on a dry run).

    `env` is exported after `setup` is sourced, so it wins over anything setup sets.
    """
    cmd = f"bash {shlex.quote(script)} " + " ".join(shlex.quote(a) for a in args)
    if env:
        cmd = " ".join(f"export {k}={shlex.quote(v)};" for k, v in env.items()) + " " + cmd
    if setup is not None:
        cmd = f"source {shlex.quote(str(setup))} && {cmd}"
    print(f"\n$ {cmd}", flush=True)
    if dry_run:
        return 0.0
    start = time.perf_counter()
    subprocess.run(["bash", "-c", cmd], cwd=REPO_ROOT, check=True)
    elapsed = time.perf_counter() - start
    print(f"\n[time] {what} took {format_duration(elapsed)}", flush=True)
    return elapsed


def generate(tests: list[dict], spec: dict, device: str, dry_run: bool, weights_dir: Path | None):
    pending = [t for t in tests if not is_generated(t)]
    for t in tests:
        if t not in pending:
            print(f"Already generated: {label(t)}")
    if not pending:
        return

    setup = resolve_setup(spec, "setup", "setup/setup.sh")
    rounds = pack_rounds(pending)
    # A model's throwaway weight cache lives until its last round, so it's
    # downloaded once per run rather than once per round.
    last_round = {t["model"]: i for i, r in enumerate(rounds, 1) for t in r}
    weight_caches: dict[str, Path] = {}

    def drop_weights(model: str):
        cache = weight_caches.pop(model, None)
        if cache is not None:
            shutil.rmtree(cache, ignore_errors=True)
            print(f"Deleted downloaded {model} weights ({cache})", flush=True)

    try:
        _generate_rounds(rounds, setup, device, dry_run, weights_dir, last_round, weight_caches, drop_weights)
    finally:
        for model in list(weight_caches):
            drop_weights(model)


def _generate_rounds(rounds, setup, device, dry_run, weights_dir, last_round, weight_caches, drop_weights):
    total, calls = 0.0, 0
    for i, round_tests in enumerate(rounds, 1):
        best = {}
        for t in round_tests:
            task, _ = DATASET_TASKS[t["dataset"]]
            # run.py prints val_pass_rate when pinning, so the key has to exist.
            best.setdefault(MODEL_DIRS[t["model"]], {}).setdefault(task, {})[t["method"]] = {
                "N": t["N"], "layer": t["layer"], "val_pass_rate": math.nan,
            }
        fd, tmp = tempfile.mkstemp(prefix="run_tests_best_configs_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(best, f, indent=2)
            by_model = defaultdict(list)
            for t in round_tests:
                by_model[t["model"]].append(t)
            print(f"\n=== Generation round {i}: {len(round_tests)} test(s) ===")
            for model, mts in by_model.items():
                datasets = sorted({t["dataset"] for t in mts})
                methods = sorted({t["method"] for t in mts}, key=METHODS.index)
                env = None
                if model in UNCACHED_MODELS:
                    if model not in weight_caches and not dry_run:
                        if weights_dir is not None:
                            weights_dir.mkdir(parents=True, exist_ok=True)
                        weight_caches[model] = Path(tempfile.mkdtemp(
                            prefix=f"hf_weights_{model}_", dir=weights_dir))
                    cache = str(weight_caches.get(model, "<temp weight cache>"))
                    env = {var: cache for var in HF_CACHE_VARS}
                total += run_script(setup, "scripts/run_steering.sh", [
                    "--model", model, "--dataset", ",".join(datasets),
                    "--split", "test", "--best-configs", tmp,
                    "--type", " ".join(methods), "--device", device,
                ], dry_run, f"Generation round {i} for {model} ({len(mts)} test(s))", env)
                calls += 1
                if last_round[model] == i:
                    drop_weights(model)
        finally:
            os.unlink(tmp)
    if calls > 1 and not dry_run:
        print(f"\n[time] All generation took {format_duration(total)}", flush=True)


def judge(tests: list[dict], spec: dict, device: str, dry_run: bool):
    pending = [t for t in tests if judged_dir(t) is None]
    if not pending:
        print("\nAll tests already judged.")
        return
    setup = resolve_setup(spec, "judge_setup", "setup/setup_judging.sh")
    models = sorted({t["model"] for t in pending})
    datasets = sorted({t["dataset"] for t in pending})
    print(f"\n=== Judging {len(pending)} test(s) ===")
    run_script(setup, "scripts/run_judging.sh", [
        "--model", ",".join(models), "--dataset", ",".join(datasets),
        "--split", "test", "--normalized", "--device", device,
    ], dry_run, f"Judge scoring ({len(pending)} test(s))")


def save_outputs(tests: list[dict]):
    """Copy each test's generations and raw judge scores into harmonic_results/."""
    print(f"\n=== Saving to {HARMONIC_ROOT} ===")
    for t in tests:
        rel, _, _ = test_paths(t)
        gen_json = generated_file(t)
        if gen_json is not None:
            dest = HARMONIC_ROOT / "generations" / rel
            dest.mkdir(parents=True, exist_ok=True)
            for src in (gen_json, gen_json.with_suffix(".txt")):
                if src.exists():
                    shutil.copy2(src, dest / src.name)
        cond_dir = judged_dir(t)
        if cond_dir is not None:
            dest = HARMONIC_ROOT / "judge_scores" / rel / cond_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for name in RATING_FILES:
                src = cond_dir / name
                if src.exists():
                    shutil.copy2(src, dest / name)
        status = ("generations + judge scores" if cond_dir is not None
                  else "generations only (not judged)" if gen_json is not None
                  else "nothing (not generated)")
        print(f"  {label(t)}: {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path, help="JSON test spec (see format above)")
    parser.add_argument("--device", default=None,
                        help="Generation/judging device (overrides the spec's 'device'; default cuda:0)")
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="Parent dir for the throwaway olmo/qwen3/llama weight downloads "
                             "(default: the system temp dir, i.e. $TMPDIR or /tmp)")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    parser.add_argument("--save-only", action="store_true",
                        help="Skip generation and judging; just copy what's on disk to harmonic_results/")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    tests = spec["tests"]
    device = args.device or spec.get("device", "cuda:0")
    warn_if_unselected(tests)

    if not args.save_only:
        generate(tests, spec, device, args.dry_run, args.weights_dir)
        judge(tests, spec, device, args.dry_run)

    if not args.dry_run:
        save_outputs(tests)


if __name__ == "__main__":
    main()
