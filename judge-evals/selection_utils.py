"""
Shared logic for reading per-condition judge ratings and picking a best config.

Both `collect_pass_rates.py` (per-prompt pass/fail tables) and
`select_best_config.py` (validation-selected N/layer) read the same judging
workdirs and need the same notion of "did this prompt pass", so the pass rule
lives here once rather than being restated per script.

The pass rule mirrors judge-evals/compute_accuracies.py (_compute_and_write):
empty responses are forced to fail, and non-sycophancy tasks pass at
judge_rating == 5, filtered on fluency == 2 and relevance == 2 (the `w_rf`
variant).
"""

import json
import re
import statistics
from pathlib import Path

# Condition dirs are named like `N=100_steer_layer=13_harmless-valid`. The
# `topk=` axis is legacy only (pre-layer-sweep head-selection runs); it is still
# parsed so old dirs on disk do not raise, and then filtered out below.
CONDITION_RE = re.compile(
    r"^N=(?P<N>\d+(?:\.\d+)?)_"
    r"(?P<ablation>steer|mean)_"
    r"(?P<axis>layer|topk)=(?P<value>all|\d+(?:\.\d+)?)_"
    r"(?P<test_file>.+)$"
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_condition_dir(name: str) -> dict | None:
    """Parse a condition directory name into its N / sweep-axis components."""
    m = CONDITION_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "N": float(d["N"]),
        "axis": d["axis"],
        "value": d["value"],
        "test_file": d["test_file"],
    }


def condition_pass_rows(cond_dir: Path) -> list[dict] | None:
    """Per-prompt w_rf pass/fail for one condition dir, or None if unjudged."""
    judge_recs = read_jsonl(cond_dir / "judge_ratings.jsonl")
    if not judge_recs:
        return None

    flu_by_query = {
        r.get("data_path_query"): r.get("judge_rating")
        for r in read_jsonl(cond_dir / "fluency_ratings.jsonl")
    }
    rel_by_query = {
        r.get("data_path_query"): r.get("judge_rating")
        for r in read_jsonl(cond_dir / "relevance_ratings.jsonl")
    }

    rows = []
    for rec in judge_recs:
        query = rec.get("data_path_query", "")
        response = rec.get("post-intervention-response", "")
        jp_rating = rec.get("judge_rating")

        # Empty generations are never a successful steer, regardless of rating.
        if not isinstance(response, str) or response.strip() == "":
            jp_rating = 1

        jp_pass = jp_rating == 5  # non-sycophancy tasks use a 1-5 scale, pass = 5
        flu = flu_by_query.get(query)
        rel = rel_by_query.get(query)
        rows.append({"prompt": query, "pass": bool(jp_pass and flu == 2 and rel == 2)})

    return rows


def map_concept(rating) -> int | None:
    """1-5 concept rating -> 0-2 (1,2,3 -> 0, 4 -> 1, 5 -> 2); None if out of range."""
    if not isinstance(rating, int) or not 1 <= rating <= 5:
        return None
    return max(rating - 3, 0)


def zero_to_two(rating) -> int | None:
    return rating if isinstance(rating, int) and 0 <= rating <= 2 else None


def harmonic_mean(*scores: int) -> float:
    if any(s == 0 for s in scores):
        return 0.0
    return len(scores) / sum(1 / s for s in scores)


def harmonic_components(rec: dict, flu, rel) -> tuple[int | None, int | None, int | None]:
    """0-2 (concept, fluency, relevance) for one judged prompt; None where a rating is unusable."""
    concept_rating = rec.get("judge_rating")
    response = rec.get("post-intervention-response", "")
    judge_prompt = str(rec.get("judge_prompt", ""))
    zero_to_two_concept = "[Steered Response Start]" in judge_prompt or "[Concept Start]" in judge_prompt
    if not isinstance(response, str) or response.strip() == "":
        concept_rating = 0 if zero_to_two_concept else 1
    concept = zero_to_two(concept_rating) if zero_to_two_concept else map_concept(concept_rating)
    return concept, zero_to_two(flu), zero_to_two(rel)


def condition_harmonic_rows(cond_dir: Path) -> list[dict] | None:
    """Per-prompt harmonic score for one condition dir, or None if unjudged."""
    judge_recs = read_jsonl(cond_dir / "judge_ratings.jsonl")
    if not judge_recs:
        return None

    flu_by_query = {
        r.get("data_path_query"): r.get("judge_rating")
        for r in read_jsonl(cond_dir / "fluency_ratings.jsonl")
    }
    rel_by_query = {
        r.get("data_path_query"): r.get("judge_rating")
        for r in read_jsonl(cond_dir / "relevance_ratings.jsonl")
    }

    rows = []
    for rec in judge_recs:
        query = rec.get("data_path_query")
        if query not in flu_by_query or query not in rel_by_query:
            continue
        scores = harmonic_components(rec, flu_by_query[query], rel_by_query[query])
        c, f, r = (0 if x is None else x for x in scores)
        rows.append({"prompt": query, "score": harmonic_mean(c, f, r)})

    return rows or None


def scan_conditions(method_root: Path, test_file: str | None = None,
                    metric: str = "pass") -> list[dict]:
    """Collect every judged condition under one steering method.

    `test_file` restricts to a single split (e.g. `harmless-valid` for the
    validation sweep, `harmless-test` for the true test run) so the two
    are never mixed into one selection.
    """
    if not method_root.is_dir():
        return []

    conditions = []
    for cond_dir in sorted(p for p in method_root.iterdir() if p.is_dir()):
        meta = parse_condition_dir(cond_dir.name)
        if meta is None:
            continue
        # Skip stale pre-layer-sweep condition dirs (legacy "topk=" naming); only
        # the current "layer=" sweep naming is considered.
        if meta["axis"] != "layer":
            continue
        if test_file is not None and meta["test_file"] != test_file:
            continue
        if metric == "harmonic":
            rows, score_key = condition_harmonic_rows(cond_dir), "score"
        else:
            rows, score_key = condition_pass_rows(cond_dir), "pass"
        if rows is None:
            continue
        conditions.append({
            **meta,
            "dir": cond_dir,
            "rows": rows,
            "rate": sum(r[score_key] for r in rows) / len(rows),
            "n": len(rows),
        })
    return conditions


def select_best(conditions: list[dict]) -> dict | None:
    """Pick the highest-scoring condition.

    Ties are broken toward the smallest N, then the layer closest to the middle
    of the swept range: among configs that score identically on validation, the
    weakest intervention is the more conservative pick, and mid-stack layers are
    where the sweep is centred.
    """
    if not conditions:
        return None

    numeric_layers = [
        float(c["value"]) for c in conditions if c["value"] != "all"
    ]
    mid_layer = statistics.median(numeric_layers) if numeric_layers else 0.0

    def sort_key(c):
        layer = float(c["value"]) if c["value"] != "all" else mid_layer
        return (-c["rate"], c["N"], abs(layer - mid_layer), layer)

    return sorted(conditions, key=sort_key)[0]


def condition_label(cond: dict) -> str:
    n_str = f"{cond['N']:g}"
    return f"N={n_str}_{cond['axis']}={cond['value']}"
