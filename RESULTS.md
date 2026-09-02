# Results

Generated via `analysis/analyze_results.py` across all four `--global` × `--attention` combinations. Full per-combo output (qualitative examples, pass rates, MMLU deltas, failure ratings) is written to `analysis/<residuals|attention>/<local|global>/`.

LEFT TABLE: pass rate per steering method (✓ = best method per row). Pass = judge_rating 5/5 and fluency 2/2 and relevance 2/2.

RIGHT TABLE: MMLU accuracy change from unsteered baseline per steering method (✓ = least degradation per row).

## Residuals / Local

<table><tr><td>

| DATASET | MODEL | LAST | MEAN | POSITIONAL |
|---|---|---|---|---|
| harmful | llama | 0.40 | 0.51 | **0.54** ✓ |
| harmful | olmo | 0.50 | 0.68 | **0.81** ✓ |
| harmful | qwen3 | 0.69 | 0.66 | **0.79** ✓ |
| sycophancy | llama | 0.78 | 0.72 | **0.88** ✓ |
| sycophancy | olmo | 0.37 | 0.50 | **0.51** ✓ |
| sycophancy | qwen3 | 0.77 | **0.95** ✓ | 0.85 |
| verse | llama | 0.69 | 0.61 | **0.87** ✓ |
| verse | olmo | 0.11 | 0.31 | **0.92** ✓ |
| verse | qwen3 | 0.70 | **0.99** ✓ | 0.98 |

</td><td>

| DATASET | MODEL | LAST | MEAN | POSITIONAL |
|---|---|---|---|---|
| harmful | llama | -0.35 | -0.30 | **-0.25** ✓ |
| harmful | olmo | -0.23 | -0.08 | **-0.01** ✓ |
| harmful | qwen3 | -0.49 | -0.12 | **+0.01** ✓ |
| sycophancy | llama | -0.34 | **-0.17** ✓ | -0.30 |
| sycophancy | olmo | -0.18 | **-0.05** ✓ | -0.33 |
| sycophancy | qwen3 | -0.50 | **-0.43** ✓ | -0.46 |
| verse | llama | -0.35 | -0.31 | **-0.27** ✓ |
| verse | olmo | -0.36 | -0.12 | **-0.02** ✓ |
| verse | qwen3 | -0.22 | **-0.05** ✓ | -0.09 |

</td></tr></table>

---

## Residuals / Global

<table><tr><td>

| DATASET | MODEL | LAST | MEAN | POSITIONAL |
|---|---|---|---|---|
| harmful | llama | **0.28** ✓ | 0.13 | 0.18 |
| harmful | olmo | 0.43 | 0.60 | **0.67** ✓ |
| harmful | qwen3 | 0.36 | **0.63** ✓ | 0.50 |
| sycophancy | llama | 0.73 | 0.82 | **0.85** ✓ |
| sycophancy | olmo | **0.51** ✓ | 0.40 | 0.36 |
| sycophancy | qwen3 | 0.50 | **0.75** ✓ | 0.74 |
| verse | llama | **1.00** ✓ | **1.00** ✓ | 1.00 |
| verse | olmo | 0.15 | 0.92 | **0.98** ✓ |
| verse | qwen3 | 0.15 | **0.99** ✓ | **0.99** ✓ |

</td><td>

| DATASET | MODEL | LAST | MEAN | POSITIONAL |
|---|---|---|---|---|
| harmful | llama | -0.17 | -0.16 | **-0.03** ✓ |
| harmful | olmo | -0.36 | -0.19 | **+0.00** ✓ |
| harmful | qwen3 | -0.18 | -0.11 | **-0.01** ✓ |
| sycophancy | llama | -0.15 | **-0.01** ✓ | -0.07 |
| sycophancy | olmo | -0.17 | -0.19 | **-0.10** ✓ |
| sycophancy | qwen3 | -0.10 | -0.06 | **-0.02** ✓ |
| verse | llama | -0.26 | -0.02 | **-0.01** ✓ |
| verse | olmo | -0.07 | -0.13 | **-0.02** ✓ |
| verse | qwen3 | -0.10 | **+0.00** ✓ | +0.00 |

</td></tr></table>

---

## Attention / Local

<table><tr><td>

| DATASET | MODEL | LAST | MEAN | POSITIONAL |
|---|---|---|---|---|
| harmful | llama | 0.21 | 0.21 | **0.33** ✓ |
| harmful | olmo | 0.20 | **0.28** ✓ | 0.24 |
| harmful | qwen3 | 0.59 | 0.48 | **0.74** ✓ |
| sycophancy | llama | 0.72 | 0.69 | **0.76** ✓ |
| sycophancy | olmo | 0.18 | **0.23** ✓ | 0.13 |
| sycophancy | qwen3 | 0.87 | **0.93** ✓ | 0.79 |
| verse | llama | 0.06 | **0.33** ✓ | 0.20 |
| verse | olmo | 0.01 | **0.02** ✓ | 0.01 |
| verse | qwen3 | **0.22** ✓ | 0.13 | 0.09 |

</td><td>

| DATASET | MODEL | LAST | MEAN | POSITIONAL |
|---|---|---|---|---|
| harmful | llama | -0.36 | **-0.28** ✓ | -0.31 |
| harmful | olmo | **-0.08** ✓ | -0.09 | -0.13 |
| harmful | qwen3 | **-0.50** ✓ | **-0.50** ✓ | **-0.50** ✓ |
| sycophancy | llama | -0.30 | **-0.03** ✓ | -0.07 |
| sycophancy | olmo | -0.05 | -0.06 | **-0.00** ✓ |
| sycophancy | qwen3 | -0.50 | **-0.21** ✓ | -0.49 |
| verse | llama | -0.36 | -0.36 | **-0.35** ✓ |
| verse | olmo | -0.03 | -0.03 | **-0.00** ✓ |
| verse | qwen3 | **-0.11** ✓ | -0.30 | -0.50 |

</td></tr></table>

---

## Attention / Global

Steering coefficient N is chosen per steering method (the argmax-validation config `select_best_config.py` pins for the held-out split); it need not match across methods on the same row.

<table><tr><td>

| DATASET | MODEL | N (LAST/MEAN/POS) | LAST | MEAN | POSITIONAL |
|---|---|---|---|---|---|
| harmful | llama | 1 / 1 / 1 | 0.00 | 0.01 | **0.24** ✓ |
| harmful | olmo | 10 / 10 / 9 | 0.72 | 0.76 | **0.77** ✓ |
| harmful | qwen3 | 6 / 4 / 5 | **0.33** ✓ | 0.33 | 0.25 |
| sycophancy | llama | 1 / 1 / 1 | 0.00 | 0.00 | **0.34** ✓ |
| sycophancy | olmo | 7 / 8 / 10 | 0.16 | **0.79** ✓ | 0.20 |
| sycophancy | qwen3 | 4 / 4 / 8 | 0.49 | **0.74** ✓ | 0.60 |
| verse | llama | 1 / 1 / 1 | **0.00** ✓ | **0.00** ✓ | **0.00** ✓ |
| verse | olmo | 10 / 10 / 8 | 0.65 | 0.81 | **0.89** ✓ |
| verse | qwen3 | 5 / 3 / 3 | 0.81 | **1.00** ✓ | 0.99 |

</td><td>

| DATASET | MODEL | LAST | MEAN | POSITIONAL |
|---|---|---|---|---|
| harmful | llama | -0.32 | -0.34 | **-0.03** ✓ |
| harmful | olmo | -0.32 | -0.33 | **-0.29** ✓ |
| harmful | qwen3 | -0.48 | -0.29 | **+0.00** ✓ |
| sycophancy | llama | -0.33 | -0.36 | **-0.11** ✓ |
| sycophancy | olmo | -0.08 | -0.12 | **-0.07** ✓ |
| sycophancy | qwen3 | -0.21 | -0.34 | **-0.05** ✓ |
| verse | llama | -0.36 | -0.35 | **-0.35** ✓ |
| verse | olmo | -0.26 | -0.26 | **-0.07** ✓ |
| verse | qwen3 | -0.22 | **-0.00** ✓ | +0.02 |

</td></tr></table>

---

## Notes

- `positional` wins the pass-rate row-best count most often across all four combos, but the margin varies a lot by stream/scope — `attention` steering generally yields lower absolute pass rates than `residuals`, and `global` scope is noisier/more extreme (harder failures like 0.00 for llama/verse under attention+global).
- Full qualitative pass/fail examples and per-combo failure-rating breakdowns live under `analysis/<stream>/<scope>/`.

---

## Best stream/scope/method per dataset/model

For each dataset/model, the single (method, stream/scope) combo with the highest held-out pass rate across all 12 candidates (3 methods × 4 stream×scope combos).

| DATASET | MODEL | BEST METHOD | BEST STREAM/SCOPE | PASS RATE |
|---|---|---|---|---|
| harmful | llama | positional | Residuals/Local | 0.54 |
| harmful | olmo | positional | Residuals/Local | 0.81 |
| harmful | qwen3 | positional | Residuals/Local | 0.79 |
| sycophancy | llama | positional | Residuals/Local | 0.88 |
| sycophancy | olmo | mean | Attention/Global | 0.79 |
| sycophancy | qwen3 | mean | Residuals/Local | 0.95 |
| verse | llama | last / mean (tied) | Residuals/Global | 1.00 |
| verse | olmo | positional | Residuals/Global | 0.98 |
| verse | qwen3 | mean | Attention/Global | 1.00 |

---

## Average MMLU change by intervention type

Averaged across all models, datasets, and steering methods (27 values per combo).

| INTERVENTION TYPE | AVG MMLU CHANGE |
|---|---|
| Residuals / Local | -0.24 |
| Residuals / Global | -0.10 |
| Attention / Local | -0.24 |
| Attention / Global | -0.22 |
