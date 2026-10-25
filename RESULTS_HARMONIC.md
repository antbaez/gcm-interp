# Harmonic scores (held-out test)

Residual-stream steering, single layer (OLMo/Qwen3: 27, Llama: 21), validation-selected N, 200 held-out prompts per combo. Harmonic score range 0-2. Bold = best method in the column (within each dataset block in the first table). Best method = the method that is best on at least 2 of the 3 models; otherwise tie.

## Positional, mean, last

| Dataset | Method | OLMo | Qwen3 | Llama | Best method |
|---|---|---|---|---|---|
| **Harmful** | positional | 0.28 | **0.46** | 0.31 | tie |
| | mean | **0.31** | 0.19 | 0.42 | |
| | last | 0.25 | 0.29 | **0.58** | |
| **Sycophancy** | positional | **1.46** | 1.30 | **1.63** | positional |
| | mean | 1.22 | **1.55** | 1.62 | |
| | last | 1.27 | 1.40 | 1.60 | |
| **Verse** | positional | **0.18** | **0.60** | 0.41 | positional |
| | mean | 0.13 | 0.10 | **0.64** | |
| | last | 0.10 | 0.50 | 0.50 | |

## All methods

| Group | Method | OLMo harm | Qwen3 harm | Llama harm | OLMo syc | Qwen3 syc | Llama syc | OLMo verse | Qwen3 verse | Llama verse | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Positional | positional | 0.28 | **0.46** | 0.31 | 1.46 | 1.30 | 1.63 | 0.18 | **0.60** | 0.41 | **0.74** |
| | unweighted-positional | 0.26 | 0.41 | 0.28 | **1.47** | 1.28 | 1.54 | **0.24** | 0.58 | 0.46 | 0.72 |
| | positional-projected | 0.23 | 0.20 | 0.36 | 1.22 | 1.42 | **1.67** | 0.03 | 0.14 | 0.45 | 0.63 |
| Mean | mean | **0.31** | 0.19 | 0.42 | 1.22 | **1.55** | 1.62 | 0.13 | 0.10 | **0.64** | 0.69 |
| Last token | last | 0.25 | 0.29 | **0.58** | 1.27 | 1.40 | 1.60 | 0.10 | 0.50 | 0.50 | 0.72 |
| | last-EOT | 0.24 | 0.42 | 0.50 | 1.27 | 1.39 | 1.49 | 0.14 | 0.06 | 0.55 | 0.67 |
| | last-prompt | 0.13 | 0.26 | 0.55 | 1.13 | 1.50 | 1.62 | 0.14 | 0.12 | 0.39 | 0.65 |
