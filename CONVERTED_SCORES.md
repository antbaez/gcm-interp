# Converted judge scores: arithmetic vs. harmonic mean

Default judge prompts, residuals / local, validation sweep. Concept ratings are converted from 1-5 to 0-2 (1, 2, 3 → 0; 4 → 1; 5 → 2); fluency and relevance are already 0-2. Per prompt, the three ratings are combined with an arithmetic or harmonic mean (any 0 zeroes the harmonic mean), averaged over the 50 validation prompts, and the max across every N and layer tested is reported. Bold marks the best method per row.

## Arithmetic mean (residual stream, local, max over validation set)

| dataset / model | last | mean-padding | positional |
|---|---|---|---|
| harmful / olmo | 1.780 | 1.900 | **1.960** |
| harmful / qwen3 | 1.920 | 1.893 | **1.947** |
| harmful / llama | 1.673 | 1.660 | **1.753** |
| sycophancy / olmo | 1.753 | **1.807** | 1.793 |
| sycophancy / qwen3 | 1.893 | **1.993** | 1.960 |
| sycophancy / llama | 1.913 | 1.907 | **1.967** |
| verse / olmo | 1.360 | 1.580 | **1.987** |
| verse / qwen3 | 1.867 | **2.000** | **2.000** |
| verse / llama | 1.833 | 1.813 | **1.980** |

## Harmonic mean (residual stream, local, max over validation set)

| dataset / model | last | mean-padding | positional |
|---|---|---|---|
| harmful / olmo | 1.504 | 1.730 | **1.880** |
| harmful / qwen3 | 1.844 | 1.824 | **1.880** |
| harmful / llama | 1.270 | 1.240 | **1.398** |
| sycophancy / olmo | 1.510 | **1.670** | 1.650 |
| sycophancy / qwen3 | 1.800 | **1.990** | 1.930 |
| sycophancy / llama | 1.850 | 1.840 | **1.950** |
| verse / olmo | 0.420 | 0.750 | **1.960** |
| verse / qwen3 | 1.600 | **2.000** | **2.000** |
| verse / llama | 1.560 | 1.720 | **1.970** |
