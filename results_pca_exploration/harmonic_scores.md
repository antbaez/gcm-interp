# Harmonic scores (held-out)

## Average over methods

| model | dataset | scope | harmonic | concept | fluency | relevance | methods |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | verse | global | 1.816 | 2.000 | 1.820 | 1.845 | last, mean |
| Llama-3.1-8B-Instruct | verse | local | 0.623 | 0.682 | 1.597 | 1.900 | last, mean |

## Per method

| model | dataset | scope | method | harmonic | concept | fluency | relevance | n | N | layer |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | verse | global | last | 1.948 | 2.000 | 1.905 | 2.000 | 200 | 0.5 | all |
| Llama-3.1-8B-Instruct | verse | global | mean | 1.684 | 2.000 | 1.735 | 1.690 | 200 | 0.5 | all |
| Llama-3.1-8B-Instruct | verse | local | last | 0.316 | 0.440 | 1.430 | 1.815 | 200 | 200.0 | 21 |
| Llama-3.1-8B-Instruct | verse | local | mean | 0.930 | 0.925 | 1.765 | 1.985 | 200 | 125.0 | 21 |
