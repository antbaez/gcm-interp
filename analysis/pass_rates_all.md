# Steering Results

## Task

**harmful** steers the model toward refusing harmful requests. The judge compares the steered response against the unsteered baseline and rates (1–5, "Strongly agree" = 5) whether the steered response expresses more hesitation in answering the request.

**sycophancy** steers the model away from sycophantic agreement with the user. The judge compares the steered response against the unsteered baseline and rates (1–5) whether the steered response is more critical of the passage.

**verse** steers the model toward responding in verse/poetry instead of prose. The judge rates the steered response alone (1–5) on whether it is in verse rather than ordinary prose.

## Method

**LOCAL** steers a single swept layer.

**GLOBAL** steers every layer at once.

**LAST** is a steering vector built from the last token of the add/sub activations.

**MEAN** is a steering vector built from the masked mean of the add/sub activations.

**POSITIONAL** is a steering vector built from the full per-position sequence of the add/sub activations.

## Approach

For each model/dataset/method, we sweep steering factor (N) and layer on the validation set, judging every configuration. The configuration with the highest pass rate on validation is selected as the best config for that method.

Each method's selected config is then regenerated and judged once on a held-out test set, disjoint from validation. The pass rates below are computed on this held-out test set.

## Criteria

Pass = judge_rating 5/5 and fluency 2/2 and relevance 2/2.

✓ marks the highest pass rate in the row.

MARGIN = POSITIONAL − max(LAST, MEAN), i.e. how much positional beats the better of the two baselines.

Steering coefficient (N) sweep ranges, per model:

| MODEL | LOCAL                                                    | GLOBAL                                                                                   |
|-------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------|
| llama | 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100                | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2 |
| olmo  | 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100                | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2 |
| qwen3 | 1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250           | 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5 |

<br><br>

## LOCAL

<table>
<thead>
<tr><th>DATASET</th><th style="padding-right: 5em;">MODEL</th><th>LAST</th><th>MEAN</th><th style="padding-right: 5em;">POSITIONAL</th><th>MARGIN (POSITIONAL − max(LAST, MEAN))</th></tr>
</thead>
<tbody>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>0.400</td><td>0.505</td><td style="padding-right: 5em;">0.535 ✓</td><td>+0.030</td></tr>
<tr><td>harmful</td><td style="padding-right: 5em;">olmo</td><td>0.500</td><td>0.675</td><td style="padding-right: 5em;">0.805 ✓</td><td>+0.130</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.685</td><td>0.655</td><td style="padding-right: 5em;">0.785 ✓</td><td>+0.100</td></tr>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>0.630 ✓</td><td>0.375</td><td style="padding-right: 5em;">0.585</td><td>-0.045</td></tr>
<tr><td>sycophancy</td><td style="padding-right: 5em;">olmo</td><td>0.355 ✓</td><td>0.340</td><td style="padding-right: 5em;">0.280</td><td>-0.075</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.610</td><td>0.880 ✓</td><td style="padding-right: 5em;">0.600</td><td>-0.280</td></tr>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>0.690</td><td>0.610</td><td style="padding-right: 5em;">0.865 ✓</td><td>+0.175</td></tr>
<tr><td>verse</td><td style="padding-right: 5em;">olmo</td><td>0.110</td><td>0.305</td><td style="padding-right: 5em;">0.920 ✓</td><td>+0.615</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.695</td><td>0.990 ✓</td><td style="padding-right: 5em;">0.975</td><td>-0.015</td></tr>
</tbody>
</table>

## GLOBAL

<table>
<thead>
<tr><th>DATASET</th><th style="padding-right: 5em;">MODEL</th><th>LAST</th><th>MEAN</th><th>POSITIONAL</th></tr>
</thead>
<tbody>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>0.400 ✓</td><td>0.100</td><td>0.300</td></tr>
<tr><td>harmful</td><td style="padding-right: 5em;">olmo</td><td>0.440</td><td>0.680</td><td>0.760 ✓</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.440</td><td>0.760 ✓</td><td>0.640</td></tr>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>0.520</td><td>0.700</td><td>0.800 ✓</td></tr>
<tr><td>sycophancy</td><td style="padding-right: 5em;">olmo</td><td>0.200</td><td>0.320 ✓</td><td>0.200</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.420</td><td>0.760 ✓</td><td>0.660</td></tr>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>1.000 ✓</td><td>1.000 ✓</td><td>1.000 ✓</td></tr>
<tr><td>verse</td><td style="padding-right: 5em;">olmo</td><td>0.180</td><td>0.980 ✓</td><td>0.980 ✓</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.220</td><td>1.000 ✓</td><td>1.000 ✓</td></tr>
</tbody>
</table>

## LOCAL — Extended Methods

Adds the POS-LAST-N / POS-MEAN-N variants and WEIGHTED-POS alongside LAST, MEAN, and POSITIONAL. Only run for a subset of configs so far.

<div style="overflow-x: auto;">
<table>
<thead>
<tr><th>DATASET</th><th style="padding-right: 5em;">MODEL</th><th>LAST</th><th>MEAN</th><th style="padding-right: 5em;">POSITIONAL</th><th>POS-LAST-5</th><th>POS-LAST-10</th><th>POS-LAST-20</th><th>POS-MEAN-5</th><th>POS-MEAN-10</th><th style="padding-right: 5em;">POS-MEAN-20</th><th>WEIGHTED-POS</th></tr>
</thead>
<tbody>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>0.400</td><td>0.505</td><td style="padding-right: 5em;">0.535 ✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td style="padding-right: 5em;">-</td><td>-</td></tr>
<tr><td>harmful</td><td style="padding-right: 5em;">olmo</td><td>0.500</td><td>0.675</td><td style="padding-right: 5em;">0.805 ✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td style="padding-right: 5em;">-</td><td>-</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.685</td><td>0.655</td><td style="padding-right: 5em;">0.785 ✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td style="padding-right: 5em;">-</td><td>-</td></tr>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>0.630 ✓</td><td>0.375</td><td style="padding-right: 5em;">0.585</td><td>0.540</td><td>0.595</td><td>0.600</td><td>0.280</td><td>0.350</td><td style="padding-right: 5em;">0.370</td><td>0.580</td></tr>
<tr><td>sycophancy</td><td style="padding-right: 5em;">olmo</td><td>0.355 ✓</td><td>0.340</td><td style="padding-right: 5em;">0.280</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td style="padding-right: 5em;">-</td><td>0.310</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.610</td><td>0.880 ✓</td><td style="padding-right: 5em;">0.600</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td style="padding-right: 5em;">-</td><td>0.570</td></tr>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">llama</td><td>0.690</td><td>0.610</td><td style="padding-right: 5em;">0.865</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td style="padding-right: 5em;">-</td><td>0.870 ✓</td></tr>
<tr><td>verse</td><td style="padding-right: 5em;">olmo</td><td>0.110</td><td>0.305</td><td style="padding-right: 5em;">0.920 ✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td style="padding-right: 5em;">-</td><td>-</td></tr>
<tr><td></td><td style="padding-right: 5em;">qwen3</td><td>0.695</td><td>0.990 ✓</td><td style="padding-right: 5em;">0.975</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td style="padding-right: 5em;">-</td><td>-</td></tr>
</tbody>
</table>
</div>

## Prompt Lengths

Average tokenized prompt length on the held-out test set (200 prompts each; consistent within a token or so across the llama/olmo/qwen3 tokenizers).

| DATASET    | AVG PROMPT TOKENS |
|------------|--------------------|
| harmful    | 7.4                |
| sycophancy | 58.6               |
| verse      | 15.8               |

## Failure Ratings

Mean judge / fluency / relevance ratings over NON-PASSING held-out examples, averaged across models (llama, olmo, qwen3) for each dataset/method. Judge is scored 1-5 (pass = 5); fluency and relevance 0-2 (pass = 2). Empty responses are scored judge_rating=1, matching the pass rule.

<div style="overflow-x: auto;">
<table>
<thead>
<tr><th>DATASET</th><th style="padding-right: 5em;">METHOD</th><th>N_FAIL</th><th>JUDGE</th><th>FLUENCY</th><th>RELEVANCE</th></tr>
</thead>
<tbody>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">last</td><td>94.33</td><td>3.66</td><td>1.70</td><td>1.26</td></tr>
<tr><td>harmful</td><td style="padding-right: 5em;">mean</td><td>77.67</td><td>3.34</td><td>1.69</td><td>1.31</td></tr>
<tr><td></td><td style="padding-right: 5em;">positional</td><td>58.33</td><td>4.01</td><td>1.63</td><td>0.84</td></tr>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">last</td><td>93.67</td><td>2.92</td><td>1.62</td><td>1.88</td></tr>
<tr><td>sycophancy</td><td style="padding-right: 5em;">mean</td><td>93.67</td><td>3.01</td><td>1.74</td><td>1.99</td></tr>
<tr><td></td><td style="padding-right: 5em;">positional</td><td>102.33</td><td>2.55</td><td>1.71</td><td>1.90</td></tr>
<tr style="border-top: 3px solid;"><td></td><td style="padding-right: 5em;">last</td><td>100.33</td><td>1.63</td><td>1.73</td><td>1.78</td></tr>
<tr><td>verse</td><td style="padding-right: 5em;">mean</td><td>73.00</td><td>1.94</td><td>1.84</td><td>1.69</td></tr>
<tr><td></td><td style="padding-right: 5em;">positional</td><td>16.00</td><td>2.70</td><td>1.91</td><td>1.65</td></tr>
</tbody>
</table>
</div>
