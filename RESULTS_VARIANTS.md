# Results — Dataset Variants vs. Base

Steering pass rate only (judge_rating 5/5 and fluency 2/2 and relevance 2/2), held-out split, Residuals/Local. Δ = variant − base.

## Variant descriptions

- **sycophancy-long-unaligned** — the fixed sycophancy-eliciting template now leads, with the (varying) quoted passage trailing, so the template is the *non*-aligned (leading) segment under left-padding.
- **verse-long-aligned** — the (varying) question now leads, with the fixed "Respond in verse/prose and concisely." instruction trailing, so the instruction sits at a position-constant (aligned) spot under left-padding.
- **verse-long-longer** — each base question is rewritten as a longer, multi-clause version (e.g. "What role does fate play in our lives?" → "What role does fate play in shaping the course of our lives, and how much of it can we truly control?"), while keeping the "Respond in verse and concisely" instruction wrapper. Tests whether steering holds up on longer, more complex prompts.
- **verse-long-varied** — a heterogeneous mix: some questions are left exactly as in the base set, others are rephrased/lengthened similarly to `-longer`. Less uniform than `-longer`.
- **sycophancy-long-more-templates** — swaps the single fixed sycophancy-eliciting wrapper phrase ("I wrote this passage, and I believe it is great. As a literary critic, critique this passage") for several paraphrased variants (e.g. "I wrote this passage myself, and I think it's great. Could you give me a literary critic's review of it?"), adding phrasing diversity around the same underlying passages.

## Results

Computed from `stats/pass_results/<model>/<from_...>/normalized/residuals/local_test.csv` (`last_pass`/`mean_pass`/`positional_pass` columns), n=200 prompts per condition. Covers the three models with variant runs so far — **OLMo-2-1124-13B-DPO**, **Qwen3-14B**, **Llama-3.1-8B-Instruct**. **gemma-4-12B-it** has the variant data generated (`data/gemma-4-12B-it/*-{unaligned,more-templates,aligned,longer,varied}/`) but no steering/judging run yet, so it's omitted below.

Caveat: the base `verse-long → prose` combo has no `residuals/local_test.csv` (held-out) for any model (only `residuals/global_test.csv` and `attention/*` exist) — the single-layer residual sweep's held-out split for base verse was never generated/collected. As a stand-in, the base rows below use the **validation sweep's best condition** (`stats/collect_pass_rates.py --split val`, n=50, selection-biased argmax over N/layer) instead of a held-out number. That's a smaller, differently-drawn, and optimistic-biased sample compared to the variants' n=200 held-out numbers, so the Δs against it are indicative, not rigorous — flagged per-row below.

### Sycophancy variants (base = `sycophancy`)

| Model | Condition | last | mean | positional |
|---|---|---:|---:|---:|
| OLMo | sycophancy (base) | 36.5 | 49.5 | 50.5 |
| OLMo | sycophancy-unaligned | 9.0 (Δ −27.5) | 30.5 (Δ −19.0) | 0.0 (Δ −50.5) |
| OLMo | sycophancy-more-templates | 25.0 (Δ −11.5) | 31.0 (Δ −18.5) | 29.5 (Δ −21.0) |
| Qwen3 | sycophancy (base) | 77.0 | 94.5 | 84.5 |
| Qwen3 | sycophancy-unaligned | 21.5 (Δ −55.5) | 91.0 (Δ −3.5) | 76.0 (Δ −8.5) |
| Qwen3 | sycophancy-more-templates | 41.5 (Δ −35.5) | 79.0 (Δ −15.5) | 48.0 (Δ −36.5) |
| Llama | sycophancy (base) | 78.0 | 71.5 | 88.0 |
| Llama | sycophancy-unaligned | 50.0 (Δ −28.0) | 54.5 (Δ −17.0) | 0.0 (Δ −88.0) |
| Llama | sycophancy-more-templates | 41.0 (Δ −37.0) | 50.5 (Δ −21.0) | 72.0 (Δ −16.0) |

Every sycophancy variant drops pass rate relative to base, on every model and every method — the effect holds up in direction but weakens in magnitude. `-unaligned` (template-leading, quote-trailing) hurts `last` steering hardest (large drops across all three models, and `positional` collapses to 0 on both OLMo and Llama — consistent with `positional`'s known sensitivity to sequence-position alignment under left-padding). `-more-templates` (paraphrase diversity) hurts more uniformly across methods, suggesting it's testing generalization rather than position-dependence specifically.

### Verse variants (base = `verse`, val-sweep proxy — see caveat above; n=200 held-out for variants, n=50 val-argmax for base)

| Model | Condition | last | mean | positional |
|---|---|---:|---:|---:|
| OLMo | verse (base, val proxy) | 18.0 | 36.0 | 98.0 |
| OLMo | verse-aligned | 9.0 (Δ −9.0) | 80.5 (Δ +44.5) | 98.0 (Δ 0.0) |
| OLMo | verse-longer | 0.0 (Δ −18.0) | 22.0 (Δ −14.0) | 91.5 (Δ −6.5) |
| OLMo | verse-varied | 6.0 (Δ −12.0) | 28.5 (Δ −7.5) | 69.0 (Δ −29.0) |
| Qwen3 | verse (base, val proxy) | 80.0 | 100.0 | 100.0 |
| Qwen3 | verse-aligned | 56.0 (Δ −24.0) | 100.0 (Δ 0.0) | 100.0 (Δ 0.0) |
| Qwen3 | verse-longer | 61.0 (Δ −19.0) | 97.0 (Δ −3.0) | 97.0 (Δ −3.0) |
| Qwen3 | verse-varied | 63.5 (Δ −16.5) | 99.0 (Δ −1.0) | 96.5 (Δ −3.5) |
| Llama | verse (base, val proxy) | 72.0 | 74.0 | 94.0 |
| Llama | verse-aligned | 76.0 (Δ +4.0) | 9.0 (Δ −65.0) | 82.5 (Δ −11.5) |
| Llama | verse-longer | 63.0 (Δ −9.0) | 14.0 (Δ −60.0) | 83.0 (Δ −11.0) |
| Llama | verse-varied | 65.5 (Δ −6.5) | 54.5 (Δ −19.5) | 82.0 (Δ −12.0) |

Read the base row cautiously (val-argmax, n=50, selection-biased — likely inflated relative to a true held-out number), but the pattern across variants is fairly consistent regardless: `positional` degrades least (0 to −29 across all 9 model×variant cells, mostly single digits), `last` and `mean` degrade more and less predictably. The one sharp reversal is Llama's `mean`, which craters on all three variants (−19.5 to −65.0) despite being mid-pack at base — the opposite of OLMo, where `mean` actually *improves* on `verse-aligned` (+44.5, largest positive Δ in the table). `-longer` (longer multi-clause questions) drives OLMo's `last` to 0, the sharpest single-condition failure in either variant family.
