# Aligned base/adapter comparison: four small models

## Finding

A fresh shared evaluation shows **mixed, model-dependent effects—not a universal adapter improvement**. Both Qwen adapters improved judged factuality, relevance, and ROUGE overlap, but the 2B adapter's overall-quality change was inconclusive and its coverage/readability scores fell. Granite 4.0-1B improved ROUGE and concision while its coverage declined; factuality and overall-quality changes were inconclusive. Granite 4.0-350M declined on several judged dimensions, including factuality and overall quality. All four adapters increased the number of outputs that stopped at the 1,024-token cap for their respective Qwen/Granite bases except the Granite 350M adapter, which reduced cap stops.

The adaptation goal is to strengthen source-grounded summarization and encourage a consistent response contract across models. The common prompt expresses that intent, but these results **do not establish cross-model response normalization**: output lengths still vary substantially, Qwen adapter means increased under this shared greedy protocol, and no formal style-variance metric was used.

| Model | Factuality: base → adapter | Overall quality: base → adapter | ROUGE-L: base → adapter | Mean words: base → adapter | 1,024-token stops: base → adapter |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 3.296 → **4.344** | 3.184 → **3.400** | 0.1400 → **0.1693** | 303.9 → 310.5 | 10 → 24 |
| Qwen3.5-2B | 3.576 → **4.520** | 3.548 → 3.596 | 0.1372 → **0.1599** | 310.6 → 342.4 | 3 → 29 |
| Granite 4.0-1B | 4.304 → 4.328 | 4.036 → 4.000 | 0.2056 → **0.2237** | 154.9 → 141.9 | 0 → 2 |
| Granite 4.0-350M | 3.636 → **2.756** | 3.148 → **2.736** | 0.2036 → 0.1956 | 179.4 → 163.8 | 11 → 3 |

Judge scores are ordinal means on a 1–5 scale, not percentages. ROUGE is reference overlap, not factuality. Each model is compared only with its own base; this table is not a cross-family model ranking.

## Primary results: 250 shared articles

Values are base and adapter means followed by adapter-minus-base difference and its **paired 95% percentile-bootstrap confidence interval**. Intervals are pointwise, with no multiple-comparison adjustment.

### Qwen3.5-0.8B

| Metric | Base | Base + adapter | Difference [95% paired CI] |
| --- | ---: | ---: | ---: |
| Factuality | 3.296 | 4.344 | +1.048 [+0.896, +1.204] |
| Coverage | 3.812 | 3.824 | +0.012 [-0.112, +0.136] |
| Relevance | 3.832 | 4.148 | +0.316 [+0.200, +0.432] |
| Concision | 2.976 | 3.004 | +0.028 [-0.132, +0.196] |
| Readability | 3.720 | 3.832 | +0.112 [-0.028, +0.256] |
| Overall quality | 3.184 | 3.400 | +0.216 [+0.076, +0.364] |
| ROUGE-1 | 0.2202 | 0.2593 | +0.0391 [+0.0257, +0.0520] |
| ROUGE-2 | 0.0864 | 0.1101 | +0.0236 [+0.0177, +0.0301] |
| ROUGE-L | 0.1400 | 0.1693 | +0.0292 [+0.0207, +0.0376] |
| Mean words | 303.9 | 310.5 | +6.6 [-22.9, +36.6] |
| Mean completion tokens | 386.5 | 397.8 | +11.3 [-26.6, +49.5] |

### Qwen3.5-2B

| Metric | Base | Base + adapter | Difference [95% paired CI] |
| --- | ---: | ---: | ---: |
| Factuality | 3.576 | 4.520 | +0.944 [+0.808, +1.076] |
| Coverage | 4.148 | 4.024 | -0.124 [-0.244, -0.004] |
| Relevance | 4.120 | 4.336 | +0.216 [+0.116, +0.316] |
| Concision | 3.232 | 3.080 | -0.152 [-0.320, +0.008] |
| Readability | 4.204 | 3.992 | -0.212 [-0.352, -0.080] |
| Overall quality | 3.548 | 3.596 | +0.048 [-0.096, +0.184] |
| ROUGE-1 | 0.2171 | 0.2443 | +0.0272 [+0.0138, +0.0403] |
| ROUGE-2 | 0.0863 | 0.1054 | +0.0191 [+0.0127, +0.0255] |
| ROUGE-L | 0.1372 | 0.1599 | +0.0226 [+0.0137, +0.0316] |
| Mean words | 310.6 | 342.4 | +31.9 [+3.1, +61.0] |
| Mean completion tokens | 391.7 | 435.8 | +44.1 [+8.7, +79.8] |

### Granite 4.0-1B

| Metric | Base | Base + adapter | Difference [95% paired CI] |
| --- | ---: | ---: | ---: |
| Factuality | 4.304 | 4.328 | +0.024 [-0.112, +0.156] |
| Coverage | 4.052 | 3.808 | -0.244 [-0.324, -0.168] |
| Relevance | 4.548 | 4.504 | -0.044 [-0.124, +0.036] |
| Concision | 3.936 | 4.116 | +0.180 [+0.096, +0.260] |
| Readability | 4.692 | 4.640 | -0.052 [-0.144, +0.036] |
| Overall quality | 4.036 | 4.000 | -0.036 [-0.140, +0.068] |
| ROUGE-1 | 0.3219 | 0.3454 | +0.0235 [+0.0150, +0.0323] |
| ROUGE-2 | 0.1299 | 0.1419 | +0.0120 [+0.0058, +0.0185] |
| ROUGE-L | 0.2056 | 0.2237 | +0.0181 [+0.0113, +0.0254] |
| Mean words | 154.9 | 141.9 | -12.9 [-24.3, +0.2] |
| Mean completion tokens | 195.3 | 178.5 | -16.7 [-29.7, -2.1] |

### Granite 4.0-350M

| Metric | Base | Base + adapter | Difference [95% paired CI] |
| --- | ---: | ---: | ---: |
| Factuality | 3.636 | 2.756 | -0.880 [-1.044, -0.712] |
| Coverage | 3.260 | 3.284 | +0.024 [-0.076, +0.128] |
| Relevance | 3.836 | 3.548 | -0.288 [-0.396, -0.176] |
| Concision | 3.292 | 3.244 | -0.048 [-0.176, +0.092] |
| Readability | 3.712 | 3.508 | -0.204 [-0.332, -0.064] |
| Overall quality | 3.148 | 2.736 | -0.412 [-0.548, -0.276] |
| ROUGE-1 | 0.3083 | 0.3044 | -0.0040 [-0.0140, +0.0069] |
| ROUGE-2 | 0.1257 | 0.1203 | -0.0054 [-0.0123, +0.0013] |
| ROUGE-L | 0.2036 | 0.1956 | -0.0079 [-0.0159, -0.0001] |
| Mean words | 179.4 | 163.8 | -15.6 [-37.3, +4.1] |
| Mean completion tokens | 225.4 | 211.5 | -13.9 [-40.0, +10.9] |

### Interpretation

- **Qwen3.5-0.8B:** factuality, relevance, overall quality, and all ROUGE changes have intervals above zero. Coverage, concision, readability, and length changes are inconclusive. Cap stops increased from 10 to 24.
- **Qwen3.5-2B:** factuality, relevance, and ROUGE improved; coverage and readability declined. Overall quality and concision changes are inconclusive. Cap stops increased from 3 to 29. This is not an across-the-board quality gain.
- **Granite 4.0-1B:** ROUGE and concision improved, coverage declined, and factuality/overall-quality differences are inconclusive. Cap stops increased from 0 to 2.
- **Granite 4.0-350M:** factuality, relevance, readability, and overall quality declined; overlap changes were small. Cap stops decreased from 11 to 3.

Cap counts are observable diagnostics, not a full human quality review. Automated judge ratings can be wrong, paired intervals do not account for judge or training-seed variation, and no multiple-testing adjustment is applied.

## Shared evaluation protocol

- **Evaluation data:** the same frozen 250-example CNN/DailyMail 3.0.0 test confirmation set for all eight candidates; SHA-256 `047f62c289d3eb08ce3dd5f3096736cf40f0414b1f602a052ee8e79a1377e75e`. It is public and previously exposed, not a sealed benchmark.
- **Prompt:** shared `summary-v3` system instruction and `{document}` user template; native chat templates retained. The instruction requests objective plain prose, source-dependent detail, no markdown, and no unsupported additions.
- **Generation:** llama.cpp CUDA commit `0878f7c304cc13915c831ae17694534c94940176`; Q4_K_XL base models, BF16 GGUF sidecar at scale 1 for adapter candidates, 8,192-token context, 1,024-token output ceiling, temperature 0 (greedy), and seed 42. Qwen requests explicitly used top-p 1, top-k 0, min-p 0, and neutral penalties; Granite source runs omitted min-p and used llama.cpp's 0.05 default, with neutral penalty defaults. At temperature 0 these cutoff filters do not change the greedy top-token choice. Qwen generations were rerun; Granite generations were reused after verifying prompt, sample, cap, temperature, seed, quantization, and server revision. No source truncation was observed.
- **Judge:** all 2,000 candidate summaries were judged with pinned `deepseek/deepseek-v4.1-flash-1017`, using the same rubric and six 1–5 dimensions. No reference summaries or private training records were sent to the judge. The first Granite 350M adapter judging batch returned one invalid rating after 36 valid responses; that partial batch was discarded and the full 250-example batch was rerun successfully. Only complete judgments enter the results.
- **ROUGE:** macro-average F1 with Porter stemming using `rouge-score`; ordinary ROUGE-L, not ROUGE-Lsum. Reference overlap is not a factuality measure.
- **Uncertainty:** 5,000 paired resamples of the same article IDs, percentile 95% intervals, seed `20261029`. Intervals are pointwise and unadjusted for multiple metrics.

The six local base-model results in [the aligned baseline comparison](aligned-baseline-comparison.md) use these same 250 articles, `summary-v3`, a 1,024-token ceiling, effective greedy decoding, and the pinned judge, so they can be read alongside this table. The hosted GPT-6 Luna control is excluded because its decoding is provider-managed and its attempted aligned rerun timed out; its earlier result remains historical. Training conditions are also not held constant across families: base candidates are unadapted, and adapters used family-specific training recipes combining public SmolTalk2 examples and private web-document/summary pairs. Private records, training data manifests, model outputs, and judge responses remain local and are not published.
