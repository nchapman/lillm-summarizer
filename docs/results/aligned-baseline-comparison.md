# Aligned results: six local models without adapters

## Summary

This is the current comparison of the six local, unadapted models. It uses the **same 250 CNN/DailyMail articles** as the aligned adapter evaluation, with the `summary-v3` prompt, a 1,024-token output ceiling, greedy decoding, and one pinned judge. The Qwen and Granite base predictions were reused from the aligned adapter runs; MiniCPM5 and LFM2.5 were newly generated for this comparison. All six candidate sets were judged under the same judge model and rubric.

The old seven-candidate table is retained as a historical record in [baseline confirmation results](baseline-confirmation.md). That run used `summary-v1`, a 256-token ceiling, model-specific sampling, and a mutable judge alias. Do not combine its numbers with this comparison.

Granite 4.0-1B leads this local cohort on the three ROUGE measures and on mean factuality and overall-quality ratings. These are descriptive results on one 250-article sample, not a universal model ranking. The adapter study is reported separately in the [aligned adapter comparison](aligned-adapter-comparison.md); its four base rows are repeated here.

## Protocol

| Setting | Value |
| --- | --- |
| Dataset | CNN/DailyMail 3.0.0 test split; frozen 250-example sample |
| Dataset SHA-256 | `047f62c289d3eb08ce3dd5f3096736cf40f0414b1f602a052ee8e79a1377e75e` |
| Prompt | `summary-v3` |
| Output ceiling | 1,024 new tokens; this is a safety cap, not a target length |
| Local decoding | Temperature 0 (greedy), seed 42 |
| Local runtime | llama.cpp commit `0878f7c304cc13915c831ae17694534c94940176`; 8,192-token context |
| Judge | `deepseek/deepseek-v4.1-flash-1017`; integer ratings from 1 to 5 |
| Uncertainty | 5,000 percentile-bootstrap resamples of article IDs; seed `20261029`; pointwise 95% intervals, not adjusted for multiple comparisons |

The Qwen, MiniCPM5, and LFM2.5 requests explicitly supplied `top_p=1`, `top_k=0`, `min_p=0`, neutral presence penalty, and repeat penalty 1. The reused Granite requests omitted `min_p` and used llama.cpp's default 0.05; they also relied on the runtime's neutral penalty defaults. Temperature 0 makes selection greedy: those cutoff filters do not change the top-token choice. Thus the effective decoding is aligned, although the recorded optional sampler flags are not byte-for-byte identical. GGUF quantization remains model-specific, as listed below.

The judge saw each article and candidate summary, not the reference summary. ROUGE measures reference overlap, not factuality. Unsupported-claim notes are counts of notes returned by the judge, not a severity-weighted error rate. Latency is recorded per-example generation time on this host and is descriptive, not a cross-provider throughput benchmark.

## Results

Means are followed by pointwise 95% bootstrap confidence intervals. Judge ratings use a 1–5 scale. `Length stops` counts outputs where the server reported `finish_reason=length`.

| Local base model | Quantization | ROUGE-1 | ROUGE-2 | ROUGE-L | Factuality | Overall quality | Unsupported-claim notes | Mean latency | Length stops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | UD-Q4_K_XL | 0.220 [0.209, 0.231] | 0.086 [0.080, 0.093] | 0.140 [0.133, 0.147] | 3.30 [3.17, 3.42] | 3.18 [3.07, 3.29] | 595 | 1.688 s | 10 |
| Qwen3.5-2B | UD-Q4_K_XL | 0.217 [0.208, 0.227] | 0.086 [0.080, 0.093] | 0.137 [0.131, 0.144] | 3.58 [3.46, 3.68] | 3.55 [3.44, 3.65] | 555 | 3.182 s | 3 |
| MiniCPM5-1B | Q4_K_M | 0.199 [0.186, 0.213] | 0.074 [0.067, 0.082] | 0.130 [0.121, 0.139] | 2.99 [2.85, 3.13] | 2.54 [2.40, 2.68] | 635 | 1.959 s | 58 |
| LFM2.5-1.2B | UD-Q4_K_XL | 0.293 [0.283, 0.303] | 0.091 [0.084, 0.097] | 0.178 [0.171, 0.185] | 3.66 [3.55, 3.78] | 3.52 [3.42, 3.61] | 409 | 0.821 s | 0 |
| **Granite 4.0-1B** | UD-Q4_K_XL | **0.322 [0.310, 0.334]** | **0.130 [0.122, 0.138]** | **0.206 [0.197, 0.214]** | **4.30 [4.19, 4.41]** | **4.04 [3.94, 4.12]** | **156** | 1.437 s | 0 |
| Granite 4.0-350M | UD-Q4_K_XL | 0.308 [0.295, 0.322] | 0.126 [0.117, 0.134] | 0.204 [0.193, 0.213] | 3.64 [3.48, 3.78] | 3.15 [3.02, 3.26] | 303 | **0.649 s** | 11 |

### All judge dimensions

| Local base model | Factuality | Coverage | Relevance | Concision | Readability | Overall quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 3.30 [3.17, 3.42] | 3.81 [3.73, 3.89] | 3.83 [3.75, 3.92] | 2.98 [2.88, 3.07] | 3.72 [3.62, 3.82] | 3.18 [3.07, 3.29] |
| Qwen3.5-2B | 3.58 [3.46, 3.68] | 4.15 [4.07, 4.22] | 4.12 [4.04, 4.19] | 3.23 [3.15, 3.31] | 4.20 [4.12, 4.28] | 3.55 [3.44, 3.65] |
| MiniCPM5-1B | 2.99 [2.85, 3.13] | 3.15 [3.03, 3.26] | 3.24 [3.13, 3.36] | 2.32 [2.20, 2.44] | 2.90 [2.75, 3.04] | 2.54 [2.40, 2.68] |
| LFM2.5-1.2B | 3.66 [3.55, 3.78] | 3.58 [3.49, 3.66] | 4.05 [3.97, 4.14] | 3.77 [3.70, 3.83] | 4.41 [4.34, 4.49] | 3.52 [3.42, 3.61] |
| Granite 4.0-1B | 4.30 [4.19, 4.41] | 4.05 [3.99, 4.12] | 4.55 [4.47, 4.62] | 3.94 [3.89, 3.98] | 4.69 [4.63, 4.75] | 4.04 [3.94, 4.12] |
| Granite 4.0-350M | 3.64 [3.48, 3.78] | 3.26 [3.16, 3.35] | 3.84 [3.73, 3.94] | 3.29 [3.18, 3.41] | 3.71 [3.59, 3.83] | 3.15 [3.02, 3.26] |

## Hosted GPT-6 Luna control

GPT-6 Luna is not included in the aligned local-model table. OpenRouter's control does not advertise the local sampler controls, so temperature and related filters cannot be set to the same greedy protocol. An attempted rerun used the same articles and prompt, a 1,024-token ceiling, seed 42, and `reasoning_effort=none`; after eight completed responses, one request exceeded the 600-second read timeout. The incomplete output was discarded, and no aligned score is reported. The old hosted score remains only in the historical [baseline confirmation results](baseline-confirmation.md); it used a different prompt, cap, and judge and must not be compared with these results.

## Interpretation and limitations

The six local baseline rows and four base/adapter pairs in the aligned adapter report share the same evaluation sample, prompt, effective greedy decoding, output ceiling, and pinned judge. Read them together for within-model base-versus-adapter comparisons. Adapter training data and recipes are not controlled identically across model families, so cross-family adapter differences are not causal evidence about architecture or size.

The judge is automated and uncalibrated; its ratings are not probabilities. Intervals describe variation across this sample, not judge variability or all summarization tasks. The report does not establish that response behavior has become more consistent across models; no style-variance metric was run.
