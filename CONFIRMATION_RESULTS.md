# CNN/DailyMail confirmation comparison

## Summary

The disjoint 250-example confirmation set supports the pilot's main local-model conclusion: **Granite 4.0-1B is the strongest local candidate** among the five Q4 models on this sample. It leads the local candidates on ROUGE-1/2/L and on all six source-grounded judge dimensions. Its paired 95% bootstrap intervals for ROUGE differences are above zero versus every other candidate, including GPT-6 Luna. GPT-6 Luna receives substantially higher judge ratings than any local model, but its ROUGE scores are below Granite's; judge scores are close to the scale ceiling and should be treated cautiously.

LFM2.5-1.2B is the local runner-up on ROUGE-1/2/L and judge quality, while MiniCPM5-1B has the lowest local mean generation latency and the third-highest local ROUGE-L. The small Qwen3.5 models received lower judge scores and ROUGE than the other local candidates. These are model-level comparisons on one random sample, not claims about all news summarization tasks.

## Protocol

- Dataset: CNN/DailyMail 3.0.0 test split; frozen, seeded sample of 250 examples, disjoint by ID and article from the frozen 50-example pilot. Sample SHA-256: `047f62c289d3eb08ce3dd5f3096736cf40f0414b1f602a052ee8e79a1377e75e`.
- All six candidates used the same examples, `summary-v1` prompt, and 256-token output cap. Local models used the model-specific Q4 files and sampling settings documented in [BASELINE_RESULTS.md](BASELINE_RESULTS.md); GPT-6 Luna used OpenRouter provider-managed decoding, seed 42, `reasoning_effort=none`, and omitted temperature.
- ROUGE-1/2/L are F1 scores from `rouge-score` with Porter stemming. ROUGE measures reference overlap, not factuality.
- The judge was `~deepseek/deepseek-flash-latest`. It saw the source and candidate summary, not the reference; dimensions are integer 1–5. Unsupported-claim counts are totals of returned notes, not a severity-weighted error rate.
- Intervals are percentile 95% bootstrap CIs from 5,000 paired resamples of the 250 example IDs (seed `20261029`). They describe sampling uncertainty on this set; they do not capture judge or model stochasticity, and comparisons are not adjusted for multiple testing.

## Results

Means are followed by 95% bootstrap confidence intervals. Judge scores are on the 1–5 scale. Latency is mean per-example generation latency from the recorded run metadata.

| Candidate | ROUGE-1 | ROUGE-2 | ROUGE-L | Judge factuality | Judge overall | Unsupported-claim notes | Mean latency | `finish_reason=length` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 0.315 [0.305, 0.325] | 0.077 [0.072, 0.083] | 0.172 [0.166, 0.179] | 1.81 [1.72, 1.91] | 1.80 [1.71, 1.90] | 1,148 | 2 |
| Qwen3.5-2B | 0.330 [0.320, 0.340] | 0.092 [0.086, 0.098] | 0.185 [0.179, 0.192] | 2.57 [2.45, 2.69] | 2.58 [2.47, 2.69] | 721 | 1 |
| MiniCPM5-1B | 0.327 [0.315, 0.339] | 0.095 [0.088, 0.104] | 0.196 [0.188, 0.204] | 2.52 [2.38, 2.66] | 2.44 [2.31, 2.56] | 617 | 2 |
| LFM2.5-1.2B | 0.344 [0.333, 0.354] | 0.101 [0.094, 0.109] | 0.211 [0.203, 0.219] | 3.98 [3.86, 4.10] | 3.89 [3.79, 3.99] | 226 | 0 |
| **Granite 4.0-1B** | **0.404 [0.392, 0.416]** | **0.156 [0.145, 0.167]** | **0.258 [0.248, 0.268]** | **4.53 [4.42, 4.63]** | **4.13 [4.04, 4.22]** | **87** | 0 |
| GPT-6 Luna (hosted) | 0.375 [0.363, 0.386] | 0.123 [0.115, 0.132] | 0.229 [0.219, 0.239] | 4.96 [4.94, 4.98] | 4.91 [4.87, 4.94] | 9 | 0 |

The full set of judge means and intervals:

| Candidate | Factuality | Coverage | Relevance | Concision | Readability | Overall quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 1.81 [1.72, 1.91] | 2.28 [2.19, 2.36] | 2.61 [2.51, 2.71] | 2.84 [2.76, 2.92] | 2.69 [2.61, 2.77] | 1.80 [1.71, 1.90] |
| Qwen3.5-2B | 2.57 [2.45, 2.69] | 2.86 [2.78, 2.94] | 3.37 [3.28, 3.46] | 3.52 [3.45, 3.60] | 3.45 [3.36, 3.53] | 2.58 [2.47, 2.69] |
| MiniCPM5-1B | 2.52 [2.38, 2.66] | 2.40 [2.30, 2.49] | 3.08 [2.97, 3.20] | 3.51 [3.42, 3.60] | 3.18 [3.08, 3.29] | 2.44 [2.31, 2.56] |
| LFM2.5-1.2B | 3.98 [3.86, 4.10] | 3.25 [3.17, 3.32] | 4.18 [4.10, 4.25] | 4.40 [4.34, 4.46] | 4.50 [4.44, 4.57] | 3.89 [3.79, 3.99] |
| Granite 4.0-1B | 4.53 [4.42, 4.63] | 3.34 [3.26, 3.41] | 4.45 [4.37, 4.52] | 4.65 [4.59, 4.71] | 4.78 [4.72, 4.83] | 4.13 [4.04, 4.22] |
| GPT-6 Luna (hosted) | 4.96 [4.94, 4.98] | 4.12 [4.08, 4.16] | 4.92 [4.89, 4.96] | 4.98 [4.97, 5.00] | 5.00 [5.00, 5.00] | 4.91 [4.87, 4.94] |

## Paired comparison

Granite's paired bootstrap difference intervals for each ROUGE metric were positive versus every other model. The mean Granite-minus-GPT-6 Luna differences were +0.029 [0.018, 0.041] ROUGE-1, +0.033 [0.022, 0.043] ROUGE-2, and +0.029 [0.019, 0.039] ROUGE-L. Granite's mean judge differences versus GPT-6 Luna were negative: -0.44 factuality [-0.54, -0.33] and -0.78 overall quality [-0.87, -0.69]. Against the strongest local judge runner-up, LFM2.5-1.2B, Granite's mean differences were +0.54 factuality [+0.41, +0.68], +0.09 coverage [+0.01, +0.17], and +0.24 overall quality [+0.12, +0.36].

The judge rated GPT-6 Luna at 5.00 readability and 4.98 concision on average, indicating a ceiling effect. These scores should not be interpreted as calibrated probabilities or as proof that GPT-6 Luna is factually perfect. The judge may also share stylistic preferences with the candidates, and no human rating was performed.

## Truncation, timing, and hosted usage

The 256-token cap was retained for all candidates. `finish_reason=length` occurred 2 times for Qwen3.5-0.8B, once for Qwen3.5-2B, twice for MiniCPM5-1B, and zero times for LFM2.5-1.2B, Granite 4.0-1B, and GPT-6 Luna. Mean generation latencies were 0.714 s, 1.164 s, 0.426 s, 0.469 s, 0.725 s, and 1.906 s, respectively. These timings reflect the recorded hardware/provider conditions and are not a controlled cross-provider throughput benchmark.

GPT-6 Luna generation usage reported `$0.037243` for 250 examples. The judge's successful-response usage records total about `$0.2222` across the six candidate sets. The judging throughput policy changed during this work: MiniCPM5, LFM2.5, and GPT-6 Luna used 60-second per-attempt timeouts, up to three transient-error retries, and 16 concurrent requests; Qwen3.5-0.8B and Granite used the earlier sequential CLI; Qwen3.5-2B combines sequential partial runs, with retries enabled for its final 78 examples. The judge model and rubric were unchanged. The new retry policy uses jittered exponential backoff and honors numeric `Retry-After` headers up to a 30-second cap; queued work is cancelled and in-flight requests stop retrying after a batch failure. An already-running HTTP call cannot be interrupted mid-call and can remain active until its per-attempt timeout. A client read timeout does not prove the provider stopped processing; a retry may create billable work whose usage is not returned to the client, so these usage totals can undercount provider-side work.

## Artifacts and limits

Predictions, judgments, per-run metadata, and ROUGE score files are preserved locally under `runs/confirmation250/` (git-ignored). The six judgment outputs each contain 250 unique IDs matching the corresponding prediction IDs in input order and have complete run metadata. The frozen sample itself is tracked at `data/eval/cnn_dailymail_test_confirmation.jsonl`.

This confirmation set is an additional generalization check, not a replacement for the 50-example pilot or a final broad benchmark. CNN/DailyMail has dataset-specific style and reference-summary biases; the sample is modest; local decoding settings differ by model recommendation; and judge outputs are automated. Granite is the strongest local LoRA candidate from these baseline results, while LFM2.5 is a useful second candidate. Do not treat these results as permission to train on the CNN/DailyMail test examples.
