# CNN/DailyMail confirmation comparison

## Summary

The disjoint 250-example confirmation set supports the pilot's main local-model conclusion: **Granite 4.0-1B remains the strongest local candidate overall**. It leads the local candidates on mean ROUGE-1/2/L and on all six source-grounded judge dimensions. The newly added Granite 4.0-350M is a striking size/quality point: at the shared 256-token cap it was about 2.1× faster than Granite 1B and came second among local models on all three ROUGE metrics. Its observed judge scores were lower, but the comparison was confounded by output length (11 cap hits versus none for Granite 1B). A 512-token rerun of 350M reduced cap hits to eight but did not clearly raise its overall judge score: the paired change was -0.036 [-0.080, +0.008]. Only 14 outputs changed, so the rerun does not establish an adequate-length quality comparison; all other models remain at 256 tokens, and eight 350M outputs still hit the higher cap. Granite 1B's paired ROUGE advantage over 350M at 256 is clear for ROUGE-1 and ROUGE-L; the ROUGE-2 difference interval includes zero.

GPT-6 Luna receives substantially higher judge ratings than any local model, but both Granite models score above it on mean ROUGE-1/2/L; for 350M versus GPT-6 Luna, the ROUGE-1 interval includes zero while ROUGE-2/L favor 350M. Judge scores are close to the scale ceiling and should be treated cautiously. LFM2.5-1.2B is the local judge runner-up. Granite 4.0-350M has the lowest local mean latency. The Qwen3.5 models received lower judge scores and ROUGE than the other local candidates. These are model-level comparisons on one random sample, not claims about all news summarization tasks.

## Protocol

- Dataset: CNN/DailyMail 3.0.0 test split; frozen, seeded sample of 250 examples, disjoint by ID and article from the frozen 50-example pilot. Sample SHA-256: `047f62c289d3eb08ce3dd5f3096736cf40f0414b1f602a052ee8e79a1377e75e`.
- All seven candidates used the same examples, `summary-v1` prompt, and 256-token output cap. Local models used model-specific Q4 files and sampling settings; the new Granite 4.0-350M used `unsloth/granite-4.0-350m-GGUF` revision `d6a88cd56438020200760fd8bc03670db0e13fa0`, file `granite-4.0-350m-UD-Q4_K_XL.gguf`, with the card's greedy generation default translated to temperature 0, top-p 1, top-k 0. Its instruct base is `ibm-granite/granite-4.0-350m` at revision `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2`. GPT-6 Luna used OpenRouter provider-managed decoding, seed 42, `reasoning_effort=none`, and omitted temperature.
- ROUGE-1/2/L are F1 scores from `rouge-score` with Porter stemming. ROUGE measures reference overlap, not factuality.
- The judge was `~deepseek/deepseek-flash-latest`. It saw the source and candidate summary, not the reference; dimensions are integer 1–5. Unsupported-claim counts are totals of returned notes, not a severity-weighted error rate.
- Intervals are percentile 95% bootstrap CIs from 5,000 paired resamples of the 250 example IDs (seed `20261029`). They describe sampling uncertainty on this set; they do not capture judge or model stochasticity, and comparisons are not adjusted for multiple testing.

## Results

Means are followed by 95% bootstrap confidence intervals. Judge scores are on the 1–5 scale. Latency is mean per-example generation latency from the recorded run metadata.

| Candidate | ROUGE-1 | ROUGE-2 | ROUGE-L | Judge factuality | Judge overall | Unsupported-claim notes | Mean latency | `finish_reason=length` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 0.315 [0.305, 0.325] | 0.077 [0.072, 0.083] | 0.172 [0.166, 0.179] | 1.81 [1.72, 1.91] | 1.80 [1.71, 1.90] | 1,148 | 0.714 s | 2 |
| Qwen3.5-2B | 0.330 [0.320, 0.340] | 0.092 [0.086, 0.098] | 0.185 [0.179, 0.192] | 2.57 [2.45, 2.69] | 2.58 [2.47, 2.69] | 721 | 1.164 s | 1 |
| MiniCPM5-1B | 0.327 [0.315, 0.339] | 0.095 [0.088, 0.104] | 0.196 [0.188, 0.204] | 2.52 [2.38, 2.66] | 2.44 [2.31, 2.56] | 617 | 0.426 s | 2 |
| LFM2.5-1.2B | 0.344 [0.333, 0.354] | 0.101 [0.094, 0.109] | 0.211 [0.203, 0.219] | 3.98 [3.86, 4.10] | 3.89 [3.79, 3.99] | 226 | 0.469 s | 0 |
| **Granite 4.0-1B** | **0.404 [0.392, 0.416]** | **0.156 [0.145, 0.167]** | **0.258 [0.248, 0.268]** | **4.53 [4.42, 4.63]** | **4.13 [4.04, 4.22]** | **87** | 0.725 s | 0 |
| Granite 4.0-350M | 0.376 [0.363, 0.390] | 0.149 [0.138, 0.160] | 0.245 [0.234, 0.256] | 3.77 [3.62, 3.92] | 3.33 [3.21, 3.44] | 220 | 0.341 s | 11 |
| GPT-6 Luna (hosted) | 0.375 [0.363, 0.386] | 0.123 [0.115, 0.132] | 0.229 [0.219, 0.239] | 4.96 [4.94, 4.98] | 4.91 [4.87, 4.94] | 9 | 1.906 s | 0 |

The full set of judge means and intervals:

| Candidate | Factuality | Coverage | Relevance | Concision | Readability | Overall quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 1.81 [1.72, 1.91] | 2.28 [2.19, 2.36] | 2.61 [2.51, 2.71] | 2.84 [2.76, 2.92] | 2.69 [2.61, 2.77] | 1.80 [1.71, 1.90] |
| Qwen3.5-2B | 2.57 [2.45, 2.69] | 2.86 [2.78, 2.94] | 3.37 [3.28, 3.46] | 3.52 [3.45, 3.60] | 3.45 [3.36, 3.53] | 2.58 [2.47, 2.69] |
| MiniCPM5-1B | 2.52 [2.38, 2.66] | 2.40 [2.30, 2.49] | 3.08 [2.97, 3.20] | 3.51 [3.42, 3.60] | 3.18 [3.08, 3.29] | 2.44 [2.31, 2.56] |
| LFM2.5-1.2B | 3.98 [3.86, 4.10] | 3.25 [3.17, 3.32] | 4.18 [4.10, 4.25] | 4.40 [4.34, 4.46] | 4.50 [4.44, 4.57] | 3.89 [3.79, 3.99] |
| Granite 4.0-1B | 4.53 [4.42, 4.63] | 3.34 [3.26, 3.41] | 4.45 [4.37, 4.52] | 4.65 [4.59, 4.71] | 4.78 [4.72, 4.83] | 4.13 [4.04, 4.22] |
| Granite 4.0-350M | 3.77 [3.62, 3.92] | 2.98 [2.90, 3.07] | 3.91 [3.82, 4.00] | 3.85 [3.74, 3.95] | 4.02 [3.91, 4.12] | 3.33 [3.21, 3.44] |
| GPT-6 Luna (hosted) | 4.96 [4.94, 4.98] | 4.12 [4.08, 4.16] | 4.92 [4.89, 4.96] | 4.98 [4.97, 5.00] | 5.00 [5.00, 5.00] | 4.91 [4.87, 4.94] |

## Granite 4.0-350M cap sensitivity rerun

At the user's request, Granite 4.0-350M alone was rerun on the same 250 IDs with the same `summary-v1` prompt, seed, greedy sampler, and a 512-token output cap. All other model results remain at the original 256-token cap, so this is a within-model sensitivity check, not a same-cap comparison across models.

| Granite 4.0-350M run | ROUGE-1 | ROUGE-2 | ROUGE-L | Judge factuality | Judge overall | `finish_reason=length` | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256-token cap | 0.376 [0.363, 0.390] | 0.149 [0.138, 0.160] | 0.245 [0.234, 0.256] | 3.77 [3.62, 3.92] | 3.33 [3.21, 3.44] | 11 | 0.341 s |
| 512-token cap | 0.373 [0.359, 0.387] | 0.147 [0.136, 0.159] | 0.243 [0.231, 0.254] | 3.73 [3.57, 3.88] | 3.29 [3.17, 3.41] | 8 | 0.384 s |

Paired 512-minus-256 changes in ROUGE were -0.0035 [-0.0059, -0.0014] ROUGE-1, -0.0017 [-0.0040, -0.0002] ROUGE-2, and -0.0027 [-0.0049, -0.0010] ROUGE-L. The full judge-dimension means and paired changes were:

| Judge dimension | 512-token mean [95% CI] | Paired change, 512 minus 256 [95% CI] |
| --- | ---: | ---: |
| Factuality | 3.73 [3.57, 3.88] | -0.040 [-0.084, +0.004] |
| Coverage | 2.94 [2.85, 3.03] | -0.044 [-0.088, -0.004] |
| Relevance | 3.87 [3.78, 3.96] | -0.040 [-0.088, +0.008] |
| Concision | 3.88 [3.76, 3.98] | +0.028 [-0.020, +0.076] |
| Readability | 3.99 [3.88, 4.10] | -0.024 [-0.068, +0.020] |
| Overall quality | 3.29 [3.17, 3.41] | -0.036 [-0.080, +0.008] |

There was no clear increase in judge overall quality. Fourteen prediction strings differed between runs: all 11 previously cap-hit outputs changed (three no longer hit the cap), and three outputs that had not hit the old cap also changed. Eight outputs still ended at the 512-token cap. This run-to-run generation variation, despite greedy decoding, and judge repeat variability on unchanged predictions mean the paired changes cannot be attributed solely to the token cap. The sensitivity check therefore does not show that the 256-token cap alone caused the lower 350M judge scores.

The 512-token run metadata and predictions are preserved under the ignored `runs/confirmation250/granite4-350m-q4-512/` directory. All 250 predictions and judgments match the frozen sample IDs in input order.

## Paired comparison

Granite 4.0-1B's paired bootstrap ROUGE difference intervals are above zero versus every other candidate except Granite 350M on ROUGE-2, where the interval includes zero. Granite 1B minus 350M is +0.027 [0.016, 0.038] ROUGE-1, +0.007 [-0.003, 0.016] ROUGE-2, and +0.013 [0.003, 0.022] ROUGE-L. In the observed 256-token judge scores, 350M minus 1B is -0.76 factuality [-0.94, -0.58] and -0.80 overall quality [-0.94, -0.66]. Since 11 of the 350M candidates hit the 256-token generation cap while none of the 1B candidates did, these judge differences are confounded by truncation. The 512-token 350M-only sensitivity rerun below still had eight cap hits and does not isolate an adequate-length comparison against Granite 1B, whose run remains at 256 tokens.

Granite 350M's 256-token run minus GPT-6 Luna is +0.002 [-0.010, +0.014] ROUGE-1 (interval includes zero), +0.026 [+0.015, +0.037] ROUGE-2, and +0.016 [+0.006, +0.026] ROUGE-L. Granite 1B's mean differences versus GPT-6 Luna were +0.029 [0.018, 0.041] ROUGE-1, +0.033 [0.022, 0.043] ROUGE-2, and +0.029 [0.019, 0.039] ROUGE-L; its judge differences were -0.44 factuality [-0.54, -0.33] and -0.78 overall quality [-0.87, -0.69]. Against the strongest local judge runner-up, LFM2.5-1.2B, Granite 1B's mean differences were +0.54 factuality [+0.41, +0.68], +0.09 coverage [+0.01, +0.17], and +0.24 overall quality [+0.12, +0.36].

The judge rated GPT-6 Luna at 5.00 readability and 4.98 concision on average, indicating a ceiling effect. These scores should not be interpreted as calibrated probabilities or as proof that GPT-6 Luna is factually perfect. The judge may also share stylistic preferences with the candidates, and no human rating was performed.

## Truncation, timing, and hosted usage

The 256-token cap was retained for the seven-candidate comparison table. `finish_reason=length` occurred 2 times for Qwen3.5-0.8B, once for Qwen3.5-2B, twice for MiniCPM5-1B, 11 times for Granite 4.0-350M, and zero times for LFM2.5-1.2B, Granite 4.0-1B, and GPT-6 Luna. Mean generation latencies were 0.714 s, 1.164 s, 0.426 s, 0.469 s, 0.725 s, 0.341 s, and 1.906 s in the order of that table. The 350M model was about 2.1× faster than Granite 1B at the 256-token cap on this host. The separate 512-token 350M-only sensitivity run averaged 0.384 s/example and still had eight cap hits. These timings reflect recorded hardware/provider conditions and are not a controlled cross-provider throughput benchmark.

GPT-6 Luna generation usage reported `$0.037243` for 250 examples. The judge's successful-response usage records total about `$0.2575` across the seven original candidate sets, plus `$0.033552` for the 350M 512-token sensitivity batch (about `$0.2910` total across eight judging batches). The sensitivity batch had 16 requests retried; as noted below, usage records can undercount provider-side work after timeouts. The judging throughput policy changed during this work: Granite 350M, MiniCPM5, LFM2.5, and GPT-6 Luna used 60-second per-attempt timeouts, up to three transient-error retries, and 16 concurrent requests; Qwen3.5-0.8B and Granite 1B used the earlier sequential CLI; Qwen3.5-2B combines sequential partial runs, with retries enabled for its final 78 examples. The judge model and rubric were unchanged. The new retry policy uses jittered exponential backoff and honors numeric `Retry-After` headers up to a 30-second cap; queued work is cancelled and in-flight requests stop retrying after a batch failure. An already-running HTTP call cannot be interrupted mid-call and can remain active until its per-attempt timeout. A client read timeout does not prove the provider stopped processing; a retry may create billable work whose usage is not returned to the client, so these usage totals can undercount provider-side work.

## Artifacts and limits

Predictions, judgments, per-run metadata, and ROUGE score files are preserved locally under `runs/confirmation250/` (git-ignored). All seven original judgment outputs and the additional 512-token Granite 350M judgment output contain 250 unique IDs matching their corresponding prediction IDs in input order and have complete run metadata. Granite 4.0-350M's GGUF is the Unsloth UD-Q4_K_XL file at the pinned revision above (SHA-256 `1fd597eb32557fabb17695eaeb4f367f286dc67d464b0f97c904d70373d067d3`); the original instruct model card is Apache-2.0. The frozen sample is a local artifact at `data/eval/cnn_dailymail_test_confirmation.jsonl`; evaluation JSONL files are Git-ignored and are not distributed with the repository.

This confirmation set is an additional generalization check, not a replacement for the 50-example pilot or a final broad benchmark. CNN/DailyMail has dataset-specific style and reference-summary biases; the sample is modest; local decoding settings differ by model recommendation; and judge outputs are automated. Granite 1B remains the strongest overall local LoRA baseline; Granite 350M is a compelling low-latency size point with close ROUGE, while its lower observed judge scores remain difficult to interpret: the 512-token rerun did not clearly raise its overall rating and still had eight cap hits. LFM2.5 remains a useful second judge-quality candidate. Do not treat these results as permission to train on the CNN/DailyMail test examples.
