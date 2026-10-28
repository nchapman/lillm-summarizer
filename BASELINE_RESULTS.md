# Q4 baseline comparison

All five candidates were evaluated on the same frozen 50-example CNN/DailyMail 3.0.0 pilot (sample SHA-256: `e45c1587fe25dc3cf510adbda8b5ac962a74417b2f51f47e3a2606e46a251d93`). Each used a pinned Q4 GGUF, the `summary-v1` prompt, llama.cpp CUDA server commit `0878f7c`, and model-recommended sampling settings. The Qwen3.5 and MiniCPM5 runs used their non-thinking/no-think modes. The judge was `~deepseek/deepseek-flash-latest`; each prediction set received 50 judgments. Judge scores are means on the 1–5 scale. These are pilot comparisons, not precise estimates of general performance.

## Results

| Candidate | Quantization | ROUGE-1 | ROUGE-2 | ROUGE-L | Judge factuality | Judge overall | Unsupported-claim notes* | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | UD-Q4_K_XL | 0.3277 | 0.0861 | 0.1778 | 1.80 | 1.74 | 240 | 0.720 s |
| Qwen3.5-2B | UD-Q4_K_XL | 0.3454 | 0.0991 | 0.1940 | 2.58 | 2.58 | 141 | 1.130 s |
| MiniCPM5-1B | Q4_K_M | 0.3214 | 0.0902 | 0.1944 | 2.24 | 2.22 | 128 | 0.429 s |
| LFM2.5-1.2B-Instruct | UD-Q4_K_XL | 0.3531 | 0.1000 | 0.2076 | 3.82 | 3.72 | 54 | 0.485 s |
| Granite 4.0-1B | UD-Q4_K_XL | **0.3985** | **0.1646** | **0.2602** | **4.58** | **4.12** | **16** | 0.787 s |

\*Count of unsupported-claim entries returned by the judge over all 50 cases; not a severity-weighted error rate. ROUGE is reference overlap, not a factuality measure.

Granite 4.0-1B led this pilot on all three ROUGE measures and judge factuality/overall quality. LFM2.5-1.2B-Instruct was runner-up on judge factuality/overall and second-fastest; MiniCPM5-1B had the lowest mean latency (0.429 s). This gives Granite the strongest initial case for a LoRA trial, with LFM as a useful second candidate. Keep conclusions provisional: the sample is small, and the candidates use different model-recommended decoders rather than identical sampling settings.

## Runs and sampling

All artifacts are local under `runs/` and git-ignored. Each folder contains `predictions.jsonl`, `predictions.scores.json`, `judgments.jsonl`, and run metadata.

| Candidate | Run directory | Sampling |
| --- | --- | --- |
| Qwen3.5-0.8B / 2B | `runs/qwen35-0.8b-q4/`, `runs/qwen35-2b-q4/` | temperature 1, top-p 1, top-k 20, min-p 0, presence penalty 2, repeat penalty 1 |
| MiniCPM5-1B | `runs/minicpm5-1b-q4/` | no-think template; temperature 0.7, top-p 0.95 |
| LFM2.5-1.2B-Instruct | `runs/lfm25-1.2b-q4/` | temperature 0.1, top-k 50, repeat penalty 1.05; other sampler values left at pinned llama.cpp defaults |
| Granite 4.0-1B | `runs/granite4-1b-q4/` | temperature 0, top-p 1, top-k 0; 16,384-token context |

The Qwen/MiniCPM samplers were seeded (seed 42) but stochastic. Qwen/Granite/LFM used `UD-Q4_K_XL`; MiniCPM5 used the official `Q4_K_M` because no matching Unsloth dynamic Q4 was available. Exact Hub repository revisions and llama.cpp launch examples are in `README.md`.

## Resource behavior and issues

Models were loaded, evaluated, judged, and unloaded sequentially. No OOM or inference failure occurred. During these runs, observed llama-server GPU-memory use was about 0.87 GiB for Qwen3.5-0.8B, 0.97 GiB for MiniCPM5, 1.06 GiB for LFM2.5, and 2.52 GiB for Granite with its 16K context. Host memory remained ample (about 114–117 GiB available during the runs); the GPU was idle after each server was stopped. The earlier LFM OOM occurred during a separate attempt with another GPU workload active and did not recur with the Q4 model run sequentially.

An earlier Q8_0/temperature-0 pilot exists at `runs/qwen35-0.8b/`, `runs/qwen35-2b/`, and `runs/minicpm5-1b/`. It used different quantization/decoding and is not included in this table. See `INITIAL_RESULTS.md` for that historical setup check.
