# Q4 baseline comparison

The five local candidates were evaluated on the same frozen 50-example CNN/DailyMail 3.0.0 pilot (sample SHA-256: `e45c1587fe25dc3cf510adbda8b5ac962a74417b2f51f47e3a2606e46a251d93`), using pinned Q4 GGUFs, the `summary-v1` prompt, llama.cpp CUDA server commit `0878f7c`, and model-recommended sampling settings. An OpenAI GPT-6 Luna hosted control was added afterward via OpenRouter on the same examples and prompt. The Qwen3.5 and MiniCPM5 runs used their non-thinking/no-think modes. All prediction sets received 50 judgments from `~deepseek/deepseek-flash-latest`; judge scores are means on the 1–5 scale. These are pilot comparisons, not precise estimates of general performance.

## Results

| Candidate | Quantization | ROUGE-1 | ROUGE-2 | ROUGE-L | Judge factuality | Judge overall | Unsupported-claim notes* | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | UD-Q4_K_XL | 0.3277 | 0.0861 | 0.1778 | 1.80 | 1.74 | 240 | 0.720 s |
| Qwen3.5-2B | UD-Q4_K_XL | 0.3454 | 0.0991 | 0.1940 | 2.58 | 2.58 | 141 | 1.130 s |
| MiniCPM5-1B | Q4_K_M | 0.3214 | 0.0902 | 0.1944 | 2.24 | 2.22 | 128 | 0.429 s |
| LFM2.5-1.2B-Instruct | UD-Q4_K_XL | 0.3531 | 0.1000 | 0.2076 | 3.82 | 3.72 | 54 | 0.485 s |
| Granite 4.0-1B | UD-Q4_K_XL | **0.3985** | **0.1646** | **0.2602** | **4.58** | **4.12** | **16** | 0.787 s |

\*Count of unsupported-claim entries returned by the judge over all 50 cases; not a severity-weighted error rate. ROUGE is reference overlap, not a factuality measure.

Among the local ≤2B candidates, Granite 4.0-1B led on all three ROUGE measures and judge factuality/overall quality. The hosted GPT-6 Luna control received higher judge means (4.98 factuality, 4.90 overall) but lower ROUGE than Granite; the judge scores are near-ceiling and should be interpreted cautiously. LFM2.5-1.2B-Instruct was runner-up among local models on judge factuality/overall and second-fastest; MiniCPM5-1B had the lowest local mean latency (0.429 s). Granite remains the strongest initial LoRA candidate, with LFM a useful second candidate. The hosted model is a quality reference only, not a trainable/quantized candidate. Keep conclusions provisional: the sample is small, and the models use different recommended/default decoding rather than identical sampling settings.

## Hosted control: GPT-6 Luna via OpenRouter

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Judge factuality | Judge coverage | Judge overall | Mean latency | Generation cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI GPT-6 Luna (`openai/gpt-6-luna`) | 0.3685 | 0.1133 | 0.2181 | 4.98 | 4.02 | 4.90 | 2.076 s | $0.008233 |

The model used OpenRouter's provider-managed hosted weights, `reasoning_effort=none`, seed 42, and a 256-token maximum. Temperature and other sampler values were omitted because OpenRouter does not advertise them as supported for this model. The generation usage records reported `$0.008233` total for 50 examples. This is not directly comparable to the local Q4 cohort: it is a much larger hosted model with provider-managed decoding. The independent judge assigned 5/5 factuality to 49 of 50 outputs and 5/5 overall quality to 45 of 50, so its strong judge results are close to the scale ceiling.

## Runs and sampling

All artifacts are local under `runs/` and git-ignored. Each folder contains `predictions.jsonl`, `predictions.scores.json`, `judgments.jsonl`, and run metadata.

| Candidate | Run directory | Sampling |
| --- | --- | --- |
| Qwen3.5-0.8B / 2B | `runs/qwen35-0.8b-q4/`, `runs/qwen35-2b-q4/` | temperature 1, top-p 1, top-k 20, min-p 0, presence penalty 2, repeat penalty 1 |
| MiniCPM5-1B | `runs/minicpm5-1b-q4/` | no-think template; temperature 0.7, top-p 0.95 |
| LFM2.5-1.2B-Instruct | `runs/lfm25-1.2b-q4/` | temperature 0.1, top-k 50, repeat penalty 1.05; other sampler values left at pinned llama.cpp defaults |
| Granite 4.0-1B | `runs/granite4-1b-q4/` | temperature 0, top-p 1, top-k 0; 16,384-token context |
| GPT-6 Luna hosted control | `runs/gpt-6-luna-openrouter/` | reasoning effort `none`; temperature omitted; provider defaults |

The Qwen/MiniCPM samplers were seeded (seed 42) but stochastic. Qwen/Granite/LFM used `UD-Q4_K_XL`; MiniCPM5 used the official `Q4_K_M` because no matching Unsloth dynamic Q4 was available. Exact Hub repository revisions and llama.cpp launch examples are in `README.md`.

## Resource behavior and issues

The local models were loaded, evaluated, judged, and unloaded sequentially. No OOM or inference failure occurred. During these runs, observed llama-server GPU-memory use was about 0.87 GiB for Qwen3.5-0.8B, 0.97 GiB for MiniCPM5, 1.06 GiB for LFM2.5, and 2.52 GiB for Granite with its 16K context. Host memory remained ample (about 114–117 GiB available during the runs); the GPU was idle after each server was stopped. GPT-6 Luna used remote OpenRouter inference and added no local model/GPU memory use. The earlier LFM OOM occurred during a separate attempt with another GPU workload active and did not recur with the Q4 model run sequentially.

An earlier Q8_0/temperature-0 pilot exists at `runs/qwen35-0.8b/`, `runs/qwen35-2b/`, and `runs/minicpm5-1b/`. It used different quantization/decoding and is not included in this table. See `INITIAL_RESULTS.md` for that historical setup check.
