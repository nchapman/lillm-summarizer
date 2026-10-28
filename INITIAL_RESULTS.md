# Historical Q8 pilot result (superseded)

The Q4 comparison with model-recommended sampling is documented in [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md). This Q8_0 result is retained only as a historical pipeline/setup check; do not compare it directly with the Q4 runs.

This is an end-to-end setup check, not a model ranking or a reliable estimate of expected performance. The 50 examples are a seeded sample from the CNN/DailyMail 3.0.0 test split. Keep this exact sample for paired comparisons; the data file is local and git-ignored.

## Qwen3.5-0.8B, Q8_0

- GGUF: `ggml-org/Qwen3.5-0.8B-GGUF`
- GGUF repo revision: `807dfd9a6fadbf703207aac3bfb0b2b3040d077c`
- Inference: llama.cpp CUDA server, commit `0878f7c`; reasoning disabled; temperature 0; seed 42; max 256 output tokens.
- Prompt: `summary-v1` in `src/tinyllm_eval/core.py`.
- Sample SHA-256: `e45c1587fe25dc3cf510adbda8b5ac962a74417b2f51f47e3a2606e46a251d93`
- Predictions: `runs/qwen35-0.8b/predictions.jsonl` (local, git-ignored).
- ROUGE F1: ROUGE-1 `0.3524`, ROUGE-2 `0.1235`, ROUGE-L `0.2034` (mean per-example F1, Porter stemming).
- Mean per-example end-to-end generation time: `0.895 s` (range `0.537–1.598 s`); all 50 outputs non-empty and ended with `stop`.

A 3-example initial smoke test showed plausible summaries but also factual errors (including an unsupported detail and a mistaken attribution). This reinforces that ROUGE alone is insufficient. The later Q4 pilot adds the requested source-grounded LLM judge; see `BASELINE_RESULTS.md` for the aggregate comparison. These 50 examples remain an initial pilot, so conclusions about model quality are provisional.
