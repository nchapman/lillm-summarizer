# Tiny-model summarization evaluation

This repo starts with **English news summarization**. The initial evaluation sample is drawn from the CNN/DailyMail 3.0.0 **test split**; it is a pilot for the inference/evaluation pipeline, not a training set. Keep the generated sample unchanged and reuse it for every model. The repo ignores downloaded data, model files, and run outputs.

See [PLAN.md](PLAN.md) for the broader benchmark and LoRA phases, and [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for the completed Q4 pilot comparison.

## Environment

Local machine: NVIDIA GB10 system (ARM64, CUDA 13.0, 128 GiB unified system memory). The baseline runs load one model server at a time and stop it before loading another; check available host memory before each new workload.

Python 3.12 is recommended. Install the small evaluation package and optional data/metric dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,metrics,dev]'
```

## Prepare the fixed pilot set

```bash
tinyllm-eval prepare-cnn-dm --count 50 --seed 20261028
```

This reads the test split as a stream and performs seeded reservoir sampling. Streaming still scans the split; it can take time on first use. Preserve `data/eval/cnn_dailymail_test_pilot.jsonl` and use this exact file across all model runs. The seed is not a substitute for preserving the generated file (dataset ordering can change upstream). The command refuses to overwrite an existing sample; pass `--overwrite` only when you intentionally want to replace it.

The rows contain `id`, `document`, `reference`, dataset/config/split, and sampling seed. Do not copy these records into LoRA training data. The CNN/DailyMail source texts and reference summaries are not tracked by Git.

## Run one model through llama.cpp

The repo's llama.cpp checkout has a CUDA build at `/home/nchapman/Code/llama.cpp/build-cuda/bin/llama-server` (commit `0878f7c`, build `0.7.0-dev`). It is external to this repo and is not modified by this harness. Hub checks found `UD-Q4_K_XL` files for Qwen3.5, LFM2.5, and Granite; MiniCPM5 uses the official `Q4_K_M` GGUF:

| Candidate | GGUF repository | File | Pinned Hub revision | Server alias |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | `unsloth/Qwen3.5-0.8B-GGUF` | `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` | `qwen35-0.8b` |
| Qwen3.5-2B | `unsloth/Qwen3.5-2B-GGUF` | `Qwen3.5-2B-UD-Q4_K_XL.gguf` | `f6d5376be1edb4d416d56da11e5397a961aca8ae` | `qwen35-2b` |
| MiniCPM5-1B | `openbmb/MiniCPM5-1B-GGUF` | `MiniCPM5-1B-Q4_K_M.gguf` | `3d55fac80935ae6456986ad2384b5cbcc4d6c948` | `minicpm5-1b` |
| LFM2.5-1.2B-Instruct | `unsloth/LFM2.5-1.2B-Instruct-GGUF` | `LFM2.5-1.2B-Instruct-UD-Q4_K_XL.gguf` | `b01cc872e5918b0fdec170003cd854851e3ed854` | `lfm25-1.2b` |
| Granite 4.0-1B | `unsloth/granite-4.0-1b-GGUF` | `granite-4.0-1b-UD-Q4_K_XL.gguf` | `cfdca559b6af6d2d6c443a5e51d8700ec04d67c9` | `granite4-1b` |

Use a pinned Hub commit and explicit local file, because llama.cpp's `-hf` short form fetches the current repo revision. Example for Qwen3.5-0.8B:

```bash
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
  --revision 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 \
  --local-dir models/qwen35-0.8b

/home/nchapman/Code/llama.cpp/build-cuda/bin/llama-server \
  -m models/qwen35-0.8b/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
  --alias qwen35-0.8b --port 8080 -ngl all -c 8192 --parallel 1 --reasoning off
```

Download each model's listed repo, file, and pinned revision. Start only one model at a time, with `--parallel 1`, and stop it before loading the next to keep unified-memory use bounded. Qwen3.5 needs `--reasoning off`; MiniCPM5 no-think mode additionally needs `--chat-template-kwargs '{"enable_thinking":false}'`.

Use the model-card sampler settings rather than forcing greedy decoding. Qwen3.5 non-thinking text: temperature 1.0, top-p 1.0, top-k 20, min-p 0.0, presence penalty 2.0. MiniCPM5-1B no-think: temperature 0.7, top-p 0.95. LFM2.5-1.2B-Instruct: temperature 0.1, top-k 50, repeat penalty 1.05 (leave unspecified samplers at the pinned llama.cpp defaults). Granite 4.0-1B: temperature 0, top-p 1, top-k 0. The CLI records sampler settings and uses seed 42; sampling models are seeded, not greedy.

Settings were checked against the [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) cards, the [MiniCPM5-1B card](https://huggingface.co/openbmb/MiniCPM5-1B), [Liquid's LFM2.5-1.2B-Instruct docs](https://docs.liquid.ai/lfm/models/lfm25-1.2b-instruct), and [Granite 4.0 Unsloth docs](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0).

In another shell, run the same frozen pilot for each served model:

```bash
tinyllm-eval run \
  --input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output runs/qwen35-0.8b-q4/smoke.jsonl \
  --model qwen35-0.8b \
  --model-revision 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 --quantization UD-Q4_K_XL \
  --server-version 0878f7c --max-new-tokens 256 \
  --temperature 1.0 --top-p 1.0 --top-k 20 --min-p 0.0 \
  --presence-penalty 2.0 --repeat-penalty 1.0 --limit 3
```

The first call uses `--limit 3` and a separate `smoke.jsonl` output as an end-to-end smoke test. After checking outputs and prompt formatting, run the full 50-example pilot to a different output path:

```bash
tinyllm-eval run \
  --input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output runs/qwen35-0.8b-q4/predictions.jsonl \
  --model qwen35-0.8b \
  --model-revision 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 --quantization UD-Q4_K_XL \
  --server-version 0878f7c --max-new-tokens 256 \
  --temperature 1.0 --top-p 1.0 --top-k 20 --min-p 0.0 \
  --presence-penalty 2.0 --repeat-penalty 1.0
```

The harness uses one fixed summarization prompt, seed 42, and requests one completion at a time. Sampler flags are sent per request and recorded in metadata; `--max-new-tokens` defaults to 192. Use the matching model-card settings above for each candidate and update its revision/quantization metadata. It saves raw output, reference, timing, token usage, and run metadata. Metadata is marked `running`, `complete`, or `failed`, with the completed row count, so a failed batch is not mistaken for complete. Output is never overwritten unless `--overwrite` is supplied.

Score prediction output with ROUGE F1:

```bash
tinyllm-eval score --input runs/qwen35-0.8b-q4/predictions.jsonl
```

ROUGE is reference-overlap only, not a factuality or overall quality judgment. For a project-specific OpenRouter credential, add this line to the repo-root `.env` file (already git-ignored):

```text
OPENROUTER_API_KEY=<your OpenRouter API key>
```

Keep `.env` restricted to your user (`chmod 600 .env`). The judge loads this project-local file without overriding an `OPENROUTER_API_KEY` already set in the process environment. Then run the LLM judge on predictions:

```bash
tinyllm-eval judge \
  --input runs/qwen35-0.8b-q4/predictions.jsonl \
  --output runs/qwen35-0.8b-q4/judgments.jsonl
```

The judge defaults to `~deepseek/deepseek-flash-latest` on OpenRouter. It sees the source document and candidate summary (not the reference), and returns 1–5 scores for factuality, coverage, relevance, concision, readability, and overall quality, plus unsupported-claim notes. Override the model with `--model` if needed. Judgment outputs include judge model and token usage; keep them alongside predictions. Treat these as an additional comparison signal alongside ROUGE, not a single combined score. Do not tune prompts against the final evaluation set.

The judge sends source articles and predictions to OpenRouter. Use only data you are permitted to send to that service. It writes incrementally to a temporary file and tracks `running`/`failed`/`complete` status and completed count in a temporary metadata sidecar. It publishes the final JSONL and `.meta.json` sidecar only on success, preserving any previous complete output and metadata if a rerun fails. Use `--overwrite` to replace them after a successful run.

The initial 50-example pilot is sufficient to test the pipeline and surface obvious differences; expand the sample if candidate rankings are close or unstable.

## Tests

```bash
python -m unittest discover -s tests -v
```
