# Tiny-model summarization evaluation

This repo starts with **English news summarization**. The initial evaluation sample is drawn from the CNN/DailyMail 3.0.0 **test split**; it is a pilot for the inference/evaluation pipeline, not a training set. Keep the generated sample unchanged and reuse it for every model. The repo ignores downloaded data, model files, and run outputs.

See [PLAN.md](PLAN.md) for the broader benchmark and LoRA phases.

## Environment

Local machine inspected for this setup: NVIDIA GB10 system (ARM64, CUDA 13.0, 128 GiB system memory). A separate process was using about 58 GiB of GPU memory at inspection time, so this initial setup uses serial inference and a single local model server. Do not launch multiple models concurrently until GPU memory is free.

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

The repo's llama.cpp checkout has a CUDA build at `/home/nchapman/Code/llama.cpp/build-cuda/bin/llama-server` (commit `0878f7c`, build `0.7.0-dev`). It is external to this repo and is not modified by this harness. Hub checks confirmed these Q8_0 GGUF files are published:

| Candidate | GGUF repository | Q8_0 file | Hub revision checked 2026-10-28 | Server alias |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | `ggml-org/Qwen3.5-0.8B-GGUF` | `Qwen3.5-0.8B-Q8_0.gguf` | `807dfd9a6fadbf703207aac3bfb0b2b3040d077c` | `qwen35-0.8b` |
| Qwen3.5-2B | `unsloth/Qwen3.5-2B-GGUF` | `Qwen3.5-2B-Q8_0.gguf` | `f6d5376be1edb4d416d56da11e5397a961aca8ae` | `qwen35-2b` |
| MiniCPM5-1B | `openbmb/MiniCPM5-1B-GGUF` | `MiniCPM5-1B-Q8_0.gguf` | `3d55fac80935ae6456986ad2384b5cbcc4d6c948` | `minicpm5-1b` |
| LFM2.5-1.2B-Instruct | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `LFM2.5-1.2B-Instruct-Q8_0.gguf` | `8ed288026e23958ad9dfa92d53ed773a8eee7125` | `lfm25-1.2b` |
| Granite 4.0-1B | `ibm-granite/granite-4.0-1b-GGUF` | `granite-4.0-1b-Q8_0.gguf` | `b27c2fe3f211b7f44e80fa620177aea371099aaa` | `granite4-1b` |

Use a pinned Hub commit and explicit local file, because llama.cpp's `-hf` short form fetches the current repo revision. Example for Qwen3.5-0.8B:

```bash
hf download ggml-org/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q8_0.gguf \
  --revision 807dfd9a6fadbf703207aac3bfb0b2b3040d077c \
  --local-dir models/qwen35-0.8b

/home/nchapman/Code/llama.cpp/build-cuda/bin/llama-server \
  -m models/qwen35-0.8b/Qwen3.5-0.8B-Q8_0.gguf \
  --alias qwen35-0.8b --port 8080 -ngl all -c 8192 --reasoning off
```

For another model, download the corresponding repo, file, and revision from the table, changing the alias as well. Start only one model at a time. For a fair quality comparison, use Q8_0 and the same inference settings for all models, and record the llama.cpp build. `--reasoning off` is important for this summarization baseline: the first smoke run with Qwen3.5 otherwise spent its output budget in `reasoning_content` and returned an empty summary. The server can be stopped before loading the next model.

In another shell, run the same frozen pilot for each served model:

```bash
tinyllm-eval run \
  --input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output runs/qwen35-0.8b/smoke.jsonl \
  --model qwen35-0.8b \
  --model-revision 807dfd9a6fadbf703207aac3bfb0b2b3040d077c --quantization Q8_0 \
  --server-version 0878f7c --max-new-tokens 256 \
  --limit 3
```

The first call uses `--limit 3` and a separate `smoke.jsonl` output as an end-to-end smoke test. After checking outputs and prompt formatting, run the full 50-example pilot to a different output path:

```bash
tinyllm-eval run \
  --input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output runs/qwen35-0.8b/predictions.jsonl \
  --model qwen35-0.8b \
  --model-revision 807dfd9a6fadbf703207aac3bfb0b2b3040d077c --quantization Q8_0 \
  --server-version 0878f7c --max-new-tokens 256
```

The harness uses a fixed summarization prompt, temperature 0, seed 42, and requests one completion at a time. `--max-new-tokens` defaults to 192; the reported initial Qwen run explicitly used 256. Use `--model minicpm5-1b` etc. for the other server aliases and update the revision/quantization metadata accordingly. It saves raw output, reference, timing, token usage, and run metadata. Metadata is marked `running`, `complete`, or `failed`, with the completed row count, so a failed batch is not mistaken for complete. Output is never overwritten unless `--overwrite` is supplied.

Score prediction output with ROUGE F1:

```bash
tinyllm-eval score --input runs/qwen35-0.8b/predictions.jsonl
```

ROUGE is reference-overlap only, not a factuality or overall quality judgment. For a project-specific OpenRouter credential, add this line to the repo-root `.env` file (already git-ignored):

```text
OPENROUTER_API_KEY=<your OpenRouter API key>
```

Keep `.env` restricted to your user (`chmod 600 .env`). The judge loads this project-local file without overriding an `OPENROUTER_API_KEY` already set in the process environment. Then run the LLM judge on predictions:

```bash
tinyllm-eval judge \
  --input runs/qwen35-0.8b/predictions.jsonl \
  --output runs/qwen35-0.8b/judgments.jsonl
```

The judge defaults to `~deepseek/deepseek-flash-latest` on OpenRouter. It sees the source document and candidate summary (not the reference), and returns 1–5 scores for factuality, coverage, relevance, concision, readability, and overall quality, plus unsupported-claim notes. Override the model with `--model` if needed. Judgment outputs include judge model and token usage; keep them alongside predictions. Treat these as an additional comparison signal alongside ROUGE, not a single combined score. Do not tune prompts against the final evaluation set.

The judge sends source articles and predictions to OpenRouter. Use only data you are permitted to send to that service. It writes incrementally to a temporary file and tracks `running`/`failed`/`complete` status and completed count in a temporary metadata sidecar. It publishes the final JSONL and `.meta.json` sidecar only on success, preserving any previous complete output and metadata if a rerun fails. Use `--overwrite` to replace them after a successful run.

The initial 50-example pilot is sufficient to test the pipeline and surface obvious differences; expand the sample if candidate rankings are close or unstable.

## Tests

```bash
python -m unittest discover -s tests -v
```
