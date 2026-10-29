# Tiny-model summarization evaluation

This repo starts with **English news summarization**. Evaluation samples are drawn from the CNN/DailyMail 3.0.0 **test split** and are never training data. The original 50-example pilot is retained as a stable pipeline/baseline set; a separate 250-example confirmation set is used to check whether its comparisons generalize. Preserve each generated file and use the same examples for every model. The repo ignores downloaded data, model files, and run outputs.

See [PLAN.md](PLAN.md) for the broader benchmark and LoRA phases, [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for the 50-example Q4 pilot, and [CONFIRMATION_RESULTS.md](CONFIRMATION_RESULTS.md) for the disjoint 250-example confirmation comparison.

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

Generate the independent confirmation set by excluding the frozen pilot's IDs:

```bash
tinyllm-eval prepare-cnn-dm \
  --count 250 --seed 20261029 \
  --exclude-input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output data/eval/cnn_dailymail_test_confirmation.jsonl
```

This is a second seeded reservoir sample from the remaining test examples, not a replacement for the original pilot. The confirmation set has 250 unique IDs and no article or ID overlap with the pilot. Its SHA-256 is `047f62c289d3eb08ce3dd5f3096736cf40f0414b1f602a052ee8e79a1377e75e`; 101/250 articles have ≤500 words, 115 have 501–1000, and 34 have >1000. A scan of the remaining valid test split found respective proportions of 35.5%, 47.0%, and 17.4%, so the random sample includes all three length bands without imposed quotas. Keep the JSONL itself for exact reuse; the exclusion-source hash is recorded in every row.

## Summarize prompt (summary-v2)

All new runs, LoRA training data, and the eventual summarize button share one fixed instruction (`PROMPT_VERSION=summary-v2`, defined in `src/tinyllm_eval/core.py`):

> Summarize the page below in a concise, objective paragraph that captures its main points and key details, rephrasing the content in your own words.

The system message carries the instruction; the user message is the raw page text. The recorded baseline and confirmation tables were produced under `summary-v1` (also kept in `core.py`), so adapter-phase comparisons must re-run each base model under v2 before comparing LoRA gains.

## Prepare LoRA training data (smoltalk2)

The training data is `HuggingFaceTB/smoltalk2`, config `SFT`, split `smoltalk_smollm3_smol_summarize_no_think` (96,061 rows; news-article and email summarization targets distilled from a larger teacher). Provenance: the news portion comes from the CNN/DailyMail 3.0.0 **train** split — never the test split our evaluation samples — and rows carry one of two generic length instructions ("up to three sentences" or "one very short sentence") in `chat_template_kwargs.custom_instructions`; the news/email split is not recoverable from the data. Both content types are kept for general web-page summarization.

```bash
tinyllm-eval prepare-smoltalk
```

The command pins and records the Hub revision, streams the split, drops exact duplicate documents and any row whose 8-gram containment against both frozen evaluation files reaches 0.2, prepends the fixed summary-v2 system prompt, and writes `data/train/smoltalk2_smol-summarize_no_think.jsonl` plus a `.meta.json` report (only the meta report is tracked). Current output: 92,704 rows kept (59,855 `three_sentence`, 32,849 `one_line`), 3,333 duplicates and 24 leakage near-duplicates dropped (max kept score 0.198). Target shape: `one_line` targets are one sentence (p50 18 words); `three_sentence` targets are p50 62 words but only half are 2–3 sentences — 16% run 4–6 sentences and 33% are 7+ (p75 163 words), i.e. the teacher frequently exceeded its "up to three sentences" instruction on long articles, so treat that tag as instruction provenance rather than a length guarantee. Rows keep `document_words`/`target_words`/`tag`/`leakage_score` fields for later mix or length ablations. As with the evaluation files, preserve the generated JSONL for exact reuse; regenerate only with `--overwrite` and an intentional revision change.

## LoRA training smoke test (phase 3)

`scripts/train_smoke.py` proves the Granite 4.0-1B training path end to end on a small slice of the prepared data:

```bash
python scripts/train_smoke.py --adapter-dir adapters/granite4-1b-smoke
```

It loads `ibm-granite/granite-4.0-1b`, attaches a LoRA adapter (r=16), trains two epochs on 197 rows with completion-only loss under the summary-v2 prompt, generates held-out base/adapter examples, saves the native adapter, and reloads it on a fresh base model via transformers + PEFT. The historical attention-only smoke run decreased loss 1.20 → 0.45 over 50 steps (~53 s) and generated coherent summaries. Its reload heuristic checked only the first 64 prompt positions: it did **not** establish assistant-output or deployment equivalence, nor prove the cause of cross-engine differences. Current scripts import Unsloth first and use `use_exact_model_name=True` to honor the requested repository/revision; historical runs were remapped to an Unsloth mirror. See the [pipeline audit](docs/granite-training-audit.md). Adapter artifacts and Unsloth's compile cache are untracked.

Architecture note: every Granite 4.0 model shares the `granitemoehybrid` config class (`GraniteMoeHybridForCausalLM`), but the layer composition is what distinguishes them — `granite-4.0-1b` is 40 attention layers with zero Mamba layers and zero routed experts (a plain dense transformer, the standard architecture the plan intended), while `granite-4.0-h-1b` is the true hybrid (36 Mamba + 4 attention). The class name is a family-wide red herring. Its MLP is `shared_mlp.{input,output}_linear` (no Llama-style gate/up/down projections exist), so the first training runs attached LoRA to attention projections only; unsloth's own ablation recommends attention + MLP together, so later runs add `input_linear`/`output_linear` to the targets. Current `train_meta.json`/`smoke_meta.json` record both the requested and the actually applied target modules from the saved `adapter_config.json`; historical smoke metadata recorded only requested targets.

Pinned stack (also recorded in `adapters/*/smoke_meta.json`): unsloth 2026.10.10, trl 1.15.0, peft 0.19.1, transformers 4.57.6, torch 2.12.1+cu130. Unsloth installed cleanly on ARM64/CUDA 13 without touching the working torch build. Extend the same script to the other finalists before building the full training pipeline; per PLAN.md, the full comparison re-runs base models under summary-v2 first.

## Full LoRA training (phase 4 pilot)

Split the prepared data into stratified train/validation sets (seeded, proportional over tag × document-length bands so short emails and long news stay represented; ~3,700 tokens/s measured on the GB10, so ~0.9 h per 24k-row epoch):

```bash
tinyllm-eval subset-training   # 24,000 train + 1,000 val rows, seed 20261128
python scripts/train_lora.py --adapter-dir adapters/granite4-1b-24k-s42
```

`train_lora.py` trains with per-epoch validation loss, early stopping (patience 1), best-checkpoint restoration, held-out base-vs-adapter generations, the limited fresh-base reload heuristic, and writes `train_meta.json` (data hashes/selected IDs, loss curves, applied LoRA targets, versions, generations). It now uses **all 1,000 validation rows by default**; an explicit `--val-limit` draws a seeded uniform sample from the entire file, not a sorted prefix. The stopped checkpoint-3000 run used the old default: 200/200 validation rows were `one_line`, so its loss **0.5058 is not mixed-task validation loss**. Do not compare that historical metric with the corrected validation split or silently resume using different validation data. Generated splits and adapters are untracked.

### Adapter deployment path (unmerged; use an F32 sidecar on this CUDA build)

Keep the Q4 base and adapter separate. For checkpoint-3000's **full attention+MLP adapter**, F16 GGUF sidecars degenerate on the pinned GB10/CUDA llama.cpp build; **F32 sidecars produce coherent summaries in controlled spot checks**. Export with an explicit dtype:

```bash
python /home/nchapman/Code/llama.cpp/convert_lora_to_gguf.py \
  adapters/granite4-1b-24k-s42/checkpoints/checkpoint-3000 \
  --base ~/.cache/huggingface/hub/models--ibm-granite--granite-4.0-1b/snapshots/6a7381ba1f54d684ff508d991aeb7dc580157103 \
  --outtype f32 --outfile /tmp/granite24k_epoch1_lora-f32.gguf
```

`--base` is the local **unquantized HF snapshot directory**, not a repo ID or GGUF. Serve using `llama-server -m models/granite4-1b/granite-4.0-1b-UD-Q4_K_XL.gguf --lora /tmp/granite24k_epoch1_lora-f32.gguf -ngl all -c 8192 --parallel 1`. This applies LoRA on activations without merging or discarding MLP targets. The F32 sidecar is ~81 MB; the base remains quantized. The [audit](docs/granite-training-audit.md) records the CPU/CUDA and dtype controls, including numerically identical F16-rounded coefficients recovering when stored as F32. Do not describe this as a gate/up conversion bug or assume training gains transfer unchanged to Q4. The earlier F16 pilot outputs are invalid for quality comparison.

The server advertises hot-swapping (`--lora-init-without-apply`, POST `/lora-adapters`) and multiple/scaled adapters. That API has **not been exercised here**; startup adapter loading and hot-swapping are separate verification steps.

## Run one model through llama.cpp

The repo's llama.cpp checkout has a CUDA build at `/home/nchapman/Code/llama.cpp/build-cuda/bin/llama-server` (commit `0878f7c`, build `0.7.0-dev`). It is external to this repo and is not modified by this harness. Hub checks found `UD-Q4_K_XL` files for Qwen3.5, LFM2.5, and Granite; MiniCPM5 uses the official `Q4_K_M` GGUF:

| Candidate | GGUF repository | File | Pinned Hub revision | Server alias |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | `unsloth/Qwen3.5-0.8B-GGUF` | `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` | `qwen35-0.8b` |
| Qwen3.5-2B | `unsloth/Qwen3.5-2B-GGUF` | `Qwen3.5-2B-UD-Q4_K_XL.gguf` | `f6d5376be1edb4d416d56da11e5397a961aca8ae` | `qwen35-2b` |
| MiniCPM5-1B | `openbmb/MiniCPM5-1B-GGUF` | `MiniCPM5-1B-Q4_K_M.gguf` | `3d55fac80935ae6456986ad2384b5cbcc4d6c948` | `minicpm5-1b` |
| LFM2.5-1.2B-Instruct | `unsloth/LFM2.5-1.2B-Instruct-GGUF` | `LFM2.5-1.2B-Instruct-UD-Q4_K_XL.gguf` | `b01cc872e5918b0fdec170003cd854851e3ed854` | `lfm25-1.2b` |
| Granite 4.0-1B | `unsloth/granite-4.0-1b-GGUF` | `granite-4.0-1b-UD-Q4_K_XL.gguf` | `cfdca559b6af6d2d6c443a5e51d8700ec04d67c9` | `granite4-1b` |
| Granite 4.0-350M | `unsloth/granite-4.0-350m-GGUF` | `granite-4.0-350m-UD-Q4_K_XL.gguf` | `d6a88cd56438020200760fd8bc03670db0e13fa0` | `granite4-350m` |

Use a pinned Hub commit and explicit local file, because llama.cpp's `-hf` short form fetches the current repo revision. Example for Qwen3.5-0.8B:

```bash
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
  --revision 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 \
  --local-dir models/qwen35-0.8b

/home/nchapman/Code/llama.cpp/build-cuda/bin/llama-server \
  -m models/qwen35-0.8b/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
  --alias qwen35-0.8b --port 8080 -ngl all -c 8192 --parallel 1 --reasoning off
```

Download each model's listed repo, file, and pinned revision. Start only one model at a time, with `--parallel 1`, and stop it before loading the next to keep unified-memory use bounded. Use `--reasoning off` for Qwen3.5 and MiniCPM5 no-thinking runs; this llama.cpp build deprecates MiniCPM5's `enable_thinking` chat-template override.

Use the model-card sampler settings rather than forcing greedy decoding. Qwen3.5 non-thinking text: temperature 1.0, top-p 1.0, top-k 20, min-p 0.0, presence penalty 2.0. MiniCPM5-1B no-think: temperature 0.7, top-p 0.95. LFM2.5-1.2B-Instruct: temperature 0.1, top-k 50, repeat penalty 1.05 (leave unspecified samplers at the pinned llama.cpp defaults). Granite 4.0-1B: temperature 0, top-p 1, top-k 0. Granite 4.0-350M's card example uses generation defaults (greedy); translated to llama.cpp, use temperature 0, top-p 1, top-k 0. The CLI records sampler settings and uses seed 42; sampling models are seeded, not greedy.

Settings were checked against the [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) cards, the [MiniCPM5-1B card](https://huggingface.co/openbmb/MiniCPM5-1B), [Liquid's LFM2.5-1.2B-Instruct docs](https://docs.liquid.ai/lfm/models/lfm25-1.2b-instruct), [Granite 4.0-1B Unsloth docs](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0), and the [Granite 4.0-350M model card](https://huggingface.co/ibm-granite/granite-4.0-350m). We use the non-H 350M dense model because IBM identifies it as the traditional architecture intended for compatibility with llama.cpp and PEFT; avoid confusing it with the separate hybrid `granite-4.0-h-350m` variant.

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

### Hosted control: GPT-6 Luna through OpenRouter

`openai/gpt-6-luna` is a hosted reference/control, not part of the local ≤2B/Q4 cohort. It uses the same prompt and frozen pilot but provider-managed weights and decoding. [OpenRouter's model page](https://openrouter.ai/openai/gpt-6-luna) and [OpenAI's model docs](https://developers.openai.com/api/docs/models/gpt-6-luna) advertise support for `max_tokens`, `seed`, and reasoning effort `none`; OpenRouter does not advertise temperature/top-p controls for this model, so the request omits them and disables reasoning. A pinned model slug and provider name are recorded in metadata.

The command reads `OPENROUTER_API_KEY` from the project `.env` (without overriding a shell value), sends the article text to OpenRouter, and does not require a llama-server or GPU:

```bash
tinyllm-eval run \
  --input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output runs/gpt-6-luna-openrouter/predictions.jsonl \
  --model openai/gpt-6-luna \
  --model-revision openai/gpt-6-luna-20260922 --quantization provider-managed \
  --server-version OpenRouter --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY --reasoning-effort none --max-new-tokens 256
```

The measured 50-example generation cost was about `$0.0082` per OpenRouter usage records; rates can change. Score and judge it like the local outputs:

```bash
tinyllm-eval score --input runs/gpt-6-luna-openrouter/predictions.jsonl
tinyllm-eval judge \
  --input runs/gpt-6-luna-openrouter/predictions.jsonl \
  --output runs/gpt-6-luna-openrouter/judgments.jsonl
```

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

The judge sends source articles and predictions to OpenRouter. Use only data you are permitted to send to that service. It runs up to 16 requests concurrently by default (`--concurrency 32` raises this to 32), with a 60-second per-attempt timeout (`--timeout`) and three retries for transient network failures and HTTP 408/425/429/5xx responses (`--max-retries`). Retries use jittered exponential backoff and honor numeric `Retry-After` headers up to a 30-second backoff cap; queued work is cancelled and in-flight requests stop retrying after a batch failure. An in-flight HTTP request cannot be interrupted mid-call and can remain active until its per-attempt timeout. Successful outputs are restored to input order; each row records its attempt count. It writes incrementally to a temporary file and tracks the request settings plus `running`/`failed`/`complete` status and completed count in a temporary metadata sidecar. It publishes the final JSONL and `.meta.json` sidecar only on success, preserving any previous complete output and metadata if a rerun fails. Usage records reflect successful responses; a provider may still have processed a request whose response timed out, so retries can incur unreported provider usage. Use `--overwrite` to replace them after a successful run.

The initial 50-example pilot is sufficient to test the pipeline and surface obvious differences; expand the sample if candidate rankings are close or unstable.

## Tests

```bash
python -m unittest discover -s tests -v
```
