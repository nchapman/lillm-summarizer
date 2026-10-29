# Tiny-model summarization evaluation

This repo starts with **English news summarization**. Evaluation samples are drawn from the CNN/DailyMail 3.0.0 **test split** and are never training data. The original 50-example pilot is retained as a stable pipeline/baseline set; a separate 250-example confirmation set is used to check whether its comparisons generalize. Preserve each generated file and use the same examples for every model. The repo ignores downloaded data, model files, and run outputs.

See [PLAN.md](PLAN.md) for the broader benchmark and LoRA phases, [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for the 50-example Q4 pilot, and [CONFIRMATION_RESULTS.md](CONFIRMATION_RESULTS.md) for the disjoint 250-example confirmation comparison.

## Environment

Local machine: NVIDIA GB10 system (ARM64, CUDA 13.0, 128 GiB unified system memory). The baseline runs load one model server at a time and stop it before loading another; check available host memory before each new workload.

Use **uv ≥0.11.23** for all Python dependencies and commands. `.python-version` pins Python 3.12.12; `pyproject.toml` declares dependencies and the committed `uv.lock` freezes their resolution for local ARM64 and remote x86_64 machines. Data preparation and ROUGE dependencies are included by default, along with the `dev` group for pytest; the default environment does not install Torch or CUDA packages.

```bash
uv sync --locked
uv run --locked tinyllm-eval --help
uv run --locked pytest tests/ -q
```

uv installs the pinned Python if needed and creates an isolated `.venv`; no activation or system-Python changes are required. Install uv using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/) if unavailable. Use `uv add` / `uv add --group train` for intentional dependency changes, inspect the resulting `pyproject.toml` and `uv.lock` diff, and commit both. Use `uv lock` only when intentionally updating the resolution; setup and execution use `--locked` so dependency drift fails rather than silently rewriting the lock.

### GPU training setup (local or remote Linux)

The `train` dependency group pins the previously verified Unsloth/TRL/PEFT/Transformers stack and CUDA 13.0 Torch/torchvision. The explicit official PyTorch CUDA index also supplies xFormers on x86_64; ARM64 does not require xFormers. Training uses Python 3.12, Linux x86_64 or ARM64, an NVIDIA driver compatible with CUDA 13.0, and a BF16-capable GPU. The pod's preinstalled Torch is not reused.

From a code-only checkout on the GPU machine:

```bash
uv sync --locked --group train
uv pip check
uv run --locked --group train pytest tests/ -q
uv run --locked --group train python scripts/train_lora.py --help
```

Keep `--group train` on GPU commands. A later plain `uv sync --locked` switches back to the lightweight evaluation environment and removes training-only packages; re-add the group to restore the locked training environment. The default evaluation environment remains usable on non-GPU machines; the Linux-only training group is not a macOS/Windows training setup.

ARM64 packaging note: NVIDIA's `nvidia-cusparselt-cu13==0.8.1` ARM wheel has an internal `manylinux2014_sbsa` tag despite an `aarch64` filename and AArch64 ELF payload. This makes `uv pip check` report that one package as built for a different platform; the same artifact exists in the previously working GB10 stack. The locked ARM64 environment passes the BF16 CUDA check below, and the x86_64 pod passes `uv pip check` cleanly. Do not hand-edit wheel metadata or ignore other dependency errors.

Before loading a model, check the GPU runtime (Unsloth must be imported first):

```bash
uv run --locked --group train python - <<'PY'
import unsloth
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item(), "Nonfinite BF16 matmul"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

These are environment checks, not an 8K model-training smoke test. Complete the model/data preflight before starting a full run. Keep private data and derived reports out of Git and public storage; transfer them separately over authenticated SSH only when ready. Disable Hub telemetry with `HF_HUB_DISABLE_TELEMETRY=1` / `DO_NOT_TRACK=1`; training already uses `report_to=[]`. Verify whether pod storage is persistent and arrange checkpoint backups before a long run. No credentials, SSH keys, or private datasets belong in the code checkout or lockfile.

## Prepare the fixed pilot set

```bash
uv run --locked tinyllm-eval prepare-cnn-dm --count 50 --seed 20261028
```

This reads the test split as a stream and performs seeded reservoir sampling. Streaming still scans the split; it can take time on first use. Preserve `data/eval/cnn_dailymail_test_pilot.jsonl` and use this exact file across all model runs. The seed is not a substitute for preserving the generated file (dataset ordering can change upstream). The command refuses to overwrite an existing sample; pass `--overwrite` only when you intentionally want to replace it.

The rows contain `id`, `document`, `reference`, dataset/config/split, and sampling seed. Do not copy these records into LoRA training data. The CNN/DailyMail source texts and reference summaries are not tracked by Git.

Generate the independent confirmation set by excluding the frozen pilot's IDs:

```bash
uv run --locked tinyllm-eval prepare-cnn-dm \
  --count 250 --seed 20261029 \
  --exclude-input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output data/eval/cnn_dailymail_test_confirmation.jsonl
```

This is a second seeded reservoir sample from the remaining test examples, not a replacement for the original pilot. The confirmation set has 250 unique IDs and no article or ID overlap with the pilot. Its SHA-256 is `047f62c289d3eb08ce3dd5f3096736cf40f0414b1f602a052ee8e79a1377e75e`; 101/250 articles have ≤500 words, 115 have 501–1000, and 34 have >1000. A scan of the remaining valid test split found respective proportions of 35.5%, 47.0%, and 17.4%, so the random sample includes all three length bands without imposed quotas. Keep the JSONL itself for exact reuse; the exclusion-source hash is recorded in every row.

## Summarize prompt (summary-v3)

New preparations, evaluation runs, and the eventual summarize button share one instruction (`PROMPT_VERSION=summary-v3`, defined in `src/tinyllm_eval/core.py`):

> Summarize the page below clearly and objectively, capturing its main points and important supporting details in your own words. Keep simple material brief, and provide more detail when the source warrants it. State the substance directly rather than describing the page, attributing opinions or arguments inline where appropriate. Write in plain prose without markdown, using multiple paragraphs when helpful, and do not add unsupported information.

Length follows source complexity: there is no sentence, word, or paragraph cap. The system message carries the instruction; the user message is the raw page text. Historical `summary-v1` and `summary-v2` contracts remain verbatim in `core.py`; recorded results and existing prepared files/checkpoints are not rewritten. Fresh adapter comparisons must re-run base models under v3. An 8K training context limit applies to the **entire templated conversation including the target and EOS**, not just the source or an arbitrary summary-length limit.

## Prepare LoRA training data (smoltalk2)

The training data is `HuggingFaceTB/smoltalk2`, config `SFT`, split `smoltalk_smollm3_smol_summarize_no_think` (96,061 rows; news-article and email summarization targets distilled from a larger teacher). Provenance: the news portion comes from the CNN/DailyMail 3.0.0 **train** split — never the test split our evaluation samples — and rows carry one of two generic length instructions ("up to three sentences" or "one very short sentence") in `chat_template_kwargs.custom_instructions`; the news/email split is not recoverable from the data. Both content types are kept for general web-page summarization.

```bash
uv run --locked tinyllm-eval prepare-smoltalk
```

The command pins and records the Hub revision, streams the split, drops exact duplicate documents and any row whose 8-gram containment against both frozen evaluation files reaches 0.2, prepends the current system prompt, and writes `data/train/smoltalk2_smol-summarize_no_think.jsonl` plus a `.meta.json` report (only the meta report is tracked). Historical v2 output: 92,704 rows kept (59,855 `three_sentence`, 32,849 `one_line`), 3,333 duplicates and 24 leakage near-duplicates dropped (max kept score 0.198). Target shape: `one_line` targets are one sentence (p50 18 words); `three_sentence` targets are p50 62 words but only half are 2–3 sentences — 16% run 4–6 sentences and 33% are 7+ (p75 163 words), i.e. the teacher frequently exceeded its "up to three sentences" instruction on long articles, so treat that tag as instruction provenance rather than a length guarantee. Rows keep `document_words`/`target_words`/`tag`/`leakage_score` fields for later mix or length ablations. As with the evaluation files, preserve the generated JSONL for exact reuse; regenerate only with `--overwrite` and an intentional revision change.

## Import a private Folio SFT export

Keep the raw export outside the repository. Its directory must contain `sft.jsonl.gz`, the paired `audit.jsonl.gz`, and `report.json`:

```bash
uv run --locked tinyllm-eval prepare-folio --source-dir /path/to/private/sft_export
```

This local-only command streams the export into `data/train/folio_sft_v1/aligned.jsonl` with a `.meta.json` sidecar. The entire `data/train/folio*/` directory is Git-ignored, including provenance and any later length audits or splits. Both generated files are owner-only (0600). **Never commit private records, excerpts, export hashes, provenance manifests, or generated reports.** If overriding `--output`, choose an equally private, ignored destination.

The importer verifies both compressed files against the source report, then checks every audit row index and document/summary hash. It replaces only the system message with our current `summary-v3` prompt; user text and assistant targets remain exact, without stripping, wrapping, or adding model-specific special tokens. The trainer applies the model's native chat template and completion-only loss as usual. Verified content hashes, original row indices, and quality flags remain available outside the messages; private URLs, titles, and other page metadata are not copied from the audit into the aligned pool. Teacher model/prompt and the source's review status are retained in the private sidecar, not promoted to an endorsement of target quality.

As with smoltalk, preparation deduplicates exact documents and excludes rows reaching 0.2 word 8-gram containment against the frozen pilot and confirmation inputs (`--eval-input` can override these). Invalid rows or broken provenance fail the import rather than being silently skipped. Existing outputs or sidecars require `--overwrite`; a failed validation leaves old files untouched.

The result is **unsplit and not context-filtered**. Before assigning train/validation sets, audit the complete chats (including targets and EOS) with the exact training tokenizer/revision and choose an explicit context-length policy. Do not let the trainer silently truncate long sources or their summaries. Review upstream quality flags and provisional review status before selecting a training pool. Keep any resulting splits and diagnostics under the ignored private directory, and use explicit `--train`/`--val` paths for a separately planned run. This import does not mix data into existing smoltalk splits, change the deployment prompt, or resume training/checkpoints.

## Combine prepared pools locally

`tinyllm_eval.combined_data.prepare_combined` accepts a source-name-to-JSONL mapping, a fresh output directory, an exact tokenizer, explicit frozen evaluation paths and tokenizer provenance. It recognizes prepared v2/v3 instructions, aligns only the system message to v3, preserves all document/target text, checks roles and document IDs, deduplicates normalized documents across sources, and reruns evaluation containment checks. Tokenizer controls (including special decoder entries absent from `all_special_tokens`), forbidden control characters and replacement characters are quarantined.

Complete chats are tokenized without truncation; over-context rows are excluded while originals remain untouched. Long and multi-paragraph targets are retained. Missing final punctuation is a **review warning**, not proof of truncation and not an automatic exclusion. Passing these mechanical checks does not certify semantic quality or readiness to train. A separate review can pass an explicit `review_exclusions` mapping (source → document ID → short reason code); unknown sources or IDs fail preparation. This supports reversible quarantine of suspect teacher cutoffs without imposing a global target-length rule. Keep the review mapping and its evidence private.

All retained rows are assigned deterministically to train or validation, proportional over source × token-length bands. Singletons stay in train; infeasible validation sizes fail. The fresh output directory (0700) contains `train.jsonl`, `val.jsonl`, `exclusions.jsonl` and `metadata.json` (0600). Atomic no-replace publication requires Linux. Use an ignored `data/train/folio*/` destination: all combined outputs, counts, fingerprints and reports are private and must remain untracked. Preparation never resumes or starts training.

## LoRA training smoke test (phase 3)

`scripts/train_smoke.py` proves the Granite 4.0-1B training path end to end on a small slice of the prepared data:

```bash
uv run --locked --group train python scripts/train_smoke.py --adapter-dir adapters/granite4-1b-smoke
```

It loads `ibm-granite/granite-4.0-1b`, attaches a LoRA adapter (r=16), trains two epochs on 197 rows with completion-only loss using the prepared rows' prompt, generates held-out base/adapter examples, saves the native adapter, and reloads it on a fresh base model via transformers + PEFT. The historical attention-only smoke run decreased loss 1.20 → 0.45 over 50 steps (~53 s) and generated coherent summaries. Its reload heuristic checked only the first 64 prompt positions: it did **not** establish assistant-output or deployment equivalence, nor prove the cause of cross-engine differences. Current scripts import Unsloth first and use `use_exact_model_name=True` to honor the requested repository/revision; historical runs were remapped to an Unsloth mirror. See the [pipeline audit](docs/granite-training-audit.md). Adapter artifacts and Unsloth's compile cache are untracked.

Architecture note: every Granite 4.0 model shares the `granitemoehybrid` config class (`GraniteMoeHybridForCausalLM`), but the layer composition is what distinguishes them — `granite-4.0-1b` is 40 attention layers with zero Mamba layers and zero routed experts (a plain dense transformer, the standard architecture the plan intended), while `granite-4.0-h-1b` is the true hybrid (36 Mamba + 4 attention). The class name is a family-wide red herring. Its MLP is `shared_mlp.{input,output}_linear` (no Llama-style gate/up/down projections exist), so the first training runs attached LoRA to attention projections only; unsloth's own ablation recommends attention + MLP together, so later runs add `input_linear`/`output_linear` to the targets. Current `train_meta.json`/`smoke_meta.json` record both the requested and the actually applied target modules from the saved `adapter_config.json`; historical smoke metadata recorded only requested targets.

Pinned stack (now declared in the uv `train` group and locked in `uv.lock`; also recorded in historical `adapters/*/smoke_meta.json`): unsloth 2026.10.10, trl 1.15.0, peft 0.19.1, transformers 4.57.6, torch 2.12.1+cu130. Unsloth installed cleanly on ARM64/CUDA 13 without touching the working torch build. Extend the same script to the other finalists before building the full training pipeline; the fresh mixed-data comparison must re-run base models under summary-v3 first.

## Fresh prepared mixed-data runs

After the data-quality gate and hardware benchmark, pass explicit `--train`, `--val`, `--prepared-manifest`, `--model-revision`, `--max-length` and a **new** `--adapter-dir` to `scripts/train_lora.py`. The manifest mode verifies file fingerprints, counts, current prompts, document IDs, stored complete-chat lengths, and train/validation separation before importing the GPU stack. It rejects resume and nonempty output directories, then checks actual trainer lengths and supervised EOS before any optimizer step.

Validation reports an aggregate loss plus separate source losses; aggregate `eval_all_loss` selects the best checkpoint and drives early stopping. `--per-device-eval-batch-size` can be tuned independently of the training batch size. `--gradient-checkpointing` accepts `unsloth` (the historical default), `standard`, or `off`; the choice is applied to both adapter setup and trainer configuration. Disable it only after measuring worst-case context batches with sufficient memory headroom, not just average-length batches. New private runs should use `--sample-count 0` during training and perform full generations in a separate local evaluation. Run under `umask 077`, disable external experiment reporting, and keep all run artifacts/metadata private. No mixed-data run should reuse the historical pilot checkpoint or its validation split.

The pinned Granite/Unsloth path can warn that `num_items_in_batch` could not be counted with gradient accumulation. The warning is not restricted to training: the SFT evaluation override omits that count and can trigger it even when training counts targets correctly. Check the warning's timing and the runtime denominator before inferring a training fallback. If training actually lacks the count, its ordinary NLL fallback correctly scales by the accumulation-group size, but averages **microbatch mean losses**, not all supervised tokens across the update. This is a different weighting objective, not an accumulation-factor gradient amplification: shorter-target microbatches give their tokens more weight, and batch composition matters. Do not assume the difference is necessarily small or that changing microbatch size while keeping the effective batch size fixed is objective-equivalent. Source-only evaluation callbacks may also warn that `eval_all_loss` is absent; they do not disable the separate aggregate-loss early-stopping callback invocation.

## Full LoRA training (phase 4 pilot)

Split the prepared data into stratified train/validation sets (seeded, proportional over tag × document-length bands so short emails and long news stay represented; ~3,700 tokens/s measured on the GB10, so ~0.9 h per 24k-row epoch):

```bash
uv run --locked tinyllm-eval subset-training   # 24,000 train + 1,000 val rows, seed 20261128
uv run --locked --group train python scripts/train_lora.py --adapter-dir adapters/granite4-1b-24k-s42
```

`train_lora.py` defaults to per-epoch validation loss and early stopping (patience 1). For earlier diagnostics, `--eval-steps N` evaluates and saves every N **optimizer steps**, with matched evaluation/save cadence for best-checkpoint restoration. `--patience` counts evaluation events, so increase it when evaluating more frequently to preserve the intended training-time horizon. Generation quality still requires separate evaluation; lower validation loss does not establish factuality. The script restores the best checkpoint, runs held-out base-vs-adapter generations and the limited fresh-base reload heuristic, and writes `train_meta.json` (data hashes/selected IDs, loss curves, applied LoRA targets, versions, generations). It now uses **all 1,000 validation rows by default**; an explicit `--val-limit` draws a seeded uniform sample from the entire file, not a sorted prefix. The stopped checkpoint-3000 run used the old default: 200/200 validation rows were `one_line`, so its loss **0.5058 is not mixed-task validation loss**. Do not compare that historical metric with the corrected validation split or silently resume using different validation data. Generated splits and adapters are untracked.

### Adapter deployment path (unmerged; use a BF16 sidecar)

Keep the Q4 base and adapter separate. **Use BF16, not F16, for this Granite adapter.** [IBM warns that Granite 4.0-1B exceeds F16's numerical range](https://huggingface.co/ibm-granite/granite-4.0-1b-GGUF#known-issues) and recommends BF16 for its full-precision model. Our captured CUDA operands confirm the same range problem in checkpoint-3000's **full attention+MLP adapter**: an MLP activation of 83,001 exceeds F16's maximum of 65,504, and the F16 LoRA matmul produces infinities. The BF16 sidecar produces finite logits and coherent summaries in controlled spot checks. Export with an explicit dtype:

```bash
uv run --locked --group train python /home/nchapman/Code/llama.cpp/convert_lora_to_gguf.py \
  adapters/granite4-1b-24k-s42/checkpoints/checkpoint-3000 \
  --base ~/.cache/huggingface/hub/models--ibm-granite--granite-4.0-1b/snapshots/6a7381ba1f54d684ff508d991aeb7dc580157103 \
  --outtype bf16 --outfile /tmp/granite24k_epoch1_lora-bf16.gguf
```

`--base` is the local **unquantized HF snapshot directory**, not a repo ID or GGUF. Serve using `llama-server -m models/granite4-1b/granite-4.0-1b-UD-Q4_K_XL.gguf --lora /tmp/granite24k_epoch1_lora-bf16.gguf -ngl all -c 8192 --parallel 1`. This applies LoRA on activations without merging or discarding MLP targets. The BF16 sidecar is ~41 MB; the base remains quantized. BF16 retains a wide exponent range in 16-bit storage; F32 export is not required for the verified path. The [audit](docs/granite-training-audit.md) records the model-specific guidance, captured-operand/reference comparison, and CPU/CUDA dtype controls. Two real held-out BF16-sidecar checks were coherent with exact HF/llama.cpp prompt-token matches; a fresh HTTP-server check also completed normally. These are correctness spot checks, **not a quality evaluation or authorization to resume training**. The earlier F16 pilot outputs remain invalid for quality comparison.

The server advertises hot-swapping (`--lora-init-without-apply`, POST `/lora-adapters`) and multiple/scaled adapters. That API has **not been exercised here**; startup adapter loading and hot-swapping are separate verification steps.

### Epoch-1 end-to-end evaluation

The [epoch-1 BF16-sidecar report](GRANITE_ADAPTER_RESULTS.md) records a complete fresh base-versus-adapter battery under `summary-v2`: 50 pilot + 250 confirmation examples per candidate, ROUGE, and 600 final source-grounded judgments. This was a successful **end-to-end pipeline checkpoint before training was finished**, not a final training verdict. At epoch 1, the adapter improved observed judge coverage but produced longer summaries, more cap hits, and lower factuality/concision/overall scores than the base. Preserve these as the reference for later checkpoints. Latency is excluded because of an overlapping GPU workload. Training was not resumed by the evaluation.

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
uv run --locked hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
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
uv run --locked tinyllm-eval run \
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
uv run --locked tinyllm-eval run \
  --input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output runs/qwen35-0.8b-q4/predictions.jsonl \
  --model qwen35-0.8b \
  --model-revision 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 --quantization UD-Q4_K_XL \
  --server-version 0878f7c --max-new-tokens 256 \
  --temperature 1.0 --top-p 1.0 --top-k 20 --min-p 0.0 \
  --presence-penalty 2.0 --repeat-penalty 1.0
```

The harness uses one fixed summarization prompt, seed 42, and requests one completion at a time. Sampler flags are sent per request and recorded in metadata; `--max-new-tokens` defaults to 1024 for source-dependent summary lengths; this is a generation safety ceiling, not a desired output length. Ensure the server context has room for both the source prompt and this output allowance, and inspect cap hits rather than treating cutoff generations as complete. Use the matching model-card settings above for each candidate and update its revision/quantization metadata. It saves raw output, reference, timing, token usage, and run metadata. Metadata is marked `running`, `complete`, or `failed`, with the completed row count, so a failed batch is not mistaken for complete. Output is never overwritten unless `--overwrite` is supplied.

### Hosted control: GPT-6 Luna through OpenRouter

`openai/gpt-6-luna` is a hosted reference/control, not part of the local ≤2B/Q4 cohort. It uses the same prompt and frozen pilot but provider-managed weights and decoding. [OpenRouter's model page](https://openrouter.ai/openai/gpt-6-luna) and [OpenAI's model docs](https://developers.openai.com/api/docs/models/gpt-6-luna) advertise support for `max_tokens`, `seed`, and reasoning effort `none`; OpenRouter does not advertise temperature/top-p controls for this model, so the request omits them and disables reasoning. A pinned model slug and provider name are recorded in metadata.

The command reads `OPENROUTER_API_KEY` from the project `.env` (without overriding a shell value), sends the article text to OpenRouter, and does not require a llama-server or GPU:

```bash
uv run --locked tinyllm-eval run \
  --input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output runs/gpt-6-luna-openrouter/predictions.jsonl \
  --model openai/gpt-6-luna \
  --model-revision openai/gpt-6-luna-20260922 --quantization provider-managed \
  --server-version OpenRouter --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY --reasoning-effort none --max-new-tokens 256
```

The measured 50-example generation cost was about `$0.0082` per OpenRouter usage records; rates can change. Score and judge it like the local outputs:

```bash
uv run --locked tinyllm-eval score --input runs/gpt-6-luna-openrouter/predictions.jsonl
uv run --locked tinyllm-eval judge \
  --input runs/gpt-6-luna-openrouter/predictions.jsonl \
  --output runs/gpt-6-luna-openrouter/judgments.jsonl
```

Score prediction output with ROUGE F1:

```bash
uv run --locked tinyllm-eval score --input runs/qwen35-0.8b-q4/predictions.jsonl
```

ROUGE is reference-overlap only, not a factuality or overall quality judgment. For a project-specific OpenRouter credential, add this line to the repo-root `.env` file (already git-ignored):

```text
OPENROUTER_API_KEY=<your OpenRouter API key>
```

Keep `.env` restricted to your user (`chmod 600 .env`). The judge loads this project-local file without overriding an `OPENROUTER_API_KEY` already set in the process environment. Then run the LLM judge on predictions:

```bash
uv run --locked tinyllm-eval judge \
  --input runs/qwen35-0.8b-q4/predictions.jsonl \
  --output runs/qwen35-0.8b-q4/judgments.jsonl
```

The judge defaults to `~deepseek/deepseek-flash-latest` on OpenRouter. It sees the source document and candidate summary (not the reference), and returns 1–5 scores for factuality, coverage, relevance, concision, readability, and overall quality, plus unsupported-claim notes. Override the model with `--model` if needed. Judgment outputs include judge model and token usage; keep them alongside predictions. Treat these as an additional comparison signal alongside ROUGE, not a single combined score. Do not tune prompts against the final evaluation set.

The judge sends source articles and predictions to OpenRouter. Use only data you are permitted to send to that service. It runs up to 16 requests concurrently by default (`--concurrency 32` raises this to 32), with a 60-second per-attempt timeout (`--timeout`) and three retries for transient network failures and HTTP 408/425/429/5xx responses (`--max-retries`). Retries use jittered exponential backoff and honor numeric `Retry-After` headers up to a 30-second backoff cap; queued work is cancelled and in-flight requests stop retrying after a batch failure. An in-flight HTTP request cannot be interrupted mid-call and can remain active until its per-attempt timeout. Successful outputs are restored to input order; each row records its attempt count. It writes incrementally to a temporary file and tracks the request settings plus `running`/`failed`/`complete` status and completed count in a temporary metadata sidecar. It publishes the final JSONL and `.meta.json` sidecar only on success, preserving any previous complete output and metadata if a rerun fails. Usage records reflect successful responses; a provider may still have processed a request whose response timed out, so retries can incur unreported provider usage. Use `--overwrite` to replace them after a successful run.

The initial 50-example pilot is sufficient to test the pipeline and surface obvious differences; expand the sample if candidate rankings are close or unstable.

## Tests

```bash
uv run --locked pytest tests/ -q
```
