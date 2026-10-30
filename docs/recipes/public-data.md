# Public-data recipe

This recipe describes a reproducible **public-only** summarization adaptation run. It uses the CNN/DailyMail test split for evaluation and the public `HuggingFaceTB/smoltalk2` summarization split for training. Never train on evaluation examples. Generated files, models, and adapters are local artifacts and are not committed.

> This does not reproduce the published Qwen adapter results: those used a public/private training mixture, a separate training environment, and selected Qwen checkpoints. This public recipe follows the checked-in Granite training path and produces a new experiment.

## 1. Environment

Use Python 3.12, uv ≥0.11.23, Linux, an NVIDIA GPU with BF16 support, and a compatible CUDA 13.0 driver. Training can consume substantial GPU memory and time. The default environment supports data preparation/evaluation without GPU dependencies.

```bash
uv sync --locked
uv run --locked lillm-summarizer --help
uv run --locked pytest -q
```

On the GPU host, install the locked training group and check the environment:

```bash
uv sync --locked --group train
uv pip check
uv run --locked --group train python - <<'PY'
import unsloth  # Import first, before Transformers/TRL/PEFT.
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

The `train` group is Linux-specific. It pins the Granite/Unsloth stack in `pyproject.toml` and `uv.lock`; keep `--group train` on GPU commands.

## 2. Freeze evaluation examples

Generate and preserve a 50-example CNN/DailyMail 3.0.0 test pilot and a disjoint 250-example test sample. Reservoir sampling scans the test split; it can take time the first time datasets are downloaded. Preserve these exact JSONL files and their hashes for all model runs. The command refuses to overwrite existing samples without `--overwrite`.

```bash
uv run --locked lillm-summarizer prepare-cnn-dm \
  --count 50 --seed 20261028 \
  --output data/eval/cnn_dailymail_test_pilot.jsonl

uv run --locked lillm-summarizer prepare-cnn-dm \
  --count 250 --seed 20261029 \
  --exclude-input data/eval/cnn_dailymail_test_pilot.jsonl \
  --output data/eval/cnn_dailymail_test_confirmation.jsonl
```

These are test examples. Never use them as training targets. The prepared files are ignored by Git because they contain the dataset text; retain them locally for exact sample reuse.

## 3. Prepare public summarization training data

The preparation command streams the public SmolTalk2 `SFT` config, split `smoltalk_smollm3_smol_summarize_no_think`, resolves and records the current dataset revision unless one is explicitly provided, normalizes examples to the project's fixed summarization instruction, removes exact duplicate documents, and filters 8-gram overlap against the frozen evaluation documents.

```bash
uv run --locked lillm-summarizer prepare-smoltalk
```

The command expects both frozen evaluation files at their default paths. Its JSONL and `.meta.json` sidecar are written under ignored `data/train/`. For exact future repetition, keep the metadata's dataset revision, the generated JSONL, and its SHA-256. An optional `--dataset-revision <hub-commit>` pins a specific Hub revision. To perform a small pipeline-only preparation, use `--limit`; do not interpret that as a quality training run.

Create a deterministic stratified training/validation subset (defaults: 24,000 train, 1,000 validation; seed 20261128):

```bash
uv run --locked lillm-summarizer subset-training
```

The split is over source-summary tag and document-length bands. Keep training and validation inputs fixed for a run; do not use the final CNN/DailyMail evaluation samples for validation.

## 4. Train Granite 4.0-1B LoRA

First inspect the trainer flags and start with a small smoke run if the GPU/model path is new:

```bash
uv run --locked --group train python scripts/train_smoke.py --help
uv run --locked --group train python scripts/train_smoke.py \
  --adapter-dir adapters/granite4-1b-smoke
```

Then run the full default training recipe (3 epochs, validation loss and early stopping, best-checkpoint restore):

```bash
uv run --locked --group train python scripts/train_lora.py \
  --model ibm-granite/granite-4.0-1b \
  --adapter-dir adapters/granite4-1b-public-smoltalk
```

Default input paths are the stratified public SmolTalk2 train/validation files. Training verifies the prepared data and completion-only supervision before loading the GPU training stack. Review saved `train_meta.json`, validation metrics, applied LoRA targets, and model revision before interpreting the run. Lower validation loss alone does not show improved factuality or summary quality.

For a bounded early-training check, use `--eval-steps N --stop-after-steps M` where `M` is divisible by `N`. This is a diagnostic run, not equivalent to completing the epoch schedule. For architecture-specific parameters, safety checks, and limitations, inspect `scripts/train_lora.py` and the [Granite audit](../results/granite-training-audit.md).

## 5. Evaluate and report

For a like-for-like comparison, evaluate base and adapter against the same frozen evaluation JSONL, prompt, decoding settings, and output budget. Save predictions and metadata under `runs/` (ignored). Score ROUGE and use the source-grounded judge only with data you are permitted to send to OpenRouter.

The project CLI exposes `run`, `score`, and `judge`; inspect current arguments with:

```bash
uv run --locked lillm-summarizer run --help
uv run --locked lillm-summarizer score --help
uv run --locked lillm-summarizer judge --help
```

ROUGE measures reference overlap, not factuality. Report all six judge dimensions, output lengths, cap/stop behavior, and example-level weaknesses. Report the exact model and dataset revisions, prompts, quantization, inference server, sampler, seeds, and evaluation-file hashes. The results in the root README summarize previously completed runs; reruns may differ because upstream models, judging services, data revisions, and hardware/runtime behavior can change.

## Local-only output policy

`data/eval/*.jsonl`, `data/train/*.jsonl`, `runs/`, `models/`, and `adapters/` are Git-ignored. Keep private or restricted data out of public issue trackers, cloud shares, model uploads, and hosted judge requests unless you have explicit authorization.
