# Tiny-model summarization evaluation and LoRA plan

## Goal

Compare the quality of openly available language models with at most 2B total parameters on a defined summarization task, then fine-tune comparable LoRA adapters and measure which model benefits most. Keep base-model quality, training gains, inference cost, and practical compatibility as separate results.

## Initial model shortlist

Use the **instruction-tuned checkpoints** for the first baseline: this tests useful summarization behavior out of the box and gives us a sensible starting point for task-specific LoRA. Record exact Hub revision, tokenizer/template, license, and checkpoint variant before running anything.

| Model | Approx. size | Candidate checkpoint | Initial notes |
| --- | ---: | --- | --- |
| Qwen3.5 | 0.8B (about 0.9B parameters) | `Qwen/Qwen3.5-0.8B` | Apache-2.0; include as the smallest Qwen point. |
| Qwen3.5 | 2B | `Qwen/Qwen3.5-2B` | Apache-2.0; tests whether the size increase is worth it. |
| MiniCPM5 | 1B | `openbmb/MiniCPM5-1B` | Apache-2.0; model card describes a standard Llama causal-LM architecture. |
| LFM2.5 | 1.2B | `LiquidAI/LFM2.5-1.2B-Instruct` | Promising speed-oriented candidate; verify its license terms and PEFT/LoRA training path before making it a core training comparison. |
| Granite 4.0 | 1B | `ibm-granite/granite-4.0-1b` | Apache-2.0; IBM distinguishes this standard architecture from its hybrid Mamba variant, specifically noting PEFT/llama.cpp compatibility as a reason to use the standard model. |

Sizes are model-family labels / approximate published counts, not a claim that every checkpoint has precisely that number of total parameters. The shortlist intentionally uses the standard Granite 4.0 1B, not `granite-4.0-h-1b` (a 1.5B hybrid model). If the strict cap means exactly 2.000B parameters rather than the published 2B class, verify model config counts before final inclusion.

## Phases

### 0. Lock the task and constraints

Before scoring, define the real target. If there is no specific application yet, start with **English, single-document news summarization** and add a second domain only after the first experiment is stable.

Record:
- Input type and realistic input-length range.
- Desired summary length and format (one paragraph, bullets, etc.).
- Whether the summary is extractive, abstractive, or either.
- Language(s), latency target, hardware/memory limits, and whether deployment is local.
- Whether fine-tuning starts from each model's instruct checkpoint (recommended for this experiment) or its base checkpoint. Do not mix the two silently.

### 0A. Choose and freeze the evaluation set

This is a dedicated dataset-design step, separate from building the inference harness and separate from sourcing training examples. Start from the use case, then select data that represents its documents, lengths, language, and desired summary style. For a general English news pilot, compare candidates such as CNN/DailyMail (moderate-length articles and multi-sentence highlights) and XSum (short, highly abstractive summaries with a greater factuality challenge); do not assume either is representative if the intended domain is different.

Before scoring models:
- Inspect dataset cards, licensing/usage constraints, available splits, examples, source/target lengths, formatting quirks, and known contamination or duplication risks.
- Decide whether to use an existing held-out test split or create a held-out set from permitted data. Keep a final test set sealed until prompt/model choices are frozen; use validation data for iteration.
- Sample a manageable but representative pilot (20–50 examples), then a scored sample (roughly 500–1,000 if available) stratified by input length and relevant subdomains. Preserve IDs and exact source/reference text.
- Set max input and output token budgets before the run. If documents exceed the input budget, define and record a truncation/chunking policy instead of silently clipping them.
- Define metrics and human-review rubric before viewing model rankings. Keep the evaluation data and references out of training and training-data generation; run deduplication/leakage checks where feasible.

### 0B. Identify or create the training data

Treat training-data design as its own workstream; do not simply train on the evaluation benchmark's test examples or assume that its training split matches the intended task. Decide whether existing public summarization data is adequate, whether we need permitted in-domain examples, or whether to create a small human-written/curated dataset. Synthetic examples can be explored later, but should be labeled and evaluated separately from human-authored targets.

For each candidate source or newly created dataset:
- Check license/terms, privacy, provenance, and permission to use documents and summaries for model training.
- Define what a good target summary looks like (length, format, coverage, factuality, style) and use the same instruction/output contract across models.
- Inspect examples for quality, duplicates, leakage from validation/test sets, noisy or inconsistent targets, and coverage across document length/topic. Keep source IDs and split assignments.
- Create distinct train, validation, and final test partitions by document/source (and preferably time or publisher when relevant), not by near-duplicate rows. Reserve final evaluation data before training.
- Begin with a small curated pilot to test the pipeline; scale only after the data and targets look sound. Record dataset versions, filtering, transformations, and any synthetic-data provenance.

### 1. Make a reproducible inference harness

Use **Unsloth for LoRA/SFT training** (with its TRL/PEFT-based workflow) and **llama.cpp for local GGUF inference/deployment** as the initial division of labor. Current Unsloth guidance covers Qwen3.5 0.8B/2B, LFM2.5-1.2B, MiniCPM5-1B, and Granite 4.0-1B, so it is a strong common starting point. Still smoke-test every model and pin a known-good package stack: architecture and Transformers-version support can differ. Save native adapters as the reference artifact, export each trained model/adapter to GGUF only where supported, and compare outputs with the HF/Unsloth path before trusting the conversion.

The harness should:
- Load pinned model and tokenizer revisions, and preserve each model's native chat template.
- Expose one common summarization prompt and generation contract.
- Save input, rendered prompt, output, token counts, stop reason, timing, model/revision, quantization, and generation settings per example.
- Write predictions incrementally, keep partial outputs inspectable, and mark run metadata as running/failed/complete so interrupted batches are distinguishable; add automatic resume only if pilot runs justify it.
- Keep raw generations immutable so metrics can be recalculated later.

For a fair quality baseline, use BF16/FP16 where the available hardware permits. If memory requires quantization, use one declared quantization level for all models (for example Q8 first, or Q4_K_M for an explicitly deployment-oriented comparison) and do not compare a quantized model against full precision without labeling the difference. Start with greedy/temperature 0 for repeatability; later test a small sampling sweep only if it reflects the intended deployment.

### 2. Baseline evaluation (no training)

Use a fixed, held-out set with examples sampled across input lengths and difficulty. Begin with a small smoke set (20–50 items), then a scored set of roughly 500–1,000 items once prompts and generation limits are stable.

For the default news task, CNN/DailyMail is a practical first benchmark; use its summarization version/splits and do not train on validation/test material. Add XSum later if we want a more aggressively abstractive, higher factuality-risk test. GovReport is a useful long-document follow-up, but its very long inputs and summaries are a separate capability regime, not a drop-in replacement for the first benchmark.

Report at least:
- ROUGE-1/2/L (with tokenizer/normalization settings stated) for reference overlap, not as the sole quality measure.
- A factuality/consistency measure and a blinded human preference check on a smaller sample. Judge factual support against the source, coverage of key information, concision/relevance, and readability.
- Length, latency, throughput, and peak memory alongside quality.
- Per-example and aggregate results, with confidence intervals or paired bootstrap comparisons where practical.

Do not use benchmark reference summaries as training targets and then present those same items as held-out results.

### 3. Check compatibility and choose the training recipe

Use Unsloth as the first training implementation. Run one adapter-training smoke test on a few dozen examples for every finalist before building the full pipeline. Confirm loss decreases, adapter saves/reloads, and outputs are valid. Pin Unsloth, Transformers, TRL, PEFT, and PyTorch versions after the smoke tests; this is especially important for newer/non-standard architectures such as LFM2.5 and multimodal-capable Qwen3.5 checkpoints.

Use a shared, documented recipe as far as architectures allow: same examples/splits, prompt format, target summary convention, optimizer family, effective batch size, token budget, epoch/early-stop policy, and evaluation procedure. A reasonable pilot is rank 8 or 16 LoRA, with learning rate and target modules verified per architecture; do not assume every model uses the same module names or supports identical adapter injection. Train on the instruct checkpoints if the purpose is to improve an already usable summarizer. Keep a no-adapter baseline for every model.

### 4. LoRA pilot and full comparison

Start with one small pilot dataset and one validation set. Tune only a limited number of recipe choices on validation, then freeze the recipe before the final held-out comparison. Save configuration, seeds, data version, checkpoints, and logs.

For each model compare:
1. Original instruct checkpoint.
2. LoRA-adapted checkpoint under the same prompt / output policy.
3. Optionally merged or quantized deployment artifact, measured separately from the high-precision training comparison.

Use at least a few training seeds for finalists if the differences are small. Measure both absolute post-training quality and gain over that model's own baseline; the best base model and the best LoRA beneficiary may differ.

### 5. Human review and decision

Blind and randomize model identities for paired human review on a manageable sample (for example 100 inputs). Use a short rubric: factuality, coverage, concision, clarity, and overall preference. Inspect factuality failures and regressions, not only averages. Select winners using a pre-agreed quality/cost criterion rather than a single ROUGE score.

### 6. Package results and iterate

Keep a reproducible results table and per-example outputs. Document known limitations, licenses, model revisions, inference/training stacks, and conversion status. Only after the first run is trusted should we add more datasets, multilingual tasks, quantization variants, or hyperparameter sweeps.

## Early technical recommendation

- **Inference/evaluation:** llama.cpp with pinned GGUFs and an OpenAI-compatible local server is a good practical fit, provided every selected model has a verified GGUF implementation and correct chat template. For initial architecture debugging, Transformers may be simpler for checkpoints without mature GGUF support.
- **LoRA training:** start with Unsloth's SFT workflow (TRL/PEFT underneath); retain the native adapter as the reference artifact. Convert and test llama.cpp GGUF adapters as a deployment/compatibility step, not as the only evaluation path.
- **Primary risk:** support parity and version constraints differ by architecture. Prove model loading, LoRA insertion, save/reload, and inference for each candidate before committing to a shared training run.

## Source notes (checked 2026-10-28)

- Qwen3.5 0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B (Apache-2.0; Hub labels it 0.9B parameters)
- Qwen3.5 2B: https://huggingface.co/Qwen/Qwen3.5-2B (Apache-2.0)
- MiniCPM5-1B: https://huggingface.co/openbmb/MiniCPM5-1B (Apache-2.0; 1,080,632,832 parameters; standard `LlamaForCausalLM`)
- LFM2.5-1.2B-Instruct: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct and GGUF variant https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF (license shown as LFM-1.0; review terms and training support)
- Granite 4.0 model family: https://www.ibm.com/granite/docs/models/granite4-0 (standard 1B recommended where Mamba2 / PEFT / llama.cpp support is a concern)
- Granite 4.0 1B instruct: https://huggingface.co/ibm-granite/granite-4.0-1b (Apache-2.0)
- CNN/DailyMail dataset card: https://huggingface.co/datasets/abisee/cnn_dailymail
- GovReport dataset card: https://huggingface.co/datasets/ccdv/govreport-summarization
- SummEval evaluation dimensions and metric limitations: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00373/100686/SummEval-Re-evaluating-Summarization-Evaluation
- llama.cpp server adapter support: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Unsloth Qwen3.5 fine-tuning: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Unsloth LFM2.5 fine-tuning: https://unsloth.ai/docs/models/tutorials/lfm2.5
- MiniCPM5 Unsloth fine-tuning guide: https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/unsloth.md
- Granite 4.0 1B on Unsloth: https://huggingface.co/unsloth/granite-4.0-1b
- Unsloth LoRA training guide: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
