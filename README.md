# Lillm Summarizer

**How well do small language models summarize out of the box, and how much can task-specific training help?** This project compares small instruction-tuned language models under practical inference settings, trains LoRA adapters, audits the supervision path, and measures factuality, coverage, relevance, concision, readability, overall quality, and reference overlap.

> **Bottom line:** Six local base models and four adapters were evaluated on the same 250 articles with `summary-v3`, a 1,024-token cap, greedy decoding, and one pinned judge. Granite 4.0-1B led the local base models; adapter effects varied by model. These results do not establish response-style normalization.

## Results: models without adapters

The six local models below were evaluated on the same frozen 250-example CNN/DailyMail test sample as the adapter comparison, with the `summary-v3` prompt, a 1,024-token output ceiling, temperature 0 (greedy decoding), and the same pinned judge. Judge ratings are means on a 1–5 scale; ROUGE is reference overlap, not factuality. The hosted GPT-6 Luna control is not included in this aligned local table; its earlier, differently configured result is retained in the historical report.

| Local base model | ROUGE-1 | ROUGE-2 | ROUGE-L | Factuality | Overall quality | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 0.220 | 0.086 | 0.140 | 3.30 | 3.18 | 1.688 s |
| Qwen3.5-2B | 0.217 | 0.086 | 0.137 | 3.58 | 3.55 | 3.182 s |
| MiniCPM5-1B | 0.199 | 0.074 | 0.130 | 2.99 | 2.54 | 1.959 s |
| LFM2.5-1.2B | 0.293 | 0.091 | 0.178 | 3.66 | 3.52 | 0.821 s |
| **Granite 4.0-1B** | **0.322** | **0.130** | **0.206** | **4.30** | **4.04** | 1.437 s |
| Granite 4.0-350M | 0.308 | 0.126 | 0.204 | 3.64 | 3.15 | **0.649 s** |

Granite 4.0-1B led the local cohort on ROUGE and mean factuality/overall quality. Granite 4.0-350M was the fastest local model and had 11/250 outputs stop at the token cap; MiniCPM5 had 58 cap stops. Judge ratings are automated, and latency is specific to this host. This is a modest sample, not a universal model ranking.

Details, all six judge dimensions, confidence intervals, and protocol notes: [aligned local baseline comparison](docs/results/aligned-baseline-comparison.md). The earlier seven-model `summary-v1`/256-token comparison, hosted-control score, and cap-sensitivity experiment remain archived as [historical baseline results](docs/results/baseline-confirmation.md).

## Results: with our adapters

Four PEFT LoRA adapters and model cards are published on Hugging Face:

- [Qwen3.5-0.8B summarization LoRA](https://huggingface.co/nchapman/qwen3.5-0.8b-summarization-lora) · [card source](docs/model-cards/qwen3.5-0.8b-summarization-lora.md)
- [Qwen3.5-2B summarization LoRA](https://huggingface.co/nchapman/qwen3.5-2b-summarization-lora) · [card source](docs/model-cards/qwen3.5-2b-summarization-lora.md)
- [Granite 4.0-1B summarization LoRA](https://huggingface.co/nchapman/granite-4.0-1b-summarization-lora) · [card source](docs/model-cards/granite-4.0-1b-summarization-lora.md)
- [Granite 4.0-350M summarization LoRA](https://huggingface.co/nchapman/granite-4.0-350m-summarization-lora) · [card source](docs/model-cards/granite-4.0-350m-summarization-lora.md)

### What the adapters are trying to learn

Each adapter adds a small set of LoRA weights to its base model, trained on **document → summary** examples rather than updating the full model. The shared public component is the `HuggingFaceTB/smoltalk2` SFT summarization split, which includes both one-line and three-sentence summary examples. The published mixed-data runs also used the author's private web-summarization pairs. Training examples were prepared for summarization, with the source document as input and the summary as the target; the evaluation articles were kept separate from training.

The goal is to strengthen source-grounded summarization and bring different small models toward a more consistent response contract: objective plain prose, concise for simple material, and more detailed when the source warrants it. “Normalize” means a shared task and style expectation—not identical summaries or a forced common length. The aligned comparison below does not establish cross-model consistency; no formal style-variance metric has been run.

### Aligned comparison: 250 articles

All four base/adapter pairs use the same frozen CNN/DailyMail examples, `summary-v3` prompt, greedy decoding, 1,024-token output ceiling, and pinned judge. The Qwen generations were rerun; verified Granite generations were reused, then all eight candidates were judged under the same protocol. Scores are means; judge ratings use a 1–5 scale and ROUGE-L is reference overlap.

| Model | Factuality: base → adapter | Coverage: base → adapter | Overall: base → adapter | ROUGE-L: base → adapter | Mean words: base → adapter | Cap stops: base → adapter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 3.296 → **4.344** | 3.812 → 3.824 | 3.184 → **3.400** | 0.1400 → **0.1693** | 303.9 → 310.5 | 10 → 24 |
| Qwen3.5-2B | 3.576 → **4.520** | 4.148 → 4.024 | 3.548 → 3.596 | 0.1372 → **0.1599** | 310.6 → 342.4 | 3 → 29 |
| Granite 4.0-1B | 4.304 → 4.328 | 4.052 → 3.808 | 4.036 → 4.000 | 0.2056 → **0.2237** | 154.9 → 141.9 | 0 → 2 |
| Granite 4.0-350M | 3.636 → **2.756** | 3.260 → 3.284 | 3.148 → **2.736** | 0.2036 → 0.1956 | 179.4 → 163.8 | 11 → 3 |

The results are mixed. Qwen 0.8B improved on factuality, relevance, overall quality, and ROUGE, but its cap stops increased. Qwen 2B improved on factuality, relevance, and ROUGE, while coverage and readability declined, overall quality was inconclusive, and cap stops increased sharply. Granite 1B improved on ROUGE and concision while coverage fell; factuality and overall quality were inconclusive. Granite 350M declined on several judged dimensions, including factuality and overall quality, though cap stops decreased. These are within-model comparisons, not a cross-family ranking or evidence that all adapters improve their bases.

Full paired intervals, all six judge dimensions, generation and judge protocol, and limitations: [aligned adapter comparison](docs/results/aligned-adapter-comparison.md). Earlier family-specific results remain available as historical experiment records: [Qwen adapter report](docs/results/qwen-adapters.md) · [Granite v3 report](docs/results/granite-v3-adapters.md) · [earlier Granite evaluation](docs/results/granite-epoch1-adapter.md) · [Granite training and deployment audit](docs/results/granite-training-audit.md).

Public and private training data were mixed for the published adapters; the private web-document/summary records and model-specific training recipes are not included in this checkout. The adapters were not trained on an identical controlled corpus across model families, so differences between family outcomes cannot be attributed solely to model architecture or size.

## Reproduce the public-data experiment

The public recipe uses CNN/DailyMail test examples only for evaluation and `HuggingFaceTB/smoltalk2` summarization examples for training. It does not use the private dataset and must be treated as a **new public-only experiment**, not an exact reproduction of the mixed-data adapter results above.

See the step-by-step [public-data recipe](docs/recipes/public-data.md), including environment setup, frozen evaluation sampling, public training-data preparation, split, and Granite LoRA training commands.

## Private-data experiment

The private Folio import and preparation code is intentionally retained, but the export itself is **not distributed**. You need an authorized export containing `sft.jsonl.gz`, `audit.jsonl.gz`, and `report.json`; without it, that recipe cannot be run. Do not add private records, reports, metadata, hashes, or derived splits/adapters to Git.

See [private-data notes](docs/recipes/private-data.md) for the preserved entry points and safety constraints. The private workflow is separate from the public-only recipe.

## Repository map

- `src/lillm_eval/` — inference/evaluation CLI, data preparation, training-data checks, and validation utilities.
- `scripts/` — Granite LoRA trainer and training smoke test.
- `tests/` — automated unit and behavior tests.
- `docs/recipes/` — concise public-only and private-data workflows.
- `docs/results/` — full experiment reports, historical evidence, and audits.
- `docs/model-cards/` — the four released LoRA adapter cards.
- `data/`, `runs/`, `models/`, `adapters/` — local inputs/artifacts; ignored by Git.

## Setup and checks

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The default environment is CPU-friendly and omits GPU training dependencies.

```bash
uv sync --locked
uv run --locked lillm-summarizer --help
uv run --locked pytest -q
```

The training dependency group is Linux/NVIDIA-specific and pins a CUDA 13.0 stack. GPU/Unsloth training also requires a compatible NVIDIA driver, BF16-capable GPU, and sufficient memory. Setup and hardware checks are documented in the [public-data recipe](docs/recipes/public-data.md).

## Evaluation and limitations

The original model comparison used pinned quantized model files and a local llama.cpp server, while source-grounded LLM judgments used OpenRouter. The shared test examples are public and have been exposed in prior experiments; later reports do not claim a sealed benchmark. LLM-judge ratings are useful comparative signals, not human truth labels. A single news dataset, one training seed, prompt/model-specific decoding, quantization, and finite sample sizes limit generalization.

The project aims to make the process inspectable: preserve generated data samples and metadata locally, record model/runtime/sampler settings, verify completion-only labels and adapter save/reload paths, and report remaining weaknesses rather than presenting a single score as the whole story.
