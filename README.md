# Lillm Summarizer

**How well do small language models summarize out of the box, and how much can task-specific training help?** This project compares small instruction-tuned language models under practical inference settings, trains LoRA adapters, audits the supervision path, and measures factuality, coverage, relevance, concision, readability, overall quality, and reference overlap.

> **Bottom line:** Granite 4.0-1B was the strongest untuned local model in the original seven-model comparison. In the aligned adapter comparison, Qwen3.5-0.8B improved on several judged measures, Qwen3.5-2B's overall-quality change was inconclusive, Granite 4.0-1B showed mixed results, and Granite 4.0-350M declined on several measures. Both comparisons use the **same frozen 250 CNN/DailyMail articles** (SHA-256 `047f62c2…1377e75e`), but the historical baseline used `summary-v1`, a 256-token cap, and model-specific decoding, while the aligned comparison uses `summary-v3`, a 1,024-token cap, common greedy decoding, and one pinned judge; keep them separate. Making response behavior more consistent is a goal, not a demonstrated result.

## Results: models without adapters

The table below reports the 250-example CNN/DailyMail test-set comparison. Local models used quantized GGUFs with model-recommended sampling; all used the same historical prompt and 256-token output cap. GPT-6 Luna is a hosted control, not part of the local small-model cohort. Judge ratings are means on a 1–5 scale; ROUGE is reference overlap, not factuality.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Factuality | Overall quality | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 0.315 | 0.077 | 0.172 | 1.81 | 1.80 | 0.714 s |
| Qwen3.5-2B | 0.330 | 0.092 | 0.185 | 2.57 | 2.58 | 1.164 s |
| MiniCPM5-1B | 0.327 | 0.095 | 0.196 | 2.52 | 2.44 | 0.426 s |
| LFM2.5-1.2B | 0.344 | 0.101 | 0.211 | 3.98 | 3.89 | 0.469 s |
| **Granite 4.0-1B** | **0.404** | 0.156 | **0.258** | **4.53** | **4.13** | 0.725 s |
| Granite 4.0-350M | 0.376 | 0.149 | 0.245 | 3.77 | 3.33 | **0.341 s** |
| GPT-6 Luna (hosted) | 0.375 | 0.123 | 0.229 | 4.96 | 4.91 | 1.906 s |

Granite 4.0-1B led the local cohort on ROUGE and judge factuality/overall quality. Granite 4.0-350M was the fastest local model, with close ROUGE but lower judged quality and 11/250 outputs stopping at the token cap. Judge ratings are automated, and the high hosted-control scores show a ceiling effect. The comparison is a modest sample, not a universal model ranking.

Details, full confidence intervals, and the 350M cap sensitivity experiment: [baseline confirmation results](docs/results/baseline-confirmation.md). The historical 50-example pilot and Q8 setup are not part of the final comparison and are not included.

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
