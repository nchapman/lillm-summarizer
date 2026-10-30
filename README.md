# Lillm Summarizer

**How well do small language models summarize out of the box, and how much can task-specific training help?** This project compares small instruction-tuned language models under practical inference settings, trains LoRA adapters, audits the supervision path, and measures factuality, coverage, relevance, concision, readability, overall quality, and reference overlap.

> **Bottom line:** the strongest untuned local model in the original multi-model comparison was Granite 4.0-1B. Qwen3.5-0.8B and Qwen3.5-2B both improved on the matched Qwen adapter experiment. Granite's reported epoch-1 adapter, by contrast, did not improve quality under its separate earlier evaluation setup. These experiments use different prompts, budgets, datasets, and training data; do not combine them into one leaderboard.

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

### Qwen3.5: measured improvement

On the larger 250-article battery, both Qwen adapters improved their own base models under matched Q4 inference, a fresh shared prompt (`summary-v3`), and the same generation settings. Ratings are means on the 1–5 judge scale. The experiment used one training seed and bounded early-training checkpoints; public and private training data were mixed.

| Model | Factuality: base → adapter | Overall quality: base → adapter | Mean words: base → adapter |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 1.596 → **1.976** | 1.600 → **1.924** | 285.2 → **134.3** |
| Qwen3.5-2B | 1.948 → **2.448** | 1.956 → **2.336** | 329.1 → **157.0** |

Factuality, relevance, concision, overall quality, and ROUGE improved; coverage was approximately unchanged. The shorter summaries still contain factual errors and occasional repetition. Public-only data has **not** been used to reproduce these adapter results. Private records and the Qwen-specific training runner/environment are not included, so the reported mixed-data training experiment is not exactly reproducible from this checkout.

Full protocol, all six judge dimensions, paired intervals, training correctness checks, and limitations: [Qwen adapter results](docs/results/qwen-adapters.md).

### Granite 4.0-1B: an earlier adapter milestone, not an improvement

The separately reported Granite epoch-1 experiment used an earlier prompt and a 256-token output cap. On its 250-example set, the adapter raised coverage but reduced factuality (4.324 → 4.048), overall quality (4.084 → 3.920), and ROUGE-L (0.2201 → 0.2007); mean output length increased from 115.9 to 149.7 words. This was an end-to-end training/export/serving milestone before training was finished, not a final verdict on the model family.

[Granite adapter evaluation](docs/results/granite-epoch1-adapter.md) · [Granite training and deployment audit](docs/results/granite-training-audit.md)

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
