---
language:
  - en
license: apache-2.0
base_model: ibm-granite/granite-4.0-350m
library_name: peft
pipeline_tag: text-generation
tags:
  - granite
  - summarization
  - lora
  - peft
---

# Granite 4.0-350M Summarization LoRA

An exploratory rank-16 LoRA adapter for English web-page summarization, trained from **ibm-granite/granite-4.0-350m**. This is a PEFT adapter, not a standalone model; load it with the base model.

## Evaluation snapshot

Matched Q4-base/BF16-sidecar evaluation on the 250-article CNN/DailyMail confirmation set, using the shared `summary-v3` instruction:

| Metric | Base | Base + adapter | Change |
| --- | ---: | ---: | ---: |
| Factuality (judge, 1–5) | 3.620 | 2.744 | -0.876 (95% paired CI: -1.044 to -0.712) |
| Coverage (judge, 1–5) | 3.256 | 3.296 | +0.040 (95% paired CI: -0.060 to +0.140) |
| Overall quality (judge, 1–5) | 3.124 | 2.736 | -0.388 (95% paired CI: -0.520 to -0.252) |
| ROUGE-1 | 0.3083 | 0.3044 | -0.0040 (95% paired CI: -0.0140 to +0.0069) |
| Mean words | 179.4 | 163.8 | -15.6 |

The outcome is **mixed across metrics and guardrails**: the adapter produced fewer cap/length-stop and automatic degeneration flags than its base, while factuality, relevance, readability, and overall judge scores were lower in this comparison. It did not pass the project's automatic quality-acceptance policy. It is shared for experimentation, not as a quality upgrade or deployment recommendation. Full paired results and limitations are documented in Lillm Summarizer at `docs/results/granite-v3-adapters.md`.

## Training and data

LoRA rank 16, alpha 16, dropout 0; evaluated artifact `granite4-350m-summary-v3-8k-s42-lr5e5/checkpoints/checkpoint-4456`. Training combined public SmolTalk2 summarization examples with the author's private web-summarization data. Private records and derived training files are not included here. Training used one seed; the adapter is not an exhaustive checkpoint or seed search.

## Use

Load the adapter with PEFT and the base Transformers model. The published comparison itself used a Q4_K_XL GGUF base plus a separately exported BF16 LoRA sidecar, so native Transformers generations may differ.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "ibm-granite/granite-4.0-350m"
adapter_id = "nchapman/granite-4.0-350m-summarization-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
```

Use the base model's tokenizer/chat template. The adapter does not bundle a tokenizer or base-model weights.

## Licensing

Adapter weights are released under Apache-2.0. The base model retains its own license and terms; consult [Granite 4.0-350M](https://huggingface.co/ibm-granite/granite-4.0-350m) before use.
