---
language:
  - en
license: apache-2.0
base_model: ibm-granite/granite-4.0-1b
library_name: peft
pipeline_tag: text-generation
tags:
  - granite
  - summarization
  - lora
  - peft
---

# Granite 4.0-1B Summarization LoRA

An exploratory rank-16 LoRA adapter for English web-page summarization, trained from **ibm-granite/granite-4.0-1b**. This is a PEFT adapter, not a standalone model; load it with the base model.

## Evaluation snapshot

Matched Q4-base/BF16-sidecar evaluation on the 250-article CNN/DailyMail confirmation set, using the shared `summary-v3` instruction:

| Metric | Base | Base + adapter | Change |
| --- | ---: | ---: | ---: |
| Factuality (judge, 1–5) | 4.288 | 4.308 | +0.020 (95% paired CI: -0.116 to +0.156) |
| Coverage (judge, 1–5) | 4.068 | 3.836 | -0.232 (95% paired CI: -0.312 to -0.156) |
| Overall quality (judge, 1–5) | 4.024 | 4.004 | -0.020 (95% paired CI: -0.128 to +0.080) |
| ROUGE-1 | 0.3219 | 0.3454 | +0.0235 (95% paired CI: +0.0150 to +0.0323) |
| Mean words | 154.9 | 141.9 | -12.9 |

The outcome is **mixed**: reference overlap and concision increased; coverage decreased; factuality and overall-quality differences were inconclusive. A few adapter outputs triggered automatic cap/length-stop/degeneration guardrails, so it did not pass the project's automatic acceptance policy. It is shared as an experiment, not a quality upgrade or deployment recommendation. Full results, all six judged dimensions, intervals, and limitations are documented in Lillm Summarizer at `docs/results/granite-v3-adapters.md`.

## Training and data

LoRA rank 16, alpha 16, dropout 0; evaluated artifact `granite4-1b-summary-v3-8k-s42-early512/checkpoints/checkpoint-128`. The training examples pair source documents with target summaries. The public component came from the `HuggingFaceTB/smoltalk2` SFT summarization split, which includes one-line and three-sentence summary styles; the mixed runs also used the author's private web-document/summary pairs.

The adaptation goal is to reinforce a shared summarization response contract—objective plain prose, brief when the source is simple and more detailed when it warrants it—across model sizes and families. This goal is not yet established as a cross-model normalization result: no formal style-variance metric was run, and the two Granite adapters show mixed, model-dependent quality outcomes. Private records and derived training files are not included here. Training used one seed; the adapter is not an exhaustive checkpoint or seed search.

## Use

Load the adapter with PEFT and the base Transformers model. The published comparison itself used a Q4_K_XL GGUF base plus a separately exported BF16 LoRA sidecar, so native Transformers generations may differ.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "ibm-granite/granite-4.0-1b"
adapter_id = "nchapman/granite-4.0-1b-summarization-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
```

Use the base model's tokenizer/chat template. The adapter does not bundle a tokenizer or base-model weights.

## Licensing

Adapter weights are released under Apache-2.0. The base model retains its own license and terms; consult [Granite 4.0-1B](https://huggingface.co/ibm-granite/granite-4.0-1b) before use.
