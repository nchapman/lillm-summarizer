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

Fresh aligned 250-article CNN/DailyMail comparison using Q4 base inference, a BF16 adapter sidecar, common greedy `summary-v3` generation, and the same pinned judge used for all four base/adapter pairs:

| Metric | Base | Base + adapter | Difference [95% paired CI] |
| --- | ---: | ---: | ---: |
| Factuality (judge, 1–5) | 4.304 | 4.328 | +0.024 [-0.112, +0.156] |
| Coverage (judge, 1–5) | 4.052 | 3.808 | -0.244 [-0.324, -0.168] |
| Concision (judge, 1–5) | 3.936 | 4.116 | +0.180 [+0.096, +0.260] |
| Overall quality (judge, 1–5) | 4.036 | 4.000 | -0.036 [-0.140, +0.068] |
| ROUGE-1 | 0.3219 | 0.3454 | +0.0235 [+0.0150, +0.0323] |
| Mean words | 154.9 | 141.9 | -12.9 [-24.3, +0.2] |
| 1,024-token cap stops | 0 | 2 | +2 |

The result is **mixed**: reference overlap and concision increased, coverage decreased, and factuality/overall-quality changes were inconclusive. Cap stops rose from zero to two. It is an exploratory adapter, not a quality upgrade or deployment recommendation. Full dimensions, protocol, intervals, and limitations: `docs/results/aligned-adapter-comparison.md` in the Lillm Summarizer repository. This shared-prompt result uses a common pinned judge and supersedes the earlier Granite snapshot's mutable judge alias.

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
