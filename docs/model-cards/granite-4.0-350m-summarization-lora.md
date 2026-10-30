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

Fresh aligned 250-article CNN/DailyMail comparison using Q4 base inference, a BF16 adapter sidecar, common greedy `summary-v3` generation, and the same pinned judge used for all four base/adapter pairs:

| Metric | Base | Base + adapter | Difference [95% paired CI] |
| --- | ---: | ---: | ---: |
| Factuality (judge, 1–5) | 3.636 | 2.756 | -0.880 [-1.044, -0.712] |
| Relevance (judge, 1–5) | 3.836 | 3.548 | -0.288 [-0.396, -0.176] |
| Readability (judge, 1–5) | 3.712 | 3.508 | -0.204 [-0.332, -0.064] |
| Overall quality (judge, 1–5) | 3.148 | 2.736 | -0.412 [-0.548, -0.276] |
| ROUGE-1 | 0.3083 | 0.3044 | -0.0040 [-0.0140, +0.0069] |
| Mean words | 179.4 | 163.8 | -15.6 [-37.3, +4.1] |
| 1,024-token cap stops | 11 | 3 | -8 |

The result is **mixed across metrics**: cap stops decreased, but factuality, relevance, readability, and overall quality were lower; reference-overlap changes were small. It is shared for experimentation, not as a quality upgrade or deployment recommendation. Full dimensions, protocol, intervals, and limitations: `docs/results/aligned-adapter-comparison.md` in the Lillm Summarizer repository. This aligned result uses a common pinned judge and supersedes the earlier Granite snapshot's mutable judge alias.

## Training and data

LoRA rank 16, alpha 16, dropout 0; evaluated artifact `granite4-350m-summary-v3-8k-s42-lr5e5/checkpoints/checkpoint-4456`. The training examples pair source documents with target summaries. The public component came from the `HuggingFaceTB/smoltalk2` SFT summarization split, which includes one-line and three-sentence summary styles; the mixed runs also used the author's private web-document/summary pairs.

The adaptation goal is to reinforce a shared summarization response contract—objective plain prose, brief when the source is simple and more detailed when it warrants it—across model sizes and families. This goal is not yet established as a cross-model normalization result: no formal style-variance metric was run, and the two Granite adapters show mixed, model-dependent quality outcomes. Private records and derived training files are not included here. Training used one seed; the adapter is not an exhaustive checkpoint or seed search.

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
