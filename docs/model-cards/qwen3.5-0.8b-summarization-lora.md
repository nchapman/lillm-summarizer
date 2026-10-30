---
language:
  - en
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
pipeline_tag: text-generation
tags:
  - qwen3.5
  - summarization
  - lora
  - peft
---

# Qwen3.5-0.8B Summarization LoRA

An experimental rank-16 LoRA adapter for English web-page summarization, trained from **Qwen/Qwen3.5-0.8B**. This is a PEFT adapter, not a standalone model; load it with the base model.

## Evaluation snapshot

On a matched 250-article CNN/DailyMail test comparison using Q4 base inference, a BF16 adapter sidecar, and the shared `summary-v3` prompt:

| Metric | Base | Base + adapter | Change |
| --- | ---: | ---: | ---: |
| Factuality (judge, 1–5) | 1.596 | 1.976 | +0.380 |
| Overall quality (judge, 1–5) | 1.600 | 1.924 | +0.324 |
| Coverage (judge, 1–5) | 2.436 | 2.476 | +0.040 |
| ROUGE-1 | 0.2129 | 0.3139 | +0.1011 |
| Mean words | 285.2 | 134.3 | -150.9 |

These are promising but bounded comparative results: one training seed, one selected early checkpoint, one public news dataset, and one automated judge. Occasional repetition and factual errors remain. The analysis guardrail marked the candidate **not promoted** due to one automatic degeneration flag; this does not erase the measured gains or qualify the adapter as deployment-ready. The complete protocol and paired intervals are documented in Lillm Summarizer at `docs/results/qwen-adapters.md`.

## Training and data

LoRA rank 16, alpha 16, dropout 0; evaluated checkpoint 128 from a 512-update run. Training combined public HuggingFaceTB/smoltalk2 summarization examples with the author's private web-summarization data. Private records, derived training files, and the training runner are not included in this repository.

## Use

Requires a Transformers release supporting Qwen3.5 and PEFT. The adapter was trained/evaluated in a Transformers 5.2.0 environment. This minimal loading example uses the native base model, rather than the Q4 GGUF representation used for the published comparison:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base_id = "Qwen/Qwen3.5-0.8B"
adapter_id = "nchapman/qwen3.5-0.8b-summarization-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForImageTextToText.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_id)
```

Use the base model's chat template and provide the source document as user content. Prompt and decoding choices affect results; see the full report before comparing generations.

## Licensing

Adapter weights are released under Apache-2.0. The base model retains its own license and terms; consult [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) before use.
