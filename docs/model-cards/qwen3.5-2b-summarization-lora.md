---
language:
  - en
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
pipeline_tag: text-generation
tags:
  - qwen3.5
  - summarization
  - lora
  - peft
---

# Qwen3.5-2B Summarization LoRA

An experimental rank-16 LoRA adapter for English web-page summarization, trained from **Qwen/Qwen3.5-2B**. This is a PEFT adapter, not a standalone model; load it with the base model.

## Evaluation snapshot

On a matched 250-article CNN/DailyMail test comparison using Q4 base inference, a BF16 adapter sidecar, and the shared `summary-v3` prompt:

| Metric | Base | Base + adapter | Change |
| --- | ---: | ---: | ---: |
| Factuality (judge, 1–5) | 1.948 | 2.448 | +0.500 |
| Overall quality (judge, 1–5) | 1.956 | 2.336 | +0.380 |
| Coverage (judge, 1–5) | 2.864 | 2.900 | +0.036 |
| ROUGE-1 | 0.1977 | 0.3010 | +0.1033 |
| Mean words | 329.1 | 157.0 | -172.2 |

These are promising but bounded comparative results: one training seed, one selected early checkpoint, one public news dataset, and one automated judge. Occasional factual errors and repetition remain. The analysis guardrail marked the candidate **not promoted** because of an automatic degeneration flag and one output reaching the length limit; these results do not qualify the adapter as deployment-ready. The complete protocol and paired intervals are documented in Lillm Summarizer at `docs/results/qwen-adapters.md`.

## Training and data

LoRA rank 16, alpha 16, dropout 0; evaluated checkpoint 128 from a 512-update run. The training examples pair a source document with its target summary. The public component came from the `HuggingFaceTB/smoltalk2` SFT summarization split, which includes one-line and three-sentence summary styles; the mixed runs also used the author's private web-document/summary pairs. Both Qwen sizes used the same frozen prepared pool and train/validation assignments.

The adaptation goal is to reinforce a shared summarization response contract—objective plain prose, brief when the source is simple and more detailed when it warrants it—across different model sizes. This is an intended direction, not a proven normalization result: no formal cross-model style-variance metric was run, and output lengths and quality changes are model-dependent. Private records, derived training files, and the training runner are not included in this repository.

## Use

Requires a Transformers release supporting Qwen3.5 and PEFT. The adapter was trained/evaluated in a Transformers 5.2.0 environment. This minimal loading example uses the native base model, rather than the Q4 GGUF representation used for the published comparison:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base_id = "Qwen/Qwen3.5-2B"
adapter_id = "nchapman/qwen3.5-2b-summarization-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForImageTextToText.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_id)
```

Use the base model's chat template and provide the source document as user content. Prompt and decoding choices affect results; see the full report before comparing generations.

## Licensing

Adapter weights are released under Apache-2.0. The base model retains its own license and terms; consult [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) before use.
