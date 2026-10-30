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

Fresh aligned 250-article CNN/DailyMail comparison using Q4 base inference, a BF16 adapter sidecar, common greedy `summary-v3` generation, and the same pinned judge used for all four base/adapter pairs:

| Metric | Base | Base + adapter | Difference [95% paired CI] |
| --- | ---: | ---: | ---: |
| Factuality (judge, 1–5) | 3.576 | 4.520 | +0.944 [+0.808, +1.076] |
| Coverage (judge, 1–5) | 4.148 | 4.024 | -0.124 [-0.244, -0.004] |
| Readability (judge, 1–5) | 4.204 | 3.992 | -0.212 [-0.352, -0.080] |
| Overall quality (judge, 1–5) | 3.548 | 3.596 | +0.048 [-0.096, +0.184] |
| ROUGE-L | 0.1372 | 0.1599 | +0.0226 [+0.0137, +0.0316] |
| Mean words | 310.6 | 342.4 | +31.9 [+3.1, +61.0] |
| 1,024-token cap stops | 3 | 29 | +26 |

Factuality, relevance, and reference-overlap scores improved in this sample, but coverage and readability declined; overall quality was inconclusive. Cap stops increased from 3 to 29. This is a mixed result—not an unconditional quality or deployment claim—and is limited to one training seed/checkpoint and an exposed public news set. Full dimensions, protocol, intervals, and limitations: `docs/results/aligned-adapter-comparison.md` in the Lillm Summarizer repository. This aligned result supersedes the earlier Qwen evaluation snapshot, which used different decoding settings.

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
