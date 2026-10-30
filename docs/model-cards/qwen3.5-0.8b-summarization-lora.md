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

Fresh aligned 250-article CNN/DailyMail comparison using Q4 base inference, a BF16 adapter sidecar, common greedy `summary-v3` generation, and the same pinned judge used for all four base/adapter pairs:

| Metric | Base | Base + adapter | Difference [95% paired CI] |
| --- | ---: | ---: | ---: |
| Factuality (judge, 1–5) | 3.296 | 4.344 | +1.048 [+0.896, +1.204] |
| Relevance (judge, 1–5) | 3.832 | 4.148 | +0.316 [+0.200, +0.432] |
| Overall quality (judge, 1–5) | 3.184 | 3.400 | +0.216 [+0.076, +0.364] |
| Coverage (judge, 1–5) | 3.812 | 3.824 | +0.012 [-0.112, +0.136] |
| ROUGE-L | 0.1400 | 0.1693 | +0.0292 [+0.0207, +0.0376] |
| Mean words | 303.9 | 310.5 | +6.6 [-22.9, +36.6] |
| 1,024-token cap stops | 10 | 24 | +14 |

Factuality, relevance, overall quality, and reference-overlap scores improved in this sample, while coverage and length differences were inconclusive. Cap stops increased from 10 to 24, so these results are not an unconditional quality or deployment claim. This is one training seed and selected early checkpoint on one exposed public news set. Full dimensions, protocol, intervals, and limitations: `docs/results/aligned-adapter-comparison.md` in the Lillm Summarizer repository. This aligned result supersedes the earlier Qwen evaluation snapshot, which used different decoding settings.

## Training and data

LoRA rank 16, alpha 16, dropout 0; evaluated checkpoint 128 from a 512-update run. The training examples pair a source document with its target summary. The public component came from the `HuggingFaceTB/smoltalk2` SFT summarization split, which includes one-line and three-sentence summary styles; the mixed runs also used the author's private web-document/summary pairs. Both Qwen sizes used the same frozen prepared pool and train/validation assignments.

The adaptation goal is to reinforce a shared summarization response contract—objective plain prose, brief when the source is simple and more detailed when it warrants it—across different model sizes. This is an intended direction, not a proven normalization result: no formal cross-model style-variance metric was run, and output lengths and quality changes are model-dependent. Private records, derived training files, and the training runner are not included in this repository.

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
