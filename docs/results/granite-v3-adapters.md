# Granite 4.0 LoRA adapters: matched v3 evaluation

## Summary

These are the missing, matched public-news evaluations for the shareable Granite adapters. Both are complete local evaluations; no rerun was needed. Results are **mixed**: the 1B adapter gains on reference overlap and concision, while judged coverage declines and automatic output guardrails flag a few cases. The 350M adapter has fewer cap/degeneration flags than its base, while several judged dimensions—including factuality and overall quality—are lower. These exploratory adapters are shared for experimentation, not presented as quality upgrades.

| Model and evaluated checkpoint | Factuality, base → adapter | Coverage, base → adapter | Overall, base → adapter | ROUGE-1, base → adapter | Mean words, base → adapter |
| --- | ---: | ---: | ---: | ---: | ---: |
| Granite 4.0-1B, `early512/checkpoint-128` | 4.288 → 4.308 | 4.068 → 3.836 | 4.024 → 4.004 | 0.3219 → 0.3454 | 154.9 → 141.9 |
| Granite 4.0-350M, `lr5e5/checkpoint-4456` | 3.620 → 2.744 | 3.256 → 3.296 | 3.124 → 2.736 | 0.3083 → 0.3044 | 179.4 → 163.8 |

Judge ratings are means on an ordinal 1–5 scale, not percentages. Reference overlap is not factuality. The evaluated candidates were selected before their full comparison battery; neither met the project's automatic quality-acceptance policy. The results remain useful as honest adaptation experiments.

## Primary confirmation results

Each row shows base and adapter means followed by **adapter minus base** with a paired 95% confidence interval. The primary set has 250 CNN/DailyMail test articles. ROUGE is macro-mean F1 on a 0–1 scale. Judge dimensions use the same source-grounded rubric within each pair.

| Metric | Granite 4.0-1B base | 1B adapter | Change [95% paired CI] | Granite 4.0-350M base | 350M adapter | Change [95% paired CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Factuality | 4.288 | 4.308 | +0.020 [-0.116, +0.156] | 3.620 | 2.744 | -0.876 [-1.044, -0.712] |
| Coverage | 4.068 | 3.836 | -0.232 [-0.312, -0.156] | 3.256 | 3.296 | +0.040 [-0.060, +0.140] |
| Relevance | 4.544 | 4.532 | -0.012 [-0.100, +0.072] | 3.784 | 3.552 | -0.232 [-0.344, -0.116] |
| Concision | 3.932 | 4.108 | +0.176 [+0.092, +0.256] | 3.276 | 3.268 | -0.008 [-0.140, +0.136] |
| Readability | 4.712 | 4.660 | -0.052 [-0.144, +0.036] | 3.708 | 3.492 | -0.216 [-0.344, -0.072] |
| Overall quality | 4.024 | 4.004 | -0.020 [-0.128, +0.080] | 3.124 | 2.736 | -0.388 [-0.520, -0.252] |
| ROUGE-1 | 0.3219 | 0.3454 | +0.0235 [+0.0150, +0.0323] | 0.3083 | 0.3044 | -0.0040 [-0.0140, +0.0069] |
| ROUGE-2 | 0.1299 | 0.1419 | +0.0120 [+0.0058, +0.0185] | 0.1257 | 0.1203 | -0.0054 [-0.0123, +0.0013] |
| ROUGE-L | 0.2056 | 0.2237 | +0.0181 [+0.0113, +0.0254] | 0.2036 | 0.1956 | -0.0079 [-0.0159, -0.0001] |
| Mean words | 154.9 | 141.9 | -12.9 [-24.3, +0.2] | 179.4 | 163.8 | -15.6 [-37.3, +4.1] |
| Mean completion tokens | 195.3 | 178.5 | -16.7 [-29.7, -2.1] | 225.4 | 211.5 | -13.9 [-40.0, +10.9] |

The 1B adapter's ROUGE and concision gains coexist with lower coverage and a small number of cap/length-stop/degeneration guardrail flags. Factuality and overall-quality intervals include zero. The 350M adapter reduced its base's observed cap/length-stop count from 11/250 to 3/250 and degeneration-flag count from 14/250 to 4/250, while its factuality and overall scores were lower. These automatic flags are diagnostics, not a complete human quality assessment.

## Training and artifacts

- **Base models:** `ibm-granite/granite-4.0-1b` and `ibm-granite/granite-4.0-350m`, pinned native revisions `6a7381ba1f54d684ff508d991aeb7dc580157103` and `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2` respectively.
- **Adapter method:** LoRA, rank 16, alpha 16, dropout 0, no bias training, attached to language attention/MLP projections. The shareable files are PEFT adapter weights/configuration, not merged base models.
- **1B artifact:** mixed-data `granite4-1b-summary-v3-8k-s42-early512/checkpoints/checkpoint-128`.
- **350M artifact:** mixed-data lower-learning-rate `granite4-350m-summary-v3-8k-s42-lr5e5/checkpoints/checkpoint-4456`.
- **Data:** aligned source-document/target-summary examples. The public component came from the `HuggingFaceTB/smoltalk2` SFT no-think summarization split, with both one-line and three-sentence summary styles; the mixed pool also included the author's private web-document/summary pairs. These varied targets suit the `summary-v3` instruction to keep simple material brief and add detail when the source warrants it, rather than forcing every model to emit a fixed-length summary. The private records and derived data files are not included in the adapter repositories.
- **Evaluation representation:** Q4_K_XL GGUF base plus an independently exported BF16 GGUF adapter sidecar. These measurements do not directly establish native-BF16 PEFT inference quality.

## Evaluation protocol and limitations

- Base and adapter each generated a fresh output for the same frozen public CNN/DailyMail pilot (50) and confirmation set (250); confirmation is primary. Combined 300 is descriptive and reuses those observations.
- Both states used the same model-specific base file, `summary-v3` instruction, native prompt rendering and decoding within a model pair. Maximum output was 1,024 new tokens. Each model was compared only with its own base.
- Source-grounded judgments used `~deepseek/deepseek-flash-latest`; one mutable judge alias means backend/version and judge variability are not fully controlled. Judge notes are annotations, not verified hallucination counts.
- Paired percentile bootstrap: 5,000 resamples, seed `20261029`, pointwise 95% intervals. Intervals describe example-resampling uncertainty, not seed, prompt, or judge uncertainty; no multiple-comparison correction was applied.
- The benchmark is English news, the public cases have been exposed in previous work, training used one seed, and neither adapter passed the automatic acceptance gate. These exploratory findings should not be generalized to all summarization domains or treated as a deployment recommendation.

## Reproduction evidence

Original, git-ignored local evidence:

- 1B: `runs/diagnostics/1b-v3-early512-step128/full-public-battery/` (`manifest.json`, `comparison.json`, predictions, judgments, inspection and verification records).
- 350M: `runs/diagnostics/350m-v3-lr5e5-epoch1-s100/full-public-battery/` (same evidence layout).
- Checkpoints remain under ignored `adapters/`; local training and evaluation outputs are not required to load the published PEFT adapters.
