# Qwen3.5: out-of-the-box summaries versus LoRA training

## Finding: training improved both small models

This experiment asks **how well small models summarize out of the box, and whether task-specific training makes them better**. It is a comparative research result, not a deployment qualification.

**Both Qwen3.5 adapters improved over their own unadapted models.** On the larger 250-article battery, factuality, relevance, concision, overall judge quality, and all three ROUGE metrics increased. Summaries became about half as long, without a clear change in coverage. The trained 2B model had the higher factuality and overall scores of the two trained models.

| Model | Factuality: base → trained | Overall quality: base → trained | Mean words: base → trained |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 1.596 → **1.976** | 1.600 → **1.924** | 285.2 → **134.3** |
| Qwen3.5-2B | 1.948 → **2.448** | 1.956 → **2.336** | 329.1 → **157.0** |

Judge scores are ordinal ratings from **1 to 5**, not accuracy percentages. The result is useful adaptation under this particular training and inference recipe; it does not establish a performance ceiling or isolate which part of training caused each improvement. Remaining factual errors and occasional repetition are findings to report, not reasons to dismiss the measured gains.

## What was trained

Two independent LoRA runs started from the corresponding native Qwen checkpoints. The 2B adapter was not initialized from the 0.8B run.

| Setting | Both models |
| --- | --- |
| Native checkpoints | `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B`, pinned revisions |
| Training representation | Native BF16 model with LoRA; not QLoRA |
| Adapter | Rank 16, alpha 16, dropout 0, no bias training |
| Targets | 186 text linear projections across full/linear attention and MLPs; 372 LoRA factor tensors |
| Frozen components | Base weights, vision branch, embeddings, output heads, and non-target parameters |
| Optimizer | 8-bit AdamW; configured peak LR `5e-5`, weight decay `0.001`, gradient clipping `1.0` |
| Batch | Microbatch 4 × gradient accumulation 8 = effective batch 32 |
| Context | 10,240 tokens for the complete native chat, including target and terminal tokens |
| Prompt / seed | `summary-v3`, thinking disabled, seed 42 |
| Run budget | 512 optimizer updates per model; validation and saves every 64 updates |
| Evaluated adapter | **Checkpoint 128 for each model**, selected before quality generation |

The runs retained the original three-epoch-derived linear learning-rate schedule and 10% warmup instead of compressing the schedule into 512 steps. Both the evaluated checkpoint and the run endpoint were still inside that warmup. These are **bounded early-training experiments**, not completed three-epoch runs. At an effective batch of 32, checkpoint 128 represents 4,096 nominal example presentations; the full 512-update run represents 16,384. Those are training exposures, not corpus sizes or a claim about unique documents.

The lowest validation loss occurred at checkpoint 512 in both runs. It was saved and restored for training verification, but it did **not** replace the preselected checkpoint 128 in this quality comparison. No checkpoint sweep, seed search, or summary reroll was used to obtain these results.

### Data and supervision

Both models used the same frozen summarization documents, targets, and train/validation assignments. The pool combined public `HuggingFaceTB/smoltalk2` summarization data with a private summarization component. Preparation included document deduplication, evaluation-overlap checks, and target-quality review. Model-specific preparation retained the frozen rows and original document/target text, with **no context truncation or target shortening**.

Native Qwen control-token literals were escaped only in the rendered user-document view; stored source documents and assistant targets remained unchanged. The native chat template, disabled-thinking prefix, assistant-only loss mask, terminal EOS/newline, and actual trainer/collator labels were checked across the prepared data before optimization. Prompt and padding tokens were excluded from supervision.

This was more than a small loading smoke test: it supplied enough training exposure to measure early adaptation, followed by a matched public comparison. It was **not** a data-volume ablation or proof that more data or longer training would stop helping. The private component's records, counts, fingerprints, and provenance reports are intentionally not published; that limits exact public reproduction of the training mix.

### Training and export checks

Both runs completed all 512 updates. Checks found finite adapter gradients, changes in all 372 intended factor tensors, unchanged frozen weights, the expected learning-rate progression, and exact restoration of the best-loss checkpoint. Fresh-process save/reload checks passed without additional optimizer updates.

| Model | Validation loss at evaluated step 128 | Validation loss at final step 512 |
| --- | ---: | ---: |
| Qwen3.5-0.8B | 1.402953 | 1.269709 |
| Qwen3.5-2B | 1.163576 | 1.025803 |

Loss is a training diagnostic, not the quality result. For the actual evaluation, both checkpoint-128 adapters were exported as separate **BF16 GGUF sidecars**. Export checks covered every factor's coefficient conversion and native base shape. Actual Q4 serving checks covered native/runtime prompt-token agreement, adapter loading and scale, a nonzero adapter effect, and EOS handling. These checks support correct training/artifact plumbing; they do not assert numerical or semantic equivalence between native BF16 and Q4 inference.

## Matched evaluation

- **Data:** the existing CNN/DailyMail 3.0.0 test pilot (50 articles) and disjoint larger battery (250 articles). These public sets had already been exposed in earlier work; the larger set is not a newly sealed confirmation set. The combined 300 is descriptive only.
- **Fresh outputs:** 300 base and 300 adapter summaries per model, **1,200 total**. Historical `summary-v1` / 256-token Qwen outputs were not reused.
- **Inference:** each model's existing `UD-Q4_K_XL` base, with the same base artifact for both conditions; the trained condition adds its BF16 sidecar. Both states were served from one loaded runtime per size, with explicit per-request adapter scale 0 or 1.
- **Runtime:** local NVIDIA GB10, llama.cpp CUDA revision `0878f7c304cc13915c831ae17694534c94940176`, one slot, 10,240-token context. No source truncation.
- **Shared prompt:** `summary-v3`: clear, objective, source-dependent detail, plain prose, no unsupported information. Both states use the same native template and document rendering.
- **Decoding:** thinking off; temperature 1, top-p 1, top-k 20, min-p 0, presence penalty 2, repeat penalty 1, frequency penalty 0, repeat window 64, seed 42, prompt caching off, maximum 1,024 new tokens. Explicit sampler order: penalties → top-k → top-p → min-p → temperature. This is fixed stochastic decoding, not greedy decoding.
- **Judge:** `deepseek/deepseek-v4.1-flash-1017` through OpenRouter, temperature 0, reasoning disabled. The unchanged six-dimension rubric sees only the public source and anonymous candidate summary, not the reference or model identity. No private training documents were sent for judging.
- **ROUGE:** macro-mean F1, Porter stemming, ordinary ROUGE-L rather than ROUGE-Lsum; scale 0–1. Reference overlap is not factuality.
- **Uncertainty:** 5,000 paired article resamples, percentile 95% confidence intervals, seed `20261029`. Changes below are **trained minus base**. Intervals are pointwise, with no multiple-comparison adjustment; they do not measure training-seed or judge variability.

### Main comparison: 250 articles

| Metric | 0.8B base | 0.8B trained | Change [95% paired CI] | 2B base | 2B trained | Change [95% paired CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Factuality | 1.596 | 1.976 | +0.380 [+0.264, +0.500] | 1.948 | 2.448 | +0.500 [+0.368, +0.632] |
| Coverage | 2.436 | 2.476 | +0.040 [-0.056, +0.140] | 2.864 | 2.900 | +0.036 [-0.080, +0.140] |
| Relevance | 2.432 | 2.816 | +0.384 [+0.276, +0.492] | 2.836 | 3.188 | +0.352 [+0.236, +0.468] |
| Concision | 2.028 | 2.784 | +0.756 [+0.648, +0.864] | 2.108 | 2.944 | +0.836 [+0.720, +0.952] |
| Readability | 2.432 | 2.504 | +0.072 [-0.016, +0.156] | 2.864 | 2.852 | -0.012 [-0.120, +0.092] |
| Overall quality | 1.600 | 1.924 | +0.324 [+0.220, +0.432] | 1.956 | 2.336 | +0.380 [+0.256, +0.496] |
| ROUGE-1 | 0.2129 | 0.3139 | +0.1011 [+0.0922, +0.1104] | 0.1977 | 0.3010 | +0.1033 [+0.0929, +0.1138] |
| ROUGE-2 | 0.0599 | 0.0873 | +0.0274 [+0.0219, +0.0331] | 0.0581 | 0.0907 | +0.0326 [+0.0270, +0.0385] |
| ROUGE-L | 0.1190 | 0.1717 | +0.0527 [+0.0468, +0.0587] | 0.1127 | 0.1728 | +0.0601 [+0.0530, +0.0673] |
| Mean words | 285.2 | 134.3 | -150.9 [-163.2, -138.6] | 329.1 | 157.0 | -172.2 [-188.0, -157.3] |
| Mean completion tokens | 351.5 | 173.3 | -178.2 [-193.6, -162.4] | 397.6 | 200.6 | -197.0 [-216.6, -178.6] |

The word-count reductions are approximately **53% for 0.8B and 52% for 2B**. Coverage and readability changes are not resolved by these intervals. The clearer gains are factuality, relevance, concision, overall quality, and reference overlap. The experiment does not isolate whether shorter outputs explain some of the factuality gain.

### Pilot and pooled results

| Set | Model | Factuality: base → trained | Overall: base → trained | Mean words: base → trained |
| --- | --- | ---: | ---: | ---: |
| Pilot (50) | 0.8B | 1.640 → 1.900 | 1.640 → 1.880 | 302.2 → 124.3 |
| Pilot (50) | 2B | 1.860 → 2.360 | 1.860 → 2.340 | 343.0 → 160.2 |
| Pooled (300) | 0.8B | 1.603 → 1.963 | 1.607 → 1.917 | 288.1 → 132.7 |
| Pooled (300) | 2B | 1.933 → 2.433 | 1.940 → 2.337 | 331.5 → 157.5 |

The pilot shows the same direction of factuality/overall gains. Pooled results reuse both sets; they are not a third replication. The trained 2B's higher judge scores do not mean it wins every metric: trained 0.8B has slightly higher ROUGE-1 on the larger battery. No paired cross-model superiority test is claimed.

## Shortcomings observed

All outputs, including failures, remain in the aggregates.

| Diagnostic, larger 250 | 0.8B base | 0.8B trained | 2B base | 2B trained |
| --- | ---: | ---: | ---: | ---: |
| 1,024-token cap / length stop | 0 | 0 | 0 | 1 |
| Automatic degeneration flags | 1 | 1 | 0 | 1 |
| Examples with judge unsupported-claim notes | 250 | 243 | 250 | 236 |
| Total unsupported-claim notes | 2,179 | 1,188 | 2,089 | 943 |

Source inspection found a genuine repeated-caption/quotation block in one 0.8B adapter summary and a lunar-eclipse caption loop that reached the cap in one 2B adapter summary. Each is **one observed case out of 250**, not an estimate precise enough to establish a rare-failure rate. The 0.8B base's automatic flag appears to be a sentence-splitting false positive around repeated “Mrs.” abbreviations; its summary still contains factual errors. The separate 2B pilot also contains one overlong base output that reached the cap.

Ordinary source checks found incorrect names, numerical details, role attribution, and confusion between article content and embedded related-link headlines. For example, a migrant-patrol article's related headline about being smuggled into Europe was incorrectly turned into the patrol officer's own history. In a football article, summaries reversed standings or who insulted whom. These illustrative checks are not a human-rated accuracy study.

Unsupported-claim notes decreased, but they are judge annotations, not a verified hallucination count. Shorter summaries also give the judge fewer claims to flag. No empty summaries or explicit thinking/native-control-token leaks were found in the 1,200 outputs.

Earlier run machinery recorded conservative output guardrails. Both adapters tripped one or more of those flags. That diagnostic is preserved, but **it is not the success criterion for this research question**: the finding is that training measurably improved these models, with the listed weaknesses still present.

## Judge repairs and verification

The first pass returned 1,195 valid judgments, four JSON responses missing `overall_quality`, and one truncated response. With explicit authorization, exactly those five cases received one additional request each: same pinned judge, rubric, source, and generated summary, with only the judge response budget increased from 512 to 1,024 tokens. All five repairs succeeded. No valid first-pass judgment was replaced, no score was invented, and no summary was regenerated. Final tables use **1,195 original judgments plus five documented repairs**.

All original and repair responses are retained separately. Reported response usage totals **$0.290694** for the original and repair calls together; this is judge cost, not training cost or a complete accounting of provider work during transport failures.

Exact input ordering, public-source identity, native/runtime prompts, runtime settings, adapter scale, stop/token evidence, and preserved artifact bindings were checked. A separate offline implementation **passed the full independent check**: all 1,200 summaries and 1,200 resolved judgments were validated, and recomputed ROUGE, all six judge dimensions, counts, paired bootstrap intervals, and recorded guardrail results agreed with the reported comparison. The original five invalid responses remain preserved. This verifies the accounting and calculations, not the automated judge's correctness.

## Scope and sharing

- This measures **Q4 inference plus BF16 LoRA sidecars**, one training seed, one decoding seed, one selected early checkpoint, and one pinned automated judge on English news. It does not establish general web/email quality, native-BF16 quality, or a model-family ranking independent of quantization and sampling.
- The public sets were already exposed. The paired results describe these samples, not a fresh sealed-test claim. They support improvement under this recipe, not exhaustive convergence or optimal data-volume claims.
- Historical baseline and Granite reports use different prompts and output budgets. Do not mix them with this fresh `summary-v3` / 1,024-token comparison to construct a matched leaderboard.
- Raw training data, private provenance, adapters, downloads, and generated artifacts remain untracked. This report publishes findings and methodology, not the private corpus.
- The Qwen-specific training runner uses a separate Transformers 5.2.0 environment; the root Granite training stack uses Transformers 4.57.6. The Qwen runner, environment, and experiment orchestration are currently preserved under ignored local run directories, **not yet packaged as a standalone public reproduction path**.

Local evidence is under `runs/training-qwen35/` (training/export) and `runs/diagnostics/qwen35-checkpoint128-public-v1/` (evaluation). The latter retains `manifest.json`, `comparison.json`, original judge responses, `judge-repairs-v1/`, `analysis-contract.json`, and inspection/verification records. Paths are evidence locations for the original checkout, not files shipped by Git.

**Next repository work:** organize the public entry points, separate current instructions from historical investigations, and extract the minimal non-private Qwen recipe and analysis code from local run orchestration. Preserve the experiment's evidence and conclusions while making the project easier to understand and share.
