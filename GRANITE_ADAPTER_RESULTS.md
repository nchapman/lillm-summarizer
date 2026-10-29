# Granite epoch-1 adapter: BF16-sidecar evaluation

## End-to-end milestone, not a final training verdict

**The epoch-1 end-to-end check succeeded.** Checkpoint-3000 was intentionally evaluated before training was finished to verify the full training → independent BF16-sidecar export → Q4-base serving → scoring/judging path. The complete battery ran successfully, and the model-specific F16 range problem was diagnosed and resolved without merging, removing MLP targets, or retraining. The user regards this as a successful single-epoch pipeline milestone, not a final model-selection decision.

The quality measurements are an **epoch-1 reference for later checkpoints**. The paired 250-example confirmation comparison favors the base on factuality, concision, overall judge quality, and all three ROUGE metrics. The adapter improves observed judge coverage while producing longer summaries and more token-cap terminations. The pilot shows the same broad trade-off. These findings do not establish that training is finished, that the checkpoint should be discarded, or what additional epochs will achieve.

This battery measures the intended **Q4 base plus separate adapter deployment** on English news under `summary-v2` and a 256-token output budget, not native HF quality on every task. Preserve the checkpoint, optimizer/scheduler/RNG state, and these results. The BF16 export issue alone is not a reason to restart epoch 1. Before resuming, explicitly handle the old biased validation/best-loss state rather than silently comparing different validation sets; see the audit. No additional training or follow-up experiment was started here.

## Protocol

- **Date:** 2026-10-29. Harness commit at launch: `90852ed1570236b29e3ef872f3aef8e3b4c86c02`.
- **Samples:** frozen CNN/DailyMail 3.0.0 test pilot (50) and confirmation (250), disjoint by ID and exact source text. The confirmation set is the primary comparison; the combined 300 is descriptive, not independent confirmation or a new sealed final test.
- **Pilot SHA-256:** `e45c1587fe25dc3cf510adbda8b5ac962a74417b2f51f47e3a2606e46a251d93`.
- **Confirmation SHA-256:** `047f62c289d3eb08ce3dd5f3096736cf40f0414b1f602a052ee8e79a1377e75e`.
- **Base:** `unsloth/granite-4.0-1b-GGUF`, revision `cfdca559b6af6d2d6c443a5e51d8700ec04d67c9`, `granite-4.0-1b-UD-Q4_K_XL.gguf`. File SHA-256: `07650fb479be74adc4d3d02fba3c55100a1ab72374658427680f43f40e1e6161`.
- **Adapter:** `adapters/granite4-1b-24k-s42/checkpoints/checkpoint-3000`, epoch 1, full attention+MLP LoRA, rank/alpha 16/16. Export: independent **BF16** sidecar, no merge or target removal, at `adapters/granite4-1b-24k-s42/epoch1-lora-bf16.gguf`; SHA-256 `c9ed97dc2cad7c26ed8b3c27b53dfc19d9c7441ce0e0365c4344ed5582b76952`.
- **Runtime:** pinned llama.cpp `0878f7c304cc13915c831ae17694534c94940176`, CUDA/all GPU layers, context 8,192, one slot. Fresh processes for each candidate. The adapter server's GET `/lora-adapters` reported the expected BF16 artifact at scale 1; the base server reported no adapters. No hot-swapping or external source modification.
- **Prompt/decoding:** both candidates freshly evaluated under the same native-template `summary-v2` instruction, raw document as user content, temperature 0, top-p 1, top-k 0, seed 42, maximum 256 new tokens. Historical `summary-v1` baselines and failed F16 outputs are **not** used in this comparison.
- **ROUGE:** macro-mean F1 from `rouge-score`, Porter stemming, on the 0–1 scale. Ordinary ROUGE-L, not ROUGE-Lsum. Reference overlap is not factuality.
- **Judge:** fresh source-grounded `~deepseek/deepseek-flash-latest` judgments for both candidates. Same rubric and request settings; source plus candidate summary only, no reference or candidate identity in the judge prompt. Scores are integers 1–5; concurrency 16, timeout 60 seconds, up to three transient-error retries.
- **Intervals:** percentile 95% CIs from 5,000 paired example resamples, seed `20261029`. Reported changes are **adapter minus base**. Intervals are pointwise, without multiple-comparison adjustment; they capture example-resampling uncertainty, not repeated-judge variability or systematic bias. Judge means summarize ordinal ratings, not percentages of quality.
- **Timing excluded:** a separate MiniCheck GPU workload overlapped part of adapter generation. At the user's direction, latency and throughput are discarded as comparison evidence. Raw timings remain preserved, but no speed or isolated LoRA-overhead claim is made.

## Primary result: 250-example confirmation

| Metric | Base Q4 | Q4 + BF16 adapter | Adapter minus base [95% paired CI] |
| --- | ---: | ---: | ---: |
| ROUGE-1 | 0.3510 | 0.3165 | -0.0345 [-0.0415, -0.0275] |
| ROUGE-2 | 0.1364 | 0.1220 | -0.0144 [-0.0202, -0.0086] |
| ROUGE-L | 0.2201 | 0.2007 | -0.0194 [-0.0249, -0.0138] |
| Judge factuality | 4.324 | 4.048 | -0.276 [-0.404, -0.148] |
| Judge coverage | 3.892 | 4.032 | +0.140 [+0.064, +0.216] |
| Judge relevance | 4.532 | 4.428 | -0.104 [-0.188, -0.020] |
| Judge concision | 4.180 | 3.912 | -0.268 [-0.336, -0.200] |
| Judge readability | 4.676 | 4.568 | -0.108 [-0.200, -0.016] |
| Judge overall quality | 4.084 | 3.920 | -0.164 [-0.268, -0.060] |
| Mean summary words | 115.9 | 149.7 | +33.8 [+30.1, +37.5] |
| Mean completion tokens | 148.2 | 194.4 | +46.2 [+41.4, +51.0] |

The adapter summaries contain about **29% more words** on average. That is compatible with higher coverage, but it does not establish why the other scores changed. These are results under the shared 256-token budget, not an uncapped quality comparison. Do not remove cap-hit examples from the aggregate or attribute every factual error to truncation.

| Output/judge diagnostic | Base Q4 | Q4 + BF16 adapter |
| --- | ---: | ---: |
| Reported `finish_reason=length` | 0/250 | 18/250 (7.2%) |
| At least 255 completion tokens | 0/250 | 22/250 |
| Judge unsupported-claim notes | 143 | 227 |
| Examples with at least one such note | 110/250 | 162/250 |

Near-cap token counts are not proof of truncation; the first smoke example ended mid-phrase with 255 completion tokens despite a reported `stop`. Unsupported-claim notes are judge annotations, not a severity-weighted hallucination rate or proof that unflagged summaries are factual. Longer summaries also provide more opportunities for unsupported claims; these counts are not normalized by claim count.

## Pilot and combined results

| Set/candidate | ROUGE-1 | ROUGE-2 | ROUGE-L | Judge factuality | Judge coverage | Judge overall | Mean words | Length stops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pilot base (50) | 0.3607 | 0.1445 | 0.2276 | 4.260 | 3.820 | 4.000 | 124.2 | 0 |
| Pilot adapter (50) | 0.3233 | 0.1240 | 0.1948 | 3.980 | 4.020 | 3.900 | 158.6 | 6 |
| Combined base (300) | 0.3526 | 0.1377 | 0.2213 | 4.313 | 3.880 | 4.070 | 117.3 | 0 |
| Combined adapter (300) | 0.3177 | 0.1223 | 0.1997 | 4.037 | 4.030 | 3.917 | 151.2 | 24 |

On the pilot, overall-quality change was -0.100 [-0.320, +0.120], so that difference alone was inconclusive. The larger confirmation result is -0.164 [-0.268, -0.060]. Combined overall change is -0.153 [-0.247, -0.060]; it reuses the pilot and confirmation observations and must not be described as another independent replication.

## Source-checked examples

These are illustrative inspected cases, not a random human-rated study:

- **Migrant patrol, pilot ID `30f5088c065de0013105c840a721c1636c74130a`:** the adapter says the crew rescued a pregnant woman, while the source says she had just died and they recovered her body. It also incorrectly makes the ship, rather than Lampedusa, “Italy's southernmost territory,” and ends at “state-of-the-art.” The base captures the patrol/rescue contrast without those particular errors. The judge flagged the woman claim but did not enumerate every source mismatch.
- **Princess Eugenie, confirmation ID `1b884cf6cb5180ea2143142d0a0b30f352865bd9`:** the adapter calls the Duchess of Cambridge “Eugenie's mother” and describes a birthday family visit to the UK. The source identifies her mother as Sarah, Duchess of York, and describes a video call to Eugenie in New York. The base does not make those particular assertions, although its summary is not free of questionable inferences about shopping purchases.
- **AP McCoy, pilot ID `6b9b36ec35fa451d6cdc189bf1ddd14225bdcf60`:** both candidates make unsupported claims. The base calls rower Steve Redgrave an Olympic swimmer; the adapter says McCoy's mental state “has improved,” which the source does not state. This is not a base-is-perfect comparison.

## Completion, recovery, and reproducibility

**All 600 predictions and 600 final judgments are present.** Verification checks exact ordered IDs and source/reference text against the frozen files, unique IDs, complete row counts/status, judgment-to-prediction hashes, prompt-version/sampler/cap metadata, artifact hashes, recorded model/runtime identities and launch commands, score schemas, and context budgets. No empty summaries or runs of 16 repeated identical characters were found. All inference requests completed; no generation failed. No inference server remains on the battery port, and training stayed stopped.

The adapter confirmation judging batch initially stopped after **187 stored valid judgments** because a response's `overall_quality` did not satisfy the required integer 1–5 schema. Those 187 judgments were retained unchanged. Only the remaining 63 IDs were requested again with the same judge/rubric/settings, then the two disjoint parts were verified and assembled in frozen input order. The original failed metadata, partial results, remaining-input file, remaining judgments, and assembly provenance are preserved. No example was omitted and no valid stored rating was selected or replaced based on its score.

Final stored judgment usage reports approximately **$0.11477** across both candidates and both sets. This excludes unrecorded provider work after the aborted batch; some in-flight requests may have completed without their results being stored. There were no transient retries recorded in the 600 selected final judgments; the separate schema-failure recovery is described above.

Artifacts are under **`runs/summary-v2/granite4-1b-epoch1-bf16/`** (git-ignored):

- `manifest.json`: pinned identities, hashes, complete fixed prompt, decoding/runtime settings, and native checkpoint hashes.
- `run_battery.py`, `commands.jsonl`, `*.log`, candidate server commands/logs, and `loaded_adapters.json`: execution provenance. The direct judge invocations are captured in the session command history and their per-batch logs/metadata.
- `base/` and `adapter/`: pilot/confirmation predictions, metadata, ROUGE scores, and judgments; the adapter directory also retains the three-example smoke test and judge-recovery parts.
- `analyze_battery.py`, `analysis.log`, and `comparison.json`: validated per-set aggregates, all paired intervals and per-example higher/equal/lower counts. Timing metrics are deliberately excluded from the comparison.
- `health_snapshot.json` and `interpretation_notes.json`: overlapping GPU-workload observation, explicit timing exclusion, and the user's end-to-end milestone framing.
- `artifact_checksums.json`: SHA-256 index of the 54 preserved run/analysis artifacts (excluding the index itself and Python caches).

Repository verification: full suite **42 passed**, `git diff --check`, and documentation review with no blocking findings. The repeated pre-existing leakage-threshold boundary suggestion was left outside this evaluation-only change.

To recompute from the preserved complete artifacts:

```bash
python runs/summary-v2/granite4-1b-epoch1-bf16/analyze_battery.py
```

Do not run Python with optimization: the analyzer requires assertions to execute. Generation/judgment outputs are not overwritten. The judge uses a `latest` alias; its resolved backend revision is not pinned in the stored responses. Ratings can vary even with temperature zero. CNN/DailyMail reference style, news-only coverage, finite samples, a single training seed/checkpoint, deployment quantization, and the output budget limit generalization. The old biased 200-row training-validation loss is not used here; full native validation loss was not recomputed by this battery. See the [training/runtime audit](docs/granite-training-audit.md) for that separate issue and the resolved F16 range diagnosis.
