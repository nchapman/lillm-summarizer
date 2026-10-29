# Granite checkpoint-3000: training and deployment audit

## Bottom line

The full native adapter produces coherent summaries. Its **F16 GGUF sidecar degenerates on this GB10/CUDA build**, while an **F32 sidecar works with the same Q4 base** in controlled spot checks. No merging, target removal, or retraining was required for that recovery. This is evidence about inference correctness, **not evidence of improved summarization quality**.

The audit also found mistakes in our training harness and earlier claims: biased validation selection, an ignored revision pin, incorrect import order, and an overstated reload check. The stopped checkpoint is preserved; no training was resumed and no new evaluation battery or hosted judging was run.

## Artifacts and scope

- Native checkpoint: `adapters/granite4-1b-24k-s42/checkpoints/checkpoint-3000` (epoch 1).
- Requested base: `ibm-granite/granite-4.0-1b` at `6a7381ba1f54d684ff508d991aeb7dc580157103`.
- Historical actual base: `unsloth/granite-4.0-1b`, cached revision `286ab97f84814537f46cb803bbd7e6d615a68690`.
- Both cached `model.safetensors` files resolve to the same content-addressed blob: `cc2def9a94efd3a18b21080b053fa64a3f0c329eef357233f0392940179058f4`. Tokenizer vocabulary JSON and special-token maps also match. Config differences inspected were version/Unsloth metadata and default maximum lengths; the mirror also embeds the chat template in tokenizer config.
- Deployment base: `models/granite4-1b/granite-4.0-1b-UD-Q4_K_XL.gguf`.
- Working full F32 sidecar retained at `adapters/granite4-1b-24k-s42/epoch1-lora-f32.gguf` (untracked); its checksum is recorded with the diagnostics.
- External llama.cpp: `/home/nchapman/Code/llama.cpp`, `0878f7c304cc13915c831ae17694534c94940176`. No source edits or rebuilds made there. The completion binary links the local `build-cuda/bin` libraries.
- Stack: Unsloth 2026.10.10, TRL 1.15.0, PEFT 0.19.1, Transformers 4.57.6, Torch 2.12.1+cu130.

Local diagnostic scripts, command records, output logs, tokenized inputs, and held-out comparisons are retained under **`runs/diagnostics/granite4-1b-epoch1-dtype/`** (untracked, as are other run artifacts). These are local evidence, not portable automated tests. Model/sidecar scratch artifacts remain under `/tmp` and must be regenerated if removed.

## Documentation used

- [Unsloth: running and saving models](https://unsloth.ai/docs/basics/running-and-saving-models): native adapter saving is distinct from merged-model export. Our deployment requirement is an independent sidecar; we do not use a merged GGUF.
- [Unsloth: troubleshooting and FAQs](https://unsloth.ai/docs/basics/troubleshooting-and-faqs): preserve the training chat template, BOS/EOS behavior, and inference formatting after export. A successful save is not an inference check.
- [Unsloth: LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide): attention and MLP adaptation together; rank/alpha, effective batch size, and training length are independent choices. Generic Llama projection names are not valid architectural introspection for Granite.

Docs recommendations were checked against the **installed source and actual checkpoint**, rather than assuming a notebook's defaults matched this environment.

## Training/data trace

| Check | Observed result | Interpretation/action |
| --- | --- | --- |
| Import order | Historical logs warn that Unsloth was imported after Transformers/TRL/PEFT. | Fixed both scripts to import Unsloth first. The warning does not prove a training corruption. |
| Revision pin | Unsloth explicitly warned that remapping IBM to its mirror discarded the IBM SHA. | Added `use_exact_model_name=True` and explicit bf16. A model-load-only check resolved the IBM repo and exact requested SHA without the remapping/import warnings. Identical cached weight blobs argue against a weight mismatch in the historical run. |
| Architecture | 40 attention layers, zero Mamba layers, zero routed experts, despite the `GraniteMoeHybridForCausalLM` family name. | Do not invoke Mamba behavior to explain this model's discrepancy. |
| Applied targets | 40 each of q/k/v/o and `shared_mlp.input_linear`/`output_linear`; 19,005,440 trainable parameters. | The corrected run really includes MLP adapters. The earlier attention-only selection was our target resolver's behavior, not evidence that Unsloth dropped requested valid MLPs. |
| Training arguments | bf16, batch 8, accumulation 1, LR 2e-4, warmup 0.1, linear schedule, AdamW 8-bit, weight decay 0.001, max grad norm 1, completion-only loss, packing false. | Read from checkpoint `training_args.bin`; effective batch is 8, not 64. |
| Sequence length | Serialized `max_length=None`, `max_seq_length=3072`; generated Unsloth preprocessing falls back to `max_seq_length`. | Do not infer unlimited preprocessing from `max_length=None` alone. Tokenizing all 24,000 native-template training examples found maximum 2,733 tokens and **zero over 3,072**. |
| Loss masks/template | Instantiated the existing trainer without training on examples from all five populated tag × length strata. Collated labels mask the prompt, retain the summary plus `<\|end_of_text\|>`/newline, and match the native prompt prefix. | Completion-only training is actually applied, not just configured. Unsloth's “double BOS” handling was inspected through prepared outputs rather than assumed to be a failure. |
| Validation | Split contains 1,000 rows: 646 `three_sentence`, 354 `one_line`. Old `--val-limit=200` took the first 200 grouped rows: **all `one_line`**. | Real harness bug. Loss **0.50579226** measures that biased subset, not mixed-task validation. Default now uses the full split; explicit caps are seeded uniform samples from the whole file. Selection IDs are recorded in completed-run metadata. |
| Checkpoint/resume | Optimizer, scheduler, RNG, trainer state and adapter retained at step 3,000. Historical `restore_callback_states_from_checkpoint=False`. | The claim of fully faithful early-stopping resume was too strong. New config enables callback restoration, but an actual resume has not been tested here. Changing validation data also makes historical best-loss/early-stop state incomparable; do **not** silently resume this checkpoint with the new defaults. |
| Reload check | Existing heuristic checks only the first 64 **prompt** positions. | It cannot establish assistant-output parity or explain differences as numerical noise. Removed those assertions from messages/docs. Separate native/sidecar generations and input-token checks are required. |

Training targets are retained from smoltalk2 while original length instructions are replaced by the fixed summary-v2 instruction. Tags describe target-generation provenance, not guaranteed lengths or identifiable news/email content. Exact duplicate/8-gram leakage filtering is useful, not proof against every form of semantic overlap. No training-data rewrite was made in this audit.

## Conversion and runtime source trace

All references below are relative to the pinned llama.cpp checkout unless otherwise noted.

1. **Fused input projection**: `conversion/granite.py:211–225` splits `shared_mlp.input_linear` into gate then up; dense `output_linear` maps to down. The generated HF/Unsloth implementation splits in the same order and computes `activation(first_half) * second_half`.
2. **LoRA split mathematics**: `convert_lora_to_gguf.py:193–203` splits B along the output dimension and shares A. Rank stays 16. Alpha 16/rank 16 gives scale 1 (`src/llama-adapter.h:52–56`). No factor-of-two correction is appropriate.
3. **Actual tensor verification**: all **240 FFN tensors across all 40 layers** in the F16 sidecar exactly equal their expected HF arrays rounded to F16: gate/up A, first/second-half B, and down A/B. No coefficient becomes nonfinite; maximum per-tensor relative rounding error is below approximately 0.023%. All 560 GGUF coefficient arrays were also checked for equality in the F16-to-F32-storage control below.
4. **Runtime composition**: `src/llama-graph.cpp:1657–1663` constructs base matmul plus scaled B(A(x)). The inspected dense Granite graph uses that composition for these FFN projections. No concrete converter or graph-formula defect was found.
5. **Internal precision**: visible F32 graph outputs do not ensure F32 arithmetic. CUDA source selects F16 internal compute/output for ordinary F16 cuBLAS inputs (`ggml/src/ggml-cuda/ggml-cuda.cu:1444–1455,1522–1580,1672–1693`); the one-column F16 MMVF path can also use half accumulation (`ggml/src/ggml-cuda/mmvf.cu:194–222,608–630`). This is a plausible mechanism to investigate, **not proof of a specific overflow or defective kernel**.

The missing-split-support explanation and gate/up swap explanation are contradicted by this checkout and its actual tensor values. The gate-only/up-only attempts that failed conversion with missing-A assertions were not valid experimental controls.

## Controlled inference results

Fresh standalone `llama-completion` processes used the same native-HF-rendered summary-v2 prompt: a short document about a $12 million water-pipe project, 93 prompt tokens, greedy decoding, 64-token maximum, context 1,024. CPU means `--device none -ngl 0 --no-op-offload` in the same build; CUDA means `-ngl all`.

| Base | Adapter | Execution | Result |
| --- | --- | --- | --- |
| Q4_K_XL | none | CUDA | Coherent summary, EOS |
| Q4_K_XL | F16 attention only, extracted from full adapter | CUDA | Coherent summary, EOS |
| Q4_K_XL | F16 attention + down | CUDA | Repeated `G`, 64-token limit |
| Q4_K_XL | F16 full | CUDA | Repeated `G`, 64-token limit |
| Q4_K_XL | F16 full | CPU | Coherent summary, EOS |
| Q4_K_XL | F16 full | CUDA, fusion disabled | Repeated `G` |
| Q4_K_XL | F16 full | CUDA, graphs disabled | Repeated `G` |
| Exact IBM BF16 GGUF | F16 full | CUDA | Repeated `G` |
| Q4_K_XL | F32 full, exported from native checkpoint | CUDA | Coherent summary, EOS |
| Q4_K_XL | **F16 coefficients promoted to F32 storage without changing their values** | CUDA | Coherent summary, EOS |

The unquantized base control was a conversion of the pinned **base only**, not a merged adapter. Individual fusion/graph controls did not fix the failure. BF16-base failure shows Q4 quantization is not necessary for it. The identical-coefficient storage control distinguishes execution dtype from coefficient-rounding loss. It does **not** by itself identify the exact CUDA operation that fails or establish behavior on other GPUs/builds.

A stock `llama-debug` callback comparison was also collected. Its tokenization differs from the completion test (137 versus 93 tokens) and callbacks introduce synchronization boundaries, so it is **not** used to claim the first divergent operation has been found.

### Real held-out and server checks

Two validation examples were then generated through both plain HF+PEFT bf16 and Q4+full-F32-sidecar CUDA, with context 8,192 and a 256-token limit:

- `69f450c26331266c` (`one_line` provenance, email): both produced coherent summaries of the Guatemala health-center update and proposed charity hike.
- `b6a24e7e3e3101c3` (`three_sentence` provenance, news): both produced coherent summaries of the Eriksson/Dell'Olio property dispute.

For **each**, every llama.cpp prompt token ID matched HF's native `apply_chat_template(..., tokenize=True)` sequence. Outputs ended normally and were read alongside their documents; they are not identical across engines and have not been quality-scored. The first diagnostic capture attempt failed while decoding partial UTF-8 token pieces in verbose stderr; rerunning with replacement decoding for diagnostic stderr completed both checks.

A fresh `llama-server` with the Q4 base, full F32 sidecar, `-ngl all -c 8192 --parallel 1` also returned a coherent water-pipe summary from `/v1/chat/completions`, with 93 prompt tokens, 46 completion tokens, `finish_reason=stop`, fingerprint `b1-0878f7c`. The temporary server was stopped afterward. Hot-swapping APIs were **not** exercised.

## What is fixed, and what is not

Fixed in this repository:

- Correct Unsloth import order, exact repository/revision loading, explicit bf16.
- Full validation by default, seeded whole-file sampling for optional limits, recorded selected IDs, and non-prefix generation sampling.
- Enable saved callback-state restoration for future resumes; actual resume remains untested.
- Smoke metadata records saved applied targets; both scripts retain their held-out generation outputs.
- Remove overstated reload/deployment claims; document F32-sidecar export explicitly.

Verification: six new row-selection tests, including a regression confirmed to fail on prefix selection; full suite **42 passed**; Python compilation and `git diff --check`; corrected model-load-only check; actual training-mask inspection; native/sidecar and HTTP-server spot checks. No new training run was performed, so these changes do not constitute a post-fix training benchmark or end-to-end resume verification. Code review found no blocking issues in this change; its two nonblocking suggestions concern pre-existing leakage-threshold boundary handling and optional-data-extra test skips, which are left outside this audit's focused fixes.

Still open:

- No valid quality comparison from the failed F16 pilot/partial confirmation runs. Do not judge or score those outputs as adapter performance.
- No proof yet of improved summarization, good length control, hot-swap behavior, or portability of the dtype finding beyond this build/hardware.
- Exact operation-level CUDA diagnosis would require a captured-operand/reference comparison; no external source patch is justified by the current evidence.
- Before further training decisions, establish representative validation under the corrected selection and explicitly handle historical best-loss/callback state rather than mixing incomparable losses.
