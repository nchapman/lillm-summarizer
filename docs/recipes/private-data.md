# Private-data workflow (not runnable without the private export)

The repository retains support for importing and preparing the private Folio SFT data used in the mixed-data experiments. **The private export and its derived records are not included or available here.** The workflow cannot be run without authorized access to the source export. The public-only alternative is [documented separately](public-data.md).

## Required local export

The importer expects an external source directory containing:

```text
sft.jsonl.gz
audit.jsonl.gz
report.json
```

Keep this export outside the repository. Do not commit or publicly share the records, audit, report, hashes, derived splits, metadata, predictions that disclose private content, or adapters trained on restricted material.

## Preserved import entry point

After placing an authorized export in an appropriately protected local path, and after generating/preserving the same frozen public evaluation files as in the public recipe, run:

```bash
umask 077
uv run --locked lillm-summarizer prepare-folio \
  --source-dir /path/to/private/sft_export \
  --output data/train/folio_sft_v1/aligned.jsonl
```

The importer validates compressed source hashes and row/document/summary alignment against the audit, replaces the system prompt with the project's current summarization prompt, retains document and target text, removes exact duplicate documents, and checks for n-gram overlap against the frozen public evaluation sources. It fails closed on invalid provenance. Output and metadata have restrictive permissions and the entire `data/train/folio*/` tree is Git-ignored.

The import is an **unsplit, not context-filtered** aligned pool. Audit full chat lengths with the exact model tokenizer and context policy before preparing train/validation partitions. Review quality flags and target quality; the importer validates data structure/provenance, not semantic truth or target quality. Never rely on silent truncation.

## Combined/private training preparation

`lillm_eval.combined_data.prepare_combined` is the retained programmatic entry point for locally combining approved prepared pools. It checks prompts, row structure, document IDs, duplicate documents, evaluation overlap, tokenizer controls, full rendered lengths, and deterministic source/length-stratified splitting. Private review exclusions can be passed explicitly. Output is atomically published to a fresh restricted directory on Linux.

This helper is a Python API, not a supported public CLI workflow. Keep its inputs, manifests, review maps, counts, and output under a private, ignored path. Training is then invoked using explicit train/validation paths and a new adapter output directory; do not reuse historic checkpoints or validation state without reviewing the relevant experiment report. See `src/lillm_eval/combined_data.py`, `src/lillm_eval/folio_data.py`, and `scripts/train_lora.py` for the preserved implementation.

## Reproducibility limits

Because the private corpus and the Qwen-specific runner/environment are not distributed, the published Qwen base-versus-adapter results are not exactly reproducible from this repository. Public SmolTalk2-only training is a useful independent experiment, but results from it must not be described as a reproduction of the mixed-data run. See the [Qwen report](../results/qwen-adapters.md) for the disclosed limitations and training/evaluation checks.
