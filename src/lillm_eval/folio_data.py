"""Stream a local Folio SFT export into the fixed summarize-button contract.

The paired audit is verified, not merged into the chat: private URLs, titles,
and other page metadata remain solely in the original export. No network calls,
summary rewriting, context truncation, or split assignment happen here.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Sequence

from .core import PROMPT_VERSION, SYSTEM_PROMPT, load_jsonl, sha256_file
from .training_data import build_eval_ngram_set, containment, count_sentences, extract_pair


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _aligned_row(record: dict, audit: dict, index: int, teacher_prompt: str) -> dict:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        raise ValueError(f"row {index}: expected system/user/assistant messages")
    if messages[0].get("role") != "system" or messages[0].get("content") != teacher_prompt:
        raise ValueError(f"row {index}: system prompt does not match source report")
    document, summary = extract_pair(messages[1:])
    if audit.get("sft_row") != index:
        raise ValueError(f"row {index}: audit sft_row is not aligned")
    source_hash, summary_hash = _text_hash(document), _text_hash(summary)
    for key, expected in (("source_sha256", source_hash), ("summary_sha256", summary_hash)):
        if audit.get(key) != expected:
            raise ValueError(f"row {index}: audit {key} does not match message content")
    flags = audit.get("flags")
    if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
        raise ValueError(f"row {index}: audit flags must be a list of strings")
    return {
        "id": source_hash[:16],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document},
            {"role": "assistant", "content": summary},
        ],
        "tag": "folio",
        "document_words": len(document.split()),
        "target_words": len(summary.split()),
        "target_sentences": count_sentences(summary),
        "source_row": index,
        "source_sha256": source_hash,
        "summary_sha256": summary_hash,
        "quality_flags": flags,
        "summary_changed": audit.get("summary_changed"),
        "finish_reason": audit.get("finish_reason"),
        "completion_tokens": audit.get("completion_tokens"),
    }


def prepare_folio(
    source_dir: Path,
    output: Path,
    *,
    eval_inputs: Sequence[Path],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify an export and write an unsplit, deduplicated, leakage-filtered pool.

    Fixed leakage threshold 0.2 matches existing preparation. Malformed rows,
    prompt drift, checksum failures, and audit misalignment fail the import;
    they are never silently dropped. Staging and rollback protect old outputs
    on ordinary publication errors, not crashes or concurrent writers.
    """
    sft_path = source_dir / "sft.jsonl.gz"
    audit_path = source_dir / "audit.jsonl.gz"
    report_path = source_dir / "report.json"
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    inputs = [sft_path, audit_path, report_path, *eval_inputs]
    if any(dst.resolve() == src.resolve() for dst in (output, metadata_path) for src in inputs):
        raise ValueError("output must not overwrite an input file")
    for dst in (output, metadata_path):
        if dst.is_dir():
            raise IsADirectoryError(f"Output destination is a directory: {dst}")
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {dst}; pass --overwrite to replace it")
    if not eval_inputs:
        raise ValueError("at least one frozen evaluation input is required")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    teacher_prompt = source_report.get("system_prompt")
    if not isinstance(teacher_prompt, str) or not teacher_prompt.strip():
        raise ValueError("source report must provide a non-empty system prompt")
    source_hashes = {path.name: sha256_file(path) for path in (sft_path, audit_path)}
    for name, digest in source_hashes.items():
        if source_report.get("output_sha256", {}).get(name) != digest:
            raise ValueError(f"source checksum mismatch for {name}")
    eval_documents = []
    eval_records = []
    for path in eval_inputs:
        records = load_jsonl(path)
        if not records:
            raise ValueError(f"empty frozen evaluation input: {path}")
        eval_documents.extend(row["document"] for row in records)
        eval_records.append({"path": str(path), "sha256": sha256_file(path), "examples": len(records)})
    eval_ngrams = build_eval_ngram_set(eval_documents)
    source_label = f"folio:{source_hashes[sft_path.name]}"
    report: dict[str, Any] = {
        "total_input_rows": 0, "kept": 0, "dropped_duplicate": 0, "dropped_leakage": 0,
        "leakage_max_seen": 0.0, "leakage_max_kept": 0.0,
    }
    seen: set[str] = set()
    flags: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary_output = Path(stream.name)
            temporary_paths.append(temporary_output)
            with gzip.open(sft_path, "rt", encoding="utf-8") as sft, gzip.open(audit_path, "rt", encoding="utf-8") as audit:
                for index, (sft_line, audit_line) in enumerate(zip_longest(sft, audit)):
                    if sft_line is None or audit_line is None:
                        raise ValueError(f"row {index}: SFT and audit row counts differ")
                    record, audit_record = json.loads(sft_line), json.loads(audit_line)
                    if not isinstance(record, dict) or not isinstance(audit_record, dict):
                        raise ValueError(f"row {index}: SFT and audit rows must be objects")
                    row = _aligned_row(record, audit_record, index, teacher_prompt)
                    report["total_input_rows"] += 1
                    source_hash = row["source_sha256"]
                    if source_hash in seen:
                        report["dropped_duplicate"] += 1
                        continue
                    seen.add(source_hash)
                    score = containment(row["messages"][1]["content"], eval_ngrams)
                    report["leakage_max_seen"] = max(report["leakage_max_seen"], score)
                    if score >= 0.2:
                        report["dropped_leakage"] += 1
                        continue
                    row.update(source=source_label, leakage_score=score)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    report["kept"] += 1
                    report["leakage_max_kept"] = max(report["leakage_max_kept"], score)
                    flags.update(row["quality_flags"])
            if report["total_input_rows"] != source_report.get("kept"):
                raise ValueError("SFT row count does not match source report kept count")
            if report["kept"] == 0:
                raise ValueError("no rows retained after filtering")
        report["retained_quality_flags"] = dict(sorted(flags.items()))
        metadata = {
            "source_dir": str(source_dir),
            "input_sha256": {**source_hashes, "report.json": sha256_file(report_path)},
            "source_status": source_report.get("status"),
            "teacher_model": source_report.get("teacher_model"),
            "source_system_prompt": teacher_prompt,
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "split": "unsplit",
            "context_policy": "no truncation or token-length filtering; audit with the training tokenizer before splitting",
            "leakage_check": {"word_ngram_size": 8, "threshold": 0.2, "eval_inputs": eval_records},
            "report": report,
            "output_sha256": sha256_file(temporary_output),
        }
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary_metadata = Path(stream.name)
            temporary_paths.append(temporary_metadata)
            stream.write(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        previous_output = None
        if output.exists():
            # Keep the old inode in the same directory without copying a large pool.
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.backup-", delete=False) as backup:
                previous_output = Path(backup.name)
                temporary_paths.append(previous_output)
            previous_output.unlink()
            os.link(output, previous_output)
        os.replace(temporary_output, output)
        try:
            os.replace(temporary_metadata, metadata_path)
        except OSError:
            if previous_output is None:
                output.unlink()
            else:
                # Retain the backup for manual recovery if rollback itself fails.
                temporary_paths.remove(previous_output)
                os.replace(previous_output, output)
            raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return metadata
