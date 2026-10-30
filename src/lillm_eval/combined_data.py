"""Combine prepared chats under the current prompt without rewriting completions.

Input mapping order defines first-wins deduplication (even if that first row is
subsequently excluded). Only input batches and accepted rows are held in memory.
Evaluation files use the harness's document/reference JSONL schema. No tokenizer
or model is loaded here, and no text is included in exclusion reports.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable, TextIO

from .core import PROMPT_VERSION, PROMPT_VERSION_V2, SYSTEM_PROMPT, SYSTEM_PROMPT_V2, load_jsonl, sha256_file
from .training_data import NGRAM_SIZE, build_eval_ngram_set, containment

BATCH_SIZE = 128
LEAKAGE_THRESHOLD = 0.2


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_row(row: Any, source: str, line: int) -> tuple[str, str]:
    """Validate before any filtering; errors deliberately omit row text."""
    prefix = f"source {source!r}, input line {line}"
    if not isinstance(row, dict):
        raise ValueError(f"{prefix}: row must be an object")
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or any(not isinstance(message, dict) for message in messages)
        or [message.get("role") for message in messages] != ["system", "user", "assistant"]
    ):
        raise ValueError(f"{prefix}: expected exact [system, user, assistant] roles")
    if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        raise ValueError(f"{prefix}: content must be a non-empty string")
    if any(set(message) != {"role", "content"} for message in messages):
        raise ValueError(f"{prefix}: messages must contain only role and content")
    if messages[0]["content"] not in (SYSTEM_PROMPT, SYSTEM_PROMPT_V2):
        raise ValueError(f"{prefix}: system prompt must be the current contract or known {PROMPT_VERSION_V2}")
    document, target = messages[1]["content"], messages[2]["content"]
    if row.get("id") != _digest(document)[:16]:
        raise ValueError(f"{prefix}: id must equal sha256(user)[:16]")
    return document, target


def _quarantine_reason(document: str, target: str, special_tokens: Sequence[str]) -> str | None:
    for text in (document, target):
        if any(token in text for token in special_tokens):
            return "special_token"
        if any((ord(char) < 32 and char not in "\t\r\n") or ord(char) == 127 for char in text):
            return "forbidden_control"
        if "\ufffd" in text:
            return "replacement_character"
    return None


def _target_warnings(target: str) -> list[str]:
    # A review signal, not proof of truncation or a reason to discard a row.
    ending = target.strip().rstrip("\"'’”)]}»`").rstrip()
    return ["missing_terminal_punctuation"] if not ending or ending[-1] not in ".!?…" else []


def _token_band(count: int) -> int:
    upper = 1024
    while count > upper:
        upper *= 2
    return upper


def _split(
    rows: list[dict[str, Any]], val_count: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bounded proportional allocation, keeping at least one train row per stratum.

    Singletons stay in train. Each other stratum gets at least one validation
    row when the budget can cover all such strata. Infeasible requests fail,
    rather than silently shrinking validation or exhausting a training stratum.
    """
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["source_dataset"], _token_band(row["token_count"]))].append(row)
    keys = sorted(strata)
    capacities = {key: len(strata[key]) - 1 for key in keys}
    if not rows or val_count > sum(capacities.values()):
        raise ValueError("insufficient retained rows for val_count while leaving train rows per stratum")
    eligible = sum(capacity > 0 for capacity in capacities.values())
    lower = {key: int(capacities[key] > 0 and val_count >= eligible) for key in keys}
    ideal = {key: val_count * len(strata[key]) / len(rows) for key in keys}
    shares = {key: max(lower[key], min(capacities[key], math.floor(ideal[key]))) for key in keys}
    while sum(shares.values()) < val_count:
        key = min(
            (key for key in keys if shares[key] < capacities[key]),
            key=lambda key: (shares[key] - ideal[key], key),
        )
        shares[key] += 1
    while sum(shares.values()) > val_count:
        key = min(
            (key for key in keys if shares[key] > lower[key]),
            key=lambda key: (ideal[key] - shares[key], key),
        )
        shares[key] -= 1
    rng = random.Random(seed)
    train, val, report = [], [], []
    for key in keys:
        group = strata[key]
        rng.shuffle(group)
        count = shares[key]
        val.extend(group[:count])
        train.extend(group[count:])
        report.append({
            "source_dataset": key[0], "token_band": f"<={key[1]}",
            "available": len(group), "train": len(group) - count, "val": count,
        })
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, report


def _open_private_text(path: Path) -> TextIO:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stream = os.fdopen(fd, "w", encoding="utf-8")
    try:
        os.fchmod(stream.fileno(), 0o600)
    except BaseException:
        stream.close()
        raise
    return stream


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with _open_private_text(path) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_directory(staged: Path, destination: Path) -> None:
    """Atomic no-replace rename, including protection against concurrent writers.

    Linux renameat2 is used rather than rename(), which can replace an existing
    empty directory. Fail closed on platforms without this primitive.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise NotImplementedError("atomic no-replace directory publication requires renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, "output directory already exists", str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def prepare_combined(
    inputs: Mapping[str, Path],
    output_dir: Path,
    tokenizer: Any,
    max_length: int = 8192,
    val_count: int = 2000,
    seed: int = 42,
    *,
    eval_inputs: Sequence[Path],
    tokenizer_provenance: dict[str, Any],
    review_exclusions: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate, filter, deduplicate, stratify and atomically publish prepared chats.

    Known v2 system instructions are migrated to the current contract; all other
    messages remain exact. Complete chats (including assistant and template EOS)
    are batch tokenized without truncation. All retained rows belong to exactly
    one split. Missing terminal punctuation is flagged, not excluded. Extra
    input fields, including upstream quality flags and historical leakage_score,
    are preserved; fresh containment measurements are reported in metadata. The
    source mapping label and measured token count replace stale corresponding
    fields. ``eval_inputs`` and ``tokenizer_provenance`` are explicit so callers
    must consciously identify the frozen evaluation files and tokenizer. Tokenizer
    provenance is caller-attested metadata, not independently verified here.
    ``review_exclusions`` is an optional source -> document ID -> reason-code map
    from a separate quality review. Every requested ID must exist in its source;
    no automatic semantic or length-based target-quality exclusion is inferred.

    ``val_count`` may be zero, but must leave one train row in every populated
    source/token-band stratum. Output publication currently requires Linux's
    renameat2; existing directories, files and dangling symlinks are refused.
    """
    if not inputs or any(not isinstance(label, str) or not label for label in inputs):
        raise ValueError("inputs must contain non-empty string source labels")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if isinstance(val_count, bool) or not isinstance(val_count, int) or val_count < 0:
        raise ValueError("val_count must be a non-negative integer")
    if not isinstance(tokenizer_provenance, dict):
        raise ValueError("tokenizer_provenance must be a dict")
    if review_exclusions is None:
        review_exclusions = {}
    if not isinstance(review_exclusions, Mapping):
        raise ValueError("review_exclusions must be a mapping")
    reviewed: dict[str, dict[str, str]] = {}
    for source, entries in review_exclusions.items():
        if source not in inputs:
            raise ValueError("review_exclusions contains an unknown source")
        if not isinstance(entries, Mapping):
            raise ValueError("review_exclusions source entries must be mappings")
        reviewed[source] = {}
        for identifier, reason in entries.items():
            if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{16}", identifier) is None:
                raise ValueError("review exclusion ID must be a 16-character document digest")
            if not isinstance(reason, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason) is None:
                raise ValueError("review exclusion reason code must be lowercase snake_case, at most 64 characters")
            reviewed[source][identifier] = reason
    requested_reviews = {(source, identifier) for source, entries in reviewed.items() for identifier in entries}
    matched_reviews: set[tuple[str, str]] = set()
    output_dir = Path(output_dir).absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"output directory already exists: {output_dir}")
    decoder_tokens = [
        entry.content for entry in getattr(tokenizer, "added_tokens_decoder", {}).values()
        if getattr(entry, "special", False)
    ]
    # Granite's role boundaries are special decoder tokens but are absent from
    # all_special_tokens. Looking only at that convenience list misses them.
    special_tokens = tuple(token for token in [*tokenizer.all_special_tokens, *decoder_tokens] if token)
    if any(not isinstance(token, str) for token in special_tokens):
        raise ValueError("tokenizer.all_special_tokens must contain strings")
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    exclusions_stream = None
    try:
        os.chmod(staged, 0o700)
        eval_reports, eval_documents = [], []
        for path in eval_inputs:
            path = Path(path)
            eval_rows = load_jsonl(path)
            eval_reports.append({"path": str(path), "sha256": sha256_file(path), "rows": len(eval_rows)})
            eval_documents.extend(row["document"] for row in eval_rows)
        eval_grams = build_eval_ngram_set(eval_documents, size=NGRAM_SIZE)
        eval_document_count = len(eval_documents)
        del eval_documents
        source_counts = {
            source: {"input": 0, "retained": 0, "excluded": 0, "train": 0, "val": 0,
                     "target_conflicts": 0, "by_reason": Counter()}
            for source in inputs
        }
        reasons: Counter[str] = Counter()
        retained_warnings: Counter[str] = Counter()
        exclusions_stream = _open_private_text(staged / "exclusions.jsonl")
        accepted: list[dict[str, Any]] = []
        pending: list[tuple[dict[str, Any], str, int, float]] = []
        seen: dict[str, str] = {}
        input_reports = {}
        max_seen_overlap = 0.0
        max_retained_overlap = 0.0

        def exclude(row: dict[str, Any], source: str, line: int, reason: str) -> None:
            exclusion = {"id": row["id"], "source": source, "input_line": line, "reason": reason}
            exclusions_stream.write(json.dumps(exclusion, ensure_ascii=False) + "\n")
            reasons[reason] += 1
            source_counts[source]["excluded"] += 1
            source_counts[source]["by_reason"][reason] += 1

        def flush() -> None:
            nonlocal max_retained_overlap
            if not pending:
                return
            tokenized = tokenizer.apply_chat_template(
                [row["messages"] for row, _, _, _ in pending],
                tokenize=True, add_generation_prompt=False, truncation=False,
            )
            if len(tokenized) != len(pending):
                raise ValueError("tokenizer returned the wrong batch size")
            for (row, source, line, overlap), tokens in zip(pending, tokenized):
                if not isinstance(tokens, (list, tuple)) or any(
                    isinstance(token, bool) or not isinstance(token, int) for token in tokens
                ) or not tokens:
                    raise ValueError("tokenizer must return a batch of non-empty token ID lists")
                count = len(tokens)
                if count > max_length:
                    exclude(row, source, line, "overlength")
                    continue
                warnings = _target_warnings(row["messages"][2]["content"])
                retained_warnings.update(warnings)
                accepted.append({**row, "source_dataset": source, "token_count": count,
                                 "preparation_warnings": warnings})
                source_counts[source]["retained"] += 1
                max_retained_overlap = max(max_retained_overlap, overlap)
            pending.clear()

        for source, path in inputs.items():
            path = Path(path)
            input_reports[source] = {"path": str(path), "sha256": sha256_file(path)}
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(f"source {source!r}, input line {line_number}: invalid JSON") from None
                    document, target = _validate_row(row, source, line_number)
                    reviewed_reason = reviewed.get(source, {}).get(row["id"])
                    if reviewed_reason is not None:
                        matched_reviews.add((source, row["id"]))
                    previous_prompt = row["messages"][0]["content"]
                    row = {
                        **row,
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *row["messages"][1:]],
                        "source_prompt_version": PROMPT_VERSION_V2 if previous_prompt == SYSTEM_PROMPT_V2 else PROMPT_VERSION,
                    }
                    source_counts[source]["input"] += 1
                    doc_hash = _digest(" ".join(document.split()).casefold())
                    target_hash = _digest(target)
                    if doc_hash in seen:
                        if seen[doc_hash] != target_hash:
                            source_counts[source]["target_conflicts"] += 1
                        exclude(row, source, line_number, "duplicate_document")
                        continue
                    seen[doc_hash] = target_hash
                    if reviewed_reason is not None:
                        exclude(row, source, line_number, reviewed_reason)
                        continue
                    reason = _quarantine_reason(document, target, special_tokens)
                    if reason:
                        exclude(row, source, line_number, reason)
                        continue
                    overlap = containment(document, eval_grams, size=NGRAM_SIZE)
                    max_seen_overlap = max(max_seen_overlap, overlap)
                    if overlap >= LEAKAGE_THRESHOLD:
                        exclude(row, source, line_number, "eval_overlap")
                        continue
                    pending.append((row, source, line_number, overlap))
                    if len(pending) >= BATCH_SIZE:
                        flush()
            input_reports[source]["rows"] = source_counts[source]["input"]
        if requested_reviews != matched_reviews:
            raise ValueError("review exclusion IDs were not found in their named source inputs")
        flush()
        exclusions_stream.flush()
        os.fsync(exclusions_stream.fileno())
        exclusions_stream.close()
        excluded_count = sum(reasons.values())
        train, val, strata = _split(accepted, val_count, seed)
        for split_name, rows in (("train", train), ("val", val)):
            for row in rows:
                source_counts[row["source_dataset"]][split_name] += 1
        outputs = {}
        for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
            _write_jsonl(staged / name, rows)
            outputs[name] = {"sha256": sha256_file(staged / name), "rows": len(rows)}
        outputs["exclusions.jsonl"] = {"sha256": sha256_file(staged / "exclusions.jsonl"), "rows": excluded_count}
        metadata = {
            "schema_version": 1, "prompt_version": PROMPT_VERSION,
            "inputs": input_reports, "tokenizer_provenance": tokenizer_provenance,
            "max_length": max_length, "val_count": val_count, "seed": seed,
            "system_prompt": SYSTEM_PROMPT,
            "target_policy": "preserve targets regardless of length, paragraphs or punctuation; flag missing terminal punctuation for review",
            "retained_warnings": dict(sorted(retained_warnings.items())),
            "review_exclusions": {
                "requested": len(requested_reviews), "matched": len(matched_reviews),
                "sha256": _digest(json.dumps(reviewed, sort_keys=True, separators=(",", ":"))),
            },
            "deduplication": {"normalization": "whitespace-collapse then casefold", "keep": "first input row"},
            "evaluation": {
                "inputs": eval_reports, "documents": eval_document_count, "unique_ngrams": len(eval_grams),
                "ngram_size": NGRAM_SIZE, "containment_threshold": LEAKAGE_THRESHOLD,
                "max_seen_containment": max_seen_overlap, "max_retained_containment": max_retained_overlap,
            },
            "counts": {
                "input": sum(counts["input"] for counts in source_counts.values()),
                "retained": len(accepted), "train": len(train), "val": len(val),
                "excluded": excluded_count,
                "target_conflicts": sum(counts["target_conflicts"] for counts in source_counts.values()),
                "by_reason": dict(sorted(reasons.items())),
                "by_source": {source: {**counts, "by_reason": dict(sorted(counts["by_reason"].items()))}
                              for source, counts in source_counts.items()},
            },
            "strata": strata, "outputs": outputs,
        }
        # A JSON object plus newline is also valid one-record JSONL.
        _write_jsonl(staged / "metadata.json", [metadata])
        directory_fd = os.open(staged, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _publish_directory(staged, output_dir)
        return metadata
    finally:
        if exclusions_stream is not None:
            exclusions_stream.close()
        if staged.exists():
            shutil.rmtree(staged)
