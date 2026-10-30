"""Prepare LoRA training data from the smoltalk2 smol-summarize subset.

The subset ``smoltalk_smollm3_smol_summarize_no_think`` mixes news-article and
email summarization, but the regeneration for SmolLM3 erased that distinction
from the instructions: every row carries one of two generic length contracts
("up to three sentences" or "one very short sentence") stored in
``chat_template_kwargs.custom_instructions``. Rows are ``[user, assistant]``
messages. Prep keeps every row, tags it by instruction variant, and prepends a
single fixed system prompt — the same instruction used at evaluation and
deployment time — so training, evaluation, and the summarize button all
present the same contract.

News articles come from the CNN/DailyMail 3.0.0 *train* split while the frozen
evaluation sets sample the *test* split; an 8-gram containment check against
the evaluation documents drops near-duplicate articles before training.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

NGRAM_SIZE = 8
LEAKAGE_THRESHOLD_DEFAULT = 0.2
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[a-z0-9']+")


# The two instruction variants observed in the smoltalk2 smol-summarize split;
# unknown instructions are tagged "other" so upstream changes surface in the report.
INSTRUCTION_TAG_MARKERS = {
    "three_sentence": ("up to three sentences",),
    "one_line": ("one very short sentence",),
}


def load_chat_rows(path: Path, limit: int | None = None, *, seed: int = 42) -> list[dict]:
    """Load all rows, or a reproducible uniform sample rather than a sorted prefix.

    Stratified split files are grouped by stratum; taking their first N rows
    silently biases validation toward one instruction/length band.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            messages = record["messages"]
            rows.append(
                {
                    "prompt": [{"role": m["role"], "content": m["content"]} for m in messages[:-1]],
                    "completion": [{"role": "assistant", "content": messages[-1]["content"]}],
                    "id": record["id"],
                }
            )
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def classify_instruction(instruction: str) -> str:
    """Tag a row by its original generation instruction.

    The split carries two generic length contracts; content type (news vs
    email) is not recoverable from the instruction, so tags describe the
    instruction variant. An absent instruction is tagged ``unknown``.
    """
    if not instruction.strip():
        return "unknown"
    lowered = instruction.lower()
    for tag, markers in INSTRUCTION_TAG_MARKERS.items():
        if any(marker in lowered for marker in markers):
            return tag
    return "other"


def extract_pair(messages: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """Return ``(user, assistant)`` from a single-turn chat row."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("messages must be a sequence of message objects")
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("each message must be an object")
    roles = [message.get("role") for message in messages]
    if roles != ["user", "assistant"]:
        raise ValueError(f"expected [user, assistant] messages, found roles {roles}")
    contents: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("each message must contain non-empty string content")
        contents.append(content)
    return contents[0], contents[1]


def original_instruction(record: dict[str, Any]) -> str:
    """Return the pipeline instruction stored in ``chat_template_kwargs``."""
    kwargs = record.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        return ""
    instruction = kwargs.get("custom_instructions")
    return instruction if isinstance(instruction, str) else ""


def word_ngrams(text: str, size: int = NGRAM_SIZE) -> list[tuple[str, ...]]:
    """Return normalized lowercase word n-grams of ``text``."""
    words = WORD_PATTERN.findall(text.lower())
    if not words:
        return []
    if len(words) < size:
        return [tuple(words)]
    return [tuple(words[index : index + size]) for index in range(len(words) - size + 1)]


def build_eval_ngram_set(documents: Iterable[str], size: int = NGRAM_SIZE) -> set[tuple[str, ...]]:
    """Collect every n-gram of the evaluation documents into one lookup set."""
    grams: set[tuple[str, ...]] = set()
    for document in documents:
        grams.update(word_ngrams(document, size))
    return grams


def containment(text: str, reference_grams: set[tuple[str, ...]], size: int = NGRAM_SIZE) -> float:
    """Fraction of ``text``'s n-grams that also appear in the reference set."""
    grams = word_ngrams(text, size)
    if not grams:
        return 0.0
    hits = sum(1 for gram in grams if gram in reference_grams)
    return hits / len(grams)


def count_sentences(text: str) -> int:
    """Rough sentence count used for target-shape statistics."""
    return len([part for part in SENTENCE_SPLIT.split(text.strip()) if part.strip()])


def _percentile(sorted_values: Sequence[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(round(fraction * (len(sorted_values) - 1)), len(sorted_values) - 1)
    return sorted_values[index]


def _bucket_sentences(counts: Iterable[int]) -> dict[str, int]:
    histogram: dict[str, int] = {str(number): 0 for number in range(1, 7)}
    histogram["7+"] = 0
    for count in counts:
        key = str(count) if count <= 6 else "7+"
        histogram[key] += 1
    return histogram


def stratified_subset(
    rows: list[dict[str, Any]],
    *,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split rows into train/val sets stratified by tag and document-length band.

    Allocation is proportional to stratum size (largest-remainder rounding,
    capped by availability with the shortfall redistributed), and sampling within
    each stratum is a seeded shuffle, so the split is deterministic for a given
    input file and seed.
    """
    if train_count <= 0 or val_count <= 0:
        raise ValueError("train_count and val_count must be greater than zero")
    if train_count + val_count > len(rows):
        raise ValueError(f"requested {train_count + val_count} rows but only {len(rows)} available")

    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["tag"], length_band(row["document_words"]))].append(row)
    for stratum in strata.values():
        rng.shuffle(stratum)

    total = train_count + val_count
    exact = {key: len(stratum) * total / len(rows) for key, stratum in strata.items()}
    targets = {key: math.floor(share) for key, share in exact.items()}
    leftover = total - sum(targets.values())
    for key in sorted(strata, key=lambda item: (-(exact[item] - targets[item]), item))[:leftover]:
        targets[key] += 1
    if sum(targets.values()) != total or any(targets[key] > len(strata[key]) for key in strata):
        raise RuntimeError("stratified allocation failed to match the requested row count")

    # Exact val split per stratum: floor shares, then distribute leftovers by
    # largest fractional part so the totals match val_count exactly.
    val_shares: dict[tuple[str, str], int] = {}
    fractional: list[tuple[float, tuple[str, str]]] = []
    for key in strata:
        raw = targets[key] * val_count / total
        val_shares[key] = min(targets[key], math.floor(raw))
        fractional.append((raw - math.floor(raw), key))
    leftover = val_count - sum(val_shares.values())
    for _, key in sorted(fractional, reverse=True)[:leftover] if leftover > 0 else []:
        if val_shares[key] < targets[key]:
            val_shares[key] += 1
            leftover -= 1
    capacity_order = sorted(strata, key=lambda key: (-(targets[key] - val_shares[key]), key))
    for key in capacity_order:
        while leftover > 0 and val_shares[key] < targets[key]:
            val_shares[key] += 1
            leftover -= 1

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "seed": seed,
        "total_input_rows": len(rows),
        "strata": {},
    }
    for key in sorted(strata):
        tag, band = key
        target = targets[key]
        val_share = val_shares[key]
        taken_val = strata[key][:val_share]
        taken_train = strata[key][val_share : val_share + (target - val_share)]
        val_rows.extend(taken_val)
        train_rows.extend(taken_train)
        report["strata"][f"{tag}/{band}"] = {
            "available": len(strata[key]),
            "train": len(taken_train),
            "val": len(taken_val),
        }
    report["train_rows"] = len(train_rows)
    report["val_rows"] = len(val_rows)
    return train_rows, val_rows, report


def length_band(document_words: int) -> str:
    """Coarse document-length band used for stratification."""
    if document_words < 100:
        return "short"
    if document_words <= 500:
        return "medium"
    return "long"


def prepare_rows(
    records: Iterable[dict[str, Any]],
    *,
    system_prompt: str,
    eval_ngrams: set[tuple[str, ...]],
    source_label: str,
    leakage_threshold: float = LEAKAGE_THRESHOLD_DEFAULT,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize, de-duplicate, and leakage-check raw smoltalk2 rows.

    Returns the training rows plus a report dict with counts, per-tag target
    shape statistics, and the highest retained leakage score.
    """
    if leakage_threshold < 0 or leakage_threshold > 1:
        raise ValueError("leakage_threshold must be between 0 and 1")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    report: dict[str, Any] = {
        "total_input_rows": 0,
        "kept": 0,
        "dropped_leakage": 0,
        "dropped_duplicate": 0,
        "invalid_rows": 0,
        "leakage_threshold": leakage_threshold,
        "leakage_max_seen": 0.0,
        "leakage_max_kept": 0.0,
        "tag_note": "tags describe the original instruction variant; news/email content type is not recoverable from the data",
        "by_tag": {},
    }
    tag_sentences: dict[str, list[int]] = {}
    tag_target_words: dict[str, list[int]] = {}
    tag_document_words: dict[str, list[int]] = {}
    tag_example_prompts: dict[str, list[str]] = {}

    for record in records:
        if limit is not None and report["total_input_rows"] >= limit:
            break
        report["total_input_rows"] += 1
        try:
            user_text, target = extract_pair(record.get("messages") or [])
        except ValueError:
            report["invalid_rows"] += 1
            continue

        row_id = hashlib.sha256(user_text.encode("utf-8")).hexdigest()[:16]
        if row_id in seen_ids:
            report["dropped_duplicate"] += 1
            continue

        score = containment(user_text, eval_ngrams)
        report["leakage_max_seen"] = max(report["leakage_max_seen"], score)
        if score >= leakage_threshold:
            report["dropped_leakage"] += 1
            continue

        tag = classify_instruction(original_instruction(record))
        seen_ids.add(row_id)
        target = target.strip()
        target_sentences = count_sentences(target)
        target_words = len(target.split())
        document_words = len(user_text.split())
        rows.append(
            {
                "id": row_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": target},
                ],
                "tag": tag,
                "target_sentences": target_sentences,
                "target_words": target_words,
                "document_words": document_words,
                "leakage_score": round(score, 4),
                "source": source_label,
            }
        )
        tag_sentences.setdefault(tag, []).append(target_sentences)
        tag_target_words.setdefault(tag, []).append(target_words)
        tag_document_words.setdefault(tag, []).append(document_words)
        examples = tag_example_prompts.setdefault(tag, [])
        instruction = original_instruction(record)
        if instruction and instruction not in examples and len(examples) < 3:
            examples.append(instruction)

    report["kept"] = len(rows)
    report["leakage_max_kept"] = max((row["leakage_score"] for row in rows), default=0.0)
    for tag in sorted(tag_sentences):
        target_words = sorted(tag_target_words[tag])
        document_words = sorted(tag_document_words[tag])
        report["by_tag"][tag] = {
            "rows": len(tag_sentences[tag]),
            "target_sentence_histogram": _bucket_sentences(tag_sentences[tag]),
            "target_word_percentiles": {
                "p25": _percentile(target_words, 0.25),
                "p50": _percentile(target_words, 0.50),
                "p75": _percentile(target_words, 0.75),
                "p95": _percentile(target_words, 0.95),
            },
            "document_word_percentiles": {
                "p50": _percentile(document_words, 0.50),
                "p95": _percentile(document_words, 0.95),
            },
            "example_instructions": tag_example_prompts[tag],
        }
    return rows, report


def format_report_summary(report: dict[str, Any]) -> str:
    """Render the prep report as a short human-readable summary."""
    lines = [
        "input rows: {total_input_rows} | kept: {kept} | "
        "dropped leakage: {dropped_leakage} | dropped duplicate: {dropped_duplicate} | "
        "invalid: {invalid_rows}".format(**report),
        "leakage: threshold {leakage_threshold:g}, max seen {leakage_max_seen:.3f}, "
        "max kept {leakage_max_kept:.3f}".format(**report),
    ]
    for tag, stats in report["by_tag"].items():
        histogram = " ".join(f"{key}:{count}" for key, count in stats["target_sentence_histogram"].items())
        target_p = stats["target_word_percentiles"]
        document_p = stats["document_word_percentiles"]
        lines.append(
            f"{tag}: {stats['rows']} rows | target sentences {histogram} | "
            f"target words p50/p95 {target_p['p50']}/{target_p['p95']} | "
            f"document words p50/p95 {document_p['p50']}/{document_p['p95']}"
        )
    return "\n".join(lines)
