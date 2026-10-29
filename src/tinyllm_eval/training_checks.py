"""CPU-only, privacy-safe checks for fresh prepared training runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def check_fresh(adapter_dir: Path, resume: Path | None) -> None:
    if resume is not None:
        raise ValueError("prepared runs cannot resume")
    if adapter_dir.exists() and (not adapter_dir.is_dir() or any(adapter_dir.iterdir())):
        raise ValueError("prepared runs require an empty adapter directory")


def load_prepared(manifest: Path, train: Path, val: Path, max_length: int,
                  system_prompt: str, prompt_version: str) -> tuple[list[dict], list[dict]]:
    """Validate full splits before sampling; never expose private values in errors."""
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if (meta["prompt_version"] != "summary-v3" or prompt_version != "summary-v3"
                or meta["system_prompt"] != system_prompt or meta["max_length"] != max_length):
            raise ValueError()
        splits, documents = [], []
        for name, path in (("train", train), ("val", val)):
            raw = path.read_bytes()
            spec = meta["outputs"][name + ".jsonl"]
            if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
                raise ValueError()
            # JSONL is delimited by LF; splitlines() also splits valid Unicode
            # separators embedded in JSON strings (e.g. U+2028), corrupting rows.
            rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
            if not rows or len(rows) != spec["rows"] or len(rows) != meta["counts"][name]:
                raise ValueError()
            converted, docs = [], set()
            for row in rows:
                messages = row["messages"]
                if ([m["role"] for m in messages] != ["system", "user", "assistant"]
                        or any(set(m) != {"role", "content"} for m in messages)
                        or any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages)
                        or messages[0]["content"] != system_prompt):
                    raise ValueError()
                source, count = row["source_dataset"], row["token_count"]
                if not isinstance(source, str) or not source.strip() or type(count) is not int or not 0 < count <= max_length:
                    raise ValueError()
                document = messages[1]["content"]
                if row["id"] != hashlib.sha256(document.encode()).hexdigest()[:16]:
                    raise ValueError()
                docs.add(hashlib.sha256(" ".join(document.split()).casefold().encode()).digest())
                converted.append({"prompt": messages[:2], "completion": messages[2:],
                                  "id": row["id"], "source_dataset": source, "token_count": count})
            splits.append(converted)
            documents.append(docs)
        if documents[0] & documents[1]:
            raise ValueError()
        return splits[0], splits[1]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("prepared data validation failed (manifest, splits, or prompt mismatch)") from None


def check_sequence(input_ids, labels, expected_length: int, eos_token_id: int) -> int:
    """Check actual collated labels; EOS need not be the final token (Granite)."""
    if len(input_ids) != expected_length or len(labels) != expected_length:
        raise ValueError("prepared sequence length mismatch")
    count = sum(label != -100 for label in labels)
    if (any(label not in (-100, token) for token, label in zip(input_ids, labels))
            or not any(token != eos_token_id and label == token for token, label in zip(input_ids, labels))
            or not any(token == eos_token_id and label == token for token, label in zip(input_ids, labels))):
        raise ValueError("prepared sequence lacks supervised target or EOS")
    return count


def source_groups(rows: list[dict]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    # Opaque keys avoid metric collisions and printing private source labels.
    mapping = {f"source_{i}": source for i, source in enumerate(sorted({r['source_dataset'] for r in rows}))}
    return ({"all": rows, **{key: [r for r in rows if r["source_dataset"] == source]
                              for key, source in mapping.items()}}, mapping)
