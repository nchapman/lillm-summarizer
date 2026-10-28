"""Core helpers for preparing, generating, and scoring summarization runs."""

from __future__ import annotations

import hashlib
import http.client
import json
import random
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROMPT_VERSION = "summary-v1"
SYSTEM_PROMPT = (
    "You are a careful summarizer. Summarize the source using only information "
    "supported by it. Preserve the most important facts and avoid adding claims. "
    "Return only the summary, without a preamble."
)
USER_PROMPT_TEMPLATE = "Summarize the following document in 2-4 sentences.\n\n{document}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and validate the fields required by the harness."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each record must be a JSON object")
            for field in ("id", "document", "reference"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{path}:{line_number}: field {field!r} must be a non-empty string")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def sample_stream(records: Any, count: int, seed: int) -> list[dict[str, Any]]:
    """Select a reproducible reservoir sample from an iterable dataset stream."""
    if count <= 0:
        raise ValueError("count must be greater than zero")
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    valid_seen = 0
    for record in records:
        candidate = {
            "id": str(record.get("id") or record.get("article_id") or valid_seen),
            "document": record.get("article") or record.get("document"),
            "reference": record.get("highlights") or record.get("summary"),
        }
        if not candidate["document"] or not candidate["reference"]:
            continue
        valid_seen += 1
        if len(sample) < count:
            sample.append(candidate)
        else:
            slot = rng.randrange(valid_seen)
            if slot < count:
                sample[slot] = candidate
    if len(sample) < count:
        raise ValueError(f"Requested {count} records but found only {len(sample)} valid records")
    return sample


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_summary(
    base_url: str,
    model: str,
    document: str,
    max_new_tokens: int,
    timeout: float,
) -> tuple[str, dict[str, Any], float]:
    """Request one chat completion from a llama.cpp-compatible endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(document=document)},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Inference endpoint returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        reason = getattr(exc, "reason", str(exc))
        raise RuntimeError(f"Inference request to {url} failed: {reason}") from exc
    elapsed = time.perf_counter() - started

    try:
        choice = payload["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat-completions response: {payload!r}") from exc
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Inference endpoint returned an empty summary")
    usage = payload.get("usage") or {}
    return text.strip(), {"usage": usage, "finish_reason": choice.get("finish_reason")}, elapsed
