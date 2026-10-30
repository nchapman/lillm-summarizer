"""Training input selection must not depend on the split file's stratum ordering."""
import json
from pathlib import Path

import pytest

from lillm_eval.training_data import load_chat_rows


def write_rows(path: Path, count: int = 100) -> None:
    records = [
        {
            "id": str(i),
            "messages": [
                {"role": "system", "content": "Summarize."},
                {"role": "user", "content": "email" if i < count // 2 else "news"},
                {"role": "assistant", "content": f"summary {i}"},
            ],
        }
        for i in range(count)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_default_loads_entire_validation_split(tmp_path):
    path = tmp_path / "validation.jsonl"
    write_rows(path, count=1000)
    rows = load_chat_rows(path)
    assert len(rows) == 1000
    assert rows[-1]["id"] == "999"
    assert rows[0]["prompt"][0] == {"role": "system", "content": "Summarize."}
    assert rows[0]["completion"] == [{"role": "assistant", "content": "summary 0"}]


def test_optional_limit_samples_whole_file_not_first_stratum(tmp_path):
    path = tmp_path / "validation.jsonl"
    write_rows(path)
    rows = load_chat_rows(path, limit=20)
    assert len(rows) == 20
    assert len({r["id"] for r in rows}) == 20
    assert {r["prompt"][-1]["content"] for r in rows} == {"email", "news"}
    assert rows == load_chat_rows(path, limit=20)
    assert rows != load_chat_rows(path, limit=20, seed=43)


def test_limit_larger_than_file_keeps_all_rows(tmp_path):
    path = tmp_path / "validation.jsonl"
    write_rows(path, count=2)
    assert len(load_chat_rows(path, limit=20)) == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_rejects_nonpositive_limit(tmp_path, limit):
    path = tmp_path / "validation.jsonl"
    write_rows(path)
    with pytest.raises(ValueError, match="^limit must be positive"):
        load_chat_rows(path, limit=limit)


def test_rejects_empty_file(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.touch()
    with pytest.raises(ValueError, match="No rows"):
        load_chat_rows(path)
