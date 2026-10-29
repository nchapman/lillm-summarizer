"""Local-only Folio imports preserve targets and verify the paired audit trail."""
import gzip
import hashlib
import json
import os

import pytest

from tinyllm_eval.cli import build_parser
from tinyllm_eval.core import SYSTEM_PROMPT, sha256_file
from tinyllm_eval.folio_data import prepare_folio
from tinyllm_eval.training_data import load_chat_rows

TEACHER_PROMPT = SYSTEM_PROMPT + " Write in plain prose without markdown formatting."


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export(tmp_path, pairs, *, change_audit=None, change_messages=None):
    source = tmp_path / "source"
    source.mkdir()
    with gzip.open(source / "sft.jsonl.gz", "wt", encoding="utf-8") as sft, gzip.open(
        source / "audit.jsonl.gz", "wt", encoding="utf-8"
    ) as audit:
        for index, (document, summary) in enumerate(pairs):
            messages = [
                {"role": "system", "content": TEACHER_PROMPT},
                {"role": "user", "content": document},
                {"role": "assistant", "content": summary},
            ]
            row = {
                "sft_row": index,
                "source_sha256": digest(document),
                "summary_sha256": digest(summary),
                "flags": ["framing"] if index == 0 else [],
                "summary_changed": True,
                "finish_reason": "stop",
                "completion_tokens": 30,
                "canonical_url": "https://private.invalid/never-copy",
                "metadata": {"title": "Never copy this private title"},
            }
            if change_messages:
                change_messages(messages)
            if change_audit:
                change_audit(row)
            sft.write(json.dumps({"messages": messages}) + "\n")
            audit.write(json.dumps(row) + "\n")
    report = {
        "kept": len(pairs),
        "status": "provisional_pending_source_and_faithfulness_review",
        "teacher_model": "test-teacher",
        "system_prompt": TEACHER_PROMPT,
        "output_sha256": {name: sha256_file(source / name) for name in ("sft.jsonl.gz", "audit.jsonl.gz")},
    }
    (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({
        "id": "frozen", "document": "A frozen evaluation article not used for training.", "reference": "Frozen summary."
    }) + "\n")
    return source, eval_path


def run(source, eval_path, output, **kwargs):
    return prepare_folio(source, output, eval_inputs=[eval_path], **kwargs)


def test_aligns_only_system_and_retains_verified_provenance(tmp_path):
    document = "  Raw café page.\n\nKeep this paragraph and whitespace.  "
    summary = "  An exact target — no stripping.\n"
    source, eval_path = export(tmp_path, [(document, summary)])
    output = tmp_path / "aligned.jsonl"
    metadata = run(source, eval_path, output)
    row = json.loads(output.read_text())
    assert row["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": document},
        {"role": "assistant", "content": summary},
    ]
    assert row["id"] == digest(document)[:16]
    assert row["source_row"] == 0
    assert row["source_sha256"] == digest(document)
    assert row["summary_sha256"] == digest(summary)
    assert row["quality_flags"] == ["framing"]
    assert row["tag"] == "folio"
    assert "canonical_url" not in row and "metadata" not in row
    assert metadata["source_status"].startswith("provisional")
    assert metadata["source_system_prompt"] == TEACHER_PROMPT
    assert metadata["split"] == "unsplit"
    assert metadata["output_sha256"] == sha256_file(output)
    assert metadata["report"]["kept"] == 1
    assert load_chat_rows(output)[0]["completion"] == [{"role": "assistant", "content": summary}]
    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.with_suffix(".jsonl.meta.json").read_text()) == metadata


def test_deduplicates_and_filters_eval_overlap_without_changing_targets(tmp_path):
    frozen = "A frozen evaluation article not used for training."
    source, eval_path = export(tmp_path, [("Unique local source.", "First target."),
                                          ("Unique local source.", "Other target."),
                                          (frozen, "Leaked target.")])
    output = tmp_path / "aligned.jsonl"
    report = run(source, eval_path, output)["report"]
    assert report["total_input_rows"] == 3
    assert report["dropped_duplicate"] == 1
    assert report["dropped_leakage"] == 1
    assert report["kept"] == 1
    assert json.loads(output.read_text())["messages"][-1]["content"] == "First target."


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(sft_row=8), "sft_row"),
    (lambda r: r.update(source_sha256="wrong"), "source_sha256"),
    (lambda r: r.update(summary_sha256="wrong"), "summary_sha256"),
])
def test_rejects_misaligned_audit_and_does_not_publish_partial_output(tmp_path, mutation, match):
    source, eval_path = export(tmp_path, [("A document.", "A target.")], change_audit=mutation)
    output = tmp_path / "aligned.jsonl"
    with pytest.raises(ValueError, match=match):
        run(source, eval_path, output)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.meta.json").exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m[0].update(content="Unexpected teacher prompt"), "system prompt"),
    (lambda m: m[2].update(role="user"), "roles"),
    (lambda m: m[1].update(content="   "), "non-empty"),
    (lambda m: m.append({"role": "assistant", "content": "Extra turn"}), "roles"),
])
def test_rejects_prompt_drift_and_malformed_chats(tmp_path, mutation, match):
    source, eval_path = export(tmp_path, [("A document.", "A target.")], change_messages=mutation)
    with pytest.raises(ValueError, match=match):
        run(source, eval_path, tmp_path / "aligned.jsonl")


def test_verifies_compressed_input_checksums(tmp_path):
    source, eval_path = export(tmp_path, [("A document.", "A target.")])
    with (source / "sft.jsonl.gz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        run(source, eval_path, tmp_path / "aligned.jsonl")


def test_protects_existing_output_metadata_and_inputs(tmp_path):
    source, eval_path = export(tmp_path, [("A document.", "A target.")])
    output = tmp_path / "aligned.jsonl"
    output.with_suffix(".jsonl.meta.json").write_text("preserve")
    with pytest.raises(FileExistsError, match="overwrite"):
        run(source, eval_path, output)
    assert output.with_suffix(".jsonl.meta.json").read_text() == "preserve"
    original = (source / "sft.jsonl.gz").read_bytes()
    with pytest.raises(ValueError, match="input"):
        run(source, eval_path, source / "sft.jsonl.gz", overwrite=True)
    assert (source / "sft.jsonl.gz").read_bytes() == original


@pytest.mark.parametrize("extra", [False, True])
def test_rejects_unequal_audit_counts(tmp_path, extra):
    source, eval_path = export(tmp_path, [("A document.", "A target.")])
    audit_path = source / "audit.jsonl.gz"
    with gzip.open(audit_path, "rt", encoding="utf-8") as stream:
        line = stream.read()
    with gzip.open(audit_path, "wt", encoding="utf-8") as stream:
        stream.write(line * 2 if extra else "")
    report_path = source / "report.json"
    report = json.loads(report_path.read_text())
    report["output_sha256"]["audit.jsonl.gz"] = sha256_file(audit_path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="row counts differ"):
        run(source, eval_path, tmp_path / "aligned.jsonl")


def test_failed_overwrite_preserves_both_old_files(tmp_path):
    source, eval_path = export(tmp_path, [("A document.", "A target.")],
                               change_audit=lambda r: r.update(sft_row=7))
    output = tmp_path / "aligned.jsonl"
    sidecar = output.with_suffix(".jsonl.meta.json")
    output.write_text("old pool")
    sidecar.write_text("old metadata")
    with pytest.raises(ValueError, match="sft_row"):
        run(source, eval_path, output, overwrite=True)
    assert output.read_text() == "old pool"
    assert sidecar.read_text() == "old metadata"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["aligned.jsonl", "aligned.jsonl.meta.json", "eval.jsonl", "source"]


@pytest.mark.parametrize("existing_pool", [False, True])
@pytest.mark.parametrize("existing_metadata", [False, True])
def test_metadata_publication_failure_restores_previous_state(
    tmp_path, monkeypatch, existing_pool, existing_metadata
):
    source, eval_path = export(tmp_path, [("A document.", "A target.")])
    output = tmp_path / "aligned.jsonl"
    sidecar = output.with_suffix(".jsonl.meta.json")
    if existing_pool:
        output.write_bytes(b"old pool\n")
        previous_inode = output.stat().st_ino
    if existing_metadata:
        sidecar.write_bytes(b"old metadata\n")
    original_entries = set(tmp_path.iterdir())
    real_replace = os.replace
    publication_attempts = []

    def fail_metadata_replace(src, dst):
        if dst == sidecar:
            publication_attempts.append(dst)
            assert json.loads(output.read_text())["messages"][-1]["content"] == "A target."
            raise OSError("injected metadata publication failure")
        return real_replace(src, dst)

    monkeypatch.setattr("tinyllm_eval.folio_data.os.replace", fail_metadata_replace)
    with pytest.raises(OSError, match="injected metadata publication failure"):
        run(source, eval_path, output, overwrite=True)

    assert publication_attempts == [sidecar]
    if existing_pool:
        assert output.read_bytes() == b"old pool\n"
        assert output.stat().st_ino == previous_inode
    else:
        assert not output.exists()
    if existing_metadata:
        assert sidecar.read_bytes() == b"old metadata\n"
    else:
        assert not sidecar.exists()
    assert set(tmp_path.iterdir()) == original_entries


def test_successful_overwrite_replaces_both_files_and_cleans_backup(tmp_path):
    source, eval_path = export(tmp_path, [("A document.", "A target.")])
    output = tmp_path / "aligned.jsonl"
    sidecar = output.with_suffix(".jsonl.meta.json")
    output.write_bytes(b"old pool\n")
    sidecar.write_bytes(b"old metadata\n")
    original_entries = set(tmp_path.iterdir())

    metadata = run(source, eval_path, output, overwrite=True)

    assert json.loads(output.read_text())["messages"][-1]["content"] == "A target."
    assert json.loads(sidecar.read_text()) == metadata
    assert metadata["output_sha256"] == sha256_file(output)
    assert set(tmp_path.iterdir()) == original_entries


@pytest.mark.parametrize("destination", ["pool", "sidecar"])
@pytest.mark.parametrize("existing_pool", [False, True])
def test_rejects_destination_directories_before_publication(
    tmp_path, monkeypatch, destination, existing_pool
):
    source, eval_path = export(tmp_path, [("A document.", "A target.")])
    output = tmp_path / "aligned.jsonl"
    sidecar = output.with_suffix(".jsonl.meta.json")
    directory = output if destination == "pool" else sidecar
    directory.mkdir()
    (directory / "keep").write_bytes(b"directory contents")
    if destination == "sidecar" and existing_pool:
        output.write_bytes(b"old pool\n")
    original_entries = set(tmp_path.iterdir())

    def unexpected_replace(src, dst):
        pytest.fail("destination directories must be rejected before publication")

    monkeypatch.setattr("tinyllm_eval.folio_data.os.replace", unexpected_replace)
    with pytest.raises(IsADirectoryError):
        run(source, eval_path, output, overwrite=True)

    assert (directory / "keep").read_bytes() == b"directory contents"
    if destination == "sidecar":
        if existing_pool:
            assert output.read_bytes() == b"old pool\n"
        else:
            assert not output.exists()
    assert set(tmp_path.iterdir()) == original_entries


def test_rejects_empty_inputs_and_empty_filtered_pool(tmp_path):
    source, eval_path = export(tmp_path, [("A frozen evaluation article not used for training.", "A target.")])
    output = tmp_path / "aligned.jsonl"
    with pytest.raises(ValueError, match="at least one"):
        prepare_folio(source, output, eval_inputs=[])
    with pytest.raises(ValueError, match="no rows retained"):
        run(source, eval_path, output)
    assert not output.exists()


def test_cli_uses_existing_frozen_eval_defaults_and_new_separate_output(tmp_path, capsys):
    source, eval_path = export(tmp_path, [("A document.", "A target.")])
    args = build_parser().parse_args(["prepare-folio", "--source-dir", str(source)])
    assert args.output.as_posix() == "data/train/folio_sft_v1/aligned.jsonl"
    assert args.eval_input is None
    args = build_parser().parse_args([
        "prepare-folio", "--source-dir", str(source), "--output", str(tmp_path / "aligned.jsonl"),
        "--eval-input", str(eval_path),
    ])
    args.func(args)
    assert "unsplit aligned pool" in capsys.readouterr().out
