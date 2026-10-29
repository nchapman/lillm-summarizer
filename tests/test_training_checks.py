import hashlib
import json
from types import SimpleNamespace

import pytest

from tinyllm_eval.training_checks import (
    check_fresh, check_sequence, checkpoint_schedule, load_prepared, source_groups,
    check_step_budget, apply_step_budget,
)


def fixture_files(tmp_path, change=None):
    meta = {"prompt_version": "summary-v3", "system_prompt": "system", "max_length": 8,
            "outputs": {}, "counts": {"train": 1, "val": 1}}
    paths = []
    for split, document in (("train", "Private Document"), ("val", "Other document")):
        row = {"id": hashlib.sha256(document.encode()).hexdigest()[:16],
               "messages": [{"role": role, "content": text} for role, text in
                            zip(("system", "user", "assistant"), ("system", document, "target"))],
               "source_dataset": "synthetic", "token_count": 4}
        if change:
            change(split, row, meta)
        path = tmp_path / (split + ".jsonl")
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        meta["outputs"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": 1}
        paths.append(path)
    manifest = tmp_path / "metadata.json"
    manifest.write_text(json.dumps(meta))
    return manifest, *paths


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_preserves_unicode_line_separators_inside_json_strings(tmp_path, separator):
    def change(split, row, meta):
        document = split + " first" + separator + "second"
        row["messages"][1]["content"] = document
        row["messages"][2]["content"] = "Summary" + separator + "continues."
        row["id"] = hashlib.sha256(document.encode()).hexdigest()[:16]
    train, val = load_prepared(*fixture_files(tmp_path, change), 8, "system", "summary-v3")
    assert train[0]["prompt"][1]["content"] == "train first" + separator + "second"
    assert val[0]["completion"][0]["content"] == "Summary" + separator + "continues."


def test_normal(tmp_path):
    train, val = load_prepared(*fixture_files(tmp_path), 8, "system", "summary-v3")
    groups, mapping = source_groups(val)
    assert groups["all"] == val
    assert mapping == {"source_0": "synthetic"}
    assert "source_dataset" not in train[0]["prompt"][0]


@pytest.mark.parametrize("field,value", [("token_count", 0), ("token_count", 9),
                                          ("token_count", True), ("source_dataset", ""), ("id", "private-id")])
def test_bad_row(tmp_path, field, value):
    files = fixture_files(tmp_path, lambda split, row, meta: row.update({field: value}))
    with pytest.raises(ValueError) as exc:
        load_prepared(*files, 8, "system", "summary-v3")
    assert "private" not in str(exc.value).lower()


@pytest.mark.parametrize("kind", ["hash", "count", "version", "length", "system", "overlap"])
def test_mismatches(tmp_path, kind):
    def change(split, row, meta):
        if kind == "overlap" and split == "val":
            doc = " private   DOCUMENT "
            row["messages"][1]["content"] = doc
            row["id"] = hashlib.sha256(doc.encode()).hexdigest()[:16]
        if kind == "system":
            row["messages"][0]["content"] = "wrong"
    manifest, train, val = fixture_files(tmp_path, change)
    meta = json.loads(manifest.read_text())
    if kind == "hash":
        meta["outputs"]["train.jsonl"]["sha256"] = "wrong"
    if kind == "count":
        meta["counts"]["train"] = 2
    if kind == "version":
        meta["prompt_version"] = "summary-v2"
    if kind == "length":
        meta["max_length"] = 7
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="prepared data validation failed"):
        load_prepared(manifest, train, val, 8, "system", "summary-v3")


def test_fresh(tmp_path):
    adapter = tmp_path / "adapter"
    check_fresh(adapter, None)
    adapter.mkdir()
    check_fresh(adapter, None)
    checkpoint = adapter / "checkpoint"
    checkpoint.write_text("preserve")
    with pytest.raises(ValueError):
        check_fresh(adapter, None)
    assert checkpoint.read_text() == "preserve"
    with pytest.raises(ValueError):
        check_fresh(tmp_path / "new", checkpoint)


def test_epoch_checkpoint_schedule_is_default():
    assert checkpoint_schedule(None) == {"eval_strategy": "epoch", "save_strategy": "epoch"}


def test_step_checkpoint_schedule_keeps_evaluation_and_saving_aligned():
    assert checkpoint_schedule(100) == {
        "eval_strategy": "steps", "save_strategy": "steps", "eval_steps": 100, "save_steps": 100,
    }


@pytest.mark.parametrize("steps", [0, -1, True, 0.5, "100"])
def test_checkpoint_schedule_rejects_invalid_intervals(steps):
    with pytest.raises(ValueError, match="positive integer"):
        checkpoint_schedule(steps)


@pytest.mark.parametrize("budget,interval", [(None, None), (None, 64), (512, 64), (64, 64)])
def test_step_budget_accepts_aligned_boundaries(budget, interval):
    check_step_budget(budget, interval)


@pytest.mark.parametrize("budget,interval", [(0, 64), (-1, 64), (True, 1), (1.5, 1),
                                               ("64", 64), (512, None), (512, 0), (513, 64)])
def test_step_budget_rejects_invalid_or_unaligned_boundaries(budget, interval):
    with pytest.raises(ValueError):
        check_step_budget(budget, interval)


@pytest.mark.parametrize("step", [64, 65])
def test_step_budget_requests_final_evaluation_save_and_normal_stop(step):
    control = SimpleNamespace(should_training_stop=False, should_evaluate=False, should_save=False)
    assert apply_step_budget(control, step, 64) is control
    assert control.should_training_stop and control.should_evaluate and control.should_save


@pytest.mark.parametrize("prior_stop", [False, True])
def test_step_budget_preserves_other_callbacks_before_boundary(prior_stop):
    control = SimpleNamespace(should_training_stop=prior_stop, should_evaluate=True, should_save=False)
    before = vars(control).copy()
    apply_step_budget(control, 63, 64)
    assert vars(control) == before


def test_labels():
    assert check_sequence([1, 2, 3, 4], [-100, 2, 3, 4], 4, 3) == 3
    for ids, labels, length in [([1, 3], [-100, 3], 2),
                                ([1, 2, 3], [-100, 9, 3], 3),
                                ([1, 3], [-100, 3], 3),
                                ([1, 3], [-100, -100], 2),
                                ([1, 3, 4], [-100, -100, 4], 3)]:
        with pytest.raises(ValueError):
            check_sequence(ids, labels, length, 3)
