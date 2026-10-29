"""Synthetic-only combined preparation tests; no model/data downloads."""

import copy
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyllm_eval import combined_data
from tinyllm_eval.combined_data import prepare_combined
from tinyllm_eval.core import PROMPT_VERSION, SYSTEM_PROMPT, SYSTEM_PROMPT_V2, sha256_file
from types import SimpleNamespace


def row(document, target="A synthetic summary.", **extra):
    return {
        "id": hashlib.sha256(document.encode()).hexdigest()[:16],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document},
            {"role": "assistant", "content": target},
        ],
        "tag": "synthetic", "document_words": len(document.split()),
        "target_words": len(target.split()), **extra,
    }


class FakeTokenizer:
    all_special_tokens = ["<eos>", "<|assistant|>"]

    def __init__(self, lengths=None):
        self.lengths = lengths or {}
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, truncation):
        # This signature rejects max_length, padding, and other unexpected kwargs.
        assert tokenize is True
        assert add_generation_prompt is False
        assert truncation is False
        assert isinstance(messages, list) and isinstance(messages[0], list)
        self.calls.append(copy.deepcopy(messages))
        result = []
        for chat in messages:
            assert [message["role"] for message in chat] == ["system", "user", "assistant"]
            # Include all contents, role boundaries and an EOS, not just the prompt.
            count = self.lengths.get(chat[1]["content"], sum(len(m["content"].split()) + 1 for m in chat) + 1)
            result.append([7] * count)
        return result


class CombinedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.provenance = {"name": "synthetic-tokenizer", "revision": "frozen", "chat_template_sha256": "abc"}

    def write_rows(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
        return path

    def read_rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def prepare(self, inputs, name="out", tokenizer=None, **kwargs):
        return prepare_combined(
            inputs, self.root / name, tokenizer or FakeTokenizer(),
            eval_inputs=kwargs.pop("eval_inputs", []),
            tokenizer_provenance=self.provenance, val_count=kwargs.pop("val_count", 0), **kwargs,
        )

    def assert_no_staging(self):
        self.assertFalse(list(self.root.glob(".*.staging-*")))

    def test_preserves_exact_messages_and_upstream_flags(self):
        original = row(
            "  Unicode Straße\tand a synthetic\r\npage.  ",
            "\nYou might prefer a verbose stylistic target!  ",
            quality_flags=["second_person", "long_target", "stylistic"],
            source_dataset="old", token_count=-1, custom={"untouched": True},
        )
        path = self.write_rows("prepared.jsonl", [original])
        tokenizer = FakeTokenizer()
        metadata = self.prepare({"news": path}, tokenizer=tokenizer)
        output = self.read_rows(self.root / "out/train.jsonl")[0]
        self.assertEqual(output["messages"], original["messages"])
        self.assertEqual(output["quality_flags"], original["quality_flags"])
        self.assertEqual(output["custom"], original["custom"])
        self.assertEqual(output["source_dataset"], "news")
        expected_count = sum(len(m["content"].split()) + 1 for m in original["messages"]) + 1
        self.assertEqual(output["token_count"], expected_count)
        self.assertEqual(tokenizer.calls, [[original["messages"]]])
        self.assertEqual(metadata["counts"]["excluded"], 0)

    def test_migrates_known_v2_system_without_changing_document_or_long_target(self):
        original = row("A legacy document.", "First paragraph.\n\nA longer second paragraph.")
        original["messages"][0]["content"] = SYSTEM_PROMPT_V2
        path = self.write_rows("legacy.jsonl", [original])
        tokenizer = FakeTokenizer()
        self.prepare({"legacy": path}, tokenizer=tokenizer)
        output = self.read_rows(self.root / "out/train.jsonl")[0]
        self.assertEqual(output["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(output["messages"][1:], original["messages"][1:])
        self.assertEqual(tokenizer.calls[0][0][0]["content"], SYSTEM_PROMPT)

    def test_applies_explicit_review_quarantine_without_rewriting_other_targets(self):
        suspect = row("A source with an unfinished teacher generation.", "This target was cut mid")
        valid = row("A substantial source.", "Complete long summary.\n\nAnother paragraph.")
        path = self.write_rows("reviewed.jsonl", [suspect, valid])
        metadata = self.prepare({"a": path}, review_exclusions={"a": {suspect["id"]: "suspected_teacher_cutoff"}})
        self.assertEqual(metadata["counts"]["by_reason"], {"suspected_teacher_cutoff": 1})
        output = self.read_rows(self.root / "out/train.jsonl")
        self.assertEqual([r["messages"] for r in output], [valid["messages"]])
        self.assertEqual(metadata["review_exclusions"]["requested"], 1)
        self.assertEqual(metadata["review_exclusions"]["matched"], 1)

    def test_rejects_review_exclusion_for_unknown_source_or_missing_id(self):
        path = self.write_rows("a.jsonl", [row("A clean source.")])
        with self.assertRaisesRegex(ValueError, "unknown source"):
            self.prepare({"a": path}, review_exclusions={"unknown": {"0" * 16: "reviewed_cutoff"}})
        with self.assertRaisesRegex(ValueError, "not found"):
            self.prepare({"a": path}, review_exclusions={"a": {"0" * 16: "reviewed_cutoff"}})
        self.assert_no_staging()
        self.assertFalse((self.root / "out").exists())

    def test_rejects_review_exclusion_reason_that_could_leak_text_into_reports(self):
        path = self.write_rows("a.jsonl", [row("A clean source.")])
        with self.assertRaisesRegex(ValueError, "reason code"):
            self.prepare({"a": path}, review_exclusions={"a": {"0" * 16: "Private article text!"}})
        self.assert_no_staging()

    def test_keeps_long_multiparagraph_and_unpunctuated_targets_with_review_warning(self):
        long_target = "A detailed summary sentence. " * 200 + "\n\nA concluding paragraph."
        rows = [row("Long article.", long_target), row("A short page.", "A short heading")]
        metadata = self.prepare({"a": self.write_rows("a.jsonl", rows)})
        output = {item["id"]: item for item in self.read_rows(self.root / "out/train.jsonl")}
        for original in rows:
            self.assertEqual(output[original["id"]]["messages"], original["messages"])
        self.assertEqual(output[rows[0]["id"]]["preparation_warnings"], [])
        self.assertEqual(output[rows[1]["id"]]["preparation_warnings"], ["missing_terminal_punctuation"])
        self.assertEqual(metadata["retained_warnings"], {"missing_terminal_punctuation": 1})
        self.assertEqual(metadata["counts"]["excluded"], 0)

    def test_quarantines_decoder_control_tokens_missing_from_all_special_tokens(self):
        tokenizer = FakeTokenizer()
        tokenizer.added_tokens_decoder = {17: SimpleNamespace(content="<|start_of_role|>", special=True)}
        rows = [row("Page with <|start_of_role|> in text."), row("A clean page.")]
        metadata = self.prepare({"a": self.write_rows("a.jsonl", rows)}, tokenizer=tokenizer)
        self.assertEqual(metadata["counts"]["by_reason"], {"special_token": 1})
        self.assertEqual(metadata["counts"]["retained"], 1)

    def test_first_wins_dedup_deterministic_disjoint_and_all_accepted_used(self):
        first = row("Straße   SYNTHETIC document", "First target")
        a = [first] + [row(f"source a document {i}") for i in range(15)]
        b = [row("STRASSE\nsynthetic\tdocument", "Conflicting target"), copy.deepcopy(first)]
        b += [row(f"source b document {i}") for i in range(16)]
        inputs = {"a": self.write_rows("a.jsonl", a), "b": self.write_rows("b.jsonl", b)}
        metadata = self.prepare(inputs, val_count=8, seed=17)
        repeated = self.prepare(inputs, "again", val_count=8, seed=17)
        self.assertEqual(metadata, repeated)
        for name in ("train.jsonl", "val.jsonl", "exclusions.jsonl", "metadata.json"):
            self.assertEqual((self.root / "out" / name).read_bytes(), (self.root / "again" / name).read_bytes())
        train = self.read_rows(self.root / "out/train.jsonl")
        val = self.read_rows(self.root / "out/val.jsonl")
        self.assertEqual((len(train), len(val)), (24, 8))
        self.assertFalse({item["id"] for item in train} & {item["id"] for item in val})
        self.assertEqual({item["id"] for item in train + val}, {item["id"] for item in a + b[2:]})
        normalized = [" ".join(item["messages"][1]["content"].split()).casefold() for item in train + val]
        self.assertEqual(len(normalized), len(set(normalized)))
        kept = next(item for item in train + val if item["id"] == first["id"])
        self.assertEqual(kept["messages"], first["messages"])
        self.assertEqual(kept["source_dataset"], "a")
        self.assertEqual(metadata["counts"]["target_conflicts"], 1)
        self.assertEqual(metadata["counts"]["by_reason"], {"duplicate_document": 2})
        self.assertEqual(metadata["counts"]["by_source"]["b"]["target_conflicts"], 1)
        # The final files are shuffled globally, not concatenated source groups.
        self.assertGreater(sum(x["source_dataset"] != y["source_dataset"] for x, y in zip(train, train[1:])), 2)
        self.prepare(inputs, "different", val_count=8, seed=18)
        self.assertNotEqual((self.root / "out/val.jsonl").read_bytes(), (self.root / "different/val.jsonl").read_bytes())

    def test_first_wins_even_if_first_duplicate_is_quarantined(self):
        rows = [row("same document", "unsafe <eos> target"), row("SAME  DOCUMENT", "safe target")]
        metadata = self.prepare({"a": self.write_rows("a.jsonl", rows + [row("independent safe document")])})
        self.assertEqual(metadata["counts"]["by_reason"], {"special_token": 1, "duplicate_document": 1})
        self.assertEqual(metadata["counts"]["target_conflicts"], 1)
        self.assertEqual(metadata["counts"]["retained"], 1)

    def test_inclusive_length_boundary_and_complete_untruncated_batch(self):
        rows = [row("below"), row("boundary"), row("above", "very long target " * 50)]
        tokenizer = FakeTokenizer({"below": 63, "boundary": 64, "above": 65})
        metadata = self.prepare({"a": self.write_rows("a.jsonl", rows)}, tokenizer=tokenizer, max_length=64)
        kept = self.read_rows(self.root / "out/train.jsonl")
        self.assertEqual({item["token_count"] for item in kept}, {63, 64})
        self.assertEqual(tokenizer.calls, [[item["messages"] for item in rows]])
        self.assertEqual(metadata["counts"]["by_reason"], {"overlength": 1})
        self.assertEqual(rows[2]["messages"][2]["content"], tokenizer.calls[0][2][2]["content"])

    def test_input_is_tokenized_in_bounded_batches(self):
        rows = [row(f"batch document {i}") for i in range(5)]
        tokenizer = FakeTokenizer()
        with patch.object(combined_data, "BATCH_SIZE", 2):
            self.prepare({"a": self.write_rows("a.jsonl", rows)}, tokenizer=tokenizer)
        self.assertEqual([len(call) for call in tokenizer.calls], [2, 2, 1])

    def test_token_bands_boundaries_extended_bands_and_minimum_validation(self):
        rows, lengths = [], {}
        for count in (1024, 1025, 2048, 2049, 4096, 4097, 8192, 8193):
            for i in range(2):
                document = f"length {count} fixture {i}"
                rows.append(row(document))
                lengths[document] = count
        metadata = self.prepare(
            {"a": self.write_rows("a.jsonl", rows)}, tokenizer=FakeTokenizer(lengths),
            val_count=5, max_length=16384,
        )
        self.assertEqual([s["token_band"] for s in metadata["strata"]],
                         ["<=1024", "<=2048", "<=4096", "<=8192", "<=16384"])
        self.assertEqual([s["available"] for s in metadata["strata"]], [2, 4, 4, 4, 2])
        self.assertTrue(all(s["val"] == 1 and s["train"] >= 1 for s in metadata["strata"]))

    def test_proportional_source_allocation_and_exact_counts(self):
        inputs = {label: self.write_rows(f"{label}.jsonl", [row(f"{label} fixture {i}") for i in range(count)])
                  for label, count in (("a", 80), ("b", 20))}
        metadata = self.prepare(inputs, val_count=20)
        self.assertEqual([s["val"] for s in metadata["strata"]], [16, 4])
        self.assertEqual([s["train"] for s in metadata["strata"]], [64, 16])

    def test_singletons_stay_in_train_and_small_validation_budget_is_exact(self):
        inputs = {label: self.write_rows(f"{label}.jsonl", [row(f"{label} fixture {i}") for i in range(count)])
                  for label, count in (("a", 4), ("b", 4), ("singleton", 1))}
        metadata = self.prepare(inputs, val_count=1)
        self.assertEqual(metadata["counts"]["val"], 1)
        self.assertEqual(metadata["counts"]["train"], 8)
        self.assertEqual(metadata["counts"]["by_source"]["singleton"]["train"], 1)

    def test_quarantines_special_tokens_controls_and_replacement_in_both_messages(self):
        unsafe = [(chr(code), "forbidden_control") for code in range(32) if code not in (9, 10, 13)]
        unsafe += [("\x7f", "forbidden_control"), ("\ufffd", "replacement_character")]
        unsafe += [(token, "special_token") for token in FakeTokenizer.all_special_tokens]
        rows = [row("safe\tfixture\r\nwith ordinary text", "Safe\tline\r\nsummary")]
        expected = {}
        for location in ("user", "assistant"):
            for index, (corruption, reason) in enumerate(unsafe):
                document = f"corrupt {location} fixture {index}"
                target = "Summary text"
                if location == "user":
                    document += corruption
                else:
                    target += corruption
                rows.append(row(document, target))
                expected[reason] = expected.get(reason, 0) + 1
        metadata = self.prepare({"a": self.write_rows("a.jsonl", rows)})
        self.assertEqual(metadata["counts"]["by_reason"], expected)
        self.assertEqual(metadata["counts"]["retained"], 1)
        exclusions = self.read_rows(self.root / "out/exclusions.jsonl")
        self.assertEqual(len(exclusions), len(rows) - 1)
        self.assertTrue(all(set(item) == {"id", "source", "input_line", "reason"} for item in exclusions))
        self.assertEqual({item["input_line"] for item in exclusions}, set(range(2, len(rows) + 1)))
        self.assertNotIn("Summary text", (self.root / "out/exclusions.jsonl").read_text())

    def test_fresh_frozen_eval_containment_inclusive_point_two_and_hashes(self):
        # Twelve words -> five 8-grams, exactly one present in the evaluation set.
        document = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
        eval_path = self.write_rows("eval.jsonl", [{
            "id": "eval-1", "document": "alpha bravo charlie delta echo foxtrot golf hotel", "reference": "Eval target",
        }])
        inputs = {"a": self.write_rows("a.jsonl", [row(document), row("unrelated safe fixture")])}
        metadata = self.prepare(inputs, eval_inputs=[eval_path])
        self.assertEqual(metadata["counts"]["by_reason"], {"eval_overlap": 1})
        self.assertEqual(metadata["evaluation"]["max_seen_containment"], 0.2)
        self.assertEqual(metadata["evaluation"]["ngram_size"], 8)
        self.assertEqual(metadata["evaluation"]["inputs"][0]["sha256"], sha256_file(eval_path))
        self.assertEqual(metadata["evaluation"]["documents"], 1)
        self.write_rows("eval.jsonl", [{"id": "eval-2", "document": "utterly different frozen evaluation", "reference": "Other"}])
        fresh = self.prepare(inputs, "fresh", eval_inputs=[eval_path])
        self.assertEqual(fresh["counts"]["retained"], 2)
        self.assertNotEqual(metadata["evaluation"]["inputs"][0]["sha256"], fresh["evaluation"]["inputs"][0]["sha256"])

    def test_malformed_rows_fail_even_after_valid_duplicate_or_overlength(self):
        valid = row("duplicate document")
        cases = []
        bad = copy.deepcopy(valid)
        bad["messages"][0]["content"] += " "
        cases.append((bad, "system"))
        bad = copy.deepcopy(valid)
        bad["messages"][2]["role"] = "user"
        cases.append((bad, "roles"))
        bad = copy.deepcopy(valid)
        bad["messages"].append({"role": "assistant", "content": "extra"})
        cases.append((bad, "roles"))
        for content in (None, 3, " ", ["not a string"]):
            bad = copy.deepcopy(valid)
            bad["messages"][2]["content"] = content
            cases.append((bad, "content"))
        bad = copy.deepcopy(valid)
        bad["id"] = "wrong-id"
        cases.extend([(bad, "id"), ([], "object"), ({"messages": "bad"}, "roles")])
        for index, (bad, message) in enumerate(cases):
            with self.subTest(index=index):
                path = self.write_rows("bad.jsonl", [valid, bad])
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare({"a": path}, name=f"bad-{index}", max_length=1)
                self.assertFalse((self.root / f"bad-{index}").exists())
                self.assert_no_staging()

    def test_invalid_json_fails_without_echoing_private_text(self):
        path = self.root / "bad.jsonl"
        path.write_text('{"private synthetic payload": invalid}\n')
        with self.assertRaises(ValueError) as error:
            self.prepare({"a": path})
        self.assertNotIn("private synthetic payload", str(error.exception))
        self.assertIn("input line 1", str(error.exception))
        self.assert_no_staging()

    def test_output_permissions_metadata_and_hashes(self):
        rows = [row("safe fixture"), row("quarantined <eos> fixture")]
        path = self.write_rows("a.jsonl", rows)
        old_umask = os.umask(0o777)
        try:
            metadata = self.prepare({"a": path})
        finally:
            os.umask(old_umask)
        output = self.root / "out"
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        self.assertEqual({file.name for file in output.iterdir()},
                         {"train.jsonl", "val.jsonl", "metadata.json", "exclusions.jsonl"})
        for file in output.iterdir():
            self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)
        self.assertEqual(json.loads((output / "metadata.json").read_text()), metadata)
        self.assertEqual(metadata["tokenizer_provenance"], self.provenance)
        self.assertEqual(metadata["prompt_version"], PROMPT_VERSION)
        self.assertEqual(metadata["inputs"]["a"]["sha256"], sha256_file(path))
        for name, report in metadata["outputs"].items():
            self.assertEqual(report["sha256"], sha256_file(output / name))
            self.assertEqual(report["rows"], len(self.read_rows(output / name)))
        self.assertNotIn("safe fixture", (output / "metadata.json").read_text())
        self.assertEqual(metadata["counts"]["input"], metadata["counts"]["retained"] + metadata["counts"]["excluded"])
        self.assert_no_staging()

    def test_refuses_existing_file_directory_and_dangling_symlink(self):
        path = self.write_rows("a.jsonl", [row("fixture")])
        (self.root / "existing-file").write_text("unchanged")
        (self.root / "existing-dir").mkdir()
        (self.root / "existing-link").symlink_to(self.root / "nonexistent")
        for name in ("existing-file", "existing-dir", "existing-link"):
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                self.prepare({"a": path}, name)
        self.assertEqual((self.root / "existing-file").read_text(), "unchanged")
        self.assert_no_staging()

    def test_failure_during_tokenization_cleans_stage_and_publishes_nothing(self):
        path = self.write_rows("a.jsonl", [row(f"fixture {i}") for i in range(3)])
        tokenizer = FakeTokenizer()
        original = tokenizer.apply_chat_template

        def fail_second(*args, **kwargs):
            if tokenizer.calls:
                raise RuntimeError("synthetic tokenizer failure")
            return original(*args, **kwargs)

        with patch.object(combined_data, "BATCH_SIZE", 1), patch.object(tokenizer, "apply_chat_template", fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic tokenizer failure"):
                self.prepare({"a": path}, tokenizer=tokenizer)
        self.assertFalse((self.root / "out").exists())
        self.assert_no_staging()

    def test_whole_directory_is_staged_before_publication_and_failure_cleans_it(self):
        path = self.write_rows("a.jsonl", [row("fixture")])

        def fail_publish(staged, destination):
            self.assertFalse(destination.exists())
            self.assertEqual({file.name for file in staged.iterdir()},
                             {"train.jsonl", "val.jsonl", "metadata.json", "exclusions.jsonl"})
            raise OSError("synthetic publication failure")

        with patch.object(combined_data, "_publish_directory", fail_publish):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.prepare({"a": path})
        self.assertFalse((self.root / "out").exists())
        self.assert_no_staging()

    def test_concurrent_destination_creation_is_not_overwritten(self):
        path = self.write_rows("a.jsonl", [row("fixture")])
        publish = combined_data._publish_directory
        inode = []

        def create_destination(staged, destination):
            destination.mkdir()
            inode.append(destination.stat().st_ino)
            publish(staged, destination)

        with patch.object(combined_data, "_publish_directory", create_destination):
            with self.assertRaises(FileExistsError):
                self.prepare({"a": path})
        self.assertEqual((self.root / "out").stat().st_ino, inode[0])
        self.assertEqual(list((self.root / "out").iterdir()), [])
        self.assert_no_staging()

    def test_infeasible_validation_request_fails_without_partial_output(self):
        path = self.write_rows("a.jsonl", [row("fixture")])
        with self.assertRaisesRegex(ValueError, "insufficient retained"):
            self.prepare({"a": path}, val_count=1)
        self.assertFalse((self.root / "out").exists())
        self.assert_no_staging()

    def test_malformed_tokenizer_results_fail_closed(self):
        path = self.write_rows("a.jsonl", [row("fixture")])
        for result in ([], [[1], [2]], [[]], [["not an id"]], {"input_ids": [[1]]}):
            with self.subTest(result=result), patch.object(FakeTokenizer, "apply_chat_template", return_value=result):
                with self.assertRaises(ValueError):
                    self.prepare({"a": path})
            self.assertFalse((self.root / "out").exists())
            self.assert_no_staging()

    def test_exclusion_line_numbers_include_blank_input_lines(self):
        path = self.root / "a.jsonl"
        path.write_text("\n" + json.dumps(row("safe fixture")) + "\n\n" + json.dumps(row("bad <eos> fixture")) + "\n")
        self.prepare({"a": path})
        self.assertEqual(self.read_rows(self.root / "out/exclusions.jsonl")[0]["input_line"], 4)


if __name__ == "__main__":
    unittest.main()
