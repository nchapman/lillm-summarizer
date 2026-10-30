import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import datasets  # noqa: F401
    import huggingface_hub  # noqa: F401

    HAVE_DATA_EXTRA = True
except ImportError:
    HAVE_DATA_EXTRA = False

from lillm_eval.cli import prepare_smoltalk
from lillm_eval.core import PROMPT_VERSION, SYSTEM_PROMPT
from lillm_eval.training_data import (
    build_eval_ngram_set,
    classify_instruction,
    containment,
    count_sentences,
    extract_pair,
    length_band,
    original_instruction,
    prepare_rows,
    stratified_subset,
    word_ngrams,
)

THREE_SENTENCE_PROMPT = (
    "Provide a concise, objective summary of the input text in up to three "
    "sentences, focusing on key actions and intentions without using second "
    "or third person pronouns."
)
ONE_LINE_PROMPT = (
    "Extract and present the main key point of the input text in one very "
    "short sentence, including essential details like dates or locations if necessary."
)


def chat_row(instruction: str, user_text: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": target},
        ],
        "chat_template_kwargs": {"custom_instructions": instruction},
    }


class TaggingTests(unittest.TestCase):
    def test_classifies_instruction_variants(self):
        self.assertEqual(classify_instruction(THREE_SENTENCE_PROMPT), "three_sentence")
        self.assertEqual(classify_instruction(ONE_LINE_PROMPT), "one_line")
        self.assertEqual(classify_instruction("Summarize the news article"), "other")
        self.assertEqual(classify_instruction(""), "unknown")

    def test_extracts_pair_and_validates_roles(self):
        row = chat_row(THREE_SENTENCE_PROMPT, "article text", "Summary sentence.")
        self.assertEqual(extract_pair(row["messages"]), ("article text", "Summary sentence."))
        self.assertEqual(original_instruction(row), THREE_SENTENCE_PROMPT)
        with self.assertRaisesRegex(ValueError, "roles"):
            extract_pair(
                [
                    {"role": "system", "content": THREE_SENTENCE_PROMPT},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            extract_pair(
                [
                    {"role": "user", "content": "  "},
                    {"role": "assistant", "content": "hello"},
                ]
            )
        with self.assertRaisesRegex(ValueError, "sequence"):
            extract_pair("not a message list")
        with self.assertRaisesRegex(ValueError, "object"):
            extract_pair(["user", "assistant"])


class NgramTests(unittest.TestCase):
    def test_word_ngrams_normalizes_and_windows(self):
        grams = word_ngrams("Hello, WORLd! hello world.", size=2)
        self.assertEqual(grams, [("hello", "world"), ("world", "hello"), ("hello", "world")])

    def test_containment_scores_overlap(self):
        reference = build_eval_ngram_set(["alpha beta gamma delta epsilon zeta"], size=3)
        identical = containment("alpha beta gamma delta epsilon zeta", reference, size=3)
        disjoint = containment("one two three four five six", reference, size=3)
        partial = containment("alpha beta gamma delta one two three four five", reference, size=3)
        self.assertEqual(identical, 1.0)
        self.assertEqual(disjoint, 0.0)
        # 7 three-grams in the candidate, of which 2 also occur in the reference.
        self.assertAlmostEqual(partial, 2.0 / 7.0)

    def test_count_sentences_is_rough_but_useful(self):
        self.assertEqual(count_sentences("One. Two sentences here! And three?"), 3)
        self.assertEqual(count_sentences(""), 0)


class StratifiedSubsetTests(unittest.TestCase):
    @staticmethod
    def _rows(count: int, tag: str, band: str) -> list[dict]:
        words = {"short": 50, "medium": 300, "long": 800}[band]
        return [
            {
                "id": f"{tag}-{band}-{index}",
                "messages": [],
                "tag": tag,
                "document_words": words,
            }
            for index in range(count)
        ]

    def test_split_is_stratified_disjoint_and_deterministic(self):
        rows = (
            self._rows(600, "three_sentence", "medium")
            + self._rows(300, "three_sentence", "long")
            + self._rows(100, "three_sentence", "short")
            + self._rows(400, "one_line", "medium")
            + self._rows(100, "one_line", "short")
        )
        train_a, val_a, report = stratified_subset(rows, train_count=250, val_count=50, seed=7)
        train_b, val_b, _ = stratified_subset(rows, train_count=250, val_count=50, seed=7)

        self.assertEqual([row["id"] for row in train_a], [row["id"] for row in train_b])
        self.assertEqual(len(train_a), 250)
        self.assertEqual(len(val_a), 50)
        train_ids = {row["id"] for row in train_a}
        val_ids = {row["id"] for row in val_a}
        self.assertFalse(train_ids & val_ids)

        bands = {length_band(row["document_words"]) for row in train_a}
        self.assertEqual(bands, {"short", "medium", "long"})
        self.assertEqual(
            report["strata"]["three_sentence/long"],
            {"available": 300, "train": 50, "val": 10},
        )
        self.assertEqual(sum(s["train"] for s in report["strata"].values()), 250)
        self.assertEqual(sum(s["val"] for s in report["strata"].values()), 50)

    def test_caps_oversubscribed_strata_and_backfills(self):
        rows = self._rows(10, "one_line", "short") + self._rows(990, "three_sentence", "medium")
        train, val, report = stratified_subset(rows, train_count=200, val_count=40, seed=1)
        self.assertEqual(len(train), 200)
        self.assertEqual(len(val), 40)
        short_stratum = report["strata"]["one_line/short"]
        self.assertLessEqual(short_stratum["train"] + short_stratum["val"], 10)
        self.assertEqual(
            sum(s["train"] + s["val"] for s in report["strata"].values()),
            240,
        )

    def test_near_full_split_returns_exact_requested_counts(self):
        rows = (
            self._rows(100, "one_line", "short")
            + self._rows(50, "three_sentence", "medium")
            + self._rows(50, "other", "long")
        )

        train, val, report = stratified_subset(rows, train_count=180, val_count=10, seed=1)

        self.assertEqual(len(train), 180)
        self.assertEqual(len(val), 10)
        self.assertEqual(report["train_rows"], 180)
        self.assertEqual(report["val_rows"], 10)
        self.assertFalse(
            {row["id"] for row in train} & {row["id"] for row in val}
        )

    def test_rejects_counts_beyond_available_rows(self):
        rows = self._rows(20, "one_line", "short")
        with self.assertRaisesRegex(ValueError, "available"):
            stratified_subset(rows, train_count=15, val_count=10, seed=1)


class PrepareRowsTests(unittest.TestCase):
    def setUp(self):
        self.eval_doc = " ".join(f"evalword{index}" for index in range(50))
        self.eval_ngrams = build_eval_ngram_set([self.eval_doc])

    def test_prepends_prompt_tags_and_drops_leaky_or_duplicate_rows(self):
        leaky_row = chat_row(THREE_SENTENCE_PROMPT, self.eval_doc, "Leaky summary.")
        records = [
            chat_row(THREE_SENTENCE_PROMPT, "news article body number one", "News summary sentence."),
            chat_row(ONE_LINE_PROMPT, "email body number one", "Email summary sentence."),
            chat_row(ONE_LINE_PROMPT, "email body number one", "Duplicate summary."),
            leaky_row,
            {"messages": [{"role": "user", "content": "no system prompt"}]},
        ]
        rows, report = prepare_rows(
            records,
            system_prompt=SYSTEM_PROMPT,
            eval_ngrams=self.eval_ngrams,
            source_label="test@revision",
        )

        self.assertEqual(report["total_input_rows"], 5)
        self.assertEqual(report["kept"], 2)
        self.assertEqual(report["dropped_leakage"], 1)
        self.assertEqual(report["dropped_duplicate"], 1)
        self.assertEqual(report["invalid_rows"], 1)
        self.assertEqual(report["by_tag"]["three_sentence"]["rows"], 1)
        self.assertEqual(report["by_tag"]["one_line"]["rows"], 1)

        self.assertEqual([row["tag"] for row in rows], ["three_sentence", "one_line"])
        for row in rows:
            self.assertEqual(row["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
            self.assertEqual(row["messages"][1]["role"], "user")
            self.assertEqual(row["messages"][2]["role"], "assistant")
            self.assertEqual(row["source"], "test@revision")
            self.assertLess(row["leakage_score"], 0.2)
        self.assertEqual(rows[0]["target_sentences"], 1)
        self.assertGreater(report["leakage_max_seen"], 0.99)
        self.assertIn("not recoverable", report["tag_note"])

    def test_counts_malformed_message_containers_as_invalid_rows(self):
        records = [
            {"messages": "broken"},
            {"messages": ["user", "assistant"]},
        ]
        rows, report = prepare_rows(
            records,
            system_prompt=SYSTEM_PROMPT,
            eval_ngrams=self.eval_ngrams,
            source_label="test@revision",
        )
        self.assertEqual(rows, [])
        self.assertEqual(report["invalid_rows"], 2)

    def test_rejects_out_of_range_threshold(self):
        with self.assertRaisesRegex(ValueError, "leakage_threshold"):
            prepare_rows(
                [],
                system_prompt=SYSTEM_PROMPT,
                eval_ngrams=set(),
                source_label="test@revision",
                leakage_threshold=1.5,
            )

    def test_limit_bounds_input_rows(self):
        records = [chat_row(ONE_LINE_PROMPT, f"email body {index}", "Summary.") for index in range(5)]
        rows, report = prepare_rows(
            records,
            system_prompt=SYSTEM_PROMPT,
            eval_ngrams=self.eval_ngrams,
            source_label="test@revision",
            limit=3,
        )
        self.assertEqual(report["total_input_rows"], 3)
        self.assertEqual(len(rows), 3)


@unittest.skipUnless(HAVE_DATA_EXTRA, "patching the loader requires the data extra")
class PrepareSmoltalkCliTests(unittest.TestCase):
    def test_refuses_to_overwrite_existing_output_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "train.jsonl"
            output.write_text("preserve this exact file\n", encoding="utf-8")
            args = SimpleNamespace(output=output, overwrite=False)

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                prepare_smoltalk(args)

            self.assertEqual(output.read_text(encoding="utf-8"), "preserve this exact file\n")

    def test_writes_rows_and_metadata_with_pinned_revision(self):
        def fake_stream_record(index: int) -> dict:
            return chat_row(
                ONE_LINE_PROMPT if index % 2 else THREE_SENTENCE_PROMPT,
                f"training article body with unique words {index}",
                f"Summary sentence {index}.",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_input = root / "eval.jsonl"
            eval_input.write_text(
                json.dumps(
                    {
                        "id": "eval-1",
                        "document": " ".join(f"evalword{index}" for index in range(40)),
                        "reference": "reference summary",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "train.jsonl"
            args = SimpleNamespace(
                output=output,
                eval_input=[eval_input],
                dataset_revision="deadbeef",
                leakage_threshold=0.2,
                limit=4,
                overwrite=False,
            )
            with patch("datasets.load_dataset", return_value=iter([fake_stream_record(i) for i in range(6)])), patch(
                "huggingface_hub.HfApi"
            ) as fake_api:
                prepare_smoltalk(args)

            fake_api.assert_not_called()  # revision was pinned explicitly
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            metadata = json.loads(output.with_suffix(".jsonl.meta.json").read_text())
            self.assertEqual(len(rows), 4)
            self.assertEqual(metadata["dataset_revision"], "deadbeef")
            self.assertEqual(metadata["prompt_version"], PROMPT_VERSION)
            self.assertEqual(metadata["leakage_check"]["eval_inputs"][0]["examples"], 1)
            self.assertEqual(metadata["report"]["kept"], 4)


if __name__ == "__main__":
    unittest.main()
