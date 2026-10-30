import unittest

from lillm_eval.core import (
    PROMPT_VERSION, SYSTEM_PROMPT, SYSTEM_PROMPT_V2, USER_PROMPT_TEMPLATE, sample_stream,
)


class SummaryPromptTests(unittest.TestCase):
    def test_v3_allows_source_dependent_length_and_preserves_legacy_contract(self):
        self.assertEqual(PROMPT_VERSION, "summary-v3")
        self.assertIn("provide more detail when the source warrants it", SYSTEM_PROMPT)
        self.assertIn("multiple paragraphs when helpful", SYSTEM_PROMPT)
        self.assertIn("do not add unsupported information", SYSTEM_PROMPT)
        self.assertEqual(USER_PROMPT_TEMPLATE, "{document}")
        self.assertEqual(SYSTEM_PROMPT_V2,
                         "Summarize the page below in a concise, objective paragraph that captures its "
                         "main points and key details, rephrasing the content in your own words.")


class SampleStreamTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": str(index), "article": f"article {index}", "highlights": f"summary {index}"}
            for index in range(20)
        ]

    def test_sampling_is_reproducible_and_has_requested_count(self):
        first = sample_stream(iter(self.records), count=5, seed=123)
        second = sample_stream(iter(self.records), count=5, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_sampling_is_not_just_first_records(self):
        sample = sample_stream(iter(self.records), count=5, seed=123)

        self.assertNotEqual([row["id"] for row in sample], ["0", "1", "2", "3", "4"])

    def test_excludes_ids_before_sampling_without_reducing_requested_count(self):
        excluded = {"0", "1", "2", "3", "4"}
        eligible_records = [row for row in self.records if row["id"] not in excluded]

        sample = sample_stream(iter(self.records), count=5, seed=123, exclude_ids=excluded)
        expected = sample_stream(iter(eligible_records), count=5, seed=123)

        self.assertEqual(sample, expected)
        self.assertEqual(len(sample), 5)
        self.assertTrue({row["id"] for row in sample}.isdisjoint(excluded))

    def test_rejects_count_larger_than_data_remaining_after_exclusions(self):
        with self.assertRaisesRegex(ValueError, "only 18 valid"):
            sample_stream(iter(self.records), count=19, seed=1, exclude_ids={"0", "1"})

    def test_excluding_generated_id_does_not_shift_later_ids_or_bias_sampling(self):
        records = [
            {"document": f"article {index}", "summary": f"summary {index}"}
            for index in range(10)
        ]
        eligible_records = [records[index] for index in range(len(records)) if index != 1]

        sample = sample_stream(iter(records), count=4, seed=123, exclude_ids={"1"})
        expected = sample_stream(iter(eligible_records), count=4, seed=123)

        sample_ids = [row["id"] for row in sample]
        source_indices = [row["document"].removeprefix("article ") for row in sample]
        sample_documents = [row["document"] for row in sample]
        expected_documents = [row["document"] for row in expected]

        self.assertEqual(sample_ids, source_indices)
        self.assertEqual(sample_documents, expected_documents)

    def test_generates_stable_ids_for_alias_only_records(self):
        records = [
            {"document": "first article", "summary": "first summary"},
            {"document": "second article", "summary": "second summary"},
        ]

        sample = sample_stream(iter(records), count=2, seed=123)

        self.assertEqual([row["id"] for row in sample], ["0", "1"])
        self.assertEqual(sample[0]["document"], "first article")
        self.assertEqual(sample[1]["reference"], "second summary")

    def test_rejects_invalid_sample_size(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            sample_stream(iter(self.records), count=0, seed=1)

    def test_rejects_count_larger_than_valid_data(self):
        with self.assertRaisesRegex(ValueError, "only 20 valid"):
            sample_stream(iter(self.records), count=21, seed=1)


if __name__ == "__main__":
    unittest.main()
