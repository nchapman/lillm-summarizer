import unittest

from tinyllm_eval.core import sample_stream


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
