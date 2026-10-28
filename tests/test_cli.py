import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tinyllm_eval.cli import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def test_refuses_to_overwrite_existing_sample_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.jsonl"
            output.write_text("preserve this exact sample\n", encoding="utf-8")
            args = SimpleNamespace(output=output, overwrite=False)

            with patch("builtins.__import__") as import_module:
                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    prepare_dataset(args)

            import_module.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve this exact sample\n")


if __name__ == "__main__":
    unittest.main()
