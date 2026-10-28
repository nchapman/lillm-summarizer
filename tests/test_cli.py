import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tinyllm_eval.cli import (
    _resolve_inference_auth,
    judge_predictions,
    load_project_env_file,
    prepare_dataset,
)


class InferenceEndpointTests(unittest.TestCase):
    def test_loads_openrouter_key_for_the_explicit_openrouter_endpoint(self):
        args = SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )
        with patch("tinyllm_eval.cli.load_project_env_file"), patch.dict(
            "os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True
        ):
            api_key, is_remote = _resolve_inference_auth(args)

        self.assertEqual(api_key, "test-key")
        self.assertTrue(is_remote)

    def test_rejects_non_loopback_non_openrouter_inference_endpoint(self):
        args = SimpleNamespace(base_url="https://example.invalid/api/v1", api_key_env=None)
        with self.assertRaisesRegex(ValueError, "local HTTP or OpenRouter HTTPS"):
            _resolve_inference_auth(args)


class PrepareDatasetTests(unittest.TestCase):
    def test_loads_project_key_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# project-local key\nOTHER=value\nOPENROUTER_API_KEY=project-key\n",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                load_project_env_file(env_file)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "project-key")

            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "shell-key"}, clear=True):
                load_project_env_file(env_file)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "shell-key")

    def test_judge_requires_openrouter_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                input=root / "predictions.jsonl",
                output=root / "judgments.jsonl",
                limit=None,
                overwrite=False,
                base_url="https://example.invalid/v1",
                model="test-judge",
                timeout=10,
            )
            with patch("tinyllm_eval.cli.load_project_env_file"), patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                    judge_predictions(args)

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
