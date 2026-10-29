import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tinyllm_eval.cli import (
    _resolve_inference_auth,
    build_parser,
    judge_predictions,
    load_project_env_file,
    prepare_dataset,
)


class InferenceLengthTests(unittest.TestCase):
    def test_default_allows_longer_summaries_and_explicit_cap_remains_supported(self):
        command = ["run", "--input", "input.jsonl", "--output", "output.jsonl", "--model", "synthetic"]
        parser = build_parser()
        self.assertEqual(parser.parse_args(command).max_new_tokens, 1024)
        self.assertEqual(parser.parse_args([*command, "--max-new-tokens", "2048"]).max_new_tokens, 2048)


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


class JudgeCliTests(unittest.TestCase):
    def test_judges_with_bounded_concurrency_and_preserves_input_order(self):
        judgment = {
            "factuality": 5,
            "coverage": 4,
            "relevance": 5,
            "concision": 4,
            "readability": 5,
            "overall_quality": 4,
            "unsupported_claims": [],
            "rationale": "Supported and clear.",
        }
        barrier = threading.Barrier(3)
        lock = threading.Lock()
        active = 0
        peak_active = 0

        def fake_request(*_args, **_kwargs):
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                barrier.wait(timeout=2)
                time.sleep(0.01)
                return judgment, {"usage": {}, "finish_reason": "stop", "attempts": 1}
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "predictions.jsonl"
            output_path = root / "judgments.jsonl"
            records = [
                {
                    "id": f"id-{index}",
                    "document": f"source {index}",
                    "reference": f"reference {index}",
                    "prediction": f"prediction {index}",
                    "model": "candidate",
                }
                for index in range(6)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                input=input_path,
                output=output_path,
                limit=None,
                overwrite=False,
                base_url="https://openrouter.ai/api/v1",
                model="test-judge",
                timeout=5,
                max_retries=0,
                retry_backoff_seconds=0,
                concurrency=3,
            )
            with patch("tinyllm_eval.cli.load_project_env_file"), patch(
                "tinyllm_eval.cli.request_judgment", side_effect=fake_request
            ), patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
                judge_predictions(args)

            outputs = [json.loads(line) for line in output_path.read_text().splitlines()]
            metadata = json.loads(output_path.with_suffix(".jsonl.meta.json").read_text())

        self.assertEqual([row["id"] for row in outputs], [row["id"] for row in records])
        self.assertEqual(peak_active, 3)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["completed_examples"], 6)
        self.assertEqual(metadata["concurrency"], 3)


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
