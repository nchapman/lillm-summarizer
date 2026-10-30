import io
import json
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lillm_eval.core import request_judgment, request_summary


class InferenceRequestTests(unittest.TestCase):
    def test_sends_chat_request_with_sampling_settings_and_reads_completion(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.setdefault("calls", []).append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                    }
                )
                response = {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "A concise summary."},
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 4},
                }
                encoded = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            summary, metadata, elapsed = request_summary(
                f"http://127.0.0.1:{server.server_port}/v1",
                "test-model",
                "The input document.",
                max_new_tokens=64,
                timeout=5,
                temperature=0.7,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                presence_penalty=1.5,
                repeat_penalty=1.05,
                api_key="test-api-key",
            )
            request_summary(
                f"http://127.0.0.1:{server.server_port}/v1",
                "test-model",
                "The input document.",
                max_new_tokens=64,
                timeout=5,
                api_key="test-api-key",
                temperature=None,
                reasoning_effort="none",
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(summary, "A concise summary.")
        self.assertEqual(metadata["usage"]["completion_tokens"], 4)
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(len(received["calls"]), 2)
        self.assertEqual(received["calls"][0]["path"], "/v1/chat/completions")
        self.assertEqual(received["calls"][0]["authorization"], "Bearer test-api-key")
        self.assertEqual(received["calls"][0]["body"]["temperature"], 0.7)
        self.assertEqual(received["calls"][0]["body"]["top_p"], 0.95)
        self.assertEqual(received["calls"][0]["body"]["top_k"], 20)
        self.assertEqual(received["calls"][0]["body"]["min_p"], 0.0)
        self.assertEqual(received["calls"][0]["body"]["presence_penalty"], 1.5)
        self.assertEqual(received["calls"][0]["body"]["repeat_penalty"], 1.05)
        self.assertEqual(received["calls"][0]["body"]["seed"], 42)
        self.assertIn("The input document.", received["calls"][0]["body"]["messages"][1]["content"])
        self.assertEqual(received["calls"][1]["authorization"], "Bearer test-api-key")
        self.assertNotIn("temperature", received["calls"][1]["body"])
        self.assertEqual(received["calls"][1]["body"]["reasoning_effort"], "none")

    def test_does_not_forward_api_key_through_redirects(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/capture")
                self.end_headers()

            def do_GET(self):
                self.server.redirect_requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"choices": []}')

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.redirect_requests = 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                request_summary(
                    f"http://127.0.0.1:{server.server_port}/v1",
                    "test-model",
                    "The input document.",
                    max_new_tokens=64,
                    timeout=5,
                    api_key="test-api-key",
                )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(server.redirect_requests, 0)


class JudgeRequestTests(unittest.TestCase):
    def test_sends_authenticated_json_judge_request(self):
        received = {}
        judgment = {
            "factuality": 5,
            "coverage": 4,
            "relevance": 5,
            "concision": 4,
            "readability": 5,
            "overall_quality": 4,
            "unsupported_claims": [],
            "rationale": "The summary is supported and clear.",
        }

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received["path"] = self.path
                received["authorization"] = self.headers.get("Authorization")
                received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                response = {
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(judgment)}}],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 40},
                }
                encoded = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result, metadata = request_judgment(
                f"http://127.0.0.1:{server.server_port}/v1",
                "test-secret",
                "test-judge",
                "The article source.",
                "A supported summary.",
                timeout=5,
                allow_insecure_loopback=True,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(result, judgment)
        self.assertEqual(metadata["usage"]["completion_tokens"], 40)
        self.assertEqual(received["path"], "/v1/chat/completions")
        self.assertEqual(received["authorization"], "Bearer test-secret")
        self.assertEqual(received["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(received["body"]["reasoning"], {"enabled": False})
        user_content = received["body"]["messages"][1]["content"]
        data = json.loads(user_content.split("\n\n", 1)[1])
        self.assertEqual(data["source"], "The article source.")
        self.assertEqual(data["candidate_summary"], "A supported summary.")

    def test_retries_timeout_and_server_error_with_exponential_backoff(self):
        judgment = {
            "factuality": 5,
            "coverage": 4,
            "relevance": 5,
            "concision": 4,
            "readability": 5,
            "overall_quality": 4,
            "unsupported_claims": [],
            "rationale": "The summary is supported and clear.",
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(judgment)}}],
            "usage": {"completion_tokens": 40},
        }).encode()
        unavailable = HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            503,
            "Service Unavailable",
            {"Retry-After": "0.4"},
            io.BytesIO(b"temporary outage"),
        )
        long_retry_after = HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            503,
            "Service Unavailable",
            {"Retry-After": "3600"},
            io.BytesIO(b"temporary outage"),
        )

        with patch("lillm_eval.core.build_opener") as build_opener, patch(
            "lillm_eval.core.time.sleep"
        ) as sleep, patch("lillm_eval.core.random.uniform", side_effect=lambda low, _high: low):
            opener = build_opener.return_value
            opener.open.side_effect = [
                URLError(TimeoutError("timed out")),
                unavailable,
                long_retry_after,
                response,
            ]
            result, metadata = request_judgment(
                "https://openrouter.ai/api/v1",
                "test-secret",
                "test-judge",
                "The article source.",
                "A supported summary.",
                timeout=5,
                max_retries=3,
                retry_backoff_seconds=0.2,
                example_id="example-1",
            )

        self.assertEqual(result, judgment)
        self.assertEqual(metadata["attempts"], 4)
        self.assertEqual(opener.open.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.1, 0.4, 30.0])

    def test_stops_before_retry_when_batch_cancellation_is_signaled(self):
        cancellation_event = threading.Event()

        def signal_cancellation(low, _high):
            cancellation_event.set()
            return low

        with patch("lillm_eval.core.build_opener") as build_opener, patch(
            "lillm_eval.core.random.uniform", side_effect=signal_cancellation
        ):
            opener = build_opener.return_value
            opener.open.side_effect = URLError(TimeoutError("timed out"))
            with self.assertRaisesRegex(RuntimeError, "cancelled because the batch is stopping"):
                request_judgment(
                    "https://openrouter.ai/api/v1",
                    "test-secret",
                    "test-judge",
                    "The article source.",
                    "A supported summary.",
                    timeout=5,
                    max_retries=3,
                    retry_backoff_seconds=1,
                    cancellation_event=cancellation_event,
                )

        self.assertEqual(opener.open.call_count, 1)

    def test_does_not_retry_non_transient_http_error(self):
        error = HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b"invalid request"),
        )
        with patch("lillm_eval.core.build_opener") as build_opener, patch(
            "lillm_eval.core.time.sleep"
        ) as sleep:
            opener = build_opener.return_value
            opener.open.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                request_judgment(
                    "https://openrouter.ai/api/v1",
                    "test-secret",
                    "test-judge",
                    "The article source.",
                    "A supported summary.",
                    timeout=5,
                    max_retries=3,
                    retry_backoff_seconds=0,
                )

        self.assertEqual(opener.open.call_count, 1)
        sleep.assert_not_called()

    def test_rejects_non_openrouter_remote_base_url(self):
        with self.assertRaisesRegex(ValueError, "OpenRouter HTTPS"):
            request_judgment(
                "https://attacker.example/api/v1",
                "test-secret",
                "test-judge",
                "source",
                "summary",
                timeout=5,
            )

    def test_rejects_out_of_range_judge_score(self):
        from lillm_eval.core import _parse_judge_response

        response = {
            "factuality": 6,
            "coverage": 4,
            "relevance": 5,
            "concision": 4,
            "readability": 5,
            "overall_quality": 4,
            "unsupported_claims": [],
            "rationale": "Invalid score.",
        }
        with self.assertRaisesRegex(ValueError, "from 1 to 5"):
            _parse_judge_response(json.dumps(response))


if __name__ == "__main__":
    unittest.main()
