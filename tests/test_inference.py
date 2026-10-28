import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tinyllm_eval.core import request_judgment, request_summary


class InferenceRequestTests(unittest.TestCase):
    def test_sends_deterministic_chat_request_and_reads_completion(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received["path"] = self.path
                received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(summary, "A concise summary.")
        self.assertEqual(metadata["usage"]["completion_tokens"], 4)
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(received["path"], "/v1/chat/completions")
        self.assertEqual(received["body"]["temperature"], 0)
        self.assertEqual(received["body"]["seed"], 42)
        self.assertIn("The input document.", received["body"]["messages"][1]["content"])


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
        from tinyllm_eval.core import _parse_judge_response

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
