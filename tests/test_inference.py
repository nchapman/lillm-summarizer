import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tinyllm_eval.core import request_summary


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


if __name__ == "__main__":
    unittest.main()
