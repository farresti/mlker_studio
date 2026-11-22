import json
from pathlib import Path
import subprocess
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PING = {"version": 1, "id": "check-1", "method": "ping", "params": {}}


def encode(request):
    return json.dumps(request).encode() + b"\n"


def exchange(data):
    process = subprocess.run(
        [sys.executable, "-m", "mlker_worker"],
        cwd=PACKAGE_ROOT, input=data, capture_output=True, timeout=5, check=True,
    )
    return [json.loads(line) for line in process.stdout.splitlines()], process.stderr


class WorkerTests(unittest.TestCase):
    def test_ping(self):
        replies, logs = exchange(encode(PING))
        self.assertEqual(replies, [{
            "version": 1, "id": "check-1", "status": "ok",
            "result": {"reply": "pong", "pythonVersion": sys.version.split()[0]},
        }])
        self.assertEqual(logs, b"")

    def test_structured_errors_and_recovery(self):
        cases = [
            (b"\xff\n", None, "invalid_json"),
            (b"[]\n", None, "invalid_request"),
            (encode({**PING, "id": ""}), None, "invalid_request"),
            (encode({**PING, "version": True}), "check-1", "invalid_request"),
            (encode({**PING, "version": 99}), "check-1", "unsupported_version"),
            (encode({**PING, "method": "missing"}), "check-1", "unknown_method"),
            (encode({**PING, "params": {"unexpected": 1}}), "check-1", "invalid_request"),
            (b"x" * 20000 + b"\n", None, "frame_too_large"),
        ]
        replies, logs = exchange(b"".join(case[0] for case in cases) + encode(PING))
        self.assertEqual(len(replies), len(cases) + 1)
        for reply, (_, request_id, code) in zip(replies, cases):
            with self.subTest(code=code):
                self.assertEqual(reply["id"], request_id)
                self.assertEqual(reply["status"], "error")
                self.assertEqual(reply["version"], 1)
                self.assertEqual(reply["error"]["code"], code)
                self.assertTrue(reply["error"]["message"])
                self.assertNotIn("result", reply)
        self.assertEqual(replies[-1]["result"]["reply"], "pong")
        self.assertIn(b"Rejected worker request", logs)

    def test_unterminated_frame(self):
        replies, _ = exchange(encode(PING).rstrip())
        self.assertEqual(replies[0]["error"]["code"], "invalid_request")
