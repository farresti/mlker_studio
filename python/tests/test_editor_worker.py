"""Exercise the persistent editor protocol through its packaged entry point."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EditorWorkerTests(unittest.TestCase):
    def test_single_stream_load_validate_save_compile_and_recover(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            sample = ROOT.parent / "examples/models/v1/cnn.yaml"
            saved = folder / "draft.yaml"
            exported = folder / "project"
            with subprocess.Popen(
                [sys.executable, "-I", "-B", ROOT / "mlker_worker", "--editor"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            ) as process:
                sequence = 0

                def request(method, **params):
                    nonlocal sequence
                    sequence += 1
                    payload = {"version": 2, "id": str(sequence), "method": method, "params": params}
                    process.stdin.write(json.dumps(payload) + "\n")
                    process.stdin.flush()
                    line = process.stdout.readline()
                    self.assertTrue(line, f"No reply to {method}")
                    response = json.loads(line)
                    self.assertEqual((response["version"], response["id"]), (2, str(sequence)))
                    return response

                try:
                    registry = request("registry")["result"]
                    self.assertEqual(registry["version"], 1)
                    conv = next(op for op in registry["operations"] if op["id"] == "core.Conv2d")
                    self.assertEqual(conv["params"]["in_channels"]["minimum"], 1)

                    document = request("load", path=str(sample))["result"]["document"]
                    graph = document["graphs"][0]
                    graph["extensions"] = {"mlker.editor": {"positions": {"node:conv": {"x": 8, "y": 9}}}}
                    document["future_field"] = {"source": ["kept"]}
                    edges, graph["edges"] = graph["edges"], []
                    self.assertEqual(request("save", path=str(saved), document=document)["status"], "ok")
                    self.assertEqual(request("load", path=str(saved))["result"]["document"], document)
                    self.assertTrue(request("validate", document=document)["result"]["diagnostics"])
                    failed = request("compile", path=str(exported), document=document)["error"]
                    self.assertEqual(failed["code"], "invalid_document")
                    self.assertTrue(failed["diagnostics"])
                    self.assertFalse(exported.exists())

                    graph["edges"] = edges
                    result = request("compile", path=str(exported), document=document)["result"]
                    self.assertEqual(result["path"], str(exported))
                    compile(result["sources"]["model.py"], "model.py", "exec")
                    self.assertEqual(request("load", path="relative.yaml")["error"]["code"], "invalid_path")
                    self.assertEqual(request("validate", document=document)["result"]["diagnostics"], [])
                finally:
                    process.stdin.close()
                    process.wait(timeout=5)
