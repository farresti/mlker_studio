from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlker_graph import ModelError, load_model, validate_model
from mlker_graph.extensions import read_registry
from mlker_graph.pytorch import compile_model
from mlker_worker.editor import registry

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
EXT = EXAMPLES / "extensions"


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.doc = load_model(EXAMPLES / "models/v2/custom-scale.yaml")
        self.node = self.doc.data["graphs"][0]["nodes"][0]
        self.trust = patch.dict(os.environ, MLKER_TRUSTED_REGISTRY=str(EXT / "registry.json"))
        self.trust.start()
        self.addCleanup(self.trust.stop)

    def test_registration_metadata_and_contracts(self):
        item = registry()["operations"][-1]
        self.assertEqual(item["implementation"], self.node["implementation"])
        self.assertEqual(item["params"]["factor"]["default"], 2.0)
        self.assertFalse(validate_model(self.doc))
        self.node["params"]["factor"] = True
        self.node["inputs"][0]["tensor"]["dtype"] = "int64"
        codes = {d.code for d in validate_model(self.doc)}
        self.assertIn("parameter_value", codes)
        self.assertGreater(len(codes), 1)
        self.assertNotIn("mlker_example_layers", sys.modules)

    def test_trust_and_permission_failures(self):
        with patch.dict(os.environ, MLKER_TRUSTED_REGISTRY=""):
            self.assertEqual(validate_model(self.doc)[0].code, "extension_trust")
        with patch.object(Path, "read_text", side_effect=PermissionError("permission denied")):
            self.assertIn("permission denied", validate_model(self.doc)[0].message)
        self.node["implementation"]["symbol"] = "AnotherClass"
        self.assertEqual(validate_model(self.doc)[0].code, "extension_trust")

    def test_duplicate_and_invalid_registrations(self):
        records = json.loads((EXT / "registry.json").read_text())
        for change in ("duplicate", "parameter", "dependency"):
            rows = deepcopy(records)
            if change == "duplicate":
                rows.extend(records)
            elif change == "parameter":
                rows[0]["parameters"][0]["id"] = "invalid-key"
            else:
                rows[0]["dependencies"] = ["--index-url alternate"]
            with patch.object(Path, "read_text", return_value=json.dumps(rows)), self.assertRaises(ModelError):
                read_registry()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "runtime requires torch")
    def test_function_registration_with_multiple_outputs(self):
        records = json.loads((EXT / "registry.json").read_text())
        registration = records[0]
        registration.update(package="torch", symbol="sort", kind="function",
                            outputs=[{"id": k, "tensor": {}} for k in ("values", "indices")],
                            parameters=[{"id": "descending", "type": "bool", "default": True}])
        self.node["implementation"].update(package="torch", symbol="sort")
        self.node.update(outputs=registration["outputs"], params={})
        self.doc.data["graphs"][0]["edges"][1]["from"]["port"] = "values"
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "registry.json"
            manifest.write_text(json.dumps(records))
            with patch.dict(os.environ, MLKER_TRUSTED_REGISTRY=str(manifest)):
                folder = compile_model(self.doc, Path(tmp) / "project")
            code = "from model import Model; import torch; assert Model()(torch.tensor([1., 3.])).tolist() == [3., 1.]"
            subprocess.run([sys.executable, "-c", code], cwd=folder, check=True)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "runtime requires torch")
    def test_nested_export_and_dependency_resolution(self):
        # A parameter traverses two nested block boundaries before reaching Python.
        leaf = self.doc.data["graphs"][0]
        port_pair = {k: deepcopy(leaf[k]) for k in ("inputs", "outputs")}
        leaf["parameters"] = [{"id": "gain", "type": "float", "default": 2.0}]
        leaf["nodes"][0]["params"]["factor"] = {"$param": "gain"}
        leaf["id"] = "leaf"
        self.doc.data["blocks"] = []
        for graph, child, gain in [("middle", "leaf", {"$param": "gain"}), ("main", "middle", 5.0)]:
            self.doc.data["blocks"].append({"id": child, "graph": child})
            self.doc.data["graphs"].append({"id": graph, **port_pair, "parameters": leaf["parameters"],
                "nodes": [{"id": "call", "kind": "block", "block": child, **port_pair, "params": {"gain": gain}}],
                "edges": [{"from": {"input": "x"}, "to": {"node": "call", "port": "x"}},
                          {"from": {"node": "call", "port": "y"}, "to": {"output": "y"}}]})
        with TemporaryDirectory() as tmp:
            folder = compile_model(self.doc, Path(tmp) / "project")
            self.assertNotIn("mlker_example_layers", sys.modules)
            self.assertFalse((folder / "mlker_example_layers.py").exists())
            code = "from model import Model\nimport torch\nx=torch.tensor([2.]); assert Model()(x).item()==10"
            def run(path):
                return subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True, text=True,
                                      env={**os.environ, "PYTHONPATH": str(path)})
            failure = run("")
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("Cannot load example.scale", failure.stderr)
            resolved = run(EXT)
            self.assertEqual(resolved.returncode, 0, resolved.stderr)
