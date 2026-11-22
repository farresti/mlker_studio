import ast
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))
from mlker_graph import ModelDocument, ModelError, load_model, save_model
from mlker_graph.pytorch import CompileError, compile_model

EXAMPLES = PYTHON_ROOT.parent / "examples/models"


def sample(name):
    return load_model(EXAMPLES / f"{name}.yaml")


def pipeline(steps, shape, output=None):
    def port(name, dims=None):
        return {"id": name, "tensor": {"shape": dims}}

    nodes, edges = [], []
    source = {"input": "data"}
    for i, (registry, params) in enumerate(steps):
        key = f"step{i}"
        kind = "module" if registry in ("Linear", "BatchNorm2d", "Dropout", "Conv2d") else "operator"
        nodes.append({"id": key, "kind": kind, "registry": f"core.{registry}", "params": params,
                      "inputs": [port("x")], "outputs": [port("out")]})
        edges.append({"from": source, "to": {"node": key, "port": "x"}})
        source = {"node": key, "port": "out"}
    edges.append({"from": source, "to": {"output": "result"}})
    return ModelDocument({
        "format": "mlker.model", "schema_version": 2,
        "model": {"id": "test", "name": "Pipeline", "revision": "r1", "entry": "g",
                  "inputs": ["data"], "outputs": ["result"]},
        "graphs": [{"id": "g", "inputs": [port("data", shape)], "outputs": [port("result", output)],
                    "nodes": nodes, "edges": edges}],
    }, "<test>")


class ExportTests(unittest.TestCase):
    def test_cli_and_deterministic_standalone_files(self):
        with TemporaryDirectory() as tmp:
            first, second = (Path(tmp) / folder for folder in ("a", "b"))
            doc = sample("v1/hierarchical")
            compile_model(doc, first)
            compile_model(doc, first)
            command = [sys.executable, "-m", "mlker_graph.pytorch", str(EXAMPLES / "v1/hierarchical.yaml"), str(second)]
            subprocess.run(command, cwd=PYTHON_ROOT, check=True, capture_output=True)
            self.assertEqual(sorted(p.name for p in first.iterdir()), sorted(p.name for p in second.iterdir()))
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
                if path.suffix == ".py":
                    ast.parse(path.read_text())
                    self.assertNotIn("mlker_graph", path.read_text())
            config = json.loads((first / "config.json").read_text())
            self.assertEqual(config["revision"], "r1")
            self.assertEqual([p["inferred_shape"] for p in config["outputs"]], [["B", 2], ["B", 16]])
            (first / "model.py").write_text("original")
            with self.assertRaises(FileExistsError):
                compile_model(doc, first)
            self.assertEqual((first / "model.py").read_text(), "original")

    def test_shape_errors_have_source_location_and_create_no_files(self):
        cases = [
            ("Linear", {"in_features": 9, "out_features": 5}, ["B", 8]),
            ("Conv2d", {"in_channels": 4, "out_channels": 2, "kernel_size": 3}, [2, 3, 8, 8]),
            ("MaxPool2d", {"kernel_size": 8, "stride": 2}, [2, 4, 2, 2]),
            ("BatchNorm2d", {"num_features": 3}, [4, 4, 3, 3]),
            ("Flatten", {"start_dim": 3}, [2, 3]),
            ("Reshape", {"shape": [4, -1]}, [3, 3]),
            ("Reshape", {"shape": [-1, -1]}, None),
            ("Permute", {"dims": [1, 1]}, None),
        ]
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "broken"
            for op, params, shape in cases:
                with self.subTest(op=op), self.assertRaises(CompileError) as failure:
                    compile_model(pipeline([(op, params)], shape), out)
                problem = failure.exception.diagnostics[0]
                self.assertEqual(problem.code, "compile_shape")
                self.assertEqual((problem.location.graph, problem.location.node), ("g", "step0"))
                self.assertFalse(out.exists())
            with self.assertRaises(CompileError):
                compile_model(pipeline([("Sigmoid", {})], [3], [5]), out)

    def test_unsupported_node_and_schema_failures_are_distinct(self):
        with TemporaryDirectory() as tmp:
            doc = sample("v1/feed-forward")
            node = doc.data["graphs"][0]["nodes"][0]
            del node["registry"]
            for target in [{"kind": "opaque", "source_op": "custom"}, {"kind": "python", "implementation": {
                    "package": "must_never_import", "symbol": "custom"}}]:
                with self.subTest(kind=target["kind"]):
                    node.pop("source_op", None)
                    node.update(target)
                    with self.assertRaises(CompileError) as invalid:
                        compile_model(doc, tmp)
                    self.assertEqual(invalid.exception.diagnostics[0].code, "compile_unsupported")
                    self.assertIn("Replace", str(invalid.exception))
            with self.assertRaises(ModelError) as invalid:
                compile_model(sample("v1/invalid/cycle"), tmp)
            self.assertNotIsInstance(invalid.exception, CompileError)

    def test_identity_conflicts_are_diagnosed(self):
        for mode, code in [("params", "compile_instance"), ("slot", "compile_parameter_slot"),
                           ("conflict", "compile_parameter_id"), ("shape", "compile_parameter_shape"),
                           ("version", "sharing_version")]:
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                doc = sample("v2/shared-branches")
                left, right, aux = doc.data["graphs"][0]["nodes"][:3]
                if mode == "params":
                    right["params"]["bias"] = False
                elif mode == "slot":
                    left["parameter_ids"] = {"missing": "a"}
                elif mode == "conflict":
                    right["parameter_ids"] = {"weight": "other"}
                elif mode == "shape":
                    aux["parameter_ids"] = {"bias": "projection_weight"}
                else:
                    doc.data["schema_version"] = 1
                with self.assertRaises(ModelError) as invalid:
                    compile_model(doc, tmp)
                self.assertEqual(invalid.exception.diagnostics[0].code, code)

    def test_constructor_bounds_and_integer_lists(self):
        with TemporaryDirectory() as tmp:
            for op, params in [("Dropout", {"p": 1.1}), ("Reshape", {"shape": [True, 2]}),
                               ("Linear", {"in_features": 2}), ("Permute", {"dims": "0,1"})]:
                with self.subTest(op=op), self.assertRaises(ModelError):
                    compile_model(pipeline([(op, params)], None), tmp)
            doc = pipeline([("Reshape", {"shape": {"$param": "dims"}})], None)
            doc.data["graphs"][0]["parameters"] = [{"id": "dims", "type": "ints", "default": [-1, 3]}]
            path = Path(tmp) / "doc.yaml"
            save_model(doc, path)
            folder = compile_model(load_model(path), Path(tmp) / "export")
            config = json.loads((folder / "config.json").read_text())
            self.assertEqual(config["outputs"][0]["inferred_shape"], [None, 3])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU runtime checks require PyTorch")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def instantiate(self, doc):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        package = "export_" + root.name
        project = compile_model(doc, root / package)
        sys.path.insert(0, str(root))
        self.addCleanup(sys.path.remove, str(root))
        module = importlib.import_module(package)
        for key in list(sys.modules):
            if key == package or key.startswith(package + "."):
                self.addCleanup(sys.modules.pop, key, None)
        return module.Model().cpu(), json.loads((project / "config.json").read_text())

    def test_examples_forward_backward_and_declared_shapes(self):
        torch = self.torch
        cases = [("feed-forward", (2, 8), [(2, 3)], 27), ("cnn", (2, 1, 8, 8), [(2, 5)], 365),
                 ("residual", (3, 7), [(3, 7)], 0), ("hierarchical", (2, 8), [(2, 2), (2, 16)], 162)]
        for name, shape, expected, parameter_count in cases:
            with self.subTest(name=name):
                doc = sample(f"v1/{name}")
                for graph in doc.data["graphs"]:
                    graph["nodes"].reverse()
                model, config = self.instantiate(doc)
                x = torch.randn(shape, device="cpu", requires_grad=True)
                result = model(x)
                outputs = result if isinstance(result, tuple) else (result,)
                self.assertEqual([tuple(y.shape) for y in outputs], expected)
                for tensor, port in zip(outputs, config["outputs"]):
                    for dim, actual in zip(port["shape"], tensor.shape):
                        if isinstance(dim, int):
                            self.assertEqual(dim, actual)
                if name == "residual":
                    torch.testing.assert_close(result, x + torch.relu(x))
                sum(y.sum() for y in outputs).backward()
                self.assertIsNotNone(x.grad)
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
                self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_public_argument_order_collision_and_shared_gradients(self):
        torch = self.torch
        doc = sample("v2/shared-branches")
        graph = doc.data["graphs"][0]
        rename = {"left": "class.x", "right": "class-x"}
        doc.data["model"]["inputs"] = ["class-x", "class.x"]
        for port in graph["inputs"]:
            port["id"] = rename[port["id"]]
        for edge in graph["edges"]:
            if "input" in edge["from"]:
                edge["from"]["input"] = rename[edge["from"]["input"]]
        # Port declaration order is independent of the registry's operand order.
        graph["nodes"][-1]["inputs"].reverse()
        model, config = self.instantiate(doc)
        self.assertEqual([p["argument"] for p in config["inputs"]], ["input_class_x", "input_class_x_2"])
        enc, aux = model.layer_encoder, model.layer_auxiliary
        self.assertIs(aux.weight, enc.weight)
        self.assertIsNot(aux.bias, enc.bias)
        self.assertEqual(len(list(model.children())), 2)
        self.assertEqual(len(list(model.parameters())), 3)
        left, right = torch.randn(2, 4), torch.randn(2, 4)
        result = model(right, left)
        torch.testing.assert_close(result, torch.cat((enc(left) + aux(left), enc(right)), dim=-1))
        result.sum().backward()
        expected = (2 * left.sum(0) + right.sum(0)).expand(3, 4)
        torch.testing.assert_close(enc.weight.grad, expected)

    def test_stateful_and_tensor_operations(self):
        torch = self.torch
        steps = [("BatchNorm2d", {"num_features": 3}), ("Dropout", {"p": 1}),
                 ("AdaptiveAvgPool2d", {"height": 2, "width": 1}),
                 ("Permute", {"dims": [0, 2, 3, 1]}), ("Reshape", {"shape": [-1, 6]}),
                 ("GELU", {}), ("SiLU", {}), ("Tanh", {}), ("Sigmoid", {})]
        model, config = self.instantiate(pipeline(steps, [2, 3, 4, 4]))
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        self.assertEqual(config["outputs"][0]["inferred_shape"], [2, 6])
        torch.testing.assert_close(model(x), torch.full((2, 6), 0.5))
        self.assertEqual(model.layer_step0.num_batches_tracked.item(), 1)
        model.eval()
        result = model(x)
        ref = torch.nn.functional.adaptive_avg_pool2d(model.layer_step0(x), (2, 1))
        ref = ref.permute(0, 2, 3, 1).reshape(2, 6)
        ref = torch.sigmoid(torch.tanh(torch.nn.functional.silu(torch.nn.functional.gelu(ref))))
        torch.testing.assert_close(result, ref)
        result.sum().backward()
        self.assertIsNotNone(model.layer_step0.weight.grad)
        self.assertIsNotNone(x.grad)

    def test_shared_block_calls_and_runtime_error(self):
        torch = self.torch
        doc = sample("v1/hierarchical")
        doc.data["schema_version"] = 2
        main = doc.data["graphs"][0]
        for node, port in zip(main["nodes"], main["outputs"]):
            node["instance_id"] = "head"
            node["params"] = {"width": 2}
            node["outputs"][0]["tensor"]["shape"] = ["B", 2]
            port["tensor"]["shape"] = ["B", 2]
        model, config = self.instantiate(doc)
        self.assertEqual(len(config["classes"]), 2)
        self.assertEqual(len(list(model.children())), 1)
        x = torch.randn(2, 8, requires_grad=True)
        a, b = model(x)
        torch.testing.assert_close(a, b)
        (a + b).sum().backward()
        expected = (2 * x.detach().sum(0)).expand(2, 8)
        torch.testing.assert_close(model.layer_head.layer_project.weight.grad, expected)
        with self.assertRaises(RuntimeError):
            model(torch.ones(2, 9))
