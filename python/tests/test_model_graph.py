"""Model format and compiler tests, runnable with unittest discovery from the repo."""

import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))
from mlker_graph import ModelError, build_ir, load_model, save_model, validate_model
from mlker_graph.ir import OpaqueRef, PythonRef
from mlker_graph.registry import REGISTRY

SAMPLES = PYTHON_ROOT.parent / "examples/models/v1"


def sample(name="feed-forward"):
    return load_model(SAMPLES / f"{name}.yaml")


class GraphTests(unittest.TestCase):
    def assert_problem(self, document, code):
        diagnostics = validate_model(document)
        self.assertEqual(diagnostics, validate_model(document))
        matches = [d for d in diagnostics if d.code == code and d.severity == "error"]
        self.assertTrue(matches, diagnostics)
        self.assertTrue(matches[0].message)
        with self.assertRaises(ModelError) as failure:
            build_ir(document)
        self.assertEqual(failure.exception.diagnostics, diagnostics)
        return matches[0].location

    def test_native_examples_and_round_trips(self):
        with TemporaryDirectory() as tmp:
            for path in sorted(SAMPLES.glob("*.yaml")):
                with self.subTest(name=path.name):
                    document = load_model(path)
                    self.assertEqual(validate_model(document), [])
                    first = build_ir(document)
                    saved = Path(tmp) / path.name
                    save_model(document, saved)
                    reopened = load_model(saved)
                    self.assertEqual(reopened.data, document.data)
                    self.assertEqual(build_ir(reopened).revision, first.revision)
                    self.assertEqual(build_ir(reopened).entry.id, first.entry.id)

    def test_invalid_fixtures(self):
        for file, code in [("cycle", "graph_cycle"), ("missing-output", "unbound_graph_output")]:
            with self.subTest(file=file):
                self.assert_problem(sample(f"invalid/{file}"), code)

    def test_cycle_through_two_nodes(self):
        doc = sample()
        graph = doc.data["graphs"][0]
        graph["edges"][0]["from"] = {"node": "activation", "port": "out"}
        self.assert_problem(doc, "graph_cycle")

    def test_schema_envelope_ids_and_types(self):
        cases = [
            (("format",), "unknown", "invalid_format"),
            (("schema_version",), 2, "unsupported_version"),
            (("schema_version",), True, "unsupported_version"),
            (("schema_version",), 1.0, "unsupported_version"),
            (("graphs",), {}, "schema"),
            (("model", "id"), "", "schema"),
            (("model", "revision"), 1, "schema"),
            (("model", "inputs"), ["features", "features"], "duplicate_id"),
            (("model", "outputs"), ["oops"], "unknown_port"),
            (("model", "inputs"), [], "missing_model_binding"),
            (("model", "entry"), "missing", "unknown_graph"),
            (("graphs", 0, "nodes", 0, "kind"), "exec", "schema"),
            (("graphs", 0, "nodes", 0, "block"), "another", "schema"),
            (("graphs", 0, "inputs", 0, "tensor", "shape"), [False], "schema"),
            (("graphs", 0, "outputs", 0, "required"), False, "port_contract"),
            (("graphs", 0, "nodes", 0, "inputs", 0, "tensor", "shape"), [-2], "schema"),
        ]
        for path, value, code in cases:
            with self.subTest(path=path, value=value):
                doc = sample()
                parent = doc.data
                for part in path[:-1]:
                    parent = parent[part]
                parent[path[-1]] = value
                self.assert_problem(doc, code)

    def test_scoped_duplicate_and_missing_ids(self):
        for path in [("graphs",), ("graphs", 0, "nodes"), ("graphs", 0, "parameters"),
                     ("graphs", 0, "inputs"), ("graphs", 0, "nodes", 0, "outputs"), ("blocks",)]:
            with self.subTest(path=path):
                doc = sample("hierarchical")
                records = doc.data
                for part in path:
                    records = records[part]
                records.append(deepcopy(records[0]))
                self.assert_problem(doc, "duplicate_id")
                del records[-1]["id"]
                self.assert_problem(doc, "missing_id")

    def test_endpoints_and_single_driver(self):
        cases = [
            ("to", {"node": "unknown", "port": "x"}, "unknown_node"),
            ("to", {"node": "dense", "port": "unknown"}, "unknown_port"),
            ("to", {"node": "dense", "port": "out"}, "wrong_direction"),
            ("from", {"node": "dense", "port": "x"}, "wrong_direction"),
            ("to", {"input": "features"}, "wrong_direction"),
            ("from", {"output": "scores"}, "wrong_direction"),
            ("from", {"input": "unknown"}, "unknown_port"),
            ("from", {"node": "dense", "input": "features"}, "schema"),
        ]
        for side, endpoint, code in cases:
            with self.subTest(endpoint=endpoint):
                doc = sample()
                doc.data["graphs"][0]["edges"][0][side] = endpoint
                loc = self.assert_problem(doc, code)
                self.assertEqual(loc.graph, "main")
                self.assertEqual(loc.path, f"/graphs/0/edges/0/{side}")
        doc = sample()
        graph = doc.data["graphs"][0]
        graph["edges"].append(graph["edges"][0])
        self.assert_problem(doc, "multiple_drivers")
        del graph["edges"][0]
        graph["edges"].pop()
        loc = self.assert_problem(doc, "unconnected_input")
        self.assertEqual(loc.node, "dense")
        self.assert_problem(doc, "unbound_graph_input")

    def test_tensor_compatibility(self):
        for spec, code in [({"dtype": "int64"}, "dtype_mismatch"), ({"shape": []}, "rank_mismatch"),
                           ({"shape": [None, 9]}, "dimension_mismatch"), ({"layout": "CN"}, "layout_mismatch")]:
            with self.subTest(spec=spec):
                doc = sample()
                graph = doc.data["graphs"][0]
                graph["inputs"][0]["tensor"]["layout"] = "NC"
                graph["nodes"][0]["inputs"][0]["tensor"] = spec
                self.assert_problem(doc, code)
        for shape in [None, [None, None], ["Batch", "Features"]]:
            doc = sample()
            doc.data["graphs"][0]["inputs"][0]["tensor"] = {"shape": shape}
            self.assertEqual(validate_model(doc), [])

    def test_registry_signatures_and_parameter_refs(self):
        cases = [
            ("registry", "core.NoSuchOp", "unknown_registry"),
            ("kind", "operator", "registry_kind"),
            ("inputs", [], "port_contract"),
            ("params", {}, "missing_parameter"),
            ("params", {"in_features": 8, "out_features": 4, "dropout": 1}, "unknown_parameter"),
            ("params", {"in_features": 8, "out_features": 0}, "parameter_value"),
            ("params", {"in_features": True, "out_features": 1}, "parameter_value"),
            ("params", {"in_features": 8, "out_features": {"$param": "missing"}}, "unknown_parameter_ref"),
            ("params", {"in_features": 8, "out_features": {"$param": "parent.width + 1"}}, "schema"),
            ("params", {"in_features": 8, "out_features": {"$param": "classes", "offset": 1}}, "schema"),
        ]
        for key, value, code in cases:
            with self.subTest(code=code, value=value):
                doc = sample()
                doc.data["graphs"][0]["nodes"][0][key] = value
                self.assertEqual(self.assert_problem(doc, code).node, "dense")
        doc = sample("cnn")
        doc.data["graphs"][0]["nodes"][0]["inputs"][0]["tensor"]["shape"] = ["B", 8]
        self.assert_problem(doc, "rank_mismatch")
        doc = sample("residual")
        node = doc.data["graphs"][0]["nodes"][1]
        node.update(registry="core.Concat", params={"axis": -1})
        self.assertEqual(build_ir(doc).entry.nodes["add"].parameters["axis"].value, -1)
        doc = sample()
        doc.data["graphs"][0]["parameters"][0]["default"] = False
        self.assert_problem(doc, "parameter_value")

    def test_typed_ir_and_snapshot(self):
        doc = sample()
        model = build_ir(doc)
        graph = model.entry
        dense = graph.nodes["dense"]
        self.assertIs(dense.target, REGISTRY["core.Linear"])
        self.assertEqual(dense.parameters["bias"].value, True)
        self.assertIs(dense.parameters["out_features"].reference, graph.parameters["classes"])
        self.assertEqual(dense.parameters["out_features"].value, 3)
        self.assertIs(graph.edges[0].source.port, model.inputs[0])
        self.assertIs(graph.edges[0].destination.port, dense.inputs["x"])
        self.assertEqual(graph.edges[0].destination.direction, "inputs")
        self.assertEqual(graph.edges[0].destination.node_id, "dense")
        self.assertEqual(model.outputs[0].tensor.shape, ("B", 3))
        self.assertEqual(dense.location.path, "/graphs/0/nodes/0")
        doc.data["model"]["metadata"]["task"] = "changed"
        self.assertEqual(model.metadata["task"], "classification")

    def test_block_bindings_and_dependency_order(self):
        doc = sample("hierarchical")
        model = build_ir(doc)
        small, large = (model.entry.nodes[k] for k in ("small", "large"))
        self.assertIs(small.target, large.target)
        self.assertIs(large.target, model.blocks["projection"])
        self.assertEqual(model.graphs[large.target.graph_id].parameters["width"].default, 4)
        self.assertEqual(small.parameters["width"].value, 2)
        self.assertEqual(large.parameters["width"].value, 16)
        self.assertIs(large.parameters["width"].reference, model.entry.parameters["wide_features"])
        doc.data["graphs"].reverse()
        self.assertEqual(build_ir(doc).entry.id, "main")

    def test_block_errors(self):
        for case, code in [("target", "unknown_graph"), ("block", "unknown_block"), ("port", "port_contract"),
                           ("dtype", "dtype_mismatch"), ("override", "parameter_value"),
                           ("ref", "unknown_parameter_ref"), ("self", "recursive_block"), ("mutual", "recursive_block")]:
            with self.subTest(case=case):
                doc = sample("hierarchical")
                main, head = doc.data["graphs"]
                node = main["nodes"][0]
                if case == "target":
                    doc.data["blocks"][0]["graph"] = "missing"
                elif case == "block":
                    node["block"] = "missing"
                elif case == "port":
                    node["outputs"][0]["id"] = "other"
                elif case == "dtype":
                    # Instance declares no dtype; the definition still requires float32.
                    main["inputs"][0]["tensor"]["dtype"] = "int32"
                elif case == "override":
                    node["params"]["width"] = -1
                elif case == "ref":
                    node["params"]["width"] = {"$param": "width"}
                elif case == "self":
                    head["nodes"].append(deepcopy(node))
                else:
                    doc.data["blocks"].append({"id": "outer", "graph": "main"})
                    head["nodes"].append({**node, "block": "outer"})
                self.assert_problem(doc, code)

    def test_transitive_instance_constraints(self):
        doc = sample("hierarchical")
        wrapper = deepcopy(doc.data["graphs"][0])
        wrapper.update(id="wrapper", nodes=[{
            "id": "call", "kind": "block", "block": "outer",
            "inputs": deepcopy(wrapper["inputs"]), "outputs": deepcopy(wrapper["outputs"]),
            "params": {"wide_features": 0},
        }])
        wrapper["edges"] = [{"from": {"input": "features"}, "to": {"node": "call", "port": "features"}}]
        for key in ["compact", "expanded"]:
            wrapper["edges"].append({"from": {"node": "call", "port": key}, "to": {"output": key}})
        doc.data["graphs"].append(wrapper)
        doc.data["blocks"].append({"id": "outer", "graph": "main"})
        location = self.assert_problem(doc, "parameter_value")
        self.assertEqual((location.graph, location.node), ("wrapper", "call"))
        wrapper["nodes"][0]["params"] = {}
        self.assertEqual(build_ir(doc).graphs["wrapper"].nodes["call"].parameters["wide_features"].value, 16)

    def test_python_and_opaque_contracts(self):
        doc = sample()
        graph = doc.data["graphs"][0]
        dense, activation = graph["nodes"]
        del dense["registry"], activation["registry"]
        dense.update(kind="python", implementation={"package": "nonexistent", "symbol": "layers.Dense", "version": "1.0"})
        activation.update(kind="opaque", source_op="vendor::Activation")
        activation["inputs"].append({"id": "optional", "required": False, "tensor": {}})
        diagnostics = validate_model(doc)
        self.assertEqual([(d.code, d.severity) for d in diagnostics], [("opaque_node", "warning")])
        model = build_ir(doc)
        self.assertIsInstance(model.entry.nodes["dense"].target, PythonRef)
        self.assertIsInstance(model.entry.nodes["activation"].target, OpaqueRef)


class DocumentTests(unittest.TestCase):
    def test_semantic_round_trip_extensions_unknowns_and_provenance(self):
        doc = sample()
        values = ["on", "2025-01-01", "1e3", True, 1e20, -0.005, None, {"研究": ["λ"]}]
        doc.data["extensions"] = {"test.tool": values}
        doc.data["graphs"][0]["nodes"][0]["future_hint"] = values
        provenance = {"source_format": "onnx", "source_identity": "fixture", "strategy": "manual",
                      "framework_version": "1", "warnings": ["partial"], "future": {"opset": 21}}
        doc.data["model"]["provenance"] = provenance
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.yaml"
            save_model(doc, path)
            reopened = load_model(path)
        self.assertEqual(reopened.data, doc.data)
        model = build_ir(reopened)
        self.assertEqual(model.provenance, provenance)
        self.assertEqual(model.extensions["test.tool"], values)

    def test_rejected_yaml_has_reader_diagnostic(self):
        invalid = ["format: 1\nformat: 2", "x: !!python/name:os.getcwd ''", "x: &a [1]\ny: *a",
                   "x: {1: value}", "x: [3", "x: 1e9999", "---\na: 1\n---\nb: 2"]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.yaml"
            for value in invalid:
                with self.subTest(value=value):
                    path.write_text(value)
                    with self.assertRaises(ModelError) as failure:
                        load_model(path)
                    problem = failure.exception.diagnostics[0]
                    self.assertEqual(problem.code, "invalid_yaml")
                    self.assertEqual(problem.location.source, str(path))

    def test_invalid_document_never_overwrites_destination(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.yaml"
            path.write_text("original")
            doc = sample()
            doc.data["schema_version"] = 99
            with self.assertRaises(ModelError):
                save_model(doc, path)
            self.assertEqual(path.read_text(), "original")

    def test_edited_extension_values_must_remain_serializable(self):
        recursive = []
        recursive.append(recursive)
        doc = sample()
        for value in [recursive, b"bytes", float("inf"), {2: "numeric key"}]:
            doc.data["extensions"] = {"vendor": value}
            diagnostics = validate_model(doc)
            self.assertEqual(diagnostics[0].code, "invalid_data")
            self.assertTrue(diagnostics[0].location.path.startswith("/extensions/vendor"))

    def test_cli_diagnostics_and_exit_code(self):
        for filename, expected in [("cnn", 0), ("invalid/cycle", 1)]:
            run = subprocess.run([sys.executable, "-m", "mlker_graph", str(SAMPLES / f"{filename}.yaml")],
                                 cwd=PYTHON_ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, expected, run.stderr)
            diagnostics = json.loads(run.stdout)
            if expected:
                problem = diagnostics[0]
                self.assertEqual(problem["code"], "graph_cycle")
                self.assertEqual(problem["location"]["graph"], "main")
                self.assertEqual(problem["severity"], "error")
            else:
                self.assertEqual(diagnostics, [])
