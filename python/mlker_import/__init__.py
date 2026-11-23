"""Standalone CPU importer. Opening a model document never calls this package."""
from copy import deepcopy
import json
from math import prod
import os
from pathlib import Path
import runpy
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4

import torch

from mlker_graph import ModelDocument, validate_model
from mlker_graph.pytorch import compile_model
from .extract import composite, extract, qualified
from .sources import construct


def examples(specs, seed):
    if not isinstance(specs, list) or len(specs) > 16:
        raise ValueError("Supply up to 16 positional tensor specifications.")
    generator = torch.Generator().manual_seed(seed)
    tensors = []
    elements = 0
    for spec in specs:
        shape = spec["shape"]
        if not isinstance(shape, list) or len(shape) > 12 or any(type(d) is not int or d < 1 for d in shape):
            raise ValueError("Example tensor dimensions must be positive integers.")
        elements += prod(shape)
        if elements > 8_000_000:
            raise ValueError("Examples exceed eight million elements.")
        dtype = spec.get("dtype", "float32")
        if dtype in ("float32", "float64"):
            tensor = torch.randn(shape, generator=generator, dtype=getattr(torch, dtype))
        elif dtype in ("int64", "int32", "bool"):
            tensor = torch.randint(0, 2, shape, generator=generator, dtype=getattr(torch, dtype))
        else:
            raise ValueError(f"Unsupported example dtype {dtype}.")
        tensors.append(tensor)
    return tensors


def tensor_spec(t):
    return {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}


def compare(model, converted, specs):
    for seed in (17, 29):
        args = examples(specs, seed)
        with torch.inference_mode():
            expected = model(*(v.clone() for v in args))
            actual = converted(*args)
        if type(expected) != type(actual):
            raise ValueError("Different output container types.")
        left = expected if isinstance(expected, tuple) else (expected,)
        right = actual if isinstance(actual, tuple) else (actual,)
        if len(left) != len(right) or not all(isinstance(t, torch.Tensor) for t in (*left, *right)):
            raise ValueError("Expected matching tensor outputs.")
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, check_dtype=True)
    return {"status": "passed", "mode": "eval", "seeds": [17, 29], "rtol": 1e-5, "atol": 1e-6,
            "outputs": [tensor_spec(t) for t in left]}


def snapshot(folder, state, model_id):
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Only tensor state_dict snapshots are supported.")
    directory = Path(folder)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("Select an existing absolute weights directory.")
    target = directory / (model_id + ".pt")
    with NamedTemporaryFile(dir=directory) as stage:
        torch.save(state, stage)
        stage.flush()
        os.link(stage.name, target)
    return str(target)


def import_model(options):
    """Construct a fresh model and return its new document. Only weights_dir writes a persistent artifact."""
    strategy = options.get("strategy", "fx")
    if strategy not in ("fx", "composite"):
        raise ValueError("Choose FX or composite import.")
    specs = options.get("inputs", [])
    inputs = examples(specs, 17)
    model, versions = construct(options)
    original_training = model.training
    hierarchy = [{"path": path, "class": qualified(m), "source_training": m.training}
                 for path, m in model.named_modules(remove_duplicate=False)]
    model.eval()
    identity = options["identity"]
    try:
        graph, modules, warnings, used = extract(model, strategy, len(inputs))
    except Exception as error:
        graph, modules, warnings, used = composite(model, f"Extraction failed: {error}", len(inputs))
    model_id = "import_" + uuid4().hex
    provenance = {"source_format": options["source"], "source_identity": identity, "framework_version": versions["torch"],
                  "strategy": used, "configuration": options.get("config", {}), "dependencies": versions,
                  "class": qualified(model), "source_training": original_training, "mode": "eval",
                  "pretrained": options.get("pretrained") is True, "source_state_dict": options.get("checkpoint"),
                  "example_inputs": specs, "hierarchy": hierarchy, "warnings": warnings}
    if options["source"] == "timm":
        cfg = getattr(model, "pretrained_cfg", {})
        provenance["pretrained_config"] = {k: cfg[k] for k in
            ("architecture", "tag", "input_size", "mean", "std", "interpolation", "num_classes")
            if k in cfg}
        # Normalize timm's tuple metadata to the document's JSON-like scalar tree.
        provenance["pretrained_config"] = json.loads(json.dumps(provenance["pretrained_config"]))
    def document():
        return ModelDocument({"format": "mlker.model", "schema_version": 2, "model": {
            "id": model_id, "name": identity.split(":")[-1], "revision": "r1", "entry": "main",
            "inputs": [p["id"] for p in graph["inputs"]], "outputs": [p["id"] for p in graph["outputs"]],
            "provenance": provenance}, "graphs": [graph]}, "<import>")
    state = None
    verification = {"status": "unverified", "mode": "eval"}
    if not any(n["kind"] == "opaque" for n in graph["nodes"]):
        try:
            with TemporaryDirectory() as tmp:
                project = compile_model(document(), Path(tmp) / "model")
                paths = json.loads((project / "config.json").read_text())["classes"][0]["module_paths"]
                state = {paths[k] + "." + slot: tensor for k, module in modules.items()
                         for slot, tensor in module.state_dict().items()}
                if options.get("verify") is True and inputs:
                    converted = runpy.run_path(str(project / "model.py"))["Model"]().eval()
                    converted.load_state_dict(state, strict=True)
                    verification = compare(deepcopy(model), converted, specs)
        except Exception as error:
            message = f"Native compilation or output comparison failed: {error}"
            graph, modules, reasons, used = composite(model, message, len(inputs))
            warnings.extend(reasons)
            provenance["strategy"] = used
            verification = {"status": "failed", "detail": str(error), "mode": "eval"}
            state = None
    if inputs and any(n["kind"] == "opaque" for n in graph["nodes"]):
        try:
            with torch.inference_mode():
                output = deepcopy(model)(*inputs)
            values = output if type(output) is tuple else (output,)
            if all(isinstance(t, torch.Tensor) for t in values):
                provenance["observed_outputs"] = [tensor_spec(t) for t in values]
                if used == "module_tree":
                    outputs = [{"id": f"result{i}", "tensor": {}} for i in range(len(values))]
                    graph["outputs"] = graph["nodes"][0]["outputs"] = outputs
                    graph["edges"] = [w for w in graph["edges"] if "output" not in w["to"]]
                    graph["edges"].extend({"from": {"node": "source", "port": p["id"]}, "to": {"output": p["id"]}}
                                          for p in outputs)
            else:
                warnings.append("The source returned a structured value outside the tensor output contract.")
        except Exception as error:
            warnings.append(f"Example execution failed: {type(error).__name__}: {error}")
    if len(inputs) == len(graph["inputs"]):
        for p, tensor in zip(graph["inputs"], inputs):
            p["extensions"] = {"mlker.example": tensor_spec(tensor)}
    for p, observed in zip(graph["outputs"], verification.get("outputs", provenance.get("observed_outputs", []))):
        p["extensions"] = {"mlker.observed": observed}
    opaque = [n["id"] for n in graph["nodes"] if n["kind"] == "opaque"]
    native = [n["id"] for n in graph["nodes"] if n["kind"] != "opaque"]
    quality = "fully editable" if not opaque else "partially editable" if native else "inspectable only"
    if verification["status"] == "unverified":
        verification["detail"] = ("Opaque operations require the original source." if opaque else
                                  "Supply example inputs and enable the eval comparison.")
    provenance["diagnostics"] = [{"code": "unsupported_operation", "severity": "warning",
                                  "graph": graph["id"], "node": n["id"], "source_op": n["source_op"]}
                                 for n in graph["nodes"] if n["kind"] == "opaque"]
    warnings.append("Captured in eval mode. Training behavior and other inputs are unverified.")
    provenance.update(quality=quality, converted=native, opaque=opaque, verification=verification,
                      weights={"format": "architecture only"})
    doc = document()
    issues = [d.message for d in validate_model(doc) if d.severity == "error"]
    if issues:
        raise ValueError("Invalid import: " + "; ".join(issues))
    if options.get("weights_dir"):
        fmt = "generated state_dict" if state is not None else "source state_dict"
        path = snapshot(options["weights_dir"], state if state is not None else model.state_dict(), model_id)
        provenance["weights"] = {"format": fmt, "path": path}
    return doc
