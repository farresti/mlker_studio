"""Editor protocol v2 operations. Graph dependencies load only for editor requests."""

from dataclasses import asdict
import json
from pathlib import Path


class EditorError(ValueError):
    def __init__(self, code, message, issues=()):
        super().__init__(message)
        self.code = code
        self.issues = [asdict(item) for item in issues]


# Operator signatures and constraints come from REGISTRY. These are UI suggestions
# for required values, independent of optional constructor defaults.
SUGGESTIONS = {"in_channels": 1, "out_channels": 4, "num_features": 4, "in_features": 36,
               "out_features": 5, "height": 1, "width": 1, "axis": 1, "dims": [0, 1], "shape": [-1, 1]}
DESCRIPTIONS = {
    "Conv2d": "Apply a learned spatial filter to an image tensor.",
    "Linear": "Project the last feature dimension through a learned affine map.",
    "BatchNorm2d": "Normalize image channels using batch or running statistics.",
    "Dropout": "Randomly zero elements during training.",
    "MaxPool2d": "Take the maximum in each spatial window.",
    "AdaptiveAvgPool2d": "Average spatial regions to a fixed output height and width.",
    "Flatten": "Combine a contiguous range of dimensions.",
    "Add": "Add two tensors using broadcasting.", "Concat": "Join two tensors along an axis.",
    "Reshape": "Rearrange dimensions while preserving the element count.",
    "Permute": "Reorder tensor axes.",
}


def registry():
    from mlker_graph.registry import REGISTRY
    entries = []
    for op in REGISTRY.values():
        name = op.id.split(".")[-1]
        params = {}
        for key, arg in op.parameters.items():
            suggestion = arg.default
            if suggestion is None:
                suggestion = SUGGESTIONS.get(key, 1)
                if key == "kernel_size":
                    suggestion = 3 if name == "Conv2d" else 2
                if key == "stride" and name == "MaxPool2d":
                    suggestion = 2
            params[key] = {**asdict(arg.rule), "default": arg.default, "initial": suggestion}
        entries.append({"id": op.id, "name": name, "kind": op.kind, "inputs": op.inputs,
                        "outputs": op.outputs, "params": params,
                        "description": DESCRIPTIONS.get(name, f"Apply the {name} activation elementwise.")})
    from mlker_graph.extensions import read_registry
    for extension in read_registry().values():
        r = extension.record
        entries.append({**{k: r[k] for k in ("id", "name", "description")}, "kind": "python",
                        "implementation": extension.identity, "contracts": {k: r[k] for k in ("inputs", "outputs")},
                        "inputs": [p["id"] for p in r["inputs"]], "outputs": [p["id"] for p in r["outputs"]],
                        "params": {k: {**asdict(a.rule), "default": a.default, "initial": a.default}
                                   for k, a in extension.parameters.items()}})
    return {"version": 1, "operations": entries}


def selected_path(params, yaml_file=False):
    value = params.get("path")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise EditorError("invalid_path", "Select an absolute local path.")
    path = Path(value)
    if yaml_file and path.suffix.lower() not in (".yaml", ".yml"):
        raise EditorError("invalid_path", "Choose a .yaml or .yml model file.")
    return path


def check_editable(document):
    from mlker_graph.schema import schema_diagnostics
    from mlker_graph import validate_model
    issues = schema_diagnostics(document)
    if not issues:
        # These defects prevent unique canvas identities or entry-graph selection.
        issues = [d for d in validate_model(document) if d.code == "duplicate_id"
                  or (d.code == "unknown_graph" and d.location.path == "/model/entry")]
    if issues:
        raise EditorError("invalid_document", "Cannot display this document.", issues)

    def safe_numbers(value):
        if isinstance(value, dict):
            return all(safe_numbers(item) for item in value.values())
        if isinstance(value, list):
            return all(safe_numbers(item) for item in value)
        return type(value) is not int or abs(value) <= 2**53 - 1

    if not safe_numbers(document.data):
        raise EditorError("integer_range", "This document contains integers outside the editor's exact numeric range.")


def handle(method, params):
    try:
        return operation(method, params)
    except ImportError as exc:
        raise EditorError("missing_dependency", f"{exc}. Install the graph dependencies in MLKER_PYTHON.") from exc
    except OSError as exc:
        raise EditorError("file_error", str(exc)) from exc


def operation(method, params):
    from mlker_graph import ModelDocument, ModelError, load_model, save_model, validate_model
    from mlker_graph.pytorch import CompileError, compile_model

    try:
        if method == "browse_timm":
            from mlker_import.sources import browse
            return browse(params.get("query", ""), params.get("pretrained_only") is True, max(0, int(params.get("offset", 0))))
        if method == "import_model":
            from mlker_import import import_model
            try:
                imported = import_model(params)
            except ImportError:
                raise
            except PermissionError as exc:
                raise EditorError("import_permission", str(exc)) from exc
            except Exception as exc:
                raise EditorError("import_failed", f"{type(exc).__name__}: {exc}") from exc
            check_editable(imported)
            return {"document": imported.data}
        if method == "registry":
            return registry()
        if method == "load":
            path = selected_path(params, True)
            if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
                raise EditorError("file_error", "Select a regular YAML file of at most 4 MiB.")
            document = load_model(path)
            check_editable(document)
            return {"document": document.data}
        document = ModelDocument(params.get("document"), params.get("source") or "<editor>")
        if method == "validate":
            return {"diagnostics": [asdict(i) for i in validate_model(document)]}
        if method == "save":
            path = selected_path(params, True)
            save_model(document, path, draft=True)
            return {"path": str(path)}
        if method == "compile":
            folder = compile_model(document, selected_path(params))
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            names = ["model.py", "blocks.py"] if len(config["classes"]) > 1 else ["model.py"]
            sources = {name: (folder / name).read_text(encoding="utf-8")
                       for name in names}
            return {"path": str(folder), "sources": sources}
        raise EditorError("unknown_method", f"Unknown editor method {method}.")
    except ModelError as exc:
        if exc.diagnostics and exc.diagnostics[0].code == "extension_registry":
            raise EditorError("extension_registry", exc.diagnostics[0].message, exc.diagnostics) from exc
        code = "compile_error" if isinstance(exc, CompileError) else "invalid_document"
        raise EditorError(code, "Compilation failed." if code == "compile_error" else "Model validation failed.",
                          exc.diagnostics) from exc
