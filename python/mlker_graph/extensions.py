"""Read the user's explicitly trusted metadata without importing implementations."""

from dataclasses import dataclass
import json
import keyword
import os
from pathlib import Path
import re

from .diagnostics import Diagnostic, Location, ModelError
from .registry import Argument, Rule, literal
from .schema import IntegerValidator, SCHEMA_V2, invalid_data

TEXT = {"type": "string", "minLength": 1}
PORTS = {"type": "array", "items": {"$ref": "#/definitions/port"}}
FIELDS = {
    "id": {"$ref": "#/definitions/id"}, "name": TEXT, "description": TEXT, "version": TEXT,
    "package": TEXT, "symbol": TEXT, "kind": {"enum": ["module", "function"]},
    "inputs": PORTS, "outputs": PORTS,
    "parameters": {"type": "array", "items": {"$ref": "#/definitions/parameter"}},
    "dependencies": {"type": "array", "items": TEXT},
}
MANIFEST = IntegerValidator({
    "type": "array", "definitions": SCHEMA_V2["definitions"],
    "items": {"type": "object", "required": list(FIELDS),
              "properties": {**FIELDS, "example_inputs": {"type": "object", "additionalProperties": {
                  "$ref": "#/definitions/tensor"}}}, "additionalProperties": False},
})


@dataclass
class Extension:
    record: dict
    parameters: dict[str, Argument]

    @property
    def identity(self):
        return {key: self.record[key] for key in ("id", "package", "symbol", "version")}


def read_registry():
    filename = os.environ.get("MLKER_TRUSTED_REGISTRY")
    if not filename:
        return {}
    try:
        path = Path(filename)
        if not path.is_absolute():
            raise ValueError("Use an absolute registry path.")
        records = json.loads(path.read_text(encoding="utf-8"))
        issues = list(invalid_data(records, Location(filename), set()))
        if issues:
            raise ValueError(issues[0].message)
        error = next(MANIFEST.iter_errors(records), None)
        if error:
            raise ValueError(f"{error.json_path}: {error.message}")
        result = {}
        for r in records:
            entry = r["id"]
            if entry in result or entry.startswith("core.") or entry in ("input", "output"):
                raise ValueError(f"Duplicate or reserved registration {entry}.")
            names = [*r["package"].split("."), r["symbol"], *(p["id"] for p in r["parameters"])]
            if any(not n.isascii() or not n.isidentifier() or keyword.iskeyword(n) or n == "__debug__" for n in names):
                raise ValueError(f"{entry}: expected Python identifiers.")
            for side in ("inputs", "outputs", "parameters"):
                ids = [p["id"] for p in r[side]]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"{entry}: duplicate {side} ID.")
            if any(not p.get("required", True) for p in r["inputs"] + r["outputs"]):
                raise ValueError(f"{entry}: all registered ports must be required.")
            if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*(?:==[a-zA-Z0-9][a-zA-Z0-9_.+-]*)?", dep)
                   for dep in r["dependencies"]):
                raise ValueError(f"{entry}: use distribution names or exact version pins.")
            params = {}
            for param in r["parameters"]:
                arg = Argument(Rule(param["type"]), literal(param["default"]))
                if not arg.rule.accepts(arg.default):
                    raise ValueError(f"{entry}: invalid default for {param['id']}.")
                params[param["id"]] = arg
            result[entry] = Extension(r, params)
        return result
    except (ValueError, OSError) as exc:
        raise ModelError([Diagnostic("extension_registry", "error", f"Cannot read trusted registry: {exc}",
                                     Location(filename))]) from exc
