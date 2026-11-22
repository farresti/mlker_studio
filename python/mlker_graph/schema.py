"""Check the version envelope before dispatching to the v1 schema."""

import json
import math
from dataclasses import replace
from importlib.resources import files

from jsonschema import Draft7Validator, validators

from .diagnostics import Diagnostic, Location
from .document import ModelDocument

# YAML distinguishes integer literals from floating point literals.
IntegerValidator = validators.extend(Draft7Validator, type_checker=Draft7Validator.TYPE_CHECKER.redefine(
    "integer", lambda checker, instance: type(instance) is int
))
SCHEMA = json.loads(files(__package__).joinpath("schema-v1.json").read_text(encoding="utf-8"))
VALIDATOR = IntegerValidator(SCHEMA)


def invalid_data(value, at: Location, ancestors: set[int]):
    """Apply the reader's data restrictions to documents edited through Python."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) not in (list, dict) or id(value) in ancestors:
        yield Diagnostic("invalid_data", "error", "Expected finite, acyclic YAML data.", at)
        return
    ancestors.add(id(value))
    entries = value.items() if isinstance(value, dict) else enumerate(value)
    for key, item in entries:
        if isinstance(value, dict) and (type(key) is not str or key == "<<"):
            yield Diagnostic("invalid_data", "error", "Expected a string key other than <<.", at)
        else:
            yield from invalid_data(item, at.child(key), ancestors)
    ancestors.remove(id(value))


def schema_diagnostics(document: ModelDocument) -> list[Diagnostic]:
    data = document.data
    root = Location(document.source)
    if not isinstance(data, dict):
        return [Diagnostic("schema", "error", "Expected a model document mapping.", root)]
    if data.get("format") != "mlker.model":
        return [Diagnostic("invalid_format", "error", "Expected format mlker.model.", root.child("format"))]
    version = data.get("schema_version")
    if type(version) is not int or version != 1:
        return [Diagnostic("unsupported_version", "error", f"Expected schema version 1, found {version!r}.",
                           root.child("schema_version"))]

    diagnostics = list(invalid_data(data, root, set()))
    if diagnostics:
        return diagnostics
    for issue in VALIDATOR.iter_errors(data):
        parts = list(issue.absolute_path)
        location = root.child(*parts)
        current = data
        for i, part in enumerate(parts):
            current = current[part]
            if isinstance(current, dict) and isinstance(current.get("id"), str):
                if parts[:1] == ["graphs"] and i == 1:
                    location = replace(location, graph=current["id"])
                if parts[:1] == ["graphs"] and parts[2:3] == ["nodes"] and i == 3:
                    location = replace(location, node=current["id"])
        code = "missing_id" if issue.validator == "required" and issue.message == "'id' is a required property" else "schema"
        diagnostics.append(Diagnostic(code, "error", issue.message, location))
    return sorted(diagnostics, key=lambda d: (d.location.path, d.message))
