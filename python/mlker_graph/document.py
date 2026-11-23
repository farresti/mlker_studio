"""Read and write the data-only YAML subset used for model documents."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from .diagnostics import Diagnostic, Location, ModelError


class DocumentLoader(yaml.SafeLoader):
    yaml_implicit_resolvers = {}

    def compose_node(self, parent, index):
        event = self.peek_event()
        if isinstance(event, yaml.AliasEvent) or event.anchor or event.tag:
            raise yaml.MarkedYAMLError(problem="Anchors, aliases and explicit tags are unsupported.",
                                       problem_mark=event.start_mark)
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in mapping or key == "<<":
                raise yaml.MarkedYAMLError(problem="Expected a unique string key; merge keys are unsupported.",
                                           problem_mark=key_node.start_mark)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping

    def construct_scalar(self, node):
        value = super().construct_scalar(node)
        if node.tag.endswith(":float") and not math.isfinite(float(value)):
            raise yaml.MarkedYAMLError(problem="Expected a finite number.", problem_mark=node.start_mark)
        return value


# JSON scalar spellings keep identifiers such as 'on', 'no' and dates as strings.
for tag, pattern, first in (
    ("null", r"null|~|", ["n", "~", ""]),
    ("bool", r"true|false", list("tf")),
    ("int", r"-?(?:0|[1-9][0-9]*)", list("-0123456789")),
    ("float", r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)", list("-0123456789")),
):
    DocumentLoader.add_implicit_resolver("tag:yaml.org,2002:" + tag, re.compile(f"(?:{pattern})\\Z"), first)


class DocumentDumper(yaml.SafeDumper):
    yaml_implicit_resolvers = DocumentLoader.yaml_implicit_resolvers

    def ignore_aliases(self, data):
        return True


@dataclass
class ModelDocument:
    data: Any
    source: str


def load_model(path: str | Path) -> ModelDocument:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            return ModelDocument(yaml.load(stream, Loader=DocumentLoader), str(path))
    except (yaml.YAMLError, UnicodeError, RecursionError, ValueError) as error:
        mark = getattr(error, "problem_mark", None)
        position = f"Line {mark.line + 1}, column {mark.column + 1}: " if mark else ""
        message = position + (getattr(error, "problem", None) or str(error))
        raise ModelError([Diagnostic("invalid_yaml", "error", message, Location(str(path)))]) from error


def save_model(document: ModelDocument, path: str | Path, *, draft: bool = False) -> None:
    from .validation import validate_model

    from .schema import schema_diagnostics

    diagnostics = schema_diagnostics(document) if draft else validate_model(document)
    if any(d.severity == "error" for d in diagnostics):
        raise ModelError(diagnostics)
    path = Path(path)
    stream = NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            yaml.dump(document.data, stream, Dumper=DocumentDumper, sort_keys=False, allow_unicode=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
