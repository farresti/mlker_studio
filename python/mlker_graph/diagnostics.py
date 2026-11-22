"""Stable diagnostic payloads shared by the reader, validator and CLI."""

from dataclasses import dataclass, replace
from typing import Literal

@dataclass(frozen=True)
class Location:
    source: str
    path: str = ""
    graph: str | None = None
    node: str | None = None

    def child(self, *parts: str | int) -> "Location":
        suffix = "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)
        return replace(self, path=self.path + suffix)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Literal["error", "warning"]
    message: str
    location: Location


class ModelError(ValueError):
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        super().__init__("\n".join(f"{d.location.path}: {d.message}" for d in diagnostics))
