"""PyTorch-specific values between lowering and source rendering."""

from dataclasses import dataclass, field
import re

from ..diagnostics import Diagnostic, Location, ModelError
from ..registry import ParameterValue


class CompileError(ModelError):
    """A validated model cannot be lowered by this backend."""


def fail(code: str, message: str, location: Location):
    raise CompileError([Diagnostic(code, "error", message, location)])


class Names:
    def __init__(self):
        self.used: set[str] = set()

    def allocate(self, prefix: str, source: str) -> str:
        # Prefixes keep keywords, digits, dunder names and nn.Module methods separate.
        base = prefix + re.sub(r"[^A-Za-z0-9_]", "_", source)
        name = base
        suffix = 2
        while name in self.used:
            name = f"{base}_{suffix}"
            suffix += 1
        self.used.add(name)
        return name


@dataclass
class Layer:
    name: str
    constructor: str
    parameters: dict[str, ParameterValue]


@dataclass
class Call:
    outputs: list[str]
    function: str
    inputs: list[str]
    parameters: dict[str, ParameterValue]


@dataclass
class ClassPlan:
    name: str
    inputs: dict[str, str]
    module_paths: dict[str, str] = field(default_factory=dict)
    layers: list[Layer] = field(default_factory=list)
    ties: list[tuple[str, str]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
