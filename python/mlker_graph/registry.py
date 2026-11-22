"""Framework-independent signatures. Registry lookup never imports a constructor."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

Scalar = int | float | bool | str
ParameterType = Literal["int", "float", "bool", "string"]


@dataclass(frozen=True)
class Rule:
    type: ParameterType
    minimum: int | None = None

    def accepts(self, value: Scalar) -> bool:
        types = {"int": (int,), "float": (int, float), "bool": (bool,), "string": (str,)}
        return type(value) in types[self.type] and (self.minimum is None or value >= self.minimum)


@dataclass(frozen=True)
class Argument:
    rule: Rule
    default: Scalar | None = None  # None denotes a required argument.


@dataclass(frozen=True)
class Operation:
    id: str
    kind: Literal["operator", "module"]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    parameters: Mapping[str, Argument]
    ranks: tuple[int, ...] = ()  # Applies to every declared tensor port.

    def __post_init__(self):
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


positive = Rule("int", 1)
nonnegative = Rule("int", 0)
unary = ("x",)
binary = ("a", "b")
out = ("out",)
operations = (
    Operation("core.Conv2d", "module", unary, out, {
        "in_channels": Argument(positive), "out_channels": Argument(positive),
        "kernel_size": Argument(positive), "stride": Argument(positive, 1),
        "padding": Argument(nonnegative, 0), "bias": Argument(Rule("bool"), True),
    }, (3, 4)),
    Operation("core.ReLU", "operator", unary, out, {}),
    Operation("core.MaxPool2d", "operator", unary, out, {
        "kernel_size": Argument(positive), "stride": Argument(positive),
    }, (3, 4)),
    Operation("core.Flatten", "operator", unary, out, {
        "start_dim": Argument(Rule("int"), 1), "end_dim": Argument(Rule("int"), -1),
    }),
    Operation("core.Linear", "module", unary, out, {
        "in_features": Argument(positive), "out_features": Argument(positive),
        "bias": Argument(Rule("bool"), True),
    }),
    Operation("core.Add", "operator", binary, out, {}),
    Operation("core.Concat", "operator", binary, out, {"axis": Argument(Rule("int"))}),
)
REGISTRY = MappingProxyType({operation.id: operation for operation in operations})
