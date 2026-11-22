"""Canonical declarations and resolved handles for a validated model revision."""

from dataclasses import dataclass
from typing import Any, Literal

from .diagnostics import Location
from .registry import Operation, ParameterType, Scalar


@dataclass(frozen=True)
class TensorSpec:
    dtype: str | None
    shape: tuple[int | str | None, ...] | None
    layout: str | None


@dataclass(frozen=True)
class Port:
    id: str
    tensor: TensorSpec
    required: bool
    location: Location
    block_contract: TensorSpec | None = None

    @property
    def effective_tensor(self) -> TensorSpec:
        """Combine compatible instance and definition declarations for wiring."""
        other = self.block_contract
        if other is None:
            return self.tensor
        shape = self.tensor.shape
        if shape is None:
            shape = other.shape
        elif other.shape is not None and len(shape) == len(other.shape):
            shape = tuple(b if a is None or isinstance(b, int) else a for a, b in zip(shape, other.shape))
        return TensorSpec(self.tensor.dtype or other.dtype, shape, self.tensor.layout or other.layout)


@dataclass(frozen=True)
class Parameter:
    id: str
    type: ParameterType
    default: Scalar
    location: Location


@dataclass(frozen=True)
class Binding:
    value: Scalar  # Effective value with the containing graph's defaults.
    reference: Parameter | None = None


@dataclass(frozen=True)
class BlockRef:
    id: str
    graph_id: str
    location: Location


@dataclass(frozen=True)
class PythonRef:
    package: str
    symbol: str
    version: str | None


@dataclass(frozen=True)
class OpaqueRef:
    source_op: str


@dataclass(frozen=True)
class Node:
    id: str
    kind: Literal["operator", "module", "block", "python", "opaque"]
    target: Operation | BlockRef | PythonRef | OpaqueRef
    inputs: dict[str, Port]
    outputs: dict[str, Port]
    parameters: dict[str, Binding]
    location: Location


@dataclass(frozen=True)
class Endpoint:
    node_id: str | None  # None denotes the graph boundary.
    direction: Literal["inputs", "outputs"]
    port: Port


@dataclass(frozen=True)
class Edge:
    source: Endpoint
    destination: Endpoint
    location: Location


@dataclass(frozen=True)
class Graph:
    id: str
    inputs: dict[str, Port]
    outputs: dict[str, Port]
    parameters: dict[str, Parameter]
    nodes: dict[str, Node]
    edges: tuple[Edge, ...]
    location: Location


@dataclass(frozen=True)
class Model:
    id: str
    revision: str
    name: str
    description: str
    entry: Graph
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    graphs: dict[str, Graph]
    blocks: dict[str, BlockRef]
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    extensions: dict[str, Any]
    location: Location
