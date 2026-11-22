"""Resolve registry and block contracts in a graph's lexical parameter scope."""

from dataclasses import dataclass, field, replace
from typing import Mapping

from .diagnostics import Diagnostic, Location
from .ir import Binding, Parameter, Port
from .registry import Argument, Operation, Rule
from .wiring import compare_tensors


@dataclass
class GraphDefinition:
    raw: dict
    location: Location
    ports: dict[str, dict[str, Port]]
    parameters: dict[str, Parameter]
    # A nested instance must satisfy every consumer of its graph parameter.
    constraints: dict[str, set[Rule]] = field(default_factory=dict)


def check_ports(actual, expected, location, issues):
    for direction in ("inputs", "outputs"):
        want = expected[direction]
        have = actual[direction]
        for key in sorted(want.keys() ^ have.keys()):
            message = f"Missing contract port {key}." if key in want else f"Unexpected contract port {key}."
            issues.append(Diagnostic("port_contract", "error", message, location.child(direction)))
        for key, port in have.items():
            if key not in want:
                continue
            contract = want[key]
            if port.required != (contract.required if contract else True):
                issues.append(Diagnostic("port_contract", "error", f"Required flag differs for {key}.", port.location))
            if contract:
                compare_tensors(port.tensor, contract.tensor, port.location, issues)
                have[key] = replace(port, block_contract=contract.tensor)


def check_operation(target: Operation, kind, ports, location, issues):
    if kind != target.kind:
        issues.append(Diagnostic("registry_kind", "error", f"{target.id} requires kind {target.kind}.", location))
    check_ports(ports, {d: dict.fromkeys(getattr(target, d)) for d in ("inputs", "outputs")}, location, issues)
    for direction in ports.values():
        for port in direction.values():
            shape = port.tensor.shape
            if target.ranks and shape is not None and len(shape) not in target.ranks:
                issues.append(Diagnostic("rank_mismatch", "error", f"{target.id} expects rank in {target.ranks}.", port.location))


def resolve_parameters(raw: dict, signature: Mapping[str, Argument] | None, graph: GraphDefinition,
                       inherited: dict[str, set[Rule]], location: Location, issues: list[Diagnostic]):
    bindings = {}
    params = dict(raw)
    if signature is not None:
        for key, arg in signature.items():
            if key not in params:
                if arg.default is None:
                    issues.append(Diagnostic("missing_parameter", "error", f"Missing parameter {key}.", location))
                else:
                    params[key] = arg.default

    for key, value in params.items():
        at = location.child(key)
        if signature is not None and key not in signature:
            issues.append(Diagnostic("unknown_parameter", "error", f"Unsupported parameter {key}.", at))
            continue
        reference = None
        if isinstance(value, dict):
            name = value["$param"]
            reference = graph.parameters.get(name)
            if reference is None:
                issues.append(Diagnostic("unknown_parameter_ref", "error", f"No graph parameter {name}.", at))
                continue
            value = reference.default

        rules = set(inherited.get(key, ()))
        if signature is not None:
            rules.add(signature[key].rule)
        for rule in sorted(rules, key=lambda r: (r.type, r.minimum is not None, r.minimum or 0)):
            if reference:
                graph.constraints.setdefault(reference.id, set()).add(rule)
            if (reference and reference.type != rule.type) or not rule.accepts(value):
                constraint = f" >= {rule.minimum}" if rule.minimum is not None else ""
                issues.append(Diagnostic("parameter_value", "error", f"{key} requires {rule.type}{constraint}.", at))
        bindings[key] = Binding(value, reference)
    return bindings
