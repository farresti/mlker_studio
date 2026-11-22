"""Graph-local endpoint resolution, single-driver checks and tensor contracts."""

from graphlib import CycleError, TopologicalSorter

from .diagnostics import Diagnostic, Location
from .ir import Edge, Endpoint, Port, TensorSpec


def compare_tensors(left: TensorSpec, right: TensorSpec, location: Location, issues: list[Diagnostic]):
    for field in ("dtype", "layout"):
        a, b = getattr(left, field), getattr(right, field)
        if a is not None and b is not None and a != b:
            issues.append(Diagnostic(f"{field}_mismatch", "error", f"Conflicting {field}: {a} and {b}.", location))
    if left.shape is None or right.shape is None:
        return
    if len(left.shape) != len(right.shape):
        issues.append(Diagnostic("rank_mismatch", "error", f"Conflicting ranks: {left.shape} and {right.shape}.", location))
        return
    for axis, (a, b) in enumerate(zip(left.shape, right.shape)):
        if isinstance(a, int) and isinstance(b, int) and a != b:
            issues.append(Diagnostic("dimension_mismatch", "error", f"Axis {axis} has sizes {a} and {b}.", location))


def resolve_endpoint(raw, source, interfaces, location, issues):
    if "node" in raw:
        owner, port_id = raw["node"], raw["port"]
        direction = "outputs" if source else "inputs"
        if owner not in interfaces:
            issues.append(Diagnostic("unknown_node", "error", f"Unknown node {owner}.", location))
            return None
    else:
        key = "input" if "input" in raw else "output"
        owner, port_id, direction = None, raw[key], key + "s"
        if (key == "input") != source:
            issues.append(Diagnostic("wrong_direction", "error", f"Graph {key} used on wrong side of edge.", location))
            return None

    port = interfaces[owner][direction].get(port_id)
    if port is None:
        opposite = "inputs" if direction == "outputs" else "outputs"
        code = "wrong_direction" if port_id in interfaces[owner][opposite] else "unknown_port"
        issues.append(Diagnostic(code, "error", f"No {direction} port {port_id} on {owner or 'graph boundary'}.", location))
        return None
    return Endpoint(owner, direction, port)


def resolve_edges(raw_edges: list, interfaces: dict[str | None, dict[str, dict[str, Port]]],
                  location: Location, issues: list[Diagnostic]) -> tuple[Edge, ...]:
    edges = []
    drivers = set()
    used_inputs = set()
    topology = TopologicalSorter()
    for i, raw in enumerate(raw_edges):
        at = location.child("edges", i)
        src = resolve_endpoint(raw["from"], True, interfaces, at.child("from"), issues)
        dst = resolve_endpoint(raw["to"], False, interfaces, at.child("to"), issues)
        if src is None or dst is None:
            continue
        if src.node_id is None:
            used_inputs.add(src.port.id)
        if (dst.node_id, dst.port.id) in drivers:
            issues.append(Diagnostic("multiple_drivers", "error", f"Multiple edges drive {dst.port.id}.", at))
        drivers.add((dst.node_id, dst.port.id))
        if src.node_id is not None and dst.node_id is not None:
            topology.add(dst.node_id, src.node_id)
        compare_tensors(src.port.effective_tensor, dst.port.effective_tensor, at, issues)
        edges.append(Edge(src, dst, at))

    for port in interfaces[None]["inputs"].values():
        if port.id not in used_inputs:
            issues.append(Diagnostic("unbound_graph_input", "error", f"No edge reads input {port.id}.", port.location))
    for owner, interface in interfaces.items():
        ports = interface["outputs" if owner is None else "inputs"]
        for port in ports.values():
            if port.required and (owner, port.id) not in drivers:
                code = "unbound_graph_output" if owner is None else "unconnected_input"
                issues.append(Diagnostic(code, "error", f"No edge supplies {port.id}.", port.location))
    try:
        topology.prepare()
    except CycleError as error:
        chain = " → ".join(error.args[1])
        issues.append(Diagnostic("graph_cycle", "error", f"Dataflow cycle: {chain}.", location.child("edges")))
    return tuple(edges)
