"""Index declarations, validate graph contracts, and publish IR only on success."""

from copy import deepcopy
from dataclasses import replace
from graphlib import CycleError, TopologicalSorter

from . import ir
from .contracts import GraphDefinition, check_operation, check_ports, resolve_parameters
from .diagnostics import Diagnostic, Location, ModelError
from .document import ModelDocument
from .registry import REGISTRY, Argument, Rule
from .schema import schema_diagnostics
from .wiring import resolve_edges


def declarations(records, location, issues):
    seen = set()
    for index, record in enumerate(records):
        key = record["id"]
        at = location.child(index)
        if key in seen:
            issues.append(Diagnostic("duplicate_id", "error", f"Duplicate ID {key}.", at.child("id")))
        else:
            seen.add(key)
            yield key, record, at


def read_ports(raw, location, issues):
    ports = {}
    for direction in ("inputs", "outputs"):
        group = {}
        for key, item, at in declarations(raw[direction], location.child(direction), issues):
            contract = item["tensor"]
            shape = contract.get("shape")
            spec = ir.TensorSpec(contract.get("dtype"), tuple(shape) if shape is not None else None, contract.get("layout"))
            required = item.get("required", True)
            if direction == "outputs" and not required:
                issues.append(Diagnostic("port_contract", "error", "Output ports must be required.", at))
            group[key] = ir.Port(key, spec, required, at)
        ports[direction] = group
    return ports


def read_definitions(raw_graphs, root, issues):
    definitions = {}
    for key, raw, at in declarations(raw_graphs, root.child("graphs"), issues):
        at = replace(at, graph=key)
        params = {}
        for name, value, loc in declarations(raw.get("parameters", []), at.child("parameters"), issues):
            param = ir.Parameter(name, value["type"], value["default"], loc)
            if not Rule(param.type).accepts(param.default):
                issues.append(Diagnostic("parameter_value", "error", f"Default for {name} must be {param.type}.", loc))
            params[name] = param
        definitions[key] = GraphDefinition(raw, at, read_ports(raw, at, issues), params)
    return definitions


def graph_order(definitions, blocks, issues):
    topology = TopologicalSorter()
    for key, definition in definitions.items():
        topology.add(key)
        for i, node in enumerate(definition.raw["nodes"]):
            if node["kind"] != "block":
                continue
            block = blocks.get(node["block"])
            if block is None:
                loc = replace(definition.location.child("nodes", i, "block"), node=node["id"])
                issues.append(Diagnostic("unknown_block", "error", f"Unknown block {node['block']}.", loc))
            elif block.graph_id in definitions:
                topology.add(key, block.graph_id)
    try:
        return tuple(topology.static_order())
    except CycleError as error:
        cycle = error.args[1]
        issues.append(Diagnostic("recursive_block", "error", "Recursive graph references: " + " → ".join(cycle),
                                 definitions[cycle[0]].location))
        return ()


def compile_graph(definition, blocks, definitions, issues):
    raw, at = definition.raw, definition.location
    interfaces = {None: definition.ports}
    nodes = {}
    for key, item, loc in declarations(raw["nodes"], at.child("nodes"), issues):
        loc = replace(loc, node=key)
        ports = interfaces[key] = read_ports(item, loc, issues)
        signature = None
        inherited = {}
        kind = item["kind"]
        if kind in ("module", "operator"):
            target = REGISTRY.get(item["registry"])
            if target is None:
                issues.append(Diagnostic("unknown_registry", "error", f"Unknown registry ID {item['registry']}.", loc))
                continue
            check_operation(target, kind, ports, loc, issues)
            signature = target.parameters
        elif kind == "block":
            target = blocks[item["block"]]
            child = definitions[target.graph_id]
            check_ports(ports, child.ports, loc, issues)
            signature = {k: Argument(Rule(p.type), p.default) for k, p in child.parameters.items()}
            inherited = child.constraints
        elif kind == "python":
            impl = item["implementation"]
            target = ir.PythonRef(impl["package"], impl["symbol"], impl.get("version"))
        else:
            target = ir.OpaqueRef(item["source_op"])
            issues.append(Diagnostic("opaque_node", "warning", "Only the declared opaque ports can be checked.", loc))
        params = resolve_parameters(item.get("params", {}), signature, definition, inherited, loc.child("params"), issues)
        nodes[key] = ir.Node(key, kind, target, ports["inputs"], ports["outputs"], params, loc)
    edges = resolve_edges(raw["edges"], interfaces, at, issues)
    return ir.Graph(raw["id"], definition.ports["inputs"], definition.ports["outputs"],
                    definition.parameters, nodes, edges, at)


def public_ports(model, entry, direction, location, issues):
    seen = set()
    ports = entry.ports[direction]
    for i, key in enumerate(model[direction]):
        at = location.child(direction, i)
        if key in seen:
            issues.append(Diagnostic("duplicate_id", "error", f"Duplicate public port {key}.", at))
        if key not in ports:
            issues.append(Diagnostic("unknown_port", "error", f"No entry {direction} port {key}.", at))
        seen.add(key)
    for key in sorted(ports.keys() - seen):
        issues.append(Diagnostic("missing_model_binding", "error", f"Expose entry port {key}.", location.child(direction)))
    return tuple(ports[k] for k in model[direction] if k in ports)


def analyze(document: ModelDocument) -> tuple[list[Diagnostic], ir.Model | None]:
    issues = schema_diagnostics(document)
    if issues:
        return issues, None
    data = document.data
    model = data["model"]
    root = Location(document.source)
    definitions = read_definitions(data["graphs"], root, issues)
    blocks = {}
    for key, item, loc in declarations(data.get("blocks", []), root.child("blocks"), issues):
        target = item["graph"]
        if target not in definitions:
            issues.append(Diagnostic("unknown_graph", "error", f"Unknown block graph {target}.", loc.child("graph")))
        blocks[key] = ir.BlockRef(key, target, loc)
    entry = definitions.get(model["entry"])
    loc = root.child("model")
    if entry is None:
        issues.append(Diagnostic("unknown_graph", "error", f"Unknown entry graph {model['entry']}.", loc.child("entry")))
    else:
        inputs = public_ports(model, entry, "inputs", loc, issues)
        outputs = public_ports(model, entry, "outputs", loc, issues)

    order = graph_order(definitions, blocks, issues)
    if issues:
        return issues, None
    graphs = {key: compile_graph(definitions[key], blocks, definitions, issues) for key in order}
    if any(problem.severity == "error" for problem in issues):
        return issues, None
    result = ir.Model(model["id"], model["revision"], model["name"], model.get("description", ""),
                      graphs[model["entry"]], inputs, outputs, graphs, blocks, deepcopy(model.get("metadata", {})),
                      deepcopy(model.get("provenance", {})), deepcopy(data.get("extensions", {})), loc)
    return issues, result


def validate_model(document: ModelDocument) -> list[Diagnostic]:
    """Return errors and warnings in deterministic validation order."""
    return analyze(document)[0]


def build_ir(document: ModelDocument) -> ir.Model:
    """Build a fresh canonical snapshot, or raise ModelError with diagnostics."""
    issues, model = analyze(document)
    if model is None:
        raise ModelError(issues)
    return model
