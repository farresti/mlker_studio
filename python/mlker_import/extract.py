"""Convert FX dependencies into a version-two MLKer graph."""
import inspect
from collections import defaultdict
from copy import deepcopy

import torch
from torch import fx, nn

from mlker_graph.registry import REGISTRY
from .mapping import module_params, native


def port(name):
    return {"id": name, "tensor": {}}


def empty_graph():
    return {"id": "main", "inputs": [], "outputs": [], "nodes": [], "edges": []}


def qualified(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def composite(model, reason, count):
    graph = empty_graph()
    # Port counts on failed traces are provisional, with no invented internal wiring.
    params = inspect.signature(model.forward).parameters
    names = [k for k, p in params.items() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    inputs = names[:count] if count else names
    if count > len(inputs):
        inputs.extend(f"arg{i}" for i in range(len(inputs), count))
    graph["inputs"] = [port(i) for i in inputs]
    graph["outputs"] = [port("result")]
    graph["nodes"].append({"id": "source", "kind": "opaque", "source_op": qualified(model),
                           "inputs": graph["inputs"], "outputs": graph["outputs"], "name": type(model).__name__})
    graph["edges"] = [{"from": {"input": i}, "to": {"node": "source", "port": i}} for i in inputs]
    graph["edges"].append({"from": {"node": "source", "port": "result"}, "to": {"output": "result"}})
    return graph, {}, [reason], "module_tree"


class BoundaryTracer(fx.Tracer):
    def is_leaf_module(self, module, path):
        return bool(path) and (not isinstance(module, (nn.Sequential, nn.ModuleList)) or super().is_leaf_module(module, path))


def extract(model, strategy, input_count):
    if strategy == "composite":
        return composite(model, "Composite import requested. The public interface is provisional.", input_count)
    if any(p.kind == p.KEYWORD_ONLY for p in inspect.signature(model.forward).parameters.values()):
        return composite(model, "Keyword-only arguments require the original forward signature.", input_count)
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or "forward" in vars(m)
           for m in model.modules()):
        return composite(model, "Hooks or instance-level forward overrides require source execution.", input_count)
    # Standard stateful operations in MLKer construct trainable float32 layers.
    if any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        return composite(model, "Frozen or non-float32 parameters require the original constructor.", input_count)
    if any(b.is_floating_point() and b.dtype != torch.float32 for b in model.buffers()):
        return composite(model, "Non-float32 buffers require the original constructor.", input_count)
    buffers = list(model.named_buffers(remove_duplicate=False))
    owners = defaultdict(set)
    for path, tensor in buffers:
        parent = path.rpartition(".")[0]
        owners[id(tensor)].add(id(model.get_submodule(parent)))
    if any(len(group) > 1 for group in owners.values()):
        return composite(model, "Buffers shared across modules require the original constructor.", input_count)
    storage = defaultdict(set)
    for p in model.parameters():
        if p.numel():
            storage[p.untyped_storage().data_ptr()].add(id(p))
    if any(len(group) > 1 for group in storage.values()):
        return composite(model, "Distinct parameters alias the same storage.", input_count)

    wrapped = bool(module_params(model))
    def trace_root():
        snapshot = deepcopy(model)
        return nn.Sequential(snapshot) if wrapped else snapshot

    warnings = []
    try:
        traced = fx.symbolic_trace(trace_root())
        strategy = "fx"
    except Exception as exc:
        warnings.append(f"FX tracing failed: {type(exc).__name__}: {exc}")
        try:
            root = trace_root()
            traced = fx.GraphModule(root, BoundaryTracer().trace(root))
            strategy = "fx_module_boundaries"
        except Exception as retry:
            return composite(model, warnings[0] + f"; module-boundary trace failed: {retry}", input_count)

    graph = empty_graph()
    sources, paths, shared, slots = {}, {}, {}, {}
    def edge(src, dst):
        graph["edges"].append({"from": sources[src], "to": dst})

    for value in traced.graph.nodes:
        if value.op == "placeholder":
            if value.args or str(value.target).startswith("*") or not value.users:
                return composite(model, "Optional, variadic or unused inputs require the original signature.", input_count)
            graph["inputs"].append(port(value.name))
            sources[value] = {"input": value.name}
        elif value.op == "output":
            result = value.args[0]
            if isinstance(result, fx.Node):
                result = (result,)
            elif type(result) is not tuple or len(result) < 2:
                return composite(model, "Output container requires the original forward contract.", input_count)
            if not all(isinstance(v, fx.Node) for v in result):
                return composite(model, "Non-tensor or nested output contract.", input_count)
            for i, v in enumerate(result):
                name = f"result{i}"
                graph["outputs"].append(port(name))
                edge(v, {"output": name})
        else:
            name = str(value.target) if isinstance(value.target, str) else getattr(value.target, "__qualname__", str(value.target))
            record = {"id": value.name, "name": name, "outputs": [port("out")],
                      "extensions": {"mlker.fx": {"op": value.op, "target": name}}}
            match = native(value, traced)
            if match:
                operation, params, args = match
                op = REGISTRY["core." + operation]
                record.update(kind=op.kind, registry=op.id, params=params, inputs=[port(k) for k in op.inputs])
                if op.kind == "module":
                    m = traced.get_submodule(value.target)
                    record["instance_id"] = shared.setdefault(id(m), value.name)
                    paths[value.name] = model if wrapped else model.get_submodule(value.target)
                    record["parameter_ids"] = {k: slots.setdefault(id(p), f"param_{len(slots)}")
                                               for k, p in m.named_parameters(recurse=False)}
            else:
                args = value.all_input_nodes
                record.update(kind="opaque", source_op=name, inputs=[port(f"arg{i}") for i in range(len(args))])
                record["extensions"]["mlker.fx"]["expression"] = value.format_node()
                warnings.append(f"{value.name}: unsupported {value.op} {name}.")
            graph["nodes"].append(record)
            for src, dest in zip(args, record["inputs"]):
                edge(src, {"node": value.name, "port": dest["id"]})
            sources[value] = {"node": value.name, "port": "out"}
    if not any(n["kind"] == "opaque" for n in graph["nodes"]):
        captured = {id(t) for m in paths.values() for t in (*m.parameters(), *m.buffers())}
        if any(id(t) not in captured for t in (*model.parameters(), *model.buffers())):
            return composite(model, "Registered parameters or buffers are absent from the captured computation.", input_count)
    return graph, paths, warnings, strategy
