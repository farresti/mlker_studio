"""Specialize block calls, resolve tensor values, and schedule a validated DAG."""

from graphlib import TopologicalSorter

from .. import ir
from ..extensions import Extension
from ..registry import Operation
from .plan import Call, ClassPlan, Layer, Names, fail
from .shapes import ShapeError, infer, merge

MODULES = {
    "core.Identity": "nn.Identity",
    "core.Linear": "nn.Linear", "core.Conv2d": "nn.Conv2d",
    "core.BatchNorm2d": "nn.BatchNorm2d", "core.Dropout": "nn.Dropout",
}
FUNCTIONS = {
    "core.ReLU": "torch.relu", "core.Sigmoid": "torch.sigmoid", "core.Tanh": "torch.tanh",
    "core.GELU": "F.gelu", "core.SiLU": "F.silu", "core.Flatten": "torch.flatten",
    "core.MaxPool2d": "F.max_pool2d", "core.AdaptiveAvgPool2d": "F.adaptive_avg_pool2d",
    "core.Permute": "torch.permute", "core.Reshape": "torch.reshape", "core.Add": "torch.add", "core.Concat": "torch.cat",
}


def checked_shape(actual, port):
    try:
        return merge(actual, port.effective_tensor.shape)
    except ShapeError as error:
        fail("compile_shape", f"Port {port.id}: {error}", port.location)


def slot_shapes(operation, params):
    if operation == "core.Linear":
        weight = (params["out_features"], params["in_features"])
    elif operation == "core.Conv2d":
        weight = (params["out_channels"], params["in_channels"], params["kernel_size"], params["kernel_size"])
    elif operation == "core.BatchNorm2d":
        weight = (params["num_features"],)
    else:
        return {}
    slots = {"weight": weight}
    if params.get("bias", True):
        slots["bias"] = (weight[0],)
    return slots


class Instances:
    """Allocate each module once and link compatible parameter slots after creation."""

    def __init__(self, plan, names):
        self.plan = plan
        self.names = names
        self.shared = {}
        self.slot_ids = {}
        self.tied = {}

    def layer(self, node, constructor, params):
        layer = self.shared.get(node.instance_id)
        if layer is None:
            layer = Layer(self.names.allocate("layer_", node.instance_id or node.id), constructor, params)
            self.plan.layers.append(layer)
            if node.instance_id is not None:
                self.shared[node.instance_id] = layer
        elif layer.constructor != constructor or layer.parameters != params:
            fail("compile_instance", f"Instance {node.instance_id} has conflicting constructors or arguments.", node.location)
        if node.parameter_ids:
            slots = slot_shapes(node.target.id, params)
            for slot, identity in node.parameter_ids.items():
                if slot not in slots:
                    fail("compile_parameter_slot", f"{node.target.id} has no active parameter {slot}.", node.location)
                path = f"{layer.name}.{slot}"
                previous = self.slot_ids.setdefault(path, identity)
                if previous != identity:
                    fail("compile_parameter_id", f"Slot {slot} has conflicting parameter IDs.", node.location)
                first, shape = self.tied.setdefault(identity, (path, slots[slot]))
                if shape != slots[slot]:
                    fail("compile_parameter_shape", f"Tied parameter {identity} has shapes {shape} and {slots[slot]}.",
                         node.location)
                tie = (path, first)
                if first != path and tie not in self.plan.ties:
                    self.plan.ties.append(tie)
        self.plan.module_paths[node.id] = layer.name
        return "self." + layer.name


class Lowering:
    def __init__(self, model: ir.Model):
        self.model = model
        self.names = Names()
        self.classes = {}
        self.cache = {}
        self.extensions = {}

    def graph(self, graph, config, input_shapes, entry=False):
        # Names and code depend on graph/config. Shape checks also depend on the call site's inputs.
        specialization = (graph.id, tuple(sorted(config.items())))
        cache_key = (specialization, tuple(sorted(input_shapes.items())))
        if cache_key in self.cache:
            return self.cache[cache_key]
        old = self.classes.get(specialization)
        name = old.name if old else ("Model" if entry else self.names.allocate("Block_", graph.id))
        names = Names()
        inputs = self.model.inputs if entry else graph.inputs.values()
        outputs = self.model.outputs if entry else graph.outputs.values()
        plan = ClassPlan(name, {p.id: names.allocate("input_", p.id) for p in inputs})
        instances = Instances(plan, names)
        values = {(None, k): (var, checked_shape(input_shapes[k], graph.inputs[k])) for k, var in plan.inputs.items()}
        drivers = {}
        order = TopologicalSorter()
        for node_id in graph.nodes:
            order.add(node_id)
        for edge in graph.edges:
            src, dst = edge.source, edge.destination
            drivers[dst.node_id, dst.port.id] = (src.node_id, src.port.id)
            if src.node_id is not None and dst.node_id is not None:
                order.add(dst.node_id, src.node_id)

        for node_id in order.static_order():
            node = graph.nodes[node_id]
            target = node.target
            params = {k: config[b.reference.id] if b.reference else b.value for k, b in node.parameters.items()}
            if isinstance(target, ir.BlockRef):
                child = self.model.graphs[target.graph_id]
                input_ids, output_ids = tuple(child.inputs), tuple(child.outputs)
            elif isinstance(target, Extension):
                input_ids, output_ids = ([p["id"] for p in target.record[k]] for k in ("inputs", "outputs"))
            elif isinstance(target, Operation) and (target.id in MODULES or target.id in FUNCTIONS):
                input_ids, output_ids = target.inputs, target.outputs
            else:
                fail("compile_unsupported", f"Cannot lower {node.kind} node {node.id}. Replace it with a registered operation or block.", node.location)
            args, shapes = [], []
            for port_id in input_ids:
                source = drivers.get((node_id, port_id))
                if source is None:
                    fail("compile_optional_input", f"Connect block input {port_id} before exporting.", node.location)
                variable, shape = values[source]
                args.append(variable)
                shapes.append(checked_shape(shape, node.inputs[port_id]))

            if isinstance(target, ir.BlockRef):
                child_plan, result_shapes = self.graph(child, params, dict(zip(input_ids, shapes)))
                function = instances.layer(node, child_plan.name, {})
                keywords = {}
            elif isinstance(target, Extension):
                registration = target.record["id"]
                if registration not in self.extensions:
                    self.extensions[registration] = (self.names.allocate("Extension_", registration), target)
                alias, _ = self.extensions[registration]
                result_shapes = [node.outputs[k].effective_tensor.shape for k in output_ids]
                if target.record["kind"] == "module":
                    function, keywords = instances.layer(node, alias, params), {}
                else:
                    function, keywords = alias, params
            else:
                try:
                    result_shapes = [infer(target, shapes, params)]
                except ShapeError as error:
                    fail("compile_shape", f"{target.id}: {error}", node.location)
                if target.id in MODULES:
                    function = instances.layer(node, MODULES[target.id], params)
                    keywords = {}
                else:
                    function, keywords = FUNCTIONS[target.id], params
            variables = []
            for port_id, shape in zip(output_ids, result_shapes):
                label = node_id if len(output_ids) == 1 else f"{node_id}_{port_id}"
                variable = names.allocate("v_", label)
                values[node_id, port_id] = variable, checked_shape(shape, node.outputs[port_id])
                variables.append(variable)
            plan.calls.append(Call(variables, function, args, keywords))

        shapes = []
        for port in outputs:
            variable, shape = values[drivers[None, port.id]]
            plan.outputs.append(variable)
            checked_shape(shape, port)
            shapes.append(shape)
        self.classes[specialization] = plan
        self.cache[cache_key] = plan, shapes
        return plan, shapes
