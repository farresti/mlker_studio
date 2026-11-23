"""Render a lowered plan using fixed Python expressions and literal arguments."""

import json

from .plan import Call, ClassPlan

COMPILER_VERSION = "0.2.0"
TORCH_REQUIREMENT = "torch>=2.0,<3"


def keywords(params):
    return [f"{key}={value!r}" for key, value in sorted(params.items())]


def expression(call: Call):
    if call.function == "torch.cat":
        return f"torch.cat(({', '.join(call.inputs)}), dim={call.parameters['axis']})"
    if call.function == "F.adaptive_avg_pool2d":
        size = (call.parameters["height"], call.parameters["width"])
        return f"F.adaptive_avg_pool2d({call.inputs[0]}, output_size={size})"
    args = call.inputs + keywords(call.parameters)
    return f"{call.function}({', '.join(args)})"


def render_class(plan: ClassPlan):
    lines = [f"class {plan.name}(nn.Module):", "    def __init__(self) -> None:", "        super().__init__()"]
    for layer in plan.layers:
        args = ", ".join(keywords(layer.parameters))
        lines.append(f"        self.{layer.name} = {layer.constructor}({args})")
    for target, owner in plan.ties:
        lines.append(f"        self.{target} = self.{owner}")
    signature = ", ".join(["self", *(f"{name}: torch.Tensor" for name in plan.inputs.values())])
    result = "torch.Tensor"
    if len(plan.outputs) != 1:
        elements = ", ".join(["torch.Tensor"] * len(plan.outputs)) or "()"
        result = f"tuple[{elements}]"
    lines.extend(["", f"    def forward({signature}) -> {result}:"])
    for call in plan.calls:
        lhs = ", ".join(call.outputs) + " = " if call.outputs else ""
        lines.append(f"        {lhs}{expression(call)}")
    lines.append("        return " + (", ".join(plan.outputs) or "()"))
    return "\n".join(lines) + "\n"


def module_source(plans, imports=()):
    header = "import torch\nfrom torch import nn\n"
    if any(call.function.startswith("F.") for plan in plans for call in plan.calls):
        header += "from torch.nn import functional as F\n"
    if imports:
        names = ", ".join(imports)
        header += f"\nif __package__:\n    from .blocks import {names}\nelse:\n    from blocks import {names}\n"
    return header + "\n\n" + "\n\n".join(render_class(plan) for plan in plans)


def interface(port):
    return {"id": port.id, "shape": port.tensor.shape, "dtype": port.tensor.dtype, "layout": port.tensor.layout}


def render_project(model, lowering, entry, inferred, schema_version):
    blocks = [plan for plan in lowering.classes.values() if plan is not entry]
    imports = [p.name for p in blocks if any(layer.constructor == p.name for layer in entry.layers)]
    config = {
        "model_id": model.id, "revision": model.revision, "schema_version": schema_version,
        "compiler": "mlker-graph", "compiler_version": COMPILER_VERSION, "requires": TORCH_REQUIREMENT,
        "entry_graph": model.entry.id,
        "inputs": [{**interface(p), "argument": entry.inputs[p.id]} for p in model.inputs],
        "outputs": [{**interface(p), "inferred_shape": shape} for p, shape in zip(model.outputs, inferred)],
        "classes": [{"class": plan.name, "graph": graph_id, "parameters": dict(params)}
                    for (graph_id, params), plan in lowering.classes.items()],
    }
    invocation = ", ".join(entry.inputs.values())
    readme = f"""# Generated PyTorch model

Model `{model.id}`, revision `{model.revision}`. Compiled by MLKer graph compiler {COMPILER_VERSION}.

Use an environment with PyTorch 2.x. For a fresh Linux CPU environment, run
`python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt`.

From this directory, load the model and define an inference function. Pass it
tensors with the shapes and dtypes listed in `config.json`.

```python
import torch
from model import Model

model = Model().eval()

@torch.no_grad()
def predict({invocation}):
    return model({invocation})
```

The forward signature follows the public input order. Outputs follow the order in
`config.json`. A single output is a tensor and multiple outputs form a tuple.
Use `model.train()` for training and call `loss.backward()` on a scalar loss.
Parameters use PyTorch's initialization defaults. Use `load_state_dict` to load
compatible trained weights. Configuration records the exported specialization.
"""
    files = {
        "model.py": module_source([entry], imports), "__init__.py": "from .model import Model\n",
        "config.json": json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        "requirements.txt": TORCH_REQUIREMENT + "\n", "README.md": readme,
    }
    if blocks:
        files["blocks.py"] = module_source(blocks)
    return files
