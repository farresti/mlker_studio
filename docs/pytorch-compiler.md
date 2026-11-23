# PyTorch graph compiler

The Python API takes a parsed document and an output directory. It validates the document and returns the directory containing a standalone Python project. Source generation runs with the ordinary graph package dependencies. Execution requires PyTorch 2.x.

```python
from mlker_graph import load_model
from mlker_graph.pytorch import compile_model

project = compile_model(load_model("examples/models/v1/cnn.yaml"), "exports/cnn")
```

The equivalent CLI is `python -m mlker_graph.pytorch examples/models/v1/cnn.yaml exports/cnn`. Choose a new directory, an empty directory, or a previous MLKer export. Recompiling replaces generated files, including local edits to them, and removes an obsolete generated `blocks.py`. Other files such as checkpoints are preserved. The compiler identifies earlier exports from their `config.json`.

## Generated project

| File | Contents |
| --- | --- |
| `model.py` | Public `Model(nn.Module)` with constructors and a topologically ordered forward method |
| `blocks.py` | Classes for reachable reusable graphs, included when the entry uses blocks |
| `__init__.py` | Package entry point exposing `Model` |
| `config.json` | Revision, compiler version, PyTorch requirement, specialized graph parameters, ordered interfaces, argument names and inferred output shapes |
| `requirements.txt`, `README.md` | Runtime dependency and instructions |

Load `Model` from the export package or import it from `model` with the export directory on the Python path. Initial weights use PyTorch defaults. The typed `forward` signature follows the graph's public input order. A single output returns one tensor, multiple outputs return an ordered tuple, and a graph without outputs returns an empty tuple. Call `train()` or `eval()` to select batch normalization and dropout behavior.

## Operations

The compiler uses the v1 registry contracts for Conv2d, Linear, ReLU, MaxPool2d, Flatten, Add and Concat. Add uses PyTorch broadcasting. Concat takes two tensors and a signed axis. Conv2d and MaxPool2d support the existing scalar spatial parameters. Additional registry contracts use input `x` and output `out`.

| Registry ID | Kind | Parameters |
| --- | --- | --- |
| `core.Sigmoid`, `core.Tanh`, `core.GELU`, `core.SiLU` | operator | None |
| `core.BatchNorm2d` | module | Positive integer `num_features`. Affine parameters and running statistics use PyTorch defaults. |
| `core.Dropout` | module | Probability `p` in [0, 1], default 0.5 |
| `core.AdaptiveAvgPool2d` | operator | Positive integers `height`, `width` |
| `core.Reshape` | operator | V2 integer list `shape`. One `-1` may request an inferred dimension. |
| `core.Permute` | operator | V2 integer list `dims` containing each input axis once. Negative axes are supported. |

## V2 identity and parameters

V2 extends the v1 document with optional graph-local identities and integer-list parameters. To migrate an existing architecture in memory, change `document.data["schema_version"]` from 1 to 2 and save to the desired path. V1 examples compile directly. A v2 integer list can be a node literal or a graph parameter declared as `{id: axes, type: ints, default: [0, 2, 3, 1]}`. Use the existing `{$param: axes}` syntax at the call site.

- Nodes with the same `instance_id` within one graph call the same module or block object. They must resolve to the same constructor and configuration. Equal constructors without an explicit shared ID create separate instances.
- `parameter_ids: {weight: projection_weight}` ties that slot across participating modules in the graph instance. Distinct layers can keep independent biases. Supported slots are active weights and biases of Linear, Conv2d and BatchNorm2d. The compiler checks slot existence and concrete shape equality.
- A reusable graph can have multiple specialized classes when its call sites specify different parameters. Repeated calls with the same specialization reuse the class. Each graph instance owns its local sharing identities.

The [two-input example](../examples/models/v2/shared-branches.yaml) uses one encoder twice and ties its weight to a third projection.

## Shape checks and errors

Shape propagation follows scheduled edges through each reachable block specialization. Known dimensions constrain convolution channels, linear features, batch normalization, pooling, broadcasting, concatenation and layout-changing operations. A symbolic or unknown dimension stays unresolved when its size depends on runtime data. The output manifest records both the declared contract and the result of inference.

Structural and parameter validation raises `ModelError`. A valid graph that cannot be lowered raises its subclass `CompileError`, with a diagnostic code and graph/node location. Catch `CompileError` first to distinguish the two. The CLI reports the stage in JSON on stderr and exits 1 for either category. File errors exit 2. Python and opaque implementations require a supported operation or an editable block before compilation. Optional block inputs must be connected for export.

Generated code uses standard PyTorch runtime errors for actual tensor shapes, dtypes, devices and training-mode restrictions. Inputs and parameters must have compatible dtypes and devices. Static checks use declared contracts and leave unresolved symbolic relationships to runtime.
