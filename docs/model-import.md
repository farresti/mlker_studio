# PyTorch and timm import

## Desktop workflow

Install the optional dependencies in the interpreter used by the worker, for example with `python -m pip install './python[importers]'`. Select that interpreter with `MLKER_PYTHON` when launching the desktop.

1. In Model Builder choose **Import model**. For the bundled residual example, choose PyTorch and enter the absolute path to `examples/import/residual.py` followed by `:Residual`. An installed class such as `torch.nn:Linear` also works. The Linear constructor requires JSON options such as `{"in_features":4,"out_features":2}`.
2. Enter `[{"shape":[2,4],"dtype":"float32"}]` for the residual example. Review the source, grant trust, choose FX conversion, enable the eval comparison and preview. A source file must be self-contained or import installed dependencies. For relative package imports, install the package and use its dotted module name.
3. Review the quality, comparison result, native and opaque nodes, and module hierarchy. **Open in editor** starts an unsaved document. **Save** chooses the YAML destination. The report records the import-time graph and results. Later edits require fresh verification.
4. For timm, search for `resnet18`. Choosing a result fills in an example image size from registry metadata. Set `{"num_classes":3}` to change the classifier. Browsing reads the installed registry. Construction defaults to random initialization. **Load pretrained weights** explicitly permits timm to fetch a checkpoint. Other supported options are `in_chans`, `global_pool` and `drop_rate`. Image preprocessing belongs to the caller.

The importer runs Python constructors, tracing and example forwards in a dedicated worker process with the user's OS permissions. Process isolation keeps the editor and protocol independent of model code. Only grant trust to reviewed source and dependencies. Selecting an architecture document in the regular **Open** dialog reads data only.

## Quality and runtime behavior

| Quality | Meaning |
| --- | --- |
| Fully editable | All captured FX operations have native mappings and the compiler accepted the graph. Numerical evidence is reported separately. |
| Partially editable | The graph has both native and opaque nodes. FX metadata and the reconstruction identity describe unsupported calls. Replace them with compatible implementations before compiling the entire graph. |
| Inspectable only | The graph consists of opaque behavior. The editor offers inspection and YAML saving. The module tree records structural containment, which does not establish execution order. |

Source construction precedes FX extraction and exact-type/signature mapping. If ordinary tracing fails, the importer retries with module boundaries. If both fail, it retains a single opaque source model with its constructor, configuration and hierarchy. Composite mode selects this representation directly. Ports on a failed trace are provisional. Example execution can supply observed output shapes and dtypes. Observations are stored separately from declared tensor contracts.

The native subset includes Linear, compatible Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, Dropout, Identity, Flatten, several activations and two-tensor addition. Shared calls and shared parameter objects retain graph-local identities. In-place ReLU can convert when its input is a fresh tensor with a single consumer, including through single-consumer identities. Other in-place activations stay opaque. Unsupported configurations, functional signatures and custom operators retain FX dependencies. Hooks, unsupported aliasing, frozen parameters, non-float32 parameters and incompatible signatures can require a composite. Dynamic control flow, training-specific branches, Python side effects and data structures can exceed FX's representation. Exact behavior outside the captured evaluation path requires the original source.

## Architecture, weights and equivalence

YAML contains provenance, configuration, dependencies, warnings, stable FX names and weight references. When **weights directory** is empty, the saved result is architecture only, including for pretrained imports. Providing an existing absolute directory writes a new uniquely named snapshot during preview. A discarded preview may therefore leave that requested checkpoint on disk.

A fully native snapshot uses the generated model's state_dict keys. A partial or composite snapshot uses the source class's keys. Input checkpoints must be plain tensor state dictionaries compatible with the chosen constructor. The loader uses `weights_only=True` and strict key/shape matching. Whole pickled models and optimizer checkpoints are unsupported. Weight paths in provenance are local references. Move the snapshot separately when transferring a project.

The optional comparison compiles a fresh MLKer export, transfers the captured parameters and buffers, then compares source and generated outputs in CPU eval mode with inference-mode autograd disabled. It checks container structure, tensor shape, dtype and values on two seeded input batches (17 and 29, relative tolerance 1e-5, absolute tolerance 1e-6). A failed comparison falls back to source inspection and records the failure. The result covers those examples and that mode. Training, gradients and untested inputs require separate verification.

The regression suite also saves and reloads a YAML document with shared linear layers, recompiles it, loads the separate generated checkpoint, and compares a new input against the original model. The real timm test constructs ResNet-18 locally with three output classes and compares its generated model on two 64 × 64 image batches.

## Standalone Python

```python
from pathlib import Path
from mlker_import import import_model
from mlker_graph import save_model

options = {
    "source": "pytorch",
    "identity": str(Path("examples/import/residual.py").resolve()) + ":Residual",
    "trusted": True,
    "strategy": "fx",
    "inputs": [{"shape": [2, 4]}],
    "verify": True,
}
document = import_model(options)
print(document.data["model"]["provenance"]["verification"])
save_model(document, "imported-residual.yaml")
```

The standalone API runs in its caller's process. Call it from a dedicated interpreter when loading reviewed external packages. The desktop uses a fresh child for each import and catalog query. Rust awaits the child's response asynchronously and drops it after the request. Import has a ten-minute timeout. The existing four-MiB protocol limit applies to metadata. Tensor contents stay in the Python process or on disk.
