# Build a reusable residual block

1. Open `examples/models/v1/residual.yaml` in Model Builder. Its input drives both ReLU and the skip connection into Add.
2. Choose **Create block from nodes**, enter the ID `residual`, check **relu** and **add**, and choose **Create**. Extraction gives the block one input and one output. The shared input still feeds both internal consumers.
3. Choose **Add instance / residual**. Select the second instance and set its input source to the first instance's output. Select the model output and set its source to the second instance. Choose **Validate**.
4. Double-click either instance, or select it and choose **Open block definition**. Both open the same graph. Use the breadcrumbs to return. Add, rename, edit or remove definition ports to update callers and their wires. Removing a port disconnects its consumers.
5. Save, reopen and compile. For input `[[-2, 3]]`, these two blocks in series return `[[-2, 12]]`. Each instance owns its modules unless the YAML explicitly assigns a shared v2 instance ID.

In a definition, select a configurable operation and choose **Expose** beside a parameter. This creates a graph parameter and binds the operation to it. Select the canvas background to change its default. Select a block instance in its parent to override it. Choose **Use default** to remove an override. Extraction forwards any existing parent parameter references. Expose a nested block's parameter the same way to forward it through another boundary.

Each graph saves its node and boundary positions. Undo and redo operate on the whole document across graphs. Extraction and caller updates each form a single edit. Navigation has no history entry. Selection must include every use of a shared module or tied parameter it moves and must leave the parent dataflow acyclic.

# Register and run a Python module

Review the implementation and its dependencies before designating a registry as trusted. The registry is a local JSON file selected through the launch environment. Opening a model cannot select a registry, install a package or import Python code. The exported model imports reviewed code when someone runs it. Use a separate environment or isolated process with appropriate permissions for that execution.

From the repository root, with the graph dependencies installed, try the bundled example:

```bash
export MLKER_TRUSTED_REGISTRY="$PWD/examples/extensions/registry.json"
npm run desktop
```

Open `examples/models/v2/custom-scale.yaml`, or select **Scale** under **Python modules**. The Python implementation selector shows registered modules and their typed settings. Changing the implementation restores its defaults and disconnects ports absent from the replacement contract. Compile the model from the editor, or try the following commands in a PyTorch environment:

```bash
export MLKER_TRUSTED_REGISTRY="$PWD/examples/extensions/registry.json"
PYTHONPATH=python python -m mlker_graph.pytorch examples/models/v2/custom-scale.yaml /tmp/mlker-scale
PYTHONPATH="$PWD/examples/extensions:/tmp/mlker-scale" python -c 'from model import Model; import torch; print(Model()(torch.tensor([2.])))'
```

The example prints a tensor containing 6. The export includes imports, dependency requirements and registration identities. Supply the implementation through an installed package or the Python import path. MLKer does not copy it into the export.

Use `examples/extensions/registry.json` as the registration template. Each entry declares an ID, display name, description, version, dotted import module in `package`, top-level `symbol`, required tensor ports in positional order, typed parameters with defaults, and dependency distribution names or exact pins. Optional `example_inputs` maps input IDs to tensor specifications. A `module` constructs an nn.Module with parameter keywords and calls it with input tensors. A `function` receives positional tensors and parameter keywords directly. Return a single tensor for one output or an ordered tuple for multiple outputs.

The model records ID, package, symbol and version. All four must match the trusted entry. Restart the desktop after changing the registry to refresh the library. Static validation checks declared contracts. Runtime package availability and custom output shapes depend on the execution environment. No runtime preview is performed. Legacy Python nodes can be opened and assigned a registered implementation in Properties.
