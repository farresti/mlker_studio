
# MLKer Studio — Project Instructions

## 1. Project objective

MLKer Studio is a modular desktop application for computer
vision, neural network inspection, model development and
machine learning experiment tracking.

It combines functionality inspired by:

- Netron
- TensorBoard
- MLflow
- Visual node-based programming environments

## 2. Technology stack

Desktop:
- Tauri 2
- Rust

Frontend:
- React
- TypeScript
- React Flow / XYFlow
- uPlot

Backend:
- Rust application services
- Python worker process
- SQLite persistent storage

Machine learning:
- PyTorch
- PyTorch Lightning
- timm
- ONNX
- ONNX Runtime
- NVIDIA CUDA

Serialization:
- YAML for visual model graphs
- JSON for process communication
- TensorBoard event format for training metrics and media

## 3. Main application modules

A. ONNX Inspector
- ONNX graph visualization
- Operator documentation
- Intermediate tensor inspection
- Dynamic inference
- Feature map visualization

B. Experiment Tracking
- TensorBoard event ingestion
- Experiment management
- Training curves
- Media visualization
- Hyperparameter comparison
- Memory-efficient metric storage

C. Visual Model Builder
- Node-based PyTorch model editor
- PyTorch and timm block registries
- Custom reusable blocks
- Hierarchical subgraphs
- YAML model serialization
- Python code generation

## 4. Architectural principles

- Keep frontend, Rust backend and Python backend independent.
- Use well-defined interfaces.
- Keep long-running operations asynchronous.
- Do not block the UI thread.
- Avoid unnecessary data copying.
- Keep large tensors and media out of JSON messages.
- Store large artifacts on disk.
- Use SQLite for structured metadata and metrics.
- Keep model graph schemas versioned.
- Preserve compatibility with previous graph schema versions.

## 5. Model graph architecture

The YAML graph format is the canonical representation
of visually designed models.

The compiler must validate:
- Graph connectivity
- Input/output ports
- Tensor shapes
- Data types
- Parameter compatibility
- Cycles
- Nested graphs

Generate normal Python PyTorch modules.

Use explicit registries for permitted module constructors.

Never execute arbitrary Python expressions from YAML.

Use PyTorch FX as an interoperability mechanism where
appropriate, not as the canonical visual graph format.

## 6. ONNX inspection

Support:
- ONNX graph parsing
- Operator domains and opsets
- Tensor metadata
- Intermediate graph outputs
- Dynamic input testing
- CPU and CUDA inference
- Numerical inspection

Provide a debugging mode that can reduce or disable
graph optimizations when intermediate values need
to be inspected.

## 7. Experiment tracking

Implement streaming TensorBoard event ingestion.

Do not load entire event files or complete experiment
histories into RAM.

Use:
- Persistent scalar storage
- Multi-resolution metric aggregation
- Lazy media loading
- Bounded image caches
- Disk-based artifacts

Keep TensorBoard compatibility separate from the
custom experiment database.

## 8. Development workflow

Before major implementation:

1. Inspect existing code.
2. Explain the architecture.
3. Present an implementation plan.
4. Identify risks.
5. Implement one milestone.
6. Run tests.
7. Report modified files and results.

Never rewrite unrelated working modules.

Never invent test results.

Explain Rust concepts for a developer experienced
in Python but new to Rust.


## Model Editor Architecture Decisions

### Product direction

MLKer Studio's model editor is visual-first, with Python escape hatches. It must support both creating new models and importing existing models.

### Canonical representation

- Use a versioned, framework-independent YAML model document as the persistent architecture format.
- Use a separate typed canonical graph IR for compiler operations.
- Preserve unknown fields, extensions, and import provenance where possible.
- Keep model architecture, trained weights, and experiment/run metadata separate.
- Use stable IDs for graphs, nodes, ports, blocks, and model revisions.

### Graph capabilities

- Support operator, module, reusable block, Python, and opaque node kinds.
- Support nested graphs and reusable blocks with explicit input/output contracts.
- Support symbolic tensor dimensions, dtype, layout, and partial shape information.
- Distinguish declared, inferred, and runtime-observed tensor metadata.
- Validate connections, graph boundaries, parameter references, cycles, and tensor compatibility.
- Provide shape inference and clear diagnostics.

### Python extensibility

- Prefer registered custom Python modules with declared input/output contracts and parameter schemas.
- Support external Python package references with explicit dependency metadata.
- Do not execute arbitrary code embedded in YAML.
- Treat imported Python modules and model files as untrusted; provide an isolated execution option.

### Compiler

- Implement the compiler as a standalone Python package.
- Pipeline: YAML loading, schema validation, reference resolution, graph validation, IR construction, shape inference, optimization, backend lowering, code generation, and runtime verification.
- Generate readable standalone PyTorch nn.Module source code.
- Preserve module sharing, parameter tying, and nested block structure.
- Keep backend-specific transformations out of the canonical YAML format.

### Import

- Support PyTorch FX tracing, module-tree inspection, and opaque module fallback.
- Support timm model discovery and model import.
- Support ONNX graph import and mapping of known operators.
- Never claim that arbitrary PyTorch models can always be fully converted into editable graphs.
- Preserve unsupported nodes as opaque nodes where possible.
- Record source format, source identity, import strategy, framework version, and conversion warnings.
- Show import quality: fully editable, partially editable, or inspectable/read-only.

### Export

- Support MLKer Studio YAML, standalone PyTorch source, PyTorch checkpoints, ONNX, and packaged model exports.
- Keep architecture and weights separate.
- Record model revision and compiler/dependency versions with generated outputs.
- Distinguish native round-trip guarantees from best-effort external model conversion.

### User interface

- Use React Flow / XYFlow for the graph canvas.
- Support node search, drag-and-drop, port connections, properties, tensor-shape inspection, nested graph navigation, reusable blocks, and validation diagnostics.
- Double-clicking a block opens its subgraph; breadcrumbs navigate back to parent graphs.
- Python nodes remain ordinary graph nodes with a dedicated implementation editor.
- Provide example-input runtime previews and intermediate activation inspection.

### Implementation priority

1. YAML schema, migration strategy, validator, and canonical IR.
2. Visual graph editor and basic PyTorch code generation.
3. Nested graphs and reusable blocks.
4. Python custom blocks.
5. PyTorch and timm importers.
6. ONNX importer/exporter.
7. Advanced optimization, runtime preview, and model debugging.