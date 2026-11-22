# MLKer model format v1

The file stores one architecture revision. The [v1 JSON Schema](../python/mlker_graph/schema-v1.json) defines its structure. The Python validator also checks references and relationships across records. The [examples](../examples/models/v1) include a feed-forward network, a CNN, a residual connection, two instances of a reusable block, and invalid fixtures.

## An identity graph

A graph boundary can connect directly to its opposite boundary. This complete document forwards a tensor without creating a node.

```yaml
format: mlker.model
schema_version: 1
model:
  id: identity
  revision: r1
  name: Identity
  entry: main
  inputs: [x]
  outputs: [y]
graphs:
  - id: main
    inputs: [{id: x, tensor: {dtype: float32, shape: [B, null]}}]
    outputs: [{id: y, tensor: {dtype: float32, shape: [B, null]}}]
    nodes: []
    edges:
      - {from: {input: x}, to: {output: y}}
```

## Document and identities

| Field | Contract |
| --- | --- |
| `format`, `schema_version` | Exactly `mlker.model` and integer `1`. The reader reports unsupported versions before interpreting their fields. |
| `model` | Required `id`, `revision`, `name`, `entry`, ordered `inputs` and `outputs`. Optional `description`, `metadata` and `provenance`. |
| `graphs` | List of graph definitions. Each has an ID, input and output port lists, node list and directed edge list. Optional `parameters` list. |
| `blocks` | Optional list of reusable definitions, each with `id` and the target graph ID in `graph`. |
| `extensions` | Optional mapping at the document level and on model, graph, block, node, port, tensor, parameter and edge records. Use a plugin-owned key for each extension. |

IDs start with an ASCII letter or underscore and continue with letters, digits, underscores, periods or hyphens. Model and revision IDs identify a revision together. Graph and block IDs occupy separate namespaces within the document. Node IDs are local to a graph. Parameter IDs are local to their defining graph. A port ID is local to its owner and direction. Connect using IDs. Display names and array indices do not determine identity.

The model's public input and output lists reference **every** corresponding entry graph port exactly once. They set the public ordering and inherit the tensor declarations. Other graphs define their own boundaries. Multiple blocks may name the same graph.

## Ports and wiring

A port requires `id` and a `tensor` mapping. The empty mapping accepts an unknown tensor. `dtype` and `layout` are optional, case-sensitive names. Use a consistent dtype vocabulary such as `float32` and `int64`. A missing or null `shape` means unknown rank. An empty shape is scalar. Dimensions can be nonnegative integers, symbolic identifiers or null. Symbols label unknown dimensions and carry no arithmetic semantics in v1.

An edge has `from` and `to`. Use `{input: x}` to read a graph input, `{output: y}` to drive a graph output, or `{node: dense, port: x}` for a node port. A node produces from its outputs and receives on its inputs. Tensor rank, dtype, layout and concrete dimensions must agree wherever both declarations specify comparable values.

A producer may feed many consumers. A destination has at most one driver. Every graph input must feed an edge and every graph output needs a driver. Node inputs default to required. A declared optional input uses `required: false` and can be left disconnected at a call site. An optional graph input still needs an internal binding. All outputs are required. Graph dataflow and graph-to-block dependencies must both be acyclic, including unused definitions.

## Node targets

Each node requires `id`, `kind`, `inputs` and `outputs`. Each kind selects one target field.

| Kind | Target | Validation |
| --- | --- | --- |
| `operator`, `module` | `registry: core.Linear`, for example | Lookup checks kind, required port IDs, parameter names, types and constraints. |
| `block` | `block: projection` | Instance ports must match target graph port IDs and requiredness. Partial tensor declarations can refine one another. The IR retains both declarations for checking external edges. |
| `python` | `implementation: {package: acme_vision, symbol: layers.Stem, version: "1.2"}` | Package and symbol are required opaque strings. Version is optional dependency metadata. Validation reads the declared interface. |
| `opaque` | `source_op: "vendor::CustomOp"` | The importer's declared ports participate in wiring checks. An `opaque_node` warning records the limited semantics. |

The initial registry uses the following contracts. Inputs and outputs shown here are required. All unary operations use input `x` and output `out`. The two binary operations use `a`, `b` and `out`.

| Registry ID | Kind | Parameters |
| --- | --- | --- |
| `core.Conv2d` | module | Positive integers `in_channels`, `out_channels`, `kernel_size`. Defaults are `stride=1`, `padding=0`, `bias=true`. Stride is positive, padding nonnegative. Rank 3 or 4 when known. |
| `core.ReLU` | operator | No parameters. |
| `core.MaxPool2d` | operator | Required positive integer `kernel_size` and `stride`. Rank 3 or 4 when known. |
| `core.Flatten` | operator | Integers `start_dim=1`, `end_dim=-1`. |
| `core.Linear` | module | Positive integers `in_features`, `out_features`. Boolean `bias=true`. |
| `core.Add` | operator | Two inputs, no parameters. |
| `core.Concat` | operator | Two inputs, required integer `axis`. |

## Parameters and hierarchy

A graph declares configurable parameters as a list, for example `[{id: width, type: int, default: 64}]`. Every definition has a scalar default. Types are `int`, `float`, `bool` and `string`. Float parameters accept finite float or integer literals. Booleans are a distinct type.

A node's optional `params` mapping binds names to scalar literals or a mapping containing exactly one key, as in `out_features: {$param: width}`. Lookup uses the containing graph. Reference types must match the receiving parameter's declared type. Dots in an ID are literal characters. A string is always a literal. Expressions, calls, parent traversal and references in parameter defaults have no interpretation.

A block call's parameter names come from its target graph. Omitted values use that graph's defaults. The validator checks an override against constraints from consumers inside the block, including uses forwarded to deeper blocks. Definitions are processed in dependency order. The IR retains each instance's bindings and the referenced definition once. A binding gives its value under the containing graph's defaults and, for a reference, the original typed parameter. A future compiler can specialize each call without mutating the shared definition. Tensor symbols and numeric parameters are separate namespaces. See the [two-head example](../examples/models/v1/hierarchical.yaml).

## Source fidelity, diagnostics and compatibility

Files contain a single UTF-8 YAML document made of string-keyed mappings, lists and JSON-like scalar values. Use lowercase booleans, decimal numbers and finite floats. Null may be written as `null`, `~` or an empty value. Quote a string that looks numeric. Dates, `on` and `off` are strings. The loader rejects duplicate keys, anchors, aliases, merge keys and explicit tags.

Unknown record fields and extension values survive a semantic load/save round trip. Core endpoint and parameter-reference objects have fixed, closed shapes. Model provenance requires `source_format`, `source_identity` and `strategy`. Optional `framework_version`, a list of `warnings` and additional source-specific fields record import history. Saving validates first and atomically replaces the destination. Whitespace, comments, quoting and other presentation details may change.

Diagnostics contain `code`, `severity`, `message` and a `location` with source path, JSON Pointer, and graph/node IDs where known. Reader errors include line and column when the parser supplies them. For example, an unconnected required input can point to `/graphs/0/nodes/1/inputs/0`. Validation stops at a failed prerequisite stage. Independent checks within a stage can return multiple findings. Warnings permit IR construction.

The format and compiler currently support declared tensor contracts, static wiring and scalar parameters. Full broadcasting, symbolic equation solving, derived shapes, dynamic control flow, weight sharing and external model execution require later compiler work. The [format decision](decisions/0003-model-graph.md) records compatibility policy and importer limitations.

For the additional registered operations, v2 sharing fields and integer-list parameters, see the [PyTorch compiler guide](pytorch-compiler.md).
