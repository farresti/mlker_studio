# 0004 · Standalone PyTorch compilation

**Status** Accepted for Milestone 2.

The compiler consumes the validated canonical IR. Lowering selects an explicit PyTorch implementation, resolves graph-local parameter bindings at each block call, schedules the dataflow DAG and checks known shapes. Its plan contains module constructors, parameter aliases, calls and tensor value names. Rendering that plan produces Python source without importing torch or interpreting user code. Generated projects require PyTorch at runtime. Runtime exceptions originate in normal Python and PyTorch.

Each reachable graph and distinct constructor configuration gets a class. A graph reused with identical configuration has one class and normally creates independent instances. The model's public port order defines its forward signature and return order. Block calls use their definition's port order. One output is a tensor, multiple outputs form a tuple. Stable source IDs pass through a scoped name allocator; only backend-owned symbols and literal values enter Python code.

Reshape and Permute need ordered integer arguments. V2 accepts integer lists as literals or graph parameters of type `ints`, which become immutable tuples in the IR. The reference grammar stays a single graph-local lookup.

Schema v2 adds optional node fields for graph-local identity. Equal `instance_id` values request one module or block object. Equal IDs in `parameter_ids` map module slots (such as weight or bias) to one parameter object. V1 readers reject the new schema before they could silently duplicate weights. Existing v1 documents retain independent instances. V1 to v2 migration consists of changing the version field; add sharing IDs only where intended. Sharing applies within each graph instance. Compilation rejects inconsistent constructors, unavailable slots and incompatible tied shapes.

Export records model identity, revision, compiler version, dependency constraint, public names and inferred shapes, alongside the source. Architecture export creates fresh parameters. Keeping weights external leaves checkpoint policy to a later milestone. The desktop worker and frontend remain independent of this backend.
