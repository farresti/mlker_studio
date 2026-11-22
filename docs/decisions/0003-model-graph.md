# 0003 · Persistent model documents and compiler graphs

**Status** Accepted for Milestone 1.

A standalone Python package owns the model format. A document carries a format identifier and integer schema version, independent of the worker and SQLite versions. The current reader supports v1. A future reader must dispatch on the original version, migrate a copy through explicit adjacent-version steps and validate the result before offering to save. Compatibility fixtures belong with each supported version. Opening a document never upgrades it on disk.

The parsed document owns the complete tree of author-supplied values. Saving serializes this tree, including extension keys, unknown fields and import provenance. The derived IR owns a snapshot of the typed contracts, resolved endpoints, parameter bindings and source locations needed for compilation. Keeping both representations avoids reconstructing source metadata from compiler data. YAML presentation details such as comments and quoting have no round-trip guarantee.

Each model and revision has an explicit ID. Graphs and reusable blocks have document-scoped IDs. Nodes have graph-scoped IDs. Ports have IDs within their owner and direction. Display names and list positions can change independently of references. The root model explicitly exposes the entry graph's input and output ports. A block names a graph whose boundary ports and parameter definitions form its contract. Multiple instances share that definition and retain their own bindings. This represents reuse of architecture. Weight identity and parameter tying require future explicit representation.

Validation checks the envelope, the v1 structural schema, identities, registry and parameter contracts, graph wiring and cycles, then the block dependency DAG. A failure carries severity, a stable code, message, source file and JSON Pointer, with graph and node IDs when available. Warnings allow an IR to be built. Declared, partially known tensor information remains distinct from future inferred or observed metadata.

External Python and opaque nodes carry declarative contracts. FX and timm imports can lose dynamic control flow, module sharing and constructor details. ONNX can introduce unfamiliar domains, opsets, variadic ports and symbolic shape expressions. Preserve source identities, conversion warnings and unknown attributes when importing. A successful structural validation makes no claim about executable behavior or external conversion fidelity.
