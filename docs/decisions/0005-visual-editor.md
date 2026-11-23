# 0005 · Visual editing of model documents

**Status** Accepted for Milestone 3.

React owns the complete parsed document while editing. Immutable changes update the entry graph and its public bindings. XYFlow projects operators and boundary ports into separate ID namespaces, with direction-qualified handle IDs. An edge's endpoints identify it. Selection and viewport are session state. Positions are optional graph extension data under `mlker.editor`. Other graphs, provenance, unfamiliar fields and existing identities survive an edit. History keeps bounded document snapshots with structural sharing.

A persistent Python subprocess serves version 2 editor requests through one asynchronous Tauri command. Rust serializes exchanges and drops the child on transport failure. The next request can start a replacement. The independent v1 health probe keeps its small frames and short lifetime. Editor requests transmit document data and file paths. Python supplies registry metadata, schema checks, semantic diagnostics, YAML serialization and standalone export. Generated source is returned for inspection only.

The UI coalesces validation while typing and accepts results only for the current document. Local checks give immediate parameter and connection feedback. The Python validator is authoritative. Schema-correct incomplete wiring can be saved as a draft. Compilation still requires successful semantic validation. Entry graphs are editable; block and Python implementation details stay out of the MVP.
