# Hierarchy and Python extensions

Keep v1/v2 graph, block and node IDs and the existing compiler specialization by graph and effective parameter values. A block definition names one graph. Instances store a reference, declared ports and overrides. Graph parameters form the public configuration. Other node literals stay internal. Graph parameters can forward a parent parameter explicitly.

Extraction moves nodes and their layout into a graph, creates ports at cut edges, and forwards referenced parameters. Reject selections that would split a shared module or tied weights, or create a cycle after contraction. Editing a definition updates its callers. Breadcrumbs track the graph path separately from the document. History covers the whole model so a single undo can restore both sides of a boundary edit.

Use the existing Python implementation mapping with an additional registration ID. Older Python references remain inspectable. Only a manifest explicitly selected by the local user can authorize generation of imports. Loading, validating and compiling read metadata only. The exported model imports the approved implementation in the user's Python runtime. The Rust host continues to exchange data with the Python worker. No schema migration or new Tauri permission is required.
