# 0001 · Desktop and worker boundaries

**Status** Accepted for Milestone 0.

React owns the interface, Tauri exposes application commands, and Rust owns process startup and storage. Python runs as a child process with a versioned JSON Lines interface over its private pipes. Only Rust resolves local data and resource paths. The interpreter is supplied by the host and the worker package ships with the application.

This keeps the ML runtime independently testable and lets the desktop stay responsive during worker I/O. Health uses one process per request and reports failures within a bounded exchange. A persistent process would need scheduling, restart and cancellation semantics. Revisit those costs when the first long-lived job lands.
