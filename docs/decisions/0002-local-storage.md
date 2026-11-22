# 0002 · Local storage and independent versions

**Status** Accepted for Milestone 0.

Rust owns an application-local SQLite file using bundled SQLite. Initialization runs in an immediate transaction and records the schema in SQLite's `user_version`. Opening a newer schema reports an error and preserves its version. The first schema contains no domain tables.

This gives future metadata and metrics migrations a single owner. Large artifacts live in files and can later be referenced through metadata. TensorBoard compatibility, model YAML schema versions and the worker protocol have their own lifecycles. A future migration must test reopening existing data as well as creating a new database.
