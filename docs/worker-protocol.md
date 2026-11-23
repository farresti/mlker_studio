# Worker protocol v1

The worker reads one UTF-8 JSON object per line from stdin and writes one response per request to stdout, in order. Each frame ends in LF and has a maximum size of 8,192 bytes including LF. The worker flushes each reply. EOF ends the process. Diagnostics go to stderr. The desktop health probe sends one request and closes stdin.

## Request

```json
{"version":1,"id":"health","method":"ping","params":{}}
```

All four fields are required. The version is an integer. The caller chooses a string ID of 1–128 characters, unique among its outstanding requests on this stream. The method is a string and params is an object. Version 1 implements only `ping`, with empty params. Additional top-level fields are ignored.

## Success

```json
{"version":1,"id":"health","status":"ok","result":{"reply":"pong","pythonVersion":"3.12.3"}}
```

The worker echoes the ID. A successful ping includes the running interpreter's version. Rust checks the envelope version, ID, status and method-specific result before reporting the worker ready.

## Error

```json
{"version":1,"id":"health","status":"error","error":{"code":"unknown_method","message":"Unrecognized worker method."}}
```

Errors carry a machine-readable code and a human-readable message. The envelope always uses the worker's supported version. The ID is null when the worker cannot recover a valid ID.

| Code | Meaning |
| --- | --- |
| `invalid_json` | Invalid JSON or UTF-8 |
| `invalid_request` | Invalid envelope, params or line termination |
| `unsupported_version` | Request uses another protocol version |
| `unknown_method` | Method is unavailable |
| `frame_too_large` | Input exceeded the frame limit and was discarded through LF or EOF |

A bad request produces one error. A later valid request on the same stream can still succeed. Spawn failures, nonzero process exits, malformed replies and timeouts are transport failures surfaced by Rust in the worker's subsystem status.

Incompatible wire-format changes require a new version. Future methods should transfer artifact references and compact metadata. Tensor buffers, images, checkpoints and event files belong on disk.

## Editor protocol v2

Rust starts the same worker with `--editor` and keeps its stdin and stdout open for the editor session. Frames are limited to 4 MiB, including LF. Rust correlates replies, allows one exchange at a time and gives each exchange 30 seconds. Protocol or transport failures replace the child on the next request. Domain errors preserve the process and may include a `diagnostics` array.

| Method | Params | Result |
| --- | --- | --- |
| `registry` | Empty object | Registry contract `version: 1` and `operations` with port IDs, categories, parameter rules and suggested values |
| `load` | Absolute YAML `path` | Parsed `document` |
| `validate` | `document`, optional `source` | Structured `diagnostics` |
| `save` | `document`, absolute YAML `path` | Written `path`. Structural schema errors block writes, and incomplete wiring can be saved. |
| `compile` | `document`, absolute output directory `path`, optional `source` | Export `path` and `sources` keyed by generated Python filename |

Registry, load, save, validation and source generation use the graph package dependencies. Running generated models is a separate operation.
