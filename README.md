# MLKer Studio

A Linux desktop workspace for visual machine learning development. Milestone 0 delivers the application shell and a live Home dashboard. Open **Home → System status** to check the Rust backend, SQLite database and Python worker. The sidebar opens Model Builder and the ONNX Inspector, Experiments, and Settings placeholders.

## Local setup on Ubuntu 24.04

Run the following in Bash. The first block installs system-wide build prerequisites. The compiler, GTK 3 and WebKitGTK 4.1 headers are required to compile the desktop application. SQLite is bundled with the Rust service. The Python worker uses only the standard library. [Tauri Linux prerequisites](https://v2.tauri.app/start/prerequisites/)

```bash
sudo apt update
sudo apt install build-essential curl file pkg-config ca-certificates python3 \
  libgtk-3-dev libwebkit2gtk-4.1-dev libxdo-dev libssl-dev \
  libayatana-appindicator3-dev librsvg2-dev
pkg-config --modversion gtk+-3.0 webkit2gtk-4.1
```

Use a stable Rust toolchain and Node.js 22.12 or later with npm. The build was verified with Rust 1.100.0, Node.js 22.16.0, npm 10.9.2 and Python 3.12.3. If rustup is already installed, start at `rustup toolchain install`. [Rust installation](https://www.rust-lang.org/tools/install)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
rustup toolchain install 1.100.0 --profile minimal --no-self-update
export RUSTUP_TOOLCHAIN=1.100.0
```

Check `node --version` in the terminal you will use to launch the app. If Node is missing or older than 22.12, the following installs the verified x86-64 Node build in your user account. For another architecture, choose the matching Linux package. [Node.js downloads](https://nodejs.org/en/download)

```bash
node_dir="$HOME/.local/share/mlker-tools"
mkdir -p "$node_dir"
(
  set -e
  cd "$node_dir"
  curl -fsSLO https://nodejs.org/dist/v22.16.0/node-v22.16.0-linux-x64.tar.xz
  curl -fsSLO https://nodejs.org/dist/v22.16.0/SHASUMS256.txt
  grep '  node-v22.16.0-linux-x64.tar.xz$' SHASUMS256.txt | sha256sum -c -
  tar -xf node-v22.16.0-linux-x64.tar.xz
)
```

If you installed that build, select it in each new terminal, including the IDE terminal, before running npm. Tauri starts Vite with the same `PATH`.

```bash
export PATH="$HOME/.local/share/mlker-tools/node-v22.16.0-linux-x64/bin:$PATH"
node --version  # v22.16.0
```

## Run from the repository root

```bash
npm ci
npm run desktop
```

This starts Vite on loopback port 1420 and opens the Tauri window in your graphical session. The first build downloads Rust crates. Keep the toolchain exports above in the launching shell. Use **Refresh status** to retry a failed service check. Set `MLKER_PYTHON` to an interpreter executable such as `/path/to/venv/bin/python` to override `python3`.

To compile and run a standalone debug executable with embedded frontend assets and the staged worker resources, use the following commands. Keep the executable in its Cargo output directory alongside its resources.

```bash
npm run desktop:build
./src-tauri/target/debug/mlker-studio
```

For a browser-only preview, use `npm run dev` and open the Vite URL. Service status requires the desktop bridge.

## Validate a model

Milestone 1 adds a standalone Python graph package. From the repository root, use Python 3.10 or later with venv support. On Ubuntu the venv module is available in `python3-venv`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ./python
python -m mlker_graph examples/models/v1/hierarchical.yaml
```

The CLI prints a JSON array of diagnostics. An empty array means validation passed. Errors produce exit code 1 and warnings permit exit code 0. Try `examples/models/v1/invalid/cycle.yaml` for a graph-cycle diagnostic. With PyYAML and jsonschema already installed, the source-tree command is `PYTHONPATH=python python3 -m mlker_graph examples/models/v1/cnn.yaml`.

```python
from mlker_graph import build_ir, load_model, save_model, validate_model

document = load_model("examples/models/v1/hierarchical.yaml")
diagnostics = validate_model(document)
model = build_ir(document)  # Raises ModelError on errors.
print(model.entry.nodes["large"].parameters["width"].value)  # 16
save_model(document, "two-heads.yaml")
```

The [format specification](docs/model-format-v1.md) describes the YAML fields, parameter lookup, hierarchy and round-trip guarantees.

## Compile and run a PyTorch model

With the graph package installed, compile the CNN from the repository root.

```bash
python -m mlker_graph.pytorch examples/models/v1/cnn.yaml exports/cnn
```

In a Python environment with PyTorch 2.x, run a CPU forward and backward pass from the repository root. For a **fresh CPU-only Linux environment**, install the generated requirement with `python -m pip install --index-url https://download.pytorch.org/whl/cpu -r exports/cnn/requirements.txt`. An existing compatible PyTorch installation can run the export directly.

```python
import torch
from exports.cnn import Model

model = Model()
scores = model(torch.randn(2, 1, 8, 8))
assert scores.shape == (2, 5)
scores.sum().backward()
assert all(p.grad is not None for p in model.parameters())
```

The [compiler guide](docs/pytorch-compiler.md) covers the Python API, emitted files, supported operators, sharing, and diagnostics. Runtime tests run when the test interpreter can import PyTorch.

## Build a model visually

Install the graph dependencies in the Python environment used by the desktop. If you installed the graph package into `.venv` as above, launch with `MLKER_PYTHON="$PWD/.venv/bin/python" npm run desktop`. The default interpreter is `python3`.

1. Open **Model Builder** and add **Input**, **Conv2d**, **ReLU**, **MaxPool2d**, **Flatten**, **Linear**, and **Output**. Click a library result to add it or drag it onto the canvas.
2. Connect each output handle to the next input handle, or choose the upstream port under **Properties → Connections**. The supplied parameter suggestions match the default `[B, 1, 8, 8]` input, with 36 features entering Linear and five output scores.
3. Select **Validate**, then **Save**. Use **New** and **Open** to reopen the saved YAML. Click a diagnostic to focus its node or port.
4. Drag either end of an edge to reconnect it. **Auto layout** arranges the graph left to right and fits it in view. Both edits support **Undo**.
5. Select **Compile** and choose an empty directory or a previous MLKer export directory. Recompiling replaces generated source files and preserves checkpoints. **Inspect generated source** opens the generated Python. To execute it, use the PyTorch example above with the generated directory on your import path.

Model Builder supports reusable blocks, nested definition editing and registered Python nodes. The [hierarchy and extensions guide](docs/hierarchy-and-extensions.md) walks through building a reusable residual block and running the Scale example. Ctrl+S saves, Ctrl+Shift+S saves as, and Ctrl+Z or Ctrl+Shift+Z undo and redo canvas edits. Use Delete to remove the selected node, port, or connection. The editor can save schema-correct incomplete drafts.

## Checks

Run these commands from the repository root after desktop and graph package setup.

```bash
npm run typecheck
npm test
npm run build
cargo check --locked --manifest-path src-tauri/Cargo.toml
cargo test --locked --manifest-path src-tauri/Cargo.toml --no-default-features
python3 -m unittest discover -s python/tests -v
```

The Rust test command exercises the service library without desktop dependencies. It starts the real Python package, checks process failures and validates database creation, repeat initialization and schema protection. To try the worker directly, send one request on stdin.

```bash
printf '%s\n' '{"version":1,"id":"manual","method":"ping","params":{}}' | python3 -I -B python/mlker_worker
```

## Data and troubleshooting

- SQLite lives at `$XDG_DATA_HOME/studio.mlker.desktop/studio.sqlite3`, defaulting to `~/.local/share/studio.mlker.desktop/studio.sqlite3`. The directory must be writable. A database from a newer schema reports its version and requires a compatible application.
- A worker startup error identifies the interpreter or exit status. The launching terminal receives Python diagnostics. A hung worker times out after five seconds.
- If Cargo reports a missing `.pc` file, check the development packages above. Runtime shared libraries alone do not supply the headers and pkg-config files used by the build.

## Design

The [architecture](docs/architecture.md) describes the service boundaries, status command, Rust concepts and next milestone. The [worker protocol](docs/worker-protocol.md) defines the wire format. The initial decisions cover [desktop and worker boundaries](docs/decisions/0001-desktop-and-worker.md) and [local storage](docs/decisions/0002-local-storage.md).

## Milestone 0 verification

On Ubuntu 24.04.4 x86-64, the host initially exposed Rust/Cargo 1.84.1 and Python 3.12.3. Node.js and npm were missing from the shell. The desktop development packages missing from the host included `libgtk-3-dev`, `libwebkit2gtk-4.1-dev`, `libjavascriptcoregtk-4.1-dev`, `libsoup-3.0-dev`, `libxdo-dev`, `libssl-dev`, `libayatana-appindicator3-dev` and `librsvg2-dev`. The Ubuntu setup command above resolves their dependencies.

Verification used temporary Node and Rust toolchains, extracted development packages and Xvfb. Type checking, the production frontend build, the desktop Cargo check, the Tauri executable build, seven Rust tests and three Python tests passed. Both the built executable and the development command opened a real Tauri window on the virtual display. Home displayed three ready services, all four placeholders opened, and status refreshed successfully. Separate launches with a missing interpreter and an unusable SQLite path each showed the affected service as unavailable while the other two stayed ready.

## Import a PyTorch or timm model

In **Model Builder → Import model**, choose a trusted PyTorch class or search the installed timm registry. Preview the graph, conversion quality and source hierarchy, then open it in the editor and save a new YAML document. The [model import guide](docs/model-import.md) includes a CPU walkthrough, checkpoint handling and the scope of numerical comparisons.
