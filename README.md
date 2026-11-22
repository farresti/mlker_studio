# MLKer Studio

A Linux desktop workspace for visual machine learning development. Milestone 0 delivers the application shell and a live Home dashboard. Open **Home → System status** to check the Rust backend, SQLite database and Python worker. The sidebar also opens the ONNX Inspector, Experiments, Model Builder and Settings placeholders.

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

If Node.js and npm are missing, the following installs the verified x86-64 Node build in your user account. For another architecture, choose the matching Linux package. [Node.js downloads](https://nodejs.org/en/download)

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
export PATH="$node_dir/node-v22.16.0-linux-x64/bin:$PATH"
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

## Checks

Run these commands from the repository root after desktop and graph package setup.

```bash
npm run typecheck
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
