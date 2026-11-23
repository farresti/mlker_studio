"""Compile an MLKer document to an independent PyTorch source project."""

import fcntl
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .. import ModelDocument, build_ir
from .codegen import render_project
from .lowering import Lowering
from .plan import CompileError

__all__ = ["CompileError", "compile_model"]


def _previous_files(destination: Path) -> set[str]:
    config_path = destination / "config.json"
    if not config_path.is_symlink() and config_path.is_file():
        try:
            config = json.loads(config_path.read_bytes())
        except (ValueError, UnicodeError):
            config = None
        if (isinstance(config, dict) and config.get("compiler") == "mlker-graph"
                and isinstance(config.get("classes"), list) and config["classes"]):
            names = {"config.json", "model.py", "__init__.py", "requirements.txt", "README.md"}
            if len(config["classes"]) > 1:
                names.add("blocks.py")
            return names
    raise FileExistsError(f"{destination} is not a recognized MLKer export. Choose an empty or previously exported directory.")


def _publish(destination: Path, files: dict[str, str]) -> None:
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise FileExistsError(f"Expected a project directory at {destination}")
    if not destination.exists() or not any(destination.iterdir()):
        with TemporaryDirectory(dir=destination.parent) as temporary:
            project = Path(temporary) / "project"
            project.mkdir()
            for name, source in files.items():
                (project / name).write_text(source, encoding="utf-8")
            project.rename(destination)
        return

    previous = _previous_files(destination)
    changed = []
    for name in sorted(previous | files.keys()):
        path = destination / name
        if path.is_symlink() or (path.exists() and (not path.is_file() or name not in previous)):
            raise FileExistsError(f"Cannot replace unrelated file or directory: {path}")
        if name in files:
            if not path.exists() or path.read_bytes() != files[name].encode("utf-8"):
                changed.append(name)
        elif path.exists():
            changed.append(name)
    if not changed:
        return
    with TemporaryDirectory(dir=destination) as temporary:
        stage = Path(temporary)
        backups = stage / "previous"
        backups.mkdir()
        for name in changed:
            if name in files:
                (stage / name).write_text(files[name], encoding="utf-8")
            if (destination / name).exists():
                os.link(destination / name, backups / name)
        applied = []
        try:
            for name in changed:
                if name in files:
                    (stage / name).replace(destination / name)
                else:
                    (destination / name).unlink()
                applied.append(name)
        except OSError:
            for name in reversed(applied):
                if (backups / name).exists():
                    (backups / name).replace(destination / name)
                else:
                    (destination / name).unlink()
            raise


def compile_model(document: ModelDocument, destination: str | Path) -> Path:
    """Validate, lower, and publish a new project or refresh an earlier MLKer export."""
    model = build_ir(document)
    lowering = Lowering(model)
    config = {key: param.default for key, param in model.entry.parameters.items()}
    shapes = {p.id: p.tensor.shape for p in model.inputs}
    entry, inferred = lowering.graph(model.entry, config, shapes, entry=True)
    files = render_project(model, lowering, entry, inferred, document.data["schema_version"])

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The parent inode stays stable across the first atomic directory rename.
    lock = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _publish(destination, files)
    finally:
        os.close(lock)
    return destination
