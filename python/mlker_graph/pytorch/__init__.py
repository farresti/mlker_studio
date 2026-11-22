"""Compile an MLKer document to an independent PyTorch source project."""

from pathlib import Path
from tempfile import TemporaryDirectory

from .. import ModelDocument, build_ir
from .codegen import render_project
from .lowering import Lowering
from .plan import CompileError

__all__ = ["CompileError", "compile_model"]


def _matches(directory: Path, files: dict[str, str]) -> bool:
    if directory.is_symlink() or not directory.is_dir():
        return False
    for name, content in files.items():
        file = directory / name
        if file.is_symlink() or not file.is_file() or file.read_bytes() != content.encode("utf-8"):
            return False
    return True


def compile_model(document: ModelDocument, destination: str | Path) -> Path:
    """Validate, lower and export atomically to a new or empty directory.

    Repeating an identical export succeeds. Existing different files are preserved.
    """
    model = build_ir(document)
    lowering = Lowering(model)
    config = {key: param.default for key, param in model.entry.parameters.items()}
    shapes = {p.id: p.tensor.shape for p in model.inputs}
    entry, inferred = lowering.graph(model.entry, config, shapes, entry=True)
    files = render_project(model, lowering, entry, inferred, document.data["schema_version"])

    destination = Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise FileExistsError(f"Expected a project directory at {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        if _matches(destination, files):
            return destination
        raise FileExistsError(f"{destination} contains different files. Choose an empty export directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        project = Path(temporary) / "project"
        project.mkdir()
        for name, source in files.items():
            (project / name).write_text(source, encoding="utf-8")
        try:
            project.rename(destination)
        except OSError:
            # Another exporter may have published the same complete project first.
            if not _matches(destination, files):
                raise
    return destination
