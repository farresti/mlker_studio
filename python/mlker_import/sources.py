"""Resolve only a user-selected constructor or a local timm registry entry."""
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys

import torch
from torch import nn


def browse(query="", pretrained_only=False, offset=0):
    import timm
    names = [n for n in timm.list_models(pretrained=pretrained_only, include_tags=False) if query.lower() in n.lower()]
    items = []
    for name in names[offset:offset + 50]:
        cfg = timm.get_pretrained_cfg(name)
        items.append({"name": name, "architecture": cfg.architecture if cfg else name,
                      "input_size": list(cfg.input_size) if cfg and cfg.input_size else None,
                      "num_classes": cfg.num_classes if cfg else None})
    return {"models": items, "total": len(names)}


def construct(options):
    kind, identity = options["source"], options["identity"]
    config = options.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Constructor configuration must be a JSON object.")
    versions = {"torch": str(torch.__version__)}
    if kind == "pytorch":
        if options.get("trusted") is not True:
            raise PermissionError("Review and explicitly trust the Python constructor before loading it.")
        package, separator, name = identity.rpartition(":")
        if not separator or not name.isidentifier():
            raise ValueError("Use package.module:Class or an absolute file.py:Class.")
        if package.endswith(".py"):
            path = Path(package)
            if not path.is_absolute() or not path.is_file():
                raise ValueError("Select an existing absolute Python source file.")
            spec = importlib.util.spec_from_file_location("_mlker_import_source", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(package)
        cls = getattr(module, name)
        if not inspect.isclass(cls) or not issubclass(cls, nn.Module):
            raise ValueError("The selected constructor must be an nn.Module class.")
        model = cls(**config)
    elif kind == "timm":
        import timm
        if identity not in timm.list_models() and identity not in timm.list_models(include_tags=True):
            raise ValueError("Choose an installed timm registry name.")
        if config.keys() - {"num_classes", "in_chans", "global_pool", "drop_rate"}:
            raise ValueError("Supported timm options are num_classes, in_chans, global_pool and drop_rate.")
        versions["timm"] = str(timm.__version__)
        model = timm.create_model(identity, pretrained=options.get("pretrained") is True, **config)
    else:
        raise ValueError("Choose a PyTorch or timm source.")
    if options.get("checkpoint"):
        state = torch.load(options["checkpoint"], map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Select a plain tensor state_dict checkpoint.")
        for key, template in model.state_dict().items():
            if key in state and isinstance(template, torch.Tensor) and state[key].dtype != template.dtype:
                raise ValueError(f"Checkpoint dtype for {key} differs from the selected constructor.")
        model.load_state_dict(state, strict=True)
    return model.cpu(), versions
