"""Lossless native mappings for exact module types and explicit FX signatures."""
import operator

import torch
from torch import nn, fx
import torch.nn.functional as F


def square(value):
    if isinstance(value, int):
        return value
    if isinstance(value, tuple) and len(value) == 2 and value[0] == value[1]:
        return value[0]


def module_params(m):
    t = type(m)
    if t is nn.Identity:
        return "Identity", {}
    if t is nn.Linear:
        return "Linear", dict(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
    if t is nn.Conv2d and m.groups == 1 and square(m.dilation) == 1 and m.padding_mode == "zeros":
        fields = {k: square(getattr(m, k)) for k in ("kernel_size", "padding", "stride")}
        if None not in fields.values():
            return "Conv2d", dict(**fields, in_channels=m.in_channels, out_channels=m.out_channels, bias=m.bias is not None)
    if t is nn.BatchNorm2d and m.eps == 1e-5 and m.momentum == 0.1 and m.affine and m.track_running_stats:
        return "BatchNorm2d", {"num_features": m.num_features}
    if t is nn.Flatten:
        return "Flatten", dict(start_dim=m.start_dim, end_dim=m.end_dim)
    if t is nn.Dropout and not m.inplace:
        return "Dropout", {"p": m.p}
    if t in (nn.Sigmoid, nn.Tanh) or (t in (nn.ReLU, nn.SiLU) and not m.inplace) or (t is nn.GELU and m.approximate == "none"):
        return t.__name__, {}
    if t is nn.AdaptiveAvgPool2d:
        dims = (m.output_size,) * 2 if isinstance(m.output_size, int) else m.output_size
        if len(dims) == 2 and all(type(d) is int and d > 0 for d in dims):
            return "AdaptiveAvgPool2d", dict(height=dims[0], width=dims[1])
    if t is nn.MaxPool2d and not m.ceil_mode and not m.return_indices and square(m.padding) is not None and square(m.dilation) == 1:
        size, stride = square(m.kernel_size), square(m.stride)
        if size and stride:
            return "MaxPool2d", dict(kernel_size=size, stride=stride, padding=square(m.padding))


def native(node, model):
    args, kwargs, target = node.args, node.kwargs, node.target
    if node.op == "call_module":
        if len(args) == 1 and isinstance(args[0], fx.Node) and not kwargs:
            module = model.get_submodule(target)
            src = args[0]
            # Out-of-place ReLU is safe only for a fresh result with this sole consumer.
            if type(module) is nn.ReLU and module.inplace and len(src.users) == 1:
                while (src.op == "call_module" and type(model.get_submodule(src.target)) is nn.Identity
                       and len(src.args) == 1 and isinstance(src.args[0], fx.Node) and len(src.args[0].users) == 1):
                    src = src.args[0]
                fresh_module = src.op == "call_module" and type(model.get_submodule(src.target)) in (
                    nn.Conv2d, nn.Linear, nn.BatchNorm2d)
                if fresh_module or (src.op == "call_function" and src.target in (torch.add, operator.add) and not src.kwargs):
                    return "ReLU", {}, args
            mapping = module_params(module)
            if mapping:
                return mapping[0], mapping[1], args
    elif node.op in ("call_function", "call_method"):
        if target in (operator.add, torch.add, "add") and len(args) == 2 and all(isinstance(a, fx.Node) for a in args) and not kwargs:
            return "Add", {}, args
        unary = {torch.relu: "ReLU", F.relu: "ReLU", "relu": "ReLU", torch.sigmoid: "Sigmoid",
                 "sigmoid": "Sigmoid", torch.tanh: "Tanh", "tanh": "Tanh"}
        if target in unary and len(args) == 1 and isinstance(args[0], fx.Node) and not kwargs:
            return unary[target], {}, args
        if target in (torch.flatten, "flatten") and 1 <= len(args) <= 3 and isinstance(args[0], fx.Node) and not kwargs:
            dims = [*args[1:], *([0, -1][len(args)-1:])]
            if all(type(d) is int for d in dims):
                return "Flatten", dict(start_dim=dims[0], end_dim=dims[1]), args[:1]
