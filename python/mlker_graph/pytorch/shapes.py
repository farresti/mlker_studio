"""Conservative shape propagation for the supported PyTorch operations."""

from math import prod

Dim = int | str | None
Shape = tuple[Dim, ...] | None


class ShapeError(ValueError):
    pass


def compatible_dim(a: Dim, b: Dim) -> Dim:
    if isinstance(a, int) and isinstance(b, int) and a != b:
        raise ShapeError(f"Incompatible dimensions {a} and {b}.")
    if isinstance(a, int) or b is None:
        return a
    return b


def merge(actual: Shape, declared: Shape) -> Shape:
    if actual is None:
        return declared
    if declared is None:
        return actual
    if len(actual) != len(declared):
        raise ShapeError(f"Inferred rank {len(actual)} conflicts with declared rank {len(declared)}.")
    return tuple(compatible_dim(a, b) for a, b in zip(actual, declared))


def axis(index: int, rank: int) -> int:
    if not -rank <= index < rank:
        raise ShapeError(f"Axis {index} is outside rank {rank}.")
    return index % rank


def binary_shape(op, a, b, params):
    if op == "core.Concat":
        if a == () or b == ():
            raise ShapeError("Concat requires tensors of rank one or greater.")
        # torch.cat accepts a one-dimensional empty tensor alongside any nonscalar.
        if a == (0,) or b == (0,):
            other = b if a == (0,) else a
            if other is not None:
                axis(params["axis"], len(other))
            return other
        known = a if a is not None else b
        if known is None:
            return None
        dim = axis(params["axis"], len(known))
        if a is None or b is None:
            return known[:dim] + (None,) + known[dim + 1:]
        if len(a) != len(b):
            raise ShapeError("Concat requires equal ranks.")
        size = a[dim] + b[dim] if isinstance(a[dim], int) and isinstance(b[dim], int) else None
        return tuple(size if i == dim else compatible_dim(x, y) for i, (x, y) in enumerate(zip(a, b)))

    if a is None or b is None:
        return None
    rank = max(len(a), len(b))
    result = []
    for x, y in zip((1,) * (rank - len(a)) + a, (1,) * (rank - len(b)) + b):
        if x == 1:
            result.append(y)
        elif y == 1 or x == y:
            result.append(x)
        elif isinstance(x, int) or isinstance(y, int):
            result.append(compatible_dim(x, y))
        else:
            result.append(None)
    return tuple(result)


def reshape(shape: Shape, target: tuple[int, ...]) -> Shape:
    if target.count(-1) > 1 or any(d < -1 for d in target):
        raise ShapeError("Reshape accepts nonnegative sizes and at most one -1.")
    fixed = prod(d for d in target if d != -1)
    if fixed == 0 and -1 in target:
        raise ShapeError("A zero-sized reshape cannot infer a -1 dimension.")
    inferred = None
    if shape is not None and all(isinstance(d, int) for d in shape):
        count = prod(shape)
        if (-1 not in target and fixed != count) or (-1 in target and count % fixed):
            raise ShapeError(f"Reshape {target} is incompatible with {count} elements.")
        if -1 in target:
            inferred = count // fixed
    return tuple(inferred if d == -1 else d for d in target)


def infer(operation, inputs: list[Shape], params) -> Shape:
    op = operation.id
    if op in ("core.Add", "core.Concat"):
        return binary_shape(op, *inputs, params)
    shape = inputs[0]
    if op == "core.Reshape":
        return reshape(shape, params["shape"])
    if op == "core.Permute":
        dims = params["dims"]
        axes = [axis(dim, len(dims)) for dim in dims]
        if len(set(axes)) != len(dims) or (shape is not None and len(dims) != len(shape)):
            raise ShapeError("Permute dims must contain each input axis exactly once.")
        return tuple(shape[i] if shape is not None else None for i in axes)
    if shape is None:
        return None
    rank = len(shape)
    if operation.ranks and rank not in operation.ranks:
        raise ShapeError(f"{op} expects rank in {operation.ranks}, received {shape}.")
    if op == "core.Linear":
        if not rank:
            raise ShapeError("Linear needs a feature dimension.")
        compatible_dim(shape[-1], params["in_features"])
        return (*shape[:-1], params["out_features"])
    if op == "core.BatchNorm2d":
        compatible_dim(shape[1], params["num_features"])
        return shape
    if op == "core.AdaptiveAvgPool2d":
        return (*shape[:-2], params["height"], params["width"])
    if op in ("core.Conv2d", "core.MaxPool2d"):
        channels = shape[-3]
        if op == "core.Conv2d":
            compatible_dim(channels, params["in_channels"])
            channels = params["out_channels"]
        spatial = []
        for size in shape[-2:]:
            if isinstance(size, int):
                result = (size + 2 * params.get("padding", 0) - params["kernel_size"]) // params["stride"] + 1
                if size == 0 or result <= 0:
                    raise ShapeError(f"{op} has an empty spatial dimension for input {shape}.")
            else:
                result = None
            spatial.append(result)
        return (*shape[:-3], channels, *spatial)
    if op == "core.Flatten":
        shape = shape or (1,)  # PyTorch treats a scalar as having one element.
        start, end = (axis(params[key], len(shape)) for key in ("start_dim", "end_dim"))
        if start > end:
            raise ShapeError("Flatten start_dim must precede or equal end_dim.")
        dims = shape[start:end + 1]
        if len(dims) == 1:
            return shape
        size = prod(dims) if all(isinstance(d, int) for d in dims) else None
        return (*shape[:start], size, *shape[end + 1:])
    return shape  # Activations and dropout preserve shape.
