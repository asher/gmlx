"""Guards in owned code against defects of stock mlx.

Each function here stands in for one stock op where a supported mlx
version computes it wrong, loses digits, or runs it as more kernels
than it needs, and is that op everywhere else.
docs/upstream-defects.md has the defect, its repro and the test of each.
"""

from __future__ import annotations

import mlx.core as mx


def addmm(c: mx.array, a: mx.array, b: mx.array) -> mx.array:
    """``mx.addmm(c, a, b)``, which is ``c + a @ b`` as one op.

    On a GPU stream of mlx 0.32.2 and 0.32.3 the op misreads a ``c`` of
    one column and more than one row when its values are not one element
    apart in memory, or when its strides say so: a column of a wider array,
    or a transposed row. Such a ``c`` goes in dense with the strides of its
    shape, through a reshape to one axis and back. That copies a view
    whose values are apart, and it is free for a dense array.
    ``mx.contiguous`` alone leaves a transposed row as it is, since its
    values are consecutive. A ``c`` of two or more columns, or of one
    row, is read right and goes in as it is, so the graph is the stock
    one."""
    if c.ndim >= 2 and c.shape[-1] == 1 and c.shape[-2] > 1:
        c = c.reshape(-1).reshape(c.shape)
    return mx.addmm(c, a, b)


# The stock pad and concatenate place every input element right in an
# output of at most this many elements.
_STOCK_MAX = 1 << 31


def _size(shape) -> int:
    size = 1
    for n in shape:
        size *= n
    return size


def _written(shape: list[int], fill, dtype, parts) -> mx.array:
    """An array of ``shape`` filled with ``fill``, with each (index,
    array) of ``parts`` written in by a slice update. A slice update
    offsets from the start of the whole array, so it is right at any
    size. mx.full keeps the fill's own dtype for a bool, for an int dtype
    with a float fill and for an array fill, so the fill is cast first."""
    out = mx.full(shape, mx.full((), fill, dtype).astype(dtype))
    for index, a in parts:
        out[index] = a.astype(dtype)
    return out


def concatenate(arrays, axis: int = 0) -> mx.array:
    """``mx.concatenate(arrays, axis)``.

    On a GPU stream of mlx 0.32.3 the op copies each input into a view of
    the output that has the output's strides and the input's element
    count, and picks 32-bit offsets from that count. In an output of more
    than 2^31 elements, each input element whose offset from the start of
    its view passes INT32_MAX is written 2^32 elements below its place,
    outside the output, and its place keeps what the memory held. Such an
    output is built here as zeros with each input written in by a slice
    update. A smaller output is the stock op's, so its graph is the stock
    one. Inputs whose other axes differ go to the stock op too, which
    raises where a slice update would broadcast."""
    arrays = list(arrays)
    ax = axis % arrays[0].ndim
    shape = list(arrays[0].shape)
    shape[ax] = sum(a.shape[ax] for a in arrays)
    rows = shape[:ax] + shape[ax + 1:]
    if _size(shape) <= _STOCK_MAX or any(
            list(a.shape[:ax] + a.shape[ax + 1:]) != rows for a in arrays):
        return mx.concatenate(arrays, axis=axis)
    parts, start = [], 0
    for a in arrays:
        index = [slice(None)] * len(shape)
        index[ax] = slice(start, start + a.shape[ax])
        parts.append((tuple(index), a))
        start += a.shape[ax]
    return _written(shape, 0, mx.result_type(*arrays), parts)


def pad(x: mx.array, pad_width, constant_values=0) -> mx.array:
    """``mx.pad(x, pad_width, constant_values=constant_values)``, the
    constant mode, with ``pad_width`` one (before, after) pair per axis.

    On a GPU stream of mlx 0.32.3 the op fills the output and copies x
    into a view of it that has the output's strides and x's element count,
    with 32-bit offsets picked from that count. In an output of more than
    2^31 elements, each element of x whose offset from the start of its
    view passes INT32_MAX is written 2^32 elements below its place,
    outside the output, and its place keeps the fill. Such an output is
    built here as the fill with x written in by a slice update. A smaller
    output is the stock op's, so its graph is the stock one."""
    pairs = [(int(lo), int(hi)) for lo, hi in pad_width]
    shape = [n + lo + hi for n, (lo, hi) in zip(x.shape, pairs, strict=True)]
    if _size(shape) <= _STOCK_MAX:
        return mx.pad(x, pad_width, constant_values=constant_values)
    index = tuple(slice(lo, lo + n) for n, (lo, _) in zip(x.shape, pairs))
    return _written(shape, constant_values, x.dtype, [(index, x)])


_FLOOR_DIV: dict = {}


def floor_divide(a: mx.array, k: int) -> mx.array:
    """``a // k`` for an integer array and a Python int.

    mlx 0.32.3 builds integer floor division as a divide and a sign fix
    of seven more ops, each its own GPU kernel. For an unsigned ``a``
    and a positive ``k`` the same ops run here as one compiled kernel,
    so the values are the stock op's. Anything else is the stock op."""
    if k <= 0 or not mx.issubdtype(a.dtype, mx.unsignedinteger):
        return a // k
    f = _FLOOR_DIV.get(k)
    if f is None:
        f = _FLOOR_DIV[k] = mx.compile(lambda v: v // k, shapeless=True)
    return f(a)


def layer_norm(x: mx.array, eps: float) -> mx.array:
    """``mx.fast.layer_norm(x, None, None, eps)``: ``x`` normalized over
    its last axis, in its dtype.

    On the CPU stream of mlx 0.32.3 the op, like every CPU reduction,
    adds a row in float32 one element after another, so its error grows
    with the row: 4e-6 relative at 524,288 elements against 5e-8 on the
    GPU. There the statistics are taken in float64, which the CPU
    reduces exactly enough. On a GPU stream it is the stock op."""
    if mx.default_device() != mx.cpu:
        return mx.fast.layer_norm(x, None, None, eps)
    f = x.astype(mx.float64)
    mean = f.mean(axis=-1, keepdims=True)
    var = mx.square(f - mean).mean(axis=-1, keepdims=True)
    return ((f - mean) * mx.rsqrt(var + eps)).astype(x.dtype)
