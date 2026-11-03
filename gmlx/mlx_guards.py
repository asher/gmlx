"""Guards in owned code against defects of stock mlx.

Each function here stands in for one stock op where a supported mlx
version computes it wrong, and is that op everywhere else.
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
    one."""
    arrays = list(arrays)
    ax = axis % arrays[0].ndim
    shape = list(arrays[0].shape)
    shape[ax] = sum(a.shape[ax] for a in arrays)
    if _size(shape) <= _STOCK_MAX:
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
