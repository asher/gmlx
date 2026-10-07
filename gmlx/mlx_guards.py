"""Guards in owned code against defects of stock mlx.

Each function here stands in for one stock op where a supported mlx
version computes it wrong, and is that op everywhere else.
docs/upstream-defects.md has the defect, its repro and the test of each.
"""

from __future__ import annotations

import mlx.core as mx


def addmm(c: mx.array, a: mx.array, b: mx.array) -> mx.array:
    """``mx.addmm(c, a, b)``, which is ``c + a @ b`` as one op.

    On a GPU stream of mlx 0.32.2 the op misreads a ``c`` of one column
    and more than one row when its values are not one element apart in
    memory, or when its strides say so: a column of a wider array, or a
    transposed row. Such a ``c`` goes in dense with the strides of its
    shape, through a reshape to one axis and back. That copies a view
    whose values are apart, and it is free for a dense array.
    ``mx.contiguous`` alone leaves a transposed row as it is, since its
    values are consecutive. A ``c`` of two or more columns, or of one
    row, is read right and goes in as it is, so the graph is the stock
    one."""
    if c.ndim >= 2 and c.shape[-1] == 1 and c.shape[-2] > 1:
        c = c.reshape(-1).reshape(c.shape)
    return mx.addmm(c, a, b)
