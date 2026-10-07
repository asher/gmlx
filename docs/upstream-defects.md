# Upstream defects

Defects of stock dependencies that gmlx guards against in its own code. Each entry gives the
dependency and the versions seen, what goes wrong, the smallest repro, the guard and the test
that fails without the guard. An entry stays until no supported version of the dependency has
the defect.

## `mx.addmm` misreads a first argument of one column on the GPU

- Dependency: mlx 0.32.2, GPU streams. The CPU stream is right.
- Affected: `mx.addmm(c, a, b)` whose result has one column and more than one row, so c is
  [..., M, 1] with M above 1, when c is a view whose M values are not consecutive in memory: a
  column of a wider array, every second row of a taller one, reversed rows, a transposed row.
  float32, float16 and bfloat16, lazy or evaluated c, with or without batch dims.
- What goes wrong: the op reads c at a fixed step from its first element that is not the row
  stride of c. For a column of a wider array it returns the elements that follow c[0, 0] in its
  row, one per output row. For a transposed row it reads past the buffer. No error is raised, and
  `a @ b` is added to the wrong values.
- Not affected: a c of two or more columns, or of one row, in any view; a dense c; a and b in
  any view; `a @ b + c`.

Smallest repro:

```python
import mlx.core as mx

wide = mx.arange(21, dtype=mx.float32).reshape(3, 7)
c = wide[:, 3:4]  # [3, 1], the values 3, 10, 17
a, b = mx.zeros((3, 2)), mx.zeros((2, 1))
print(mx.addmm(c, a, b).reshape(-1))  # GPU: [3, 4, 5]
print(mx.addmm(c, a, b, stream=mx.cpu).reshape(-1))  # [3, 10, 17]
print(mx.addmm(mx.contiguous(c), a, b).reshape(-1))  # GPU: [3, 10, 17]
```

- Where gmlx meets it: a LoRA layer adds its delta to the base output with `mx.addmm`, the base
  output as c. That is `LoRAKQuantLinear` in `gmlx/load/modules.py` for a text model and the
  pair placements of `gmlx/image/lora/install.py` for an image model, where a pair on a band of
  one output column adds into a one-column slice of the base output.
- Guard: `addmm` in `gmlx/mlx_guards.py`, which the three call sites use. A c of one column and
  more than one row goes in as `c.reshape(-1).reshape(c.shape)`, which copies a view to dense
  memory, gives a transposed row the strides of its shape and costs a dense c nothing.
  `mx.contiguous(c)` is not enough: a transposed row is consecutive in memory, so it passes
  through unchanged and is still misread. A c of two or more columns, or of one row, goes in as
  it is, so the graph of a layer wider than one output is unchanged.
- Test: `tests/test_mlx_guards.py`. It holds the guard on each kind of view, a text LoRA layer
  of one output over a column of a wider array and an image LoRA band of one output column
  against a float64 sum, and it holds that a wider c reaches the stock op as the same array.
