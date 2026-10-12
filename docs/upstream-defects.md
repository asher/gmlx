# Upstream defects

Defects of stock dependencies that gmlx guards against in its own code. Each entry gives the
dependency and the versions seen, what goes wrong, the smallest repro, the guard and the test
that fails without the guard. An entry stays until no supported version of the dependency has
the defect.

## `mx.addmm` misreads a first argument of one column on the GPU

- Dependency: mlx 0.32.2 and 0.32.3, GPU streams. The CPU stream is right.
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

## The diffusers single-file loader leaves a BF16 tensor outside a linear layer undecoded

- Dependency: diffusers at e0abab83, the revision the image reference scripts under
  `scripts/image_oracle/` pin. gmlx does not import diffusers: only those scripts do, to write
  the fixtures the image tests read.
- Affected: `from_single_file` on a GGUF with a BF16 tensor that is not the weight of a linear
  layer. In the Qwen-Image 2.1 Q4_K_M file that is `txt_in.text_norm.weight`, the only BF16
  vector of the file.
- What goes wrong: the loader keeps the tensor in its GGUF bytes, and only a GGUF linear layer
  decodes them, in its call. Every other module reads the parameter as a plain tensor. On the
  bytes of a BF16 vector that is a tensor of twice the length, so the forward of the model stops
  at the text norm with a shape error.
- Guard: `load_model` in `scripts/image_oracle/check3_qwen_image21.py` decodes each such
  parameter once, to the values the file holds, and refuses a codec other than BF16 there. It
  lists the names it decoded as `results.oracle.plain_decoded` in the manifest of every fixture
  it loads the DiT for.
- Test: `test_the_reference_loader_decoded_the_bf16_vectors_of_the_file` in
  `tests/image/test_image_oracle_harness.py` holds that list, in each committed manifest, to the
  BF16 vectors of the header inventory of the file. Without the guard no such fixture can be
  written.

## pytest 9.1 hides a conftest's fixtures from a directory named twice

- Dependency: pytest 9.1.1, which the test job resolves. 9.0.3 is not affected. gmlx does not
  import pytest: only its test suite does.
- Affected: a command line whose test paths name one directory, then another, then the first
  again, as in `tests/serve/a.py tests/b.py tests/serve/c.py`.
- What goes wrong: pytest collects the directory once for each run of consecutive paths in it,
  and from 9.1 a conftest's fixtures are visible only below the directory node it was collected
  for. The tests of the second node see no fixture of that conftest, its autouse fixtures
  included, so a test that requests one errors with `fixture not found` and the tests that rely
  on an autouse restore fail. A run of the whole suite names each directory once and passes.

Smallest repro:

```
pytest tests/serve/test_admit_gate.py tests/test_envflags.py \
  tests/serve/test_server_patches.py::test_the_seam_restore_lets_the_media_salt_install_again
```

- Guard: `pytest_configure` in `tests/conftest.py` orders the test paths with `grouped` in
  `tests/arg_order.py`, so the paths of each directory are consecutive, in the order of first
  appearance at every level.
- Test: `tests/test_arg_order.py` runs the repro in a child pytest and checks the grouping.
