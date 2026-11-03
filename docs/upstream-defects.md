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

## `mx.pad` misplaces its input past 2^31 output elements on the GPU

- Dependency: mlx 0.32.3, which mlx-kquant 0.5 pins, measured on GPU streams. 0.32.4 has the same
  code. The CPU stream is right. MLX main fixes it in afc03c5 (#4744), which no release carries yet.
- Affected: `mx.pad` in the constant mode, when the output holds more than 2^31 elements, the input
  holds at most INT32_MAX, and the input's last element lands more than INT32_MAX elements past
  its first one in the output. That takes a padded axis after a non-unit one, such as the token axis
  of [B, H, S, D] keys, or the width of an image of more than one row. Any dtype.
- What goes wrong: the op fills the output and copies the input into a view of it that has the
  output's strides and the input's element count. The copy picks 32-bit offsets from that count, so
  each element whose offset from the start of the view passes INT32_MAX is written 2^32 elements
  below its place, outside the output, and its place keeps the fill. In two of gmlx's three
  measurements those writes landed on the op's own input, so a later read of the input is wrong
  too. No error is raised.
- Not affected: an output of at most 2^31 elements; an input of more than INT32_MAX elements, which
  gets 64-bit offsets; a pad of the first non-unit axis only; the edge, reflect and symmetric modes,
  which build the output from slice updates; a slice update into the output (`out[:, :16] = x`),
  which offsets from the start of the whole output.

Smallest repro (a 2 GiB output):

```python
import mlx.core as mx

rows, width = 65600, 32768
x = (mx.arange(rows) % 251).astype(mx.uint8)[:, None] + mx.zeros((1, 16), mx.uint8)
want = (mx.arange(rows) % 251).astype(mx.uint8)
out = mx.pad(x, [(0, 0), (0, width - 16)])  # [65600, 32768], 2.1e9 elements
print((out[:, 0] != want).sum())  # GPU: 64, the rows from 65536
out = mx.pad(x, [(0, 0), (0, width - 16)], stream=mx.cpu)
print((out[:, 0] != want).sum())  # 0
out = mx.zeros((rows, width), mx.uint8)
out[:, :16] = x  # a slice update
print((out[:, 0] != want).sum())  # 0
```

- Where gmlx meets it:
  - `BatchKVCache.extend` in `gmlx/cache/kv_cache.py` pads the keys and values of each side to the
    joint width when rows join a decoding batch of `gmlx serve`. With 4 KV heads of 256, as on
    Qwen3.5-4B and 9B, a row of 293,377 tokens that joins 7 rows at 293,376 is the first that
    passes. That is past the trained context, which the server admits.
  - `BatchRotatingKVCache.extend` in the same file pads the caches of sliding-window layers the same
    way. A joining row's buffer is its window plus its last prefill call, which a model that keeps
    only the last logits runs whole, so it widens with `--prefill-step-size`. On gemma-4-31B served
    text-only, with 16 KV heads of 256 in a window of 1024, `GMLX_DECODE_BATCH=32` and
    `--prefill-step-size 16384`, a prompt of 32,306 tokens that joins 31 decoding rows is the first
    that passes. At the default 8 rows gemma-4-26B-A4B passes with a step of 262,144 and a prompt of
    152,502 tokens. `--max-kv-size` does not build these caches in `gmlx serve`.
  - `BatchQuantizedKVCache.extend` in the same file pads the packed keys and values and their scales
    and biases the same way under `--kv-bits` with `--kv-quant-scheme uniform`. On gemma-4-26B-A4B
    at 8 bits, with 2 KV heads of 512 in its global layers and `GMLX_DECODE_BATCH=64`, a row of
    129,537 tokens that joins 63 decoding rows is the first that passes.
  - `Downsample` in `gmlx/image/vae/flux1_ae.py` pads the bottom and the right of the FLUX.1 VAE
    encoder's 128 channels. Up to 8192 a side, 38 canvases from 16,771,072 to 16,776,960 pixels
    pass, 4080x4112 and 4064x4128 among them. `gmlx image` reaches them with an init image, which it
    cuts down to whole multiples of the stride.
- Guard: `pad` in `gmlx/mlx_guards.py`, which all four call. Past 2^31 output elements it builds
  the output as the fill with the input written in by a slice update. At or under 2^31 it is the
  stock op, so the graph of every smaller shape is unchanged. At the tested shapes it holds the
  stock op's memory peak. It stays until an MLX release carries afc03c5 and an A/B on these shapes
  shows the stock op right.
- Test: `test_a_row_joining_a_batch_past_2_31_elements_keeps_every_key`,
  `test_a_row_joining_sliding_window_rows_past_2_31_elements_keeps_every_key` and
  `test_a_row_joining_a_quantized_batch_past_2_31_elements_keeps_every_key` in
  `tests/cache/test_kv_past_int32.py`, and
  `test_a_vae_downsample_past_2_31_elements_keeps_every_pixel` in
  `tests/image/test_image_past_int32.py`. Each runs the caller at the first shape that passes, on
  a real Apple GPU, and checks every element of the output and of the inputs against a formula.
  With the stock op, the first leaves 12,058,624 keys of batch row 6, head 3 at the fill and writes
  as many over the 7 rows' own keys. The second leaves 48,896 keys of batch row 30, head 15 at the
  fill. The third leaves the 2,228,224 packed words of the keys of batch rows 61 and 62 at the
  fill. The fourth leaves 489,344 elements of the last image row at the fill and writes 472,960 of
  them over the start of x. `test_pad_takes_a_slice_update_only_past_the_limit` in
  `tests/test_mlx_guards.py` holds the routing on small arrays in every run. The past-limit tests
  are nightly. `tests/past_int32.py` runs them one at a time, on a real Apple GPU with a recommended
  working set of at least 27.6 GiB, twice the largest test's footprint, so `gate.py run --full` on
  such a Mac covers them.

## `mx.concatenate` misplaces an input past 2^31 output elements on the GPU

- Dependency: mlx 0.32.3, which mlx-kquant 0.5 pins, measured on GPU streams. 0.32.4 has the same
  code. The CPU stream is right. MLX main fixes it in df46f58 (#4758), which no release carries yet.
  `mx.stack` on a later axis goes through concatenate.
- Affected: `mx.concatenate` when the output holds more than 2^31 elements, an input holds at most
  INT32_MAX, and that input's last element lands more than INT32_MAX elements past its first one in
  the output. That takes a joined axis after a non-unit one, such as the token axis of [B, H, S, D]
  keys or the last axis of [1, L, D] rows. Any dtype.
- What goes wrong: the op copies each input into a view of the output that has the output's strides
  and the input's element count, and picks 32-bit offsets from that count, as `mx.pad` does. Each
  element whose offset from the start of its view passes INT32_MAX is written 2^32 elements below
  its place, outside the output, and its place keeps what the memory held. In gmlx's measurements
  those writes landed on the op's own inputs, and when the copy read such an input afterwards, rows
  of the output under the limit came out wrong too. No error is raised.
- Not affected: an output of at most 2^31 elements; the first non-unit axis; an input of more than
  INT32_MAX elements, which gets 64-bit offsets, though a smaller input beside it is still
  misplaced; a slice update into the output.

Smallest repro (a 2 GiB output):

```python
import mlx.core as mx

rows, width = 65600, 32768
x = (mx.arange(rows) % 251).astype(mx.uint8)[:, None] + mx.zeros((1, 16), mx.uint8)
want = (mx.arange(rows) % 251).astype(mx.uint8)
out = mx.concatenate([x, mx.zeros((rows, width - 16), mx.uint8)], axis=1)
print((out[:, 0] != want).sum())  # GPU: 64, the rows from 65536
out = mx.concatenate([x, mx.zeros((rows, width - 16), mx.uint8)], axis=1, stream=mx.cpu)
print((out[:, 0] != want).sum())  # 0
```

- Where gmlx meets it:
  - `grow` in `gmlx/cache/kv_append.py` regrows a KV cache by concatenating zeros on the token axis.
    With 8 rows decoding and 4 KV heads of 256, as on Qwen3.5-4B and 9B, the regrow from 260,608 to
    268,800 tokens is the first that passes, inside their trained 262,144. gemma-4-26B-A4B passes at
    the same point. Plain `gmlx serve` reaches it, and the keys of batch row 7, head 3 from token
    55,808 are lost. Each later regrow writes its zeros 8 GiB below.
  - `_pad_shared_kv_seq` in `gmlx/spec/speculative.py` pads the full-attention keys and values that
    a gemma-4 assistant drafter shares, when rows join an MTP batch. Two rows of gemma-4-E4B at
    1,048,575 tokens that are padded to 1,048,577 are the first that pass, past the trained context.
  - `SingleBlock.out` in `gmlx/image/families/flux2_klein/model.py` joins a FLUX.2 klein block's
    attention and feed-forward rows on the last axis. 9B passes from 131,073 rows and 4B from
    174,763, which the ComfyUI steps route reaches with several large references and `gmlx image`
    with a large canvas.
- Guard: `concatenate` in `gmlx/mlx_guards.py`, which all three call. Past 2^31 output elements it
  builds the output as zeros with each input written in by a slice update. At or under 2^31 it is
  the stock op, so the graph of every smaller shape is unchanged. At the tested shapes it holds the
  stock op's memory peak. It stays until an MLX release carries df46f58 and an A/B on these shapes
  shows the stock op right.
- Test: `test_a_decode_regrow_past_2_31_elements_keeps_every_key` in
  `tests/cache/test_kv_past_int32.py`, `test_joining_rows_padded_past_2_31_elements_keep_every_key`
  in `tests/spec/test_shared_kv_past_int32.py` and
  `test_a_single_block_past_2_31_elements_keeps_every_row` in
  `tests/image/test_image_past_int32.py`. Each runs the caller at the first shape that passes, on a
  real Apple GPU, and checks every element of the output and of the inputs against a formula. With
  the stock op the regrow leaves 52,428,800 keys unwritten and writes as many over the old keys, the
  shared-KV pad 1,024, and the single block its last row of 16,384, written over its attention
  rows. `test_concatenate_takes_a_slice_update_only_past_the_limit` in `tests/test_mlx_guards.py`
  holds the routing on small arrays in every run. The past-limit tests run as the pad's do.

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
