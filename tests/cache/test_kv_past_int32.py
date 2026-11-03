"""The KV cache's regrow and batch extends at outputs past 2^31 elements.

On a GPU stream of mlx 0.32.3 a pad or a concatenate on a later axis
misplaces input elements past 2^31 output elements (docs/upstream-defects.md).
The regrow concatenate (gmlx/cache/kv_append.py) and the extend pads
(BatchKVCache.extend, BatchRotatingKVCache.extend and
BatchQuantizedKVCache.extend) go through gmlx.mlx_guards. Each test runs
the cache at the first shape a serve reaches past the limit and checks
every key of the output and of the inputs against the formula of
tests/past_int32.py. The values take the same path at any width, so they
are narrow here.
"""
from __future__ import annotations

import mlx.core as mx

import past_int32 as p32
from gmlx.cache import kv_append
from gmlx.cache.kv_cache import BatchKVCache, BatchQuantizedKVCache, BatchRotatingKVCache

pytestmark = p32.MARKS
# The fixture MARKS uses, where pytest finds it.
settled = p32.settled

# Qwen3.5-4B and 9B: 4 KV heads of 256, bfloat16.
H, D, DV = 4, 256, 8


def _regrow(b: int, s_old: int) -> tuple:
    """A decode step of ``b`` rows at a full buffer of ``s_old`` tokens:
    the regrown cache's shape, the wrong elements of rows b - 1, head 3
    from token 55,808, of every key and of the old keys."""
    rows = b * H
    keys = p32.bf16((b, H, s_old, D))
    values = p32.bf16((b, H, s_old, DV), salt=50)
    cache = BatchKVCache([0] * b)
    cache.keys, cache.values, cache._idx = keys, values, s_old
    cache.offset = mx.array([s_old] * b)
    cache.update_and_fetch(p32.bf16((b, H, 1, D), s0=s_old),
                           p32.bf16((b, H, 1, DV), s0=s_old, salt=50))
    mx.eval(cache.keys, cache.values)
    s_new = cache.keys.shape[2]

    def kept(r, s0, n):
        return p32.placed(r, s0, n, D, 0, s_old + 1)

    tail = cache.keys[b - 1:, H - 1:, 55_808:s_old]
    region = p32.wrong(tail, (1, s_old - 55_808, D),
                       lambda r, s0, n: p32.pattern(rows - 1, 1, 55_808 + s0, n, D)[0])
    out = p32.wrong(cache.keys, (rows, s_new, D), kept)
    old = p32.wrong(keys, (rows, s_old, D), lambda r, s0, n: p32.pattern(r, 1, s0, n, D)[0])
    return tuple(cache.keys.shape), region, out, old


def test_a_decode_regrow_past_2_31_elements_keeps_every_key():
    """8 rows decode at a full buffer of 260,608 tokens, and the next token
    regrows it to 268,800 on the capacity grid. The stock concatenate left
    the keys of batch row 7, head 3 from token 55,808 unwritten and wrote
    them 8 GiB below, over the old keys of batch row 0."""
    assert kv_append.grid_capacity(260_609, BatchKVCache.step) == 268_800
    shape, region, out, old = _regrow(8, 260_608)
    assert shape == (8, H, 268_800, D)
    # {(row, first token of a span): wrong elements}, all three at once.
    assert {"region": region, "keys": out, "old keys": old} == {
        "region": {}, "keys": {}, "old keys": {}}


def _extend(n: int, idx: int) -> tuple:
    """A row one token past ``n`` rows at ``idx`` tokens joins them: the
    joined cache's shape, the wrong elements of rows n - 1, head 3 from
    token 246,273, of every key, of the n rows' keys and of the row's."""
    cap = kv_append.grid_capacity(idx + 1, BatchKVCache.step)
    rows = n * H
    keys = p32.bf16((n, H, idx, D))
    joiner = p32.bf16((1, H, cap, D), r0=rows)
    a = BatchKVCache([0] * n)
    a.keys, a.values, a._idx = keys, p32.bf16((n, H, idx, DV), salt=50), idx
    a.offset = mx.array([idx] * n)
    b = BatchKVCache([0])
    b.keys, b.values = joiner, p32.bf16((1, H, cap, DV), r0=rows, salt=50)
    b._idx, b.offset = idx + 1, mx.array([idx + 1])
    a.extend(b)
    mx.eval(a.keys, a.values)

    def joined(r, s0, n_):
        if r >= rows:
            return p32.pattern(r, 1, s0, n_, D)[0]
        # One token of left padding, then the row's keys, then zeros.
        return p32.placed(r, s0, n_, D, 1, idx + 1, shift=1)

    tail = a.keys[n - 1:n, H - 1:, 246_273:idx + 1]
    region = p32.wrong(tail, (1, idx - 246_272, D),
                       lambda r, s0, n_: p32.pattern(rows - 1, 1, 246_272 + s0, n_, D)[0])
    out = p32.wrong(a.keys, ((n + 1) * H, cap, D), joined)
    own = p32.wrong(keys, (rows, idx, D), lambda r, s0, n_: p32.pattern(r, 1, s0, n_, D)[0])
    other = p32.wrong(joiner, (H, cap, D),
                      lambda r, s0, n_: p32.pattern(rows + r, 1, s0, n_, D)[0])
    return tuple(a.keys.shape), region, out, own, other


def test_a_row_joining_a_batch_past_2_31_elements_keeps_every_key():
    """A row of 293,377 tokens joins 7 rows at 293,376, so extend pads the
    7 rows' keys by one token on the left and 8,191 on the right. The stock
    pad left the keys of batch row 6, head 3 from token 246,272 at the fill
    and wrote them 8 GiB below, over the 7 rows' own keys at batch row 1,
    head 1."""
    shape, region, out, own, other = _extend(7, 293_376)
    assert shape == (8, H, 301_568, D)
    assert {"region": region, "keys": out, "7 rows' keys": own, "row's keys": other} == {
        "region": {}, "keys": {}, "7 rows' keys": {}, "row's keys": {}}


# gemma-4-31B's sliding-window layers: 16 KV heads of 256, a window of 1024.
WH, WIN = 16, 1024


def _rotated(r: int, s0: int, n: int, left: int, k: int) -> mx.array:
    """Tokens s0..s0+n of row r of the decoding side after the extend: the
    fill for the first ``left`` tokens, then the window in temporal order,
    which starts at buffer slot k."""
    window = p32.pattern(r, 1, 0, WIN, D)[0]
    s = mx.arange(s0, s0 + n)
    slot = (mx.maximum(s - left, 0) + k) % WIN
    got = mx.take(window, slot, axis=0)
    return mx.where((s >= left)[:, None], got, mx.zeros_like(got))


def _rotating_extend(n: int, k: int, width: int, prompt: int) -> tuple:
    """A row whose window buffer is ``width`` tokens wide after its prefill
    joins ``n`` rows that decode at a full window with buffer slot k
    next: the joined cache's shape, the wrong elements of rows n - 1,
    head 15 from token 16,754, of every key, of the n rows' keys and of
    the row's."""
    rows = n * WH
    left = width - WIN
    keys = p32.bf16((n, WH, WIN, D))
    joiner = p32.bf16((1, WH, width, D), r0=rows)
    a = BatchRotatingKVCache(WIN, [0] * n)
    a.keys, a.values = keys, p32.bf16((n, WH, WIN, DV), salt=50)
    a._idx, a._offset, a.rotated = k, 4 * WIN + k, True
    a.offset = mx.array([4 * WIN + k] * n)
    b = BatchRotatingKVCache(WIN, [0])
    b.keys, b.values = joiner, p32.bf16((1, WH, width, DV), r0=rows, salt=50)
    b._idx, b._offset, b.offset = width, prompt, mx.array([prompt])
    a.extend(b)
    mx.eval(a.keys, a.values)

    def joined(r, s0, n_):
        if r >= rows:
            return p32.pattern(r, 1, s0, n_, D)[0]
        return _rotated(r, s0, n_, left, k)

    tail = a.keys[n - 1:n, WH - 1:, 16_754:width]
    region = p32.wrong(tail, (1, width - 16_754, D),
                       lambda r, s0, n_: _rotated(rows - 1, 16_754 + s0, n_, left, k))
    out = p32.wrong(a.keys, ((n + 1) * WH, width, D), joined)
    own = p32.wrong(keys, (rows, WIN, D), lambda r, s0, n_: p32.pattern(r, 1, s0, n_, D)[0])
    other = p32.wrong(joiner, (WH, width, D),
                      lambda r, s0, n_: p32.pattern(rows + r, 1, s0, n_, D)[0])
    return tuple(a.keys.shape), region, out, own, other


def test_a_row_joining_sliding_window_rows_past_2_31_elements_keeps_every_key():
    """A serve of gemma-4-31B text with GMLX_DECODE_BATCH=32 and
    --prefill-step-size 16384: a prompt of 32,306 tokens runs its last
    15,922 in one call, so its window buffer is 16,945 tokens wide when it
    joins 31 rows that decode at their full window. extend pads their
    windows by 15,921 tokens on the left. The stock pad left the keys of
    batch row 30, head 15 from token 16,754 at the fill and wrote them
    8 GiB below."""
    shape, region, out, own, other = _rotating_extend(31, 300, 16_945, 32_306)
    assert shape == (32, WH, 16_945, D)
    assert {"region": region, "keys": out, "31 rows' keys": own, "row's keys": other} == {
        "region": {}, "keys": {}, "31 rows' keys": {}, "row's keys": {}}


# gemma-4-26B-A4B's global layers under --kv-bits 8 --kv-quant-scheme
# uniform: 2 KV heads of 512, packed 128 uint32 wide, with scales and
# biases 8 wide at group size 64.
QH, QP, QS = 2, 128, 8


def _quant(n: int, tokens: int, packed: int, scales: int, r0: int, salt: int) -> tuple:
    """(packed, scales, biases) of ``n`` rows of ``tokens``, the packed part
    the pattern of twice its width viewed as uint32."""
    return (p32.bf16((n, QH, tokens, 2 * packed), r0=r0, salt=salt).view(mx.uint32),
            p32.bf16((n, QH, tokens, scales), r0=r0, salt=salt + 10),
            p32.bf16((n, QH, tokens, scales), r0=r0, salt=salt + 20))


def _quant_extend(n: int, idx: int, join: int) -> tuple:
    """A row of ``join`` tokens joins ``n`` rows that decode at a full
    buffer of ``idx`` tokens: the joined packed keys' shape, the wrong
    elements of batch rows n - 2 and n - 1, of every packed key, of the n
    rows' and of the row's."""
    cap = kv_append.grid_capacity(join, BatchQuantizedKVCache.step)
    rows, left, w = n * QH, join - idx, 2 * QP
    a = BatchQuantizedKVCache([0] * n, group_size=64, bits=8)
    a.keys, a.values = _quant(n, idx, QP, QS, 0, 0), _quant(n, idx, 4, 1, 0, 50)
    a._idx, a.offset = idx, mx.array([idx] * n)
    b = BatchQuantizedKVCache([0], group_size=64, bits=8)
    b.keys, b.values = _quant(1, cap, QP, QS, rows, 0), _quant(1, cap, 4, 1, rows, 50)
    b._idx, b.offset = join, mx.array([join])
    keys, joiner = a.keys[0], b.keys[0]
    a.extend(b)
    mx.eval(a.keys, a.values)
    packed = a.keys[0]

    def joined(r, s0, n_):
        if r >= rows:
            return p32.pattern(r, 1, s0, n_, w)[0]
        return p32.placed(r, s0, n_, w, left, join, shift=left)

    tail = packed[n - 2:n, :, left:join]
    region = p32.wrong(tail, (2 * QH, idx, w),
                       lambda r, s0, n_: p32.pattern(rows - 2 * QH + r, 1, s0, n_, w)[0])
    out = p32.wrong(packed, ((n + 1) * QH, cap, w), joined)
    own = p32.wrong(keys, (rows, idx, w), lambda r, s0, n_: p32.pattern(r, 1, s0, n_, w)[0])
    other = p32.wrong(joiner, (QH, cap, w),
                      lambda r, s0, n_: p32.pattern(rows + r, 1, s0, n_, w)[0])
    return tuple(packed.shape), region, out, own, other


def test_a_row_joining_a_quantized_batch_past_2_31_elements_keeps_every_key():
    """A serve of gemma-4-26B-A4B with --kv-bits 8 --kv-quant-scheme
    uniform and GMLX_DECODE_BATCH=64: a row of 129,537 tokens, its buffer
    137,728 on the capacity grid, joins 63 rows that decode at a full
    buffer of 4,352. extend pads their packed keys by 125,185 tokens on
    the left and 8,191 on the right. The stock pad left the packed keys of
    batch rows 61 and 62, 2,228,224 words, at the fill, and the inputs
    intact."""
    shape, region, out, own, other = _quant_extend(63, 4352, 129_537)
    assert shape == (64, QH, 137_728, QP)
    assert {"region": region, "keys": out, "63 rows' keys": own, "row's keys": other} == {
        "region": {}, "keys": {}, "63 rows' keys": {}, "row's keys": {}}
