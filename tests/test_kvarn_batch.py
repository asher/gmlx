"""BatchKVarNKVCache: row/single-stream bit-equality, batched fused decode
with per-row starts (incl. rows fully inside the tail), mask registration,
filter/extend geometry, and serialization."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.kvarn_cache import BatchKVarNKVCache, KVarNKVCache, KVarNView
from gmlx.kvarn_sdpa import kvarn_attention

_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="kvarn kernels are Metal-only; needs the GPU device",
)

B = 3
H = 2
HQ = 8
D = 128
SCALE = D**-0.5


def _slab(n, b=B, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((b, H, n, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((b, H, n, D)).astype(np.float16))
    return k, v


def _filled(n, pads, tail=256, seed=0):
    c = BatchKVarNKVCache(pads, tail_tokens=tail)
    c.update_and_fetch(*_slab(n, b=len(pads), seed=seed))
    return c


def _make_q(b=B, seed=1, dtype=mx.float16):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((b, HQ, 1, D)).astype(np.float16)).astype(dtype)


def _ref_batch_decode(q, cache, pads):
    """fp32 qL=1 attention over the exact values the route attends, with
    per-row start masking."""
    import mlx_kquant as kq

    n = cache._idx
    t = min(cache.tail_len, n)
    mat_k, mat_v = cache.materialize()
    parts_k, parts_v = [mat_k[:, :, : n - t]], [mat_v[:, :, : n - t]]
    if t:
        tk, tv = cache.tail_slices(t)
        parts_k.append(kq.kvarn_rotate(tk))
        parts_v.append(kq.kvarn_rotate(tv))
    b = q.shape[0]
    k = mx.concatenate(parts_k, axis=2).astype(mx.float32)
    v = mx.concatenate(parts_v, axis=2).astype(mx.float32)
    qr = kq.kvarn_rotate(q.astype(mx.float16)).astype(mx.float32)
    qg = qr.reshape(b, H, HQ // H, 1, D)
    s = (qg @ k[:, :, None].transpose(0, 1, 2, 4, 3)) * SCALE
    kpos = mx.arange(n)[None, None, None, None, :]
    starts = mx.array(pads).reshape(b, 1, 1, 1, 1)
    s = mx.where(kpos >= starts, s, mx.array(-np.inf, mx.float32))
    o = (mx.softmax(s, axis=-1) @ v[:, :, None]).reshape(b, HQ, 1, D)
    return kq.kvarn_rotate(o.astype(mx.float16)).astype(mx.float32)


def _assert_close(out, ref, atol=5e-3):
    d = np.abs(np.array(out.astype(mx.float32)) - np.array(ref)).max()
    assert d < atol, f"max|d|={d}"


# -- construction ------------------------------------------------------------


@_NEEDS_GPU
def test_batch_rows_match_single_stream():
    k, v = _slab(600)
    batch = BatchKVarNKVCache([0, 64, 200], tail_tokens=256)
    batch.update_and_fetch(k, v)
    for b in range(B):
        solo = KVarNKVCache(tail_tokens=256)
        solo.update_and_fetch(k[b : b + 1], v[b : b + 1])
        bk, bv = batch.materialize()
        sk, sv = solo.materialize()
        assert np.array_equal(np.array(bk[b : b + 1]), np.array(sk))
        assert np.array_equal(np.array(bv[b : b + 1]), np.array(sv))


@_NEEDS_GPU
def test_views_and_offsets():
    c = _filled(300, [0, 10, 20])
    kv, vv = c.update_and_fetch(*_slab(1, seed=9))
    assert isinstance(kv, KVarNView) and isinstance(vv, KVarNView)
    with pytest.raises(RuntimeError, match="kvarn SDPA route"):
        np.array(kv)
    assert np.array_equal(np.array(c.offset), [301, 291, 281])
    assert not c.is_trimmable() and c.trim(5) == 0


# -- batched decode ----------------------------------------------------------


def _decode_setup(n, pads, tail=256, seed=0):
    """Fill to n the way serve does: the decode mask is created before the
    step's token lands in the cache, so mask width == post-update _idx."""
    c = BatchKVarNKVCache(pads, tail_tokens=tail)
    k, v = _slab(n, b=len(pads), seed=seed)
    c.update_and_fetch(k[:, :, :-1], v[:, :, :-1])
    mask = c.make_mask(1, window_size=None)
    c.update_and_fetch(k[:, :, -1:], v[:, :, -1:])
    return c, mask


@_NEEDS_GPU
def test_batch_decode_matches_reference():
    # Row starts hit every region: records (150), and past the body/tail
    # boundary (400 > 600-256): that row's body leg attends zero keys and
    # must contribute nothing through the LSE merge.
    pads = [0, 150, 400]
    c, mask = _decode_setup(600, pads)
    q = _make_q()
    out = kvarn_attention(q, c, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, c, pads))


@_NEEDS_GPU
def test_batch_decode_no_tail():
    pads = [0, 150, 296]
    c, mask = _decode_setup(600, pads, tail=0)
    q = _make_q()
    out = kvarn_attention(q, c, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, c, pads))


@_NEEDS_GPU
def test_batch_decode_all_tail():
    pads = [0, 50, 150]
    c, mask = _decode_setup(200, pads, tail=256)
    q = _make_q()
    out = kvarn_attention(q, c, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, c, pads))


@_NEEDS_GPU
def test_unregistered_mask_falls_back():
    # A foreign array mask (no provenance) must still be correct via the
    # materialize path: same values, mask applied by mx.fast.
    pads = [0, 150, 296]
    c, _ = _decode_setup(600, pads)
    q = _make_q()
    from mlx_vlm.models.cache import create_causal_mask

    mask = create_causal_mask(1, offset=c._idx - 1, left_padding=c.left_padding)
    out = kvarn_attention(q, c, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, c, pads), atol=2e-2)


def test_make_mask_registers_starts():
    from gmlx.quantized_sdpa_fix import _registered_starts

    c = BatchKVarNKVCache([0, 5, 9])
    mask = c.make_mask(1, window_size=None)
    starts = _registered_starts(mask)
    assert starts is not None
    assert np.array_equal(np.array(starts), [0, 5, 9])
    windowed = c.make_mask(1, window_size=64)
    assert _registered_starts(windowed) is None


# -- batch ops ---------------------------------------------------------------


@_NEEDS_GPU
def test_filter_selects_rows_without_compaction():
    c = _filled(300, [0, 64, 128])
    before_k = np.array(c.materialize()[0])
    c.filter(mx.array([2, 0]))
    assert c._idx == 300  # no left-pad compaction: records never re-index
    assert np.array_equal(np.array(c.left_padding), [128, 0])
    after_k = np.array(c.materialize()[0])
    assert np.array_equal(after_k, before_k[[2, 0]])


@_NEEDS_GPU
def test_extend_equal_idx_is_bit_exact():
    a = _filled(300, [0, 32], seed=0)
    b = _filled(300, [16], seed=5)
    ak = np.array(a.materialize()[0])
    bk = np.array(b.materialize()[0])
    a.extend(b)
    assert np.array_equal(np.array(a.left_padding), [0, 32, 16])
    mk = np.array(a.materialize()[0])
    assert np.array_equal(mk[:2], ak)
    assert np.array_equal(mk[2:], bk)


@_NEEDS_GPU
def test_extend_realigns_shorter_side():
    a = _filled(640, [0, 32], seed=0)
    b = _filled(400, [8], seed=5)
    b_raw = [np.array(x) for x in b._raw_rows()]
    a.extend(b)
    assert a._idx == 640
    assert np.array_equal(np.array(a.left_padding), [0, 32, 248])
    # The admitted row re-quantized once at the new alignment: its content
    # matches the original raw rows within one extra quantization pass.
    import mlx_kquant as kq

    mk, mv = a.materialize()
    got_k = np.array(kq.kvarn_rotate(mk[2:3]).astype(mx.float32))[:, :, 240:]
    got_v = np.array(kq.kvarn_rotate(mv[2:3]).astype(mx.float32))[:, :, 240:]
    assert np.abs(got_k - b_raw[0].astype(np.float32)).max() < 0.35
    assert np.abs(got_v - b_raw[1].astype(np.float32)).max() < 0.35
    # Decode over the merged batch stays reference-correct.
    q = _make_q()
    mask = a.make_mask(1, window_size=None)
    out = kvarn_attention(q, a, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, a, [0, 32, 248]))


@_NEEDS_GPU
def test_extend_mismatches_raise():
    a = _filled(300, [0])
    with pytest.raises(ValueError, match="mismatched"):
        a.extend(BatchKVarNKVCache([0], k_bits=4, tail_tokens=256))
    with pytest.raises(ValueError, match="empty batch"):
        a.extend(BatchKVarNKVCache([0], tail_tokens=256))


@_NEEDS_GPU
def test_state_round_trip():
    c = _filled(300, [0, 64, 128])
    fresh = BatchKVarNKVCache([0])
    fresh.state = tuple(mx.contiguous(x) for x in c.state)
    fresh.meta_state = c.meta_state
    assert fresh._idx == 300 and fresh.n_sealed == c.n_sealed
    for x, y in zip(fresh.materialize(), c.materialize(), strict=True):
        assert np.array_equal(np.array(x), np.array(y))


def test_finalize_declines_right_padding():
    c = BatchKVarNKVCache([0, 0])
    c.prepare(right_padding=[0, 4])
    with pytest.raises(RuntimeError, match="right padding"):
        c.finalize()
