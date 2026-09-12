"""BatchKVarNKVCache: row/single-stream bit-equality, batched fused decode
and verify with a geometry per row (incl. rows fully inside the tail),
ragged rollback replay, mask registration, filter/extend geometry, and
serialization."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.cache.kvarn_cache import BatchKVarNKVCache, KVarNKVCache, KVarNView
from gmlx.cache.kvarn_sdpa import kvarn_attention
from kvarn_testlib import D, H, needs_kvarn_ops

B = 3
HQ = 8
SCALE = D**-0.5
needs_row_ends = pytest.mark.needs_kvarn_row_ends
needs_row_ends = pytest.mark.needs_kvarn_row_ends


def _slab(n, b=B, seed=0, d=D):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((b, H, n, d)).astype(np.float16))
    v = mx.array(rng.standard_normal((b, H, n, d)).astype(np.float16))
    return k, v


def _filled(n, pads, tail=256, seed=0):
    c = BatchKVarNKVCache(pads, tail_tokens=tail)
    c.update_and_fetch(*_slab(n, b=len(pads), seed=seed))
    return c


def _make_q(b=B, seed=1, dtype=mx.float16, d=D, qL=1):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((b, HQ, qL, d)).astype(np.float16)).astype(dtype)


def _ref_batch_decode(q, cache):
    """fp32 attention over the exact values the route attends, one row at
    a time: the row's rotated body from its start to tail_cap short of its
    end, then its rotated tail rows, every query clamped at the row's own
    causal position. q is [B, HQ, qL, d]."""
    import mlx_kquant as kq

    b, hq, qL, d = q.shape
    h = cache.stage_k.shape[1]
    mat_k, mat_v = cache.materialize()
    qr = kq.kvarn_rotate(q.astype(mx.float16)).astype(mx.float32)
    outs = []
    for r in range(b):
        s, e = cache.starts[r], cache.ends[r]
        t = min(e - s, cache.tail_cap)
        parts_k, parts_v = [mat_k[r : r + 1, :, s : e - t]], [mat_v[r : r + 1, :, s : e - t]]
        if t:
            te = cache.tail_ends[r]
            parts_k.append(kq.kvarn_rotate(cache.tail_k[r : r + 1, :, te - t : te]))
            parts_v.append(kq.kvarn_rotate(cache.tail_v[r : r + 1, :, te - t : te]))
        k = mx.concatenate(parts_k, axis=2).astype(mx.float32)
        v = mx.concatenate(parts_v, axis=2).astype(mx.float32)
        n = e - s
        qg = qr[r : r + 1].reshape(1, h, hq // h, qL, d)
        sc = (qg @ k[:, :, None].transpose(0, 1, 2, 4, 3)) * (d**-0.5)
        kpos = mx.arange(n)[None, None, None, None, :]
        qpos = (n - qL + mx.arange(qL))[None, None, None, :, None]
        sc = mx.where(kpos <= qpos, sc, mx.array(-np.inf, mx.float32))
        outs.append((mx.softmax(sc, axis=-1) @ v[:, :, None]).reshape(1, hq, qL, d))
    o = mx.concatenate(outs, axis=0)
    return kq.kvarn_rotate(o.astype(mx.float16)).astype(mx.float32)


def _assert_close(out, ref, atol=5e-3):
    d = np.abs(np.array(out.astype(mx.float32)) - np.array(ref)).max()
    assert d < atol, f"max|d|={d}"


def _assert_shadow(c):
    """The shadow geometry the stack's fp16 layers see, against the
    physical rows: _idx - left_padding[b] == ends[b] - starts[b]."""
    pads = c.left_padding.tolist()
    assert pads == c._pads
    for b in range(len(c.ends)):
        assert c._idx - pads[b] == c.ends[b] - c.starts[b], (b, c._idx, pads, c.starts, c.ends)


def _assert_row_equals(c, b, ref):
    """Row b of c against row 0 of a one-row cache: records and stage in
    the rotated domain, recovered K/V (tail rows included), and the seal
    count, all bit-exact."""
    assert c.n_sealed_row(b) == ref.n_sealed_row(0)
    assert c.ends[b] - c.starts[b] == ref.ends[0] - ref.starts[0]
    for x, y in zip(c._materialize_row(b), ref._materialize_row(0), strict=True):
        assert np.array_equal(np.array(x), np.array(y))
    for x, y in zip(c._raw_row(b), ref._raw_row(0), strict=True):
        assert np.array_equal(np.array(x), np.array(y))


def _spec_round(c, blk_k, blk_v, keep):
    """One verify round on a batch cache: the block lands on every row,
    then the rollback harden_mtp_rollback applies (a uniform trim to the
    best row, then each row's own right padding through finalize)."""
    c.update_and_fetch(blk_k, blk_v)
    best = max(keep)
    trim = blk_k.shape[2] - best
    if trim > 0:
        assert c.trim(trim) == trim
    right = [best - k for k in keep]
    if any(right):
        c.prepare(right_padding=right)
        c.finalize()


def _ragged_rounds(c, refs, rounds, seed=3, block=4):
    """Drive ragged verify rounds on c and the matching one-row refs: row
    b keeps (round + b) % block + 1 tokens of every block, so every
    rejected count from 0 to block - 1 lands on every row at every seal
    phase. Returns the per-row token counts kept."""
    rng = np.random.default_rng(seed)
    nb = len(refs)
    kept = [0] * nb
    for r in range(rounds):
        keep = [(r + b) % block + 1 for b in range(nb)]
        blk_k = mx.array(rng.standard_normal((nb, H, block, D)).astype(np.float16))
        blk_v = mx.array(rng.standard_normal((nb, H, block, D)).astype(np.float16))
        _spec_round(c, blk_k, blk_v, keep)
        for b in range(nb):
            refs[b].update_and_fetch(
                blk_k[b : b + 1, :, : keep[b]], blk_v[b : b + 1, :, : keep[b]]
            )
            kept[b] += keep[b]
        _assert_shadow(c)
    return kept


def _ragged_cache(pads=(0, 150, 100), right=(0, 30, 300), n=600, tail=256, d=D):
    """Rows of different lengths from one prefill and per-row trims: with
    the defaults, lengths 600 (full body), 420 (body shorter than two
    groups) and 200 (inside its tail: empty body)."""
    c = BatchKVarNKVCache(list(pads), tail_tokens=tail)
    k, v = _slab(n, b=len(pads), seed=2, d=d)
    c.update_and_fetch(k, v)
    if any(right):
        c.prepare(right_padding=list(right))
        c.finalize()
    return c


# -- construction ------------------------------------------------------------


@needs_kvarn_ops
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


@needs_kvarn_ops
def test_views_and_offsets():
    c = _filled(300, [0, 10, 20])
    kv, vv = c.update_and_fetch(*_slab(1, seed=9))
    assert isinstance(kv, KVarNView) and isinstance(vv, KVarNView)
    with pytest.raises(RuntimeError, match="kvarn SDPA route"):
        np.array(kv)
    assert np.array_equal(np.array(c.offset), [301, 291, 281])
    assert c.is_trimmable() and c.trim(5) == 5
    assert c.ends == [296, 296, 296]
    assert np.array_equal(np.array(c.offset), [296, 286, 276])
    _assert_shadow(c)


# -- batched decode ----------------------------------------------------------


def _decode_setup(n, pads, tail=256, seed=0, d=D):
    """Fill to n the way serve does: the decode mask is created before the
    step's token lands in the cache, so mask width == post-update _idx."""
    c = BatchKVarNKVCache(pads, tail_tokens=tail)
    k, v = _slab(n, b=len(pads), seed=seed, d=d)
    c.update_and_fetch(k[:, :, :-1], v[:, :, :-1])
    mask = c.make_mask(1, window_size=None)
    c.update_and_fetch(k[:, :, -1:], v[:, :, -1:])
    return c, mask


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
def test_batch_decode_matches_reference(d):
    # Row starts hit every region: records (150), and past the body/tail
    # boundary (400 > 600-256): that row's body leg attends zero keys and
    # must contribute nothing through the LSE merge.
    pads = [0, 150, 400]
    c, mask = _decode_setup(600, pads, d=d)
    q = _make_q(d=d)
    out = kvarn_attention(q, c, d**-0.5, mask)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_kvarn_ops
def test_batch_decode_d512_gqa16_shipped_shape():
    # gemma-4 global layers on serve: 1 kv head, 16 q heads; gqa sits on
    # _decode_batch's <= 16 limit.
    pads = [0, 150, 400]
    c = BatchKVarNKVCache(pads, tail_tokens=256)
    rng = np.random.default_rng(7)
    k = mx.array(rng.standard_normal((3, 1, 600, 512)).astype(np.float16))
    v = mx.array(rng.standard_normal((3, 1, 600, 512)).astype(np.float16))
    c.update_and_fetch(k[:, :, :-1], v[:, :, :-1])
    mask = c.make_mask(1, window_size=None)
    c.update_and_fetch(k[:, :, -1:], v[:, :, -1:])
    q = mx.array(rng.standard_normal((3, 16, 1, 512)).astype(np.float16))
    out = kvarn_attention(q, c, 512**-0.5, mask)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_kvarn_ops
def test_batch_decode_no_tail():
    pads = [0, 150, 296]
    c, mask = _decode_setup(600, pads, tail=0)
    q = _make_q()
    out = kvarn_attention(q, c, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_kvarn_ops
def test_batch_decode_all_tail():
    pads = [0, 50, 150]
    c, mask = _decode_setup(200, pads, tail=256)
    q = _make_q()
    out = kvarn_attention(q, c, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_kvarn_ops
def test_unregistered_mask_falls_back():
    # A foreign array mask carries nothing the batch route needs: the
    # geometry comes off the cache, so the result matches the reference
    # whatever the mask describes.
    pads = [0, 150, 296]
    c, _ = _decode_setup(600, pads)
    q = _make_q()
    from mlx_vlm.models.cache import create_causal_mask

    mask = create_causal_mask(1, offset=c._idx - 1, left_padding=c.left_padding)
    out = kvarn_attention(q, c, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, c), atol=2e-2)


def test_make_mask_registers_starts():
    from gmlx.upstream.quantized_sdpa_fix import (_registered_geometry,
                                                  _registered_starts)

    c = BatchKVarNKVCache([0, 5, 9])
    mask = c.make_mask(1, window_size=None)
    starts = _registered_starts(mask)
    assert starts is not None
    assert np.array_equal(np.array(starts), [0, 5, 9])
    # the physical geometry after the pending update, at any width
    geo = _registered_geometry(mask)
    assert np.array_equal(np.array(geo[1]), [1, 1, 1])
    wide = c.make_mask(4, window_size=None)
    assert np.array_equal(np.array(_registered_geometry(wide)[1]), [4, 4, 4])
    windowed = c.make_mask(1, window_size=64)
    assert _registered_starts(windowed) is None


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
def test_explicit_starts_match_registered_mask(d):
    # Owned dispatches (qwen3.5) pass cache.left_padding directly; the
    # fused result must match the mask-provenance route bit for bit.
    pads = [0, 150, 400]
    c, mask = _decode_setup(600, pads, d=d)
    q = _make_q(d=d)
    via_mask = kvarn_attention(q, c, d**-0.5, mask)
    via_starts = kvarn_attention(q, c, d**-0.5, None, starts=c.starts_mx, ends=c.ends_mx)
    assert np.array_equal(np.array(via_starts), np.array(via_mask))


@needs_kvarn_ops
def test_explicit_starts_masked_fallback(monkeypatch):
    # Declined fused decode with explicit starts must mask pad rows on
    # the materialize path, not attend them.
    monkeypatch.setenv("GMLX_KVARN_SDPA", "0")
    pads = [0, 150, 296]
    c, _ = _decode_setup(600, pads)
    q = _make_q()
    out = kvarn_attention(q, c, SCALE, None, starts=c.starts_mx, ends=c.ends_mx)
    _assert_close(out, _ref_batch_decode(q, c), atol=2e-2)


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
def test_qwen35_arm_uses_cache_pads(d):
    # The qwen3.5 decode protocol strips the mask to None and carries the
    # pads on the cache; the arm must recover per-row starts from it.
    # d=256 is the shape every real qwen3.5/3.6 checkpoint dispatches.
    pytest.importorskip("mlx_vlm.models.qwen3_5")
    from gmlx.models.qwen35.attn import _kvarn_attention

    pads = [0, 150, 296]
    c, _ = _decode_setup(600, pads, d=d)
    q = _make_q(d=d)
    out = _kvarn_attention(q, cache=c, scale=d**-0.5, mask=None)
    _assert_close(out, _ref_batch_decode(q, c))


# -- batch ops ---------------------------------------------------------------


@needs_kvarn_ops
def test_filter_selects_rows_bit_exactly_when_a_row_keeps_no_padding():
    c = _filled(300, [0, 64, 128])
    before_k = np.array(c.materialize()[0])
    c.filter(mx.array([2, 0]))
    assert c._idx == 300  # a zero-pad row survives: nothing to compact
    assert np.array_equal(np.array(c.left_padding), [128, 0])
    after_k = np.array(c.materialize()[0])
    assert np.array_equal(after_k, before_k[[2, 0]])


@needs_kvarn_ops
def test_filter_compacts_shared_padding_like_batch_kv_cache():
    """The stack shares one mask. BatchKVCache.filter shifts left by the
    minimum padding, so a kvarn layer that kept it would leave the mask
    wider than an fp16 layer's keys."""
    from mlx_vlm.models.cache import BatchKVCache

    pads = [0, 64, 128]
    c = _filled(300, pads)
    ref = BatchKVCache(pads)
    ref.update_and_fetch(*_slab(300))
    keep = mx.array([1, 2])
    c.filter(keep)
    ref.filter(keep)
    assert c._idx == ref._idx == 236
    assert np.array_equal(np.array(c.left_padding), np.array(ref.left_padding))
    assert np.array_equal(np.array(c.offset), np.array(ref.offset))
    # Only the shadow moved: the rows stay in place, bit-exact.
    assert c.starts == [64, 128] and c.ends == [300, 300]
    _assert_shadow(c)
    q = _make_q(b=2)
    mask = c.make_mask(1, window_size=None)
    out = kvarn_attention(q, c, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_kvarn_ops
def test_filter_to_all_padding_rows_empties_the_cache():
    c = _filled(128, [0, 128])
    c.filter(mx.array([1]))
    assert c._idx == 0
    assert np.array_equal(np.array(c.left_padding), [0])
    assert c.stage_k is None


@needs_kvarn_ops
def test_filter_to_no_rows_then_extend_adopts_without_padding():
    """The spec loop's all-rows-finished adoption empties the batch and
    extends it with the injected row. The shadow watermark resets with
    the rows, so the adopted row carries no padding: a one-row batch
    decodes at the watermark's position."""
    c = _filled(300, [0, 64])
    c.filter(mx.array([], dtype=mx.int32))
    assert c._idx == 0 and c.stage_k is None
    assert c.left_padding.shape == (0,) and c.offset.shape == (0,)
    row = _filled(200, [0], seed=3)
    c.extend(row)
    assert c._idx == 200 and c._pads == [0]
    assert c.starts == [0] and c.ends == [200]
    _assert_shadow(c)
    ref = _filled(200, [0], seed=3)
    assert np.array_equal(np.array(c.materialize()[0]), np.array(ref.materialize()[0]))
    q = _make_q(b=1)
    mask = c.make_mask(1, window_size=None)
    _assert_close(kvarn_attention(q, c, SCALE, mask), _ref_batch_decode(q, c))


@needs_kvarn_ops
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


@needs_kvarn_ops
def test_extend_keeps_rows_in_place():
    """A shorter batch joins without realignment: its rows keep their
    physical starts and ends (no re-quantization), and only the shadow
    right-aligns to the longer side the way BatchKVCache does."""
    a = _filled(640, [0, 32], seed=0)
    b = _filled(400, [8], seed=5)
    b_raw = [np.array(x) for x in b._raw_rows()]
    b_mat = [np.array(x) for x in b.materialize()]
    a.extend(b)
    assert a._idx == 640
    assert np.array_equal(np.array(a.left_padding), [0, 32, 248])
    assert a.starts == [0, 32, 8] and a.ends == [640, 640, 400]
    _assert_shadow(a)
    rk, rv = a._raw_row(2)
    assert np.array_equal(np.array(rk), b_raw[0])
    assert np.array_equal(np.array(rv), b_raw[1])
    mk, mv = a.materialize()
    assert np.array_equal(np.array(mk[2:3, :, :400]), b_mat[0])
    assert np.array_equal(np.array(mv[2:3, :, :400]), b_mat[1])
    assert not np.array(a._raw_rows()[0][2:3, :, 400:]).any()
    # Decode over the merged batch stays reference-correct.
    q = _make_q()
    mask = a.make_mask(1, window_size=None)
    out = kvarn_attention(q, a, SCALE, mask)
    _assert_close(out, _ref_batch_decode(q, a))


@needs_kvarn_ops
def test_extend_mismatches_raise():
    a = _filled(300, [0])
    with pytest.raises(ValueError, match="mismatched"):
        a.extend(BatchKVarNKVCache([0], k_bits=4, tail_tokens=256))
    with pytest.raises(ValueError, match="empty batch"):
        a.extend(BatchKVarNKVCache([0], tail_tokens=256))


@needs_kvarn_ops
def test_state_round_trip():
    c = _filled(300, [0, 64, 128])
    fresh = BatchKVarNKVCache([0])
    fresh.state = tuple(mx.contiguous(x) for x in c.state)
    fresh.meta_state = c.meta_state
    assert fresh._idx == 300 and fresh.n_sealed == c.n_sealed
    for x, y in zip(fresh.materialize(), c.materialize(), strict=True):
        assert np.array_equal(np.array(x), np.array(y))


@needs_kvarn_ops
def test_extract_and_merge_carry_content_only():
    c = _filled(200, [0])
    row = c.extract(0)
    assert row.nbytes < c.nbytes / 3
    back = BatchKVarNKVCache.merge([row])
    assert back.nbytes <= row.nbytes
    k, v = _slab(700, b=1, seed=5)
    back.update_and_fetch(k, v)
    c.update_and_fetch(k, v)
    for x, y in zip(back.materialize(), c.materialize(), strict=True):
        assert np.array_equal(np.array(x), np.array(y))
    assert back.tail_len == c.tail_len
    for x, y in zip(back.tail_slices(back.tail_len), c.tail_slices(c.tail_len)):
        assert np.array_equal(np.array(x), np.array(y))


@needs_kvarn_ops
def test_finalize_applies_per_row_trims():
    """Right padding is the ragged step of a speculative rollback: each
    row drops its own count, the shadow pad grows by it, and the row
    equals a cache that never held those tokens."""
    k, v = _slab(300, b=2)
    c = BatchKVarNKVCache([0, 0], tail_tokens=256)
    c.update_and_fetch(k, v)
    c.prepare(right_padding=[0, 4])
    c.finalize()
    assert c.ends == [300, 296] and c._idx == 300
    assert np.array_equal(np.array(c.left_padding), [0, 4])
    _assert_shadow(c)
    ref = BatchKVarNKVCache([0], tail_tokens=256)
    ref.update_and_fetch(k[1:2, :, :296], v[1:2, :, :296])
    _assert_row_equals(c, 1, ref)
    # more than a row holds, or a wrong count, is refused before any trim
    c.prepare(right_padding=[0, 400])
    with pytest.raises(RuntimeError, match="right padding"):
        c.finalize()
    c.prepare(right_padding=[1])
    with pytest.raises(ValueError, match="right padding"):
        c.finalize()
    assert c.ends == [300, 296]


# -- ragged rollback --------------------------------------------------------


@needs_kvarn_ops
def test_ragged_rounds_replay_bit_exactly():
    """Three rows reject different counts every round across seal
    boundaries and tail normalizations: after each rollback every row
    equals a one-row cache with the same pad fed only the tokens it kept,
    and the shadow invariant holds throughout."""
    pads = [0, 5, 130]
    k, v = _slab(300, seed=8)
    c = BatchKVarNKVCache(pads, tail_tokens=256)
    c.update_and_fetch(k, v)
    refs = []
    for b, p in enumerate(pads):
        r = BatchKVarNKVCache([p], tail_tokens=256)
        r.update_and_fetch(k[b : b + 1], v[b : b + 1])
        refs.append(r)
    kept = _ragged_rounds(c, refs, rounds=163)
    assert c.ends == [300 + n for n in kept]
    assert len(set(c.ends)) == 3
    assert min(c.n_sealed_row(b) for b in range(3)) > 2
    # the ring wrapped for at least one row (per-row normalization)
    assert max(c.tail_ends) <= c.tail_k.shape[2]
    for b in range(3):
        _assert_row_equals(c, b, refs[b])
    # and the rows still decode against the per-row reference
    q = _make_q()
    out = kvarn_attention(q, c, SCALE, None)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_kvarn_ops
def test_trim_refuses_past_the_shortest_row():
    c = _ragged_cache()
    assert c.ends == [600, 570, 300]
    # row 2 holds 200 keys past its start of 100
    assert c._can_trim(200) and not c._can_trim(201)
    assert c.trim(201) == 0 and c.ends == [600, 570, 300]
    assert c.trim(200) == 200 and c.ends == [400, 370, 100]
    _assert_shadow(c)


@needs_kvarn_ops
def test_wide_trim_tops_up_the_tail_from_the_body():
    """A trim wider than the ring's slack (never a verify round) rewinds
    a row's window past its history: the missing rows come back from the
    body, so the tail leg's window is whole and the recovered row equals
    a replay of the tokens kept (the topped-up rows are the body's own
    values, one rotation round trip)."""
    k, v = _slab(700, b=2, seed=31)
    c = BatchKVarNKVCache([0, 0], tail_tokens=256)
    c.update_and_fetch(k, v)
    ring = c.tail_k.shape[2]
    drop = ring - 128
    c.prepare(right_padding=[0, drop])
    c.finalize()
    assert c.ends == [700, 700 - drop]
    assert c.tail_ends[1] == 256
    ref = BatchKVarNKVCache([0], tail_tokens=256)
    ref.update_and_fetch(k[1:2, :, : 700 - drop], v[1:2, :, : 700 - drop])
    for x, y in zip(c._materialize_row(1), ref._materialize_row(0), strict=True):
        assert np.array_equal(np.array(x), np.array(y))
    got = np.array(c._raw_row(1)[0].astype(mx.float32))
    want = np.array(ref._raw_row(0)[0].astype(mx.float32))
    assert np.abs(got - want).max() < 0.35
    q = _make_q(b=2, qL=2)
    c.update_and_fetch(*_slab(2, b=2, seed=32))
    out = kvarn_attention(q, c, SCALE, None)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_kvarn_ops
def test_width_gate_cycle_replays():
    """A batch over the width cap decodes plain (one-token ragged appends),
    drains to one row through filter, then a capture round and a
    speculative rollback run on the survivor: bit-exact against a one-row
    replay at every step."""
    pads = [0, 40]
    k, v = _slab(300, b=2, seed=4)
    c = BatchKVarNKVCache(pads, tail_tokens=256)
    c.update_and_fetch(k, v)
    ref = BatchKVarNKVCache([40], tail_tokens=256)
    ref.update_and_fetch(k[1:2], v[1:2])
    rng = np.random.default_rng(6)
    for _ in range(70):
        tk = mx.array(rng.standard_normal((2, H, 1, D)).astype(np.float16))
        tv = mx.array(rng.standard_normal((2, H, 1, D)).astype(np.float16))
        c.update_and_fetch(tk, tv)
        ref.update_and_fetch(tk[1:2], tv[1:2])
    c.filter(mx.array([1]))
    assert c.starts == [40] and c.ends == [370]
    _assert_shadow(c)
    _assert_row_equals(c, 0, ref)
    for r in range(12):
        keep = [r % 4 + 1]
        bk = mx.array(rng.standard_normal((1, H, 4, D)).astype(np.float16))
        bv = mx.array(rng.standard_normal((1, H, 4, D)).astype(np.float16))
        _spec_round(c, bk, bv, keep)
        ref.update_and_fetch(bk[:, :, : keep[0]], bv[:, :, : keep[0]])
        _assert_shadow(c)
    _assert_row_equals(c, 0, ref)


@needs_kvarn_ops
def test_filter_keeps_ragged_rows_in_place():
    c = _ragged_cache()
    before = [tuple(np.array(x) for x in c._raw_row(b)) for b in range(3)]
    c.filter(mx.array([2, 0]))
    assert c.starts == [100, 0] and c.ends == [300, 600]
    _assert_shadow(c)
    for i, b in enumerate((2, 0)):
        for x, y in zip(c._raw_row(i), before[b], strict=True):
            assert np.array_equal(np.array(x), y)


@needs_kvarn_ops
def test_tail_normalizes_per_row():
    """Rows whose tail windows fill the ring at different times normalize
    one at a time (the ragged tail path), and every row's recovered rows
    stay bit-exact against its replay."""
    pads = [0, 0]
    k, v = _slab(200, b=2, seed=12)
    c = BatchKVarNKVCache(pads, tail_tokens=128)
    c.update_and_fetch(k, v)
    refs = []
    for b in range(2):
        r = BatchKVarNKVCache([0], tail_tokens=128)
        r.update_and_fetch(k[b : b + 1], v[b : b + 1])
        refs.append(r)
    ring = c.tail_k.shape[2]
    rounds = 0
    while max(c.tail_ends) < ring + 64 and rounds < 400:
        # row 0 keeps 1 of 4, row 1 keeps 3 of 4: the windows drift apart
        rng = np.random.default_rng(100 + rounds)
        bk = mx.array(rng.standard_normal((2, H, 4, D)).astype(np.float16))
        bv = mx.array(rng.standard_normal((2, H, 4, D)).astype(np.float16))
        _spec_round(c, bk, bv, [1, 3])
        refs[0].update_and_fetch(bk[0:1, :, :1], bv[0:1, :, :1])
        refs[1].update_and_fetch(bk[1:2, :, :3], bv[1:2, :, :3])
        rounds += 1
        assert len(set(c.tail_ends)) == 2
        for b in range(2):
            assert c.tail_ends[b] >= min(c.ends[b], c.tail_cap)
    for b in range(2):
        _assert_row_equals(c, b, refs[b])


# -- serialization of ragged rows -------------------------------------------


@needs_kvarn_ops
def test_state_round_trip_ragged():
    c = _ragged_cache()
    fresh = BatchKVarNKVCache([0])
    fresh.meta_state = c.meta_state
    fresh.state = tuple(mx.contiguous(x) for x in c.state)
    assert (fresh.starts, fresh.ends, fresh.tail_ends) == (c.starts, c.ends, c.tail_ends)
    assert fresh.horizon_valid == c.horizon_valid and fresh._idx == c._idx
    assert fresh.left_padding.tolist() == c.left_padding.tolist()
    _assert_shadow(fresh)
    for x, y in zip(fresh._raw_rows(), c._raw_rows(), strict=True):
        assert np.array_equal(np.array(x), np.array(y))


def test_state_refuses_an_older_batch_layout():
    c = BatchKVarNKVCache([0, 3])
    meta = list(c.meta_state)
    meta[1] = "1"
    fresh = BatchKVarNKVCache([0])
    with pytest.raises(ValueError, match="batch cache state version"):
        fresh.meta_state = tuple(meta)


# -- batched verify on the kernel route ------------------------------------


@needs_row_ends
@pytest.mark.parametrize("qL", [1, 2, 3, 4])
def test_batch_verify_matches_reference(qL):
    """Ragged rows at verify width on the kernel route: a full body, a
    body shorter than two groups, and a row inside its tail whose body
    leg is empty and must carry no weight in the merge."""
    c = _ragged_cache()
    k, v = _slab(qL, seed=21)
    c.update_and_fetch(k, v)
    q = _make_q(qL=qL, seed=22)
    out = kvarn_attention(q, c, SCALE, None)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_row_ends
@pytest.mark.parametrize("qL", [1, 4])
def test_batch_verify_no_tail(qL):
    # tail_cap 0: the body leg alone, under the per-row causal clamp.
    c = _ragged_cache(tail=0)
    k, v = _slab(qL, seed=23)
    c.update_and_fetch(k, v)
    q = _make_q(qL=qL, seed=24)
    out = kvarn_attention(q, c, SCALE, None)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_row_ends
def test_batch_verify_short_tail_keeps_the_clamp():
    # A tail shorter than the block: some queries' causal positions fall
    # inside the body, so the body leg runs clamped, not full-visibility.
    c = _ragged_cache(tail=128)
    k, v = _slab(4, seed=25)
    c.update_and_fetch(k, v)
    q = _make_q(qL=4, seed=26)
    out = kvarn_attention(q, c, SCALE, None)
    _assert_close(out, _ref_batch_decode(q, c))


@needs_kvarn_ops
@pytest.mark.parametrize("qL", [1, 4, 5])
def test_batch_verify_materialize_path(qL, monkeypatch):
    """The fallback (the kill switch, or a block wider than the kernels
    take) attends the same values under the per-row mask."""
    from gmlx.cache import kvarn_sdpa

    if qL <= 4:
        monkeypatch.setattr(kvarn_sdpa, "_sdpa_env", False)
    monkeypatch.setattr(kvarn_sdpa, "_decode_batch",
                        lambda *a, **k: pytest.fail("fused route taken"))
    c = _ragged_cache()
    k, v = _slab(qL, seed=27)
    c.update_and_fetch(k, v)
    q = _make_q(qL=qL, seed=28)
    out = kvarn_attention(q, c, SCALE, None)
    _assert_close(out, _ref_batch_decode(q, c), atol=2e-2)


@needs_kvarn_ops
def test_batch_route_declines_without_row_ends(monkeypatch):
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_row_ends_result", (False,))
    monkeypatch.setattr(kvarn_sdpa, "_decode_batch",
                        lambda *a, **k: pytest.fail("fused route taken"))
    c = _ragged_cache()
    q = _make_q()
    out = kvarn_attention(q, c, SCALE, None)
    _assert_close(out, _ref_batch_decode(q, c), atol=2e-2)


@needs_row_ends
@pytest.mark.parametrize("d", [128, 256])
def test_qwen35_arm_routes_fused_at_verify_width(d, monkeypatch):
    """The owned qwen3.5 dispatch at verify width hands the batch cache's
    own geometry to the kernel route; the sliced stack mask it carries
    is not consulted. The failure mode is a throughput cliff with correct
    output, so the route is asserted, not just the values."""
    pytest.importorskip("mlx_vlm.models.qwen3_5")
    from gmlx.cache import kvarn_sdpa
    from gmlx.models.qwen35.attn import _kvarn_attention

    c = _ragged_cache(d=d)
    mask = c.make_mask(4, window_size=None)[..., : c._idx + 4]
    k, v = _slab(4, seed=29, d=d)
    c.update_and_fetch(k, v)
    calls = []
    fused = kvarn_sdpa._decode_batch

    def _spy(*a, **kw):
        calls.append(a[0].shape[2])
        return fused(*a, **kw)

    monkeypatch.setattr(kvarn_sdpa, "_decode_batch", _spy)
    monkeypatch.setattr(kvarn_sdpa, "_prefill",
                        lambda *a, **k: pytest.fail("materialized"))
    q = _make_q(qL=4, seed=30, d=d)
    out = _kvarn_attention(q, cache=c, scale=d**-0.5, mask=mask)
    assert calls == [4]
    _assert_close(out, _ref_batch_decode(q, c))
