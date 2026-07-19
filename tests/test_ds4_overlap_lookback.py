"""Ratio-4 overlap compressor cross-call lookback parity.

The overlap compressor pools each window's b-half with the PREVIOUS
window's a-half. A one-shot prefill computes that linkage inside a single
call; decode and chunked prefill hand windows over one call at a time, so
PoolingCache must carry the previous completed window's raw projections
across calls. Without it every first-window-of-a-call pooled row had its
a-half lanes forced to -inf (rows diverged from the one-shot pool by ~3.0).
"""
import types

import mlx.core as mx

from gmlx.deepseek_v4_cache import BatchPoolingCache, PoolingCache
from gmlx.deepseek_v4_model import Compressor

D_MODEL = 64
HEAD = 32
# Not 1e-5: MLX dispatches f32 GEMM through the tensor cores at TF32
# precision where available, while a 1-row GEMV stays full f32, so decode
# and one-shot prefill projections legitimately differ by ~1e-3. The bug
# these tests pin (a-half lanes forced to -inf) diverges by ~3.0; 1e-2
# keeps the suite on production numerics with margin on both sides.
TOL = 1e-2


def _mk_compressor():
    cfg = types.SimpleNamespace(
        hidden_size=D_MODEL,
        qk_rope_head_dim=16,
        compress_rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        head_dim=-1,          # sidestep the fp8/fp4 QAT round-trips
        index_head_dim=-2,
    )
    return Compressor(cfg, compress_ratio=4, head_dim=HEAD)


def _x(n, seed=0, b=1):
    mx.random.seed(seed)
    return mx.random.normal((b, n, D_MODEL)).astype(mx.float32)


def _maxdiff(a, b):
    assert a.shape == b.shape, f"{a.shape} vs {b.shape}"
    return mx.abs(a - b).max().item()


def test_decode_matches_oneshot():
    comp = _mk_compressor()
    x = _x(12)
    ref = comp(x, PoolingCache(4), 0)
    c = PoolingCache(4)
    out = None
    for t in range(12):
        out = comp(x[:, t : t + 1], c, t)
    assert ref.shape == (1, 3, HEAD)
    assert _maxdiff(ref, out) < TOL


def test_chunked_prefill_matches_oneshot():
    comp = _mk_compressor()
    x = _x(13)
    ref = comp(x, PoolingCache(4), 0)
    c = PoolingCache(4)
    comp(x[:, :5], c, 0)
    out = comp(x[:, 5:], c, 5)
    assert _maxdiff(ref, out) < TOL


def test_prefill_then_decode_matches_oneshot():
    comp = _mk_compressor()
    x = _x(15)
    ref = comp(x[:, :12], PoolingCache(4), 0)
    c = PoolingCache(4)
    comp(x[:, :7], c, 0)
    out = None
    for t in range(7, 15):
        out = comp(x[:, t : t + 1], c, t)
    assert _maxdiff(ref, out) < TOL


def test_state_roundtrip_preserves_lookback():
    comp = _mk_compressor()
    x = _x(14)
    c = PoolingCache(4)
    for t in range(10):
        comp(x[:, t : t + 1], c, t)
    c2 = PoolingCache(4)
    c2.meta_state = c.meta_state
    c2.state = c.state
    out = None
    for t in range(10, 14):
        out = comp(x[:, t : t + 1], c2, t)
    ref = comp(x[:, :12], PoolingCache(4), 0)
    assert _maxdiff(ref, out) < TOL


def test_trim_restores_lookback():
    comp = _mk_compressor()
    x = _x(14)
    c = PoolingCache(4)
    for t in range(11):
        comp(x[:, t : t + 1], c, t)     # 2 windows + remainder 3
    # MTP-verify-style 2-wide update that completes window 3, then a full
    # rejection: the lookback must roll back to window 2's rows.
    comp(x[:, 11:13], c, 11)
    assert c.trim(2) == 2
    out = None
    for t in range(11, 14):
        out = comp(x[:, t : t + 1], c, t)
    ref = comp(x[:, :12], PoolingCache(4), 0)
    assert _maxdiff(ref, out) < TOL


def test_batch_matches_scalar():
    comp = _mk_compressor()
    xa = _x(18, seed=1)
    xb = _x(18, seed=2)

    ra = rb = None
    ca, cb = PoolingCache(4), PoolingCache(4)
    for t in range(14):
        ra = comp(xa[:, t : t + 1], ca, t)
        rb = comp(xb[:, t : t + 1], cb, t)

    # Run each row scalar to t=6 (past one window), merge, continue batched.
    c1, c2 = PoolingCache(4), PoolingCache(4)
    for t in range(6):
        comp(xa[:, t : t + 1], c1, t)
        comp(xb[:, t : t + 1], c2, t)
    bc = BatchPoolingCache.merge([c1, c2])
    xab = mx.concatenate([xa, xb], axis=0)
    out = None
    for t in range(6, 14):
        out = comp(xab[:, t : t + 1], bc, mx.array([t, t]))
    assert _maxdiff(out[0:1], ra) < TOL
    assert _maxdiff(out[1:2], rb) < TOL

    # And back out to scalar caches with the lookback intact.
    e0 = bc.extract(0)
    c3 = PoolingCache(4)
    c3.meta_state = e0.meta_state
    c3.state = e0.state
    out3 = None
    for t in range(14, 18):
        out3 = comp(xa[:, t : t + 1], c3, t)
    ref3 = comp(xa[:, :16], PoolingCache(4), 0)
    assert _maxdiff(ref3, out3) < TOL


def test_batch_filter_and_extend_carry_lookback():
    comp = _mk_compressor()
    xa = _x(12, seed=3)
    xb = _x(12, seed=4)
    ref_b = comp(xb, PoolingCache(4), 0)

    c1, c2 = PoolingCache(4), PoolingCache(4)
    for t in range(6):
        comp(xa[:, t : t + 1], c1, t)
        comp(xb[:, t : t + 1], c2, t)
    bc = BatchPoolingCache.merge([c1])
    bc.extend(BatchPoolingCache.merge([c2]))
    bc.filter(mx.array([1], dtype=mx.int32))   # keep only row b
    out = None
    for t in range(6, 12):
        out = comp(xb[:, t : t + 1], bc, mx.array([t]))
    assert _maxdiff(ref_b, out) < TOL
