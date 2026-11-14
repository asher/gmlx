"""Every cache growth keeps the buffer's dtype.

A step run off float32 state (decode leaves the GDN conv states float32)
hands a bfloat16 cache float32 rows. The slice write casts them to the
buffer's dtype; each growth branch must do the same, or its pad, new
buffer or concatenate widens the whole cache to float32. An empty cache
still takes its first rows' dtype. Each case fills a cache in bfloat16,
then drives one growth branch with float32 rows."""

from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx.cache import kv_cache as kc

BF, F32 = mx.bfloat16, mx.float32
D = 8
QD = 64  # the quantized caches' head width: one group


def _rows(n, dtype, b=1, d=D, seed=0):
    mx.random.seed(seed + n)
    return mx.random.normal((b, 1, n, d)).astype(dtype)


def _kv(n, dtype, b=1, d=D):
    return _rows(n, dtype, b, d, 1), _rows(n, dtype, b, d, 2)


def _parts(side):
    return tuple(side) if isinstance(side, (tuple, list)) else (side,)


def _float_dtypes(c):
    out = []
    for side in (c.keys, c.values):
        out.extend(a.dtype for a in _parts(side) if a.dtype != mx.uint32)
    return out


def _kv_cache(n=250):
    c = kc.KVCache()
    c.update_and_fetch(*_kv(n, BF))
    return c


def _batch_kv(lengths=(40, 33)):
    rows = []
    for n in lengths:
        c = kc.KVCache()
        c.update_and_fetch(*_kv(n, BF))
        rows.append(c)
    return kc.BatchKVCache.merge(rows)


def kv_cache_regrow():
    c = _kv_cache()
    c.update_and_fetch(*_kv(10, F32))
    return c


def quantized_regrow():
    c = kc.QuantizedKVCache(group_size=64, bits=8)
    c.update_and_fetch(*_kv(250, BF, d=QD))
    c.update_and_fetch(*_kv(10, F32, d=QD))
    return c


def batch_kv_regrow():
    c = _batch_kv()
    c.update_and_fetch(*_kv(1, F32, b=2))
    return c


def batch_kv_row_ends_regrow():
    c = _batch_kv()
    c._set_row_ends([40, 39])
    assert c._row_ends is not None
    c.update_and_fetch(*_kv(1, F32, b=2))
    return c


def batch_quantized_regrow():
    c = kc.BatchQuantizedKVCache([0], group_size=64, bits=8)
    c.update_and_fetch(*_kv(250, BF, d=QD))
    c.update_and_fetch(*_kv(10, F32, d=QD))
    return c


def rotating_step_regrow():
    c = kc.RotatingKVCache(max_size=512)
    c.update_and_fetch(*_kv(256, BF))
    c.update_and_fetch(*_kv(1, F32))
    return c


def rotating_concat():
    c = kc.RotatingKVCache(max_size=64)
    c.update_and_fetch(*_kv(16, BF))
    c.update_and_fetch(*_kv(8, F32))
    return c


def chunked_regrow():
    c = kc.ChunkedKVCache(chunk_size=4096)
    c.update_and_fetch(*_kv(250, BF))
    c.update_and_fetch(*_kv(10, F32))
    return c


def batch_rotating_step_regrow():
    c = kc.BatchRotatingKVCache(512, [0])
    c.update_and_fetch(*_kv(256, BF))
    c.update_and_fetch(*_kv(1, F32))
    return c


def batch_rotating_concat():
    c = kc.BatchRotatingKVCache(64, [0])
    c.update_and_fetch(*_kv(16, BF))
    c.update_and_fetch(*_kv(8, F32))
    return c


def buffered_rotating_regrow():
    c = kc.BufferedRotatingKVCache(max_size=64, buffer_size=8)
    c.update_and_fetch(*_kv(16, BF))
    cap = c.keys.shape[2]
    c.update_and_fetch(*_kv(cap, F32))
    assert c.keys.shape[2] > cap
    return c


def concatenate_append():
    c = kc.ConcatenateKVCache()
    c.update_and_fetch(*_kv(4, BF))
    c.update_and_fetch(*_kv(2, F32))
    return c


def simple_append():
    c = kc.SimpleKVCache()
    c.update_and_fetch(*_kv(4, BF))
    c.update_and_fetch(*_kv(2, F32))
    return c


def static_prefix_regrow():
    c = kc.StaticPrefixKVCache(max_size=16, step=16)
    c.update_and_fetch(*_kv(8, BF))
    c.update_and_fetch(*_kv(16, F32))
    assert c.keys.shape[2] > 16
    return c


def batch_kv_extend():
    c = _batch_kv((8,))
    o = kc.KVCache()
    o.update_and_fetch(*_kv(8, F32))
    c.extend(kc.BatchKVCache.merge([o]))
    return c


def batch_kv_extend_empty_side():
    c = kc.BatchKVCache([0])
    c.extend(_batch_kv((8,)))
    return c


def batch_rotating_extend():
    c = kc.BatchRotatingKVCache(64, [0])
    c.update_and_fetch(*_kv(8, BF))
    o = kc.BatchRotatingKVCache(64, [0])
    o.update_and_fetch(*_kv(8, F32))
    c.extend(o)
    return c


def batch_rotating_extend_empty_side():
    c = kc.BatchRotatingKVCache(64, [0])
    o = kc.BatchRotatingKVCache(64, [0])
    o.update_and_fetch(*_kv(8, BF))
    c.extend(o)
    return c


def batch_quantized_extend():
    c = kc.BatchQuantizedKVCache([0], group_size=64, bits=8)
    c.update_and_fetch(*_kv(8, BF, d=QD))
    o = kc.BatchQuantizedKVCache([0], group_size=64, bits=8)
    o.update_and_fetch(*_kv(8, F32, d=QD))
    c.extend(o)
    return c


_KV_CASES = [
    kv_cache_regrow, quantized_regrow, batch_kv_regrow,
    batch_kv_row_ends_regrow, batch_quantized_regrow, rotating_step_regrow,
    rotating_concat, chunked_regrow, batch_rotating_step_regrow,
    batch_rotating_concat, buffered_rotating_regrow, concatenate_append,
    simple_append, static_prefix_regrow, batch_kv_extend,
    batch_kv_extend_empty_side, batch_rotating_extend,
    batch_rotating_extend_empty_side, batch_quantized_extend,
]


@pytest.mark.parametrize("case", _KV_CASES, ids=lambda f: f.__name__)
def test_growth_keeps_the_buffer_dtype(case):
    c = case()
    mx.eval([a for side in (c.keys, c.values) for a in _parts(side)])
    assert _float_dtypes(c) and set(_float_dtypes(c)) == {BF}


def test_an_empty_cache_takes_its_first_rows_dtype():
    c = kc.KVCache()
    c.update_and_fetch(*_kv(4, F32))
    assert c.keys.dtype == F32
    c = kc.RotatingKVCache(max_size=64)
    c.update_and_fetch(*_kv(1, F32))
    assert c.keys.dtype == F32


def test_regrown_rows_land_cast():
    """The rows that forced the growth are written, in the base dtype."""
    c = _kv_cache()
    k, v = _kv(10, F32)
    out_k, _ = c.update_and_fetch(k, v)
    assert mx.array_equal(out_k[..., 250:, :], k.astype(BF)).item()


def _pooling_px(n, dtype):
    mx.random.seed(n)
    return mx.random.normal((1, n, D)).astype(dtype)


def test_pooling_append_keeps_the_dtype():
    c = kc.PoolingCache(ratio=4)
    c.update_and_fetch(_pooling_px(2, BF))
    out = c.update_and_fetch(_pooling_px(1, F32))
    assert out.shape[1] == 3 and out.dtype == BF


def test_batch_pooling_regrow_keeps_the_dtype():
    c = kc.BatchPoolingCache(ratio=4, left_padding=[0])
    c._processed, c.remainder, c._pool_lengths = [8], [0], [0]
    c.update_and_fetch(_pooling_px(2, BF))
    c._processed = [12]
    out = c.update_and_fetch(_pooling_px(1, F32))
    assert out.shape[1] == 3 and out.dtype == BF


def test_batch_pooling_extend_keeps_the_dtype():
    def one(dtype):
        c = kc.BatchPoolingCache(ratio=4, left_padding=[0])
        c._processed, c.remainder, c._pool_lengths = [8], [0], [0]
        c.update_and_fetch(_pooling_px(2, dtype))
        c.buf_kv = mx.zeros((1, 4, D), dtype)
        c.buf_gate = mx.zeros((1, 4, D), dtype)
        return c

    c = one(BF)
    c.extend(one(F32))
    assert (c.pooled.dtype, c.buf_kv.dtype, c.buf_gate.dtype) == (BF,) * 3


# Model-owned caches with their own growth.

def _pool(n, dtype, d=QD):
    mx.random.seed(n)
    return mx.random.normal((1, n, d)).astype(dtype)


def test_ds4_pool_regrow_keeps_the_dtype():
    from gmlx.models.deepseek_v4.cache import DS4PoolingCache

    c = DS4PoolingCache(ratio=4)
    step = c.POOL_STEP
    c.append_pooled(_pool(step, BF))
    c.append_pooled(_pool(1, F32))
    assert c._pbuf.shape[1] > step and c._pbuf.dtype == BF


def test_ds4_quantized_pool_regrow_keeps_the_dtype():
    from gmlx.models.deepseek_v4.cache import DS4PoolingCache

    c = DS4PoolingCache(ratio=4)
    c.quantize_storage(group_size=64, bits=8)
    step = c.POOL_STEP
    c.append_pooled(_pool(step, BF))
    c.append_pooled(_pool(1, F32))
    assert c._pbuf[0].shape[1] > step
    assert (c._pbuf[1].dtype, c._pbuf[2].dtype) == (BF, BF)


def test_ds4_fp4_pool_unpacks_in_its_first_rows_dtype():
    import mlx_kquant as kq
    from gmlx.models.deepseek_v4.cache import DS4PoolingCache

    if not hasattr(kq, "latent_fp4_pack"):
        pytest.skip("no latent_fp4_pack in this mlx_kquant")
    c = DS4PoolingCache(ratio=4)
    c.pack_fp4(BF)
    c.append_pooled(_pool(4, BF, d=16))
    c.append_pooled(_pool(1, F32, d=16))
    assert c._fp4_dtype == BF


def _ds4_batch(dtype, rows=1):
    from gmlx.models.deepseek_v4.cache import DS4BatchPoolingCache

    c = DS4BatchPoolingCache(4, [0] * rows)
    c._processed, c.remainder, c._pool_lengths = [8] * rows, [0] * rows, [0] * rows
    mx.random.seed(rows)
    c.update_and_fetch(mx.random.normal((rows, 2, D)).astype(dtype))
    return c


def test_ds4_batch_pool_regrow_keeps_the_dtype():
    c = _ds4_batch(BF)
    c._processed = [12]
    out = c.update_and_fetch(_pool(1, F32, d=D))
    assert out.shape[1] == 3 and out.dtype == BF


def test_ds4_batch_lookback_keeps_the_dtype():
    """One row completes a window off float32 rows, the other keeps its
    lookback window: both stay in the window's dtype."""
    from gmlx.models.deepseek_v4.cache import DS4BatchPoolingCache

    c = DS4BatchPoolingCache(4, [0, 0])
    c.buf_kv = mx.zeros((2, 4, D), BF)
    c.buf_gate = mx.zeros((2, 4, D), BF)
    c.remainder = [3, 0]
    c._prev_kv = mx.zeros((2, 4, D), BF)
    c._prev_gate = mx.full((2, 4, D), -mx.inf, dtype=BF)
    kv = mx.random.normal((2, 1, D))
    c.accumulate_windows(kv, mx.random.normal((2, 1, D)), 4)
    assert (c._prev_kv.dtype, c._prev_gate.dtype) == (BF, BF)


def test_ds4_batch_extend_keeps_the_dtype():
    c = _ds4_batch(BF)
    c.buf_kv, c.buf_gate = mx.zeros((1, 4, D), BF), mx.zeros((1, 4, D), BF)
    c._prev_kv, c._prev_gate = mx.zeros((1, 4, D), BF), mx.zeros((1, 4, D), BF)
    o = _ds4_batch(F32)
    o.buf_kv, o.buf_gate = mx.zeros((1, 4, D)), mx.zeros((1, 4, D))
    o._prev_kv, o._prev_gate = mx.zeros((1, 4, D)), mx.zeros((1, 4, D))
    c.extend(o)
    got = (c.pooled, c.buf_kv, c.buf_gate, c._prev_kv, c._prev_gate)
    assert [a.dtype for a in got] == [BF] * 5


def test_minimax_msa_index_stream_regrow_keeps_the_dtype():
    from gmlx.models.minimax_m3 import MSAKVCache

    c = MSAKVCache()
    ik = lambda n, dt: _rows(n, dt, seed=5)  # noqa: E731
    c.update_and_fetch_msa(*_kv(250, BF), ik(250, BF))
    c.update_and_fetch_msa(*_kv(10, F32), ik(10, F32))
    assert (c.keys.dtype, c.values.dtype, c.ik.dtype) == (BF,) * 3


def test_hybrid_quant_uniform_append_keeps_the_dtype():
    from types import SimpleNamespace

    from gmlx._vendor.mlx_vlm.turboquant import HybridQuantKVCache

    spec = SimpleNamespace(scheme="uniform", bits=8, group_size=64)
    c = HybridQuantKVCache(SimpleNamespace(key=spec, value=spec))
    c.update_and_fetch(*_kv(4, BF, d=QD))
    c.update_and_fetch(*_kv(2, F32, d=QD))
    assert _float_dtypes(c) and set(_float_dtypes(c)) == {BF}


def test_vendored_hy_v4_regrow_keeps_the_dtype():
    from gmlx._vendor.mlx_vlm.models.hy_v4.cache import HyV4KVCache

    c = HyV4KVCache()
    c.update_and_fetch(*_kv(4, BF))
    cap = c.keys.shape[2]
    c.update_and_fetch(*_kv(cap, F32))
    assert c.keys.shape[2] > cap
    assert (c.keys.dtype, c.values.dtype) == (BF, BF)


@pytest.mark.parametrize("batch", [False, True])
def test_vendored_minimax_index_stream_regrow_keeps_the_dtype(batch):
    from gmlx._vendor.mlx_vlm.models.minimax_m3_vl import language as mm

    c = mm.MiniMaxM3BatchKVCache([0]) if batch else mm.MiniMaxM3KVCache()
    c.update_index_and_fetch(_rows(250, BF))
    c.update_index_and_fetch(_rows(10, F32))
    assert c.index_keys.dtype == BF
