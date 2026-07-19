"""Quantized at-rest pooled KV storage (--kv-bits on deepseek4).

The growing compressor pools pack to affine-quantized triples on append;
fetches dequantize (full pool for prefill/compressed reads, gathered top-k
rows for sparse decode). Trim/undo are watermark moves, so replay must be
bit-identical between a rolled-back cache and a control that only ever saw
the confirmed tokens -- same contract as the fp16 tests in
test_deepseek_v4_mtp.py. Real-GGUF behavior is gated separately (chat/run
--kv-bits smoke on the V4 model)."""

import mlx.core as mx
import pytest

from gmlx.deepseek_v4_cache import PoolingCache
from gmlx.generation import quantize_pooled_caches

D = 64  # pooled row width; mx.quantize needs group >= 32 dividing D


def _qcache(ratio=4, bits=8, group=32):
    c = PoolingCache(ratio)
    c.quantize_storage(group_size=group, bits=bits)
    return c


def test_pooled_roundtrip_tolerance():
    mx.random.seed(0)
    rows = mx.random.normal((1, 7, D)).astype(mx.float16)
    ref = PoolingCache(4)
    q = _qcache()
    ref.update_and_fetch(rows)
    q.update_and_fetch(rows)
    row = mx.random.normal((1, 1, D)).astype(mx.float16)
    for c in (ref, q):
        c.update_and_fetch(row)
    assert q.size() == ref.size() == 8
    err = mx.abs(q.pooled - ref.pooled).max().item()
    assert err < 0.05, f"8-bit pooled roundtrip error {err}"


def test_gather_matches_dense_dequant():
    from gmlx.deepseek_v4_model import _sparse_topk_gather

    mx.random.seed(1)
    q = _qcache()
    q.update_and_fetch(mx.random.normal((1, 30, D)).astype(mx.float16))
    topk = mx.random.randint(0, 30, (1, 2, 5))
    gathered = q.gather_pooled(topk)
    dense = _sparse_topk_gather(q.pooled, topk, 2, D)
    assert gathered.shape == dense.shape == (1, 1, 2, 5, D)
    assert mx.array_equal(gathered, dense)


def _pool_px(r_kv, ratio):
    B, usable, d = r_kv.shape
    return r_kv.reshape(B, usable // ratio, ratio, d).mean(axis=2)


@pytest.mark.parametrize("pre_count,n_trim", [(3, 1), (2, 2), (3, 2)])
def test_quantized_pool_trim_replay(pre_count, n_trim):
    # Mirror of test_pooling_cache_trim_window_recompletion with packed
    # storage on both arms: rollback + confirmed replay must be bit-equal
    # to a control that only saw the confirmed tokens.
    mx.random.seed(13)
    ratio, G = 4, 2

    pre = [mx.random.normal((1, 1, D)).astype(mx.float16) for _ in range(pre_count)]
    pre_g = [mx.random.normal((1, 1, G)) for _ in range(pre_count)]
    upd = mx.random.normal((1, 3, D)).astype(mx.float16)
    upd_g = mx.random.normal((1, 3, G))

    test_cache = _qcache(ratio)
    for t, g in zip(pre, pre_g):
        r_kv, _, _ = test_cache.accumulate_windows(t, g, 0)
        assert r_kv.shape[1] == 0
    r_kv, _, _ = test_cache.accumulate_windows(upd, upd_g, 0)
    if r_kv.shape[1] > 0:
        test_cache.update_and_fetch(_pool_px(r_kv, ratio))
    assert test_cache._can_trim(n_trim)
    assert test_cache.trim(n_trim) == n_trim

    control = _qcache(ratio)
    k = 3 - n_trim
    for i in range(pre_count + k):
        t = pre[i] if i < pre_count else upd[:, i - pre_count : i - pre_count + 1]
        g = (
            pre_g[i]
            if i < pre_count
            else upd_g[:, i - pre_count : i - pre_count + 1]
        )
        r_kv, _, _ = control.accumulate_windows(t, g, 0)
        if r_kv.shape[1] > 0:
            control.update_and_fetch(_pool_px(r_kv, ratio))

    assert control.remainder == test_cache.remainder
    assert control.size() == test_cache.size()
    if control.remainder > 0:
        assert mx.array_equal(
            control.buf_kv[:, : control.remainder],
            test_cache.buf_kv[:, : test_cache.remainder],
        )
    if control.size() > 0:
        assert mx.array_equal(control.pooled, test_cache.pooled)
    else:
        assert test_cache.pooled is None


def test_quantize_pooled_caches_selects_compressor_pools():
    from mlx_lm.models.cache import CacheList, RotatingKVCache

    idx_pool = PoolingCache(4)
    idx_pool.quantizable = False
    comp_pool = PoolingCache(4)
    caches = [
        RotatingKVCache(max_size=8),
        CacheList(RotatingKVCache(max_size=8), comp_pool, idx_pool),
    ]
    assert quantize_pooled_caches(caches, 8, 32) == 1
    assert comp_pool.is_quantized
    assert not idx_pool.is_quantized


def test_quantize_pooled_caches_rejects_unknown_bits():
    assert quantize_pooled_caches([PoolingCache(4)], 16, 32) == 0


def test_incompatible_row_width_disarms(capsys):
    q = _qcache(group=32)
    q.update_and_fetch(mx.random.normal((1, 5, 16)).astype(mx.float16))
    assert not q.is_quantized  # disarmed, stayed fp16
    assert q.size() == 5
    assert "stays fp16" in capsys.readouterr().err


def test_quantize_storage_incompatible_width_stays_fp16():
    c = PoolingCache(4)
    c.update_and_fetch(mx.random.normal((1, 2, 16)).astype(mx.float16))
    c.quantize_storage(group_size=32, bits=8)  # 16-wide rows can't pack
    assert not c.is_quantized
    assert c.size() == 2
