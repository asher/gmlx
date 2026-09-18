"""FP4 at-rest storage for the DeepSeek-V4.1 latent pool.

The latent QAT lands every pooled row on the E2M1 grid with an E4M3 scale
per 16, so mlx-kquant's latent_fp4_pack stores it exactly; the sparse
kernels read the packed form directly and every other reader unpacks it
bit-for-bit. Metal-only kernels: skipped under KQUANT_FORCE_CPU.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

import mlx.core as mx
import pytest

from gmlx.models.deepseek_v4 import model as v4
from gmlx.models.deepseek_v4.cache import PackedPool, PoolingCache
from gmlx.models.deepseek_v41.model import Model, _latent_qat

D = 64


def _fp4_available() -> bool:
    if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
        return False
    try:
        import mlx_kquant as kq
    except ImportError:
        return False
    return hasattr(kq, "latent_fp4_pack")


pytestmark = pytest.mark.skipif(
    not _fp4_available(), reason="latent_fp4_pack is a Metal kernel"
)


def _grid_rows(n, d=D, seed=0):
    mx.random.seed(seed)
    return _latent_qat(mx.random.normal((1, n, d)).astype(mx.float16))


def _packed():
    c = PoolingCache(4)
    c.pack_fp4()
    return c


def _model_tests():
    p = pathlib.Path(__file__).with_name("test_deepseek_v41_model.py")
    spec = importlib.util.spec_from_file_location("_v41_model_tests_fp4", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- cache -----------------------------------------------------------------


def test_packed_rows_round_trip_exactly():
    rows = _grid_rows(37)
    ref = PoolingCache(4)
    ref.update_and_fetch(rows)
    c = _packed()
    got = c.update_and_fetch(rows)
    assert isinstance(got, PackedPool)
    assert got.shape == (1, 37, D) and got.dtype == mx.float16
    assert c.is_packed and not c.is_quantized
    assert mx.array_equal(c.pooled_rows, ref.pooled)
    assert c.nbytes < ref.nbytes / 3
    more = _grid_rows(3, seed=1)
    for x in (ref, c):
        x.update_and_fetch(more)
    assert c.size() == 40
    assert mx.array_equal(c.pooled_rows, ref.pooled)


def test_gather_matches_the_dense_gather():
    c = _packed()
    c.update_and_fetch(_grid_rows(30))
    topk = mx.random.randint(0, 30, (1, 2, 5))
    got = c.gather_pooled(topk)
    want = v4._sparse_topk_gather(c.pooled_rows, topk, 2, D)
    assert got.shape == (1, 1, 2, 5, D)
    assert mx.array_equal(got, want)


def test_pack_fp4_repacks_landed_rows_and_ignores_kv_bits():
    rows = _grid_rows(9)
    c = PoolingCache(4)
    c.update_and_fetch(rows)
    c.pack_fp4()
    assert c.is_packed and c.size() == 9
    assert mx.array_equal(c.pooled_rows, rows)
    c.quantize_storage(group_size=32, bits=8)
    assert c.is_packed and not c.is_quantized
    q = PoolingCache(4)
    q.quantize_storage(group_size=32, bits=8)
    q.pack_fp4()
    assert q.is_quantized and not q.is_packed


def test_state_and_meta_state_rebuild_a_packed_cache():
    c = _packed()
    c.update_and_fetch(_grid_rows(11))
    assert c.meta_state == (4, True, "float16")
    d = PoolingCache.from_state(c.state, c.meta_state)
    assert d.is_packed and d.size() == 11
    assert isinstance(d.pooled, PackedPool)
    assert mx.array_equal(d.pooled_rows, c.pooled_rows)


def _pool_px(r_kv, ratio):
    B, usable, d = r_kv.shape
    return r_kv.reshape(B, usable // ratio, ratio, d).mean(axis=2)


@pytest.mark.parametrize("pre_count,n_trim", [(3, 1), (2, 2), (3, 2)])
def test_packed_pool_trim_replay(pre_count, n_trim):
    """Rollback plus confirmed replay is bit-equal to a control that only
    saw the confirmed tokens (the --kv-bits contract, packed arms)."""
    mx.random.seed(13)
    ratio, G = 4, 2
    pre = [mx.random.normal((1, 1, D)).astype(mx.float16) for _ in range(pre_count)]
    pre_g = [mx.random.normal((1, 1, G)) for _ in range(pre_count)]
    upd = mx.random.normal((1, 3, D)).astype(mx.float16)
    upd_g = mx.random.normal((1, 3, G))

    test_cache = _packed()
    for t, g in zip(pre, pre_g):
        r_kv, _, _ = test_cache.accumulate_windows(t, g, 0)
        assert r_kv.shape[1] == 0
    r_kv, _, _ = test_cache.accumulate_windows(upd, upd_g, 0)
    if r_kv.shape[1] > 0:
        test_cache.update_and_fetch(_pool_px(r_kv, ratio))
    assert test_cache._can_trim(n_trim)
    assert test_cache.trim(n_trim) == n_trim

    control = _packed()
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
    if control.size() > 0:
        assert mx.array_equal(control.pooled_rows, test_cache.pooled_rows)
    else:
        assert test_cache.pooled is None


# --- kernels ---------------------------------------------------------------


def test_the_kernel_wrappers_pass_the_packed_pool(monkeypatch):
    seen = []

    def fake_decode(q, window, pool, idx, scale, sinks=None, win_mask=None,
                    sel_mask=None, splits=0, pool_scales=None):
        seen.append(("decode", pool.dtype, pool_scales is not None))
        return mx.zeros_like(q)

    def fake_prefill(q, window, pool, idx, scale, band, sinks=None,
                     sel_mask=None, pool_scales=None):
        seen.append(("prefill", pool.dtype, pool_scales is not None))
        return mx.zeros_like(q)

    monkeypatch.setitem(
        sys.modules, "mlx_kquant",
        types.SimpleNamespace(
            sdpa_sparse_decode=fake_decode, sdpa_sparse_prefill=fake_prefill
        ),
    )
    Dk = 128
    q = mx.zeros((1, 2, 3, Dk), mx.bfloat16)
    kv = mx.zeros((1, 1, 8, Dk), mx.bfloat16)
    packed = PackedPool(
        mx.zeros((1, 12, Dk // 2), mx.uint8),
        mx.zeros((1, 12, Dk // 16), mx.uint8),
        mx.bfloat16,
    )
    topk = mx.zeros((1, 3, 4), mx.uint32)
    assert v4._sparse_kernel_attention(
        q, kv, packed, topk, None, None, 1.0, None
    ) is not None
    assert v4._sparse_kernel_prefill(
        q, kv, packed, topk, None, None, 1.0, None, 4
    ) is not None
    assert seen == [("decode", mx.uint8, True), ("prefill", mx.uint8, True)]


def test_the_probe_declines_a_kernel_without_pool_scales(monkeypatch):
    import mlx_kquant as real

    def old(q, window, pool, idx, scale, sinks=None, win_mask=None,
            sel_mask=None, splits=0):
        return real.sdpa_sparse_decode(
            q, window, pool, idx, scale,
            sinks=sinks, win_mask=win_mask, sel_mask=sel_mask,
        )

    fake = types.SimpleNamespace(
        sdpa_sparse_decode=old,
        latent_fp4_pack=real.latent_fp4_pack,
        latent_fp4_unpack=real.latent_fp4_unpack,
    )
    monkeypatch.setitem(sys.modules, "mlx_kquant", fake)
    monkeypatch.setitem(v4._SPARSE_KERNEL, "on", True)
    monkeypatch.setitem(v4._SPARSE_KERNEL, "fp4", None)
    assert v4._pool_fp4_ok() is False


def test_the_probe_accepts_the_real_kernels(monkeypatch):
    for key in ("on", "wide", "fp4"):
        monkeypatch.setitem(v4._SPARSE_KERNEL, key, None)
    if not v4._sparse_kernel_ok():
        pytest.skip("the sparse decode kernel is not armed on this box")
    assert v4._pool_fp4_ok() is True


# --- model -----------------------------------------------------------------


def _pools(cache, args):
    return [list(cache[i])[1] for i in args.kv_source_layers]


def test_arming_follows_the_qat_and_the_switch(monkeypatch):
    t = _model_tests()
    args = t._args()
    model = Model(args)
    monkeypatch.setitem(v4._SPARSE_KERNEL, "fp4", True)
    pools = _pools(model.make_cache(), args)
    assert all(p.is_packed for p in pools)
    assert not any(list(cache_entry)[2].is_packed for cache_entry in
                   (model.make_cache()[i] for i in args.kv_source_layers))
    monkeypatch.setenv("GMLX_DS41_QAT", "0")
    assert not any(p.is_packed for p in _pools(model.make_cache(), args))
    monkeypatch.delenv("GMLX_DS41_QAT")
    monkeypatch.setitem(v4._SPARSE_KERNEL, "fp4", None)
    monkeypatch.setenv("GMLX_DS41_POOL_FP4", "0")
    assert v4._pool_fp4_ok() is False


def test_packed_pool_gives_the_fp16_pool_logits(monkeypatch):
    """Prefill (dense pooled path), then decode steps past index_topk (the
    gathered chain: the tiny head dim is outside the kernel's)."""
    t = _model_tests()
    args = t._args()
    mx.random.seed(0)
    model = t._randomized(Model(args))
    prompt = mx.array([[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]])
    outs = []
    for packed in (False, True):
        monkeypatch.setitem(v4._SPARSE_KERNEL, "fp4", packed)
        cache = model.make_cache()
        pools = _pools(cache, args)
        assert all(p.is_packed == packed for p in pools)
        logits = [model(prompt, cache=cache)]
        for tok in (33, 35, 37):
            logits.append(model(mx.array([[tok]]), cache=cache))
        mx.eval(*logits)
        assert all(p.size() > args.index_topk for p in pools[1:])
        outs.append(logits)
    for a, b in zip(*outs):
        assert mx.array_equal(a, b)
