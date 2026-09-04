"""KVarN SDPA route: fused decode + tail merge parity against an fp32
reference over the exact segment values, prefill materialize parity, and
dispatch hygiene (sweep install, passthrough, kill switch)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import mlx.core as mx

from gmlx.cache.kvarn_cache import KVarNKVCache
from gmlx.cache.kvarn_sdpa import install_kvarn_sdpa, kvarn_attention
from kvarn_testlib import D, H, filled, needs_kvarn_ops

HQ = 8
SCALE = D**-0.5


def _make_q(qL, seed=1, dtype=mx.float16, d=D):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, HQ, qL, d)).astype(np.float16)).astype(
        dtype
    )


def _ref_attention(q, cache, qL):
    """fp32 attention over the exact values the route attends: rotated
    materialized body plus rotated tail rows, causal-clamped per query."""
    import mlx_kquant as kq

    d = q.shape[-1]
    n = cache.offset
    t = min(cache.tail_len, n)
    if 0 < n - t < qL:
        t = n - qL
    mat_k, mat_v = cache.materialize()
    parts_k, parts_v = [mat_k[:, :, : n - t]], [mat_v[:, :, : n - t]]
    if t:
        tk, tv = cache.tail_slices(t)
        parts_k.append(kq.kvarn_rotate(tk))
        parts_v.append(kq.kvarn_rotate(tv))
    k = mx.concatenate(parts_k, axis=2).astype(mx.float32)
    v = mx.concatenate(parts_v, axis=2).astype(mx.float32)
    qr = kq.kvarn_rotate(q.astype(mx.float16)).astype(mx.float32)
    qg = qr.reshape(1, H, HQ // H, qL, d)
    s = (qg @ k[:, :, None].transpose(0, 1, 2, 4, 3)) * (d**-0.5)
    kpos = mx.arange(n)[None, None, None, None, :]
    qpos = (n - qL + mx.arange(qL))[None, None, None, :, None]
    s = mx.where(kpos <= qpos, s, mx.array(-np.inf, mx.float32))
    o = (mx.softmax(s, axis=-1) @ v[:, :, None]).reshape(1, HQ, qL, d)
    return kq.kvarn_rotate(o.astype(mx.float16)).astype(mx.float32)


def _assert_close(out, ref, atol=5e-3):
    d = np.abs(np.array(out.astype(mx.float32)) - np.array(ref)).max()
    assert d < atol, f"max|d|={d}"


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("ql", [1, 2, 4])
def test_decode_with_tail_matches_reference(ql, d):
    cache = filled(700, d=d)
    q = _make_q(ql, d=d)
    out = kvarn_attention(q, cache, d**-0.5, "causal" if ql > 1 else None)
    _assert_close(out, _ref_attention(q, cache, ql))


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("n", [130, 300, 700])
def test_decode_no_tail_matches_reference(n, d):
    cache = filled(n, tail=0, d=d)
    q = _make_q(1, d=d)
    out = kvarn_attention(q, cache, d**-0.5, None)
    _assert_close(out, _ref_attention(q, cache, 1))


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
def test_decode_all_tail_matches_reference(d):
    # Shallow cache: the tail covers every token, no body call.
    cache = filled(200, tail=384, d=d)
    q = _make_q(1, d=d)
    out = kvarn_attention(q, cache, d**-0.5, None)
    _assert_close(out, _ref_attention(q, cache, 1))


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
def test_bfloat16_query(d):
    cache = filled(700, d=d)
    q = _make_q(1, dtype=mx.bfloat16, d=d)
    out = kvarn_attention(q, cache, d**-0.5, None)
    assert out.dtype == mx.bfloat16
    _assert_close(out, _ref_attention(q, cache, 1), atol=2e-2)


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
def test_prefill_matches_reference(d):
    cache = filled(700, d=d)
    q = _make_q(16, d=d)
    scale = d**-0.5
    out = kvarn_attention(q, cache, scale, "causal")
    # Prefill never consults the tail: reference over the materialized cache.
    import mlx_kquant as kq

    k, v = cache.materialize()
    k, v = k.astype(mx.float32), v.astype(mx.float32)
    qr = kq.kvarn_rotate(q).astype(mx.float32)
    qg = qr.reshape(1, H, HQ // H, 16, d)
    s = (qg @ k[:, :, None].transpose(0, 1, 2, 4, 3)) * scale
    n = cache.offset
    kpos = mx.arange(n)[None, None, None, None, :]
    qpos = (n - 16 + mx.arange(16))[None, None, None, :, None]
    s = mx.where(kpos <= qpos, s, mx.array(-np.inf, mx.float32))
    o = (mx.softmax(s, axis=-1) @ v[:, :, None]).reshape(1, HQ, 16, d)
    ref = kq.kvarn_rotate(o.astype(mx.float16)).astype(mx.float32)
    _assert_close(out, ref)


@needs_kvarn_ops
def test_kill_switch_forces_materialize(monkeypatch):
    cache = filled(700)
    q = _make_q(1)
    monkeypatch.setenv("GMLX_KVARN_SDPA", "0")
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_sdpa_env", None)
    off = kvarn_attention(q, cache, SCALE, None)
    ref = kvarn_sdpa._prefill(q, cache, SCALE, None)
    assert np.array_equal(np.array(off), np.array(ref))


@needs_kvarn_ops
def test_sinks_raise_loudly():
    cache = filled(300)
    q = _make_q(1)
    with pytest.raises(RuntimeError, match="sinks"):
        kvarn_attention(q, cache, SCALE, None, sinks=mx.zeros((HQ,)))


def test_install_sweeps_and_passes_through():
    llama = importlib.import_module("mlx_lm.models.llama")
    base = importlib.import_module("mlx_lm.models.base")
    n = install_kvarn_sdpa()
    assert n >= 2
    for mod in (llama, base):
        assert getattr(mod.scaled_dot_product_attention, "_gmlx_kvarn", False)
    # Idempotent: a second sweep leaves the same wrappers in place.
    fn = llama.scaled_dot_product_attention
    install_kvarn_sdpa()
    assert llama.scaled_dot_product_attention is fn

    # Non-kvarn calls pass through to the original bit-for-bit.
    from mlx_lm.models.cache import KVCache

    rng = np.random.default_rng(3)
    q = mx.array(rng.standard_normal((1, 4, 1, 64)).astype(np.float16))
    k = mx.array(rng.standard_normal((1, 2, 8, 64)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, 2, 8, 64)).astype(np.float16))
    wrapped = fn(q, k, v, cache=KVCache(), scale=1.0, mask=None)
    orig = fn._gmlx_orig(q, k, v, cache=KVCache(), scale=1.0, mask=None)
    assert np.array_equal(np.array(wrapped), np.array(orig))


@needs_kvarn_ops
def test_installed_wrapper_routes_views():
    install_kvarn_sdpa()
    base = importlib.import_module("mlx_lm.models.base")
    cache = filled(300)
    kv, vv = cache.update_and_fetch(
        mx.zeros((1, H, 1, D), mx.float16), mx.zeros((1, H, 1, D), mx.float16)
    )
    q = _make_q(1)
    out = base.scaled_dot_product_attention(
        q, kv, vv, cache=cache, scale=SCALE, mask=None
    )
    _assert_close(out, _ref_attention(q, cache, 1))


@needs_kvarn_ops
def test_probe_pins_the_record_layout_version(monkeypatch):
    # gmlx and mlx-kquant each carry the wire layout version; a mismatch
    # declines the scheme before any record is written.
    import mlx_kquant as kq

    from gmlx.cache import kvarn_sdpa

    assert kq.KVARN_RECORD_VERSION == KVarNKVCache.kvarn_layout_version
    assert kvarn_sdpa._probe() is None
    monkeypatch.setattr(kq, "KVARN_RECORD_VERSION", 99)
    assert "record layout 99" in kvarn_sdpa._probe()
    monkeypatch.delattr(kq, "KVARN_RECORD_VERSION")
    assert "record layout None" in kvarn_sdpa._probe()


@needs_kvarn_ops
def test_threadgroup_cap_gates_the_fused_route(monkeypatch):
    # A GPU whose pipeline cap is below the dispatch width would raise at
    # eval, so the route consults the probed cap and materializes instead.
    from gmlx.cache import kvarn_sdpa

    cache = filled(700)
    q = _make_q(4)
    calls = []
    real = kvarn_sdpa._decode
    monkeypatch.setattr(kvarn_sdpa, "_decode", lambda *a: calls.append(1) or real(*a))
    assert kvarn_sdpa._tg_threads(16, 4) == 1024
    assert kvarn_sdpa._tg_threads(HQ // H, 4) <= kvarn_sdpa._tg_limit()
    kvarn_attention(q, cache, SCALE, "causal")
    assert calls == [1]
    monkeypatch.setattr(kvarn_sdpa, "_tg_limit_cache", 32)
    capped = kvarn_attention(q, cache, SCALE, "causal")
    ref = kvarn_sdpa._prefill(q, cache, SCALE, "causal")
    assert calls == [1]
    assert np.array_equal(np.array(capped), np.array(ref))
