"""KVarN SDPA route: fused decode + tail merge parity against an fp32
reference over the exact segment values, prefill materialize parity, and
dispatch hygiene (sweep install, passthrough, kill switch)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import mlx.core as mx

from gmlx.kvarn_cache import KVarNKVCache
from gmlx.kvarn_sdpa import install_kvarn_sdpa, kvarn_attention

_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="kvarn kernels are Metal-only; needs the GPU device",
)

H = 2
HQ = 8
D = 128
SCALE = D**-0.5


def _filled(n, tail=384, seed=0, d=D):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, n, d)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, H, n, d)).astype(np.float16))
    c = KVarNKVCache(tail_tokens=tail)
    c.update_and_fetch(k, v)
    return c


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


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("ql", [1, 2, 4])
def test_decode_with_tail_matches_reference(ql, d):
    cache = _filled(700, d=d)
    q = _make_q(ql, d=d)
    out = kvarn_attention(q, cache, d**-0.5, "causal" if ql > 1 else None)
    _assert_close(out, _ref_attention(q, cache, ql))


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("n", [130, 300, 700])
def test_decode_no_tail_matches_reference(n, d):
    cache = _filled(n, tail=0, d=d)
    q = _make_q(1, d=d)
    out = kvarn_attention(q, cache, d**-0.5, None)
    _assert_close(out, _ref_attention(q, cache, 1))


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256])
def test_decode_all_tail_matches_reference(d):
    # Shallow cache: the tail covers every token, no body call.
    cache = _filled(200, tail=384, d=d)
    q = _make_q(1, d=d)
    out = kvarn_attention(q, cache, d**-0.5, None)
    _assert_close(out, _ref_attention(q, cache, 1))


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256])
def test_bfloat16_query(d):
    cache = _filled(700, d=d)
    q = _make_q(1, dtype=mx.bfloat16, d=d)
    out = kvarn_attention(q, cache, d**-0.5, None)
    assert out.dtype == mx.bfloat16
    _assert_close(out, _ref_attention(q, cache, 1), atol=2e-2)


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256])
def test_prefill_matches_reference(d):
    cache = _filled(700, d=d)
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


@_NEEDS_GPU
def test_kill_switch_forces_materialize(monkeypatch):
    cache = _filled(700)
    q = _make_q(1)
    monkeypatch.setenv("GMLX_KVARN_SDPA", "0")
    from gmlx import kvarn_sdpa

    off = kvarn_attention(q, cache, SCALE, None)
    ref = kvarn_sdpa._prefill(q, cache, SCALE, None)
    assert np.array_equal(np.array(off), np.array(ref))


@_NEEDS_GPU
def test_sinks_raise_loudly():
    cache = _filled(300)
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


@_NEEDS_GPU
def test_installed_wrapper_routes_views():
    install_kvarn_sdpa()
    base = importlib.import_module("mlx_lm.models.base")
    cache = _filled(300)
    kv, vv = cache.update_and_fetch(
        mx.zeros((1, H, 1, D), mx.float16), mx.zeros((1, H, 1, D), mx.float16)
    )
    q = _make_q(1)
    out = base.scaled_dot_product_attention(
        q, kv, vv, cache=cache, scale=SCALE, mask=None
    )
    _assert_close(out, _ref_attention(q, cache, 1))
