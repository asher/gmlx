"""The sliding-window SDPA route: which calls take kq's window kernel, and
that its output matches stock attention under the cache's band mask over a
prompt that crosses the window."""

from __future__ import annotations

import mlx.core as mx
import pytest

import gmlx.cache.window_sdpa as window_sdpa
from gmlx.cache.kv_cache import BatchRotatingKVCache, RotatingKVCache

_OUT = mx.zeros((1,))
_WINDOW = 513


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _OUT


def _qkv(L, S, d=128, dtype=mx.bfloat16):
    q = mx.zeros((1, 32, L, d), dtype=dtype)
    k = mx.zeros((1, 4, S, d), dtype=dtype)
    return q, k, k


@pytest.fixture
def spies(monkeypatch):
    kernel, stock = _Spy(), _Spy()
    monkeypatch.setattr(window_sdpa, "_kq_window_kernel", lambda: kernel)
    monkeypatch.setattr(window_sdpa, "scaled_dot_product_attention", stock)
    monkeypatch.setattr(window_sdpa, "_on_gpu", lambda: True)
    return kernel, stock


@pytest.mark.parametrize(
    "cache,L,env,routed",
    [
        (RotatingKVCache(_WINDOW), 2048, None, True),
        (RotatingKVCache(_WINDOW), 64, None, True),
        (RotatingKVCache(_WINDOW), 63, None, False),
        (RotatingKVCache(_WINDOW, keep=4), 2048, None, False),
        (BatchRotatingKVCache(_WINDOW, [0]), 2048, None, False),
        (RotatingKVCache(_WINDOW), 2048, "0", False),
    ],
    ids=["chunk", "one_block", "verify_width", "keep", "batch", "switch_off"],
)
def test_route(spies, monkeypatch, cache, L, env, routed):
    kernel, stock = spies
    if env is not None:
        monkeypatch.setenv("GMLX_SWA_KQ", env)
    q, k, v = _qkv(L, L + _WINDOW - 1)
    sinks = mx.zeros((32,))
    out = window_sdpa.sliding_window_attention(
        q, k, v, cache, 0.5, "mask", _WINDOW, sinks=sinks)
    assert out is _OUT
    assert (len(kernel.calls), len(stock.calls)) == ((1, 0) if routed else (0, 1))
    if routed:
        args, kwargs = kernel.calls[0]
        assert args[3:] == (0.5, _WINDOW) and kwargs == {"sinks": sinks}
    else:
        assert stock.calls[0][1]["mask"] == "mask"


def test_window_kernel_matches_stock_over_a_rotating_cache():
    # Three chunks of a 2500-token prompt through two caches at Flint's
    # sliding-layer geometry: the routed one and stock under the band
    # mask the cache makes. The second chunk crosses the window.
    if window_sdpa._kq_window_kernel() is None:
        pytest.skip("needs an mlx_kquant build with sdpa_prefill_window on NAX")
    D, scale = 128, 128**-0.5
    sinks = mx.random.normal((32,), key=mx.random.key(1)).astype(mx.bfloat16)
    routed, stock = RotatingKVCache(_WINDOW), RotatingKVCache(_WINDOW)
    for n, L in enumerate((1024, 1024, 452)):
        keys = mx.random.split(mx.random.key(10 + n), 3)
        q = mx.random.normal((1, 32, L, D), key=keys[0]).astype(mx.bfloat16)
        k = mx.random.normal((1, 4, L, D), key=keys[1]).astype(mx.bfloat16)
        v = mx.random.normal((1, 4, L, D), key=keys[2]).astype(mx.bfloat16)
        mask = stock.make_mask(L, window_size=_WINDOW)
        sk, sv = stock.update_and_fetch(k, v)
        want = mx.fast.scaled_dot_product_attention(
            q, sk, sv, scale=scale, mask=mask, sinks=sinks)
        rk, rv = routed.update_and_fetch(k, v)
        got = window_sdpa.sliding_window_attention(
            q, rk, rv, routed, scale, None, _WINDOW, sinks=sinks)
        err = mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max()
        assert err.item() < 2e-2, f"chunk {n}: max abs err {err.item():.3e}"
