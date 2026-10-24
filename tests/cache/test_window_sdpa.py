"""The sliding-window SDPA route: which calls window_sdpa.claim takes onto
kq's window kernel, and that its output matches stock attention under the
cache's band mask over a prompt that crosses the window."""

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
def kernel(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(window_sdpa, "_kq_window_kernel", lambda: spy)
    monkeypatch.setattr(window_sdpa, "_on_gpu", lambda: True)
    return spy


def _right_padded():
    cache = BatchRotatingKVCache(_WINDOW, [0, 0])
    cache.prepare(lengths=[2048, 1900], right_padding=[0, 148])
    return cache


@pytest.mark.parametrize(
    "cache,L,window,env,routed",
    [
        (RotatingKVCache(_WINDOW), 2048, _WINDOW, None, True),
        (RotatingKVCache(_WINDOW), 64, _WINDOW, None, True),
        (RotatingKVCache(_WINDOW), 63, _WINDOW, None, False),
        (RotatingKVCache(_WINDOW, keep=4), 2048, _WINDOW, None, False),
        (BatchRotatingKVCache(_WINDOW, [0]), 2048, _WINDOW, None, True),
        (_right_padded(), 2048, _WINDOW, None, False),
        (RotatingKVCache(_WINDOW), 2048, None, None, False),
        (RotatingKVCache(_WINDOW), 2048, _WINDOW, "0", False),
    ],
    ids=["chunk", "one_block", "verify_width", "keep", "batch",
         "batch_right_padded", "no_window", "switch_off"],
)
def test_route(kernel, monkeypatch, cache, L, window, env, routed):
    if env is not None:
        monkeypatch.setenv("GMLX_SWA_KQ", env)
    q, k, v = _qkv(L, L + _WINDOW - 1)
    sinks = mx.zeros((32,))
    out = window_sdpa.claim(q, k, v, cache, 0.5, window, sinks)
    assert out is (_OUT if routed else None)
    assert len(kernel.calls) == int(routed)
    if routed:
        args, kwargs = kernel.calls[0]
        assert args[3:] == (0.5, _WINDOW) and kwargs.pop("sinks") is sinks
        assert set(kwargs) == ({"starts"} if type(cache) is BatchRotatingKVCache else set())


def test_window_kernel_matches_stock_over_a_rotating_cache():
    # Three chunks of a 2500-token prompt through two caches at Flint's
    # sliding-layer geometry: the routed one and stock under the band
    # mask the cache makes. The second chunk crosses the window.
    if window_sdpa._kq_window_kernel() is None or not window_sdpa._on_gpu():
        pytest.skip("needs an mlx_kquant build with sdpa_prefill_window on NAX")
    routed, stock = RotatingKVCache(_WINDOW), RotatingKVCache(_WINDOW)
    for n, L in enumerate(_CHUNKS):
        got, want = _chunk_pair(routed, stock, 1, L, seed=10 + n)
        err = _max_err(got, want)
        assert err < 2e-2, f"chunk {n}: max abs err {err:.3e}"


def test_window_kernel_matches_stock_over_a_left_padded_batch():
    # Prompts of 2500, 1900 and 700 tokens, left-padded into one batch and
    # prefilled in the same three chunks. Over the chunks a row's padding is
    # inside the buffer, trimmed past, or covers the whole chunk. Rows of
    # padding are compared only for being finite.
    if window_sdpa._kq_window_kernel() is None or not window_sdpa._on_gpu():
        pytest.skip("needs an mlx_kquant build with sdpa_prefill_window on NAX")
    pads = [0, 600, 1800]
    routed = BatchRotatingKVCache(_WINDOW, pads)
    stock = BatchRotatingKVCache(_WINDOW, pads)
    col = 0
    for n, L in enumerate(_CHUNKS):
        got, want = _chunk_pair(routed, stock, len(pads), L, seed=20 + n)
        assert mx.isfinite(got).all().item(), f"chunk {n}"
        for b, pad in enumerate(pads):
            first = max(0, pad - col)
            if first < L:
                err = _max_err(got[b, :, first:], want[b, :, first:])
                assert err < 2e-2, f"chunk {n} row {b}: max abs err {err:.3e}"
        col += L


_CHUNKS = (1024, 1024, 452)
_SINKS = mx.random.normal((32,), key=mx.random.key(1)).astype(mx.bfloat16)


def _chunk_pair(routed, stock, B, L, seed, D=128):
    """One prefill chunk at Flint's sliding-layer geometry through the
    routed cache and through stock attention under the band mask the stock
    cache makes."""
    scale = D**-0.5
    keys = mx.random.split(mx.random.key(seed), 3)
    q = mx.random.normal((B, 32, L, D), key=keys[0]).astype(mx.bfloat16)
    k = mx.random.normal((B, 4, L, D), key=keys[1]).astype(mx.bfloat16)
    v = mx.random.normal((B, 4, L, D), key=keys[2]).astype(mx.bfloat16)
    mask = stock.make_mask(L, window_size=_WINDOW)
    sk, sv = stock.update_and_fetch(k, v)
    want = mx.fast.scaled_dot_product_attention(
        q, sk, sv, scale=scale, mask=mask, sinks=_SINKS)
    rk, rv = routed.update_and_fetch(k, v)
    got = window_sdpa.claim(q, rk, rv, routed, scale, _WINDOW, _SINKS)
    assert got is not None
    return got, want


def _max_err(got, want) -> float:
    return mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item()
