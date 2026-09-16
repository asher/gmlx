"""The mlx-kquant sparse decode attention kernel behind DeepSeek-V4.1
decode: the first-use probe, the mask shaping, the fallbacks, and (on a GPU
whose mlx-kquant carries the kernel) its agreement with the chain."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import mlx.core as mx
import pytest

from gmlx.models.deepseek_v4 import model as v4

_SIBLING = pathlib.Path(__file__).with_name("test_deepseek_v41_model.py")


def _model_module():
    spec = importlib.util.spec_from_file_location("_v41_model_tests", _SIBLING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chain_kernel(calls=None):
    """A stand-in kernel that runs the fp32 chain over the kernel's own
    argument shapes, so a call through it also checks the shaping."""

    def fake(q, window, pool, idx, scale, sinks=None, win_mask=None,
             sel_mask=None, splits=0):
        B, _, L, D = q.shape
        assert window.shape[:2] == (B, 1) and window.shape[3] == D
        assert pool.shape[0] == B and pool.shape[2] == D
        assert idx.shape == (B, L, idx.shape[-1])
        assert idx.dtype in (mx.int32, mx.uint32)
        if win_mask is not None:
            assert win_mask.dtype == mx.bool_
            assert win_mask.shape[1:] == (L, window.shape[2])
            win_mask = win_mask[:, None]
        if sel_mask is not None:
            assert sel_mask.dtype == mx.bool_
            assert sel_mask.shape[1:] == (L, idx.shape[-1])
            sel_mask = sel_mask[:, None]
        if calls is not None:
            calls.append((L, win_mask is not None, sel_mask is not None))
        f32 = lambda a: a.astype(mx.float32)  # noqa: E731
        g = v4._sparse_topk_gather(f32(pool), idx, L, D)
        out = v4._sparse_gathered_attention(
            f32(q), f32(window), g, win_mask, sel_mask, scale,
            None if sinks is None else f32(sinks),
        )
        return out.astype(q.dtype)

    return fake


def _install_kq(monkeypatch, fn):
    """An mlx_kquant whose sdpa_sparse_decode is ``fn``; every other
    attribute is the real package's, so the rest of the model still runs."""
    real = sys.modules.get("mlx_kquant")
    if real is None:
        try:
            import mlx_kquant as real
        except ImportError:  # pragma: no cover - bare test env
            real = None

    class _Fake:
        sdpa_sparse_decode = staticmethod(fn)

        def __getattr__(self, name):
            if real is None:
                raise AttributeError(name)
            return getattr(real, name)

    monkeypatch.setitem(sys.modules, "mlx_kquant", _Fake())


def _reset(monkeypatch):
    monkeypatch.setitem(v4._SPARSE_KERNEL, "on", None)


def test_switch_off_keeps_the_chain(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("GMLX_DS41_SPARSE_KERNEL", "0")
    assert v4._sparse_kernel_ok() is False


def test_probe_declines_a_kquant_without_the_kernel(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_kquant", types.SimpleNamespace())
    assert v4._sparse_kernel_probe() is False


def test_probe_rejects_a_wrong_kernel(monkeypatch):
    _install_kq(monkeypatch, lambda q, *a, **k: mx.zeros_like(q))
    assert v4._sparse_kernel_probe() is False


def test_probe_accepts_a_kernel_as_close_as_the_chain(monkeypatch):
    calls = []
    _install_kq(monkeypatch, _chain_kernel(calls))
    assert v4._sparse_kernel_probe() is True
    assert calls == [(2, True, True)]


def test_wrapper_declines_what_the_kernel_cannot_take(monkeypatch):
    calls = []
    _install_kq(monkeypatch, _chain_kernel(calls))
    B, H, L, D, W, P, N = 1, 2, 1, 128, 8, 12, 4
    q = mx.random.normal((B, H, L, D)).astype(mx.bfloat16)
    kv = mx.random.normal((B, 1, W, D)).astype(mx.bfloat16)
    pooled = mx.random.normal((B, P, D)).astype(mx.bfloat16)
    topk = mx.array([[[0, 3, 5, 9]]], mx.uint32)
    sinks = mx.zeros((H,), mx.bfloat16)
    args = (q, kv, pooled, topk, None, None, D ** -0.5, sinks)

    assert v4._sparse_kernel_attention(*args) is not None
    assert calls == [(1, False, False)]
    # head dim outside the kernel's set
    narrow = (q[..., :64], kv[..., :64], pooled[..., :64]) + args[3:]
    assert v4._sparse_kernel_attention(*narrow) is None
    # too many queries
    wide = (mx.broadcast_to(q, (B, H, 17, D)), kv, pooled,
            mx.broadcast_to(topk, (B, 17, N))) + args[4:]
    assert v4._sparse_kernel_attention(*wide) is None
    # an additive float mask
    add = args[:4] + (mx.zeros((L, W), mx.float32), None) + args[6:]
    assert v4._sparse_kernel_attention(*add) is None
    # a 64-bit index list
    i64 = args[:3] + (topk.astype(mx.int64),) + args[4:]
    assert v4._sparse_kernel_attention(*i64) is None
    # a bool mask whose size is not [L, W]
    odd = args[:4] + (mx.ones((L, W + 1), mx.bool_), None) + args[6:]
    assert v4._sparse_kernel_attention(*odd) is None
    assert len(calls) == 1


def test_decode_steps_take_the_kernel_and_match_the_chain(monkeypatch):
    """One-token and two-token steps on a full window and an outgrown
    pool route through the kernel (so does the 14-token prompt, which is
    inside the kernel's query width); their logits match the chain."""
    tm = _model_module()
    calls = []
    _install_kq(monkeypatch, _chain_kernel(calls))
    monkeypatch.setattr(v4, "_SPARSE_KERNEL_DIMS", (16,))
    monkeypatch.setattr(v4, "_SPARSE_KERNEL_DTYPES", (mx.float32,))
    args = tm._args()
    model = tm._randomized(tm.Model(args))
    prompt = mx.array([[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]])
    steps = [mx.array([[33]]), mx.array([[35, 37]])]

    outs = {}
    for on in (True, False):
        monkeypatch.setitem(v4._SPARSE_KERNEL, "on", on)
        cache = model.make_cache()
        mx.eval(model(prompt, cache=cache))
        n = len(calls)
        got = []
        for tok in steps:
            out = model(tok, cache=cache)
            mx.eval(out)
            got.append(out)
        outs[on] = got
        assert (len(calls) > n) == on
    widths = sorted({c[0] for c in calls})
    assert widths == [1, 2, 14]
    assert all(c[1] == (c[0] > 1) for c in calls), calls
    for a, b in zip(outs[True], outs[False]):
        assert mx.allclose(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    mx.default_device() != mx.gpu, reason="Metal-only kernel"
)
def test_probe_passes_with_the_real_kernel(monkeypatch):
    kq = pytest.importorskip("mlx_kquant")
    if not hasattr(kq, "sdpa_sparse_decode"):
        pytest.skip("mlx-kquant without sdpa_sparse_decode")
    _reset(monkeypatch)
    monkeypatch.delenv("GMLX_DS41_SPARSE_KERNEL", raising=False)
    assert v4._sparse_kernel_probe() is True
    assert v4._sparse_kernel_ok() is True
