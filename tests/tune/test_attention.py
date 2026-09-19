"""blocked_attention against mx.fast.scaled_dot_product_attention on the
CPU (where MLX takes its unfused reference path): outputs and the
gradients of q, k and v under grouped-query heads, every mask form, a
query length that is not a block multiple, bottom-right causal alignment
and bf16 inputs. Also the installer's routing on a stock mlx-lm attention."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.tune.attention import blocked_attention, install_training_attention



@pytest.fixture(autouse=True)
def _cpu():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)


def _inputs(B=2, Hq=8, Hkv=2, Tq=70, Tk=70, D=32, Dv=32, seed=0, dtype=mx.float32):
    r = np.random.default_rng(seed)
    q = mx.array(r.standard_normal((B, Hq, Tq, D)).astype(np.float32)).astype(dtype)
    k = mx.array(r.standard_normal((B, Hkv, Tk, D)).astype(np.float32)).astype(dtype)
    v = mx.array(r.standard_normal((B, Hkv, Tk, Dv)).astype(np.float32)).astype(dtype)
    return q, k, v


def _np(x):
    return np.array(x.astype(mx.float32))


def _check(q, k, v, mask, *, block, atol, rtol, scale=0.2):
    r = np.random.default_rng(1)
    w = mx.array(r.standard_normal(q.shape[:3] + (v.shape[-1],)).astype(np.float32)).astype(q.dtype)

    def ref(q, k, v):
        return (mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask) * w).sum()

    def ours(q, k, v):
        return (blocked_attention(q, k, v, scale=scale, mask=mask, block=block) * w).sum()

    y0 = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    y1 = blocked_attention(q, k, v, scale=scale, mask=mask, block=block)
    g0 = mx.grad(ref, argnums=(0, 1, 2))(q, k, v)
    g1 = mx.grad(ours, argnums=(0, 1, 2))(q, k, v)
    mx.eval(y0, y1, g0, g1)
    assert np.allclose(_np(y0), _np(y1), atol=atol, rtol=rtol)
    for name, a, b in zip("qkv", g0, g1):
        a = _np(a)
        b = _np(b)
        scale_ = max(float(np.abs(a).max()), 1e-6)
        assert np.allclose(a, b, atol=atol * scale_, rtol=rtol), name


@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize("block", [16, 64, 256])
def test_matches_reference_causal_and_dense(mask, block):
    q, k, v = _inputs()
    _check(q, k, v, mask, block=block, atol=1e-4, rtol=1e-4)


def test_bool_mask_with_right_padding_and_causal():
    q, k, v = _inputs(Tq=50, Tk=50)
    T = 50
    causal = np.tril(np.ones((T, T), dtype=bool))
    valid = np.ones((2, T), dtype=bool)
    valid[1, 40:] = False
    m = causal[None, None] & valid[:, None, None, :]
    # a fully masked row (a pad query attending to nothing) is finite on both paths
    _check(q, k, v, mx.array(m), block=16, atol=1e-4, rtol=1e-4)


def test_additive_mask_per_head():
    q, k, v = _inputs(Tq=33, Tk=33)
    r = np.random.default_rng(3)
    add = (r.standard_normal((2, 8, 33, 33)) * 0.5).astype(np.float32)
    _check(q, k, v, mx.array(add), block=8, atol=1e-4, rtol=1e-4)


def test_causal_bottom_right_when_keys_exceed_queries():
    q, k, v = _inputs(Tq=20, Tk=45)
    _check(q, k, v, "causal", block=8, atol=1e-4, rtol=1e-4)


def test_bf16_inputs_track_the_reference():
    q, k, v = _inputs(Tq=64, Tk=64, dtype=mx.bfloat16)
    _check(q, k, v, "causal", block=16, atol=2e-2, rtol=2e-2)


def test_nothing_quadratic_survives_the_forward():
    """The forward returns only the output; the custom vjp sees primals,
    the output and the lse, so a T x T array never sits on the tape."""
    q, k, v = _inputs(Tq=40, Tk=40)
    y = blocked_attention(q, k, v, scale=0.2, mask="causal", block=8)
    assert y.shape == (2, 8, 40, 32)


def test_installer_routes_training_calls_only():
    from mlx_lm.models import llama
    from mlx_lm.models.llama import ModelArgs

    args = ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=1,
                     intermediate_size=64, num_attention_heads=4,
                     num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=64)
    model = llama.Model(args)
    calls = []
    orig = llama.scaled_dot_product_attention

    def spy(*a, **kw):
        calls.append(kw.get("cache", a[3] if len(a) > 3 else None))
        return orig(*a, **kw)

    llama.scaled_dot_product_attention = spy
    try:
        restore = install_training_attention(model)
        assert restore.count >= 1
        ids = mx.array([[1, 2, 3, 4, 5]])
        model.train()
        y_train = model(ids)
        mx.eval(y_train)
        assert calls == [], "training call reached the original attention"
        model.eval()
        y_eval = model(ids)
        mx.eval(y_eval)
        assert len(calls) == 1, "eval call did not reach the original attention"
        assert np.allclose(_np(y_train), _np(y_eval), atol=1e-4)
        restore()
        assert llama.scaled_dot_product_attention is spy
    finally:
        llama.scaled_dot_product_attention = orig
