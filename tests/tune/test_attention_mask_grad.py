"""blocked_attention on the GPU with an additive mask that carries
parameters, as MLA models pass their positional scores: the gradient
reaches the mask and matches the unfused float32 reference."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from gmlx.tune.attention import blocked_attention

pytestmark = pytest.mark.skipif(
    os.environ.get("KQUANT_FORCE_CPU") == "1" or not mx.metal.is_available()
    or mx.default_device() != mx.gpu, reason="runs on the GPU only")


def _reference(q, k, v, mask, scale):
    rep = q.shape[1] // k.shape[1]
    k = mx.repeat(k, rep, axis=1)
    v = mx.repeat(v, rep, axis=1)
    s = (q @ k.swapaxes(-1, -2)) * scale + mask
    return mx.softmax(s, axis=-1) @ v


@pytest.mark.parametrize("mask_shape", ["full", "heads", "shared"])
@pytest.mark.parametrize("block", [16, 256])
def test_additive_mask_gets_its_gradient(mask_shape, block):
    B, Hq, Hkv, T, D = 2, 4, 2, 40, 16
    r = np.random.default_rng(0)

    def arr(*shape):
        return mx.array(r.standard_normal(shape).astype(np.float32))

    q, k, v = arr(B, Hq, T, D), arr(B, Hkv, T, D), arr(B, Hkv, T, D)
    lead = {"full": (B, Hq), "heads": (1, Hq), "shared": (B, 1)}[mask_shape]
    a, b = arr(*lead, T, 4), arr(*lead, T, 4)
    causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    w = arr(B, Hq, T, D)
    scale = 0.25

    def loss(attend, q, k, v, a, b):
        m = mx.where(causal, a @ b.swapaxes(-1, -2), mx.finfo(mx.float32).min)
        return (attend(q, k, v, m) * w).sum()

    def ours(q, k, v, m):
        return blocked_attention(q, k, v, scale=scale, mask=m, block=block)

    def ref(q, k, v, m):
        return _reference(q, k, v, m, scale)

    args = (q, k, v, a, b)
    g1 = mx.grad(lambda *x: loss(ours, *x), argnums=(0, 1, 2, 3, 4))(*args)
    g0 = mx.grad(lambda *x: loss(ref, *x), argnums=(0, 1, 2, 3, 4))(*args)
    mx.eval(g0, g1)
    for name, x0, x1 in zip(("q", "k", "v", "a", "b"), g0, g1):
        x0, x1 = np.array(x0), np.array(x1)
        top = max(float(np.abs(x0).max()), 1e-6)
        assert float(np.abs(x0 - x1).max()) < 1e-2 * top, name
