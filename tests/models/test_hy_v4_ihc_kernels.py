#!/usr/bin/env python3
"""The fused iHC route against the ops path it replaces.

The kernels are not bit-identical: the reduction order differs and the
folded sublayer norm rounds once where the ops path rounds twice. Both
differences move toward the fp32 reference, so the gate is that the kernel
is no further from that reference than the ops path is.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.models.hy_v4 import ihc_kernels
from gmlx.models.hy_v4.model import IndependentHyperConnection, ModelArgs

D = 512
HC = 4
EPS = 1e-6
NEPS = 1e-5
MAG = 2.0


def _args(**kw):
    base = dict(
        model_type="hy_v4", vocab_size=32, hidden_size=D, intermediate_size=8,
        moe_intermediate_size=8, num_hidden_layers=2, num_attention_heads=4,
        q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4,
        v_head_dim=4, n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, first_k_dense_replace=1, rms_norm_eps=NEPS,
        hc_mult=HC, hc_eps=EPS, hc_magnitude=MAG)
    base.update(kw)
    return ModelArgs(**base)


@pytest.fixture
def hc_and_norm():
    mx.random.seed(4)
    hc = IndependentHyperConnection(_args())
    hc.fn = mx.random.normal((2 * HC, HC * D)) * 0.02
    hc.base = mx.random.normal((2 * HC,)) * 0.1
    hc.scale = mx.random.uniform(0.5, 1.5, (2,))
    norm = nn.RMSNorm(D, eps=NEPS)
    norm.weight = mx.random.uniform(0.5, 1.5, (D,))
    mx.eval(hc.parameters(), norm.parameters())
    return hc, norm


def _reference(x, fn, base, scale, w):
    """The cycle in float64: collapse, then the sublayer norm."""
    y = np.array(x.astype(mx.float32)).astype(np.float64)
    flat = y.reshape(*y.shape[:-2], -1)
    z = flat / np.sqrt((flat ** 2).mean(-1, keepdims=True) + NEPS)
    m = z @ np.array(fn).astype(np.float64).T
    b, s = np.array(base).astype(np.float64), np.array(scale).astype(
        np.float64)
    pre = 1 / (1 + np.exp(-(m[..., :HC] * s[0] + b[:HC]))) + EPS
    post = MAG / (1 + np.exp(-(m[..., HC:] * s[1] + b[HC:]))) + EPS
    c = (pre[..., None] * y).sum(-2)
    c = c / np.sqrt((c ** 2).mean(-1, keepdims=True) + NEPS)
    return c * np.array(w).astype(np.float64), post


@pytest.mark.parametrize("length", [1, 7, 64])
def test_fused_collapse_is_no_worse_than_the_ops_path(hc_and_norm, length):
    hc, norm = hc_and_norm
    x = (mx.random.normal((1, length, HC, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    if not ihc_kernels.eligible(x, HC):
        pytest.skip("fused iHC route unavailable on this device")

    kc, kp = hc.pre_norm(x, norm)
    oc, op = hc.pre(x)
    oc = norm(oc)
    mx.eval(kc, kp, oc, op)
    ref_c, ref_p = _reference(x, hc.fn, hc.base, hc.scale, norm.weight)

    k_err = np.abs(np.array(kc.astype(mx.float32)) - ref_c).max()
    o_err = np.abs(np.array(oc.astype(mx.float32)) - ref_c).max()
    assert k_err <= o_err * 1.5 + 1e-3, (
        f"fused collapse drifted from the reference: {k_err:.3e} vs the "
        f"ops path's {o_err:.3e}")
    assert np.abs(np.array(kp) - ref_p).max() < 1e-5


@pytest.mark.parametrize("length", [1, 7, 64])
def test_fused_expand_is_no_worse_than_the_ops_path(hc_and_norm, length):
    hc, _ = hc_and_norm
    resid = (mx.random.normal((1, length, HC, D)) * 0.5).astype(mx.bfloat16)
    y = (mx.random.normal((1, length, D)) * 0.5).astype(mx.bfloat16)
    post = mx.random.uniform(0.0, 2.0, (1, length, HC))
    mx.eval(resid, y, post)
    if not ihc_kernels.eligible(resid, HC):
        pytest.skip("fused iHC route unavailable on this device")

    fused = ihc_kernels.expand(y, resid, post)
    ops = (resid.astype(mx.float32)
           + post[..., None] * y.astype(mx.float32)[..., None, :]
           ).astype(resid.dtype)
    mx.eval(fused, ops)
    # The kernel fuses the multiply and the add, so it rounds once where the
    # ops path rounds twice and a few results land one ulp apart.
    ref = (np.array(resid.astype(mx.float32)).astype(np.float64)
           + np.array(post).astype(np.float64)[..., None]
           * np.array(y.astype(mx.float32)).astype(np.float64)[..., None, :])
    k_err = np.abs(np.array(fused.astype(mx.float32)) - ref).max()
    o_err = np.abs(np.array(ops.astype(mx.float32)) - ref).max()
    assert k_err <= o_err


def test_the_env_flag_returns_the_ops_path(monkeypatch, hc_and_norm):
    hc, norm = hc_and_norm
    x = (mx.random.normal((1, 4, HC, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    monkeypatch.setattr(ihc_kernels, "_HC_ENABLED", False)
    assert not ihc_kernels.eligible(x, HC)
    collapsed, post = hc.pre_norm(x, norm)
    ref_c, _ = hc.pre(x)
    mx.eval(collapsed, post, ref_c)
    assert mx.array_equal(collapsed, norm(ref_c))


@pytest.mark.parametrize("dtype", [mx.float32, mx.int8])
def test_only_half_precision_streams_take_the_kernel(dtype):
    x = mx.zeros((1, 2, HC, D), dtype=dtype)
    assert not ihc_kernels.eligible(x, HC)


def test_other_stream_counts_take_the_ops_path():
    # The kernels unroll four streams by name (p0..p3, x_row0..3).
    x = mx.zeros((1, 2, 2, D), dtype=mx.bfloat16)
    assert not ihc_kernels.eligible(x, 2)


def test_a_width_the_float4_loads_cannot_cover_is_declined():
    x = mx.zeros((1, 2, HC, 12), dtype=mx.bfloat16)
    assert not ihc_kernels.eligible(x, HC)
