#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Fused single-token KDA decode kernel: kernel-level parity against the
eager op chain it replaces (conv, l2 norms, decay, delta rule, out-norm,
gate) and glm5_next decode parity with the route on vs off.

Tiny dims, random weights, no GGUF. Metal only (the kernel has no CPU
implementation; the eager path stays the CPU route)."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm.models.gated_delta import gated_delta_kernel

from gmlx.models import kda_fused
from gmlx.models.kimi_k3 import ShortConv1d, _kda_decay_lb
from gmlx.models.kda_fused import kda_decode_fused

from test_glm5_next import _random_model, _tiny_args

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="fused KDA decode kernel is Metal-only")

H, D, KW = 4, 32, 4
C = H * D
LB, SCALE = -5.0, D ** -0.5


def _eager(xq, xk, xv, sq, sk, sv, convs, a_raw, dt_bias, a_folded, b_logit,
           gate, state, w, B):
    qc, sq2 = convs[0](xq, sq, None, None)
    kc, sk2 = convs[1](xk, sk, None, None)
    vc, sv2 = convs[2](xv, sv, None, None)
    q = qc.reshape(B, 1, H, D)
    k = kc.reshape(B, 1, H, D)
    v = vc.reshape(B, 1, H, D)
    q = (SCALE ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = SCALE * mx.fast.rms_norm(k, None, 1e-6)
    g = _kda_decay_lb(a_folded, a_raw.reshape(B, 1, H, D), dt_bias.reshape(H, D), LB)
    beta = mx.sigmoid(b_logit.reshape(B, 1, H))
    out, st2 = gated_delta_kernel(q, k, v, g, beta, state, None)
    o = mx.fast.rms_norm(out.reshape(B, 1, H, D), w, 1e-5)
    y = (o * mx.sigmoid(gate.reshape(B, 1, H, D))).reshape(B, 1, -1)
    return y, sq2, sk2, sv2, st2


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_kernel_matches_eager_chain(B, dtype):
    mx.random.seed(3)
    convs = [ShortConv1d(C, KW) for _ in range(3)]
    for c in convs:
        c.conv.weight = (mx.random.normal((C, KW, 1)) * 0.5).astype(dtype)
    xq, xk, xv, a_raw, gate = (
        mx.random.normal((B, 1, C)).astype(dtype) for _ in range(5))
    sq, sk, sv = (mx.random.normal((B, KW - 1, C)).astype(dtype) for _ in range(3))
    dt_bias = mx.random.normal((C,)) * 0.5
    a_folded = -mx.random.uniform(low=1.0, high=4.0, shape=(H,))
    b_logit = mx.random.normal((B, 1, H)).astype(dtype)
    state = mx.random.normal((B, H, D, D)) * 0.1
    w = (1 + 0.1 * mx.random.normal((D,))).astype(dtype)
    ws = [c.conv.weight for c in convs]

    ref = _eager(xq, xk, xv, sq, sk, sv, convs, a_raw, dt_bias, a_folded,
                 b_logit, gate, state, w, B)
    got = kda_decode_fused(
        xq, xk, xv, sq, sk, sv, ws[0], ws[1], ws[2], a_raw, dt_bias, a_folded,
        b_logit, gate, state, w, lb=LB, scale=SCALE, l2_eps=1e-6, norm_eps=1e-5,
        num_heads=H, head_dim=D, conv_kernel=KW)
    mx.eval(ref, got)

    tol = 1e-4 if dtype == mx.float32 else 3e-2
    for name, a, b in zip(("y", "state"), (ref[0], ref[4]), (got[0], got[4])):
        a = np.array(a.astype(mx.float32))
        b = np.array(b.astype(mx.float32))
        scale = np.abs(a).max() + 1e-9
        assert np.abs(a - b).max() / scale < tol, name
    # shifted conv tails are a pure copy: bit-exact
    for a, b in zip(ref[1:4], got[1:4]):
        assert mx.array_equal(a, b)
    assert got[0].dtype == dtype and got[4].dtype == mx.float32


def test_glm5_decode_route_matches_eager(monkeypatch):
    # Metal kernel needs Dk % 32 == 0 (the eager fallback covers the rest).
    args = _tiny_args(kda_head_dim=32)
    model = _random_model(args, seed=5)
    toks = [3, 9, 27, 40, 11, 5, 33, 60]

    def run(enabled):
        monkeypatch.setattr(kda_fused, "_ENABLED", enabled)
        cache = model.make_cache()
        outs = [model(mx.array([[t]]), cache=cache) for t in toks]
        mx.eval(outs)
        ssm = [c[3] for c, ly in zip(cache, model.layers) if ly.is_linear]
        conv = [c[0] for c, ly in zip(cache, model.layers) if ly.is_linear]
        mx.eval(ssm, conv)
        return outs, ssm, conv

    on, ssm_on, conv_on = run(True)
    off, ssm_off, conv_off = run(False)
    for t, (a, b) in enumerate(zip(on, off)):
        a = np.array(a[0, 0], dtype=np.float32)
        b = np.array(b[0, 0], dtype=np.float32)
        np.testing.assert_allclose(a, b, rtol=2e-2, atol=2e-2)
        assert a.argmax() == b.argmax(), f"argmax diverged at step {t}"
    for a, b in zip(ssm_on, ssm_off):
        np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-3, atol=1e-4)
    for a, b in zip(conv_on, conv_off):
        assert mx.array_equal(a, b)


def test_route_gating():
    # Fused only for the plain decode shape: T == 1 with a cache, no ssm
    # mask, no per-row lengths.
    args = _tiny_args(kda_head_dim=32)
    model = _random_model(args, seed=1)
    cache = model.make_cache()
    x = mx.zeros((1, 1, args.hidden_size))
    kda_cache = cache[model.model.ssm_idx]
    assert kda_fused.fused_ok(x, None, kda_cache)
    assert not kda_fused.fused_ok(mx.zeros((1, 2, args.hidden_size)), None, kda_cache)
    assert not kda_fused.fused_ok(x, mx.ones((1, 1), dtype=mx.bool_), kda_cache)
    assert not kda_fused.fused_ok(x, None, None)
