#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Fused KDA decode kernel: kernel-level parity against the eager op chain
it replaces (conv, l2 norms, decay, delta rule, out-norm, gate), the block
form against the single-token kernel chained per token, and glm5_next
decode parity with the route on vs off.

Tiny dims, random weights, no GGUF. Metal only (the kernel has no CPU
implementation; the eager path stays the CPU route)."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm.models.gated_delta import gated_delta_kernel

from gmlx.models import kda_fused
from gmlx.models.kimi_k3 import ShortConv1d, _kda_decay_lb
from gmlx.models.kda_fused import kda_decode_fused, kda_decode_fused_block

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


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("hd", [32, 128])
def test_block_matches_chained_single_token(B, hd):
    """One NT-token dispatch (state carried in registers, per-token states
    on the side output) equals NT chained single-token dispatches: the
    same f32 evaluation order, so bit-exact. Head dim 128 takes the float4
    register path, other head dims the scalar one."""
    NT, h = 3, 2
    c = h * hd
    mx.random.seed(11)
    dtype = mx.bfloat16
    ws = [(mx.random.normal((c, KW, 1)) * 0.5).astype(dtype) for _ in range(3)]
    xq, xk, xv, a_raw, gate = (
        mx.random.normal((B, NT, c)).astype(dtype) for _ in range(5))
    sq, sk, sv = (mx.random.normal((B, KW - 1, c)).astype(dtype) for _ in range(3))
    dt_bias = mx.random.normal((c,)) * 0.5
    a_folded = -mx.random.uniform(low=1.0, high=4.0, shape=(h,))
    b_logit = mx.random.normal((B, NT, h)).astype(dtype)
    state = mx.random.normal((B, h, hd, hd)) * 0.1
    w = (1 + 0.1 * mx.random.normal((hd,))).astype(dtype)
    kw = dict(lb=LB, scale=hd ** -0.5, l2_eps=1e-6, norm_eps=1e-5,
              num_heads=h, head_dim=hd, conv_kernel=KW)

    st = (sq, sk, sv, state)
    ys, mids = [], []
    for t in range(NT):
        sl = slice(t, t + 1)
        y, *st = kda_decode_fused(
            xq[:, sl], xk[:, sl], xv[:, sl], st[0], st[1], st[2], *ws,
            a_raw[:, sl], dt_bias, a_folded, b_logit[:, sl], gate[:, sl],
            st[3], w, **kw)
        ys.append(y)
        mids.append(st[3])
    got = kda_decode_fused_block(
        xq, xk, xv, sq, sk, sv, *ws, a_raw, dt_bias, a_folded, b_logit, gate,
        state, w, **kw)
    mx.eval(ys, st, got)

    assert got[0].shape == (B, NT, c) and got[5].shape == (NT - 1, B, h, hd, hd)
    assert mx.array_equal(got[0], mx.concatenate(ys, axis=1))
    for a, b in zip(st[:3], got[1:4]):
        assert mx.array_equal(a, b)
    assert mx.array_equal(got[4], st[3])
    for t in range(NT - 1):
        assert mx.array_equal(got[5][t], mids[t])


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
    # The kernel copies its inputs into the tails, so the first linear
    # layer's tails are exact. Later layers' inputs pass through the fused
    # route, which agrees with eager to rounding (one f32 ulp on an M3
    # Max), not bit-exactly.
    assert mx.array_equal(conv_on[0], conv_off[0])
    for a, b in zip(conv_on[1:], conv_off[1:]):
        np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-5, atol=1e-7)


def test_route_gating():
    # Fused only for the plain decode shape: T == 1 with a cache, no ssm
    # mask, no per-row lengths.
    args = _tiny_args(kda_head_dim=32)
    model = _random_model(args, seed=1)
    cache = model.make_cache()
    x = mx.zeros((1, 1, args.hidden_size))
    kda_cache = cache[model.model.ssm_idx]
    # Metal-only: a CPU default device (KQUANT_FORCE_CPU on hosted CI)
    # routes the plain decode step through the ops path too.
    assert kda_fused.fused_ok(x, None, kda_cache) == (
        mx.default_device() == mx.gpu)
    with mx.stream(mx.cpu):
        assert not kda_fused.fused_ok(x, None, kda_cache)
    assert not kda_fused.fused_ok(mx.zeros((1, 2, args.hidden_size)), None, kda_cache)
    assert not kda_fused.fused_ok(x, mx.ones((1, 1), dtype=mx.bool_), kda_cache)
    assert not kda_fused.fused_ok(x, None, None)


def test_launch_width_narrows_to_what_the_gpu_accepts(monkeypatch):
    # maxTotalThreadsPerThreadgroup is a per-pipeline limit that follows from
    # register pressure, and a GPU with a smaller register file refuses the
    # 512- and 1024-thread launches this kernel asks for first. The probe
    # steps the width down until one launches, and every block length takes
    # the same width, so a block stays bit-identical to the same tokens
    # stepped one at a time.
    real = kda_fused._launches
    monkeypatch.setattr(kda_fused, "_SG_FIT", {})
    monkeypatch.setattr(kda_fused, "_launches",
                        lambda vec, sg, *a: sg <= 8 and real(vec, sg, *a))
    assert kda_fused.sg_for(mx.bfloat16, 1, 2, 128, KW) == 8
    test_block_matches_chained_single_token(1, 128)
    assert kda_fused._SG_FIT[(str(mx.bfloat16), 1, 2, 128, KW)] == 8


def test_no_width_fits_routes_the_eager_path(monkeypatch):
    # A GPU that takes no width at all keeps the eager op chain, rather than
    # raising out of the decode step.
    class _Cache:
        lengths = None

    monkeypatch.setattr(kda_fused, "_SG_FIT", {})
    monkeypatch.setattr(kda_fused, "_launches", lambda *a: False)
    assert kda_fused.sg_for(mx.bfloat16, 1, 2, 128, KW) is None
    x = mx.zeros((1, 1, 256), dtype=mx.bfloat16)
    assert not kda_fused.fused_ok(x, None, _Cache(), num_heads=2,
                                  head_dim=128, conv_kernel=KW)
    assert kda_fused.fused_ok(x, None, _Cache()) == (
        mx.default_device() == mx.gpu)
