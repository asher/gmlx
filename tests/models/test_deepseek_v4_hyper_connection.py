"""HyperConnection parity: module output vs a naive-ops reference.

On a Metal GPU the module dispatches the fused norm+mix+sinkhorn+collapse
kernel; under KQUANT_FORCE_CPU (or non-Metal) it takes the compiled ops
path. Both are checked against the same eager reference.
"""

import mlx.core as mx
import pytest

from gmlx.models.deepseek_v4.hyper_connection import (
    HyperConnection,
    hc_expand,
    hc_expand_collapse,
)


class _Cfg:
    hc_mult = 4
    hc_sinkhorn_iters = 5
    hc_eps = 1e-6
    rms_norm_eps = 1e-6
    hidden_size = 256


def _reference(x, fn, base, scale, cfg):
    hc, eps = cfg.hc_mult, cfg.hc_eps
    y = x.astype(mx.float32)
    z = mx.fast.rms_norm(y.flatten(-2), None, cfg.rms_norm_eps)
    mixes = z @ fn.T
    pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * mx.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc) * scale[2]
    comb = comb + base[2 * hc :].reshape(hc, hc)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    collapsed = (pre[..., None] * y).sum(axis=2).astype(x.dtype)
    return collapsed, post, comb


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 4, 13])
def test_hyper_connection_matches_reference(dtype, length):
    cfg = _Cfg()
    mx.random.seed(11)
    hc = HyperConnection(cfg)
    mix_rows = (2 + cfg.hc_mult) * cfg.hc_mult
    hc.fn = (
        mx.random.normal((mix_rows, cfg.hc_mult * cfg.hidden_size)) * 0.02
    ).astype(mx.float32)
    hc.base = (mx.random.normal((mix_rows,)) * 0.5).astype(mx.float32)
    hc.scale = mx.array([1.1, 0.9, 1.3], dtype=mx.float32)

    x = (
        mx.random.normal((1, length, cfg.hc_mult, cfg.hidden_size)) * 1.7
    ).astype(dtype)
    collapsed, post, comb = hc(x)
    ref_collapsed, ref_post, ref_comb = _reference(
        x, hc.fn, hc.base, hc.scale, cfg
    )
    mx.eval(collapsed, post, comb, ref_collapsed, ref_post, ref_comb)

    ctol = 2e-2 if dtype == mx.float16 else 1e-1
    assert collapsed.shape == ref_collapsed.shape
    assert (
        mx.abs(collapsed.astype(mx.float32) - ref_collapsed.astype(mx.float32))
        .max()
        .item()
        < ctol
    )
    assert mx.abs(post - ref_post).max().item() < 3e-3
    assert mx.abs(comb - ref_comb).max().item() < 3e-3


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 4, 13])
def test_hc_expand_collapse_matches_pair(dtype, length):
    cfg = _Cfg()
    mx.random.seed(23)
    hc = HyperConnection(cfg)
    mix_rows = (2 + cfg.hc_mult) * cfg.hc_mult
    hc.fn = (
        mx.random.normal((mix_rows, cfg.hc_mult * cfg.hidden_size)) * 0.02
    ).astype(mx.float32)
    hc.base = (mx.random.normal((mix_rows,)) * 0.5).astype(mx.float32)
    hc.scale = mx.array([1.1, 0.9, 1.3], dtype=mx.float32)

    x = (mx.random.normal((1, length, cfg.hidden_size)) * 1.7).astype(dtype)
    residual = (
        mx.random.normal((1, length, cfg.hc_mult, cfg.hidden_size)) * 1.7
    ).astype(dtype)
    post = (mx.random.normal((1, length, cfg.hc_mult)) * 0.5 + 1.0).astype(
        mx.float32
    )
    comb = mx.softmax(
        mx.random.normal((1, length, cfg.hc_mult, cfg.hc_mult)), axis=-1
    ).astype(mx.float32)

    h, collapsed, post2, comb2 = hc_expand_collapse(hc, x, residual, post, comb)
    ref_h = hc_expand(x, residual, post, comb)
    ref_collapsed, ref_post2, ref_comb2 = _reference(
        ref_h, hc.fn, hc.base, hc.scale, cfg
    )
    mx.eval(h, collapsed, post2, comb2, ref_h, ref_collapsed, ref_post2,
            ref_comb2)

    htol = 1e-3 if dtype == mx.float16 else 8e-3
    ctol = 2e-2 if dtype == mx.float16 else 1e-1
    assert (
        mx.abs(h.astype(mx.float32) - ref_h.astype(mx.float32)).max().item()
        < htol
    )
    assert (
        mx.abs(collapsed.astype(mx.float32) - ref_collapsed.astype(mx.float32))
        .max()
        .item()
        < ctol
    )
    assert mx.abs(post2 - ref_post2).max().item() < 3e-3
    assert mx.abs(comb2 - ref_comb2).max().item() < 3e-3
