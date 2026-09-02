"""HY4 independent hyper-connections (iHC), against a NumPy reference.

iHC is DeepSeek-V4's hyper-connection minus the sinkhorn/comb term: one
[2*hc, hc*hidden] mixer over the weightless RMSNorm of the flattened residual
streams gives ``hc`` collapse gates and ``hc`` redistribute gates. The whole
cycle runs in fp32 because 78 layers x 2 sublayers of bf16 rounding on the
streams compounds.

The reference here is transcribed from llama.cpp ``src/models/hyv4.cpp``, not
from the MLX implementation, so a transcription error in either shows up as a
mismatch rather than as agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.models.hy_v4.model import IndependentHyperConnection, ModelArgs

HC = 4
DIM = 8
EPS = 1e-6
HC_EPS = 1e-6
MAGNITUDE = 2.0


def _args(**over):
    base = dict(
        model_type="hy_v4", vocab_size=16, hidden_size=DIM, intermediate_size=16,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4,
        v_head_dim=4, n_routed_experts=2, num_experts_per_tok=1,
        n_shared_experts=1, first_k_dense_replace=1, rms_norm_eps=EPS,
        hc_mult=HC, hc_eps=HC_EPS, hc_magnitude=MAGNITUDE,
    )
    base.update(over)
    return ModelArgs(**base)


def _hc(seed=0):
    rng = np.random.default_rng(seed)
    hc = IndependentHyperConnection(_args())
    hc.fn = mx.array(rng.normal(scale=0.3, size=(2 * HC, HC * DIM)).astype("float32"))
    hc.base = mx.array(rng.normal(scale=0.5, size=(2 * HC,)).astype("float32"))
    hc.scale = mx.array(rng.normal(loc=1.0, scale=0.2, size=(2,)).astype("float32"))
    return hc


def _ref_gates(x, fn, base, scale):
    """llama.cpp hyv4 iHC front, in fp64 NumPy."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(x.shape[:-2] + (HC * DIM,))
    rms = np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + EPS)
    z = flat / rms
    mixes = z @ np.asarray(fn, dtype=np.float64).T
    base = np.asarray(base, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    pre = sig(mixes[..., :HC] * scale[0] + base[:HC]) + HC_EPS
    post = MAGNITUDE * sig(mixes[..., HC:] * scale[1] + base[HC:]) + HC_EPS
    return pre, post


def test_pre_collapse_matches_reference():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 3, HC, DIM)).astype("float32")
    hc = _hc()
    got, post = hc.pre(mx.array(x))
    mx.eval(got, post)

    pre_ref, post_ref = _ref_gates(x, hc.fn, hc.base, hc.scale)
    collapsed = (pre_ref[..., None] * x.astype(np.float64)).sum(axis=-2)

    assert np.allclose(np.array(got), collapsed, atol=2e-5)
    assert np.allclose(np.array(post), post_ref, atol=2e-5)
    assert got.shape == (2, 3, DIM)
    assert post.shape == (2, 3, HC)


def test_expand_matches_reference():
    rng = np.random.default_rng(11)
    residual = rng.normal(size=(2, 3, HC, DIM)).astype("float32")
    y = rng.normal(size=(2, 3, DIM)).astype("float32")
    post = rng.uniform(0.0, 2.0, size=(2, 3, HC)).astype("float32")

    hc = _hc()
    got = hc.expand(mx.array(y), mx.array(residual), mx.array(post))
    mx.eval(got)

    ref = (residual.astype(np.float64)
           + post.astype(np.float64)[..., None] * y.astype(np.float64)[..., None, :])
    assert np.allclose(np.array(got), ref, atol=2e-5)
    assert got.shape == residual.shape


def test_expand_is_identity_when_post_is_zero():
    rng = np.random.default_rng(3)
    residual = mx.array(rng.normal(size=(1, 2, HC, DIM)).astype("float32"))
    y = mx.array(rng.normal(size=(1, 2, DIM)).astype("float32"))
    got = _hc().expand(y, residual, mx.zeros((1, 2, HC), dtype=mx.float32))
    mx.eval(got)
    assert np.array_equal(np.array(got), np.array(residual))


def test_gates_are_strictly_positive_and_bounded():
    # pre in (hc_eps, 1 + hc_eps), post in (hc_eps, magnitude + hc_eps): a
    # collapse gate that reached 0 would drop a residual stream entirely.
    rng = np.random.default_rng(5)
    x = mx.array((rng.normal(size=(1, 4, HC, DIM)) * 50).astype("float32"))
    hc = _hc(seed=2)
    collapsed, post = hc.pre(x)
    mx.eval(collapsed, post)
    p = np.array(post)
    assert (p > HC_EPS / 2).all() and (p < MAGNITUDE + 2 * HC_EPS).all()
    assert np.isfinite(np.array(collapsed)).all()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_cycle_runs_in_fp32_and_returns_input_dtype(dtype):
    # The parameters stay fp32 leaves; only the returned arrays follow the
    # stream dtype. A cast of fn/base/scale would show up as a dtype change.
    rng = np.random.default_rng(13)
    x = mx.array(rng.normal(size=(1, 2, HC, DIM)).astype("float32")).astype(dtype)
    hc = _hc()
    collapsed, post = hc.pre(x)
    out = hc.expand(collapsed, x, post)
    mx.eval(collapsed, post, out)
    assert hc.fn.dtype == mx.float32
    assert hc.base.dtype == mx.float32
    assert hc.scale.dtype == mx.float32
    assert collapsed.dtype == dtype
    assert post.dtype == mx.float32       # gates stay fp32 into expand
    assert out.dtype == dtype


def test_no_sinkhorn_parameters():
    # The delta from DeepSeek-V4 HC. A comb/sinkhorn tensor here would mean
    # the wrong reference was ported.
    hc = _hc()
    names = {k for k, _ in hc.parameters().items()}
    assert names == {"fn", "base", "scale"}
    assert not hasattr(hc, "comb")
    assert not hasattr(hc, "sinkhorn_iters")
