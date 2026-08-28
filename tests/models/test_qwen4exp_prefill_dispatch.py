"""qwen4exp prefill-path coverage: HC prefill kernels (norm/epi/combine/
inject) vs the eager ops, the MoE residual-dtype contract, and the QSA
prefill dispatch (split-regime and ragged-L) vs the dense token-mask
reference."""

from __future__ import annotations

import os
import re

import mlx.core as mx
import pytest

import gmlx.models.qwen4_exp.model as q4
from gmlx.models.qwen4_exp.model import (
    Attention,
    HyperConnection,
    ModelArgs,
    QSAKVCache,
    SparseMoeBlock,
)


def _real_apple_gpu() -> bool:
    if os.environ.get("KQUANT_FORCE_CPU"):
        return False
    try:
        name = str(mx.device_info().get("device_name", ""))
    except Exception:
        return False
    return bool(re.search(r"Apple M\d", name))


gpu_only = pytest.mark.skipif(
    not _real_apple_gpu(), reason="needs a real Apple GPU")


# HC prefill kernels


def _hc_module(hidden=256, hc=4, lowrank=32, dtype=mx.bfloat16):
    mx.random.seed(11)
    m = HyperConnection(hidden, hc, lowrank, eps=1e-6)
    m.norm.weight = mx.random.normal((hc * hidden,)).astype(dtype) * 0.1 + 1.0
    m.down.weight = (mx.random.normal(m.down.weight.shape) * 0.05).astype(dtype)
    m.up.weight = (mx.random.normal(m.up.weight.shape) * 0.05).astype(dtype)
    # the loader keeps inject weights fp32
    m.inject.weight = mx.random.normal(m.inject.weight.shape) * 0.05
    return m


def _eager_arm(fn):
    """Run ``fn`` with the HC prefill kernels disarmed (ops fallback)."""
    saved = q4._hc_prefill_kerns
    q4._hc_prefill_kerns = False
    try:
        return fn()
    finally:
        q4._hc_prefill_kerns = saved


@gpu_only
def test_hc_inject_kern_matches_fp32_matmul():
    B, T, hc, hidden = 1, 32, 4, 256
    mx.random.seed(3)
    xf = mx.random.normal((B, T, hc * hidden)).astype(mx.bfloat16)
    w = mx.random.normal((hc, hc * hidden)) * 0.05
    got = q4._hc_inject_kern(xf, w)
    assert got is not None
    mx.eval(got)
    # float64 reference: mx.matmul's fp32 GEMM runs TF32 on M5-class GPUs,
    # so it is the noisier arm; the kernel accumulates exact fp32
    import numpy as np
    ref = (np.asarray(xf.astype(mx.float32), dtype=np.float64)
           @ np.asarray(w, dtype=np.float64).T)
    assert got.dtype == mx.float32
    assert float(np.abs(np.asarray(got, dtype=np.float64) - ref).max()) < 1e-5


@gpu_only
def test_hc_inject_kern_guards():
    hc, n = 4, 1024
    w = mx.zeros((hc, n))
    small = mx.zeros((1, 4, n), dtype=mx.bfloat16)
    assert q4._hc_inject_kern(small, w) is None          # B*T <= 8
    odd = mx.zeros((1, 32, n + 32), dtype=mx.bfloat16)
    assert q4._hc_inject_kern(odd, mx.zeros((hc, n + 32))) is None  # N % 256
    f32x = mx.zeros((1, 32, n))
    assert q4._hc_inject_kern(f32x, w) is None           # fp32 activations
    bf16w = mx.zeros((hc, n), dtype=mx.bfloat16)
    x = mx.zeros((1, 32, n), dtype=mx.bfloat16)
    assert q4._hc_inject_kern(x, bf16w) is None          # non-fp32 weight


@gpu_only
def test_hc_call_kernel_path_matches_eager():
    m = _hc_module()
    B, T, hc, hidden = 1, 32, 4, 256
    mx.random.seed(5)
    h = mx.random.normal((B, T, hc, hidden)).astype(mx.bfloat16)
    mixed_k, inj_k = m(h)
    mixed_e, inj_e = _eager_arm(lambda: m(h))
    mx.eval(mixed_k, inj_k, mixed_e, inj_e)
    assert mixed_k.dtype == mixed_e.dtype == mx.bfloat16
    dm = mx.abs(mixed_k.astype(mx.float32) - mixed_e.astype(mx.float32))
    di = mx.abs(inj_k - inj_e)
    # the kernels mirror the eager rounding points; residual is the fp32
    # sigmoid lsb class plus the inject GEMV's reduction order
    assert float(dm.max()) < 2e-2
    assert float(di.max()) < 2e-2


@gpu_only
def test_hc_combine_kernel_matches_ops():
    B, T, hc, hidden = 1, 32, 4, 256
    mx.random.seed(7)
    h = mx.random.normal((B, T, hc, hidden)).astype(mx.bfloat16)
    out = mx.random.normal((B, T, hidden)).astype(mx.bfloat16)
    # live inj is 2*sigmoid(...), so (0, 2)
    inj = 2.0 * mx.sigmoid(mx.random.normal((B, T, hc)))
    got = q4._hc_combine_kern(h, out, inj)
    assert got is not None
    # the ops fallback keeps the fp32 promotion (that is the escape
    # hatch's contract); the kernel result differs by its two InT
    # roundings, so compare in the bf16 grid at a couple of ulps
    ref = q4._hc_combine_ops(h, out, inj).astype(mx.bfloat16)
    mx.eval(got, ref)
    assert got.dtype == mx.bfloat16
    d = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    assert float(d.max()) < 6e-2


# MoE residual-dtype contract (the fp32 stream poison regression)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_moe_output_keeps_activation_dtype(dtype):
    args = ModelArgs(hidden_size=64, num_experts=8, num_experts_per_tok=2,
                     moe_intermediate_size=32,
                     shared_expert_intermediate_size=32,
                     num_hidden_layers=1)
    moe = SparseMoeBlock(args)
    moe.set_dtype(dtype)
    # the loader keeps router weights fp32; the weighted sum must still
    # return the activation dtype or every downstream layer runs fp32
    moe.gate.weight = moe.gate.weight.astype(mx.float32)
    # the expert matmuls return the activation dtype on every backend;
    # stub them (CPU gather_mm is fp32-only) so the test pins the
    # routing-scores promotion path itself
    moe.switch_mlp = lambda x, inds: mx.ones(
        inds.shape + (x.shape[-1],), dtype=x.dtype)
    x = mx.random.normal((1, 16, 64)).astype(dtype)
    y = moe(x)
    mx.eval(y)
    assert y.dtype == dtype


# QSA prefill dispatch vs the dense token-mask reference


def _qsa_args():
    return ModelArgs(hidden_size=128, num_hidden_layers=1,
                     num_attention_heads=12, num_key_value_heads=1,
                     head_dim=256, indexer_budget=8, compress_ratios=[4],
                     layer_types=["full_attention"])


def _qsa_layer(dtype=mx.bfloat16):
    mx.random.seed(13)
    layer = Attention(_qsa_args(), 0)
    layer.set_dtype(dtype)
    for lin in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                layer.indexer.q_proj, layer.indexer.k_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.05).astype(dtype)
    return layer


def _run_arm(layer, xs, kernel: bool):
    """Forward each window in ``xs`` through a fresh cache; ``kernel=False``
    disarms the kq block-sparse handle so dispatch falls back to the dense
    token-mask SDPA reference."""
    saved = q4._kq_bs_prefill
    if not kernel:
        q4._kq_bs_prefill = lambda: None
    try:
        cache = QSAKVCache(ratio=4)
        outs = [layer(x, mask=None, cache=cache) for x in xs]
        mx.eval(*outs)
        return outs
    finally:
        q4._kq_bs_prefill = saved


def _assert_close(a, b, tol):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    assert float(d.max()) < tol, float(d.max())


@gpu_only
@pytest.mark.parametrize("L", [44, 45])
def test_qsa_prefill_dispatch_matches_dense_mask(L):
    """L=44: aligned split-regime window. L=45: ragged-L window (split
    head + gathered tail) -- the serve one-shot shape class."""
    if q4._kq_bs_prefill() is None:
        pytest.skip("kq block-sparse prefill kernel unavailable")
    layer = _qsa_layer()
    mx.random.seed(17)
    x = mx.random.normal((1, L, 128)).astype(mx.bfloat16)
    (got,) = _run_arm(layer, [x], kernel=True)
    (ref,) = _run_arm(layer, [x], kernel=False)
    _assert_close(got, ref, 2e-2)


@gpu_only
def test_qsa_ragged_all_sparse_matches_dense_mask():
    """Ragged second window past the sparse boundary (block-sparse head +
    gathered tail at offset > 0)."""
    if q4._kq_bs_prefill() is None:
        pytest.skip("kq block-sparse prefill kernel unavailable")
    layer = _qsa_layer()
    mx.random.seed(19)
    x1 = mx.random.normal((1, 64, 128)).astype(mx.bfloat16)
    x2 = mx.random.normal((1, 13, 128)).astype(mx.bfloat16)
    _, got = _run_arm(layer, [x1, x2], kernel=True)
    _, ref = _run_arm(layer, [x1, x2], kernel=False)
    _assert_close(got, ref, 2e-2)
