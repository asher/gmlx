"""Verify-width groups of k-quant projections that share an input run as
one kq.quantized_matmul_multi call: attention q/k/v and MLP gate/up
through verify_linears, and the gated-delta qkv, z and b/a in the fused
verify body. An mlx-kquant without the op keeps the per-weight calls."""

from __future__ import annotations

import importlib
import os

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_kquant as kq
from mlx_kquant.nn import KQuantLinear

from gmlx.models.qwen35.verify_linear import verify_linears

gp = importlib.import_module("gmlx.upstream.gdn_patches")
K = 64


def _kq(rows, seed, codec="q4_0", k=K):
    mx.random.seed(seed)
    lin = KQuantLinear(k, rows, bias=False, codec=codec)
    lin.weight = kq.quantize(mx.random.normal((rows, k)) * 0.1, codec)[0]
    return lin


def _each(linears, x):
    return [lin(x) for lin in linears]


@pytest.fixture
def calls(monkeypatch):
    """A stand-in op that records each call's weight rows and runs the
    weights one by one."""
    seen = []
    sc = mx.zeros((1,), mx.uint8)

    def multi(x, weights, codec):
        seen.append([int(w.shape[0]) for w in weights])
        return [kq.quantized_matmul(x, w, sc, codec, transpose=True)
                for w in weights]

    monkeypatch.setattr(kq, "quantized_matmul_multi", multi, raising=False)
    return seen


def test_a_verify_width_group_runs_as_one_call(calls):
    lins = (_kq(96, 1), _kq(32, 2), _kq(32, 3))
    x = mx.random.normal((1, 6, K)).astype(mx.bfloat16)
    out = gp.kquant_multi(lins, x)
    assert calls == [[96, 32, 32]]
    for got, want in zip(out, _each(lins, x)):
        assert mx.array_equal(got, want)


@pytest.mark.parametrize("rows", [1, 9])
def test_outside_the_verify_widths_it_does_not_apply(calls, rows):
    lins = (_kq(32, 1), _kq(32, 2))
    assert gp.kquant_multi(lins, mx.zeros((1, rows, K), mx.bfloat16)) is None
    assert not calls


def test_a_mixed_or_wrapped_group_does_not_apply(calls):
    x = mx.zeros((1, 4, K), mx.bfloat16)
    assert gp.kquant_multi((_kq(32, 1), _kq(32, 2, codec="q8_0")), x) is None

    class Wrapped(KQuantLinear):
        pass

    w = _kq(32, 2)
    w.__class__ = Wrapped
    assert gp.kquant_multi((_kq(32, 1), w), x) is None
    assert gp.kquant_multi((_kq(32, 1), nn.Linear(K, 32, bias=False)), x) is None
    assert not calls


def test_an_mlx_kquant_without_the_op_keeps_the_per_weight_calls(monkeypatch):
    monkeypatch.delattr(kq, "quantized_matmul_multi", raising=False)
    x = mx.zeros((1, 4, K), mx.bfloat16)
    assert gp.kquant_multi((_kq(32, 1), _kq(32, 2)), x) is None


def test_verify_linears_takes_the_op_at_verify_only(calls):
    lins = (_kq(64, 1), _kq(32, 2), _kq(32, 3))
    x = mx.random.normal((1, 6, K)).astype(mx.bfloat16)
    got = verify_linears(lins, x, True)
    assert calls == [[64, 32, 32]]
    for g, w in zip(got, _each(lins, x)):
        assert mx.array_equal(g, w)
    verify_linears(lins, x, False)
    verify_linears(lins, x[:, :1], True)
    assert len(calls) == 1


def _gdn(cat_ba: bool):
    from gmlx._vendor.mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=K, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        rms_norm_eps=1e-6, vocab_size=32, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4,
        full_attention_interval=4, head_dim=32, rope_theta=1e4,
        max_position_embeddings=128)
    mx.random.seed(0)
    gdn = GatedDeltaNet(args)
    mx.eval(gdn.parameters())
    gdn.in_proj_qkv = _kq(gdn.conv_dim, 1)
    gdn.in_proj_z = _kq(gdn.value_dim, 2)
    gdn.in_proj_b = _kq(gdn.num_v_heads, 3)
    gdn.in_proj_a = _kq(gdn.num_v_heads, 4)
    if cat_ba:
        assert gp._gdn_try_cat_ba(gdn)
    return gdn


@pytest.mark.parametrize("cat_ba", [True, False])
def test_the_fused_gdn_verify_runs_its_projections_as_one_call(
        monkeypatch, calls, cat_ba):
    gdn = _gdn(cat_ba)
    x = mx.random.normal((1, 3, K)).astype(mx.bfloat16)
    want_ba = mx.concatenate([gdn.in_proj_b(x), gdn.in_proj_a(x)], axis=-1)
    seen = {}

    def kernel(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
        seen["conv_input"], seen["ba"] = inputs[0], inputs[2]
        return [mx.zeros(s, d) for s, d in zip(output_shapes, output_dtypes)]

    monkeypatch.setattr(gp, "_gdn_fused_verify_kernel", kernel)
    gp._gdn_fused_verify_body(gdn, x, None, None, [])
    hv = gdn.num_v_heads
    rest = [2 * hv] if cat_ba else [hv, hv]
    assert calls == [[gdn.conv_dim, gdn.value_dim, *rest]]
    assert mx.array_equal(seen["ba"], want_ba)
    assert mx.array_equal(seen["conv_input"][:, -3:], gdn.in_proj_qkv(x))


@pytest.mark.skipif(bool(os.environ.get("KQUANT_FORCE_CPU")),
                    reason="the op's joint dispatch is a Metal route")
@pytest.mark.skipif(not hasattr(kq, "quantized_matmul_multi"),
                    reason="mlx-kquant without quantized_matmul_multi")
@pytest.mark.parametrize("m", [2, 6, 8])
def test_the_real_op_matches_the_per_weight_calls(m):
    lins = (_kq(256, 1, k=512), _kq(96, 2, k=512), _kq(48, 3, k=512))
    x = mx.random.normal((1, m, 512)).astype(mx.bfloat16)
    for got, want in zip(gp.kquant_multi(lins, x), _each(lins, x)):
        assert mx.allclose(got.astype(mx.float32), want.astype(mx.float32),
                           atol=2e-2, rtol=2e-2)
