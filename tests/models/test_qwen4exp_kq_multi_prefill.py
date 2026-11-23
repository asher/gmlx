"""Prefill rows of qwen4_exp: the k-quant projections that share an input
(GDN qkv and z, attention q, k and v, shared expert gate and up) run as one
kq.quantized_matmul_multi call past 128 rows, with the bits of a call per
weight (GMLX_KQ_MULTI_PREFILL=0)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_kquant as kq
from mlx_kquant.nn import KQuantLinear

import gmlx.models.qwen4_exp.model as q4

gp = importlib.import_module("gmlx.upstream.gdn_patches")
H = 256

if not hasattr(kq, "quantized_matmul_multi"):
    pytest.skip("mlx-kquant without quantized_matmul_multi",
                allow_module_level=True)


def _kq(rows, seed, k=H, codec="q6_k"):
    mx.random.seed(seed)
    lin = KQuantLinear(k, rows, bias=False, codec=codec)
    lin.weight = kq.quantize(mx.random.normal((rows, k)) * 0.05, codec)[0]
    return lin


@pytest.fixture
def calls(monkeypatch):
    seen = []
    real = kq.quantized_matmul_multi

    def multi(x, weights, codec, **kw):
        seen.append([int(w.shape[0]) for w in weights])
        return real(x, weights, codec, **kw)

    monkeypatch.setattr(kq, "quantized_matmul_multi", multi)
    return seen


def _x(rows, seed=3):
    return (mx.random.normal((1, rows, H), key=mx.random.key(seed)) * 0.5
            ).astype(mx.bfloat16)


def _both(monkeypatch, run):
    got = run()
    mx.eval(got)
    monkeypatch.setenv("GMLX_KQ_MULTI_PREFILL", "0")
    want = run()
    mx.eval(want)
    monkeypatch.delenv("GMLX_KQ_MULTI_PREFILL")
    return got, want


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(a, b).item())


def test_the_gdn_projections_run_as_one_call(monkeypatch, calls):
    args = SimpleNamespace(
        hidden_size=H, linear_num_value_heads=6, linear_num_key_heads=2,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6)
    mx.random.seed(0)
    layer = q4.GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    layer.in_proj_qkv = _kq(layer.conv_dim, 1)
    layer.in_proj_z = _kq(layer.value_dim, 2)
    layer.eval()
    x = _x(160)
    got, want = _both(monkeypatch, lambda: layer(x, cache=[None, None]))
    assert calls == [[layer.conv_dim, layer.value_dim]]
    assert _same(got, want)


def test_the_attention_projections_run_as_one_call(monkeypatch, calls):
    args = q4.ModelArgs(hidden_size=H, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=1,
                        head_dim=128, layer_types=["full_attention"])
    mx.random.seed(13)
    layer = q4.Attention(args, 0)
    layer.set_dtype(mx.bfloat16)
    layer.q_proj = _kq(4 * 2 * 128, 1)
    layer.k_proj = _kq(128, 2)
    layer.v_proj = _kq(128, 3)
    x = _x(200)
    got, want = _both(monkeypatch, lambda: list(layer._project(x)))
    assert calls == [[4 * 2 * 128, 128, 128]]
    for g, w in zip(got, want):
        assert (g is None and w is None) or _same(g, w)


def test_the_shared_expert_gate_and_up_run_as_one_call(monkeypatch, calls):
    mlp = q4.MLP(H, 640)
    mlp.gate_proj = _kq(640, 1)
    mlp.up_proj = _kq(640, 2)
    mlp.down_proj = _kq(H, 3, k=640, codec="q8_0")
    x = _x(300)
    got, want = _both(monkeypatch, lambda: mlp(x))
    assert calls == [[640, 640]]
    assert _same(got, want)


@pytest.mark.parametrize("rows", [6, 64, 128])
def test_up_to_128_rows_keep_the_calls_per_weight(calls, rows):
    lins = (_kq(640, 1), _kq(640, 2))
    assert gp.kquant_multi(lins, _x(rows), prefill=True) is None
    assert not calls
