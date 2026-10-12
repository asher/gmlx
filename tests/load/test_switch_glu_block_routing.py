"""MoE verify blocks on gmlx switch linears take one gather per projection.

The mlx-vlm 0.7 SwitchGLU runs 1 < T <= 8 positions one at a time when its
exact kernels decline, and they decline on every gmlx switch linear. The
gmlx SwitchGLU classes (the block-routed swap, and the fused K-quant and
native-fp fallbacks) keep the mlx-vlm 0.6 routing: each projection sees the
whole block once. Projection calls are recorded, not computed, so no
kernel runs.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

vsl = pytest.importorskip("gmlx._vendor.mlx_vlm.models.switch_layers")
import gmlx._vendor.mlx_vlm.models.linear as vlm_linear  # noqa: E402

import gmlx.load.modules as modules  # noqa: E402

_B, _T, _K, _D, _H, _E = 1, 4, 2, 256, 256, 4


class _Holder(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.experts = glu


def _block():
    x = mx.zeros((_B, _T, _D), dtype=mx.bfloat16)
    idx = mx.zeros((_B, _T, _K), dtype=mx.uint32)
    return x, idx


def _record(monkeypatch, leaf_cls):
    """Replace ``leaf_cls.__call__`` with a recorder that returns zeros in
    the gather output layout, and return the recorded index shapes."""
    calls = []

    def fake(self, x, indices, sorted_indices=False):
        calls.append(tuple(indices.shape))
        return mx.zeros((*indices.shape, 1, self.weight.shape[1]), x.dtype)

    monkeypatch.setattr(leaf_cls, "__call__", fake)
    return calls


def _no_tokenwise(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("SwitchGLU took the per-position loop")

    monkeypatch.setattr(vlm_linear, "tokenwise", fail)


def _kquant_glu():
    model = _Holder(vsl.SwitchGLU(_D, _H, _E))
    modules.install_kquant_modules(
        model, {f"experts.{p}.weight": "q8_0"
                for p in ("gate_proj", "up_proj", "down_proj")})
    model.eval()
    return model


def test_stock_glu_runs_a_kquant_verify_block_per_position(monkeypatch):
    """The premise: the stock 0.7 call on K-quant leaves loops over the
    block, one gather per position and projection."""
    from mlx_kquant.nn import KQuantSwitchLinear

    model = _kquant_glu()
    calls = _record(monkeypatch, KQuantSwitchLinear)
    y = model.experts(*_block())
    assert y.shape == (_B, _T, _K, _D)
    assert calls == [(_B, 1, _K)] * (3 * _T)


def test_block_routing_swaps_kquant_glus_and_gathers_the_block(monkeypatch):
    from mlx_kquant.nn import KQuantSwitchLinear

    model = _kquant_glu()
    assert modules.install_switch_glu_block_routing(model) == 1
    assert type(model.experts).__name__ == "_BlockRoutedSwitchGLU"
    assert isinstance(model.experts, vsl.SwitchGLU)
    assert modules.install_switch_glu_block_routing(model) == 0

    calls = _record(monkeypatch, KQuantSwitchLinear)
    _no_tokenwise(monkeypatch)
    y = model.experts(*_block())
    assert y.shape == (_B, _T, _K, _D)
    assert calls == [(_B, _T, _K)] * 3


def test_block_routing_keeps_the_stock_combine(monkeypatch):
    """Weights, shared output and the sum over routed slots apply as in the
    stock call, for callers on the mlx-vlm 0.7 MoE signature."""
    from mlx_kquant.nn import KQuantSwitchLinear

    model = _kquant_glu()
    modules.install_switch_glu_block_routing(model)
    _record(monkeypatch, KQuantSwitchLinear)
    x, idx = _block()
    w = mx.ones((_B, _T, _K), dtype=mx.bfloat16)
    shared = mx.ones((_B, _T, _D), dtype=mx.bfloat16)
    y = model.experts(x, idx, w, shared)
    assert y.shape == (_B, _T, _D)
    assert mx.array_equal(y, shared)


def test_block_routing_leaves_stock_quantized_glus_alone():
    glu = vsl.SwitchGLU(_D, _H, _E)
    glu.gate_proj = glu.gate_proj.to_quantized(64, 4)
    glu.up_proj = glu.up_proj.to_quantized(64, 4)
    model = _Holder(glu)
    assert modules.install_switch_glu_block_routing(model) == 0
    assert type(model.experts) is vsl.SwitchGLU


def test_fused_kquant_fallback_gathers_the_block(monkeypatch):
    from mlx_kquant.nn import KQuantSwitchLinear

    model = _kquant_glu()
    fused = modules._make_fused_kquant(vsl.SwitchGLU, modules._FusedMoeCaps())
    model.experts.__class__ = fused
    monkeypatch.setattr(modules, "_FUSED_MOE_ENABLED", False)
    calls = _record(monkeypatch, KQuantSwitchLinear)
    _no_tokenwise(monkeypatch)
    y = model.experts(*_block())
    assert y.shape == (_B, _T, _K, _D)
    assert calls == [(_B, _T, _K)] * 3


def test_fused_nativefp_fallback_gathers_the_block(monkeypatch):
    glu = vsl.SwitchGLU(_D, _H, _E)
    glu.eval()
    fused = modules._make_fused_nativefp(vsl.SwitchGLU, modules._FusedMoeCaps())
    glu.__class__ = fused
    monkeypatch.setattr(modules, "_FUSED_MOE_ENABLED", False)
    calls = _record(monkeypatch, vsl.SwitchLinear)
    _no_tokenwise(monkeypatch)
    y = glu(*_block())
    assert y.shape == (_B, _T, _K, _D)
    assert calls == [(_B, _T, _K)] * 3
