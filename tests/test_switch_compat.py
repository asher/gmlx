#!/usr/bin/env python3
"""Dual-origin switch-layer recognition. mlx-vlm 0.6.4 vendored mlx-lm's
switch_layers module, so MoE models built from it instantiate functionally
identical but distinct SwitchLinear/SwitchGLU classes; the leaf swap and the
fused-GLU installer must accept either origin, and a codec'd leaf nobody
recognizes must fail loud instead of loading wire bytes into a stock float
module (which surfaces later as an opaque gather_mm shape error)."""
from __future__ import annotations

import sys
import types

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.models import switch_layers as lm_sl

from gmlx import modules


@pytest.fixture
def vlm_switch_module(monkeypatch):
    """The mlx_vlm switch_layers module: the real one when installed
    (mlx-vlm >= 0.6.4), else a stand-in registered at its import path so the
    dual-origin mechanism is exercised on 0.6.3 too."""
    try:
        from mlx_vlm.models import switch_layers as vlm_sl
    except ImportError:
        vlm_sl = types.ModuleType("mlx_vlm.models.switch_layers")

        class SwitchLinear(lm_sl.SwitchLinear):
            pass

        class SwitchGLU(lm_sl.SwitchGLU):
            pass

        SwitchLinear.__module__ = vlm_sl.__name__
        SwitchGLU.__module__ = vlm_sl.__name__
        vlm_sl.SwitchLinear = SwitchLinear
        vlm_sl.SwitchGLU = SwitchGLU
        monkeypatch.setitem(
            sys.modules, "mlx_vlm.models.switch_layers", vlm_sl)
        import mlx_vlm.models as vlm_models
        monkeypatch.setattr(
            vlm_models, "switch_layers", vlm_sl, raising=False)
    monkeypatch.setattr(modules, "_SWITCH_TYPES", None)
    yield vlm_sl
    modules._SWITCH_TYPES = None


def _glu(sl_mod, dim=64, hidden=32, experts=4):
    return sl_mod.SwitchGLU(dim, hidden, experts)


def _meta_for(prefix, codec="q8_0"):
    return {f"{prefix}.{p}.weight": codec
            for p in ("gate_proj", "up_proj", "down_proj")}


class _Holder(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.experts = glu


def test_types_always_include_mlx_lm():
    lin, glu = modules.switch_layer_types()
    assert lm_sl.SwitchLinear in lin
    assert lm_sl.SwitchGLU in glu


def test_types_include_vlm_origin(vlm_switch_module):
    lin, glu = modules.switch_layer_types()
    assert vlm_switch_module.SwitchLinear in lin
    assert vlm_switch_module.SwitchGLU in glu


@pytest.mark.parametrize("origin", ["mlx_lm", "mlx_vlm"])
def test_kquant_swap_recognizes_both_origins(origin, vlm_switch_module):
    sl_mod = lm_sl if origin == "mlx_lm" else vlm_switch_module
    model = _Holder(_glu(sl_mod))
    n = modules.install_kquant_modules(model, _meta_for("experts"))
    assert n == 3
    for proj in ("gate_proj", "up_proj", "down_proj"):
        leaf = getattr(model.experts, proj)
        assert type(leaf).__name__ == "KQuantSwitchLinear", (origin, proj)


@pytest.mark.parametrize("origin", ["mlx_lm", "mlx_vlm"])
def test_fused_glu_swaps_both_origins(origin, vlm_switch_module):
    import mlx_kquant as kq
    if not hasattr(kq, "moe_glu_gather_kq"):
        pytest.skip("mlx_kquant without fused MoE kernels")
    sl_mod = lm_sl if origin == "mlx_lm" else vlm_switch_module
    # q8_0 geometry the fused path accepts: K % 256 == 0 (both matmuls,
    # so dim and hidden), N % 8 == 0.
    model = _Holder(_glu(sl_mod, dim=256, hidden=256))
    base = type(model.experts)
    modules.install_kquant_modules(model, _meta_for("experts"))
    n = modules.install_fused_moe_glu(model)
    assert n == 1, origin
    assert type(model.experts).__name__ == "_FusedKQuantSwitchGLU"
    assert isinstance(model.experts, base), (
        "fused class must subclass the instance's own origin")
    # Idempotent: a second pass must not re-wrap the fused instance.
    assert modules.install_fused_moe_glu(model) == 0


def test_unrecognized_codec_leaf_fails_loud():
    class _Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((4, 8, 8))

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.odd = _Odd()

    with pytest.raises(ValueError, match="not\\s+recognized"):
        modules.install_kquant_modules(_M(), {"odd.weight": "q8_0"})
