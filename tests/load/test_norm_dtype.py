"""GMLX_NORM_DTYPE: which models keep their F32 norm weights float32, and the
cast that keeps a float32 norm weight from widening the residual stream."""

from __future__ import annotations

import pytest

import mlx.core as mx
import mlx.nn as nn

from gmlx.load.loader import cast_fp32_norm_outputs, fp32_keep_for


@pytest.mark.parametrize("mode, beam, qwen3", [
    ("auto", True, False),
    ("fp32", True, True),
    ("activation", False, False),
    ("nonsense", True, False),
])
def test_the_mode_picks_which_models_pin_their_norms(monkeypatch, mode, beam,
                                                     qwen3):
    monkeypatch.setenv("GMLX_NORM_DTYPE", mode)
    assert ("norm.weight" in fp32_keep_for("beam")) is beam
    assert ("norm.weight" in fp32_keep_for("qwen3")) is qwen3
    # The model type's own pins stay either way.
    assert ".gate_bias" in fp32_keep_for("beam")


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(8)
        self.ln = nn.LayerNorm(8)
        self.proj = nn.Linear(8, 8)


def test_norms_with_fp32_weights_return_the_input_dtype():
    m = _Block()
    m.proj.update({"weight": m.proj.weight.astype(mx.bfloat16),
                   "bias": m.proj.bias.astype(mx.bfloat16)})
    x = mx.random.normal((2, 8)).astype(mx.bfloat16)
    assert m.input_layernorm(x).dtype == mx.float32
    assert cast_fp32_norm_outputs(m, mx.bfloat16) == 2
    assert m.input_layernorm(x).dtype == mx.bfloat16
    assert m.ln(x).dtype == mx.bfloat16
    assert isinstance(m.input_layernorm, nn.RMSNorm)
    assert m.proj(x).dtype == mx.bfloat16
    # A float32 input keeps float32, and a second pass changes nothing.
    assert m.input_layernorm(x.astype(mx.float32)).dtype == mx.float32
    assert cast_fp32_norm_outputs(m, mx.bfloat16) == 0


def test_bf16_norm_weights_and_fp32_graphs_are_left_alone():
    m = _Block()
    assert cast_fp32_norm_outputs(m, mx.float32) == 0
    m.input_layernorm.weight = m.input_layernorm.weight.astype(mx.bfloat16)
    m.ln.weight = m.ln.weight.astype(mx.bfloat16)
    assert cast_fp32_norm_outputs(m, mx.bfloat16) == 0
    assert type(m.input_layernorm) is nn.RMSNorm


def test_a_wider_activation_dtype_still_widens_a_narrow_input(monkeypatch):
    """With float32 activations, a bf16 input (a BF16 embedding) comes out
    float32, as a norm weight in the activation dtype would give."""
    from gmlx.models.beam.model import _Norm

    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float32")
    x = mx.random.normal((2, 8)).astype(mx.bfloat16)
    assert _Norm(8)(x).dtype == mx.float32
    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "bfloat16")
    assert _Norm(8)(x).dtype == mx.bfloat16
    m = _Block()
    cast_fp32_norm_outputs(m, mx.float16)
    assert m.input_layernorm(x).dtype == mx.float32
