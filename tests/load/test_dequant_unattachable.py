"""Codec'd tensors on raw-array carrier modules dequantize to f32 at load.

Some community GGUF quantizers codec the MoE router gate (llama.cpp's own
quantize leaves it F32). The deepseek-family MoEGate holds that weight as a
raw array, so no KQuant* module can attach; the loader reconstructs the float
tensor instead of failing. Large unattachable tensors keep failing loud.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import pytest

from gmlx.modules import dequantize_unattachable_leaves, install_kquant_modules


class RawGate(nn.Module):
    """MoEGate-style carrier: bare weight array, no Linear."""

    def __init__(self, n_experts, dims):
        super().__init__()
        self.weight = mx.zeros((n_experts, dims))


class Block(nn.Module):
    def __init__(self, n_experts, dims):
        super().__init__()
        self.gate = RawGate(n_experts, dims)
        self.proj = nn.Linear(dims, dims, bias=False)


def _quantized_block():
    mx.random.seed(0)
    model = Block(8, 256)
    w = mx.random.normal((8, 256))
    wq, scales = kq.quantize(w, "q8_0")
    hf_weights = {"gate.weight": wq, "gate.scales": scales}
    meta = {"gate.weight": "q8_0", "proj.weight": "q8_0"}
    return model, wq, scales, hf_weights, meta


def test_small_raw_leaf_dequantized():
    model, wq, scales, hf_weights, meta = _quantized_block()
    handled = dequantize_unattachable_leaves(model, hf_weights, meta)

    assert handled == ["gate (q8_0)"]
    # Gate is now a plain f32 tensor with the scales sidecar dropped; the
    # Linear leaf stays codec'd for install_kquant_modules.
    assert "gate.weight" not in meta
    assert meta == {"proj.weight": "q8_0"}
    assert "gate.scales" not in hf_weights
    deq = hf_weights["gate.weight"]
    assert deq.shape == (8, 256)
    assert deq.dtype == mx.float32
    ref = kq.dequantize(wq, scales, "q8_0", mx.float32).reshape(8, 256)
    assert mx.array_equal(deq, ref)

    n = install_kquant_modules(model, meta)
    assert n == 1  # proj swapped; gate left as the raw float carrier


def test_large_raw_leaf_still_fails_loud():
    model, _, _, hf_weights, meta = _quantized_block()
    handled = dequantize_unattachable_leaves(
        model, hf_weights, meta, max_bytes=1024)

    assert handled == []
    assert meta["gate.weight"] == "q8_0"
    assert "gate.scales" in hf_weights
    with pytest.raises(ValueError, match="no recognized module class"):
        install_kquant_modules(model, meta)
