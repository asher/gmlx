"""Codec'd tensors on raw-array carrier modules dequantize to f32 at load.

Some community GGUF quantizers codec the MoE router gate (llama.cpp's own
quantize leaves it F32). The deepseek-family MoEGate holds that weight as a
raw array, so no KQuant* module can attach; the loader reconstructs the float
tensor instead of failing. The same applies to raw leaves under names other
than ``weight`` (hyper-connection ``fn``). Large unattachable tensors fail
loud in the dequantizer itself - the installer's own loud path only covers
``.weight``-keyed leaves, so deferring would be silent for the rest.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import pytest

from gmlx.load.modules import dequantize_unattachable_leaves, install_kquant_modules


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

    assert handled == ["gate.weight (q8_0)"]
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
    with pytest.raises(ValueError, match="too large to dequantize"):
        dequantize_unattachable_leaves(
            model, hf_weights, meta, max_bytes=1024)

    # Nothing consumed before the raise; the wire stays intact for triage.
    assert meta["gate.weight"] == "q8_0"
    assert "gate.scales" in hf_weights


def test_nonweight_raw_leaf_dequantized():
    # Raw leaves under names other than ``weight`` (hyper-connection fn):
    # scales sidecar lives at ``<key>.scales``, not ``<module>.scales``.
    class HC(nn.Module):
        def __init__(self):
            super().__init__()
            self.fn = mx.zeros((8, 256))
            self.base = mx.zeros((8,))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_hc = HC()

    mx.random.seed(1)
    model = Layer()
    w = mx.random.normal((8, 256))
    wq, scales = kq.quantize(w, "q8_0")
    hf_weights = {"attn_hc.fn": wq, "attn_hc.fn.scales": scales,
                  "attn_hc.base": mx.zeros((8,))}
    meta = {"attn_hc.fn": "q8_0"}
    handled = dequantize_unattachable_leaves(model, hf_weights, meta)

    assert handled == ["attn_hc.fn (q8_0)"]
    assert meta == {}
    assert "attn_hc.fn.scales" not in hf_weights
    deq = hf_weights["attn_hc.fn"]
    assert deq.shape == (8, 256) and deq.dtype == mx.float32
    ref = kq.dequantize(wq, scales, "q8_0", mx.float32).reshape(8, 256)
    assert mx.array_equal(deq, ref)
