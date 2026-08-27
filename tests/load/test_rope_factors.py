"""Llama-3.x rope-factor restoration.

llama.cpp converts the HF "llama3" rope_scaling config into a per-dim factors
tensor (``rope_freqs.weight``) and writes NO scaling KV, so a synthesized
config gets a plain unscaled rope - coherent below the 8k original context,
degenerate beyond it (the gated ``test_long_prefill_parity[llama]`` failure).
The loader copies the tensor out and rebuilds each attention rope from it;
these tests anchor that reconstruction to mlx-lm's ``Llama3RoPE`` reference.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.loader import _FactoredRoPE, _patch_rope_factors


def _llama3_factors(dims: int, base: float, cfg: dict) -> np.ndarray:
    """The factors tensor exactly as llama.cpp's convert script computes it
    (the HF llama3 formula: divisor per rotary dim)."""
    factor = cfg["factor"]
    low, high = cfg["low_freq_factor"], cfg["high_freq_factor"]
    old_ctx = cfg["original_max_position_embeddings"]
    inv_freq = base ** (-np.arange(0, dims, 2, dtype=np.float64) / dims)
    wavelen = 2 * math.pi / inv_freq
    out = np.ones(dims // 2)
    for i, w in enumerate(wavelen):
        if w > old_ctx / low:            # low-freq band: full stretch
            out[i] = factor
        elif w > old_ctx / high:         # medium band: smooth ramp
            smooth = (old_ctx / w - low) / (high - low)
            out[i] = 1.0 / ((1 - smooth) / factor + smooth)
    return out.astype(np.float32)


# Llama-3.2 values (Llama-3.1 uses factor=8.0; same formula).
_CFG = {"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192}
_DIMS, _BASE = 64, 500000.0


def test_factored_rope_matches_llama3_reference():
    from mlx_lm.models.rope_utils import Llama3RoPE

    ref = Llama3RoPE(_DIMS, traditional=False, base=_BASE,
                     scaling_config=dict(_CFG, rope_type="llama3"))
    factors = mx.array(_llama3_factors(_DIMS, _BASE, _CFG))
    ours = _FactoredRoPE(_DIMS, _BASE, False, 1.0, factors)

    x = mx.random.normal((1, 2, 8, _DIMS))
    for offset in (0, 4096, 16384, 131000):
        a, b = ref(x, offset=offset), ours(x, offset=offset)
        assert mx.allclose(a, b, atol=1e-4, rtol=1e-4), f"offset={offset}"


def test_patch_swaps_matching_ropes_only():
    factors = mx.array(_llama3_factors(_DIMS, _BASE, _CFG))

    class Attn(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.rope = nn.RoPE(dims, traditional=False, base=_BASE)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Attn(_DIMS), Attn(_DIMS)]
            self.other = Attn(_DIMS * 2)  # width mismatch: must be untouched

    model = Toy()
    _patch_rope_factors(model, factors)
    assert all(isinstance(lyr.rope, _FactoredRoPE) for lyr in model.layers)
    assert type(model.other.rope) is nn.RoPE

    # Scaling only bites past the original 8k context: identical rotation at
    # offset 0 for the high-freq (factor 1.0) dims, different deep in.
    x = mx.random.normal((1, 1, 4, _DIMS))
    deep_ref = nn.RoPE(_DIMS, traditional=False, base=_BASE)(x, offset=32768)
    deep_ours = model.layers[0].rope(x, offset=32768)
    assert not mx.allclose(deep_ref, deep_ours, atol=1e-3)


def test_all_ones_factors_are_a_no_op():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = nn.RoPE(_DIMS, traditional=False, base=_BASE)

    m = Attn()
    _patch_rope_factors(m, mx.ones(_DIMS // 2))
    assert type(m.rope) is nn.RoPE


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_ROPE_FACTORS", "0")

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = nn.RoPE(_DIMS, traditional=False, base=_BASE)

    m = Attn()
    _patch_rope_factors(m, mx.array(_llama3_factors(_DIMS, _BASE, _CFG)))
    assert type(m.rope) is nn.RoPE
