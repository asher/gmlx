"""Granite Switch adapter deltas through mlx-kquant's stacked LoRA epilogue
(``lora_ids`` and ``lora_table`` on a dense K-quant matmul) against the
GatherMM path, on the GPU."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

import gmlx.models.granite_switch.model as gs

pytestmark = pytest.mark.skipif(
    not gs._LORA_STACK or not mx.metal.is_available()
    or mx.default_device() != mx.gpu,
    reason="needs mlx-kquant's stacked LoRA epilogue on the GPU")


def _q8_linear(n_out, n_in, rng):
    from gguf import GGMLQuantizationType as GT
    from gguf import quants
    from mlx_kquant.nn import KQuantLinear

    w = (rng.standard_normal((n_out, n_in)) * 0.05).astype(np.float32)
    lin = KQuantLinear(in_dims=n_in, out_dims=n_out, bias=False, codec="q8_0")
    lin.weight = mx.array(quants.quantize(w, GT.Q8_0).astype(np.uint8))
    return lin


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("rows", [1, 3, 24])
def test_epilogue_matches_gather_path(dtype, rows, monkeypatch):
    args = gs.ModelArgs(
        model_type="granite_switch", hidden_size=256, num_hidden_layers=1,
        intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-6, vocab_size=64, logits_scaling=1.0,
        attention_multiplier=0.125, embedding_multiplier=1.0,
        residual_multiplier=1.0, num_adapters=2, max_lora_rank=16,
        adapter_token_ids=[60, 61], adapter_substitute_ids=[5, 9])
    rng = np.random.default_rng(0)
    lora = gs.SwitchLoRA(args)
    lora.load_weights([
        (k, mx.array((rng.standard_normal(v.shape) * 0.1).astype(np.float32))
         .astype(dtype))
        for k, v in lora.parameters().items()])
    n_in, n_out, _ = gs.lora_shapes(args)["qkv"]
    base = _q8_linear(n_out, n_in, rng)
    x = mx.array((rng.standard_normal((1, rows, n_in)) * 0.5)
                 .astype(np.float32)).astype(dtype)
    slots = mx.array(rng.integers(0, 3, (1, rows)).astype(np.uint32))
    route = gs.SlotRoute(slots)
    fused = lora.project("qkv", base, x, route)
    monkeypatch.setattr(gs.SwitchLoRA, "EPILOGUE_MAX_ROWS", 0)
    plain = lora.project("qkv", base, x, route)
    bare = base(x)
    mx.eval(fused, plain, bare)
    f, p, b = (np.array(t.astype(mx.float32)) for t in (fused, plain, bare))
    # Slot-0 rows take no delta at all.
    zero = np.array(slots[0]) == 0
    assert np.array_equal(f[0, zero], b[0, zero])
    delta = p - b
    err = np.linalg.norm((f - b) - delta) / (np.linalg.norm(delta) + 1e-6)
    assert err < 2e-2
