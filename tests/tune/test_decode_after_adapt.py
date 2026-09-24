"""A model adapted in process after load decodes through its adapters.
The loader builds fused weights from the base projections (the b/a
concat of a gated-delta layer, the batched-decode wires of the occupancy
fuse), and a wrapped projection must take its own forward instead."""
from __future__ import annotations

import os

import mlx.core as mx
import pytest

from .tiny_train_archs import build, eval_logits, process_patches, rel

pytestmark = pytest.mark.skipif(
    os.environ.get("KQUANT_FORCE_CPU") == "1" or not mx.metal.is_available()
    or mx.default_device() != mx.gpu, reason="the fused decode kernels run on the GPU only")


def _float_qwen35():
    from gmlx.load.loader import build_model

    mx.random.seed(0)
    model, _ = build_model(dict(
        model_type="qwen3_5", hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, rms_norm_eps=1e-6, vocab_size=512,
        max_position_embeddings=4096, linear_num_value_heads=4,
        linear_num_key_heads=4, linear_key_head_dim=64,
        linear_value_head_dim=64, linear_conv_kernel_dim=4,
        tie_word_embeddings=False, full_attention_interval=2,
        rope_parameters={"type": "default", "mrope_section": [3, 3, 2],
                         "rope_theta": 1e7, "partial_rotary_factor": 0.25}))
    model.set_dtype(mx.bfloat16)
    model.eval()
    return model


def _randomize_lora_b(model, std):
    from mlx_lm.tuner.lora import LoRALinear

    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_b = mx.random.normal(m.lora_b.shape, key=mx.random.key(n)) * std
            n += 1
    return n


def test_fused_gdn_decode_serves_adapted_b_and_a():
    from gmlx.tune.lora import prepare_lora_student
    from gmlx.upstream.gdn_patches import _patch_gated_delta_fused_decode

    model = _float_qwen35()
    _patch_gated_delta_fused_decode(model)
    assert any(getattr(m, "_gdn_ba_weight", None) is not None for m in model.modules())
    prepare_lora_student(model, rank=4, scale=20.0, num_layers=None,
                         keys=["linear_attn.in_proj_b", "linear_attn.in_proj_a"])
    assert _randomize_lora_b(model, 0.5) == 2

    tokens = mx.random.randint(0, 512, (2, 12), key=mx.random.key(7))
    full, dec = eval_logits(model, tokens)
    assert rel(full[:, -1:], dec) < 2e-2


def test_occupancy_wires_built_before_the_adapter_serve_it(monkeypatch):
    from gmlx.tune.lora import LORA_KEYS, prepare_lora_student

    monkeypatch.setenv("GMLX_OCCUPANCY_FUSE", "1")
    with process_patches():
        model, _, _ = build("qwen3_5")
        tokens = mx.random.randint(1, 512, (2, 12), key=mx.random.key(7))
        base_full, _ = eval_logits(model, tokens)   # a batched decode builds the wires
        wires = [m for m in model.modules()
                 if getattr(m, "_kq_wqkv", None) is not None
                 or getattr(m, "_kq_wgu", None) is not None]
        assert wires
        prepare_lora_student(model, rank=4, scale=20.0, num_layers=None, keys=LORA_KEYS)
        assert _randomize_lora_b(model, 0.02) > 0
        full, dec = eval_logits(model, tokens)
        monkeypatch.setenv("GMLX_OCCUPANCY_FUSE", "0")
        _, stock = eval_logits(model, tokens)
    assert rel(full, base_full) > 0.1
    assert rel(stock, dec) < 1e-3
