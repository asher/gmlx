"""Regression test for the glm-dsa / DeepSeek-V3.2 mask-path decode mitigation
(``dsv32_patches._patch_dsv32_mask_decode``).

mlx-lm's ``DeepseekV32Attention`` applies the DSA indexer top-k by GATHERING keys on
the L==1 decode step but MASKING full keys on the L>1 prefill step. The mask path
shipped as a corruption fix on an early stack; the corruption no longer reproduces,
so stock gather is the default and the mask path is an opt-in mitigation
(``GMLX_DSV32_MASK_DECODE=1``).

No GGUF / no GPU: builds a tiny random-weight stock ``glm_moe_dsa`` model and checks
(1) the opt-in installs + flags every attention module, (2) the mask rewrite
reproduces the stock decode logits at small scale, exercising the indexer's sparse
branch, (3) the default and an explicit ``0`` both leave the model on stock gather,
and (4) unflagged instances stay on the stock path even with the class patched.
"""

import mlx.core as mx
import pytest

import gmlx.dsv32_patches as dsv32_patches
from mlx_lm.models.deepseek_v32 import DeepseekV32Attention
from mlx_lm.models.glm_moe_dsa import Model, ModelArgs


@pytest.fixture(autouse=True)
def _cpu_device():
    # CPU numerics by design; restore so the flip never leaks to other files.
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _tiny_args(index_topk: int = 4) -> ModelArgs:
    return ModelArgs.from_dict(
        {
            "model_type": "glm_moe_dsa",
            "vocab_size": 128,
            "hidden_size": 32,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "index_topk": index_topk,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "routed_scaling_factor": 1.0,
            "kv_lora_rank": 16,
            "q_lora_rank": 24,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8,
            "qk_nope_head_dim": 8,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
            "n_group": 1,
            "topk_group": 1,
            "num_experts_per_tok": 2,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000.0},
            "attention_bias": False,
        }
    )


def _build(index_topk: int = 4) -> Model:
    m = Model(_tiny_args(index_topk=index_topk))
    m.eval()
    mx.eval(m.parameters())
    return m


def _decode_logits(model: Model, prefill: int, n: int) -> mx.array:
    """Prefill ``prefill`` tokens then decode ``n`` more one at a time (L==1),
    returning the stacked per-step logits. With prefill > index_topk the indexer's
    sparse branch fires at every decode step."""
    total = prefill + n
    seq = mx.array([[(i * 7 + 3) % 128 for i in range(total)]], dtype=mx.int32)
    cache = model.make_cache()
    model(seq[:, :prefill], cache=cache)
    out = [model(seq[:, i : i + 1], cache=cache) for i in range(prefill, total)]
    return mx.concatenate(out, axis=1)


def _n_attn(model: Model) -> int:
    return sum(isinstance(m, DeepseekV32Attention) for m in model.modules())


def test_patch_installs_and_flags_every_attention(monkeypatch):
    monkeypatch.setenv("GMLX_DSV32_MASK_DECODE", "1")
    model = _build()
    n_attn = _n_attn(model)
    assert n_attn > 0

    dsv32_patches._patch_dsv32_mask_decode(model)

    assert dsv32_patches._MASK_DECODE_PATCH.installed
    assert DeepseekV32Attention.__call__ is dsv32_patches._dsv32_mask_decode_call
    flagged = sum(
        getattr(m, "_dsv32_mask_decode", False)
        for m in model.modules()
        if isinstance(m, DeepseekV32Attention)
    )
    assert flagged == n_attn


def test_mask_decode_matches_stock_gather_at_small_scale(monkeypatch):
    monkeypatch.setenv("GMLX_DSV32_MASK_DECODE", "1")
    # index_topk small + prefill above it -> sparse indexer branch fires every step.
    prefill, n = 6, 5
    model = _build(index_topk=4)

    # Stock baseline: instances unflagged -> stock-gather fallback (regardless of
    # whether the class patch was already installed by another test).
    for m in model.modules():
        if isinstance(m, DeepseekV32Attention):
            m._dsv32_mask_decode = False
    stock = _decode_logits(model, prefill, n)

    dsv32_patches._patch_dsv32_mask_decode(model)  # flips the flag on -> mask path
    fixed = _decode_logits(model, prefill, n)

    d = float(mx.max(mx.abs(fixed - stock)).item())
    assert d < 1e-4, f"mask decode diverges from stock gather at small scale: {d}"
    assert bool(mx.all(mx.argmax(fixed, -1) == mx.argmax(stock, -1)).item())


def test_default_off_skips_patch(monkeypatch):
    # Stock gather decode is the default; the mask path is opt-in.
    monkeypatch.delenv("GMLX_DSV32_MASK_DECODE", raising=False)
    model = _build()
    dsv32_patches._patch_dsv32_mask_decode(model)
    assert not any(
        getattr(m, "_dsv32_mask_decode", False)
        for m in model.modules()
        if isinstance(m, DeepseekV32Attention)
    )


def test_explicit_zero_skips_patch(monkeypatch):
    monkeypatch.setenv("GMLX_DSV32_MASK_DECODE", "0")
    model = _build()
    dsv32_patches._patch_dsv32_mask_decode(model)
    assert not any(
        getattr(m, "_dsv32_mask_decode", False)
        for m in model.modules()
        if isinstance(m, DeepseekV32Attention)
    )


def test_unflagged_instance_uses_stock_path(monkeypatch):
    # Even with the class patch installed, an unflagged instance must hit the stock
    # fallback. Compare an all-unflagged model's decode to the saved stock call run
    # directly on the same modules.
    monkeypatch.setenv("GMLX_DSV32_MASK_DECODE", "1")
    model = _build(index_topk=4)
    dsv32_patches._patch_dsv32_mask_decode(model)  # ensure class is patched
    for m in model.modules():
        if isinstance(m, DeepseekV32Attention):
            m._dsv32_mask_decode = False
    unflagged = _decode_logits(model, 6, 4)

    assert dsv32_patches._MASK_DECODE_PATCH.stock is not None
    # Re-run through the explicit stock call to confirm identical behaviour.
    stock_call = dsv32_patches._MASK_DECODE_PATCH.stock
    patched_call = DeepseekV32Attention.__call__
    try:
        DeepseekV32Attention.__call__ = stock_call
        direct = _decode_logits(model, 6, 4)
    finally:
        DeepseekV32Attention.__call__ = patched_call
    assert float(mx.max(mx.abs(unflagged - direct)).item()) < 1e-6
