#!/usr/bin/env python3
"""DiffusionGemma plumbing: arch gate, config synthesis, remap, and build.

DiffusionGemma is an encoder-decoder *block-diffusion* model on the Gemma-4 MoE
backbone - non-autoregressive, and its model class lives in mlx-vlm, not mlx-lm.
These CPU tests pin the gmlx side of loading it: the GGUF arch maps and
synthesizes into a nested config the mlx-vlm ``diffusion_gemma.Model`` accepts,
and the remap table targets that model's exact (``model.decoder.*`` /
``model.encoder.*``) parameter tree. No model is loaded and no GPU kernel runs;
the real 26B weights + denoising generation are exercised by the integration
tier.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.diffusion_gemma")

from mlx.utils import tree_flatten  # noqa: E402

from gmlx import config_synth  # noqa: E402
from gmlx.config_synth import supported_arches, synthesize_config  # noqa: E402
from gmlx.loader import _dequantize_diffusion_embedding, build_model  # noqa: E402
from gmlx.remap import parse_gguf_name  # noqa: E402

ARCH = "diffusion-gemma"
VOCAB = 64


def _tiny_meta() -> dict:
    """Minimal GGUF KV metadata for a tiny all-sliding DiffusionGemma.

    All-sliding (no full-attention layer) keeps the fixture free of the
    K-eq-V tensor-shape derivation the Gemma-4 synth does for full layers - that
    path is covered by the remap test (which builds a real full layer) and the
    integration tier. Carries the diffusion knobs (canvas length + entropy-bound
    sampler) the synth lifts into ``generation_config``.
    """
    return {
        "general.architecture": ARCH,
        f"{ARCH}.embedding_length": 32,
        f"{ARCH}.block_count": 3,
        f"{ARCH}.attention.head_count": 4,
        f"{ARCH}.attention.head_count_kv": 2,
        f"{ARCH}.feed_forward_length": 48,
        f"{ARCH}.context_length": 1024,
        f"{ARCH}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{ARCH}.attention.key_length": 16,        # full-attn dim -> global_head_dim
        f"{ARCH}.attention.key_length_swa": 8,     # sliding dim -> head_dim
        f"{ARCH}.attention.shared_kv_layers": 0,
        f"{ARCH}.attention.sliding_window": 8,
        f"{ARCH}.final_logit_softcapping": 30.0,
        f"{ARCH}.attention.sliding_window_pattern": [True, True, True],
        f"{ARCH}.rope.freq_base": 10000.0,
        f"{ARCH}.rope.freq_base_swa": 10000.0,
        f"{ARCH}.expert_count": 4,
        f"{ARCH}.expert_used_count": 2,
        f"{ARCH}.expert_feed_forward_length": 16,
        "diffusion.canvas_length": 16,
        "diffusion.eb_max_steps": 8,
        "diffusion.eb_t_min": 0.4,
        "diffusion.eb_t_max": 0.8,
        "diffusion.eb_entropy_bound": 0.1,
        "diffusion.eb_stability_threshold": 1,
        "diffusion.eb_confidence_threshold": 0.005,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.tokens": ["t"] * VOCAB,
    }


def test_arch_is_supported():
    """The loader gate admits diffusion-gemma (no ``--force`` needed)."""
    assert ARCH in supported_arches()
    assert config_synth.GGUF_ARCH_TO_MODEL_TYPE[ARCH] == "diffusion_gemma"


def test_config_synth_nests_text_and_diffusion_fields():
    cfg = synthesize_config(_tiny_meta(), {})

    # Top level is the mlx-vlm Model config, not a flat text config.
    assert cfg["model_type"] == "diffusion_gemma"
    assert cfg["canvas_length"] == 16
    # eos is the canonical DiffusionGemma stop list (scalar GGUF eos + the
    # turn-end ids 106/50); top-level too (server resets stop off it).
    assert cfg["eos_token_id"] == [1, 106, 50]

    tc = cfg["text_config"]
    assert tc["model_type"] == "diffusion_gemma_text"
    assert tc["hidden_size"] == 32
    assert tc["num_hidden_layers"] == 3
    assert tc["vocab_size"] == VOCAB
    assert tc["tie_word_embeddings"] is True       # no output.weight in shapes
    # Gemma-4 MoE fields the shared synth produced under the diffusion prefix.
    assert tc["num_experts"] == 4
    assert tc["top_k_experts"] == 2
    assert tc["moe_intermediate_size"] == 16
    assert tc["global_head_dim"] == 16
    assert tc["head_dim"] == 8
    assert tc["final_logit_softcapping"] == 30.0
    assert tc["layer_types"] == ["sliding_attention"] * 3

    # Entropy-bound sampler lifted out of the diffusion.* metadata.
    gen = cfg["generation_config"]
    assert gen["eos_token_id"] == [1, 106, 50]
    assert gen["max_denoising_steps"] == 8
    assert gen["linear_temperature_schedule_config"] == {"t_min": 0.4, "t_max": 0.8}
    assert gen["sampler_config"]["entropy_bound"] == 0.1
    assert gen["diffusion_stopping_config"] == {
        "confidence_threshold": 0.005,
        "stability_threshold": 1,
    }


def test_config_synth_bakes_sampler_defaults_when_gguf_lacks_eb_keys():
    """The real llama.cpp conversion drops every ``diffusion.eb_*`` key (it keeps
    only ``diffusion.canvas_length`` + the scalar eos). The synth must still emit
    a complete entropy-bound ``generation_config`` from the canonical defaults -
    most critically ``diffusion_stopping_config`` (else the denoiser runs the full
    48-step schedule with no early stop, ~3-4x slower) and the turn-end eos ids
    (else the canvas never stops on ``<end_of_turn>`` and runs to max_tokens)."""
    meta = _tiny_meta()
    for k in list(meta):
        if k.startswith("diffusion.eb_"):
            del meta[k]
    assert "diffusion.canvas_length" in meta  # the one diffusion knob that survives

    gen = synthesize_config(meta, {})["generation_config"]
    assert gen["eos_token_id"] == [1, 106, 50]
    assert gen["max_denoising_steps"] == 48
    assert gen["linear_temperature_schedule_config"] == {"t_min": 0.4, "t_max": 0.8}
    assert gen["sampler_config"] == {
        "_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1}
    assert gen["diffusion_stopping_config"] == {
        "confidence_threshold": 0.005, "stability_threshold": 1}


def test_build_model_instantiates_mlx_vlm_diffusion_class():
    """The synthesized config drives the real mlx-vlm Model __init__ (no eval).

    This is the strongest plumbing guard: a missing or mis-named text field that
    ``ModelConfig.from_dict`` / ``Model.__init__`` needs fails right here, on
    tiny dims, with no weights and no GPU.
    """
    cfg = synthesize_config(_tiny_meta(), {})
    model, returned = build_model(cfg)
    assert returned["model_type"] == "diffusion_gemma"

    import mlx_vlm.models.diffusion_gemma as dg

    assert isinstance(model, dg.Model)
    # canvas_length on the live config is what mlx-vlm's generate dispatch keys
    # on to route into the denoiser instead of the autoregressive loop.
    from mlx_vlm.generate.diffusion import is_diffusion_model

    assert is_diffusion_model(model) is True


def test_remap_targets_match_the_model_parameter_tree():
    """Every GGUF tensor a real conversion writes maps to a live model param -
    nothing unmapped, nothing mis-homed to a non-existent slot - and an unknown
    tensor fails closed (no silent fall-through to the gemma-4 canonical map)."""
    import mlx_vlm.models.diffusion_gemma as dg

    text = dict(
        model_type="diffusion_gemma_text", vocab_size=VOCAB, hidden_size=32,
        intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
        head_dim=8, global_head_dim=16, num_experts=4, top_k_experts=2,
        sliding_window=8,
    )
    model = dg.Model(dg.ModelConfig.from_dict(
        {"model_type": "diffusion_gemma", "text_config": text, "canvas_length": 16}))
    params = {k for k, _ in tree_flatten(model.parameters())}

    # Layer 2 is full-attention -> K-eq-V -> no v_proj; the conversion omits attn_v
    # there too, so drive that off the live param tree.
    per_layer = [
        "attn_q.weight", "attn_k.weight", "attn_output.weight", "attn_q_norm.weight",
        "attn_k_norm.weight", "attn_norm.weight", "post_attention_norm.weight",
        "ffn_norm.weight", "pre_ffw_norm_2.weight", "post_ffw_norm.weight",
        "post_ffw_norm_1.weight", "post_ffw_norm_2.weight", "ffn_gate.weight",
        "ffn_up.weight", "ffn_down.weight", "ffn_gate_inp.weight", "ffn_gate_inp.scale",
        "ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_down_exps.scale",
        "layer_output_scale", "enc_layer_output_scale",
    ]
    gguf_names = [
        "token_embd.weight", "output_norm.weight", "self_cond_pre_norm.weight",
        "self_cond_gate.weight", "self_cond_up.weight", "self_cond_down.weight",
    ]
    for layer in range(3):
        gguf_names += [f"blk.{layer}.{t}" for t in per_layer]
        if f"model.decoder.layers.{layer}.self_attn.v_proj.weight" in params:
            gguf_names.append(f"blk.{layer}.attn_v.weight")

    targets = set()
    for name in gguf_names:
        dec = parse_gguf_name(ARCH, name)
        assert dec.kind == dec.KIND_MAP, f"{name} did not map: {dec.reason}"
        targets.add(dec.hf_name)

    assert targets <= params, f"mapped to non-existent slots: {sorted(targets - params)}"
    assert params <= targets, f"model params with no GGUF source: {sorted(params - targets)}"

    # Fail-closed: an unrecognized tensor must NOT silently resolve via the shared
    # gemma-4 canonical templates (which would mis-home it to `model.layers.*`).
    unknown = parse_gguf_name(ARCH, "blk.0.some_future_tensor.weight")
    assert unknown.kind == unknown.KIND_FAIL

    # ...but an expected never-loaded tensor (precomputed RoPE freqs) is an
    # explicit skip, not a failure - even under the fail-closed guard.
    rope = parse_gguf_name(ARCH, "rope_freqs.weight")
    assert rope.kind == rope.KIND_SKIP


def test_is_diffusion_model_detects_the_built_model():
    """``run``/``chat`` key on this to route into the denoiser; it must fire on a
    real diffusion model and stay quiet on an ordinary one."""
    import types

    from gmlx.diffusion import is_diffusion_model

    model, _ = build_model(synthesize_config(_tiny_meta(), {}))
    assert is_diffusion_model(model) is True
    assert is_diffusion_model(
        types.SimpleNamespace(config=types.SimpleNamespace())) is False


def test_dequantize_diffusion_embedding_swaps_kquant_to_bf16():
    """The denoiser's ``probs @ embed_tokens.weight`` soft-embedding step needs a
    dense float table, so the loader replaces the kquant ``embed_tokens`` with a
    bf16 ``nn.Embedding`` post-load. Verify the swap and that its values match a
    reference dequant of the same q8_0 wire bytes (CPU-only, tiny table)."""
    import types

    import mlx.nn as nn
    import numpy as np
    from gguf.constants import GGMLQuantizationType as GT
    from gguf.quants import dequantize, quantize

    from gmlx.modules import KQuantEmbedding

    rows, dims = 80, 64
    ref_tab = (np.random.RandomState(0).randn(rows, dims) * 0.1).astype(np.float32)
    qbytes = quantize(ref_tab, GT.Q8_0)              # (rows, bytes_per_row) uint8
    ref_deq = dequantize(qbytes, GT.Q8_0).astype(np.float32)

    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        emb = KQuantEmbedding(rows, dims, "q8_0")
        emb.weight = mx.array(qbytes)
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                decoder=types.SimpleNamespace(embed_tokens=emb)))
        _dequantize_diffusion_embedding(model, lambda *_: None)

        new = model.model.decoder.embed_tokens
        assert isinstance(new, nn.Embedding)
        assert not isinstance(new, KQuantEmbedding)
        assert new.weight.shape == (rows, dims)
        assert new.weight.dtype == mx.bfloat16
        got = np.array(new.weight.astype(mx.float32))
        # Same bytes both ways -> only bf16 rounding (~0.4% rel) separates them.
        assert np.allclose(got, ref_deq, atol=2e-3, rtol=0.02)

        # A plain (already-float) embedding is left untouched.
        plain = nn.Embedding(rows, dims)
        model2 = types.SimpleNamespace(
            model=types.SimpleNamespace(
                decoder=types.SimpleNamespace(embed_tokens=plain)))
        _dequantize_diffusion_embedding(model2, lambda *_: None)
        assert model2.model.decoder.embed_tokens is plain
    finally:
        mx.set_default_device(prev)
