#!/usr/bin/env python3
"""Config synthesis: per-arch field correctness, real mlx-lm instantiation, and
the dual-mode (GGUFReader vs decoded-dict) equality.

The instantiation tests are the strongest regression guard: a synthesized
config that's missing or mis-naming a field that mlx-lm's ``ModelArgs`` /
``Model.__init__`` needs fails here, on tiny dims, with no GPU and no model
download. Hybrid/MoE arches (gemma4, qwen35*, mistral3, nemotron) need
intricate tensor-shape fixtures to drive their derivations and are covered by
the integration tests instead.
"""

from __future__ import annotations

import pytest

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.utils import _get_classes  # noqa: E402

from gmlx.config_synth import synthesize_config  # noqa: E402

VOCAB = 32


def _base_meta(arch: str) -> dict:
    """Minimal universal-field KV metadata for a tiny dense model."""
    return {
        "general.architecture": arch,
        f"{arch}.embedding_length": 64,
        f"{arch}.block_count": 2,
        f"{arch}.attention.head_count": 4,
        f"{arch}.attention.head_count_kv": 2,
        f"{arch}.feed_forward_length": 128,
        f"{arch}.context_length": 1024,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{arch}.attention.key_length": 16,
        f"{arch}.rope.freq_base": 10000.0,
        "tokenizer.ggml.tokens": ["t"] * VOCAB,
    }


def _meta_for(arch: str) -> dict:
    m = _base_meta(arch)
    if arch == "gemma2":
        m[f"{arch}.attn_logit_softcapping"] = 50.0
        m[f"{arch}.final_logit_softcapping"] = 30.0
        m[f"{arch}.attention.sliding_window"] = 4096
    elif arch == "gemma3":
        m[f"{arch}.attention.sliding_window"] = 512
        m[f"{arch}.attention.sliding_window_pattern"] = 6
        # mlx-lm's gemma3 forward indexes cache[pattern-1], so the tiny model
        # needs at least `pattern` layers to exercise a real forward.
        m[f"{arch}.block_count"] = 6
    elif arch == "llama":
        m[f"{arch}.rope.dimension_count"] = 16
    elif arch == "glm4":
        # partial rotary: rotate 8 of head_dim=16 (key_length) -> factor 0.5.
        m[f"{arch}.rope.dimension_count"] = 8
    elif arch == "granite":
        # the four runtime multipliers granite.ModelArgs requires (no defaults).
        m[f"{arch}.embedding_scale"] = 12.0
        m[f"{arch}.residual_scale"] = 1.0
        m[f"{arch}.attention.scale"] = 0.015625
        m[f"{arch}.logit_scale"] = 6.0
    return m


# model_type each arch must resolve to, for the instantiation sweep.
DENSE_ARCHES = ["qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3", "glm4",
                "llama", "seed_oss", "smollm3", "granite"]


@pytest.mark.parametrize("arch", DENSE_ARCHES)
def test_synth_config_instantiates_in_mlx_lm(arch):
    config = synthesize_config(_meta_for(arch), tensor_shapes={})
    Model, ModelArgs = _get_classes(config)
    model = Model(ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    # a tiny forward proves the synthesized dims (head_dim, kv heads, sliding
    # window, softcaps, rope) are mutually consistent end to end.
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[0] == 1 and out.shape[-1] == VOCAB


def test_qwen2_universal_fields():
    c = synthesize_config(_meta_for("qwen2"), tensor_shapes={})
    assert c["model_type"] == "qwen2"
    assert c["hidden_size"] == 64
    assert c["num_hidden_layers"] == 2
    assert c["num_attention_heads"] == 4
    assert c["num_key_value_heads"] == 2
    assert c["intermediate_size"] == 128
    assert c["head_dim"] == 16
    assert c["vocab_size"] == VOCAB
    # no output.weight in the (empty) inventory => tied.
    assert c["tie_word_embeddings"] is True


def test_untied_when_output_weight_present():
    c = synthesize_config(_meta_for("qwen2"),
                          tensor_shapes={"output.weight": [64, VOCAB]})
    assert c["tie_word_embeddings"] is False


def test_seed_oss_head_dim_from_key_length():
    # Seed-OSS head_dim rides in GGUF as attention.key_length and is *not*
    # hidden//heads on the real model; the synth must surface the key_length value
    # (16 in the base meta), not derive 64//4=16-by-coincidence - so use a
    # key_length that differs from hidden//heads to pin the source.
    m = _base_meta("seed_oss")
    m["seed_oss.attention.key_length"] = 24
    c = synthesize_config(m, tensor_shapes={})
    assert c["model_type"] == "seed_oss"
    assert c["head_dim"] == 24
    # no rope scaling KV => no rope_scaling key.
    assert "rope_scaling" not in c


def test_seed_oss_head_dim_fallback_without_key_length():
    m = _base_meta("seed_oss")
    del m["seed_oss.attention.key_length"]
    c = synthesize_config(m, tensor_shapes={})
    assert c["head_dim"] == 64 // 4  # hidden//heads fallback


def test_seed_oss_attention_bias_from_tensor_presence():
    # Seed-OSS-36B carries q/k/v biases but no output bias; GGUF has no KV
    # flag for them, so both flags derive from tensor presence. Without the
    # flags the mlx-lm class builds bias-less Linears and strict-load fails
    # (or, pre-remap-fix, the bias overwrote the quant weight slot).
    m = _base_meta("seed_oss")
    c = synthesize_config(m, tensor_shapes={"blk.0.attn_q.bias": [16]})
    assert c["attention_bias"] is True
    assert c["attention_out_bias"] is False
    c = synthesize_config(m, tensor_shapes={})
    assert c["attention_bias"] is False
    assert c["attention_out_bias"] is False


def test_smollm3_nope_default_interval_4():
    # llama.cpp hardcodes n_no_rope_layer_step=4 (no GGUF KV); mlx-lm's default
    # no_rope_layer_interval=4 + formula must match: use_rope = (i+1)%4 != 0, so
    # over 4 layers only the last (index 3) is NoPE.
    m = _base_meta("smollm3")
    m["smollm3.block_count"] = 4
    c = synthesize_config(m, tensor_shapes={})
    assert c["model_type"] == "smollm3"
    _Model, ModelArgs = _get_classes(c)
    args = ModelArgs.from_dict(c)
    assert args.no_rope_layers == [1, 1, 1, 0]


def test_smollm3_nope_interval_override():
    m = _base_meta("smollm3")
    m["smollm3.block_count"] = 4
    m["smollm3.attention.n_no_rope_layer_step"] = 2
    c = synthesize_config(m, tensor_shapes={})
    assert c["no_rope_layer_interval"] == 2
    _Model, ModelArgs = _get_classes(c)
    args = ModelArgs.from_dict(c)
    assert args.no_rope_layers == [1, 0, 1, 0]


def test_granite_multipliers_and_bias_inference():
    # The four runtime multipliers must reach the config exactly (mlx-lm applies
    # them at runtime; a wrong/missing value silently mis-scales the model).
    c = synthesize_config(_meta_for("granite"), tensor_shapes={})
    assert c["model_type"] == "granite"
    assert c["embedding_multiplier"] == 12.0
    assert c["residual_multiplier"] == 1.0
    assert c["attention_multiplier"] == 0.015625
    assert c["logits_scaling"] == 6.0
    assert c["attention_bias"] is False and c["mlp_bias"] is False
    # bias inferred from tensor presence
    c2 = synthesize_config(_meta_for("granite"),
                           tensor_shapes={"blk.0.attn_q.bias": [64]})
    assert c2["attention_bias"] is True


def test_granite_missing_multiplier_raises():
    m = _meta_for("granite")
    del m["granite.logit_scale"]
    with pytest.raises(ValueError, match="logit_scale"):
        synthesize_config(m, tensor_shapes={})


def test_gemma2_specific_fields():
    c = synthesize_config(_meta_for("gemma2"), tensor_shapes={})
    # query_pre_attn_scalar == head_dim for non-27B sizes (27B uses hidden//heads).
    assert c["query_pre_attn_scalar"] == c["head_dim"] == 16
    assert c["attn_logit_softcapping"] == 50.0
    assert c["final_logit_softcapping"] == 30.0
    assert c["sliding_window"] == 4096


def _meta_gemma_size(arch: str, *, hidden, heads, kv_heads, head_dim, layers):
    """Real published gemma-2/3 geometry over the base meta (query_pre_attn_scalar
    is derived from these dims, so the tiny defaults won't do)."""
    m = _meta_for(arch)
    m[f"{arch}.embedding_length"] = hidden
    m[f"{arch}.attention.head_count"] = heads
    m[f"{arch}.attention.head_count_kv"] = kv_heads
    m[f"{arch}.attention.key_length"] = head_dim
    m[f"{arch}.block_count"] = layers
    return m


def test_gemma2_27b_query_pre_attn_scalar_is_hidden_over_heads():
    # gemma-2-27b: 46 layers, hidden 4608 / 32 heads = 144 != head_dim 128
    # (gemma_pytorch get_config_for_27b; llama.cpp gemma2.cpp LLM_TYPE_27B).
    m = _meta_gemma_size("gemma2", hidden=4608, heads=32, kv_heads=16,
                         head_dim=128, layers=46)
    c = synthesize_config(m, tensor_shapes={})
    assert c["query_pre_attn_scalar"] == 144
    assert c["head_dim"] == 128


def test_gemma3_27b_query_pre_attn_scalar_is_hidden_over_heads():
    # gemma-3-27b: 62 layers, hidden 5376 / 32 heads = 168 != head_dim 128
    # (gemma_pytorch get_config_for_27b_v3; llama.cpp gemma3.cpp LLM_TYPE_27B).
    m = _meta_gemma_size("gemma3", hidden=5376, heads=32, kv_heads=16,
                         head_dim=128, layers=62)
    c = synthesize_config(m, tensor_shapes={})
    assert c["query_pre_attn_scalar"] == 168
    assert c["head_dim"] == 128


def test_gemma3_12b_query_pre_attn_scalar_stays_head_dim():
    # gemma-3-12b: 48 layers, head_dim 256 (hidden 3840 / 16 heads = 240 must
    # NOT be used - only the 27B scales by hidden//heads).
    m = _meta_gemma_size("gemma3", hidden=3840, heads=16, kv_heads=8,
                         head_dim=256, layers=48)
    c = synthesize_config(m, tensor_shapes={})
    assert c["query_pre_attn_scalar"] == 256


def _meta_gemma_embedding() -> dict:
    """EmbeddingGemma metadata: the gemma3 backbone fields plus the two
    encoder-specific keys (pooling type + bidirectional attention)."""
    a = "gemma-embedding"
    m = _base_meta(a)
    m[f"{a}.attention.sliding_window"] = 512
    m[f"{a}.attention.sliding_window_pattern"] = 6
    m[f"{a}.block_count"] = 6
    m[f"{a}.pooling_type"] = 1            # MEAN
    m[f"{a}.attention.causal"] = False    # bidirectional sentence encoder
    return m


def test_gemma_embedding_synth_fields_and_pooling():
    c = synthesize_config(_meta_gemma_embedding(), tensor_shapes={})
    assert c["model_type"] == "gemma_embedding"     # distinct from the decoder tag
    # backbone fields come straight from the reused gemma3 synth.
    assert c["hidden_size"] == 64
    assert c["num_hidden_layers"] == 6
    assert c["sliding_window"] == 512
    # encoder-specific fields surfaced from the embedding metadata.
    assert c["pooling_type"] == 1
    assert c["attention_causal"] is False


def test_gemma_embedding_synth_omits_absent_encoder_fields():
    m = _meta_gemma_embedding()
    del m["gemma-embedding.pooling_type"]
    del m["gemma-embedding.attention.causal"]
    c = synthesize_config(m, tensor_shapes={})
    assert "pooling_type" not in c
    assert "attention_causal" not in c


def test_gemma_embedding_synth_builds_mlx_embeddings_encoder():
    # The strongest guard: the synthesized config must instantiate the
    # mlx-embeddings encoder and run a tiny forward (dims mutually consistent),
    # mirroring the loader's build_model gemma_embedding branch.
    import importlib

    c = synthesize_config(_meta_gemma_embedding(), tensor_shapes={})
    ge = importlib.import_module("mlx_embeddings.models.gemma3_text")
    model = ge.Model(ge.ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]), attention_mask=mx.ones((1, 3)))
    mx.eval(out.text_embeds)
    assert out.text_embeds.shape == (1, c["hidden_size"])
    assert len(model.dense) == 2                    # the 2-layer projection head


def test_glm4_partial_rotary_and_attention_bias():
    m = _meta_for("glm4")
    # attention_bias is inferred from the presence of a q-proj bias tensor.
    c = synthesize_config(m, tensor_shapes={"blk.0.attn_q.bias": [64]})
    assert c["model_type"] == "glm4"
    assert c["partial_rotary_factor"] == 0.5   # rope.dimension_count 8 / head_dim 16
    assert c["attention_bias"] is True
    c2 = synthesize_config(m, tensor_shapes={})
    assert c2["attention_bias"] is False


def test_qwen3moe_synth_instantiates():
    m = _base_meta("qwen3moe")
    m["qwen3moe.expert_count"] = 4
    m["qwen3moe.expert_used_count"] = 2
    m["qwen3moe.expert_feed_forward_length"] = 64
    c = synthesize_config(m, tensor_shapes={})
    assert c["model_type"] == "qwen3_moe"
    assert c["num_experts"] == 4 and c["num_experts_per_tok"] == 2
    assert c["moe_intermediate_size"] == 64
    assert c["decoder_sparse_step"] == 1 and c["mlp_only_layers"] == []
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_qwen2moe_synth_instantiates():
    # The public Qwen1.5-MoE GGUFs omit expert_feed_forward_length, so
    # moe_intermediate_size is derived from the stacked ffn_gate_exps shape
    # ([hidden, moe_intermediate, n_experts]); shared_expert_intermediate_size
    # falls back to feed_forward_length. Exercise both derivations here.
    m = _base_meta("qwen2moe")
    m["qwen2moe.expert_count"] = 4
    m["qwen2moe.expert_used_count"] = 2
    c = synthesize_config(
        m, tensor_shapes={"blk.0.ffn_gate_exps.weight": [64, 48, 4]})
    assert c["model_type"] == "qwen2_moe"
    assert c["num_experts"] == 4 and c["num_experts_per_tok"] == 2
    assert c["moe_intermediate_size"] == 48          # ffn_gate_exps[1]
    assert c["shared_expert_intermediate_size"] == 128  # feed_forward_length
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_qwen2moe_prefers_expert_ffn_kv_when_present():
    # If a conversion *does* write expert_feed_forward_length, trust it over the
    # tensor shape (and shared length over feed_forward_length).
    m = _base_meta("qwen2moe")
    m["qwen2moe.expert_count"] = 4
    m["qwen2moe.expert_used_count"] = 2
    m["qwen2moe.expert_feed_forward_length"] = 80
    m["qwen2moe.expert_shared_feed_forward_length"] = 96
    c = synthesize_config(
        m, tensor_shapes={"blk.0.ffn_gate_exps.weight": [64, 48, 4]})
    assert c["moe_intermediate_size"] == 80
    assert c["shared_expert_intermediate_size"] == 96


def _ernie_meta() -> dict:
    arch = "ernie4_5-moe"
    m = _base_meta(arch)
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 48
    m[f"{arch}.expert_shared_count"] = 1
    m[f"{arch}.leading_dense_block_count"] = 1   # layer 0 dense, layer 1 MoE
    m[f"{arch}.interleave_moe_layer_step"] = 1
    return m


def test_ernie4_5_moe_synth_instantiates():
    # ERNIE-4.5-MoE: stacked routed experts (SwitchGLU) + a shared expert behind
    # `leading_dense_block_count` dense layers. With block_count=2 / start=1 the
    # tiny model has one dense decoder layer and one MoE layer, exercising both.
    c = synthesize_config(_ernie_meta(), tensor_shapes={})
    assert c["model_type"] == "ernie4_5_moe"
    assert c["moe_num_experts"] == 4 and c["moe_k"] == 2
    assert c["moe_intermediate_size"] == 48
    assert c["moe_num_shared_experts"] == 1
    assert c["moe_layer_start_index"] == 1 and c["moe_layer_interval"] == 1
    assert c["moe_gate_act"] == "softmax"      # no expert_gating_func KV => default
    assert c["use_bias"] is False              # no attn_q.bias tensor
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_ernie4_5_moe_intermediate_from_stacked_tensor():
    # Without expert_feed_forward_length, moe_intermediate_size derives from the
    # stacked ffn_gate_exps shape ([hidden, moe_intermediate, n_experts]).
    m = _ernie_meta()
    del m["ernie4_5-moe.expert_feed_forward_length"]
    c = synthesize_config(
        m, tensor_shapes={"blk.0.ffn_gate_exps.weight": [64, 40, 4]})
    assert c["moe_intermediate_size"] == 40


def test_ernie4_5_moe_sigmoid_gating():
    # expert_gating_func == 2 selects the aux-free sigmoid gate.
    m = _ernie_meta()
    m["ernie4_5-moe.expert_gating_func"] = 2
    c = synthesize_config(m, tensor_shapes={})
    assert c["moe_gate_act"] == "sigmoid" and c["moe_use_aux_free"] is True


def _minimax_meta() -> dict:
    arch = "minimax-m2"
    m = _base_meta(arch)
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 48
    m[f"{arch}.rope.dimension_count"] = 8    # partial rotary (< head_dim 16)
    return m


# untied (minimax always ships output.weight) + the full-width qk-norm present.
_MINIMAX_SHAPES = {
    "output.weight": [64, VOCAB],
    "blk.0.attn_q_norm.weight": [64],
}


def test_minimax_synth_instantiates():
    # Every-layer fine-grained sigmoid MoE (no dense, no shared expert); the
    # expert FFN width IS intermediate_size, the router/experts nest under
    # block_sparse_moe, partial rotary, full-width qk-norm.
    c = synthesize_config(_minimax_meta(), tensor_shapes=_MINIMAX_SHAPES)
    assert c["model_type"] == "minimax"
    assert c["num_local_experts"] == 4 and c["num_experts_per_tok"] == 2
    assert c["intermediate_size"] == 48        # expert FFN width (no dense layers)
    assert c["shared_intermediate_size"] == 0  # required-but-unused
    assert c["scoring_func"] == "sigmoid"
    assert c["rotary_dim"] == 8                 # partial rotary
    assert c["use_qk_norm"] is True
    assert c["tie_word_embeddings"] is False
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_minimax_head_dim_from_key_length_not_hidden_over_heads():
    # head_dim must come from attention.key_length (!= hidden//heads for
    # MiniMax-M2); the full-width qk-norm shape depends on getting it right.
    m = _minimax_meta()
    m["minimax-m2.attention.key_length"] = 32   # hidden//heads would be 16
    c = synthesize_config(m, tensor_shapes=_MINIMAX_SHAPES)
    assert c["head_dim"] == 32


def test_minimax_expert_ffn_from_stacked_tensor():
    # Without expert_feed_forward_length, the expert width (-> intermediate_size)
    # derives from the stacked ffn_gate_exps middle dim.
    m = _minimax_meta()
    del m["minimax-m2.expert_feed_forward_length"]
    shapes = dict(_MINIMAX_SHAPES)
    shapes["blk.0.ffn_gate_exps.weight"] = [64, 40, 4]
    c = synthesize_config(m, tensor_shapes=shapes)
    assert c["intermediate_size"] == 40


def test_minimax_qk_norm_flag_from_tensor_presence():
    # use_qk_norm follows whether the GGUF carries attn_q_norm.
    c = synthesize_config(_minimax_meta(),
                          tensor_shapes={"output.weight": [64, VOCAB]})
    assert c["use_qk_norm"] is False


def _minimax_m3_meta() -> dict:
    arch = "minimax-m3"
    m = _base_meta(arch)
    m[f"{arch}.block_count"] = 3                  # 1 dense + 2 MoE
    m[f"{arch}.leading_dense_block_count"] = 1
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 48
    m[f"{arch}.expert_shared_count"] = 1
    m[f"{arch}.expert_gating_func"] = 2
    m[f"{arch}.expert_weights_scale"] = 2.0
    m[f"{arch}.rope.dimension_count"] = 8         # partial rotary (< head_dim 16)
    return m


# untied + per-head qk-norm ([head_dim]) + a shared expert on the first MoE
# layer (its width has no GGUF KV - read off the tensor shape).
_MINIMAX_M3_SHAPES = {
    "output.weight": [64, VOCAB],
    "blk.0.attn_q_norm.weight": [16],
    "blk.1.ffn_gate_shexp.weight": [64, 24],
}


def test_minimax_m3_synth_instantiates():
    # DeepSeek-V3-shaped MoE on the M2 GQA base: leading dense layers, shared
    # expert, sigmoid + correction bias, routed scaling; SwiGLU-OAI activation.
    # The model class is the gmlx-vendored mlx-lm PR #1401 module,
    # registered into mlx_lm.models exactly as the loader does it.
    from gmlx import minimax_m3_model
    minimax_m3_model.ensure_registered()

    c = synthesize_config(_minimax_m3_meta(), tensor_shapes=_MINIMAX_M3_SHAPES)
    assert c["model_type"] == "minimax_m3"
    assert c["num_local_experts"] == 4 and c["num_experts_per_tok"] == 2
    assert c["intermediate_size"] == 48           # expert FFN width
    assert c["dense_intermediate_size"] == 128    # the arch feed_forward_length
    assert c["shared_intermediate_size"] == 24    # from the shexp tensor shape
    assert c["scoring_func"] == "sigmoid"
    assert c["routed_scaling_factor"] == 2.0
    assert c["mlp_layer_types"] == ["dense", "sparse", "sparse"]
    assert c["rotary_dim"] == 8
    assert c["use_qk_norm"] is True
    assert c["tie_word_embeddings"] is False
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB
    # The dispatch list must have built 1 dense + 2 MoE layers.
    assert hasattr(model.model.layers[0], "mlp")
    assert hasattr(model.model.layers[1], "block_sparse_moe")


def test_minimax_m3_head_dim_from_key_length_not_hidden_over_heads():
    m = _minimax_m3_meta()
    m["minimax-m3.attention.key_length"] = 32   # hidden//heads would be 16
    c = synthesize_config(m, tensor_shapes=_MINIMAX_M3_SHAPES)
    assert c["head_dim"] == 32


def test_minimax_m3_shared_expert_width_required_when_declared():
    # expert_shared_count > 0 with no shexp tensor to size it must fail loudly.
    with pytest.raises(ValueError, match="ffn_gate_shexp"):
        synthesize_config(_minimax_m3_meta(),
                          tensor_shapes={"output.weight": [64, VOCAB]})


def test_minimax_m3_all_sparse_without_leading_dense_kv():
    # leading_dense_block_count absent -> every layer is MoE.
    m = _minimax_m3_meta()
    del m["minimax-m3.leading_dense_block_count"]
    c = synthesize_config(m, tensor_shapes=_MINIMAX_M3_SHAPES)
    assert c["mlp_layer_types"] == ["sparse"] * 3


def _hunyuan_meta() -> dict:
    arch = "hunyuan-moe"
    m = _base_meta(arch)
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 48
    m[f"{arch}.expert_shared_feed_forward_length"] = 32
    return m


# Per-head qk-norm (head_dim=16) + shared expert present; tied output (Hunyuan
# ties - no output.weight, and mlx-lm always uses embed_tokens.as_linear).
_HUNYUAN_SHAPES = {
    "blk.0.attn_q_norm.weight": [16],
    "blk.0.ffn_gate_shexp.weight": [64, 32],
}


def test_hunyuan_synth_instantiates():
    c = synthesize_config(_hunyuan_meta(), tensor_shapes=_HUNYUAN_SHAPES)
    assert c["model_type"] == "hunyuan"
    assert c["num_experts"] == 4 and c["moe_topk"] == 2
    assert c["moe_intermediate_size"] == 48     # routed-expert FFN width
    assert c["intermediate_size"] == 32         # shared width (num_shared = 1)
    assert c["num_shared_expert"] == 1
    assert c["use_mixed_mlp_moe"] is True
    assert c["use_qk_norm"] is True
    assert c["attention_bias"] is False
    assert c["use_cla"] is False
    assert c["rope_scaling"]["alpha"] == 1.0
    assert {"factor", "type"} <= set(c["rope_scaling"])  # __post_init__ requires
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_hunyuan_rope_alpha_override_from_gguf():
    # A GGUF that carries an explicit NTK alpha overrides the 1.0 default.
    m = _hunyuan_meta()
    m["hunyuan-moe.rope.scaling.alpha"] = 1000.0
    c = synthesize_config(m, tensor_shapes=_HUNYUAN_SHAPES)
    assert c["rope_scaling"]["alpha"] == 1000.0


def test_hunyuan_shared_width_from_tensor_when_kv_absent():
    # Without expert_shared_feed_forward_length, the shared width (->
    # intermediate_size) derives from the ffn_gate_shexp tensor's middle dim.
    m = _hunyuan_meta()
    del m["hunyuan-moe.expert_shared_feed_forward_length"]
    c = synthesize_config(m, tensor_shapes=_HUNYUAN_SHAPES)
    assert c["intermediate_size"] == 32


def _hy_v3_meta() -> dict:
    arch = "hy_v3"
    m = _base_meta(arch)
    # 3 trunk layers + 1 NextN/MTP block appended past the trunk.
    m[f"{arch}.block_count"] = 4
    m[f"{arch}.nextn_predict_layers"] = 1
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 48
    m[f"{arch}.expert_shared_feed_forward_length"] = 48   # 1 shared expert
    m[f"{arch}.expert_weights_scale"] = 2.826
    m[f"{arch}.expert_weights_norm"] = True
    m[f"{arch}.expert_gating_func"] = 2                   # sigmoid
    return m


# Layer 0 dense (no router tensor), layers 1-2 MoE, block 3 = the MTP block
# (also MoE - it must not end the leading-dense derivation early). Untied
# output head; per-head qk-norm; suffix-less expert bias (Hy3 wire quirk).
_HY_V3_SHAPES = {
    "output.weight": [64, VOCAB],
    "blk.0.attn_q_norm.weight": [16],
    "blk.1.ffn_gate_inp.weight": [64, 4],
    "blk.1.exp_probs_b": [4],
    "blk.2.ffn_gate_inp.weight": [64, 4],
    "blk.2.exp_probs_b": [4],
    "blk.3.ffn_gate_inp.weight": [64, 4],
    "blk.3.exp_probs_b": [4],
}


def test_hy_v3_synth_instantiates():
    # Sigmoid-gated MoE + selection-only expert bias + ungated shared expert
    # behind one leading dense layer; NextN block excluded from the trunk.
    # The model class is the gmlx-vendored mlx-lm PR #1485 module,
    # registered into mlx_lm.models exactly as the loader does it.
    from gmlx import hy_v3_model
    hy_v3_model.ensure_registered()

    c = synthesize_config(_hy_v3_meta(), tensor_shapes=_HY_V3_SHAPES)
    assert c["model_type"] == "hy_v3"
    assert c["num_hidden_layers"] == 3            # block_count 4 - nextn 1
    assert c["num_nextn_predict_layers"] == 1
    assert c["num_experts"] == 4 and c["num_experts_per_tok"] == 2
    assert c["expert_hidden_dim"] == 48           # routed-expert FFN width
    assert c["intermediate_size"] == 128          # dense-layer width
    assert c["num_shared_experts"] == 1
    assert c["first_k_dense_replace"] == 1        # from tensor presence
    assert c["router_scaling_factor"] == 2.826
    assert c["route_norm"] is True
    assert c["moe_router_use_sigmoid"] is True
    assert c["moe_router_enable_expert_bias"] is True
    assert c["qk_norm"] is True
    assert c["head_dim"] == 16                    # from key_length, not h//heads
    assert c["tie_word_embeddings"] is False
    assert c["enable_lm_head_fp32"] is False      # pinned off (parity oracle)
    assert c["rope_parameters"] == {"rope_theta": 10000.0,
                                    "rope_type": "default"}
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB
    # Layer 0 dense MLP, layer 1 MoE (router + switch_mlp + shared_mlp).
    assert hasattr(model.model.layers[0].mlp, "gate_proj")
    assert hasattr(model.model.layers[1].mlp, "switch_mlp")
    assert hasattr(model.model.layers[1].mlp, "router")
    assert model.model.layers[1].mlp.shared_mlp is not None


def test_hy_v3_sanitize_strips_nextn_block():
    from gmlx import hy_v3_model
    hy_v3_model.ensure_registered()
    c = synthesize_config(_hy_v3_meta(), tensor_shapes=_HY_V3_SHAPES)
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    weights = {
        "model.layers.2.self_attn.q_proj.weight": mx.zeros((64, 64)),
        "model.layers.3.self_attn.q_proj.weight": mx.zeros((64, 64)),
        "model.layers.3.nextn.eh_proj.weight": mx.zeros((64, 128)),
    }
    out = model.sanitize(weights)
    assert "model.layers.2.self_attn.q_proj.weight" in out
    assert not any(k.startswith("model.layers.3.") for k in out)


def test_hy_v3_head_dim_required():
    m = _hy_v3_meta()
    del m["hy_v3.attention.key_length"]
    with pytest.raises(ValueError, match="key_length"):
        synthesize_config(m, tensor_shapes=_HY_V3_SHAPES)


def test_hy_v3_yarn_rope_scaling_translated():
    # Long-context (1M) conversions may carry yarn scaling in the KV.
    m = _hy_v3_meta()
    m["hy_v3.rope.scaling.type"] = "yarn"
    m["hy_v3.rope.scaling.factor"] = 4.0
    m["hy_v3.rope.scaling.original_context_length"] = 262144
    c = synthesize_config(m, tensor_shapes=_HY_V3_SHAPES)
    assert c["rope_parameters"] == {
        "rope_theta": 10000.0,
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 262144,
    }


def test_hy_v3_no_shared_expert_when_absent():
    m = _hy_v3_meta()
    del m["hy_v3.expert_shared_feed_forward_length"]
    c = synthesize_config(m, tensor_shapes=_HY_V3_SHAPES)
    assert c["num_shared_experts"] == 0


def _granitehybrid_meta() -> dict:
    arch = "granitehybrid"
    m = _base_meta(arch)
    # Per-layer head_count_kv: 0 => recurrent (Mamba2). Layer 0 mamba, layer 1
    # attention - exercises both block types in a 2-layer model.
    m[f"{arch}.attention.head_count_kv"] = [0, 2]
    m[f"{arch}.ssm.conv_kernel"] = 4
    m[f"{arch}.ssm.inner_size"] = 128       # 2*hidden
    m[f"{arch}.ssm.state_size"] = 16
    m[f"{arch}.ssm.time_step_rank"] = 4     # mamba_n_heads -> d_head = 32
    m[f"{arch}.ssm.group_count"] = 1
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_shared_feed_forward_length"] = 96
    m[f"{arch}.embedding_scale"] = 12.0
    m[f"{arch}.residual_scale"] = 0.22
    m[f"{arch}.attention.scale"] = 0.0078125
    m[f"{arch}.logit_scale"] = 8.0
    return m


def test_granitehybrid_synth_instantiates():
    c = synthesize_config(_granitehybrid_meta(), tensor_shapes={})
    assert c["model_type"] == "granitemoehybrid"
    assert c["layer_types"] == ["mamba", "attention"]
    # element 0 of the per-layer kv array is 0 (mamba) - must NOT become the
    # model-wide num_key_value_heads.
    assert c["num_key_value_heads"] == 2
    assert c["embedding_multiplier"] == 12.0
    assert c["residual_multiplier"] == 0.22
    assert c["attention_multiplier"] == 0.0078125
    assert c["logits_scaling"] == 8.0
    assert c["mamba_n_heads"] == 4 and c["mamba_d_head"] == 32
    assert c["mamba_d_state"] == 16 and c["mamba_d_conv"] == 4
    assert c["mamba_n_groups"] == 1 and c["mamba_conv_bias"] is False
    assert c["num_local_experts"] == 4 and c["num_experts_per_tok"] == 2
    assert c["shared_intermediate_size"] == 96
    assert c["position_embedding_type"] == "rope"
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_granitehybrid_nope_and_neutral_multiplier_defaults():
    # rope.scaling.finetuned=false => NoPE; absent multipliers default neutral
    # (attention_multiplier falls back to 1/sqrt(head_dim)).
    m = _granitehybrid_meta()
    for k in ("embedding_scale", "residual_scale", "attention.scale",
              "logit_scale"):
        del m[f"granitehybrid.{k}"]
    m["granitehybrid.rope.scaling.finetuned"] = False
    c = synthesize_config(m, tensor_shapes={})
    assert c["position_embedding_type"] == "nope"
    assert c["embedding_multiplier"] == 1.0
    assert c["residual_multiplier"] == 1.0
    assert c["logits_scaling"] == 1.0
    assert c["attention_multiplier"] == (64 // 4) ** -0.5
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_granitehybrid_dense_variant_without_experts():
    # No expert_count => dense-MLP hybrid: MoE fields stay unset and mlx-lm
    # builds GraniteMoeHybridMLP (use_moe False).
    m = _granitehybrid_meta()
    for k in ("expert_count", "expert_used_count",
              "expert_shared_feed_forward_length"):
        del m[f"granitehybrid.{k}"]
    c = synthesize_config(m, tensor_shapes={})
    assert "num_local_experts" not in c
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def _falcon_h1_meta() -> dict:
    arch = "falcon-h1"
    m = _base_meta(arch)
    m[f"{arch}.ssm.conv_kernel"] = 4
    m[f"{arch}.ssm.inner_size"] = 128       # 2*hidden
    m[f"{arch}.ssm.state_size"] = 16
    m[f"{arch}.ssm.time_step_rank"] = 4     # mamba_n_heads -> d_head = 32
    m[f"{arch}.ssm.group_count"] = 1
    return m


# conv_dim = inner + 2*groups*state = 128 + 32 = 160
_FALCON_H1_SHAPES = {
    "output.weight": [64, VOCAB],
    "blk.0.ssm_norm.weight": [128],
    "blk.0.ssm_conv1d.bias": [160],
}


def test_falcon_h1_synth_instantiates():
    c = synthesize_config(_falcon_h1_meta(), tensor_shapes=_FALCON_H1_SHAPES)
    assert c["model_type"] == "falcon_h1"
    assert c["head_dim"] == 16                       # key_length, not hidden//heads
    assert c["mamba_d_ssm"] == 128
    assert c["mamba_n_heads"] == 4 and c["mamba_d_head"] == 32
    assert c["mamba_d_state"] == 16 and c["mamba_d_conv"] == 4
    assert c["mamba_n_groups"] == 1
    assert c["mamba_rms_norm"] is True               # ssm_norm tensor present
    assert c["mamba_conv_bias"] is True              # conv bias tensor present
    assert c["mamba_norm_before_gate"] is False      # llama.cpp gates THEN norms
    assert c["tie_word_embeddings"] is False         # output.weight present
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    # Forward with a real cache: every falcon-h1 layer runs attention AND the
    # Mamba2 mixer in parallel, so this exercises both block types per layer.
    out = model(mx.array([[1, 2, 3]]), cache=model.make_cache())
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_falcon_h1_neutral_multipliers_and_tied_variant():
    # The muP multiplier zoo is folded into the wire weights at convert; the
    # synth must pin every ModelArgs multiplier neutral (the class defaults are
    # NON-neutral, and the tied-embed lm_head path applies
    # lm_head_multiplier/embedding_multiplier at runtime).
    c = synthesize_config(_falcon_h1_meta(), tensor_shapes={})
    for k in ("embedding_multiplier", "attention_in_multiplier",
              "attention_out_multiplier", "key_multiplier",
              "lm_head_multiplier", "ssm_in_multiplier", "ssm_out_multiplier"):
        assert c[k] == 1.0, f"{k}: {c[k]}"
    assert c["mlp_multipliers"] == [1.0, 1.0]
    assert c["ssm_multipliers"] == [1.0] * 5
    # No tensors => tied embeddings, no gated norm, no conv bias.
    assert c["tie_word_embeddings"] is True
    assert c["mamba_rms_norm"] is False
    assert c["mamba_conv_bias"] is False
    assert c["attention_bias"] is False and c["mlp_bias"] is False
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]), cache=model.make_cache())
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def _qwen3next_meta() -> dict:
    arch = "qwen3next"
    m = _base_meta(arch)
    # 4 layers so the interval-4 schedule exercises BOTH block types:
    # idx 0-2 gated-DeltaNet, idx 3 full attention.
    m[f"{arch}.block_count"] = 4
    m[f"{arch}.full_attention_interval"] = 4
    m[f"{arch}.rope.dimension_count"] = 4    # head_dim 16 * 0.25
    m[f"{arch}.ssm.conv_kernel"] = 4
    m[f"{arch}.ssm.inner_size"] = 64         # head_v 16 * num_v 4
    m[f"{arch}.ssm.state_size"] = 16         # linear_key_head_dim
    m[f"{arch}.ssm.time_step_rank"] = 4      # linear_num_value_heads
    m[f"{arch}.ssm.group_count"] = 2         # linear_num_key_heads
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 32
    m[f"{arch}.expert_shared_feed_forward_length"] = 48
    return m


def test_qwen3next_synth_instantiates():
    c = synthesize_config(_qwen3next_meta(), tensor_shapes={})
    assert c["model_type"] == "qwen3_next"
    assert c["head_dim"] == 16
    assert c["partial_rotary_factor"] == 0.25
    assert c["linear_num_value_heads"] == 4
    assert c["linear_num_key_heads"] == 2
    assert c["linear_key_head_dim"] == 16
    assert c["linear_value_head_dim"] == 16  # inner // num_v
    assert c["linear_conv_kernel_dim"] == 4
    assert c["kv_head_layout"] == "grouped"
    assert c["full_attention_interval"] == 4
    assert c["num_experts"] == 4 and c["num_experts_per_tok"] == 2
    assert c["moe_intermediate_size"] == 32
    assert c["shared_expert_intermediate_size"] == 48
    assert c["norm_topk_prob"] is True
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]), cache=model.make_cache())
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_qwen3next_split_layout_flagged():
    # The newer split GDN layout (attn_qkv/attn_gate) sets gdn_split_layout so
    # the loader restructures the GDN modules; the legacy fused layout leaves
    # the flag unset (stock in_proj_qkvz path).
    c = synthesize_config(_qwen3next_meta(),
                          tensor_shapes={"blk.0.attn_qkv.weight": [64, 96]})
    assert c["gdn_split_layout"] is True
    c = synthesize_config(_qwen3next_meta(),
                          tensor_shapes={"blk.0.ssm_in.weight": [128, 64]})
    assert "gdn_split_layout" not in c


def test_qwen3next_recurrent_layers_array_mismatch_rejected():
    # An explicit attention.recurrent_layers array that contradicts the
    # interval pattern is unrepresentable in mlx-lm's qwen3_next.
    m = _qwen3next_meta()
    m["qwen3next.attention.recurrent_layers"] = [True, True, True, True]
    with pytest.raises(ValueError, match="recurrent_layers"):
        synthesize_config(m, tensor_shapes={})
    # ...while a matching array is accepted.
    m["qwen3next.attention.recurrent_layers"] = [True, True, True, False]
    c = synthesize_config(m, tensor_shapes={})
    assert c["full_attention_interval"] == 4


def test_qwen3next_tiled_v_patch_excluded():
    # qwen3next has asymmetric linear K/V heads (the tiled-V predicate's
    # trigger) but its wire V order is HF-grouped - the qwen3.5 tiled fixup
    # must NOT engage for it.
    from gmlx.gdn_patches import _needs_tiled_v_patch
    c = synthesize_config(_qwen3next_meta(), tensor_shapes={})
    assert c["linear_num_key_heads"] != c["linear_num_value_heads"]
    assert _needs_tiled_v_patch(c) is False
    # The qwen3.5 shape of the same config still triggers it.
    assert _needs_tiled_v_patch({"model_type": "qwen3_5_moe",
                                 "linear_num_key_heads": 2,
                                 "linear_num_value_heads": 4}) is True


def test_qwen3next_split_gdn_patch_forward_parity():
    # The split-GDN loader patch must be numerically equivalent to the stock
    # fused path. Build a stock tiny model, split each in_proj_qkvz exactly
    # the way llama.cpp's converter does (per-k-head de-interleave: q|k|v
    # flattened head-major -> attn_qkv, z -> attn_gate), load the halves into a
    # patched copy, and compare logits - prefill and a cached decode step.
    from gmlx.gdn_patches import _patch_qwen3next_split_gdn
    c = synthesize_config(_qwen3next_meta(), tensor_shapes={})
    Model, ModelArgs = _get_classes(c)
    mx.random.seed(0)
    stock = Model(ModelArgs.from_dict(c))
    mx.eval(stock.parameters())

    split = Model(ModelArgs.from_dict(c))
    split.update(stock.parameters())
    _patch_qwen3next_split_gdn(split)

    nk = c["linear_num_key_heads"]
    dn = c["linear_key_head_dim"]
    vg = (c["linear_num_value_heads"] // nk) * c["linear_value_head_dim"]
    n_gdn = 0
    for s_layer, p_layer in zip(stock.layers, split.layers):
        if not hasattr(s_layer, "linear_attn"):
            continue
        W = s_layer.linear_attn.in_proj_qkvz.weight
        h = W.shape[-1]
        w = W.reshape(nk, -1, h)        # per-k-head interleaved [q|k|v|z]
        q, k = w[:, :dn], w[:, dn:2 * dn]
        v, z = w[:, 2 * dn:2 * dn + vg], w[:, 2 * dn + vg:]
        gdn = p_layer.linear_attn
        gdn.in_proj_qkv.weight = mx.concatenate(
            [q.reshape(-1, h), k.reshape(-1, h), v.reshape(-1, h)], axis=0)
        gdn.in_proj_z.weight = z.reshape(-1, h)
        # structural: fused module gone, split shapes correct
        assert not hasattr(gdn, "in_proj_qkvz")
        kd, vd = gdn.key_dim, gdn.value_dim
        assert gdn.in_proj_qkv.weight.shape == (2 * kd + vd, h)
        assert gdn.in_proj_z.weight.shape == (vd, h)
        n_gdn += 1
    assert n_gdn == 3                    # interval-4 schedule: layers 0-2 GDN
    mx.eval(split.parameters())
    assert not any("in_proj_qkvz" in name
                   for name, _ in tree_flatten(split.parameters()))

    toks = mx.array([[1, 2, 3, 4, 5]])
    cache_s, cache_p = stock.make_cache(), split.make_cache()
    out_s = stock(toks, cache=cache_s)
    out_p = split(toks, cache=cache_p)
    mx.eval(out_s, out_p)
    assert mx.allclose(out_s, out_p, atol=1e-5, rtol=1e-5)
    # one cached decode step (exercises conv_state + recurrent state reuse)
    step = mx.array([[7]])
    out_s = stock(step, cache=cache_s)
    out_p = split(step, cache=cache_p)
    mx.eval(out_s, out_p)
    assert mx.allclose(out_s, out_p, atol=1e-5, rtol=1e-5)


def test_mixtral_synth_instantiates():
    # Mixtral ships as general.architecture='llama' + an expert count; the llama
    # synth must switch model_type to mixtral and add the MoE fields, then the
    # rest of the (shared) llama backbone instantiates + forwards.
    m = _meta_for("llama")
    m["llama.expert_count"] = 4
    m["llama.expert_used_count"] = 2
    c = synthesize_config(m, tensor_shapes={"output.weight": [64, VOCAB]})
    assert c["model_type"] == "mixtral"
    assert c["num_local_experts"] == 4
    assert c["num_experts_per_tok"] == 2
    assert c["tie_word_embeddings"] is False     # output.weight present => untied
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_llama_without_experts_stays_dense():
    # No expert count => plain dense llama (regression guard for the discriminator).
    c = synthesize_config(_meta_for("llama"), tensor_shapes={})
    assert c["model_type"] == "llama"
    assert "num_local_experts" not in c


def _deepseek2_meta() -> dict:
    arch = "deepseek2"
    m = _base_meta(arch)
    m[f"{arch}.attention.head_count_kv"] = 1          # MLA: single absorbed kv
    m[f"{arch}.expert_gating_func"] = 2               # V3 sigmoid gating
    m[f"{arch}.attention.q_lora_rank"] = 32
    m[f"{arch}.attention.kv_lora_rank"] = 16
    m[f"{arch}.rope.dimension_count"] = 4             # qk_rope_head_dim
    m[f"{arch}.expert_count"] = 8
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 32
    m[f"{arch}.expert_shared_count"] = 1
    m[f"{arch}.leading_dense_block_count"] = 1        # layer 0 dense, layer 1 MoE
    m[f"{arch}.expert_group_count"] = 2
    m[f"{arch}.expert_group_used_count"] = 2
    m[f"{arch}.expert_weights_scale"] = 2.5
    m[f"{arch}.expert_weights_norm"] = True
    m[f"{arch}.vocab_size"] = VOCAB
    return m


# attn_q_b: [q_lora, num_heads * q_head_dim] (q_head_dim = nope+rope = 4+4);
# attn_v_b: [kv_lora, v_head_dim, num_heads] (per-head stacked up-projection).
_DEEPSEEK2_SHAPES = {
    "blk.0.attn_q_b.weight": [32, 32],
    "blk.0.attn_v_b.weight": [16, 8, 4],
}


def test_deepseek2_synth_instantiates():
    # deepseek_v3: MLA head-dim decomposition derived from the per-head up-proj
    # tensor shapes, plus the V3 fine-grained MoE block (group routing, sigmoid
    # gating, leading dense block). A tiny forward proves they're consistent.
    c = synthesize_config(_deepseek2_meta(), tensor_shapes=_DEEPSEEK2_SHAPES)
    assert c["model_type"] == "deepseek_v3"
    assert "head_dim" not in c                        # MLA has no single head_dim
    assert c["q_lora_rank"] == 32 and c["kv_lora_rank"] == 16
    assert c["qk_rope_head_dim"] == 4
    assert c["qk_nope_head_dim"] == 4                 # q_head_dim 8 - rope 4
    assert c["v_head_dim"] == 8                        # attn_v_b[1]
    assert c["n_routed_experts"] == 8 and c["num_experts_per_tok"] == 2
    assert c["moe_intermediate_size"] == 32
    assert c["first_k_dense_replace"] == 1
    assert c["n_group"] == 2 and c["topk_group"] == 2
    assert c["routed_scaling_factor"] == 2.5
    assert c["scoring_func"] == "sigmoid" and c["topk_method"] == "noaux_tc"
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_deepseek2_v2_softmax_gating_rejected():
    # A softmax-gated V2 GGUF (no correction bias) isn't the deepseek_v3 mlx-lm
    # model; the synth must refuse rather than emit a half-built V3 config.
    m = _deepseek2_meta()
    m["deepseek2.expert_gating_func"] = 1             # softmax (V2)
    with pytest.raises(NotImplementedError):
        synthesize_config(m, tensor_shapes=_DEEPSEEK2_SHAPES)


def _deepseek4_meta() -> dict:
    arch = "deepseek4"
    m = _base_meta(arch)
    m[f"{arch}.block_count"] = 3
    m[f"{arch}.attention.head_count_kv"] = 1      # single shared KV latent
    m[f"{arch}.expert_gating_func"] = 4           # sqrt-softplus (V4 Flash)
    m[f"{arch}.expert_count"] = 8
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 32
    m[f"{arch}.expert_shared_count"] = 1
    m[f"{arch}.expert_weights_scale"] = 1.5
    m[f"{arch}.expert_weights_norm"] = True
    m[f"{arch}.hash_layer_count"] = 1
    m[f"{arch}.attention.q_lora_rank"] = 32
    m[f"{arch}.attention.output_lora_rank"] = 16
    m[f"{arch}.attention.output_group_count"] = 2
    m[f"{arch}.rope.dimension_count"] = 8         # tail rope on 8 of 16 dims
    m[f"{arch}.attention.sliding_window"] = 16
    m[f"{arch}.attention.indexer.head_count"] = 2
    m[f"{arch}.attention.indexer.key_length"] = 16
    m[f"{arch}.attention.indexer.top_k"] = 8
    # block_count + nextn entries; the tail one belongs to the MTP layer that
    # ships in a SEPARATE GGUF.
    m[f"{arch}.attention.compress_ratios"] = [0, 128, 4, 0]
    m[f"{arch}.attention.compress_rope_freq_base"] = 160000.0
    m[f"{arch}.rope.scaling.type"] = "yarn"
    m[f"{arch}.rope.scaling.factor"] = 16.0
    m[f"{arch}.rope.scaling.original_context_length"] = 512
    m[f"{arch}.rope.scaling.yarn_beta_fast"] = 32.0
    m[f"{arch}.rope.scaling.yarn_beta_slow"] = 1.0
    m[f"{arch}.hyper_connection.count"] = 4
    m[f"{arch}.hyper_connection.sinkhorn_iterations"] = 5
    m[f"{arch}.hyper_connection.epsilon"] = 1e-5
    m[f"{arch}.swiglu_clamp_exp"] = [10.0, 10.0, 10.0, 10.0]
    m[f"{arch}.nextn_predict_layers"] = 1
    m[f"{arch}.vocab_size"] = VOCAB
    return m


_DEEPSEEK4_SHAPES = {"output.weight": [64, VOCAB]}


def test_deepseek4_synth_instantiates():
    # DeepSeek V4 Flash: MLA-lite low-rank attention in three per-layer
    # variants (local / compressed / sparse-indexed via compress_ratios),
    # hyper-connections, sqrt-softplus MoE with hash routing. A tiny forward
    # over one layer of each attention variant proves config/model agree.
    # Model class = gmlx-vendored mlx-lm PR #1192, registered into
    # mlx_lm.models exactly as the loader does it.
    from gmlx import deepseek_v4_model
    deepseek_v4_model.ensure_registered()

    c = synthesize_config(_deepseek4_meta(), tensor_shapes=_DEEPSEEK4_SHAPES)
    assert c["model_type"] == "deepseek_v4"
    assert c["scoring_func"] == "sqrtsoftplus"
    assert c["n_routed_experts"] == 8 and c["num_experts_per_tok"] == 2
    assert c["moe_intermediate_size"] == 32 and c["n_shared_experts"] == 1
    assert c["routed_scaling_factor"] == 1.5 and c["norm_topk_prob"] is True
    assert c["num_hash_layers"] == 1
    assert c["q_lora_rank"] == 32
    assert c["o_lora_rank"] == 16 and c["o_groups"] == 2
    assert c["qk_rope_head_dim"] == 8 and c["sliding_window"] == 16
    assert c["index_n_heads"] == 2 and c["index_head_dim"] == 16
    assert c["index_topk"] == 8
    assert c["compress_rope_theta"] == 160000.0
    assert c["rope_scaling"]["type"] == "yarn"
    assert c["rope_scaling"]["factor"] == 16.0
    assert c["rope_scaling"]["original_max_position_embeddings"] == 512
    assert c["hc_mult"] == 4 and c["hc_sinkhorn_iters"] == 5
    assert c["swiglu_limit"] == 10.0
    assert c["tie_word_embeddings"] is False
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    cache = model.make_cache()
    out = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_deepseek4_layer_count_not_nextn_subtracted():
    # nextn_predict_layers=1 is in the metadata but the MTP layer ships in a
    # separate GGUF - block_count already excludes it. The universal
    # `num_hidden_layers = block_count - nextn` cut (right for glm4moe/glm-dsa,
    # whose MTP layer is IN the file) must be undone, and compress_ratios
    # (block_count + nextn entries; tail = the MTP layer's) truncated.
    c = synthesize_config(_deepseek4_meta(), tensor_shapes=_DEEPSEEK4_SHAPES)
    assert c["num_hidden_layers"] == 3            # not 2
    assert c["compress_ratios"] == [0, 128, 4]    # MTP tail entry dropped
    assert c["num_nextn_predict_layers"] == 1


def test_deepseek4_non_sqrtsoftplus_gating_rejected():
    # The vendored model implements gating func 4 (sqrt-softplus) only; a
    # sigmoid-gated GGUF must refuse loudly, not mis-gate silently.
    m = _deepseek4_meta()
    m["deepseek4.expert_gating_func"] = 2
    with pytest.raises(ValueError, match="expert_gating_func"):
        synthesize_config(m, tensor_shapes=_DEEPSEEK4_SHAPES)


def test_deepseek4_nonuniform_swiglu_clamp_rejected():
    # swiglu_clamp_exp is a per-layer GGUF array but the model class takes one
    # scalar; a non-uniform array would silently mis-clamp -> hard error.
    m = _deepseek4_meta()
    m["deepseek4.swiglu_clamp_exp"] = [10.0, 8.0, 10.0, 10.0]
    with pytest.raises(ValueError, match="swiglu_clamp_exp"):
        synthesize_config(m, tensor_shapes=_DEEPSEEK4_SHAPES)


def test_deepseek4_short_compress_ratios_rejected():
    # Fewer ratio entries than layers means the GGUF is malformed (every layer
    # needs an attention variant); building anyway would mis-shape the stack.
    m = _deepseek4_meta()
    m["deepseek4.attention.compress_ratios"] = [0, 128]
    with pytest.raises(ValueError, match="compress_ratios"):
        synthesize_config(m, tensor_shapes=_DEEPSEEK4_SHAPES)


def _glm4moe_meta() -> dict:
    arch = "glm4moe"
    m = _base_meta(arch)
    m[f"{arch}.block_count"] = 3                  # 2 real + 1 MTP
    m[f"{arch}.nextn_predict_layers"] = 1         # -> num_hidden_layers = 2
    m[f"{arch}.rope.dimension_count"] = 8         # partial rotary 8/16 = 0.5
    m[f"{arch}.expert_count"] = 8
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 32
    m[f"{arch}.expert_shared_count"] = 1
    m[f"{arch}.leading_dense_block_count"] = 1    # layer 0 dense, layer 1 MoE
    m[f"{arch}.expert_group_count"] = 2
    m[f"{arch}.expert_group_used_count"] = 2
    m[f"{arch}.expert_weights_scale"] = 2.5
    m[f"{arch}.expert_weights_norm"] = True
    return m


def test_glm4moe_synth_instantiates():
    # GLM-4.5/4.6: standard MHA (partial rotary, qk-norm from tensor inventory)
    # + V3-style fine-grained MoE; the MTP layer is excluded from num_hidden_layers.
    c = synthesize_config(
        _glm4moe_meta(),
        tensor_shapes={"output.weight": [64, VOCAB],
                       "blk.0.attn_q_norm.weight": [16]})
    assert c["model_type"] == "glm4_moe"
    assert c["num_hidden_layers"] == 2            # 3 blocks - 1 nextn
    assert c["partial_rotary_factor"] == 0.5      # rope_dim 8 / head_dim 16
    assert c["use_qk_norm"] is True               # attn_q_norm tensor present
    assert c["n_routed_experts"] == 8 and c["num_experts_per_tok"] == 2
    assert c["moe_intermediate_size"] == 32 and c["first_k_dense_replace"] == 1
    assert c["n_group"] == 2 and c["topk_group"] == 2
    assert c["scoring_func"] == "sigmoid" and c["topk_method"] == "noaux_tc"
    assert c["rope_scaling"] is None              # no yarn => explicit None (required field)
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_glm4moe_qk_norm_absent_when_no_tensor():
    # use_qk_norm is inventory-driven: no attn_q_norm tensor => False.
    c = synthesize_config(_glm4moe_meta(),
                          tensor_shapes={"output.weight": [64, VOCAB]})
    assert c["use_qk_norm"] is False


def _gpt_oss_meta() -> dict:
    arch = "gpt-oss"
    m = _base_meta(arch)
    # gpt-oss has no dense MLP; the SwitchGLU hidden dim is the *expert* width.
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = 32
    m[f"{arch}.attention.sliding_window"] = 8
    # YaRN rope (as shipped): type + factor + original context length.
    m[f"{arch}.rope.scaling.type"] = "yarn"
    m[f"{arch}.rope.scaling.factor"] = 32.0
    m[f"{arch}.rope.scaling.original_context_length"] = 4096
    return m


def test_gpt_oss_synth_instantiates():
    # gpt-oss: MoE (no dense MLP) with per-head sinks, alternating sliding/full
    # attention, and YaRN rope. The MXFP4 experts are a load-time concern; here
    # the unquantized model must build + forward from the synthesized config.
    c = synthesize_config(_gpt_oss_meta(),
                          tensor_shapes={"output.weight": [64, VOCAB]})
    assert c["model_type"] == "gpt_oss"
    assert c["num_local_experts"] == 4 and c["num_experts_per_tok"] == 2
    # intermediate_size is the EXPERT width (expert_feed_forward_length), not the
    # universal feed_forward_length - gpt-oss has no dense MLP.
    assert c["intermediate_size"] == 32
    assert c["sliding_window"] == 8
    assert c["head_dim"] == 16                    # key_length
    assert c["rope_theta"] == 10000.0
    assert c["rope_scaling"]["type"] == "yarn"
    assert c["rope_scaling"]["factor"] == 32.0
    assert c["rope_scaling"]["original_max_position_embeddings"] == 4096
    # Per-layer pattern: sliding (even) / full (odd), starting sliding.
    assert c["layer_types"] == ["sliding_attention", "full_attention"]
    # vocab from the lm_head rows (untied: output.weight present).
    assert c["vocab_size"] == VOCAB and c["tie_word_embeddings"] is False
    Model, ModelArgs = _get_classes(c)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB


def test_unsupported_but_mapped_arch_raises_not_implemented(monkeypatch):
    # An arch can be mapped to an mlx-lm model_type before its synthesizer
    # lands; that interim state must hard-fail with NotImplementedError, not
    # silently emit a half-built config. Every shipped arch currently has a
    # synthesizer, so inject a fake mapping to exercise the branch.
    import gmlx.config_synth as cs
    monkeypatch.setitem(cs.GGUF_ARCH_TO_MODEL_TYPE, "_fake_pending", "llama")
    with pytest.raises(NotImplementedError):
        synthesize_config({"general.architecture": "_fake_pending"},
                          tensor_shapes={})


def test_unmapped_arch_raises_value_error():
    with pytest.raises(ValueError):
        synthesize_config({"general.architecture": "not-a-real-arch"},
                          tensor_shapes={})


# dual-mode: GGUFReader path == decoded-dict path
def test_reader_and_dict_paths_agree(tmp_path):
    """The KV-access helpers must yield an identical config whether ``meta`` is
    a gguf-py GGUFReader or the decoded dict from ``kq.load_gguf``."""
    from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType as GT
    import numpy as np

    arch = "qwen2"
    meta = _meta_for(arch)

    # Write the same scalar values into a real GGUF (+ a token_embd tensor so
    # the file is well-formed and the tie probe has an inventory to look at).
    p = tmp_path / "dual.gguf"
    w = GGUFWriter(str(p), arch)  # sets general.architecture
    w.add_uint32(f"{arch}.embedding_length", 64)
    w.add_uint32(f"{arch}.block_count", 2)
    w.add_uint32(f"{arch}.attention.head_count", 4)
    w.add_uint32(f"{arch}.attention.head_count_kv", 2)
    w.add_uint32(f"{arch}.feed_forward_length", 128)
    w.add_uint32(f"{arch}.context_length", 1024)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32(f"{arch}.attention.key_length", 16)
    w.add_float32(f"{arch}.rope.freq_base", 10000.0)
    w.add_array("tokenizer.ggml.tokens", ["t"] * VOCAB)
    w.add_tensor("token_embd.weight",
                 np.zeros((VOCAB, 64), dtype=np.float32), raw_dtype=GT.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    reader = GGUFReader(str(p), "r")
    from_reader = synthesize_config(reader)  # shapes derived from reader.tensors
    from_dict = synthesize_config(meta, tensor_shapes={"token_embd.weight": [64, VOCAB]})

    assert set(from_reader) == set(from_dict)
    for k in from_reader:
        a, b = from_reader[k], from_dict[k]
        if isinstance(a, float) or isinstance(b, float):
            # GGUF stores eps as float32; the decoded dict carries a Python
            # float - compare numerically, not bit-for-bit.
            assert a == pytest.approx(b, rel=1e-6), k
        else:
            assert a == b, k


def _write_meta_gguf(path, arch: str, meta: dict) -> None:
    """Write a decoded-dict fixture into a real GGUF, typed KV by KV, so the
    GGUFReader path sees genuine BOOL / array-typed fields."""
    from gguf import GGUFWriter
    import numpy as np

    w = GGUFWriter(str(path), arch)  # sets general.architecture
    for k, v in meta.items():
        if k == "general.architecture":
            continue
        if isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        elif isinstance(v, float):
            w.add_float32(k, v)
        elif isinstance(v, str):
            w.add_string(k, v)
        else:
            w.add_array(k, v)
    w.add_tensor("token_embd.weight",
                 np.zeros((VOCAB, 64), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _assert_configs_equal(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], float) or isinstance(b[k], float):
            # GGUF stores floats as float32; the decoded dict carries Python
            # floats - compare numerically, not bit-for-bit.
            assert a[k] == pytest.approx(b[k], rel=1e-6), k
        else:
            assert a[k] == b[k], k


def test_reader_and_dict_paths_agree_bool_and_array_kvs(tmp_path):
    """Same dual-mode contract for the typed KVs the qwen2 case never touches:
    a real GGUF BOOL (expert_weights_norm), an int array (per-layer
    head_count_kv), and a bool array (recurrent_layers)."""
    from gguf import GGUFReader

    # bool KV: qwen3moe expert_weights_norm=False (False so the read value is
    # distinguishable from the absent-KV default of True).
    m = _base_meta("qwen3moe")
    m["qwen3moe.expert_count"] = 4
    m["qwen3moe.expert_used_count"] = 2
    m["qwen3moe.expert_feed_forward_length"] = 64
    m["qwen3moe.expert_weights_norm"] = False

    # int-array KV: granitehybrid per-layer head_count_kv (0 => mamba layer).
    gh = _granitehybrid_meta()
    assert gh["granitehybrid.attention.head_count_kv"] == [0, 2]

    # bool-array KV: qwen3next explicit recurrent_layers schedule.
    qn = _qwen3next_meta()
    qn["qwen3next.attention.recurrent_layers"] = [True, True, True, False]

    for arch, meta in [("qwen3moe", m), ("granitehybrid", gh),
                       ("qwen3next", qn)]:
        p = tmp_path / f"{arch}.gguf"
        _write_meta_gguf(p, arch, meta)
        from_reader = synthesize_config(GGUFReader(str(p), "r"))
        from_dict = synthesize_config(
            meta, tensor_shapes={"token_embd.weight": [64, VOCAB]})
        _assert_configs_equal(from_reader, from_dict)

    # The typed KVs were actually read (not defaulted) on the reader path.
    m_cfg = synthesize_config(
        GGUFReader(str(tmp_path / "qwen3moe.gguf"), "r"))
    assert m_cfg["norm_topk_prob"] is False
    gh_cfg = synthesize_config(
        GGUFReader(str(tmp_path / "granitehybrid.gguf"), "r"))
    assert gh_cfg["layer_types"] == ["mamba", "attention"]
    assert gh_cfg["num_key_value_heads"] == 2
    qn_cfg = synthesize_config(
        GGUFReader(str(tmp_path / "qwen3next.gguf"), "r"))
    assert qn_cfg["full_attention_interval"] == 4


def test_hunyuan_norm_topk_patch():
    # mlx-lm's hunyuan.MoeBlock omits the norm_topk_prob rescale that the HF
    # reference and llama.cpp's hunyuan-moe graph both apply; the loader patch
    # adds it. On real A13B the missing rescale degenerates generation from
    # the first token, so pin the patch's exact semantics: stock output with
    # scores renormalized to sum to 1 over the selected top-k.
    from gmlx.loader import _patch_hunyuan_norm_topk
    from mlx_lm.models.hunyuan import MoeBlock

    c = synthesize_config(_hunyuan_meta(), tensor_shapes=_HUNYUAN_SHAPES)
    Model, ModelArgs = _get_classes(c)
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(c))
    mx.eval(model.parameters())

    blocks = [m for m in model.modules() if type(m) is MoeBlock]
    assert blocks, "fixture has no MoeBlock"
    blk = blocks[0]
    x = mx.random.normal((1, 3, c["hidden_size"]))

    # manual reference: stock routing math + the renormalization
    gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
    k = blk.top_k
    inds = mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    ref = (blk.switch_mlp(x, inds)
           * scores[..., None].astype(mx.float32)).sum(axis=-2)
    if blk.use_shared_mlp:
        ref = ref + blk.shared_mlp(x)

    _patch_hunyuan_norm_topk(model)
    assert type(blk).__name__ == "_NormTopKMoE"
    assert all(type(m).__name__ == "_NormTopKMoE"
               for m in model.modules() if isinstance(m, MoeBlock))
    out = blk(x)
    mx.eval(ref, out)
    assert mx.allclose(ref.astype(mx.float32), out.astype(mx.float32),
                       atol=1e-5, rtol=1e-5)


def test_zero_count_guard_is_a_named_refusal():
    # A corrupt GGUF declaring head_count=0 reaches a division somewhere in
    # per-arch synthesis; the entry points must surface a ValueError (clean
    # CLI error), not a ZeroDivisionError traceback.
    from gmlx.config_synth import _zero_count_guard

    @_zero_count_guard
    def synth():
        return 1 // 0

    with pytest.raises(ValueError, match="zero head/expert"):
        synth()


def test_mistral3_without_yarn_factor_is_plain_rope():
    """A mistral3 GGUF with no rope.scaling.factor KV must not emit
    type=yarn (mlx-lm's yarn init KeyErrors on the missing factor)."""
    m = _base_meta("mistral3")
    c = synthesize_config(m, tensor_shapes={})
    rp = c["rope_parameters"]
    assert rp["type"] == "default" and rp["rope_type"] == "default"
    assert "factor" not in rp
    assert rp["rope_theta"] == 10000.0
    assert rp["llama_4_scaling_beta"] == 0.0
    assert rp["original_max_position_embeddings"] == 1024


def test_mistral3_with_yarn_factor_keeps_yarn():
    m = _base_meta("mistral3")
    m["mistral3.rope.scaling.factor"] = 4.0
    m["mistral3.rope.scaling.original_context_length"] = 256
    c = synthesize_config(m, tensor_shapes={})
    rp = c["rope_parameters"]
    assert rp["type"] == "yarn" and rp["factor"] == 4.0
    assert rp["original_max_position_embeddings"] == 256


def test_qwen3_rope_scaling_none_omitted():
    m = _base_meta("qwen3")
    m["qwen3.rope.scaling.type"] = "none"
    c = synthesize_config(m, tensor_shapes={})
    assert "rope_scaling" not in c
