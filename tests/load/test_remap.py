#!/usr/bin/env python3
"""Per-architecture GGUF->HF tensor-name remap decisions.

Pure string/decision logic (``parse_gguf_name``) - no tensors, no GPU. These
pin the arch-specific routing that's easy to break on a gguf-py upgrade or a
new-arch addition: the FFN_NORM/FFN_PRE_NORM collision resolution, the gemma
norm-unbake gating, qwen2's QKV biases, phi3's fused projections, the gemma-4
MoE split + architectural ``.scale`` claims, and the universal skip/fail edges.
"""

from __future__ import annotations


from gmlx.load.remap import parse_gguf_name, RemapDecision  # noqa: E402

MAP = RemapDecision.KIND_MAP
SKIP = RemapDecision.KIND_SKIP
FAIL = RemapDecision.KIND_FAIL


def d(arch, name):
    return parse_gguf_name(arch, name)


# llama: qk_permute on Q/K only, ffn_norm collision pinned
def test_llama_qk_permute_q_and_k():
    for t in ("attn_q", "attn_k"):
        r = d("llama", f"blk.3.{t}.weight")
        assert r.kind == MAP and r.bid == 3
        assert r.transform == "qk_permute"
    # V is never permuted.
    rv = d("llama", "blk.3.attn_v.weight")
    assert rv.kind == MAP and rv.transform == "passthrough"
    assert rv.hf_name == "model.layers.3.self_attn.v_proj.weight"


def test_llama_ffn_norm_is_post_attention():
    r = d("llama", "blk.0.ffn_norm.weight")
    assert r.kind == MAP and r.transform == "passthrough"
    assert r.hf_name == "model.layers.0.post_attention_layernorm.weight"


def test_llama_global_tensors():
    assert d("llama", "token_embd.weight").hf_name == "model.embed_tokens.weight"
    assert d("llama", "output_norm.weight").hf_name == "model.norm.weight"
    assert d("llama", "output.weight").hf_name == "lm_head.weight"


# seed_oss: NEOX rope (no qk_permute), ffn_norm collision pinned
def test_seed_oss_attn_is_passthrough_not_permuted():
    # NEOX rope => Q/K must NOT be permuted (the qk_permute gate is LLAMA-only).
    for t in ("attn_q", "attn_k", "attn_v"):
        r = d("seed_oss", f"blk.5.{t}.weight")
        assert r.kind == MAP and r.bid == 5
        assert r.transform == "passthrough"
    assert d("seed_oss", "blk.5.attn_q.weight").hf_name == \
        "model.layers.5.self_attn.q_proj.weight"
    assert d("seed_oss", "blk.5.attn_output.weight").hf_name == \
        "model.layers.5.self_attn.o_proj.weight"


def test_seed_oss_ffn_norm_is_post_attention():
    r = d("seed_oss", "blk.0.ffn_norm.weight")
    assert r.kind == MAP and r.transform == "passthrough"
    assert r.hf_name == "model.layers.0.post_attention_layernorm.weight"


def test_seed_oss_attn_biases_claimed_as_bias():
    # Seed-OSS-36B ships F32 q/k/v biases. The canonical path strips ".bias"
    # and re-emits the ".weight" target, which would overwrite the quant
    # weight slot (the real-GGUF failure mode: KQuantLinear w not uint8) -
    # the override must claim them with their ".bias" HF names.
    for t, proj in (("attn_q", "q_proj"), ("attn_k", "k_proj"),
                    ("attn_v", "v_proj")):
        r = d("seed_oss", f"blk.3.{t}.bias")
        assert r.kind == MAP and r.transform == "passthrough"
        assert r.hf_name == f"model.layers.3.self_attn.{proj}.bias"


def test_seed_oss_ffn_and_global_tensors():
    assert d("seed_oss", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    assert d("seed_oss", "blk.0.ffn_down.weight").hf_name == \
        "model.layers.0.mlp.down_proj.weight"
    assert d("seed_oss", "token_embd.weight").hf_name == "model.embed_tokens.weight"
    assert d("seed_oss", "output_norm.weight").hf_name == "model.norm.weight"


# smollm3: reuses the LLAMA alias (NORM rope => qk_permute, ffn_norm pin)
def test_smollm3_reuses_llama_qk_permute():
    for t in ("attn_q", "attn_k"):
        r = d("smollm3", f"blk.2.{t}.weight")
        assert r.kind == MAP and r.transform == "qk_permute"
    assert d("smollm3", "blk.2.attn_v.weight").transform == "passthrough"
    assert d("smollm3", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"
    assert d("smollm3", "token_embd.weight").hf_name == "model.embed_tokens.weight"


# granite: reuses the LLAMA alias (NORM rope => qk_permute, ffn_norm pin)
def test_granite_reuses_llama_qk_permute():
    for t in ("attn_q", "attn_k"):
        assert d("granite", f"blk.1.{t}.weight").transform == "qk_permute"
    assert d("granite", "blk.1.attn_v.weight").transform == "passthrough"
    assert d("granite", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"
    assert d("granite", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"


# ernie4_5-moe: NORM rope but traditional=True => NO qk_permute
def test_ernie4_5_moe_attn_is_passthrough_despite_norm_rope():
    # CRITICAL: llama.cpp tags ernie4_5-moe LLAMA_ROPE_TYPE_NORM, but mlx-lm's
    # ernie4_5_moe uses traditional=True rope, which consumes the GGUF's HF-native
    # (un-permuted) Q/K directly. So Q/K must PASS THROUGH - applying the llama
    # qk_permute here would silently mis-attend. (Verified end-to-end by 16k
    # token-parity vs llama.cpp.)
    for t in ("attn_q", "attn_k", "attn_v"):
        r = d("ernie4_5-moe", f"blk.5.{t}.weight")
        assert r.kind == MAP and r.bid == 5 and r.transform == "passthrough"
    assert d("ernie4_5-moe", "blk.5.attn_q.weight").hf_name == \
        "model.layers.5.self_attn.q_proj.weight"
    assert d("ernie4_5-moe", "blk.5.attn_output.weight").hf_name == \
        "model.layers.5.self_attn.o_proj.weight"


def test_ernie4_5_moe_experts_shared_router_and_ffn_norm():
    # Routed experts (already stacked) -> switch_mlp; shared expert kept separate;
    # router ffn_gate_inp -> mlp.gate; ffn_norm pins past the collision.
    assert d("ernie4_5-moe", "blk.3.ffn_gate_exps.weight").hf_name == \
        "model.layers.3.mlp.switch_mlp.gate_proj.weight"
    assert d("ernie4_5-moe", "blk.3.ffn_down_exps.weight").hf_name == \
        "model.layers.3.mlp.switch_mlp.down_proj.weight"
    assert d("ernie4_5-moe", "blk.3.ffn_up_shexp.weight").hf_name == \
        "model.layers.3.mlp.shared_experts.up_proj.weight"
    assert d("ernie4_5-moe", "blk.3.ffn_gate_inp.weight").hf_name == \
        "model.layers.3.mlp.gate.weight"
    assert d("ernie4_5-moe", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"


def test_ernie4_5_moe_correction_bias_is_dropped():
    # mlx-lm's ernie4_5_moe.sanitize DROPS e_score_correction_bias and gates
    # without it, so exp_probs_b must SKIP - routing it to a live param would
    # leave an unfilled slot and fail strict-load.
    assert d("ernie4_5-moe", "blk.3.exp_probs_b.bias").kind == SKIP


def test_ernie4_5_moe_leading_dense_and_globals():
    # Leading dense layers use a plain MLP (no _exps suffix).
    assert d("ernie4_5-moe", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    assert d("ernie4_5-moe", "blk.0.ffn_down.weight").hf_name == \
        "model.layers.0.mlp.down_proj.weight"
    assert d("ernie4_5-moe", "token_embd.weight").hf_name == \
        "model.embed_tokens.weight"
    assert d("ernie4_5-moe", "output_norm.weight").hf_name == "model.norm.weight"


# minimax-m2: NEOX (no permute), MoE nested under block_sparse_moe.*
def test_minimax_attn_passthrough_and_full_width_qk_norm():
    # NEOX rope => Q/K NOT permuted; the full-width qk-norms resolve via the
    # canonical ATTN_Q_NORM/ATTN_K_NORM targets (passthrough).
    for t in ("attn_q", "attn_k", "attn_v"):
        assert d("minimax-m2", f"blk.4.{t}.weight").transform == "passthrough"
    assert d("minimax-m2", "blk.4.attn_q_norm.weight").hf_name == \
        "model.layers.4.self_attn.q_norm.weight"
    assert d("minimax-m2", "blk.4.attn_k_norm.weight").hf_name == \
        "model.layers.4.self_attn.k_norm.weight"
    assert d("minimax-m2", "blk.4.attn_output.weight").hf_name == \
        "model.layers.4.self_attn.o_proj.weight"


def test_minimax_moe_nested_under_block_sparse_moe():
    # Router, experts, and correction bias all nest under block_sparse_moe.* -
    # the canonical mlp.* targets would be wrong.
    assert d("minimax-m2", "blk.0.ffn_gate_inp.weight").hf_name == \
        "model.layers.0.block_sparse_moe.gate.weight"
    assert d("minimax-m2", "blk.0.ffn_gate_exps.weight").hf_name == \
        "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"
    assert d("minimax-m2", "blk.0.ffn_down_exps.weight").hf_name == \
        "model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"
    assert d("minimax-m2", "blk.0.ffn_up_exps.weight").hf_name == \
        "model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"
    # Correction bias KEPT (minimax uses it at runtime, unlike ERNIE).
    rb = d("minimax-m2", "blk.0.exp_probs_b.bias")
    assert rb.kind == MAP
    assert rb.hf_name == \
        "model.layers.0.block_sparse_moe.e_score_correction_bias"


def test_minimax_ffn_norm_and_globals():
    assert d("minimax-m2", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"
    assert d("minimax-m2", "token_embd.weight").hf_name == \
        "model.embed_tokens.weight"
    assert d("minimax-m2", "output_norm.weight").hf_name == "model.norm.weight"
    assert d("minimax-m2", "output.weight").hf_name == "lm_head.weight"


# minimax-m3: shares the MINIMAX alias; adds a shared expert, leading dense
# layers (canonical mlp.*), and gemma-+1-baked norms (unbaked on load).
def test_minimax_m3_shared_expert_under_block_sparse_moe():
    assert d("minimax-m3", "blk.3.ffn_gate_shexp.weight").hf_name == \
        "model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight"
    assert d("minimax-m3", "blk.3.ffn_up_shexp.weight").hf_name == \
        "model.layers.3.block_sparse_moe.shared_experts.up_proj.weight"
    assert d("minimax-m3", "blk.3.ffn_down_shexp.weight").hf_name == \
        "model.layers.3.block_sparse_moe.shared_experts.down_proj.weight"
    # Routed experts + router + correction bias: inherited MINIMAX routing.
    assert d("minimax-m3", "blk.3.ffn_gate_exps.weight").hf_name == \
        "model.layers.3.block_sparse_moe.switch_mlp.gate_proj.weight"
    rb = d("minimax-m3", "blk.3.exp_probs_b.bias")
    assert rb.kind == MAP and rb.hf_name == \
        "model.layers.3.block_sparse_moe.e_score_correction_bias"


def test_minimax_m3_leading_dense_layers_canonical_mlp():
    # blk.0-2 are dense on M3 - plain ffn_gate/up/down resolve canonically.
    assert d("minimax-m3", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    assert d("minimax-m3", "blk.0.ffn_up.weight").hf_name == \
        "model.layers.0.mlp.up_proj.weight"
    assert d("minimax-m3", "blk.0.ffn_down.weight").hf_name == \
        "model.layers.0.mlp.down_proj.weight"


def test_minimax_m3_norms_unbaked_m2_untouched():
    # llama.cpp's converter bakes gemma +1 into every M3 *norm.weight (incl.
    # the per-head qk-norms); the model's GemmaRMSNorm re-adds 1 at runtime.
    for name, target in (
        ("blk.4.attn_norm.weight", "model.layers.4.input_layernorm.weight"),
        ("blk.4.ffn_norm.weight", "model.layers.4.post_attention_layernorm.weight"),
        ("blk.4.attn_q_norm.weight", "model.layers.4.self_attn.q_norm.weight"),
        ("blk.4.attn_k_norm.weight", "model.layers.4.self_attn.k_norm.weight"),
        ("output_norm.weight", "model.norm.weight"),
    ):
        r = d("minimax-m3", name)
        assert r.kind == MAP and r.hf_name == target
        assert r.transform == "gemma_norm_minus_one"
    # M2 shares the MINIMAX alias but its norms are NOT +1-baked.
    assert d("minimax-m2", "blk.4.ffn_norm.weight").transform == "passthrough"
    assert d("minimax-m2", "blk.4.attn_q_norm.weight").transform == "passthrough"
    # Non-norm tensors on M3 stay passthrough (NEOX rope - no qk-permute).
    assert d("minimax-m3", "blk.4.attn_q.weight").transform == "passthrough"


# deepseek4 (DeepSeek V4 Flash, dwarfstar arch - not llama.cpp): complete
# override block, everything passthrough (NEOX-style tail rope applied after
# the low-rank projections; no norm bakes).
def test_deepseek4_mla_lite_attention_routing():
    assert d("deepseek4", "blk.2.attn_q_a.weight").hf_name == \
        "model.layers.2.attn.wq_a.weight"
    assert d("deepseek4", "blk.2.attn_q_a_norm.weight").hf_name == \
        "model.layers.2.attn.q_norm.weight"
    assert d("deepseek4", "blk.2.attn_q_b.weight").hf_name == \
        "model.layers.2.attn.wq_b.weight"
    assert d("deepseek4", "blk.2.attn_kv.weight").hf_name == \
        "model.layers.2.attn.wkv.weight"
    assert d("deepseek4", "blk.2.attn_kv_a_norm.weight").hf_name == \
        "model.layers.2.attn.kv_norm.weight"
    # wo_a passes through 2D; the vendored sanitize() reshapes the wire bytes
    # to the 3D (o_groups, o_lora, -1) MultiLinear layout on load.
    r = d("deepseek4", "blk.2.attn_output_a.weight")
    assert r.hf_name == "model.layers.2.attn.wo_a.weight"
    assert r.transform == "passthrough"
    assert d("deepseek4", "blk.2.attn_output_b.weight").hf_name == \
        "model.layers.2.attn.wo_b.weight"
    # Per-head fp32 sinks: raw array target, no `.weight`.
    assert d("deepseek4", "blk.2.attn_sinks.weight").hf_name == \
        "model.layers.2.attn.attn_sink"


def test_deepseek4_compressor_and_indexer_routing():
    assert d("deepseek4", "blk.2.attn_compressor_kv.weight").hf_name == \
        "model.layers.2.attn.compressor.wkv.weight"
    assert d("deepseek4", "blk.2.attn_compressor_gate.weight").hf_name == \
        "model.layers.2.attn.compressor.wgate.weight"
    # ape is a raw positional table (no `.weight` on the module attr).
    assert d("deepseek4", "blk.2.attn_compressor_ape.weight").hf_name == \
        "model.layers.2.attn.compressor.ape"
    assert d("deepseek4", "blk.2.attn_compressor_norm.weight").hf_name == \
        "model.layers.2.attn.compressor.norm.weight"
    # The indexer's own tensors are dotted; its private compressor's are
    # underscore-joined (both spellings as in the real GGUF).
    assert d("deepseek4", "blk.2.indexer.attn_q_b.weight").hf_name == \
        "model.layers.2.attn.indexer.wq_b.weight"
    assert d("deepseek4", "blk.2.indexer.proj.weight").hf_name == \
        "model.layers.2.attn.indexer.weights_proj.weight"
    assert d("deepseek4", "blk.2.indexer_compressor_kv.weight").hf_name == \
        "model.layers.2.attn.indexer.compressor.wkv.weight"
    assert d("deepseek4", "blk.2.indexer_compressor_ape.weight").hf_name == \
        "model.layers.2.attn.indexer.compressor.ape"


def test_deepseek4_hyper_connections_and_moe_routing():
    # HC params are raw fp32 arrays (fn/base/scale, no `.weight`), per block
    # and at the head.
    assert d("deepseek4", "blk.1.hc_attn_fn.weight").hf_name == \
        "model.layers.1.attn_hc.fn"
    assert d("deepseek4", "blk.1.hc_ffn_scale.weight").hf_name == \
        "model.layers.1.ffn_hc.scale"
    assert d("deepseek4", "output_hc_fn.weight").hf_name == "model.hc_head.fn"
    assert d("deepseek4", "output_hc_base.weight").hf_name == \
        "model.hc_head.base"
    # MoE under ffn.* (not the canonical mlp.*): router, selection-only
    # correction bias, and the raw I32 hash-route table.
    assert d("deepseek4", "blk.1.ffn_gate_inp.weight").hf_name == \
        "model.layers.1.ffn.gate.weight"
    assert d("deepseek4", "blk.1.exp_probs_b.bias").hf_name == \
        "model.layers.1.ffn.gate.e_score_correction_bias"
    assert d("deepseek4", "blk.0.ffn_gate_tid2eid.weight").hf_name == \
        "model.layers.0.ffn.gate.tid2eid"
    assert d("deepseek4", "blk.1.ffn_gate_exps.weight").hf_name == \
        "model.layers.1.ffn.switch_mlp.gate_proj.weight"
    assert d("deepseek4", "blk.1.ffn_down_shexp.weight").hf_name == \
        "model.layers.1.ffn.shared_experts.down_proj.weight"


def test_deepseek4_block_norms_and_everything_passthrough():
    # Block norms live on the block as attn_norm/ffn_norm (not the canonical
    # input/post_attention_layernorm), and NOTHING is transformed.
    assert d("deepseek4", "blk.3.attn_norm.weight").hf_name == \
        "model.layers.3.attn_norm.weight"
    assert d("deepseek4", "blk.3.ffn_norm.weight").hf_name == \
        "model.layers.3.ffn_norm.weight"
    for name in ("blk.3.attn_q_a.weight", "blk.3.attn_norm.weight",
                 "blk.3.hc_attn_fn.weight", "blk.3.ffn_gate_exps.weight",
                 "token_embd.weight", "output.weight", "output_norm.weight"):
        r = d("deepseek4", name)
        assert r.kind == MAP and r.transform == "passthrough", name
    assert d("deepseek4", "token_embd.weight").hf_name == \
        "model.embed_tokens.weight"
    assert d("deepseek4", "output.weight").hf_name == "lm_head.weight"


# hyv4 (HY4-preview): complete override block, everything passthrough. The MLA
# and MoE rows follow mlx_lm's deepseek_v32 naming so KQuantMultiLinear
# engages; the HY4-only rows (gate, sinks, iHC) have no canonical enum.
def test_hyv4_absorbed_mla_routing():
    assert d("hyv4", "blk.2.attn_q_a.weight").hf_name == \
        "model.layers.2.self_attn.q_a_proj.weight"
    assert d("hyv4", "blk.2.attn_q_a_norm.weight").hf_name == \
        "model.layers.2.self_attn.q_a_layernorm.weight"
    assert d("hyv4", "blk.2.attn_q_b.weight").hf_name == \
        "model.layers.2.self_attn.q_b_proj.weight"
    assert d("hyv4", "blk.2.attn_kv_a_mqa.weight").hf_name == \
        "model.layers.2.self_attn.kv_a_proj_with_mqa.weight"
    assert d("hyv4", "blk.2.attn_kv_a_norm.weight").hf_name == \
        "model.layers.2.self_attn.kv_a_layernorm.weight"
    # k_b/v_b are absorbed into the MultiLinear pair the loader swaps to
    # KQuantMultiLinear - the names must be exactly embed_q/unembed_out.
    assert d("hyv4", "blk.2.attn_k_b.weight").hf_name == \
        "model.layers.2.self_attn.embed_q.weight"
    assert d("hyv4", "blk.2.attn_v_b.weight").hf_name == \
        "model.layers.2.self_attn.unembed_out.weight"
    assert d("hyv4", "blk.2.attn_output.weight").hf_name == \
        "model.layers.2.self_attn.o_proj.weight"


def test_hyv4_attention_gate_and_sinks_routing():
    # The sigmoid gate on the decompressed attention output.
    assert d("hyv4", "blk.2.attn_gate.weight").hf_name == \
        "model.layers.2.self_attn.attn_gate.weight"
    # Per-head fp32 sinks: a raw array target, no `.weight` (gpt-oss spelling).
    assert d("hyv4", "blk.2.attn_sinks.weight").hf_name == \
        "model.layers.2.self_attn.sinks"


def test_hyv4_indexer_routing_includes_k_norm_bias():
    # k_norm is a LayerNorm here (weight AND bias), unlike the RMSNorm the
    # other DSA arches carry - dropping the bias loads clean and mis-selects.
    assert d("hyv4", "blk.5.indexer.attn_q_b.weight").hf_name == \
        "model.layers.5.self_attn.indexer.wq_b.weight"
    assert d("hyv4", "blk.5.indexer.attn_k.weight").hf_name == \
        "model.layers.5.self_attn.indexer.wk.weight"
    assert d("hyv4", "blk.5.indexer.k_norm.weight").hf_name == \
        "model.layers.5.self_attn.indexer.k_norm.weight"
    assert d("hyv4", "blk.5.indexer.k_norm.bias").hf_name == \
        "model.layers.5.self_attn.indexer.k_norm.bias"
    assert d("hyv4", "blk.5.indexer.proj.weight").hf_name == \
        "model.layers.5.self_attn.indexer.weights_proj.weight"


def test_hyv4_hyper_connection_routing():
    # iHC params are raw fp32 arrays (fn/base/scale, no `.weight`), one pair
    # per block plus the final-collapse head.
    assert d("hyv4", "blk.1.hc_attn_fn.weight").hf_name == \
        "model.layers.1.attn_hc.fn"
    assert d("hyv4", "blk.1.hc_attn_base.weight").hf_name == \
        "model.layers.1.attn_hc.base"
    assert d("hyv4", "blk.1.hc_attn_scale.weight").hf_name == \
        "model.layers.1.attn_hc.scale"
    assert d("hyv4", "blk.1.hc_ffn_fn.weight").hf_name == \
        "model.layers.1.ffn_hc.fn"
    assert d("hyv4", "blk.1.hc_ffn_base.weight").hf_name == \
        "model.layers.1.ffn_hc.base"
    assert d("hyv4", "blk.1.hc_ffn_scale.weight").hf_name == \
        "model.layers.1.ffn_hc.scale"
    assert d("hyv4", "output_hc_fn.weight").hf_name == "model.hc_head.fn"
    assert d("hyv4", "output_hc_base.weight").hf_name == "model.hc_head.base"
    assert d("hyv4", "output_hc_scale.weight").hf_name == "model.hc_head.scale"


def test_hyv4_moe_and_dense_mlp_routing():
    # MoE under the canonical mlp.* (unlike deepseek4's ffn.*).
    assert d("hyv4", "blk.1.ffn_gate_inp.weight").hf_name == \
        "model.layers.1.mlp.gate.weight"
    assert d("hyv4", "blk.1.exp_probs_b.bias").hf_name == \
        "model.layers.1.mlp.gate.e_score_correction_bias"
    assert d("hyv4", "blk.1.ffn_gate_exps.weight").hf_name == \
        "model.layers.1.mlp.switch_mlp.gate_proj.weight"
    assert d("hyv4", "blk.1.ffn_up_exps.weight").hf_name == \
        "model.layers.1.mlp.switch_mlp.up_proj.weight"
    assert d("hyv4", "blk.1.ffn_down_exps.weight").hf_name == \
        "model.layers.1.mlp.switch_mlp.down_proj.weight"
    assert d("hyv4", "blk.1.ffn_gate_shexp.weight").hf_name == \
        "model.layers.1.mlp.shared_experts.gate_proj.weight"
    assert d("hyv4", "blk.1.ffn_down_shexp.weight").hf_name == \
        "model.layers.1.mlp.shared_experts.down_proj.weight"
    # The leading dense block keeps the plain MLP names.
    assert d("hyv4", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    assert d("hyv4", "blk.0.ffn_up.weight").hf_name == \
        "model.layers.0.mlp.up_proj.weight"
    assert d("hyv4", "blk.0.ffn_down.weight").hf_name == \
        "model.layers.0.mlp.down_proj.weight"


# Every distinct tensor-name template in
# AngelSlim__Hy4-preview-GGUF/Hy4-preview-STQ1_0.gguf (2134 tensors), with the
# block index folded to {bid}. A closed-set pin: a new spelling on the wire, or
# a row deleted from the override block, breaks this rather than the 229 GB
# load.
_HYV4_WIRE_TEMPLATES = (
    "blk.{bid}.attn_gate.weight",
    "blk.{bid}.attn_k_b.weight",
    "blk.{bid}.attn_kv_a_mqa.weight",
    "blk.{bid}.attn_kv_a_norm.weight",
    "blk.{bid}.attn_norm.weight",
    "blk.{bid}.attn_output.weight",
    "blk.{bid}.attn_q_a.weight",
    "blk.{bid}.attn_q_a_norm.weight",
    "blk.{bid}.attn_q_b.weight",
    "blk.{bid}.attn_sinks.weight",
    "blk.{bid}.attn_v_b.weight",
    "blk.{bid}.exp_probs_b.bias",
    "blk.{bid}.ffn_down.weight",
    "blk.{bid}.ffn_down_exps.weight",
    "blk.{bid}.ffn_down_shexp.weight",
    "blk.{bid}.ffn_gate.weight",
    "blk.{bid}.ffn_gate_exps.weight",
    "blk.{bid}.ffn_gate_inp.weight",
    "blk.{bid}.ffn_gate_shexp.weight",
    "blk.{bid}.ffn_norm.weight",
    "blk.{bid}.ffn_up.weight",
    "blk.{bid}.ffn_up_exps.weight",
    "blk.{bid}.ffn_up_shexp.weight",
    "blk.{bid}.hc_attn_base.weight",
    "blk.{bid}.hc_attn_fn.weight",
    "blk.{bid}.hc_attn_scale.weight",
    "blk.{bid}.hc_ffn_base.weight",
    "blk.{bid}.hc_ffn_fn.weight",
    "blk.{bid}.hc_ffn_scale.weight",
    "blk.{bid}.indexer.attn_k.weight",
    "blk.{bid}.indexer.attn_q_b.weight",
    "blk.{bid}.indexer.k_norm.bias",
    "blk.{bid}.indexer.k_norm.weight",
    "blk.{bid}.indexer.proj.weight",
    "output.weight",
    "output_hc_base.weight",
    "output_hc_fn.weight",
    "output_hc_scale.weight",
    "output_norm.weight",
    "token_embd.weight",
)


def test_hyv4_every_wire_tensor_maps_to_a_unique_target():
    targets = {}
    for tpl in _HYV4_WIRE_TEMPLATES:
        name = tpl.format(bid=7)
        r = d("hyv4", name)
        assert r.kind == MAP, name
        assert r.transform == "passthrough", name
        assert r.hf_name not in targets, (name, targets.get(r.hf_name))
        targets[r.hf_name] = name
    assert len(targets) == len(_HYV4_WIRE_TEMPLATES)


def test_hyv4_global_tensors():
    assert d("hyv4", "token_embd.weight").hf_name == "model.embed_tokens.weight"
    assert d("hyv4", "output_norm.weight").hf_name == "model.norm.weight"
    assert d("hyv4", "output.weight").hf_name == "lm_head.weight"


# hunyuan-moe: NEOX, per-head qk-norm (query/key_layernorm), shared expert
def test_hunyuan_attn_passthrough_and_per_head_qk_norm_naming():
    # NEOX rope => Q/K passthrough; per-head qk-norms map to mlx-lm's
    # query_layernorm/key_layernorm (NOT the canonical q_norm/k_norm).
    for t in ("attn_q", "attn_k", "attn_v"):
        assert d("hunyuan-moe", f"blk.2.{t}.weight").transform == "passthrough"
    assert d("hunyuan-moe", "blk.2.attn_q_norm.weight").hf_name == \
        "model.layers.2.self_attn.query_layernorm.weight"
    assert d("hunyuan-moe", "blk.2.attn_k_norm.weight").hf_name == \
        "model.layers.2.self_attn.key_layernorm.weight"
    assert d("hunyuan-moe", "blk.2.attn_output.weight").hf_name == \
        "model.layers.2.self_attn.o_proj.weight"


def test_hunyuan_moe_router_experts_and_shared():
    # Router wraps in a Gate module (.wg); experts -> switch_mlp; per-layer shared
    # expert -> shared_mlp (a plain MLP).
    assert d("hunyuan-moe", "blk.0.ffn_gate_inp.weight").hf_name == \
        "model.layers.0.mlp.gate.wg.weight"
    assert d("hunyuan-moe", "blk.0.ffn_gate_exps.weight").hf_name == \
        "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    assert d("hunyuan-moe", "blk.0.ffn_down_exps.weight").hf_name == \
        "model.layers.0.mlp.switch_mlp.down_proj.weight"
    assert d("hunyuan-moe", "blk.0.ffn_up_shexp.weight").hf_name == \
        "model.layers.0.mlp.shared_mlp.up_proj.weight"
    assert d("hunyuan-moe", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"


def test_hunyuan_globals():
    assert d("hunyuan-moe", "blk.0.attn_norm.weight").hf_name == \
        "model.layers.0.input_layernorm.weight"
    assert d("hunyuan-moe", "token_embd.weight").hf_name == \
        "model.embed_tokens.weight"
    assert d("hunyuan-moe", "output_norm.weight").hf_name == "model.norm.weight"


# hy_v3: NEOX, canonical per-head qk-norm names, sigmoid MoE + expert bias
def test_hy_v3_attn_passthrough_canonical_qk_norm():
    # NEOX rope => Q/K passthrough; qk-norms land on mlx-lm's stock
    # q_norm/k_norm (unlike hunyuan-moe's query/key_layernorm).
    for t in ("attn_q", "attn_k", "attn_v"):
        r = d("hy_v3", f"blk.2.{t}.weight")
        assert r.kind == MAP and r.transform == "passthrough"
    assert d("hy_v3", "blk.2.attn_q_norm.weight").hf_name == \
        "model.layers.2.self_attn.q_norm.weight"
    assert d("hy_v3", "blk.2.attn_k_norm.weight").hf_name == \
        "model.layers.2.self_attn.k_norm.weight"
    assert d("hy_v3", "blk.2.ffn_norm.weight").hf_name == \
        "model.layers.2.post_attention_layernorm.weight"


def test_hy_v3_router_experts_shared_and_bias():
    # Router wraps in a MoEGate module (mlp.router); experts -> switch_mlp;
    # shared expert -> shared_mlp. The selection bias is stored SUFFIX-LESS
    # on Hy3 GGUFs (deepseek/glm4moe use .bias) - both spellings map.
    assert d("hy_v3", "blk.1.ffn_gate_inp.weight").hf_name == \
        "model.layers.1.mlp.router.gate.weight"
    assert d("hy_v3", "blk.1.exp_probs_b").hf_name == \
        "model.layers.1.mlp.router.expert_bias"
    assert d("hy_v3", "blk.1.exp_probs_b.bias").hf_name == \
        "model.layers.1.mlp.router.expert_bias"
    assert d("hy_v3", "blk.1.ffn_gate_exps.weight").hf_name == \
        "model.layers.1.mlp.switch_mlp.gate_proj.weight"
    assert d("hy_v3", "blk.1.ffn_down_exps.weight").hf_name == \
        "model.layers.1.mlp.switch_mlp.down_proj.weight"
    assert d("hy_v3", "blk.1.ffn_up_shexp.weight").hf_name == \
        "model.layers.1.mlp.shared_mlp.up_proj.weight"


def test_hy_v3_dense_layer_and_globals_canonical():
    # The leading dense layer resolves via the canonical map (mlp.*).
    assert d("hy_v3", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    assert d("hy_v3", "blk.0.ffn_down.weight").hf_name == \
        "model.layers.0.mlp.down_proj.weight"
    assert d("hy_v3", "token_embd.weight").hf_name == \
        "model.embed_tokens.weight"
    assert d("hy_v3", "output.weight").hf_name == "lm_head.weight"
    assert d("hy_v3", "output_norm.weight").hf_name == "model.norm.weight"


def test_hy_v3_nextn_tensors_skip():
    # The MTP block's nextn.* extras have no HF target in the trunk (the
    # drafter loads them via remap_mtp_arrays); they must SKIP, not FAIL.
    for t in ("nextn.eh_proj.weight", "nextn.enorm.weight",
              "nextn.hnorm.weight", "nextn.shared_head_norm.weight"):
        assert d("hy_v3", f"blk.80.{t}").kind == SKIP
    # Its standard decoder tensors map onto model.layers.80.* (stripped by
    # the vendored sanitize).
    assert d("hy_v3", "blk.80.attn_q.weight").hf_name == \
        "model.layers.80.self_attn.q_proj.weight"


# granitehybrid: Mamba2+attn hybrid, NORM rope => qk_permute, fused shexp
def test_granitehybrid_attention_layers_permute():
    # NORM rope + mlx-lm traditional=False => Q/K must be un-permuted (explicit
    # rows - the canonical qk_permute gate is LLAMA-alias-only).
    for t in ("attn_q", "attn_k"):
        r = d("granitehybrid", f"blk.5.{t}.weight")
        assert r.kind == MAP and r.transform == "qk_permute"
    assert d("granitehybrid", "blk.5.attn_v.weight").transform == "passthrough"
    assert d("granitehybrid", "blk.5.attn_output.weight").hf_name == \
        "model.layers.5.self_attn.o_proj.weight"
    assert d("granitehybrid", "blk.5.attn_norm.weight").hf_name == \
        "model.layers.5.input_layernorm.weight"


def test_granitehybrid_mamba_family():
    cases = {
        "blk.2.ssm_in.weight": ("model.layers.2.mamba.in_proj.weight",
                                "passthrough"),
        "blk.2.ssm_conv1d.weight": ("model.layers.2.mamba.conv1d.weight",
                                    "conv1d_unsqueeze"),
        "blk.2.ssm_conv1d.bias": ("model.layers.2.mamba.conv1d.bias",
                                  "passthrough"),
        "blk.2.ssm_dt.bias": ("model.layers.2.mamba.dt_bias", "passthrough"),
        "blk.2.ssm_a": ("model.layers.2.mamba.A_log", "ssm_a_to_a_log"),
        "blk.2.ssm_d": ("model.layers.2.mamba.D", "flatten"),
        "blk.2.ssm_norm.weight": ("model.layers.2.mamba.norm.weight",
                                  "flatten"),
        "blk.2.ssm_out.weight": ("model.layers.2.mamba.out_proj.weight",
                                 "passthrough"),
    }
    for name, (hf, transform) in cases.items():
        r = d("granitehybrid", name)
        assert r.kind == MAP and (r.hf_name, r.transform) == (hf, transform), \
            f"{name}: got ({r.hf_name}, {r.transform})"


def test_granitehybrid_moe_router_experts_and_fused_shared():
    # Router wraps a TopKGating module (.layer); experts -> switch_mlp; the
    # shared MLP consumes the loader-fused gate_up tensor as input_linear.
    assert d("granitehybrid", "blk.0.ffn_gate_inp.weight").hf_name == \
        "model.layers.0.block_sparse_moe.router.layer.weight"
    assert d("granitehybrid", "blk.0.ffn_gate_exps.weight").hf_name == \
        "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"
    assert d("granitehybrid", "blk.0.ffn_down_exps.weight").hf_name == \
        "model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"
    assert d("granitehybrid", "blk.0.ffn_gate_up_shexp.weight").hf_name == \
        "model.layers.0.shared_mlp.input_linear.weight"
    assert d("granitehybrid", "blk.0.ffn_down_shexp.weight").hf_name == \
        "model.layers.0.shared_mlp.output_linear.weight"
    assert d("granitehybrid", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"


def test_granitehybrid_dense_variant_and_globals():
    # Non-MoE hybrid variants use a plain dense MLP (canonical mlp.* targets).
    assert d("granitehybrid", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    assert d("granitehybrid", "blk.0.ffn_down.weight").hf_name == \
        "model.layers.0.mlp.down_proj.weight"
    assert d("granitehybrid", "token_embd.weight").hf_name == \
        "model.embed_tokens.weight"
    assert d("granitehybrid", "output_norm.weight").hf_name == \
        "model.norm.weight"


# falcon-h1: parallel attn+Mamba2 every layer, NEOX rope => no permute
def test_falcon_h1_attention_no_permute():
    # NEOX rope => Q/K resolve canonically WITHOUT qk_permute (the canonical
    # permute gate is LLAMA-alias-only and FALCON_H1 claims no attention rows).
    for t in ("attn_q", "attn_k", "attn_v"):
        r = d("falcon-h1", f"blk.5.{t}.weight")
        assert r.kind == MAP and r.transform == "passthrough", \
            f"{t}: got ({r.kind}, {r.transform})"
    assert d("falcon-h1", "blk.5.attn_q.weight").hf_name == \
        "model.layers.5.self_attn.q_proj.weight"
    assert d("falcon-h1", "blk.5.attn_output.weight").hf_name == \
        "model.layers.5.self_attn.o_proj.weight"
    assert d("falcon-h1", "blk.5.attn_norm.weight").hf_name == \
        "model.layers.5.input_layernorm.weight"


def test_falcon_h1_mamba_family():
    # Same Mamba2 wire family as granitehybrid, housed under `mamba.*`.
    cases = {
        "blk.2.ssm_in.weight": ("model.layers.2.mamba.in_proj.weight",
                                "passthrough"),
        "blk.2.ssm_conv1d.weight": ("model.layers.2.mamba.conv1d.weight",
                                    "conv1d_unsqueeze"),
        "blk.2.ssm_conv1d.bias": ("model.layers.2.mamba.conv1d.bias",
                                  "passthrough"),
        "blk.2.ssm_dt.bias": ("model.layers.2.mamba.dt_bias", "passthrough"),
        "blk.2.ssm_a": ("model.layers.2.mamba.A_log", "ssm_a_to_a_log"),
        "blk.2.ssm_d": ("model.layers.2.mamba.D", "flatten"),
        "blk.2.ssm_norm.weight": ("model.layers.2.mamba.norm.weight",
                                  "flatten"),
        "blk.2.ssm_out.weight": ("model.layers.2.mamba.out_proj.weight",
                                 "passthrough"),
    }
    for name, (hf, transform) in cases.items():
        r = d("falcon-h1", name)
        assert r.kind == MAP and (r.hf_name, r.transform) == (hf, transform), \
            f"{name}: got ({r.hf_name}, {r.transform})"


def test_falcon_h1_ffn_and_globals():
    # mlx-lm houses the dense MLP under feed_forward.* (not mlp.*), the pre-MLP
    # norm as pre_ff_layernorm, and the final norm as model.final_layernorm.
    # llama.cpp writes falcon-h1's ffn_norm with NO .weight suffix - both
    # spellings must resolve.
    assert d("falcon-h1", "blk.3.ffn_gate.weight").hf_name == \
        "model.layers.3.feed_forward.gate_proj.weight"
    assert d("falcon-h1", "blk.3.ffn_up.weight").hf_name == \
        "model.layers.3.feed_forward.up_proj.weight"
    assert d("falcon-h1", "blk.3.ffn_down.weight").hf_name == \
        "model.layers.3.feed_forward.down_proj.weight"
    assert d("falcon-h1", "blk.3.ffn_norm").hf_name == \
        "model.layers.3.pre_ff_layernorm.weight"
    assert d("falcon-h1", "blk.3.ffn_norm.weight").hf_name == \
        "model.layers.3.pre_ff_layernorm.weight"
    assert d("falcon-h1", "output_norm.weight").hf_name == \
        "model.final_layernorm.weight"
    assert d("falcon-h1", "token_embd.weight").hf_name == \
        "model.embed_tokens.weight"
    assert d("falcon-h1", "output.weight").hf_name == "lm_head.weight"


# qwen3next: gated-DeltaNet hybrid, NEOX, fused + split GDN layouts
def test_qwen3next_attention_no_permute_and_norms():
    # NEOX => canonical un-permuted attention; the attention output gate rides
    # fused inside attn_q so it needs no row of its own. The +1-baked norms
    # pass through (mlx-lm's runtime weights carry the +1).
    for t in ("attn_q", "attn_k"):
        r = d("qwen3next", f"blk.7.{t}.weight")
        assert r.kind == MAP and r.transform == "passthrough", \
            f"{t}: got ({r.kind}, {r.transform})"
    assert d("qwen3next", "blk.7.attn_q_norm.weight").hf_name == \
        "model.layers.7.self_attn.q_norm.weight"
    assert d("qwen3next", "blk.7.attn_norm.weight").hf_name == \
        "model.layers.7.input_layernorm.weight"
    assert d("qwen3next", "blk.7.post_attention_norm.weight").hf_name == \
        "model.layers.7.post_attention_layernorm.weight"


def test_qwen3next_gdn_family():
    # Legacy fused input projections claim explicit rows (canonical SSM_IN
    # targets qwen3.5's in_proj); conv/dt/A/norm/out resolve canonically onto
    # linear_attn.* with the standard transforms.
    cases = {
        "blk.2.ssm_in.weight": (
            "model.layers.2.linear_attn.in_proj_qkvz.weight", "passthrough"),
        "blk.2.ssm_ba.weight": (
            "model.layers.2.linear_attn.in_proj_ba.weight", "passthrough"),
        "blk.2.ssm_conv1d.weight": (
            "model.layers.2.linear_attn.conv1d.weight", "conv1d_unsqueeze"),
        "blk.2.ssm_dt.bias": (
            "model.layers.2.linear_attn.dt_bias", "passthrough"),
        "blk.2.ssm_a": (
            "model.layers.2.linear_attn.A_log", "ssm_a_to_a_log"),
        "blk.2.ssm_norm.weight": (
            "model.layers.2.linear_attn.norm.weight", "passthrough"),
        "blk.2.ssm_out.weight": (
            "model.layers.2.linear_attn.out_proj.weight", "passthrough"),
    }
    for name, (hf, transform) in cases.items():
        r = d("qwen3next", name)
        assert r.kind == MAP and (r.hf_name, r.transform) == (hf, transform), \
            f"{name}: got ({r.hf_name}, {r.transform})"


def test_qwen3next_split_layout_maps_to_split_modules():
    # The newer split GDN layout resolves through the canonical
    # ATTN_QKV/ATTN_GATE slots onto the qwen3.5 in_proj_qkv/in_proj_z names -
    # gdn_patches._patch_qwen3next_split_gdn creates exactly those modules
    # (gated by the synth's gdn_split_layout flag).
    r = d("qwen3next", "blk.0.attn_qkv.weight")
    assert r.kind == MAP and r.transform == "passthrough"
    assert r.hf_name == "model.layers.0.linear_attn.in_proj_qkv.weight"
    r = d("qwen3next", "blk.0.attn_gate.weight")
    assert r.kind == MAP and r.transform == "passthrough"
    assert r.hf_name == "model.layers.0.linear_attn.in_proj_z.weight"


def test_qwen3next_moe_and_shared_expert():
    assert d("qwen3next", "blk.0.ffn_gate_inp.weight").hf_name == \
        "model.layers.0.mlp.gate.weight"
    assert d("qwen3next", "blk.0.ffn_up_exps.weight").hf_name == \
        "model.layers.0.mlp.switch_mlp.up_proj.weight"
    assert d("qwen3next", "blk.0.ffn_down_shexp.weight").hf_name == \
        "model.layers.0.mlp.shared_expert.down_proj.weight"
    rg = d("qwen3next", "blk.0.ffn_gate_inp_shexp.weight")
    assert rg.hf_name == "model.layers.0.mlp.shared_expert_gate.weight"
    assert rg.transform == "gate_1d_unsqueeze"


# mixtral: llama-arch sparse MoE -> block_sparse_moe (experts coalesced)
def test_mixtral_moe_experts_and_router():
    # Mixtral ships as general.architecture='llama'; after the loader coalesces
    # the per-expert split tensors into `_exps`, the LLAMA overrides route them
    # (and the router) to mlx-lm mixtral's block_sparse_moe (SwitchGLU).
    rg = d("llama", "blk.0.ffn_gate_exps.weight")
    assert rg.kind == MAP and rg.transform == "passthrough"
    assert rg.hf_name == "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"
    assert d("llama", "blk.0.ffn_up_exps.weight").hf_name == \
        "model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"
    assert d("llama", "blk.0.ffn_down_exps.weight").hf_name == \
        "model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"
    rr = d("llama", "blk.0.ffn_gate_inp.weight")
    assert rr.kind == MAP and rr.hf_name == "model.layers.0.block_sparse_moe.gate.weight"


def test_mixtral_attention_still_permuted_like_llama():
    # Mixtral uses the llama attention layout, so Q/K keep the llama qk_permute
    # and the standard self_attn targets (the MoE overrides don't touch attn).
    rq = d("llama", "blk.0.attn_q.weight")
    assert rq.transform == "qk_permute"
    assert rq.hf_name == "model.layers.0.self_attn.q_proj.weight"
    assert d("llama", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"


# qwen2: no qk_permute, QKV biases, ffn_norm pinned
def test_qwen2_q_proj_not_permuted():
    r = d("qwen2", "blk.0.attn_q.weight")
    assert r.kind == MAP and r.transform == "passthrough"
    assert r.hf_name == "model.layers.0.self_attn.q_proj.weight"


def test_qwen2_qkv_biases_map_to_bias_slots():
    for t, proj in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj")):
        r = d("qwen2", f"blk.5.{t}.bias")
        assert r.kind == MAP and r.bid == 5
        assert r.hf_name == f"model.layers.5.self_attn.{proj}.bias"
        assert r.transform == "passthrough"


def test_qwen2_ffn_norm_is_post_attention():
    r = d("qwen2", "blk.0.ffn_norm.weight")
    assert r.hf_name == "model.layers.0.post_attention_layernorm.weight"


# gemma2/gemma3: norm-unbake; gemma4: NOT unbaked
def test_gemma2_norms_get_minus_one_transform():
    # input norm (canonical) and the final norm both end in "norm.weight".
    assert d("gemma2", "blk.0.attn_norm.weight").transform == "gemma_norm_minus_one"
    assert d("gemma2", "output_norm.weight").transform == "gemma_norm_minus_one"
    # ffn_norm is pinned to pre_feedforward (collision) AND unbaked.
    r = d("gemma2", "blk.0.ffn_norm.weight")
    assert r.hf_name == "model.layers.0.pre_feedforward_layernorm.weight"
    assert r.transform == "gemma_norm_minus_one"


def test_gemma3_ffn_norm_is_pre_feedforward_and_unbaked():
    r = d("gemma3", "blk.1.ffn_norm.weight")
    assert r.hf_name == "model.layers.1.pre_feedforward_layernorm.weight"
    assert r.transform == "gemma_norm_minus_one"


def test_gemma4_norms_are_not_unbaked():
    # gemma4 shares the GEMMA3 remap alias but its mlx_lm RMSNorm uses the
    # weight as-is - no +1 bake to undo.
    assert d("gemma4", "blk.0.attn_norm.weight").transform == "passthrough"
    assert d("gemma4", "output_norm.weight").transform == "passthrough"


def test_gemma4_moe_split_and_router():
    r = d("gemma4", "blk.0.ffn_gate_up_exps.weight")
    assert r.kind == MAP and r.transform == "moe_split_gate_up"
    assert r.hf_name == "model.layers.0.experts.switch_glu.gate_up_proj.weight"
    rr = d("gemma4", "blk.0.ffn_gate_inp.weight")
    assert rr.hf_name == "model.layers.0.router.proj.weight"


def test_gemma4_architectural_scale_claimed_before_universal_skip():
    # ffn_gate_inp.scale is a real router tensor, not Unsloth-UD metadata -
    # the arch-priority override must claim it before the universal .scale skip.
    r = d("gemma4", "blk.0.ffn_gate_inp.scale")
    assert r.kind == MAP and r.hf_name == "model.layers.0.router.scale"
    r2 = d("gemma4", "blk.0.ffn_down_exps.scale")
    assert r2.kind == MAP and r2.hf_name == "model.layers.0.router.per_expert_scale"


# gemma-embedding (EmbeddingGemma): GEMMA3 backbone + a 2-layer dense head
def test_gemma_embedding_reuses_gemma3_backbone_unbake():
    # backbone norms unbake like gemma3; ffn_norm pins to pre_feedforward.
    assert (d("gemma-embedding", "blk.0.attn_norm.weight").transform
            == "gemma_norm_minus_one")
    r = d("gemma-embedding", "blk.1.ffn_norm.weight")
    assert r.hf_name == "model.layers.1.pre_feedforward_layernorm.weight"
    assert r.transform == "gemma_norm_minus_one"
    assert (d("gemma-embedding", "blk.0.attn_q.weight").hf_name
            == "model.layers.0.self_attn.q_proj.weight")


def test_gemma_embedding_globals_target_encoder_tree():
    assert (d("gemma-embedding", "token_embd.weight").hf_name
            == "model.embed_tokens.weight")
    on = d("gemma-embedding", "output_norm.weight")
    assert on.hf_name == "model.norm.weight"
    assert on.transform == "gemma_norm_minus_one"


def test_gemma_embedding_dense_head_claimed_before_universal_skip():
    # dense_2/dense_3 are the projection head; the arch-priority override must
    # claim them as dense.0/dense.1 before gguf-py's GEMMA_EMBEDDING templates
    # short-circuit them to a canonical SKIP.
    r2 = d("gemma-embedding", "dense_2.weight")
    assert r2.kind == MAP and r2.transform == "passthrough"
    assert r2.hf_name == "dense.0.weight"
    r3 = d("gemma-embedding", "dense_3.weight")
    assert r3.kind == MAP and r3.transform == "passthrough"
    assert r3.hf_name == "dense.1.weight"


# gemma3n: standalone LanguageModel (unprefixed), AltUp/LAuReL, no unbake
def test_gemma3n_norms_are_passthrough_not_unbaked():
    # gemma3n uses plain RMSNorm (weight as-is) - no +1 bake to undo.
    assert d("gemma3n", "blk.0.attn_norm.weight").transform == "passthrough"
    assert d("gemma3n", "output_norm.weight").transform == "passthrough"


def test_gemma3n_targets_are_unprefixed_language_model_paths():
    # built as a bare LanguageModel, so targets carry no model./language_model.
    assert d("gemma3n", "token_embd.weight").hf_name == "embed_tokens.weight"
    assert d("gemma3n", "output_norm.weight").hf_name == "norm.weight"
    r = d("gemma3n", "blk.3.attn_q.weight")
    assert r.kind == MAP and r.hf_name == "layers.3.self_attn.q_proj.weight"


def test_gemma3n_four_norm_sandwich():
    assert (d("gemma3n", "blk.1.post_attention_norm.weight").hf_name
            == "layers.1.post_attention_layernorm.weight")
    assert (d("gemma3n", "blk.1.ffn_norm.weight").hf_name
            == "layers.1.pre_feedforward_layernorm.weight")
    assert (d("gemma3n", "blk.1.post_ffw_norm.weight").hf_name
            == "layers.1.post_feedforward_layernorm.weight")


def test_gemma3n_altup_stacked_projections_split():
    r = d("gemma3n", "altup_proj.weight")
    assert r.kind == MAP and r.transform == "altup_split"
    assert r.hf_name == "altup_projections.weight"
    rr = d("gemma3n", "altup_unembd_proj.weight")
    assert rr.transform == "altup_split"
    assert rr.hf_name == "altup_unembed_projections.weight"


def test_gemma3n_altup_correct_scale_is_raw_param_no_weight_suffix():
    # mlx_lm's correct_output_scale is a bare mx.array, not a Linear - the
    # target must NOT carry a .weight suffix.
    r = d("gemma3n", "blk.2.altup_correct_scale.weight")
    assert r.kind == MAP and r.hf_name == "layers.2.altup.correct_output_scale"
    assert r.transform == "passthrough"
    # the coef Linears DO keep .weight
    assert (d("gemma3n", "blk.2.altup_predict_coef.weight").hf_name
            == "layers.2.altup.prediction_coefs.weight")


def test_gemma3n_laurel_and_per_layer_input_blocks():
    assert (d("gemma3n", "blk.4.laurel_l.weight").hf_name
            == "layers.4.laurel.linear_left.weight")
    assert (d("gemma3n", "blk.4.laurel_post_norm.weight").hf_name
            == "layers.4.laurel.post_laurel_norm.weight")
    assert (d("gemma3n", "blk.4.inp_gate.weight").hf_name
            == "layers.4.per_layer_input_gate.weight")
    assert (d("gemma3n", "blk.4.proj.weight").hf_name
            == "layers.4.per_layer_projection.weight")
    assert (d("gemma3n", "blk.4.post_norm.weight").hf_name
            == "layers.4.post_per_layer_input_norm.weight")


# gemma (Gemma-1): llama-style norms, unbaked; no qk_permute
def test_gemma1_norms_unbaked_and_ffn_is_post_attention():
    # Gemma-1 has only input + post-attention norms (like Llama), so ffn_norm
    # pins to post_attention_layernorm - NOT pre_feedforward (gemma-2/3).
    r = d("gemma", "blk.0.ffn_norm.weight")
    assert r.hf_name == "model.layers.0.post_attention_layernorm.weight"
    assert r.transform == "gemma_norm_minus_one"
    # input + final norms also get the +1 bake undone.
    assert d("gemma", "blk.0.attn_norm.weight").transform == "gemma_norm_minus_one"
    assert d("gemma", "output_norm.weight").transform == "gemma_norm_minus_one"


def test_gemma1_q_not_permuted():
    # Gemma (unlike Llama) does not permute Q/K at convert time.
    r = d("gemma", "blk.0.attn_q.weight")
    assert r.kind == MAP and r.transform == "passthrough"
    assert r.hf_name == "model.layers.0.self_attn.q_proj.weight"


# phi3: fused qkv + fused gate_up
def test_phi3_fused_projections():
    rq = d("phi3", "blk.0.attn_qkv.weight")
    assert rq.kind == MAP and rq.hf_name == "model.layers.0.self_attn.qkv_proj.weight"
    ru = d("phi3", "blk.0.ffn_up.weight")
    assert ru.kind == MAP and ru.hf_name == "model.layers.0.mlp.gate_up_proj.weight"
    rn = d("phi3", "blk.0.ffn_norm.weight")
    assert rn.hf_name == "model.layers.0.post_attention_layernorm.weight"


# glm4: 4 norms (non-standard names) + fused gate_up + qkv bias
def test_glm4_four_norms_map_to_mlx_lm_names():
    # mlx_lm glm4's four per-layer norms use non-standard attribute names; the
    # canonical reverse-map would mis-target three of them, so all are pinned.
    assert d("glm4", "blk.0.attn_norm.weight").hf_name == \
        "model.layers.0.input_layernorm.weight"
    assert d("glm4", "blk.0.post_attention_norm.weight").hf_name == \
        "model.layers.0.post_self_attn_layernorm.weight"
    assert d("glm4", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"
    assert d("glm4", "blk.0.post_ffw_norm.weight").hf_name == \
        "model.layers.0.post_mlp_layernorm.weight"


def test_glm4_fused_gate_up_and_qkv_bias_no_permute():
    # ffn_up IS the fused [gate; up] (phi-3 precedent).
    ru = d("glm4", "blk.0.ffn_up.weight")
    assert ru.kind == MAP and ru.transform == "passthrough"
    assert ru.hf_name == "model.layers.0.mlp.gate_up_proj.weight"
    # q/k/v biases claimed explicitly (canonical .bias path would mis-name them).
    rb = d("glm4", "blk.3.attn_k.bias")
    assert rb.kind == MAP and rb.hf_name == "model.layers.3.self_attn.k_proj.bias"
    # Q weight goes canonical, NOT permuted (GLM uses interleaved rope natively).
    rq = d("glm4", "blk.3.attn_q.weight")
    assert rq.kind == MAP and rq.transform == "passthrough"
    assert rq.hf_name == "model.layers.3.self_attn.q_proj.weight"


# qwen35moe: switch_mlp experts + shared-expert gate unsqueeze
def test_qwen35moe_experts_and_shared_gate():
    rg = d("qwen35moe", "blk.0.ffn_gate_exps.weight")
    assert rg.hf_name == "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    rs = d("qwen35moe", "blk.0.ffn_gate_inp_shexp.weight")
    assert rs.kind == MAP and rs.transform == "gate_1d_unsqueeze"
    assert rs.hf_name == "model.layers.0.mlp.shared_expert_gate.weight"


# qwen4exp: qwen35 GDN tensors via the canonical map; hyper-connections, PLE,
# QSA indexer and MoE targets via the priority overrides
def test_qwen4exp_hyper_connections_ple_indexer_routing():
    for w, tr in (("attn_qkv", "passthrough"), ("attn_gate", "passthrough"),
                  ("ssm_conv1d", "conv1d_unsqueeze")):
        r = d("qwen4exp", f"blk.0.{w}.weight")
        assert r.kind == MAP and r.transform == tr, w
        assert r.hf_name.startswith("model.layers.0.linear_attn.")
    ra = d("qwen4exp", "blk.0.ssm_a")
    assert ra.transform == "ssm_a_to_a_log"
    assert ra.hf_name == "model.layers.0.linear_attn.A_log"
    assert d("qwen4exp", "blk.0.ssm_dt.bias").hf_name == \
        "model.layers.0.linear_attn.dt_bias"
    for hc in ("hc_attn", "hc_ffn"):
        for leaf in ("norm", "down", "up", "inject"):
            r = d("qwen4exp", f"blk.2.{hc}_{leaf}.weight")
            assert r.kind == MAP and r.transform == "passthrough"
            assert r.hf_name == f"model.layers.2.{hc}.{leaf}.weight"
    for leaf in ("norm", "down", "up"):
        r = d("qwen4exp", f"output_hc_{leaf}.weight")
        assert r.kind == MAP and r.hf_name == f"model.hc_head.{leaf}.weight"
    # no output_norm: the head mixer carries it; a stray one would be a fail
    rt = d("qwen4exp", "per_layer_token_embd.weight")
    assert rt.kind == MAP and rt.hf_name == "model.ple_embed.weight"
    assert rt.transform == "passthrough"
    rc = d("qwen4exp", "blk.1.ple_conv1d.weight")
    assert rc.transform == "conv1d_unsqueeze"
    assert rc.hf_name == "model.layers.1.ple.conv1d.weight"
    for w, t in (("ple_key", "key_proj"), ("ple_value", "value_proj"),
                 ("ple_norm_key", "norm_key"), ("ple_norm_query", "norm_query"),
                 ("ple_norm_conv", "norm_conv")):
        assert d("qwen4exp", f"blk.1.{w}.weight").hf_name == \
            f"model.layers.1.ple.{t}.weight"
    for w in ("q_proj", "k_proj", "q_norm", "k_norm"):
        r = d("qwen4exp", f"blk.3.indexer.{w}.weight")
        assert r.kind == MAP and r.transform == "passthrough"
        assert r.hf_name == f"model.layers.3.self_attn.indexer.{w}.weight"
    # gated attention: q_proj carries [q | gate]; plain canonical targets
    assert d("qwen4exp", "blk.3.attn_q.weight").hf_name == \
        "model.layers.3.self_attn.q_proj.weight"
    assert d("qwen4exp", "blk.3.attn_k_norm.weight").hf_name == \
        "model.layers.3.self_attn.k_norm.weight"
    rg = d("qwen4exp", "blk.0.ffn_gate_exps.weight")
    assert rg.hf_name == "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    rs = d("qwen4exp", "blk.0.ffn_gate_inp_shexp.weight")
    assert rs.kind == MAP and rs.transform == "gate_1d_unsqueeze"
    assert rs.hf_name == "model.layers.0.mlp.shared_expert_gate.weight"
    assert d("qwen4exp", "output.weight").hf_name == "lm_head.weight"


# qwen2moe: switch_mlp experts + shared expert + qkv bias + ffn_norm
def test_qwen2moe_experts_shared_and_qkv_bias():
    rg = d("qwen2moe", "blk.0.ffn_gate_exps.weight")
    assert rg.hf_name == "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    rd = d("qwen2moe", "blk.0.ffn_down_exps.weight")
    assert rd.hf_name == "model.layers.0.mlp.switch_mlp.down_proj.weight"
    # shared expert + its 1D sigmoid gate (unsqueezed), as for qwen35moe.
    rsh = d("qwen2moe", "blk.0.ffn_up_shexp.weight")
    assert rsh.hf_name == "model.layers.0.mlp.shared_expert.up_proj.weight"
    rsg = d("qwen2moe", "blk.0.ffn_gate_inp_shexp.weight")
    assert rsg.kind == MAP and rsg.transform == "gate_1d_unsqueeze"
    assert rsg.hf_name == "model.layers.0.mlp.shared_expert_gate.weight"
    # router, ffn_norm pin, and a QKV bias (qwen2_moe builds q/k/v with bias).
    assert d("qwen2moe", "blk.0.ffn_gate_inp.weight").hf_name == \
        "model.layers.0.mlp.gate.weight"
    assert d("qwen2moe", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"
    rb = d("qwen2moe", "blk.5.attn_v.bias")
    assert rb.kind == MAP and rb.hf_name == "model.layers.5.self_attn.v_proj.bias"


# qwen3moe: switch_mlp experts (no shared expert) + ffn_norm pinned
def test_qwen3moe_experts_and_ffn_norm():
    rg = d("qwen3moe", "blk.0.ffn_gate_exps.weight")
    assert rg.hf_name == "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    rd = d("qwen3moe", "blk.0.ffn_down_exps.weight")
    assert rd.hf_name == "model.layers.0.mlp.switch_mlp.down_proj.weight"
    rr = d("qwen3moe", "blk.0.ffn_gate_inp.weight")
    assert rr.hf_name == "model.layers.0.mlp.gate.weight"
    rn = d("qwen3moe", "blk.0.ffn_norm.weight")
    assert rn.hf_name == "model.layers.0.post_attention_layernorm.weight"


# deepseek2: MLA absorbed up-projections + fine-grained MoE
def test_deepseek2_mla_projections():
    # q path: down-proj to a latent, RMSNorm, up-proj to per-head q.
    assert d("deepseek2", "blk.0.attn_q_a.weight").hf_name == \
        "model.layers.0.self_attn.q_a_proj.weight"
    assert d("deepseek2", "blk.0.attn_q_a_norm.weight").hf_name == \
        "model.layers.0.self_attn.q_a_layernorm.weight"
    assert d("deepseek2", "blk.0.attn_q_b.weight").hf_name == \
        "model.layers.0.self_attn.q_b_proj.weight"
    # kv path: joint down-proj (with the decoupled-rope MQA key), RMSNorm.
    assert d("deepseek2", "blk.0.attn_kv_a_mqa.weight").hf_name == \
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
    assert d("deepseek2", "blk.0.attn_kv_a_norm.weight").hf_name == \
        "model.layers.0.self_attn.kv_a_layernorm.weight"
    # The "absorbed" per-head up-projections: GGUF stacks them per head, so they
    # land on the MultiLinear embed_q / unembed_out (-> KQuantMultiLinear), NOT a
    # standard kv_b_proj. All passthrough - the byte layout already matches.
    rk = d("deepseek2", "blk.0.attn_k_b.weight")
    assert rk.kind == MAP and rk.transform == "passthrough"
    assert rk.hf_name == "model.layers.0.self_attn.embed_q.weight"
    rv = d("deepseek2", "blk.0.attn_v_b.weight")
    assert rv.kind == MAP and rv.transform == "passthrough"
    assert rv.hf_name == "model.layers.0.self_attn.unembed_out.weight"
    assert d("deepseek2", "blk.0.attn_output.weight").hf_name == \
        "model.layers.0.self_attn.o_proj.weight"


def test_deepseek2_no_qk_permute():
    # MLA q/k come out of the latent up-projection already in the right layout;
    # llama.cpp's deepseek2 convert does not permute (unlike llama).
    for t in ("attn_q_b", "attn_k_b"):
        assert d("deepseek2", f"blk.0.{t}.weight").transform == "passthrough"


def test_deepseek2_moe_experts_shared_and_gate_bias():
    # Fine-grained routed experts -> switch_mlp; shared experts kept separate.
    assert d("deepseek2", "blk.3.ffn_gate_exps.weight").hf_name == \
        "model.layers.3.mlp.switch_mlp.gate_proj.weight"
    assert d("deepseek2", "blk.3.ffn_down_exps.weight").hf_name == \
        "model.layers.3.mlp.switch_mlp.down_proj.weight"
    assert d("deepseek2", "blk.3.ffn_up_shexp.weight").hf_name == \
        "model.layers.3.mlp.shared_experts.up_proj.weight"
    # router gate + its sigmoid bias-correction term (V3 noaux_tc routing).
    assert d("deepseek2", "blk.3.ffn_gate_inp.weight").hf_name == \
        "model.layers.3.mlp.gate.weight"
    rb = d("deepseek2", "blk.3.exp_probs_b.bias")
    assert rb.kind == MAP
    assert rb.hf_name == "model.layers.3.mlp.gate.e_score_correction_bias"


def test_deepseek2_leading_dense_block():
    # Layers below first_k_dense_replace are a plain dense MLP (no _exps suffix).
    assert d("deepseek2", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    assert d("deepseek2", "blk.0.ffn_down.weight").hf_name == \
        "model.layers.0.mlp.down_proj.weight"


# kimi-k3: hybrid KDA + nope-only MLA, latent MoE, residual-attention scores
def test_kimi_k3_mla_rows_match_deepseek2_targets():
    for t, hf in (("attn_q_a", "q_a_proj"), ("attn_q_a_norm", "q_a_layernorm"),
                  ("attn_q_b", "q_b_proj"),
                  ("attn_kv_a_mqa", "kv_a_proj_with_mqa"),
                  ("attn_kv_a_norm", "kv_a_layernorm"),
                  ("attn_k_b", "embed_q"), ("attn_v_b", "unembed_out"),
                  ("attn_output", "o_proj")):
        r = d("kimi-k3", f"blk.2.{t}.weight")
        assert r.kind == MAP and r.transform == "passthrough"
        assert r.hf_name == f"model.layers.2.self_attn.{hf}.weight"
    # K3-only MLA sigmoid output gate.
    assert d("kimi-k3", "blk.2.attn_gate.weight").hf_name == \
        "model.layers.2.self_attn.attn_gate.weight"


def test_kimi_k3_kda_rows():
    # In-projections (also the MLA no-q_lora fallback for attn_q) + decay path.
    for t, hf in (("attn_q", "q_proj"), ("attn_k", "k_proj"),
                  ("attn_v", "v_proj"), ("ssm_f_a", "f_a_proj"),
                  ("ssm_f_b", "f_b_proj"), ("ssm_beta", "b_proj"),
                  ("ssm_g", "g_proj")):
        r = d("kimi-k3", f"blk.0.{t}.weight")
        assert r.kind == MAP and r.transform == "passthrough"
        assert r.hf_name == f"model.layers.0.self_attn.{hf}.weight"
    assert d("kimi-k3", "blk.0.ssm_norm.weight").hf_name == \
        "model.layers.0.self_attn.o_norm.weight"
    # dt bias is a raw array on the module (no .weight).
    rdt = d("kimi-k3", "blk.0.ssm_dt.bias")
    assert rdt.kind == MAP and rdt.transform == "passthrough"
    assert rdt.hf_name == "model.layers.0.self_attn.dt_bias"


def test_kimi_k3_conv_weights_get_kda_conv_transform():
    for t, hf in (("ssm_conv1d_q", "q_conv"), ("ssm_conv1d_k", "k_conv"),
                  ("ssm_conv1d_v", "v_conv")):
        r = d("kimi-k3", f"blk.0.{t}.weight")
        assert r.kind == MAP and r.transform == "kda_conv_weight"
        assert r.hf_name == f"model.layers.0.self_attn.{hf}.conv.weight"


def test_kimi_k3_ssm_a_stays_folded_never_a_log():
    # ssm_a is the folded -exp(A_log); the canonical ssm_a_to_a_log transform
    # would take log of a negative (NaN). The override must claim it first
    # with a plain flatten onto a_folded.
    r = d("kimi-k3", "blk.0.ssm_a")
    assert r.kind == MAP and r.transform == "flatten"
    assert r.hf_name == "model.layers.0.self_attn.a_folded"


def test_kimi_k3_res_scores_are_raw_arrays():
    ra = d("kimi-k3", "blk.4.attn_res_score.weight")
    assert ra.kind == MAP and ra.hf_name == "model.layers.4.attn_res_score"
    rf = d("kimi-k3", "blk.4.ffn_res_score.weight")
    assert rf.kind == MAP and rf.hf_name == "model.layers.4.ffn_res_score"
    ro = d("kimi-k3", "output_res_score.weight")
    assert ro.kind == MAP and ro.hf_name == "model.output_res_score"


def test_kimi_k3_latent_moe_and_bias_on_moe_module():
    # Latent projections around the routed experts.
    assert d("kimi-k3", "blk.3.ffn_routed_down.weight").hf_name == \
        "model.layers.3.mlp.routed_down.weight"
    assert d("kimi-k3", "blk.3.ffn_routed_up.weight").hf_name == \
        "model.layers.3.mlp.routed_up.weight"
    assert d("kimi-k3", "blk.3.ffn_routed_norm.weight").hf_name == \
        "model.layers.3.mlp.routed_norm.weight"
    # Router on mlp.gate; the correction bias lives on the MoE module itself
    # (kimi_linear layout), NOT mlp.gate as in deepseek2.
    assert d("kimi-k3", "blk.3.ffn_gate_inp.weight").hf_name == \
        "model.layers.3.mlp.gate.weight"
    assert d("kimi-k3", "blk.3.exp_probs_b.bias").hf_name == \
        "model.layers.3.mlp.e_score_correction_bias"
    # Routed experts + fused shared experts.
    assert d("kimi-k3", "blk.3.ffn_up_exps.weight").hf_name == \
        "model.layers.3.mlp.switch_mlp.up_proj.weight"
    assert d("kimi-k3", "blk.3.ffn_gate_shexp.weight").hf_name == \
        "model.layers.3.mlp.shared_experts.gate_proj.weight"


def test_kimi_k3_norms_dense_mlp_and_globals():
    assert d("kimi-k3", "blk.0.attn_norm.weight").hf_name == \
        "model.layers.0.input_layernorm.weight"
    assert d("kimi-k3", "blk.0.ffn_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"
    assert d("kimi-k3", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    # Globals resolve via the canonical map.
    assert d("kimi-k3", "token_embd.weight").hf_name == \
        "model.embed_tokens.weight"
    assert d("kimi-k3", "output_norm.weight").hf_name == "model.norm.weight"
    assert d("kimi-k3", "output.weight").hf_name == "lm_head.weight"


def test_kimi_k3_kda_conv_weight_array_transform():
    # Both wire layouts reshape to mlx Conv1d (d_inner, d_conv, 1) exactly.
    import mlx.core as mx
    import numpy as np
    from gmlx.load.loader import remap_arrays
    d_inner, d_conv = 6, 4
    base = np.arange(d_inner * d_conv, dtype=np.float32)
    arrays = {
        # bf16-style 4D numpy layout (1, d_inner, 1, d_conv).
        "blk.0.ssm_conv1d_q.weight":
            mx.array(base.reshape(1, d_inner, 1, d_conv)),
        # quantized 3D layout (d_inner, 1, d_conv).
        "blk.0.ssm_conv1d_k.weight":
            mx.array(base.reshape(d_inner, 1, d_conv)),
        # folded -exp(A_log), padded [1, n_head]: flatten, no log (NaN trap).
        "blk.0.ssm_a": mx.array([[-0.5, -1.5]]),
    }
    hf, _meta, stats = remap_arrays(arrays, {}, "kimi-k3")
    q = hf["model.layers.0.self_attn.q_conv.conv.weight"]
    k = hf["model.layers.0.self_attn.k_conv.conv.weight"]
    assert q.shape == (d_inner, d_conv, 1) and k.shape == (d_inner, d_conv, 1)
    assert np.array_equal(np.array(q), base.reshape(d_inner, d_conv, 1))
    assert np.array_equal(np.array(q), np.array(k))
    a = hf["model.layers.0.self_attn.a_folded"]
    assert a.shape == (2,)
    assert np.array_equal(np.array(a), [-0.5, -1.5])
    assert stats["kda_conv_weight"] == 2


# glm4moe: MHA + V3-style fine-grained MoE (router bias, experts, shexp)
def test_glm4moe_norms_and_attention_via_canonical():
    # glm4_moe's two per-layer norms resolve via the canonical map (no glm4-dense
    # 4-norm remap): attn_norm -> input_layernorm, post_attention_norm ->
    # post_attention_layernorm. qk-norm + attention go canonical too.
    assert d("glm4moe", "blk.0.attn_norm.weight").hf_name == \
        "model.layers.0.input_layernorm.weight"
    assert d("glm4moe", "blk.0.post_attention_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"
    assert d("glm4moe", "blk.0.attn_q_norm.weight").hf_name == \
        "model.layers.0.self_attn.q_norm.weight"
    # q/k/v weights AND biases must land on distinct targets. GLM-4.5-Air has
    # q/k/v bias; the canonical path strips ".bias" and re-emits ".weight", so
    # without explicit bias overrides the bias collides onto {q,k,v}_proj.weight
    # and the quant matrix never loads (regression: 138 params silently dropped).
    for p in ("q", "k", "v"):
        rw = d("glm4moe", f"blk.0.attn_{p}.weight")
        rb = d("glm4moe", f"blk.0.attn_{p}.bias")
        assert rw.kind == MAP and rw.hf_name == \
            f"model.layers.0.self_attn.{p}_proj.weight"
        assert rb.kind == MAP and rb.hf_name == \
            f"model.layers.0.self_attn.{p}_proj.bias"
    # o_proj has no bias in GLM-4.5/4.6.
    assert d("glm4moe", "blk.0.attn_output.weight").hf_name == \
        "model.layers.0.self_attn.o_proj.weight"


def test_glm4moe_moe_experts_shared_and_gate_bias():
    assert d("glm4moe", "blk.1.ffn_gate_exps.weight").hf_name == \
        "model.layers.1.mlp.switch_mlp.gate_proj.weight"
    assert d("glm4moe", "blk.1.ffn_down_exps.weight").hf_name == \
        "model.layers.1.mlp.switch_mlp.down_proj.weight"
    assert d("glm4moe", "blk.1.ffn_up_shexp.weight").hf_name == \
        "model.layers.1.mlp.shared_experts.up_proj.weight"
    # router gate (canonical) + its sigmoid correction bias (override).
    assert d("glm4moe", "blk.1.ffn_gate_inp.weight").hf_name == \
        "model.layers.1.mlp.gate.weight"
    rb = d("glm4moe", "blk.1.exp_probs_b.bias")
    assert rb.kind == MAP
    assert rb.hf_name == "model.layers.1.mlp.gate.e_score_correction_bias"


def test_glm4moe_leading_dense_and_mtp_skip():
    # leading dense block uses plain MLP (no _exps); NextN/MTP tensors auto-skip
    # (canonical enum, no HF target).
    assert d("glm4moe", "blk.0.ffn_gate.weight").hf_name == \
        "model.layers.0.mlp.gate_proj.weight"
    assert d("glm4moe", "blk.46.nextn.eh_proj.weight").kind == SKIP
    assert d("glm4moe", "blk.46.nextn.embed_tokens.weight").kind == SKIP


# gpt-oss: attention sinks + router + per-expert MXFP4 + every-bias
def test_gpt_oss_norms_and_attention_via_canonical():
    # The two per-layer norms resolve via the canonical map: attn_norm ->
    # input_layernorm, post_attention_norm -> post_attention_layernorm.
    assert d("gpt-oss", "blk.0.attn_norm.weight").hf_name == \
        "model.layers.0.input_layernorm.weight"
    assert d("gpt-oss", "blk.0.post_attention_norm.weight").hf_name == \
        "model.layers.0.post_attention_layernorm.weight"
    # q/k/v/o weights AND biases land on distinct targets - gpt-oss has a bias
    # on all four projections (incl. o_proj, unlike qwen2/glm). The canonical
    # path strips ".bias" and re-emits ".weight", so each bias must be claimed
    # explicitly or it collides onto the (quant) weight slot.
    for p in ("q", "k", "v"):
        rw = d("gpt-oss", f"blk.0.attn_{p}.weight")
        rb = d("gpt-oss", f"blk.0.attn_{p}.bias")
        assert rw.kind == MAP and rw.transform == "passthrough"
        assert rw.hf_name == f"model.layers.0.self_attn.{p}_proj.weight"
        assert rb.kind == MAP and rb.hf_name == \
            f"model.layers.0.self_attn.{p}_proj.bias"
    # o_proj weight + bias (gpt-oss is the one arch here with an o_proj bias).
    assert d("gpt-oss", "blk.0.attn_output.weight").hf_name == \
        "model.layers.0.self_attn.o_proj.weight"
    assert d("gpt-oss", "blk.0.attn_output.bias").hf_name == \
        "model.layers.0.self_attn.o_proj.bias"
    # NeoX rope -> no qk_permute (mlx_lm builds rope traditional=False).
    assert d("gpt-oss", "blk.0.attn_q.weight").transform == "passthrough"


def test_gpt_oss_attention_sinks_no_weight_suffix():
    # Per-head learned sinks are a raw array on the module (self.sinks), NOT a
    # sub-module, so the HF target has no ".weight". No canonical enum carries
    # the GGUF `attn_sinks` name, so mapping requires the override.
    r = d("gpt-oss", "blk.7.attn_sinks.weight")
    assert r.kind == MAP and r.bid == 7 and r.transform == "passthrough"
    assert r.hf_name == "model.layers.7.self_attn.sinks"


def test_gpt_oss_router_weight_and_bias():
    # gpt-oss names the MoE router `mlp.router` (mlx_lm), not the canonical
    # `mlp.gate` that FFN_GATE_INP would map to; both weight and bias are pinned.
    assert d("gpt-oss", "blk.0.ffn_gate_inp.weight").hf_name == \
        "model.layers.0.mlp.router.weight"
    assert d("gpt-oss", "blk.0.ffn_gate_inp.bias").hf_name == \
        "model.layers.0.mlp.router.bias"


def test_gpt_oss_expert_weights_canonical_and_biases_distinct():
    # Routed-expert *weights* (the MXFP4 tensors) resolve via the canonical map
    # onto mlx_lm's SwitchGLU (mlp.experts.*); the native-fp repack handles the
    # codec, not the remap. Each expert *bias* is claimed explicitly so it lands
    # on its own ".bias" slot rather than colliding onto the packed weight.
    for proj in ("gate", "up", "down"):
        rw = d("gpt-oss", f"blk.2.ffn_{proj}_exps.weight")
        rb = d("gpt-oss", f"blk.2.ffn_{proj}_exps.bias")
        assert rw.kind == MAP and rw.transform == "passthrough"
        assert rw.hf_name == f"model.layers.2.mlp.experts.{proj}_proj.weight"
        assert rb.kind == MAP and rb.hf_name == \
            f"model.layers.2.mlp.experts.{proj}_proj.bias"


def test_gpt_oss_global_tensors():
    assert d("gpt-oss", "token_embd.weight").hf_name == "model.embed_tokens.weight"
    assert d("gpt-oss", "output_norm.weight").hf_name == "model.norm.weight"
    # Untied (output.weight present in this GGUF).
    assert d("gpt-oss", "output.weight").hf_name == "lm_head.weight"


# nemotron_h_moe: backbone prefix + ssm A_log + permuted attn
def test_nemotron_backbone_prefix_and_ssm():
    assert d("nemotron_h_moe", "token_embd.weight").hf_name == "backbone.embeddings.weight"
    ra = d("nemotron_h_moe", "blk.0.ssm_a")
    assert ra.kind == MAP and ra.transform == "ssm_a_to_a_log"
    assert ra.hf_name == "backbone.layers.0.mixer.A_log"
    rq = d("nemotron_h_moe", "blk.0.attn_q.weight")
    assert rq.transform == "qk_permute"
    assert rq.hf_name == "backbone.layers.0.mixer.q_proj.weight"


# universal edges: VLM skip, .scale skip, rope skip, unknown
def test_vlm_tower_tensors_skip():
    assert d("llama", "v.blk.0.attn_q.weight").kind == SKIP
    assert d("qwen2", "mm.0.weight").kind == SKIP


def test_unsloth_ud_scale_skipped_when_not_claimed():
    # On qwen2 nothing claims a bare .scale -> universal Unsloth-UD skip.
    assert d("qwen2", "blk.0.ffn_gate.scale").kind == SKIP


def test_rope_freqs_skipped():
    assert d("llama", "rope_freqs.weight").kind == SKIP


def test_unknown_arch_skips_not_fails():
    # No ARCH_ALIAS entry -> skip with an actionable reason (not a hard fail).
    r = d("totally-made-up-arch", "blk.0.attn_q.weight")
    assert r.kind == SKIP


def test_unknown_tensor_on_known_arch_hard_fails():
    r = d("llama", "blk.0.this_is_not_a_real_tensor.weight")
    assert r.kind == FAIL


# muse-glimmer: sandwich norms with the +1 already baked (consumed as-is),
# an attention output gate, and interleaved Q/K consumed without a permute.
def test_muse_glimmer_ffn_norm_pins_to_pre_feedforward():
    # FFN_NORM/FFN_PRE_NORM collide in TENSOR_NAMES; muse-glimmer's ffn_norm
    # is the PRE-FFN norm, same as gemma2/gemma3.
    r = d("muse-glimmer", "blk.1.ffn_norm.weight")
    assert r.kind == MAP
    assert r.hf_name == "model.layers.1.pre_feedforward_layernorm.weight"


def test_muse_glimmer_norms_are_not_unbaked():
    # The converter folds the +1 into the four per-layer norms and a plain
    # nn.RMSNorm consumes them directly - no gemma-style unbake.
    for t in ("attn_norm", "ffn_norm", "post_attention_norm", "post_ffw_norm"):
        assert d("muse-glimmer", f"blk.0.{t}.weight").transform == "passthrough"
    assert d("muse-glimmer", "output_norm.weight").transform == "passthrough"


def test_muse_glimmer_post_norms_land_on_the_sandwich_slots():
    assert (d("muse-glimmer", "blk.2.post_attention_norm.weight").hf_name
            == "model.layers.2.post_attention_layernorm.weight")
    assert (d("muse-glimmer", "blk.2.post_ffw_norm.weight").hf_name
            == "model.layers.2.post_feedforward_layernorm.weight")


def test_muse_glimmer_attn_gate_claims_the_gate_projection():
    # CANONICAL_HF["ATTN_GATE"] points at qwen3.5's linear_attn.in_proj_z, so
    # muse-glimmer has to claim this row first.
    r = d("muse-glimmer", "blk.0.attn_gate.weight")
    assert r.kind == MAP
    assert r.hf_name == "model.layers.0.self_attn.gate_proj.weight"
    assert "linear_attn" not in r.hf_name


def test_muse_glimmer_qk_are_not_permuted():
    # LLAMA_ROPE_TYPE_NORM: the converter already un-permuted HF's rotate_half
    # into interleaved, and the model class ropes with traditional=True. A
    # qk_permute here would double-apply it (and force _own() copies).
    for t in ("attn_q", "attn_k"):
        r = d("muse-glimmer", f"blk.3.{t}.weight")
        assert r.kind == MAP and r.bid == 3
        assert r.transform == "passthrough"


def test_muse_glimmer_qk_norms_and_projections_resolve_canonically():
    for gguf, hf in (
        ("blk.0.attn_q_norm.weight", "model.layers.0.self_attn.q_norm.weight"),
        ("blk.0.attn_k_norm.weight", "model.layers.0.self_attn.k_norm.weight"),
        ("blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight"),
        ("blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight"),
        ("blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight"),
        ("blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight"),
        ("blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight"),
    ):
        r = d("muse-glimmer", gguf)
        assert r.kind == MAP and r.hf_name == hf, gguf


def test_muse_glimmer_globals_resolve():
    assert d("muse-glimmer", "token_embd.weight").hf_name == "model.embed_tokens.weight"
    assert d("muse-glimmer", "output_norm.weight").hf_name == "model.norm.weight"
    assert d("muse-glimmer", "output.weight").hf_name == "lm_head.weight"
