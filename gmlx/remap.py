"""GGUF tensor name -> HF tensor name remap (pure string/decision logic).

A single arch-aware remapping shared by the load and dequant paths, so the
rule tables aren't duplicated per consumer.

This module is intentionally numpy/mlx-free: it only inspects names and
decides what to do. Callers own the array manipulation (dequant, layout
transforms, wire-byte splits, etc.).
"""

from __future__ import annotations

import re


# Naming policy / why each lookup layer exists:
#
# - GGUF stores tensors with names like `blk.0.attn_q.weight`. HF/MLX checkpoints
#   want `model.layers.0.self_attn.q_proj.weight`. Remap is per-arch.
# - `gguf-py.constants.TENSOR_NAMES` provides the canonical GGUF->enum direction
#   for most tensors. We invert it in `_gguf_to_enum()`. Then `CANONICAL_HF`
#   maps the enum to the HF target template.
# - Some archs (gemma-4-MoE) use names that diverge from HF stock or have
#   `.scale` tensors that the universal Unsloth-UD `.scale`-skip would drop.
#   `ARCH_PRIORITY_OVERRIDES` runs first so an arch can claim those before
#   the universal skip / canonical lookup.
# - Some archs (gemma-4 dual-FFN norms) have tensors not in upstream
#   gguf-py templates - `EXTRA_OVERRIDES` is the per-arch fallback.
# - Tensors not mapped by any TensorNameMap entry - hard-fail (genuine unknown).


# GGUF arch string -> MODEL_ARCH enum used for TENSOR_NAMES lookup.
# Older gguf-py releases have no GEMMA4 enum yet; gemma-4 GGUFs advertise
# general.architecture='gemma4' but reuse GEMMA3's tensor templates plus a few
# extra dual-FFN norms (handled via EXTRA_OVERRIDES below).
ARCH_ALIAS = {
    "gemma4": "GEMMA3",
    "gemma3": "GEMMA3",
    # EmbeddingGemma: gemma3 backbone; dense head (dense_2/3) in EXTRA_OVERRIDES.
    "gemma-embedding": "GEMMA3",
    "gemma3n": "GEMMA3N",
    "gemma2": "GEMMA2",
    "gemma": "GEMMA",
    "phi3": "PHI3",
    "glm4": "GLM4",
    "qwen2": "QWEN2",
    "qwen2moe": "QWEN2MOE",
    "qwen3": "QWEN3",
    "qwen3moe": "QWEN3MOE",
    # Qwen3-VL / Qwen3-Omni text tower: identical tensor layout to Qwen3-MoE
    # (qk-norm + stacked routed experts), so it reuses the QWEN3MOE templates +
    # overrides verbatim. The vision/audio towers live in a separate mmproj.
    "qwen3vlmoe": "QWEN3MOE",
    "qwen35": "QWEN35",
    "qwen35moe": "QWEN35MOE",
    "llama": "LLAMA",
    # llama.cpp's 'mistral3' arch (Ministral-3 / Mistral-Small-3.1) uses the
    # canonical Llama tensor layout (q/k permuted at convert time, ffn_norm
    # = post_attention_layernorm). The LLAMA arch overrides + qk_permute
    # transform apply unchanged.
    "mistral3": "LLAMA",
    # SmolLM3 is a Llama subclass (same tensor layout) with NoPE on selected
    # layers. NORM rope => q/k permuted at convert, so it reuses the LLAMA alias
    # verbatim (qk_permute + ffn_norm pin); NoPE is a model-side no-op needing no
    # tensor remap. The inert Mixtral expert entries never fire (dense model).
    "smollm3": "LLAMA",
    # IBM Granite 3.x/4.x dense: Llama tensor layout with four runtime scalar
    # multipliers (config-only, not folded into weights). NORM rope => qk_permute,
    # so it reuses the LLAMA alias verbatim.
    "granite": "LLAMA",
    "nemotron_h_moe": "NEMOTRON_H_MOE",
    "deepseek2": "DEEPSEEK2",
    # GLM-5.2 (llama.cpp arch 'glm-dsa') is the DeepSeek-V3.2 layout: MLA + the
    # same fine-grained MoE as deepseek2, so it reuses the DEEPSEEK2 tensor map
    # verbatim. Its DSA "lightning indexer" adds per-layer indexer.* tensors,
    # mapped by the indexer block appended to DEEPSEEK2 (inert for plain deepseek2
    # GGUFs, which carry no such tensors). Routes to model_type glm_moe_dsa.
    "glm-dsa": "DEEPSEEK2",
    # DeepSeek V4 Flash (dwarfstar/antirez 'deepseek4' arch; not llama.cpp).
    # Nothing shares the layout: low-rank q + single shared KV latent + grouped
    # low-rank output proj (wq_a/wkv/wo_a/wo_b under `attn.*`), per-layer
    # compressor + lightning-indexer tensors, hyper-connection params (raw
    # arrays, no .weight), MoE under `ffn.*` with hash-router tid2eid tables.
    # Complete override block; targets follow gmlx.deepseek_v4_model
    # (vendored mlx-lm PR #1192). NEOX-style tail rope on q/kv after the
    # low-rank projections => everything passes through un-permuted.
    "deepseek4": "DEEPSEEK4",
    "glm4moe": "GLM4MOE",
    # OpenAI gpt-oss (llama.cpp arch 'gpt-oss' / LLM_ARCH_OPENAI_MOE): MoE with
    # attention sinks, sliding/full alternating attention, and MXFP4 experts.
    "gpt-oss": "GPT_OSS",
    # ByteDance Seed-OSS (llama.cpp 'seed_oss'): plain Llama-shaped dense with an
    # explicit head_dim. NEOX rope, so attn_q/attn_k pass through (no qk_permute);
    # the only divergence from canonical is the FFN_NORM collision pin below.
    "seed_oss": "SEED_OSS",
    # Baidu ERNIE-4.5-MoE (llama.cpp 'ernie4_5-moe'): fine-grained MoE with a
    # shared expert behind leading dense layers. mlx-lm's ernie4_5_moe rope is
    # traditional=True (HF-native interleaved), so attn_q/attn_k pass through the
    # canonical map un-permuted despite llama.cpp tagging the arch NORM - see the
    # ERNIE4_5_MOE override block for the routing + dropped correction bias.
    "ernie4_5-moe": "ERNIE4_5_MOE",
    # MiniMax-M2 (llama.cpp 'minimax-m2'): every-layer fine-grained MoE (no dense
    # layers, no shared expert) with full attention + full-width qk-norm and
    # partial rotary. NEOX rope => Q/K pass through (no qk_permute). The router +
    # experts + sigmoid correction bias live under `block_sparse_moe.*`, not the
    # canonical `mlp.*` - see the MINIMAX override block.
    "minimax-m2": "MINIMAX",
    # MiniMax-M3 (llama.cpp 'minimax-m3'): the M2 layout plus leading dense
    # layers (canonical mlp.*), a per-layer shared expert (ffn_*_shexp patterns
    # appended to the shared MINIMAX block - inert for M2, which has none), and
    # per-head qk-norm (same canonical q_norm/k_norm targets, [head_dim] shape).
    # NEOX rope => Q/K pass through. Every *norm.weight is gemma-+1-baked at
    # conversion => the arch is in _GEMMA_NORM_BAKED_ARCHS (unbaked on load).
    "minimax-m3": "MINIMAX",
    # Tencent Hunyuan-A13B (llama.cpp 'hunyuan-moe'): softmax-gated fine-grained
    # MoE + a per-layer shared expert, per-head qk-norm, NTK-alpha rope. NEOX rope
    # => Q/K pass through. mlx-lm names the qk-norms query_layernorm/key_layernorm
    # and the router mlp.gate.wg, and the shared expert mlp.shared_mlp - none of
    # which match the canonical targets, so see the HUNYUAN_MOE override block.
    "hunyuan-moe": "HUNYUAN_MOE",
    # Tencent Hy3 299B-A21B (llama.cpp PR #25395 'hy_v3' - the underscore
    # spelling; early community conversions used 'hy-v3' and are not mapped).
    # Sigmoid-gated fine-grained MoE + selection-only expert bias + one ungated
    # shared expert behind a single leading dense layer, per-head qk-norm on
    # the canonical q_norm/k_norm names, NEOX rope => Q/K pass through. One
    # NextN/MTP block past the trunk (stripped by sanitize; drafted separately).
    "hy_v3": "HY_V3",
    # IBM Granite 4.x hybrid (llama.cpp 'granitehybrid'): alternating Mamba2 +
    # attention layers, each followed by softmax MoE + a fused-input shared MLP
    # (or a dense MLP on non-MoE variants), plus the granite runtime multipliers.
    # NORM rope + mlx-lm traditional=False => qk_permute on the attention layers
    # (claimed explicitly - the canonical gate only fires on the LLAMA alias).
    # The loader pre-fuses ffn_{gate,up}_shexp -> ffn_gate_up_shexp for the
    # shared MLP's fused input_linear (see transforms.fuse_shexp_gate_up).
    "granitehybrid": "GRANITE_HYBRID",
    # TII Falcon-H1 (llama.cpp 'falcon-h1'): parallel attention + Mamba2 in
    # every layer (both tensor families on every blk, outputs summed), then a
    # dense MLP mlx-lm houses under `feed_forward.*` (not `mlp.*`). NEOX rope =>
    # Q/K pass through. The Falcon muP multiplier zoo is folded into the wire
    # weights at convert time by llama.cpp, so the synth pins every multiplier
    # neutral - see _synth_falcon_h1. See the FALCON_H1 override block for the
    # ffn/final-norm renames and the (suffix-less) ffn_norm pin.
    "falcon-h1": "FALCON_H1",
    # Qwen3-Next 80B-A3B (llama.cpp 'qwen3next'): gated-DeltaNet linear
    # attention on 3 of every 4 layers (the 4th is gated full attention - the
    # output gate rides fused inside attn_q, both sides), every layer a
    # 512-expert MoE + shared expert. NEOX rope => no qk_permute (do not copy
    # qwen35's handling - different packing: fused in_proj_qkvz/in_proj_ba vs
    # qwen3.5's split in_proj_qkv/in_proj_z/in_proj_a/in_proj_b). Both GDN
    # wire layouts load: legacy fused ssm_in -> in_proj_qkvz passthrough; the
    # newer split attn_qkv/attn_gate resolve canonically onto qwen3.5's
    # in_proj_qkv/in_proj_z names, which gdn_patches._patch_qwen3next_split_gdn
    # creates on the model (flagged by _synth_qwen3next gdn_split_layout).
    # V heads stay HF-GROUPED on the wire in both layouts (legacy is the raw
    # HF tensor; the split converter's de-interleave preserves group order), so
    # the qwen3.5 tiled-V patch must not fire - see _needs_tiled_v_patch.
    "qwen3next": "QWEN3NEXT",
    # DiffusionGemma (llama.cpp 'diffusion-gemma'): an encoder-decoder block-
    # diffusion model on the Gemma-4 MoE backbone. The decoder backbone uses the
    # exact Gemma-4 GGUF tensor names, but the mlx-vlm Model nests them under
    # `model.decoder.*` (vs gemma-4's `model.*`) and keeps the expert gate_up
    # fused (vs gemma-4's switch_glu split). It also adds three diffusion-only
    # tensors (the encoder per-layer scalar + the self-conditioning gated MLP).
    # None of that fits the shared GEMMA3 templates, so it gets its own complete,
    # fail-closed override table (see DIFFUSION_GEMMA below) - there is no
    # canonical fallback for this alias.
    "diffusion-gemma": "DIFFUSION_GEMMA",
}

# GGUF archs whose RMSNorm weights llama.cpp stores with +1 baked in, and whose
# mlx_lm model re-adds 1 at runtime (rms_norm(x, 1.0 + weight)). The
# `gemma_norm_minus_one` transform undoes the bake on load. gemma4 is excluded:
# it shares the GEMMA3 remap alias but its mlx_lm RMSNorm uses the weight as-is.
# gemma-embedding is included: its backbone is gemma3 and the mlx-embeddings
# encoder builds the same mlx_lm gemma3_text RMSNorm (the +1 form).
_GEMMA_NORM_BAKED_ARCHS = frozenset({"gemma", "gemma2", "gemma3", "gemma-embedding",
                                     # minimax-m3: llama.cpp's converter bakes
                                     # +1 into every *norm.weight (incl. the
                                     # per-head qk-norms); mlx-lm minimax_m3's
                                     # GemmaRMSNorm re-adds 1 at runtime.
                                     "minimax-m3"})


def _gemma_norm_transform(arch_string: str, hf_name: str | None,
                          default: str) -> str:
    """Force the gemma norm-unbake transform for any RMSNorm weight on a
    +1-baked gemma arch, regardless of which remap path produced it (the four
    gemma norms are split across ARCH_PRIORITY_OVERRIDES and TENSOR_NAMES). All
    norm targets - input/post/pre/post-ffn layernorms, q/k_norm, the final
    norm - end in ``norm.weight`` (``layernorm.weight`` does too)."""
    if (arch_string in _GEMMA_NORM_BAKED_ARCHS and hf_name is not None
            and hf_name.endswith("norm.weight")):
        return "gemma_norm_minus_one"
    return default

# HF stock name (with `model.` prefix and `{bid}` placeholder) per MODEL_TENSOR.
# Names follow the de-facto HF Llama-family layout. mlx_lm-specific renames
# (e.g., `experts.switch_glu.*`) are not applied here; the loader emits an
# HF-format checkpoint that downstream tooling (`mlx_lm.convert`,
# `mlx_vlm.utils.convert`) can transform if/when needed.
CANONICAL_HF = {
    "TOKEN_EMBD":           "model.embed_tokens.weight",
    "OUTPUT":               "lm_head.weight",
    "OUTPUT_NORM":          "model.norm.weight",
    "ROPE_FREQS":           None,  # never serialized; skip
    "ATTN_NORM":            "model.layers.{bid}.input_layernorm.weight",
    "ATTN_POST_NORM":       "model.layers.{bid}.post_attention_layernorm.weight",
    "ATTN_Q":               "model.layers.{bid}.self_attn.q_proj.weight",
    "ATTN_K":               "model.layers.{bid}.self_attn.k_proj.weight",
    "ATTN_V":               "model.layers.{bid}.self_attn.v_proj.weight",
    "ATTN_OUT":             "model.layers.{bid}.self_attn.o_proj.weight",
    "ATTN_Q_NORM":          "model.layers.{bid}.self_attn.q_norm.weight",
    "ATTN_K_NORM":          "model.layers.{bid}.self_attn.k_norm.weight",
    "FFN_NORM":             "model.layers.{bid}.post_attention_layernorm.weight",
    "FFN_PRE_NORM":         "model.layers.{bid}.pre_feedforward_layernorm.weight",
    "FFN_POST_NORM":        "model.layers.{bid}.post_feedforward_layernorm.weight",
    "FFN_GATE":             "model.layers.{bid}.mlp.gate_proj.weight",
    "FFN_UP":               "model.layers.{bid}.mlp.up_proj.weight",
    "FFN_DOWN":             "model.layers.{bid}.mlp.down_proj.weight",
    "FFN_GATE_INP":         "model.layers.{bid}.mlp.gate.weight",  # MoE router
    "FFN_GATE_EXP":         "model.layers.{bid}.mlp.experts.gate_proj.weight",
    "FFN_UP_EXP":           "model.layers.{bid}.mlp.experts.up_proj.weight",
    "FFN_DOWN_EXP":         "model.layers.{bid}.mlp.experts.down_proj.weight",
    "FFN_GATE_UP_EXP":      "model.layers.{bid}.mlp.experts.gate_up_proj.weight",  # fused (gemma-4)
    # Gemma-3/4 per-layer features:
    "PER_LAYER_TOKEN_EMBD": "model.embed_tokens_per_layer.weight",
    "PER_LAYER_MODEL_PROJ": "model.per_layer_model_projection.weight",
    "PER_LAYER_PROJ_NORM":  "model.per_layer_projection_norm.weight",
    "PER_LAYER_INP_GATE":   "model.layers.{bid}.per_layer_input_gate.weight",
    "PER_LAYER_PROJ":       "model.layers.{bid}.per_layer_projection.weight",
    "PER_LAYER_POST_NORM":  "model.layers.{bid}.post_per_layer_input_norm.weight",
    # Qwen3.5/3.6 hybrid Mamba+attn (linear_attn-housed) features:
    "ATTN_QKV":             "model.layers.{bid}.linear_attn.in_proj_qkv.weight",
    "ATTN_GATE":            "model.layers.{bid}.linear_attn.in_proj_z.weight",
    "SSM_IN":               "model.layers.{bid}.linear_attn.in_proj.weight",
    "SSM_A":                "model.layers.{bid}.linear_attn.A_log",
    "SSM_ALPHA":            "model.layers.{bid}.linear_attn.in_proj_a.weight",
    "SSM_BETA":             "model.layers.{bid}.linear_attn.in_proj_b.weight",
    "SSM_CONV1D":           "model.layers.{bid}.linear_attn.conv1d.weight",
    # GGUF SSM_DT is stored as a 1D bias ('blk.{N}.ssm_dt.bias', shape (48,)).
    # MLX names it `dt_bias`, not `dt_proj.weight` - it's a bias vector, not a projection.
    "SSM_DT":               "model.layers.{bid}.linear_attn.dt_bias",
    "SSM_NORM":             "model.layers.{bid}.linear_attn.norm.weight",
    "SSM_OUT":              "model.layers.{bid}.linear_attn.out_proj.weight",
}

# Per-arch overrides for tensors not in `gguf.constants.TENSOR_NAMES` -
# typically architecture quirks not yet upstreamed in gguf-py.
#
# Each entry is a regex matching the GGUF name (sans `.weight` suffix; see
# parse_gguf_name) -> HF format string with optional {bid} placeholder.
# Order matters: the first matching pattern wins.
EXTRA_OVERRIDES: dict[str, list[tuple[re.Pattern, str | None]]] = {
    # Empty by design. Tensors that gguf-py also ships native templates for
    # (e.g. Gemma-4's dual-FFN norms + layer_output_scale, EmbeddingGemma's
    # dense head) must go in ARCH_PRIORITY_OVERRIDES instead: the canonical
    # reverse-map matches the native template first and skips it (no
    # CANONICAL_HF target), short-circuiting before this after-canonical
    # override ever runs.
}

# Arch-priority overrides - checked before the universal `.scale`-skip and
# canonical TENSOR_NAMES lookup. Lets an arch (a) redirect a tensor to a name
# that diverges from the HF/Llama stock layout (e.g., gemma-4-MoE's
# `router.proj` vs HF-stock `mlp.gate`) and (b) keep architectural `.scale`
# tensors that the universal Unsloth-UD skip would otherwise drop.
#
# Entry: (pattern, hf_format_string, transform). transform=`passthrough` for
# pure renames; `moe_split_gate_up` for fused gate_up split; etc.
#
# For gemma-4-MoE specifically, MLX's mlx-vlm gemma-4 implementation uses:
#   model.layers.{N}.experts.switch_glu.{gate,up,down}_proj.weight
#   model.layers.{N}.router.proj.weight     (GGUF: blk.{N}.ffn_gate_inp.weight)
#   model.layers.{N}.router.scale           (GGUF: blk.{N}.ffn_gate_inp.scale)
#   model.layers.{N}.router.per_expert_scale  (GGUF: blk.{N}.ffn_down_exps.scale)
#
# Other `.scale` tensors in Unsloth UD GGUFs are recipe metadata and continue
# to be skipped by `_is_unsloth_ud_scale`.
#
# For LLAMA arch, FFN_NORM is shadowed by FFN_PRE_NORM in the global reverse-map
# (TENSOR_NAMES collision: both enums use `blk.{bid}.ffn_norm`). The override
# below claims `blk.{N}.ffn_norm.weight` first so Llama gets the right HF target
# (post_attention_layernorm); without it, Llama would get pre_feedforward_layernorm,
# which silently produces a misnamed checkpoint.
ARCH_PRIORITY_OVERRIDES: dict[str, list[tuple[re.Pattern, str | None, str]]] = {
    # FFN_NORM/FFN_PRE_NORM TENSOR_NAMES collision: both enums use
    # `blk.{bid}.ffn_norm`, so the global reverse-map's resolution depends on
    # gguf-py enum ordering. Both archs that consume `ffn_norm` get an
    # explicit override below so a gguf-py version bump can't silently flip
    # the mapping.
    "GEMMA3": [
        # Router (architectural .scale must be claimed before universal skip).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.router.proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.scale$"),
         "model.layers.{bid}.router.scale", "passthrough"),
        # Per-expert routing scale: Unsloth UD GGUFs ship this under the
        # `ffn_down_exps.scale` name (verified bit-exact against the upstream
        # `router.per_expert_scale` bf16 tensor). Must be claimed before the
        # universal `.scale` skip; without it the student loads with the
        # tensor missing and routing is silently mis-scaled.
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.scale$"),
         "model.layers.{bid}.router.per_expert_scale", "passthrough"),
        # MoE expert weights - namespace divergence: HF stock uses `mlp.experts.*`,
        # mlx-vlm gemma-4 uses `experts.switch_glu.*`. Override the canonical map
        # so --out-dir produces a directly mlx_lm.load-able checkpoint.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_up_exps\.weight$"),
         "model.layers.{bid}.experts.switch_glu.gate_up_proj.weight", "moe_split_gate_up"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.experts.switch_glu.down_proj.weight", "passthrough"),
        # FFN_PRE_NORM is what gemma-3/4 wants for `blk.{bid}.ffn_norm`. Pin it
        # explicitly so the resolution doesn't drift on a gguf-py upgrade.
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.pre_feedforward_layernorm.weight", "passthrough"),
        # Gemma-4 dual-FFN extra norms + per-layer output scale. gguf-py >=0.19
        # ships a native GEMMA4 arch whose templates (FFN_POST_NORM_1/2,
        # FFN_PRE_NORM_2, LAYER_OUT_SCALE) match these names in the canonical
        # reverse-map but have no CANONICAL_HF target, so they'd silently skip.
        # Claim them here (before the canonical lookup) so the mapping holds on
        # both the old gguf-py (no GEMMA4 enum) and new (>=0.19). The `.weight`
        # is optional because scalar tensors can arrive without it.
        (re.compile(r"^blk\.(\d+)\.post_ffw_norm_1(?:\.weight)?$"),
         "model.layers.{bid}.post_feedforward_layernorm_1.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_ffw_norm_2(?:\.weight)?$"),
         "model.layers.{bid}.post_feedforward_layernorm_2.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.pre_ffw_norm_2(?:\.weight)?$"),
         "model.layers.{bid}.pre_feedforward_layernorm_2.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.layer_output_scale(?:\.weight)?$"),
         "model.layers.{bid}.layer_scalar", "passthrough"),  # no .weight suffix in MLX
        # EmbeddingGemma dense head. gguf-py's GEMMA_EMBEDDING templates put
        # dense_2/dense_3 in the global reverse-map with no CANONICAL_HF target
        # (they would skip), so claim them here onto the mlx-embeddings Model's
        # dense = [Linear(h, 4h), Linear(4h, h)]. Only gemma-embedding GGUFs carry
        # them, so this is inert for plain gemma3/gemma4.
        (re.compile(r"^dense_2\.weight$"), "dense.0.weight", "passthrough"),
        (re.compile(r"^dense_3\.weight$"), "dense.1.weight", "passthrough"),
    ],
    # DiffusionGemma: a complete table (every backbone tensor + the diffusion-
    # only tensors), because the mlx-vlm Model homes the Gemma-4 backbone under
    # `model.decoder.*` - diverging from canonical on every name - and parse
    # fails closed for this arch (no canonical fallback). The decoder layer runs
    # a dense MLP (`mlp.*`) and routed experts (`experts.*` + `router.*`) in
    # parallel; the expert gate_up stays fused (passthrough, not moe_split).
    "DIFFUSION_GEMMA": [
        # --- globals ---
        (re.compile(r"^token_embd\.weight$"),
         "model.decoder.embed_tokens.weight", "passthrough"),
        (re.compile(r"^output_norm\.weight$"),
         "model.decoder.norm.weight", "passthrough"),
        # lm_head is tied to the decoder embedding; a stray output head is dropped.
        (re.compile(r"^output\.weight$"), None, "passthrough"),
        # self-conditioning gated MLP (decoder-only).
        (re.compile(r"^self_cond_pre_norm(?:\.weight)?$"),
         "model.decoder.self_conditioning.pre_norm.weight", "passthrough"),
        (re.compile(r"^self_cond_gate\.weight$"),
         "model.decoder.self_conditioning.gate_proj.weight", "passthrough"),
        (re.compile(r"^self_cond_up\.weight$"),
         "model.decoder.self_conditioning.up_proj.weight", "passthrough"),
        (re.compile(r"^self_cond_down\.weight$"),
         "model.decoder.self_conditioning.down_proj.weight", "passthrough"),
        # --- attention ---
        (re.compile(r"^blk\.(\d+)\.attn_q\.weight$"),
         "model.decoder.layers.{bid}.self_attn.q_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.weight$"),
         "model.decoder.layers.{bid}.self_attn.k_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.weight$"),
         "model.decoder.layers.{bid}.self_attn.v_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_output\.weight$"),
         "model.decoder.layers.{bid}.self_attn.o_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_q_norm\.weight$"),
         "model.decoder.layers.{bid}.self_attn.q_norm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k_norm\.weight$"),
         "model.decoder.layers.{bid}.self_attn.k_norm.weight", "passthrough"),
        # --- norms (the Gemma-4 dual-FFN norm zoo) ---
        (re.compile(r"^blk\.(\d+)\.attn_norm\.weight$"),
         "model.decoder.layers.{bid}.input_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_attention_norm\.weight$"),
         "model.decoder.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.decoder.layers.{bid}.pre_feedforward_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.pre_ffw_norm_2\.weight$"),
         "model.decoder.layers.{bid}.pre_feedforward_layernorm_2.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_ffw_norm\.weight$"),
         "model.decoder.layers.{bid}.post_feedforward_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_ffw_norm_1\.weight$"),
         "model.decoder.layers.{bid}.post_feedforward_layernorm_1.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_ffw_norm_2\.weight$"),
         "model.decoder.layers.{bid}.post_feedforward_layernorm_2.weight", "passthrough"),
        # --- dense MLP (the shared, always-on branch) ---
        (re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$"),
         "model.decoder.layers.{bid}.mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
         "model.decoder.layers.{bid}.mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down\.weight$"),
         "model.decoder.layers.{bid}.mlp.down_proj.weight", "passthrough"),
        # --- routed experts (gate_up stays fused) + router ---
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.decoder.layers.{bid}.router.proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.scale$"),
         "model.decoder.layers.{bid}.router.scale", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.scale$"),
         "model.decoder.layers.{bid}.router.per_expert_scale", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_up_exps\.weight$"),
         "model.decoder.layers.{bid}.experts.gate_up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.decoder.layers.{bid}.experts.down_proj.weight", "passthrough"),
        # --- per-layer output scalars (decoder + the tied encoder's own) ---
        (re.compile(r"^blk\.(\d+)\.layer_output_scale(?:\.weight)?$"),
         "model.decoder.layers.{bid}.layer_scalar", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.enc_layer_output_scale(?:\.weight)?$"),
         "model.encoder.language_model.layers.{bid}.layer_scalar", "passthrough"),
    ],
    "LLAMA": [
        # FFN_NORM (post_attention_layernorm) - Llama has no separate pre-FFN norm;
        # the post-attn norm is the pre-FFN norm.
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # Mixtral (llama-arch sparse MoE): llama.cpp emits Mixtral under
        # general.architecture='llama' with an expert count, and mlx-lm models it
        # as the `mixtral` model_type (block_sparse_moe + SwitchGLU). After the
        # loader coalesces the legacy per-expert split weights into the stacked
        # `_exps` form, route the experts + router here. The attn (q/k permuted),
        # norms, and globals stay on the shared LLAMA/canonical path. All inert on
        # dense Llama, which carries none of these tensors.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.block_sparse_moe.gate.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.down_proj.weight", "passthrough"),
    ],
    "GEMMA": [
        # Gemma-1 has only input + post-attention norms (like Llama), so ffn_norm
        # is post_attention_layernorm - not the pre_feedforward norm gemma-2/3 use.
        # Pin past the FFN_NORM/FFN_PRE_NORM collision. The gemma_norm_minus_one
        # bake-undo applies to all gemma-1 norms (gemma is in _GEMMA_NORM_BAKED_ARCHS).
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
    ],
    "QWEN3": [
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
    ],
    "SEED_OSS": [
        # Seed-OSS has only input + post-attention norms (Llama-shaped), so
        # ffn_norm is post_attention_layernorm. Pin past the FFN_NORM/
        # FFN_PRE_NORM collision; everything else resolves via the canonical map
        # (NEOX rope => attn_q/attn_k pass through, no qk_permute).
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # Seed-OSS-36B ships q/k/v attention biases (F32). The canonical path
        # strips ".bias" to match the enum and re-emits the ".weight" target,
        # which would overwrite the quant weight slot - claim them explicitly
        # with their ".bias" HF names (the qwen2/glm4 precedent).
        (re.compile(r"^blk\.(\d+)\.attn_q\.bias$"),
         "model.layers.{bid}.self_attn.q_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.bias$"),
         "model.layers.{bid}.self_attn.k_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.bias$"),
         "model.layers.{bid}.self_attn.v_proj.bias", "passthrough"),
    ],
    "ERNIE4_5_MOE": [
        # Baidu ERNIE-4.5-MoE: shared-expert fine-grained MoE with leading dense
        # layers. mlx-lm's ernie4_5_moe uses traditional=True rope, which consumes
        # the GGUF's HF-native (un-permuted) Q/K directly - so attn_q/attn_k fall
        # through to the canonical PASSTHROUGH (not qk_permute), even though
        # llama.cpp tags the arch LLAMA_ROPE_TYPE_NORM. (qk_permute would be a
        # silent mis-attention here; the 16k parity gate is the check.)
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # Routed experts (already stacked in GGUF) -> SwitchGLU.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        # Shared expert.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.down_proj.weight", "passthrough"),
        # Router.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.gate.weight", "passthrough"),
        # Aux-free routing correction bias: mlx-lm's ernie4_5_moe drops
        # e_score_correction_bias and gates without it, so drop it on load (a
        # mapped-then-dropped name would also work, but skipping is cleaner).
        (re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$"), None, "passthrough"),
    ],
    "MINIMAX": [
        # MiniMax-M2: NEOX rope => attn_q/attn_k/attn_v + the full-width qk-norms
        # (attn_q_norm/attn_k_norm -> self_attn.q_norm/k_norm) resolve via the
        # canonical map as PASSTHROUGH. mlx-lm's minimax nests the whole MoE under
        # `block_sparse_moe.*` (not the canonical `mlp.*`), so claim each tensor.
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # Router -> block_sparse_moe.gate (canonical FFN_GATE_INP would mis-target
        # mlp.gate).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.block_sparse_moe.gate.weight", "passthrough"),
        # Routed experts (already stacked) -> block_sparse_moe.switch_mlp.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.gate_proj.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.up_proj.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.down_proj.weight",
         "passthrough"),
        # Sigmoid-gating correction bias (no .weight; kept - minimax adds it to
        # the routing scores at runtime, unlike ERNIE which drops it).
        (re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$"),
         "model.layers.{bid}.block_sparse_moe.e_score_correction_bias",
         "passthrough"),
        # MiniMax-M3 shared expert (M2 GGUFs carry no *_shexp tensors, so these
        # are inert for M2 - same shared-block pattern as glm-dsa's indexer
        # entries in DEEPSEEK2). mlx-lm minimax_m3 nests it as a plain MLP under
        # block_sparse_moe.shared_experts. M3's leading dense layers resolve via
        # the canonical map (ffn_gate/up/down -> mlp.*).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.block_sparse_moe.shared_experts.gate_proj.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.block_sparse_moe.shared_experts.up_proj.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.block_sparse_moe.shared_experts.down_proj.weight",
         "passthrough"),
        # MiniMax-M3 MSA indexer (llama.cpp PR #24908) - one q projection per
        # GQA group + a single shared k head, each per-head gemma-normed. M2
        # GGUFs carry none, so these are inert for M2. Two GGUF spellings:
        # the PR's `indexer.*` namespace, and the pre-rename community
        # spelling `index_*` (avar6-era conversions). The norm targets end in
        # `norm.weight`, so the gemma +1 unbake fires on them like every
        # other minimax-m3 norm.
        (re.compile(r"^blk\.(\d+)\.indexer\.q_proj\.weight$"),
         "model.layers.{bid}.self_attn.index_q_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer\.k_proj\.weight$"),
         "model.layers.{bid}.self_attn.index_k_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer\.q_norm\.weight$"),
         "model.layers.{bid}.self_attn.index_q_norm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer\.k_norm\.weight$"),
         "model.layers.{bid}.self_attn.index_k_norm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.index_q_proj\.weight$"),
         "model.layers.{bid}.self_attn.index_q_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.index_k_proj\.weight$"),
         "model.layers.{bid}.self_attn.index_k_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.index_q_norm\.weight$"),
         "model.layers.{bid}.self_attn.index_q_norm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.index_k_norm\.weight$"),
         "model.layers.{bid}.self_attn.index_k_norm.weight", "passthrough"),
    ],
    "HUNYUAN_MOE": [
        # Hunyuan-A13B: NEOX rope => attn_q/attn_k/attn_v/attn_output pass through
        # via the canonical map. The qk-norms are PER-HEAD (RMSNorm over head_dim)
        # and mlx-lm names them query_layernorm/key_layernorm (not q_norm/k_norm),
        # so claim them explicitly.
        (re.compile(r"^blk\.(\d+)\.attn_q_norm\.weight$"),
         "model.layers.{bid}.self_attn.query_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k_norm\.weight$"),
         "model.layers.{bid}.self_attn.key_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # Router: mlx-lm's hunyuan wraps the gate Linear in a Gate module (.wg),
        # so the target is mlp.gate.wg (canonical FFN_GATE_INP -> mlp.gate is wrong).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.gate.wg.weight", "passthrough"),
        # Routed experts (already stacked) -> switch_mlp.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        # Per-layer shared expert -> mlp.shared_mlp (a plain MLP, not a SwitchGLU).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_mlp.down_proj.weight", "passthrough"),
    ],
    "HY_V3": [
        # Tencent Hy3 (llama.cpp PR #25395 'hy_v3'): NEOX rope => attn_q/attn_k
        # pass through via the canonical map, and the per-head qk-norms land on
        # the canonical q_norm/k_norm targets (hy_v3 keeps mlx-lm's stock
        # names, unlike hunyuan-moe). The NextN/MTP block's nextn.* tensors map
        # to canonical enums with no HF target -> auto-SKIP; its standard
        # decoder tensors land on model.layers.{num_hidden_layers}.* and the
        # vendored hy_v3.sanitize strips them (the MTP drafter loads them
        # separately). Only the MoE family needs explicit targets.
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # Router: hy_v3 wraps the gate Linear in a MoEGate module (mlp.router).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.router.gate.weight", "passthrough"),
        # Selection-only expert bias (F32, sigmoid routing). Hy3 GGUFs store it
        # SUFFIX-LESS (blk.N.exp_probs_b, unlike deepseek/glm4moe's .bias form,
        # for compatibility with the first published files); accept both.
        (re.compile(r"^blk\.(\d+)\.exp_probs_b(?:\.bias)?$"),
         "model.layers.{bid}.mlp.router.expert_bias", "passthrough"),
        # Routed experts (already stacked) -> switch_mlp (SwitchGLU).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        # Ungated shared expert -> mlp.shared_mlp (a plain MLP).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_mlp.down_proj.weight", "passthrough"),
    ],
    "GRANITE_HYBRID": [
        # Granite 4.x hybrid (granitemoehybrid): alternating Mamba2 + attention
        # layers (a layer is recurrent iff its head_count_kv entry is 0), each
        # followed by softmax MoE + a shared MLP - or a plain dense MLP on
        # non-MoE variants (those resolve via the canonical map). attn_norm ->
        # input_layernorm and attn_v/attn_output also resolve canonically.
        #
        # NORM rope + mlx-lm traditional=False => the attention layers' Q/K were
        # permuted at convert and must be un-permuted (explicit qk_permute -
        # the canonical gate only fires on the LLAMA alias).
        (re.compile(r"^blk\.(\d+)\.attn_q\.weight$"),
         "model.layers.{bid}.self_attn.q_proj.weight", "qk_permute"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.weight$"),
         "model.layers.{bid}.self_attn.k_proj.weight", "qk_permute"),
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # Mamba2 mixer (mlx-lm module name `mamba`)
        (re.compile(r"^blk\.(\d+)\.ssm_in\.weight$"),
         "model.layers.{bid}.mamba.in_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_conv1d\.weight$"),
         "model.layers.{bid}.mamba.conv1d.weight", "conv1d_unsqueeze"),
        (re.compile(r"^blk\.(\d+)\.ssm_conv1d\.bias$"),
         "model.layers.{bid}.mamba.conv1d.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_dt\.bias$"),
         "model.layers.{bid}.mamba.dt_bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_a$"),
         "model.layers.{bid}.mamba.A_log", "ssm_a_to_a_log"),
        (re.compile(r"^blk\.(\d+)\.ssm_d$"),
         "model.layers.{bid}.mamba.D", "flatten"),
        # GGUF stores the gated norm grouped 2-D [d_inner/n_groups, n_groups];
        # mlx-lm's RMSNormGated weight is flat [d_inner].
        (re.compile(r"^blk\.(\d+)\.ssm_norm\.weight$"),
         "model.layers.{bid}.mamba.norm.weight", "flatten"),
        (re.compile(r"^blk\.(\d+)\.ssm_out\.weight$"),
         "model.layers.{bid}.mamba.out_proj.weight", "passthrough"),
        # MoE: router wraps a Gate module (`router.layer`), experts stacked
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.block_sparse_moe.router.layer.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.gate_proj.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.up_proj.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.block_sparse_moe.switch_mlp.down_proj.weight",
         "passthrough"),
        # Shared MLP: mlx-lm wants the fused [gate; up] input_linear. The
        # loader pre-fuses ffn_{gate,up}_shexp into this name (gate rows first)
        # via transforms.fuse_shexp_gate_up before remap; the raw split names
        # must not reach here (if they do - fusion skipped - strict-load fails
        # loudly on the unfilled input_linear). ---
        (re.compile(r"^blk\.(\d+)\.ffn_gate_up_shexp\.weight$"),
         "model.layers.{bid}.shared_mlp.input_linear.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.shared_mlp.output_linear.weight", "passthrough"),
    ],
    "FALCON_H1": [
        # TII Falcon-H1: every layer carries both an attention block and a
        # Mamba2 mixer in parallel (input_layernorm feeds both; outputs are
        # summed), then a dense gated MLP. Attention (attn_q/k/v/output,
        # attn_norm) and the embeddings resolve canonically - NEOX rope, so no
        # permute. Everything below diverges from canonical:
        #
        # mlx-lm falcon_h1 names the MLP `feed_forward.*` (canonical targets
        # `mlp.*`), the pre-MLP norm `pre_ff_layernorm`, and the final norm
        # `model.final_layernorm`. llama.cpp writes ffn_norm / ssm_a / ssm_d
        # with no ".weight" suffix for this arch (see falcon-h1.cpp
        # load_arch_tensors), so the norm pin matches the bare name too.
        (re.compile(r"^blk\.(\d+)\.ffn_norm(?:\.weight)?$"),
         "model.layers.{bid}.pre_ff_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$"),
         "model.layers.{bid}.feed_forward.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
         "model.layers.{bid}.feed_forward.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down\.weight$"),
         "model.layers.{bid}.feed_forward.down_proj.weight", "passthrough"),
        (re.compile(r"^output_norm\.weight$"),
         "model.final_layernorm.weight", "passthrough"),
        # Mamba2 mixer (mlx-lm module name `mamba`) - same family/shapes
        # as GRANITE_HYBRID; in_proj split order verified identical
        # ([gate, conv_input, dt] on both sides). ---
        (re.compile(r"^blk\.(\d+)\.ssm_in\.weight$"),
         "model.layers.{bid}.mamba.in_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_conv1d\.weight$"),
         "model.layers.{bid}.mamba.conv1d.weight", "conv1d_unsqueeze"),
        (re.compile(r"^blk\.(\d+)\.ssm_conv1d\.bias$"),
         "model.layers.{bid}.mamba.conv1d.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_dt\.bias$"),
         "model.layers.{bid}.mamba.dt_bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_a$"),
         "model.layers.{bid}.mamba.A_log", "ssm_a_to_a_log"),
        (re.compile(r"^blk\.(\d+)\.ssm_d$"),
         "model.layers.{bid}.mamba.D", "flatten"),
        # GGUF stores the gated norm grouped 2-D [d_inner/n_groups, n_groups];
        # mlx-lm's FalconH1RMSNormGated weight is flat [d_inner].
        (re.compile(r"^blk\.(\d+)\.ssm_norm\.weight$"),
         "model.layers.{bid}.mamba.norm.weight", "flatten"),
        (re.compile(r"^blk\.(\d+)\.ssm_out\.weight$"),
         "model.layers.{bid}.mamba.out_proj.weight", "passthrough"),
    ],
    "QWEN3NEXT": [
        # Qwen3-Next: gated-DeltaNet layers + gated full-attention layers +
        # every-layer MoE w/ shared expert. Attention (attn_q with the fused
        # output gate, attn_k/v/output, q/k norms, attn_norm, attn_post_norm)
        # and the GDN's conv1d / dt_bias / A_log / norm / out_proj all resolve
        # canonically onto self_attn.* / linear_attn.* (NEOX => no permute; the
        # converter bakes +1 into every non-gated norm, which is exactly what
        # mlx-lm's runtime weights expect => passthrough). Claimed here:
        #
        # The newer split GDN input layout (current convert_hf_to_gguf
        # de-interleaves in_proj_qkvz into attn_qkv + attn_gate) needs no row
        # here: canonical ATTN_QKV/ATTN_GATE resolve to qwen3.5's
        # linear_attn.in_proj_qkv / in_proj_z names, and the loader's
        # _patch_qwen3next_split_gdn creates exactly those modules on the
        # qwen3_next GDN (gated by _synth_qwen3next's gdn_split_layout flag).
        # legacy fused GDN input projections - the raw HF tensors, per-k-head
        # interleaved [q,k,v,z] / [b,a]: exactly the layout mlx-lm's
        # fix_query_key_value_ordering consumes. (Canonical SSM_IN targets
        # qwen3.5's in_proj.weight - wrong module name here.)
        (re.compile(r"^blk\.(\d+)\.ssm_in\.weight$"),
         "model.layers.{bid}.linear_attn.in_proj_qkvz.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_ba\.weight$"),
         "model.layers.{bid}.linear_attn.in_proj_ba.weight", "passthrough"),
        # MoE: routed experts stored stacked -> mlp.switch_mlp.* (canonical
        # would target mlp.experts.*); shared expert + its 1-D sigmoid gate.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.down_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert_gate.weight", "gate_1d_unsqueeze"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.gate.weight", "passthrough"),
    ],
    "GLM4": [
        # GLM-4 has four norms/layer whose mlx_lm attribute names do not match the
        # canonical HF slots, so pin all four. mlx_lm glm4 calls them:
        #   input_layernorm (pre-attn)      <- attn_norm
        #   post_self_attn_layernorm        <- post_attention_norm  (canonical would
        #                                      mis-target post_attention_layernorm)
        #   post_attention_layernorm (pre-MLP) <- ffn_norm  (also the FFN_NORM/
        #                                      FFN_PRE_NORM collision; pin it)
        #   post_mlp_layernorm              <- post_ffw_norm (canonical would
        #                                      mis-target post_feedforward_layernorm)
        (re.compile(r"^blk\.(\d+)\.attn_norm\.weight$"),
         "model.layers.{bid}.input_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_attention_norm\.weight$"),
         "model.layers.{bid}.post_self_attn_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_ffw_norm\.weight$"),
         "model.layers.{bid}.post_mlp_layernorm.weight", "passthrough"),
        # Fused gate_up: GGUF ffn_up is the [gate; up] tensor (2*intermediate wide);
        # mlx_lm glm4 splits it in mlp.gate_up_proj (phi-3 precedent). Canonical
        # FFN_UP would mis-map to mlp.up_proj.
        (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
         "model.layers.{bid}.mlp.gate_up_proj.weight", "passthrough"),
        # QKV biases - GLM-4 has q/k/v bias (o_proj has none). The canonical path
        # strips ".bias" to match the enum then re-emits the ".weight" target, so
        # claim the three biases explicitly (as for qwen2). Q/K weights stay on
        # the canonical path: GLM uses interleaved rope natively (mlx_lm glm4 sets
        # rope traditional=True to match), so no qk_permute - verified by 16k
        # greedy-parity vs llama.cpp.
        (re.compile(r"^blk\.(\d+)\.attn_q\.bias$"),
         "model.layers.{bid}.self_attn.q_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.bias$"),
         "model.layers.{bid}.self_attn.k_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.bias$"),
         "model.layers.{bid}.self_attn.v_proj.bias", "passthrough"),
    ],
    "GEMMA2": [
        # gemma-2 has four norms/layer: attn_norm (input), post_attention_norm,
        # ffn_norm (== pre_feedforward), post_ffw_norm (post_feedforward). The
        # latter three resolve via the canonical map (ATTN_POST_NORM / FFN_POST_NORM),
        # but ffn_norm hits the FFN_NORM/FFN_PRE_NORM collision - pin it to
        # pre_feedforward_layernorm (gemma-2/3 want the pre-FFN norm). The
        # gemma_norm_minus_one bake-undo is applied to all four (gemma2 is in
        # _GEMMA_NORM_BAKED_ARCHS).
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.pre_feedforward_layernorm.weight", "passthrough"),
    ],
    # gemma-3n (E2B / E4B): a MatFormer text tower with AltUp (alternating
    # updates), LAuReL (low-rank augmented residual), per-layer input
    # embeddings, and KV-sharing. mlx_lm models the text tower as a standalone
    # `LanguageModel` (no `model.`/`language_model.` wrapper prefix), which the
    # loader builds directly - so every target here is the unprefixed
    # `LanguageModel` attribute path (`layers.{bid}.*`, `embed_tokens.weight`,
    # `norm.weight`, ...). The block therefore claims all gemma-3n tensors; none
    # fall through to the canonical (`model.`-prefixed) reverse-map. Norms are
    # used as-is: both ggml and mlx_lm apply a plain RMSNorm to the stored
    # weight, so there is no +1 bake to undo (gemma3n is not in
    # _GEMMA_NORM_BAKED_ARCHS). Q/K stay passthrough - gemma uses NEOX rope
    # (nn.RoPE traditional=False), so convert applies no permute.
    "GEMMA3N": [
        # globals (LanguageModel level)
        (re.compile(r"^token_embd\.weight$"),
         "embed_tokens.weight", "passthrough"),
        (re.compile(r"^output_norm\.weight$"),
         "norm.weight", "passthrough"),
        # Per-layer input embeddings + their projection/norm.
        (re.compile(r"^per_layer_token_embd\.weight$"),
         "embed_tokens_per_layer.weight", "passthrough"),
        (re.compile(r"^per_layer_model_proj\.weight$"),
         "per_layer_model_projection.weight", "passthrough"),
        (re.compile(r"^per_layer_proj_norm\.weight$"),
         "per_layer_projection_norm.weight", "passthrough"),
        # AltUp stream (un)projections: GGUF stores one stacked 3-D tensor
        # [n_embd, n_embd, altup_num_inputs-1]; mlx_lm wants a list of
        # (altup_num_inputs-1) separate Linears. Split along the stack axis into
        # `altup_projections.{i}.weight`.
        (re.compile(r"^altup_proj\.weight$"),
         "altup_projections.weight", "altup_split"),
        (re.compile(r"^altup_unembd_proj\.weight$"),
         "altup_unembed_projections.weight", "altup_split"),
        # per-layer attention
        (re.compile(r"^blk\.(\d+)\.attn_norm\.weight$"),
         "layers.{bid}.input_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_q\.weight$"),
         "layers.{bid}.self_attn.q_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.weight$"),
         "layers.{bid}.self_attn.k_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.weight$"),
         "layers.{bid}.self_attn.v_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_output\.weight$"),
         "layers.{bid}.self_attn.o_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_q_norm\.weight$"),
         "layers.{bid}.self_attn.q_norm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k_norm\.weight$"),
         "layers.{bid}.self_attn.k_norm.weight", "passthrough"),
        # per-layer MLP (separate gate/up/down) + the 4-norm sandwich
        (re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$"),
         "layers.{bid}.mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
         "layers.{bid}.mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down\.weight$"),
         "layers.{bid}.mlp.down_proj.weight", "passthrough"),
        # attn_norm=input, post_attention_norm=post-attn, ffn_norm=pre-FFN,
        # post_ffw_norm=post-FFN (the gemma 4-norm sandwich).
        (re.compile(r"^blk\.(\d+)\.post_attention_norm\.weight$"),
         "layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "layers.{bid}.pre_feedforward_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_ffw_norm\.weight$"),
         "layers.{bid}.post_feedforward_layernorm.weight", "passthrough"),
        # per-layer LAuReL block
        (re.compile(r"^blk\.(\d+)\.laurel_l\.weight$"),
         "layers.{bid}.laurel.linear_left.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.laurel_r\.weight$"),
         "layers.{bid}.laurel.linear_right.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.laurel_post_norm\.weight$"),
         "layers.{bid}.laurel.post_laurel_norm.weight", "passthrough"),
        # per-layer AltUp coefs/router (correct_scale is a raw param: no
        #     .weight suffix on the mlx_lm side) ---
        (re.compile(r"^blk\.(\d+)\.altup_correct_coef\.weight$"),
         "layers.{bid}.altup.correction_coefs.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.altup_correct_scale\.weight$"),
         "layers.{bid}.altup.correct_output_scale", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.altup_predict_coef\.weight$"),
         "layers.{bid}.altup.prediction_coefs.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.altup_router\.weight$"),
         "layers.{bid}.altup.modality_router.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.altup_router_norm\.weight$"),
         "layers.{bid}.altup.router_norm.weight", "passthrough"),
        # per-layer per-layer-input gate/projection/norm
        (re.compile(r"^blk\.(\d+)\.inp_gate\.weight$"),
         "layers.{bid}.per_layer_input_gate.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.proj\.weight$"),
         "layers.{bid}.per_layer_projection.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.post_norm\.weight$"),
         "layers.{bid}.post_per_layer_input_norm.weight", "passthrough"),
    ],
    "PHI3": [
        # phi-3 ships fused projections that the canonical map gets wrong:
        #  - attn_qkv: canonical ATTN_QKV -> Qwen3.5 linear_attn.in_proj_qkv (wrong
        #    arch). mlx_lm phi3 wants a fused self_attn.qkv_proj it splits at
        #    forward (op_size = (n_heads + 2*n_kv_heads)*head_dim).
        #  - ffn_up: canonical FFN_UP -> mlp.up_proj, but phi-3's ffn_up is the
        #    fused [gate; up] (2*intermediate wide); mlx_lm splits it in
        #    mlp.gate_up_proj. Pass through fused (HF gate_up_proj order preserved).
        (re.compile(r"^blk\.(\d+)\.attn_qkv\.weight$"),
         "model.layers.{bid}.self_attn.qkv_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
         "model.layers.{bid}.mlp.gate_up_proj.weight", "passthrough"),
        # ffn_norm -> post_attention_layernorm (phi-3 has only input + post-attn
        # norms; pin past the FFN_NORM/FFN_PRE_NORM collision, as LLAMA/QWEN).
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
    ],
    "QWEN2": [
        # ffn_norm -> post_attention_layernorm (pin past the FFN_NORM/FFN_PRE_NORM
        # TENSOR_NAMES collision, as for QWEN3/LLAMA).
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # QKV biases - qwen2's signature (vs qwen3's q/k-norm). The canonical
        # path strips the ".bias" suffix to match the enum, then re-emits the
        # CANONICAL_HF ".weight" target, so a bias would be mis-named onto the
        # weight slot. Claim the three biases explicitly with their ".bias" HF
        # names (mlx_lm qwen2 builds q/k/v_proj with bias=True; o_proj has none).
        (re.compile(r"^blk\.(\d+)\.attn_q\.bias$"),
         "model.layers.{bid}.self_attn.q_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.bias$"),
         "model.layers.{bid}.self_attn.k_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.bias$"),
         "model.layers.{bid}.self_attn.v_proj.bias", "passthrough"),
    ],
    "QWEN2MOE": [
        # Qwen1.5-MoE (qwen2_moe): standard attn_norm + ffn_norm transformer with
        # routed experts and a shared expert (the qwen35moe expert layout, minus
        # the hybrid backbone). ffn_norm -> post_attention_layernorm (pin past the
        # FFN_NORM/FFN_PRE_NORM collision, as for QWEN2/QWEN3/LLAMA).
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # QKV biases - qwen2_moe builds q/k/v_proj with bias=True (o_proj has none).
        # The canonical path strips ".bias" and re-emits the ".weight" target, so
        # claim the three biases explicitly (as for QWEN2).
        (re.compile(r"^blk\.(\d+)\.attn_q\.bias$"),
         "model.layers.{bid}.self_attn.q_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.bias$"),
         "model.layers.{bid}.self_attn.k_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.bias$"),
         "model.layers.{bid}.self_attn.v_proj.bias", "passthrough"),
        # Routed experts - stored already-stacked; mlx-lm qwen2_moe wants
        # mlp.switch_mlp.* (SwitchGLU). Same layout as qwen35moe.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        # Shared expert - not in gguf-py TENSOR_NAMES.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.down_proj.weight", "passthrough"),
        # Shared expert gate: 1D [hidden_size] in GGUF -> nn.Linear(dim, 1) wants
        # [1, hidden_size]. Needs unsqueeze (as for qwen35moe).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert_gate.weight", "gate_1d_unsqueeze"),
        # Router.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.gate.weight", "passthrough"),
    ],
    "QWEN3MOE": [
        # ffn_norm -> post_attention_layernorm (pin past the FFN_NORM/FFN_PRE_NORM
        # collision, as for QWEN3/QWEN2/LLAMA).
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # Routed experts - stored already-stacked; mlx-lm qwen3_moe wants
        # mlp.switch_mlp.* (SwitchGLU), same as qwen35moe but with no shared
        # expert. The router is mlp.gate.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.gate.weight", "passthrough"),
    ],
    "QWEN35MOE": [
        # Routed experts - GGUF has separate gate/up/down (not fused like gemma-4).
        # Model-native path is mlp.switch_mlp.*, not mlp.experts.* (which would
        # need qwen3_5_moe.Model.sanitize to rename, but we build TextModel
        # directly to avoid the language_model. prefix).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        # Shared expert - not in gguf-py TENSOR_NAMES.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert.down_proj.weight", "passthrough"),
        # Shared expert gate: 1D [hidden_size] in GGUF -> nn.Linear(dim, 1)
        # expects [1, hidden_size]. Needs unsqueeze.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_expert_gate.weight", "gate_1d_unsqueeze"),
        # Router - explicit claim for self-containment.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.gate.weight", "passthrough"),
    ],
    "NEMOTRON_H_MOE": [
        # nemotron_h uses backbone.layers prefix and unified `mixer` module name
        # for all block types (attention, mamba, MoE, dense MLP). Every tensor
        # must be claimed here because CANONICAL_HF uses model.layers which is
        # wrong for this arch.
        #
        # Global tensors
        (re.compile(r"^token_embd\.weight$"),
         "backbone.embeddings.weight", "passthrough"),
        (re.compile(r"^output_norm\.weight$"),
         "backbone.norm_f.weight", "passthrough"),
        (re.compile(r"^output\.weight$"),
         "lm_head.weight", "passthrough"),
        # Per-layer norm (single norm per block, regardless of block type)
        (re.compile(r"^blk\.(\d+)\.attn_norm\.weight$"),
         "backbone.layers.{bid}.norm.weight", "passthrough"),
        # Attention (Q/K permuted by convert_hf_to_gguf via GraniteHybridModel)
        (re.compile(r"^blk\.(\d+)\.attn_q\.weight$"),
         "backbone.layers.{bid}.mixer.q_proj.weight", "qk_permute"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.weight$"),
         "backbone.layers.{bid}.mixer.k_proj.weight", "qk_permute"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.weight$"),
         "backbone.layers.{bid}.mixer.v_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_output\.weight$"),
         "backbone.layers.{bid}.mixer.o_proj.weight", "passthrough"),
        # Mamba2 SSM
        (re.compile(r"^blk\.(\d+)\.ssm_in\.weight$"),
         "backbone.layers.{bid}.mixer.in_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_conv1d\.weight$"),
         "backbone.layers.{bid}.mixer.conv1d.weight", "conv1d_unsqueeze"),
        (re.compile(r"^blk\.(\d+)\.ssm_conv1d\.bias$"),
         "backbone.layers.{bid}.mixer.conv1d.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_dt\.bias$"),
         "backbone.layers.{bid}.mixer.dt_bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ssm_a$"),
         "backbone.layers.{bid}.mixer.A_log", "ssm_a_to_a_log"),
        (re.compile(r"^blk\.(\d+)\.ssm_d$"),
         "backbone.layers.{bid}.mixer.D", "flatten"),
        (re.compile(r"^blk\.(\d+)\.ssm_norm\.weight$"),
         "backbone.layers.{bid}.mixer.norm.weight", "flatten"),
        (re.compile(r"^blk\.(\d+)\.ssm_out\.weight$"),
         "backbone.layers.{bid}.mixer.out_proj.weight", "passthrough"),
        # MoE router + expert correction bias
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "backbone.layers.{bid}.mixer.gate.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$"),
         "backbone.layers.{bid}.mixer.gate.e_score_correction_bias", "passthrough"),
        # MoE routed experts (stacked: shape [intermediate, hidden, n_experts])
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "backbone.layers.{bid}.mixer.switch_mlp.fc1.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "backbone.layers.{bid}.mixer.switch_mlp.fc2.weight", "passthrough"),
        # MoE latent projections (dimensionality reduction before/after experts)
        (re.compile(r"^blk\.(\d+)\.ffn_latent_down\.weight$"),
         "backbone.layers.{bid}.mixer.fc1_latent_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_latent_up\.weight$"),
         "backbone.layers.{bid}.mixer.fc2_latent_proj.weight", "passthrough"),
        # MoE shared expert
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "backbone.layers.{bid}.mixer.shared_experts.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "backbone.layers.{bid}.mixer.shared_experts.down_proj.weight", "passthrough"),
        # Dense MLP (non-MoE layers, if present in the pattern)
        (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
         "backbone.layers.{bid}.mixer.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down\.weight$"),
         "backbone.layers.{bid}.mixer.down_proj.weight", "passthrough"),
    ],
    "GLM4MOE": [
        # GLM-4.5/4.6 (GLM-4.5-Air): standard MHA (q/k/v + optional qk-norm,
        # partial rotary) with deepseek-V3-style fine-grained MoE - sigmoid
        # gating + correction bias + group routing + a shared expert, behind a
        # leading dense block. The attention, the two per-layer norms (attn_norm
        # -> input_layernorm, post_attention_norm -> post_attention_layernorm), the
        # dense-MLP layers, and the router (ffn_gate_inp -> mlp.gate) all resolve
        # via the canonical map. The NextN/MTP tensors (blk.{N}.nextn.*) map to
        # canonical enums with no HF target -> auto-SKIP; the MTP block's standard
        # tensors land on model.layers.{num_hidden_layers}.* and mlx-lm's
        # glm4_moe.sanitize drops them. Q/K stay passthrough - GLM4-MoE uses NeoX
        # rope in ggml (LLAMA_ROPE_TYPE_NEOX), so llama.cpp's converter does not
        # permute Q/K, matching mlx_lm glm4_moe's rope traditional=False; no
        # qk_permute to undo (the dense glm4 NORM-rope/traditional=True case is the
        # mirror image, also passthrough). The MoE + bias tensors below need
        # explicit targets.
        #
        # QKV biases - GLM-4.5/4.6 has q/k/v bias (o_proj none). The canonical path
        # strips ".bias" to match the enum then re-emits the ".weight" target, so
        # claim the three biases explicitly (as for dense glm4 / qwen2); otherwise
        # they collide onto {q,k,v}_proj.weight and the quant matrices never load.
        (re.compile(r"^blk\.(\d+)\.attn_q\.bias$"),
         "model.layers.{bid}.self_attn.q_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.bias$"),
         "model.layers.{bid}.self_attn.k_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.bias$"),
         "model.layers.{bid}.self_attn.v_proj.bias", "passthrough"),
        # exp_probs_b: sigmoid-gating correction bias (no .weight; lands on
        # mlp.gate.e_score_correction_bias). No CANONICAL_HF target -> would skip.
        (re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$"),
         "model.layers.{bid}.mlp.gate.e_score_correction_bias", "passthrough"),
        # Routed experts (already stacked) -> switch_mlp (SwitchGLU).
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        # Shared expert.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.down_proj.weight", "passthrough"),
    ],
    "DEEPSEEK2": [
        # DeepSeek-V2/V3 (and GLM-4.x conversions that llama.cpp emits as the
        # 'deepseek2' arch): MLA attention + fine-grained MoE with a shared
        # expert. The MLA tensor names (q_a/q_b/kv_a_mqa/k_b/v_b) aren't in the
        # canonical CANONICAL_HF map, so every tensor is claimed explicitly here
        # (as for NEMOTRON_H_MOE). All passthrough - DeepSeek RoPE acts only on
        # the qk_rope split *after* q_b/kv_a, so there's no llama-style Q/K
        # permute to undo. Targets follow mlx_lm.models.deepseek_v3.
        #
        # MLA attention (absorbed layout; mlx_lm uses MultiLinear)
        # Query: low-rank A->norm->B. attention_bias is False on these GGUFs
        # (no q_a/kv_a/o_proj .bias tensors), matching mlx_lm's deepseek_v3.
        (re.compile(r"^blk\.(\d+)\.attn_q_a\.weight$"),
         "model.layers.{bid}.self_attn.q_a_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_q_a_norm\.weight$"),
         "model.layers.{bid}.self_attn.q_a_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_q_b\.weight$"),
         "model.layers.{bid}.self_attn.q_b_proj.weight", "passthrough"),
        # KV: compressed latent (kv_lora + qk_rope) -> norm; then the per-head
        # up-projections k_b / v_b. llama.cpp ships k_b/v_b already split and
        # per-head-stacked ([.., .., n_head]); their byte layout matches mlx_lm's
        # embed_q / unembed_out MultiLinear exactly, so they pass through onto a
        # KQuantMultiLinear (see modules.py) with no transform.
        (re.compile(r"^blk\.(\d+)\.attn_kv_a_mqa\.weight$"),
         "model.layers.{bid}.self_attn.kv_a_proj_with_mqa.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_kv_a_norm\.weight$"),
         "model.layers.{bid}.self_attn.kv_a_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k_b\.weight$"),
         "model.layers.{bid}.self_attn.embed_q.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v_b\.weight$"),
         "model.layers.{bid}.self_attn.unembed_out.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_output\.weight$"),
         "model.layers.{bid}.self_attn.o_proj.weight", "passthrough"),
        # per-layer norms (standard 2-norm transformer block)
        (re.compile(r"^blk\.(\d+)\.attn_norm\.weight$"),
         "model.layers.{bid}.input_layernorm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.post_attention_layernorm.weight", "passthrough"),
        # dense MLP (the leading first_k_dense_replace layers)
        (re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$"),
         "model.layers.{bid}.mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
         "model.layers.{bid}.mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down\.weight$"),
         "model.layers.{bid}.mlp.down_proj.weight", "passthrough"),
        # MoE: router + sigmoid correction bias
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.gate.weight", "passthrough"),
        # exp_probs_b is the V3 sigmoid-gating correction bias (no .weight in
        # MLX: gate.e_score_correction_bias). F32 -> cast to bf16 at load.
        (re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$"),
         "model.layers.{bid}.mlp.gate.e_score_correction_bias", "passthrough"),
        # MoE: routed experts (already stacked) -> switch_mlp
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.mlp.switch_mlp.down_proj.weight", "passthrough"),
        # MoE: shared expert
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.mlp.shared_experts.down_proj.weight", "passthrough"),
        # DSA "lightning indexer" (GLM-5.2 / glm-dsa arch only; targets
        # mlx_lm.models.deepseek_v32.Indexer). A cheap per-layer Q/K scorer that
        # selects the top-k keys for sparse attention. Inert for plain deepseek2
        # GGUFs (no indexer.* tensors). GGUF-native shapes: attn_q_b
        # [q_lora, n_heads*head_dim] -> wq_b; attn_k [hidden, head_dim] -> wk;
        # proj [hidden, n_heads] -> weights_proj. k_norm is a LayerNorm (w + b).
        (re.compile(r"^blk\.(\d+)\.indexer\.attn_q_b\.weight$"),
         "model.layers.{bid}.self_attn.indexer.wq_b.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer\.attn_k\.weight$"),
         "model.layers.{bid}.self_attn.indexer.wk.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer\.k_norm\.weight$"),
         "model.layers.{bid}.self_attn.indexer.k_norm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer\.k_norm\.bias$"),
         "model.layers.{bid}.self_attn.indexer.k_norm.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer\.proj\.weight$"),
         "model.layers.{bid}.self_attn.indexer.weights_proj.weight", "passthrough"),
    ],
    "DEEPSEEK4": [
        # DeepSeek V4 Flash (dwarfstar 'deepseek4', not a llama.cpp arch).
        # Complete override block - none of the MLA-lite / compressor /
        # hyper-connection names exist in the canonical map, and the ones that
        # do (attn_norm, ffn_norm, ffn_gate_inp, exp_probs_b) target different
        # module paths in the vendored gmlx.deepseek_v4_model (block attrs
        # `attn_norm`/`ffn_norm`, MoE under `ffn.gate`). All passthrough: no
        # gemma unbake, and the NEOX-style tail rope is applied to q/kv after
        # the low-rank projections, so there's no llama Q/K permute to undo.
        #
        # Query: low-rank A -> RMSNorm -> B (per-head q_norms are weightless).
        (re.compile(r"^blk\.(\d+)\.attn_q_a\.weight$"),
         "model.layers.{bid}.attn.wq_a.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_q_a_norm\.weight$"),
         "model.layers.{bid}.attn.q_norm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_q_b\.weight$"),
         "model.layers.{bid}.attn.wq_b.weight", "passthrough"),
        # KV: one shared 512-dim latent (K == V), RMSNorm'd before rope.
        (re.compile(r"^blk\.(\d+)\.attn_kv\.weight$"),
         "model.layers.{bid}.attn.wkv.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_kv_a_norm\.weight$"),
         "model.layers.{bid}.attn.kv_norm.weight", "passthrough"),
        # Output: grouped low-rank wo_a (GGUF ships it 2D [o_groups*o_lora,
        # hidden]; the vendored sanitize() reshapes the wire bytes to the 3D
        # (o_groups, o_lora, -1) MultiLinear layout - pure header reshape,
        # rows unchanged) -> wo_b.
        (re.compile(r"^blk\.(\d+)\.attn_output_a\.weight$"),
         "model.layers.{bid}.attn.wo_a.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_output_b\.weight$"),
         "model.layers.{bid}.attn.wo_b.weight", "passthrough"),
        # Per-head fp32 attention sinks - raw array, no `.weight` (gpt-oss
        # `self_attn.sinks` precedent below).
        (re.compile(r"^blk\.(\d+)\.attn_sinks\.weight$"),
         "model.layers.{bid}.attn.attn_sink", "passthrough"),
        # Attention compressor (ratio-4 and ratio-128 layers): softmax-pooled
        # window summaries. `ape` is a raw (ratio, out_dim) positional table.
        (re.compile(r"^blk\.(\d+)\.attn_compressor_kv\.weight$"),
         "model.layers.{bid}.attn.compressor.wkv.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_compressor_gate\.weight$"),
         "model.layers.{bid}.attn.compressor.wgate.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_compressor_ape\.weight$"),
         "model.layers.{bid}.attn.compressor.ape", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_compressor_norm\.weight$"),
         "model.layers.{bid}.attn.compressor.norm.weight", "passthrough"),
        # Lightning indexer (ratio-4 layers): top-k selector over pooled rows.
        # Its own tensors are dotted (indexer.*); its private compressor's are
        # underscore-joined (indexer_compressor_*) - both spellings verified
        # against the real GGUF.
        (re.compile(r"^blk\.(\d+)\.indexer\.attn_q_b\.weight$"),
         "model.layers.{bid}.attn.indexer.wq_b.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer\.proj\.weight$"),
         "model.layers.{bid}.attn.indexer.weights_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer_compressor_kv\.weight$"),
         "model.layers.{bid}.attn.indexer.compressor.wkv.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer_compressor_gate\.weight$"),
         "model.layers.{bid}.attn.indexer.compressor.wgate.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer_compressor_ape\.weight$"),
         "model.layers.{bid}.attn.indexer.compressor.ape", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.indexer_compressor_norm\.weight$"),
         "model.layers.{bid}.attn.indexer.compressor.norm.weight",
         "passthrough"),
        # Block norms - the vendored block names them attn_norm/ffn_norm, not
        # the canonical input/post_attention_layernorm.
        (re.compile(r"^blk\.(\d+)\.attn_norm\.weight$"),
         "model.layers.{bid}.attn_norm.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
         "model.layers.{bid}.ffn_norm.weight", "passthrough"),
        # Hyper-connections: raw fp32 arrays on HyperConnection (fn/base/scale,
        # no `.weight`).
        (re.compile(r"^blk\.(\d+)\.hc_attn_fn\.weight$"),
         "model.layers.{bid}.attn_hc.fn", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.hc_attn_base\.weight$"),
         "model.layers.{bid}.attn_hc.base", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.hc_attn_scale\.weight$"),
         "model.layers.{bid}.attn_hc.scale", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.hc_ffn_fn\.weight$"),
         "model.layers.{bid}.ffn_hc.fn", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.hc_ffn_base\.weight$"),
         "model.layers.{bid}.ffn_hc.base", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.hc_ffn_scale\.weight$"),
         "model.layers.{bid}.ffn_hc.scale", "passthrough"),
        # MoE router: sqrt-softplus gate; e_score_correction_bias is the
        # selection-only bias; tid2eid is the raw I32 [vocab, top_k] hash-route
        # table on the first num_hash_layers layers.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.ffn.gate.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$"),
         "model.layers.{bid}.ffn.gate.e_score_correction_bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_tid2eid\.weight$"),
         "model.layers.{bid}.ffn.gate.tid2eid", "passthrough"),
        # Routed experts (pre-stacked [n_experts, ...]) + shared expert.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
         "model.layers.{bid}.ffn.switch_mlp.gate_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
         "model.layers.{bid}.ffn.switch_mlp.up_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
         "model.layers.{bid}.ffn.switch_mlp.down_proj.weight", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
         "model.layers.{bid}.ffn.shared_experts.gate_proj.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
         "model.layers.{bid}.ffn.shared_experts.up_proj.weight",
         "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
         "model.layers.{bid}.ffn.shared_experts.down_proj.weight",
         "passthrough"),
        # Final-collapse HyperHead (non-blk; no capture group -> target used
        # as-is). token_embd/output_norm/output resolve via the canonical map.
        (re.compile(r"^output_hc_fn\.weight$"),
         "model.hc_head.fn", "passthrough"),
        (re.compile(r"^output_hc_base\.weight$"),
         "model.hc_head.base", "passthrough"),
        (re.compile(r"^output_hc_scale\.weight$"),
         "model.hc_head.scale", "passthrough"),
    ],
    "GPT_OSS": [
        # OpenAI gpt-oss (mlx_lm.models.gpt_oss). The two per-layer norms
        # (attn_norm -> input_layernorm, post_attention_norm ->
        # post_attention_layernorm), the q/k/v/o weights, and the routed-expert
        # *weights* (ffn_{gate,up,down}_exps -> mlp.experts.{gate,up,down}_proj,
        # MXFP4) all resolve correctly via the canonical map. Q/K stay
        # passthrough - gpt-oss uses NeoX rope, so llama.cpp doesn't permute and
        # mlx_lm initialises rope traditional=False. The override-only tensors:
        #
        # Attention sinks - a per-head learned bias (mlx_lm: self.sinks, a raw
        # array, not a module), so the target has no `.weight`. No canonical enum
        # carries it to an HF target -> it would SKIP/FAIL without this.
        (re.compile(r"^blk\.(\d+)\.attn_sinks\.weight$"),
         "model.layers.{bid}.self_attn.sinks", "passthrough"),
        # Router - gpt-oss names it `mlp.router` (mlx_lm), not the canonical
        # `mlp.gate` that FFN_GATE_INP maps to.
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
         "model.layers.{bid}.mlp.router.weight", "passthrough"),
        # Biases - gpt-oss has q/k/v and o_proj bias, a router bias, and a bias
        # per expert projection. The canonical path strips `.bias` to match the
        # enum then re-emits the `.weight` target, so every bias must be claimed
        # explicitly or it collides onto the (quant) weight slot and overwrites it.
        (re.compile(r"^blk\.(\d+)\.attn_q\.bias$"),
         "model.layers.{bid}.self_attn.q_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.bias$"),
         "model.layers.{bid}.self_attn.k_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.bias$"),
         "model.layers.{bid}.self_attn.v_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.attn_output\.bias$"),
         "model.layers.{bid}.self_attn.o_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.bias$"),
         "model.layers.{bid}.mlp.router.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.bias$"),
         "model.layers.{bid}.mlp.experts.gate_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_up_exps\.bias$"),
         "model.layers.{bid}.mlp.experts.up_proj.bias", "passthrough"),
        (re.compile(r"^blk\.(\d+)\.ffn_down_exps\.bias$"),
         "model.layers.{bid}.mlp.experts.down_proj.bias", "passthrough"),
    ],
}

# GGUF tensor-name prefixes that indicate vision/audio tower content. These
# are skipped (not hard-failed); loading them into MLX needs mlx_vlm.utils
# plumbing well outside this loader's scope.
VLM_PREFIXES = (
    "v.",            # vision tower (per gguf TENSOR_NAMES convention: V_*)
    "v_enc.",
    "vision_tower.",
    "mm.",           # multi-modal projector
    "a.",            # audio tower (A_*)
    "audio_tower.",
    "resampler.",
)

# Tensors some GGUF conversions emit but no MLX model ever loads - precomputed
# values MLX derives at runtime. These are *expected* drops, not gaps: classify
# them as KIND_SKIP up front so even fail-closed arches (e.g. diffusion-gemma)
# don't misreport them as failures.
EXPECTED_SKIP_TENSORS = frozenset({
    "rope_freqs.weight",   # per-dim rope factors; loader copies it out before
                           # remap and rebuilds the ropes (_patch_rope_factors)
})


# Unsloth UD `.scale` tensors are metadata for Unsloth's MLX recipe (per-tensor
# additional scaling on top of the base K-quant). They have no HF/MLX-stock
# counterpart; skip with warning.
def _is_unsloth_ud_scale(name: str) -> bool:
    return name.endswith(".scale")


# Module-level cache: {gguf_name_template: MODEL_TENSOR_enum}. The reverse of
# `gguf.constants.TENSOR_NAMES` is global - TENSOR_NAMES is a single dict[enum,
# name] shared across archs, so there's nothing arch-specific to cache. Per-arch
# routing is handled by ARCH_PRIORITY_OVERRIDES (which runs first) and CANONICAL_HF
# (the post-lookup HF/MLX target naming). Built lazily on first call to keep the
# `from gguf.constants import TENSOR_NAMES` import cost out of module-load time.
_GGUF_TO_ENUM_CACHE: dict[str, "object"] | None = None


def _gguf_to_enum() -> dict[str, "object"]:
    global _GGUF_TO_ENUM_CACHE
    if _GGUF_TO_ENUM_CACHE is None:
        from gguf.constants import TENSOR_NAMES
        _GGUF_TO_ENUM_CACHE = {name: enum for enum, name in TENSOR_NAMES.items()}
    return _GGUF_TO_ENUM_CACHE


class RemapDecision:
    __slots__ = ("kind", "hf_name", "transform", "reason", "bid")
    KIND_MAP = "map"        # produce HF tensor (possibly with transform)
    KIND_SKIP = "skip"      # log warning + drop from output
    KIND_FAIL = "fail"      # hard error: unrecognized tensor

    def __init__(self, kind: str, *, hf_name: str | None = None,
                 transform: str = "passthrough", reason: str = "",
                 bid: int | None = None):
        self.kind = kind
        self.hf_name = hf_name
        self.transform = transform   # passthrough | qk_permute | moe_split_gate_up | conv1d_unsqueeze
        self.reason = reason
        self.bid = bid


# Layout transforms keyed on the matched tensor enum (bodies in transforms.py):
# FFN_GATE_UP_EXP - fused gate+up expert tensor, split into the two HF tensors.
# SSM_CONV1D - Qwen3.5/3.6 hybrid Mamba: GGUF stores conv1d as (out_ch, kernel),
#   MLX Conv1d expects (out_ch, kernel, in_ch_per_group=1) for depthwise.
# SSM_A - GGUF stores -exp(A_log); the MLX model expects raw A_log.
_ENUM_TRANSFORMS = {
    "FFN_GATE_UP_EXP": "moe_split_gate_up",
    "SSM_CONV1D": "conv1d_unsqueeze",
    "SSM_A": "ssm_a_to_a_log",
}


def parse_gguf_name(arch_string: str, gguf_name: str) -> RemapDecision:
    """Decide what to do with a single GGUF tensor name."""
    # VLM/audio tower skip - universal, no arch lookup needed.
    if any(gguf_name.startswith(p) for p in VLM_PREFIXES):
        return RemapDecision(RemapDecision.KIND_SKIP,
                             reason="vision/audio tower (prefix match)")

    arch_alias = ARCH_ALIAS.get(arch_string)

    # Arch-priority overrides - run before the universal .scale skip so an
    # arch can claim architectural .scale tensors (e.g., gemma-4-MoE router).
    if arch_alias is not None:
        for pat, hf_fmt, transform in ARCH_PRIORITY_OVERRIDES.get(arch_alias, []):
            m = pat.match(gguf_name)
            if m:
                if hf_fmt is None:
                    return RemapDecision(RemapDecision.KIND_SKIP,
                                         reason=f"arch-priority drops {gguf_name!r}")
                bid = int(m.group(1)) if m.groups() else None
                hf_name = hf_fmt.format(bid=bid) if bid is not None else hf_fmt
                transform = _gemma_norm_transform(arch_string, hf_name, transform)
                return RemapDecision(RemapDecision.KIND_MAP,
                                     hf_name=hf_name, transform=transform, bid=bid)

    # Expected, never-loaded tensors (e.g. precomputed RoPE freqs) - an explicit
    # skip, not a failure. Checked after arch overrides (so an arch could still
    # claim one) but before any fail-closed guard.
    if gguf_name in EXPECTED_SKIP_TENSORS:
        return RemapDecision(RemapDecision.KIND_SKIP,
                             reason=f"{gguf_name!r} is computed at runtime, never loaded")

    # DiffusionGemma is fail-closed: its override table above is complete, and
    # the shared canonical map would mis-home its backbone tensors to `model.*`
    # (no `decoder.` infix). Anything unclaimed is a genuine gap to extend, not a
    # tensor to guess at - so stop here rather than fall through.
    if arch_string == "diffusion-gemma":
        return RemapDecision(
            RemapDecision.KIND_FAIL,
            reason=f"no diffusion-gemma override for {gguf_name!r}; extend the "
                   f"DIFFUSION_GEMMA table")

    # Universal Unsloth-UD .scale skip (after arch-priority claims).
    if _is_unsloth_ud_scale(gguf_name):
        return RemapDecision(RemapDecision.KIND_SKIP,
                             reason="Unsloth UD .scale metadata tensor")

    # Strip the trailing ".weight" / ".bias" if present, since
    # `TENSOR_NAMES` templates omit it.
    base = gguf_name
    for s in (".weight", ".bias"):
        if base.endswith(s):
            base = base[: -len(s)]
            break

    if arch_alias is None:
        return RemapDecision(RemapDecision.KIND_SKIP,
                             reason=f"arch {arch_string!r} has no remap support; --no-remap or extend ARCH_ALIAS")

    # Try the canonical TENSOR_NAMES reverse map. The templates use {bid};
    # match by pattern.
    rev = _gguf_to_enum()
    bid: int | None = None
    matched_enum = None
    for tmpl, enum in rev.items():
        if "{bid}" in tmpl:
            # Replace {bid} with capture group, anchor full match.
            pat = "^" + re.escape(tmpl).replace(r"\{bid\}", r"(\d+)") + "$"
            m = re.match(pat, base)
            if m:
                bid = int(m.group(1))
                matched_enum = enum
                break
        else:
            if tmpl == base:
                matched_enum = enum
                break

    if matched_enum is not None:
        canonical = CANONICAL_HF.get(matched_enum.name)
        if canonical is None:
            # Mapped to an enum we deliberately drop (e.g., ROPE_FREQS).
            return RemapDecision(RemapDecision.KIND_SKIP,
                                 reason=f"{matched_enum.name} not serialized to HF/MLX")
        hf_name = canonical.format(bid=bid) if bid is not None else canonical
        # If the GGUF name didn't carry .weight (e.g., scalar tensors) and the HF
        # form does, that's fine - caller can still write under hf_name.
        transform = _ENUM_TRANSFORMS.get(matched_enum.name, "passthrough")
        if matched_enum.name in ("ATTN_Q", "ATTN_K") and arch_alias == "LLAMA":
            transform = "qk_permute"
        # gemma/gemma2/gemma3 (not gemma4) RMSNorm: llama.cpp bakes +1 into the
        # norm weights at convert; mlx_lm's gemma RMSNorm re-adds 1 at runtime,
        # so undo the bake. Keyed on arch_string, not arch_alias, because gemma4
        # shares the GEMMA3 alias but uses its norm weight directly.
        transform = _gemma_norm_transform(arch_string, hf_name, transform)
        return RemapDecision(RemapDecision.KIND_MAP,
                             hf_name=hf_name, transform=transform, bid=bid)

    # Per-arch overrides for tensors not in upstream TENSOR_NAMES.
    for pat, hf_fmt in EXTRA_OVERRIDES.get(arch_alias, []):
        m = pat.match(base)
        if m:
            if hf_fmt is None:
                return RemapDecision(RemapDecision.KIND_SKIP,
                                     reason=f"override drops {base!r}")
            bid = int(m.group(1)) if m.groups() else None
            hf_name = hf_fmt.format(bid=bid) if bid is not None else hf_fmt
            transform = _gemma_norm_transform(arch_string, hf_name, "passthrough")
            return RemapDecision(RemapDecision.KIND_MAP,
                                 hf_name=hf_name, transform=transform, bid=bid)

    # Genuine unknown - hard-fail so we extend the override table.
    return RemapDecision(RemapDecision.KIND_FAIL,
                         reason=f"no remap entry for {gguf_name!r} (arch={arch_string})")


# GGUF metadata + arch detection
def _read_string_field(reader, field_name: str) -> str | None:
    """Read a GGUF string KV field; return None if missing."""
    f = reader.fields.get(field_name)
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def detect_arch(reader) -> str:
    """Read `general.architecture` from GGUF metadata. Returns the raw string
    (e.g., "gemma4", "gemma3", "qwen3", "qwen3moe", "llama"). Caller maps to
    a TensorNameMap arch enum.
    """
    arch = _read_string_field(reader, "general.architecture")
    if arch is None:
        raise ValueError("GGUF missing 'general.architecture' KV field - can't detect arch")
    return arch
