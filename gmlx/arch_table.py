"""GGUF architecture -> mlx-lm model mapping, and the load gate.

A GGUF is loadable iff:
  (a) its ``general.architecture`` maps to a ``model_type`` (this table),
  (b) the *installed* backend package defines ``class Model`` for that
      ``model_type`` - usually mlx-lm (``mlx_lm/models/<model_type>.py``), but a
      few archs are backed by mlx-vlm (``_MLX_VLM_BACKED``; e.g. DiffusionGemma) or
      mlx-embeddings (``_MLX_EMBEDDINGS_BACKED``; encoder embedders) instead, and
  (c) a config synthesizer exists for the arch
      (``config_synth.supported_arches()``) or an ``hf_source`` config override
      is supplied.

A few archs are implemented but *disabled* (``config_synth.DISABLED_ARCHES``):
the synth/remap/loader code is complete, but no correctly-converted GGUF is
known to exist, so the gate refuses them regardless of ``hf_source``.

``model_type`` and ``remap_alias`` are pulled from ``config_synth`` and ``remap``
so this module never becomes a second source for them - it only adds the
``family`` grouping and free-text ``notes``. "Has a synthesizer" is likewise
derived from ``config_synth.supported_arches()``, never hand-kept here, to avoid
the split-brain that overcounts coverage.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from . import config_synth
from .remap import ARCH_ALIAS


@dataclass(frozen=True)
class ArchEntry:
    gguf_arch: str       # general.architecture, e.g. "qwen35moe"
    model_type: str      # model_type, e.g. "qwen3_5_moe"
    family: str          # config-synth family that owns it
    remap_alias: str     # remap.ARCH_ALIAS target, e.g. "QWEN35MOE" ("" if none)
    notes: str = ""
    backend: str = "mlx-lm"   # which package ships `class Model` (mlx-lm | mlx-vlm)


# Archs whose model class lives in mlx-vlm rather than mlx-lm. Almost every text
# GGUF builds an mlx-lm class; the exception is a model that is fundamentally not
# a plain autoregressive LM (e.g. DiffusionGemma's block-diffusion encoder-decoder
# is implemented only in mlx-vlm), so the gate checks the right package for it.
_MLX_VLM_BACKED = {"diffusion-gemma"}

# Archs whose model class lives in mlx-embeddings (bidirectional encoder
# embedders, not autoregressive LMs). model_type -> mlx-embeddings module file
# (the module name differs from model_type, like mlx-vlm packages do).
_MLX_EMBEDDINGS_BACKED = {"gemma-embedding"}
_MLX_EMBEDDINGS_MODULE = {"gemma_embedding": "gemma3_text"}


# gguf_arch -> (family, notes). family/notes are this module's contribution;
# model_type + remap_alias are filled from config_synth/remap below.
_FAMILY_NOTES = {
    "gemma4":         ("gemma",    "gemma-4 text tower (incl. the E2B/E4B omni and 12B unified-embedder LLMs); norm +1 bake undone on load; tied embeddings. Pairs with the gemma-4 mmprojs (--mmproj) for vision/audio, and with the gemma-4 assistant drafter (--draft-gguf) for MTP speculative decoding"),
    "gemma3":         ("gemma",    "gemma-3 1B/4B/12B; norm +1 bake undone on load; pass hf_source for 27B (query_pre_attn_scalar differs)"),
    "gemma3n":        ("gemma",    "gemma-3n E2B/E4B text tower (MatFormer); built as a standalone LanguageModel with a text_config-nested config (no wrapper prefix). AltUp (4 streams, predict/correct, coef clip 120) + LAuReL low-rank residual + per-layer input embeddings; alternating sliding(512)/full attention with separate sliding/global rope bases; KV-sharing on the last layers; tied embeddings; final logit softcap 30. Plain RMSNorm (no +1 bake). The stacked altup (un)projections are split per-stream on load"),
    "gemma2":         ("gemma",    "gemma-2 2B/9B; norm +1 bake undone; logit softcaps; pass hf_source for 27B (query_pre_attn_scalar differs). Note: mlx-lm's gemma2 has no sliding-window attention, so output is exact vs the GGUF reference only up to the 4096 window, and diverges above it (upstream mlx-lm limitation, not a loader issue)"),
    "gemma":          ("gemma",    "Gemma-1 2B/7B; norm +1 bake undone; head_dim (256) from key_length, not hidden//heads; tied embeddings"),
    "phi3":           ("phi3",     "Phi-3 mini/small/medium; fused qkv + gate_up; pass hf_source for 128K long-context (su/longrope) variants"),
    "glm4":           ("glm",      "GLM-4 9B/32B 0414 dense; 4 norms/layer (input/post-self-attn/pre-MLP/post-MLP); fused gate_up; partial rotary 0.5; QKV bias; untied lm_head"),
    "qwen2":          ("qwen2",    "Qwen2 / Qwen2.5 dense; QKV biases, tied embeddings on 0.5B/1.5B"),
    "qwen2moe":       ("qwen2",    "Qwen1.5-MoE-A2.7B; routed switch_mlp experts + shared expert + sigmoid shared gate; QKV biases; moe_intermediate from ffn_gate_exps shape (GGUF omits the KV key)"),
    "qwen3":          ("qwen3",    "Qwen3 dense (0.6B-32B); per-head qk-norm, NEOX rope (no qk-permute)"),
    "qwen35":         ("qwen3",    "Qwen3.5 dense hybrid: gated-DeltaNet linear attention with a full-attention layer every full_attention_interval; fused-GDN Metal kernels at runtime (GMLX_FUSED_GDN=0 disables); native-head MTP (nextn) -> --speculative needs no companion GGUF"),
    "qwen35moe":      ("qwen3",    "Qwen3.5/3.6 MoE (e.g. Qwen3.6-27B): the qwen35 gated-DeltaNet hybrid + fine-grained MoE with shared expert; fused-GDN kernels + native-head MTP as on qwen35"),
    "qwen3moe":       ("qwen3",    "Qwen3-MoE (30B-A3B / 235B-A22B); all-MoE switch_mlp, no shared expert"),
    "qwen3vlmoe":     ("qwen3",    "Qwen3-Omni thinker text tower (MoE, qwen3moe layout); pairs with the Qwen3-Omni mmproj (--mmproj) for vision + audio input"),
    "llama":          ("llama",    "Llama-2/3, Mistral-7B-as-llama, Vicuna; SPM merges reconstructed from scores. Sparse-MoE variants (Mixtral-8x7B/8x22B) ship under this arch with an expert count -> routed to model_type=mixtral (block_sparse_moe + SwitchGLU); legacy per-expert split weights are coalesced to the stacked form on load"),
    "mistral3":       ("llama",    "llama.cpp 'mistral3' = Ministral-3 / Mistral-Small-3.1, Llama layout"),
    "nemotron_h_moe": ("nemotron", "NVIDIA Nemotron-H MoE hybrid: Mamba2 SSM layers + sparse attention layers + MoE MLPs; layer_norm_epsilon (not rms_norm_eps)"),
    "deepseek2":       ("deepseek", "DeepSeek-V3/R1 + GLM-4.x MLA conversions (GLM-4.7-Flash); MLA attention (absorbed embed_q/unembed_out via KQuantMultiLinear, native Q8_0), fine-grained sigmoid-gated MoE + shared expert + group routing, leading dense block. V2 (softmax gating) not yet supported"),
    "deepseek4":      ("deepseek", "DeepSeek V4 Flash (256x8.4B, dwarfstar 'deepseek4' arch, not a llama.cpp conversion; parity reference is the ds4 engine). MLA-lite attention (low-rank q, single shared 512-dim KV latent K=V, grouped low-rank output proj, per-head fp32 sinks, tail-64 NEOX rope with inverse-rope'd output) in three per-layer variants from compress_ratios: sliding-window(128), +compressed pool, +lightning-indexer top-512 sparse. Manifold-constrained hyper-connections (4-stream 4D hidden, Sinkhorn mixing). Every-layer 256-expert MoE, sqrt-softplus gating + selection-only correction bias, first 3 layers hash-routed (tid2eid), shared expert, clamped SwiGLU. QAT round-trips (fp8-E4M3 KV, Hadamard+fp4-E2M1 indexer) reproduced on-path for logit parity. Model class vendored from mlx-lm PR #1192; separate MTP drafter GGUF (--draft-gguf) for speculative decoding"),
    "glm-dsa":        ("deepseek", "GLM-5.2 (DeepSeek-V3.2): deepseek2 MLA + fine-grained sigmoid-gated MoE plus a per-layer DSA 'lightning indexer' (top-k sparse-attention key selection) + an MTP/nextn layer (dropped on load). Reuses the DEEPSEEK2 remap (indexer.* patterns appended) -> mlx-lm model_type glm_moe_dsa (subclasses deepseek_v32)"),
    "glm4moe":        ("glm",      "GLM-4.5 / 4.6 (incl. GLM-4.5-Air); standard MHA (qk-norm, partial rotary) + DeepSeek-V3-style fine-grained sigmoid-gated MoE (shared expert, group routing, leading dense block). NextN/MTP layer dropped on load"),
    "gpt-oss":        ("gpt-oss",  "OpenAI gpt-oss 20B/120B; MoE (no dense MLP) with per-head attention sinks, alternating sliding/full attention, YaRN rope, and native MXFP4 experts (packed-repacked in RAM, or zero-copy GGUF wire bytes for streaming/over-RAM - never dequantized). Attn/embed/output Q8_0; router/norms/biases F32"),
    "seed_oss":       ("seed_oss", "ByteDance Seed-OSS 36B dense; plain Llama layout, explicit head_dim from attention.key_length (!= hidden//heads), NEOX rope (no qk-permute); F32 q/k/v attention biases claimed explicitly (attention_bias derived from tensor presence); tie_word_embeddings derived from output.weight presence"),
    "smollm3":        ("llama",    "HuggingFace SmolLM3-3B; Llama backbone (reuses the LLAMA remap alias incl. qk-permute, NORM rope) with NoPE on every 4th layer - llama.cpp hardcodes the step to 4, matching mlx-lm's no_rope_layer_interval default, so no GGUF KV drives it"),
    "granite":        ("granite",  "IBM Granite 3.x/4.x dense; Llama layout (reuses the LLAMA remap alias incl. qk-permute, NORM rope) + four runtime scalar multipliers (embedding_scale/residual_scale/attention.scale/logit_scale) applied at runtime, not folded into weights"),
    "ernie4_5-moe":   ("ernie",    "Baidu ERNIE-4.5-MoE (21B-A3B); fine-grained MoE (stacked routed experts -> SwitchGLU) + shared expert behind leading dense layers, softmax or aux-free sigmoid gating. NORM rope but mlx-lm uses traditional=True => Q/K pass through un-permuted (HF-native); the e_score_correction_bias is dropped on load (mlx-lm gates without it)"),
    "minimax-m2":     ("minimax",  "MiniMax-M2 (230B-A10B); every-layer fine-grained sigmoid-gated MoE (no dense layers, no shared expert) with full attention, full-width qk-norm (RMSNorm over head_dim*n_heads), and partial rotary (rotary_dim < head_dim). NEOX rope (no qk-permute); router/experts/correction-bias nested under block_sparse_moe.*; head_dim != hidden//heads (from key_length)"),
    "minimax-m3":     ("minimax_m3", "MiniMax-M3 (428B-A23B); M2's GQA base (partial rotary, head_dim from key_length, NEOX rope) plus gemma-style +1 RMSNorms (unbaked on load), per-head qk-norm, and a DeepSeek-V3-shaped MoE: leading dense layers, sigmoid gating + correction bias, routed weights renormalized x expert_weights_scale, per-layer shared expert (block_sparse_moe.shared_experts). SwiGLU-OAI activation. MSA sparse attention (llama.cpp PR #24908 semantics): a per-GQA-group indexer max-pools scores into 128-token blocks and top-16 blocks (local block forced) bound attention to 2048 KV per query - the form the model is trained with; runs whenever the GGUF carries the blk.N.indexer.* tensors or a `*indexer*.gguf` sidecar sits next to the model (GMLX_INDEXER_SIDECAR overrides discovery, GMLX_MSA_DISABLE=1 forces dense for A/B). Indexless GGUFs fall back to dense with a one-time warning (exact to 2048 tokens, degrades beyond - reasoning loops). Thinking tags are `<mm:think>`/`</mm:think>` (template-detected; the vocab's legacy `</think>` entries are decoys). Model class vendored from mlx-lm PR #1401 (+ gmlx MSA extension) until upstream ships models/minimax_m3.py"),
    "hunyuan-moe":    ("hunyuan",  "Tencent Hunyuan-A13B; softmax-gated fine-grained MoE + per-layer shared expert, per-head qk-norm (named query/key_layernorm), NTK-alpha rope. NEOX rope (no qk-permute); router -> mlp.gate.wg, shared expert -> mlp.shared_mlp; the GGUF materializes k/v on every layer so use_cla=False; rope alpha defaults to 1.0 (folded into freq_base); top-k router scores renormalized at load (norm_topk_prob - upstream mlx-lm omits it and degenerates)"),
    "hy_v3":          ("hunyuan",  "Tencent Hy3 (299B-A21B, llama.cpp PR #25395); sigmoid-gated fine-grained MoE (192 experts top-8) with selection-only expert bias (exp_probs_b, stored suffix-less) + top-k renorm x expert_weights_scale + one ungated shared expert, single leading dense layer (derived from tensor presence - no KV), per-head qk-norm, plain NEOX rope theta 11.16M (no qk-permute). Native MTP/NextN block past the trunk (stripped from the trunk on load; drafts via HyV3MTPDrafter - single-depth head, block_size 2, GMLX_HY3_MTP_BLOCK raises it). Router gate + expert bias pinned fp32 (llama.cpp routes fp32). HF enable_lm_head_fp32 pinned off (llama.cpp, the parity oracle, also computes the head in compute dtype). Early community GGUFs with arch 'hy-v3' (dash) are not mapped - reconvert. Model class vendored from mlx-lm PR #1485 with the MTP hidden-state wiring fixed to the vLLM-verified post-final-norm form"),
    "granitehybrid":  ("granite",  "IBM Granite 4.x hybrid (H-Micro/H-Tiny/H-Small); alternating Mamba2 + attention (layer_types from per-layer head_count_kv==0), softmax MoE + fused-input shared MLP (loader pre-fuses ffn_{gate,up}_shexp -> input_linear), granite runtime multipliers, NoPE via rope.scaling.finetuned=false. NORM rope => qk_permute on attention layers"),
    "falcon-h1":      ("falcon",   "TII Falcon-H1 (0.5B-34B); parallel attention + Mamba2 in every layer (one input_layernorm feeds both, outputs summed), dense gated MLP under feed_forward.*, explicit head_dim from key_length. NEOX rope (no qk-permute). The muP multiplier zoo is folded into the wire weights at convert => synth pins every multiplier neutral; ffn_norm/ssm_a/ssm_d stored with no .weight suffix"),
    "qwen3next":      ("qwen",     "Qwen3-Next-80B-A3B; gated-DeltaNet linear attention (3 of every 4 layers) + gated full attention (gate fused in attn_q), every-layer 512-expert MoE + shared expert. NEOX rope (no qk-permute, not qwen35's packing). Both GDN wire layouts load: legacy fused ssm_in -> in_proj_qkvz; the newer split attn_qkv/attn_gate via a load-time module split (loader swaps in_proj_qkvz for in_proj_qkv/in_proj_z, skipping the runtime de-interleave). V heads HF-grouped => the qwen3.5 tiled-V patch is excluded; +1 norm bake is what mlx-lm expects (passthrough)"),
    "gemma-embedding": ("gemma",   "EmbeddingGemma 300M; a gemma3 text backbone run as a bidirectional sentence encoder (mean pool + a 2-layer dense head), built into the mlx-embeddings gemma3_text.Model. Reuses the GEMMA3 remap/synth; norm +1 bake undone; dense_2/dense_3 -> the Model's dense.0/dense.1; tied embeddings. Needs the mlx-embeddings package (a core dependency)"),
    "diffusion-gemma": ("gemma",   "DiffusionGemma 26B-A4B; a non-autoregressive encoder-decoder block-diffusion model on the gemma-4 MoE backbone (each decoder layer runs a dense MLP + routed experts in parallel; encoder and decoder share weights bar a per-layer scalar). The decoder denoises a fixed-length canvas over reverse-diffusion steps with an entropy-bound sampler. Model class + denoising engine live in mlx-vlm (model_type diffusion_gemma); the backbone homes under model.decoder.* with the expert gate_up kept fused, plus the self-conditioning gated MLP. Single text-only GGUF (vision tower dropped at convert)"),
}


def _build_table() -> dict[str, ArchEntry]:
    table: dict[str, ArchEntry] = {}
    for arch, model_type in config_synth.GGUF_ARCH_TO_MODEL_TYPE.items():
        family, notes = _FAMILY_NOTES.get(arch, (arch, ""))
        table[arch] = ArchEntry(
            gguf_arch=arch,
            model_type=model_type,
            family=family,
            remap_alias=ARCH_ALIAS.get(arch, ""),
            notes=notes,
            backend=("mlx-vlm" if arch in _MLX_VLM_BACKED
                     else "mlx-embeddings" if arch in _MLX_EMBEDDINGS_BACKED
                     else "mlx-lm"),
        )
    return table


ARCH_TABLE = _build_table()


class UnsupportedArchError(Exception):
    """A GGUF architecture the runtime loader can't build a model for."""


def has_synth(gguf_arch: str) -> bool:
    """True iff ``config_synth`` produces a complete config for this arch."""
    return gguf_arch in config_synth.supported_arches()


# mlx-lm model_types whose module gmlx vendors (grafted into the
# mlx_lm.models namespace by the loader at build time) while the upstream PR is
# unmerged. Value: the vendoring gmlx module. An installed mlx-lm that
# ships the module wins over the vendored copy - once it does, drop the entry
# (and the vendored file).
_VENDORED_MLX_LM_MODULES = {
    # mlx-lm PR #1401 (MiniMax-M3 text backbone).
    "minimax_m3": "gmlx.minimax_m3_model",
    # mlx-lm PR #1192 (DeepSeek V4 Flash), + vendored hyper_connection and
    # PoolingCache/BatchPoolingCache companions injected by ensure_registered.
    "deepseek_v4": "gmlx.deepseek_v4_model",
    # mlx-lm PR #1485 (Tencent Hy3, supersedes #1211).
    "hy_v3": "gmlx.hy_v3_model",
}


def mlx_lm_has_model(model_type: str) -> bool:
    """True iff the installed mlx-lm has ``mlx_lm/models/<model_type>.py`` with a
    ``class Model`` - the class mlx-lm's ``_get_classes`` imports - or gmlx
    vendors the module (``_VENDORED_MLX_LM_MODULES``; the loader registers it
    into the mlx_lm.models namespace before ``_get_classes`` runs).
    Source-scanned (no import) so it has no side effects and tolerates a model
    whose import needs args the bare check can't supply.
    """
    candidates = [f"mlx_lm.models.{model_type}"]
    vendored = _VENDORED_MLX_LM_MODULES.get(model_type)
    if vendored is not None:
        candidates.append(vendored)
    for module in candidates:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError, ModuleNotFoundError):
            continue
        if spec is None or not spec.origin or not os.path.isfile(spec.origin):
            continue
        with open(spec.origin, encoding="utf-8") as f:
            if "class Model" in f.read():
                return True
    return False


def mlx_vlm_has_model(model_type: str) -> bool:
    """True iff the installed mlx-vlm ships ``class Model`` for ``model_type``.

    mlx-vlm models are packages (``mlx_vlm/models/<model_type>/`` with the class
    in a submodule), so when the spec resolves to a package ``__init__`` we scan
    its directory for ``class Model``; a single-file module is scanned directly.
    Source-scanned (no import) to stay side-effect-free, like
    :func:`mlx_lm_has_model`.
    """
    try:
        spec = importlib.util.find_spec(f"mlx_vlm.models.{model_type}")
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return False
    if os.path.basename(spec.origin) == "__init__.py":
        pkg_dir = os.path.dirname(spec.origin)
        for fn in sorted(os.listdir(pkg_dir)):
            if fn.endswith(".py"):
                with open(os.path.join(pkg_dir, fn), encoding="utf-8") as f:
                    if "class Model" in f.read():
                        return True
        return False
    with open(spec.origin, encoding="utf-8") as f:
        return "class Model" in f.read()


def mlx_embeddings_has_model(model_type: str) -> bool:
    """True iff the installed mlx-embeddings ships ``class Model`` for the encoder
    ``model_type``. The model_type maps to an mlx-embeddings module file via
    ``_MLX_EMBEDDINGS_MODULE`` (e.g. gemma_embedding -> gemma3_text). Source-scanned
    (no import), like :func:`mlx_lm_has_model`.
    """
    module = _MLX_EMBEDDINGS_MODULE.get(model_type, model_type)
    try:
        spec = importlib.util.find_spec(f"mlx_embeddings.models.{module}")
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return False
    with open(spec.origin, encoding="utf-8") as f:
        return "class Model" in f.read()


def gate(gguf_arch: str, *, hf_source: str | None = None) -> ArchEntry:
    """Resolve a GGUF arch to its ``ArchEntry`` or raise ``UnsupportedArchError``
    with an actionable message. The (a)/(b)/(c) failure modes, plus an explicit
    "implemented but disabled" refusal for archs with no known-good GGUF
    (``config_synth.DISABLED_ARCHES``) - unconditional, since the defect is in
    the weights and an ``hf_source`` config override can't fix it."""
    entry = ARCH_TABLE.get(gguf_arch)
    if entry is None:
        mapped = ", ".join(sorted(ARCH_TABLE))
        raise UnsupportedArchError(
            f"GGUF architecture {gguf_arch!r} is not supported: it has no mapping "
            f"to an mlx-lm model_type. Mapped architectures: {mapped}. "
            f"See docs/arch-coverage.md.")
    disabled = config_synth.DISABLED_ARCHES.get(gguf_arch)
    if disabled is not None:
        raise UnsupportedArchError(
            f"GGUF architecture {gguf_arch!r} is implemented but disabled: "
            f"{disabled}")
    if entry.backend == "mlx-vlm":
        has_model, where = mlx_vlm_has_model, f"mlx_vlm/models/{entry.model_type}/"
    elif entry.backend == "mlx-embeddings":
        module = _MLX_EMBEDDINGS_MODULE.get(entry.model_type, entry.model_type)
        has_model, where = (mlx_embeddings_has_model,
                            f"mlx_embeddings/models/{module}.py")
    else:
        has_model, where = mlx_lm_has_model, f"mlx_lm/models/{entry.model_type}.py"
    if not has_model(entry.model_type):
        pkg = entry.backend
        raise UnsupportedArchError(
            f"GGUF architecture {gguf_arch!r} maps to {pkg} model_type "
            f"{entry.model_type!r}, but the installed {pkg} has no "
            f"{where} defining `class Model`. Upgrade {pkg} to a release "
            f"that ships it.")
    if hf_source is None and not has_synth(gguf_arch):
        raise UnsupportedArchError(
            f"GGUF architecture {gguf_arch!r} (model_type {entry.model_type!r}) "
            f"has no config synthesizer yet. Pass hf_source=<hf-id-or-dir> to "
            f"supply a config.json, or see docs/arch-coverage.md.")
    return entry
