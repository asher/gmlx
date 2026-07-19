"""Two-GGUF vision-language loading.

A vision-language model ships as two GGUFs: a K-quant **LLM** GGUF (a normal
text architecture) and a separate **float** ``mmproj`` GGUF
(``general.architecture = "clip"``) holding the vision encoder + cross-modal
projector. They carry no cross-reference; the pair is associated by an explicit
``mmproj`` path, the same contract as llama.cpp ``mtmd``.

This module loads both into a single ``mlx-vlm`` ``Model`` (``vision_tower`` /
``language_model`` / projector): the LLM tensors are remapped under the
``language_model.`` prefix and K-quant-swapped as usual, while the float vision
tensors are remapped onto the vision tower and left native. mlx-kquant owns the
GGUF side (parse / remap / config synth / build); ``mlx-vlm`` is an optional
dependency supplying the model classes + image processor.

The text-only :func:`..loader.load_model` path is untouched: VLM mode is entered
only through :func:`load_vlm_model` (CLI ``--mmproj``).
"""

from __future__ import annotations

import re

import mlx.core as mx
from transformers import BaseVideoProcessor as _BaseVideoProcessor
from transformers.image_processing_utils import ImageProcessingMixin

from . import loadlog
from .config_synth import synthesize_config
from .gdn_patches import (
    _needs_tiled_v_patch,
    _patch_gated_delta_tiled_v,
    _tiled_v_patch_applied,
)
from .gguf_meta import first_nonzero_int, read_int
from .loader import (
    _install_and_load,
    load_gguf_wire_bytes,
    remap_arrays,
)
from .preflight import preflight
from .transforms import coalesce_split_experts


class UnsupportedVLMError(Exception):
    """A (LLM arch, mmproj projector) pair with no VLM model mapping yet."""


# Model-type resolution: (LLM arch, mmproj metadata) -> mlx-vlm model_type

def resolve_vlm_model_type(llm_arch: str, mm_meta: dict) -> str:
    """Pick the mlx-vlm model_type for an LLM GGUF + its mmproj.

    The mmproj's ``clip.*`` metadata names the projector; the LLM arch
    disambiguates families that share one (e.g. gemma vision towers)."""
    if mm_meta.get("clip.has_llava_projector"):
        return "llava"
    # Older mmproj GGUFs carry a top-level ``clip.projector_type``; newer ones
    # (Qwen3-VL / Qwen3-Omni) put it under ``clip.vision.projector_type``.
    proj = (mm_meta.get("clip.projector_type")
            or _mm(mm_meta, "clip.vision.projector_type"))
    if proj == "pixtral":
        # Mistral Pixtral: a plain-float Pixtral ViT (2-D RoPE, RMSNorm, SiLU MLP)
        # + a 2-layer GELU projector onto a Mistral-Nemo (llama-arch) text tower.
        return "pixtral"
    if proj == "qwen2vl_merger":
        # Resolvable in principle (mlx-vlm has qwen2_vl), but none of the
        # vision remap / config synth / processor synth paths exist for the
        # Qwen2-VL ViT yet. Fail here, naming the family, rather than three
        # stages later with a remap error that reads like a loader bug.
        raise UnsupportedVLMError(
            "Qwen2-VL / Qwen2.5-VL mmprojs (projector 'qwen2vl_merger') are "
            "not supported yet")
    if proj == "qwen3vl_merger":
        # Qwen3-Omni (llama.cpp arch tag qwen3vlmoe) - a qwen3_moe thinker text
        # tower with both a Qwen3-VL vision encoder and a qwen3a Conformer audio
        # encoder (the mmproj sets clip.has_audio_encoder). The text/vision/audio
        # all nest under thinker.* in mlx-vlm's qwen3_omni_moe Model.
        if llm_arch == "qwen3vlmoe":
            return "qwen3_omni_moe"
        # Qwen3.6 (llama.cpp arch tags qwen35 / qwen35moe) - hybrid delta-net +
        # attention text model with a Qwen3-VL vision tower. The LLM arch picks
        # dense vs MoE; both share one vision encoder (deepstack disabled).
        if llm_arch == "qwen35moe":
            return "qwen3_5_moe"
        return "qwen3_5"
    if llm_arch == "gemma4":
        # gemma-4 ships two omni families: the E-series (E2B/E4B) with CLIP +
        # Conformer encoders (projector ``gemma4v``/``gemma4a``), and the 12B
        # "unified" model that drops both encoders for a lightweight embedder
        # (projector ``gemma4uv``/``gemma4ua``, ``block_count`` 0). The vision
        # projector type disambiguates.
        if _mm(mm_meta, "clip.vision.projector_type") == "gemma4uv":
            return "gemma4_unified"
        return "gemma4"
    raise UnsupportedVLMError(
        f"no VLM model mapping for LLM arch {llm_arch!r} + mmproj projector "
        f"{proj!r} (has_llava_projector={mm_meta.get('clip.has_llava_projector')})")


# Vision / projector remap  (mmproj clip names -> mlx-vlm module paths)

# LLaVA-1.5: a plain-float CLIP ViT-L/14-336 vision tower + a 2-layer MLP
# projector, mapped onto mlx_vlm.models.llava (VisionModel = ClipVisionModel,
# multi_modal_projector = LlavaMultiModalProjector).
# All vision towers share the same block-name shape; one compiled regex.
_VISION_BLK_RE = re.compile(r"^v\.blk\.(\d+)\.(.+)$")
_LLAVA_BLK_SUBMAP = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ln1": "layer_norm1",
    "ln2": "layer_norm2",
    # The clip export stores the FFN projections under inverted names: the
    # tensor named ``ffn_down`` is CLIP's ``fc1`` (hidden->intermediate up
    # projection) and ``ffn_up`` is ``fc2`` (intermediate->hidden). llama.cpp's
    # own loader swaps them back when ``ff_down_w->ne[0] == n_embd`` (clip.cpp).
    "ffn_down": "mlp.fc1",
    "ffn_up": "mlp.fc2",
}
_VM = "vision_tower.vision_model"
_LLAVA_TOP_MAP = {
    "v.patch_embd.weight": f"{_VM}.embeddings.patch_embedding.weight",
    "v.class_embd": f"{_VM}.embeddings.class_embedding",
    "v.position_embd.weight": f"{_VM}.embeddings.position_embedding.weight",
    "v.pre_ln.weight": f"{_VM}.pre_layrnorm.weight",   # mlx-vlm's spelling
    "v.pre_ln.bias": f"{_VM}.pre_layrnorm.bias",
    "v.post_ln.weight": f"{_VM}.post_layernorm.weight",
    "v.post_ln.bias": f"{_VM}.post_layernorm.bias",
    "mm.0.weight": "multi_modal_projector.linear_1.weight",
    "mm.0.bias": "multi_modal_projector.linear_1.bias",
    "mm.2.weight": "multi_modal_projector.linear_2.weight",
    "mm.2.bias": "multi_modal_projector.linear_2.bias",
}


def _llava_vision_name(name: str) -> str | None:
    """Map one mmproj clip tensor name to its mlx-vlm llava path, or None to
    skip (e.g. audio or unrecognized aux tensors)."""
    hit = _LLAVA_TOP_MAP.get(name)
    if hit is not None:
        return hit
    m = _VISION_BLK_RE.match(name)
    if m is None:
        return None
    bid, rest = m.group(1), m.group(2)
    tail = ""
    for suf in (".weight", ".bias"):
        if rest.endswith(suf):
            rest, tail = rest[: -len(suf)], suf
            break
    tgt = _LLAVA_BLK_SUBMAP.get(rest)
    if tgt is None:
        return None
    return f"{_VM}.encoder.layers.{bid}.{tgt}{tail}"


# Pixtral vision tower onto mlx_vlm.models.pixtral (VisionModel wraps a
# PixtralVisionModel under .vision_model). A plain ViT: a patch conv, one
# pre-encoder RMSNorm, N blocks (full-attn with 2-D RoPE, two RMSNorms, a
# SiLU-gated MLP), then a 2-layer GELU projector (multi_modal_projector). All
# leaf weights sit directly on their module (no ClippableLinear nesting). The
# text LLM rides under language_model.* via the usual llama remap.
# Note (upstream defect): community Pixtral mmproj GGUFs ship mangled vision
# attn_q/attn_k weights - a RoPE-permutation mismatch in llama.cpp's mmproj
# conversion (Pixtral ViT is 2-D RoPE). Only q/k are affected (attn_v/out, FFN,
# norms, patch conv, projector are faithful); it has no clean loader-side inverse
# and llama.cpp's own mtmd shows the same degradation. The remap below is correct
# - GGUF *vision* quality is just capped until a re-converted mmproj appears.
# See docs/vlm.md "Known upstream conversion defects".
_PIXTRAL_BLK_SUBMAP = {
    "attn_q": "attention.q_proj",
    "attn_k": "attention.k_proj",
    "attn_v": "attention.v_proj",
    "attn_out": "attention.o_proj",
    "ln1": "attention_norm",      # pre-attention RMSNorm
    "ffn_gate": "feed_forward.gate_proj",
    "ffn_up": "feed_forward.up_proj",
    "ffn_down": "feed_forward.down_proj",
    "ln2": "ffn_norm",            # pre-FFN RMSNorm
}
_PVM = "vision_tower.vision_model"
_PIXTRAL_TOP_MAP = {
    "v.pre_ln.weight": f"{_PVM}.ln_pre.weight",
    "mm.1.weight": "multi_modal_projector.linear_1.weight",
    "mm.1.bias": "multi_modal_projector.linear_1.bias",
    "mm.2.weight": "multi_modal_projector.linear_2.weight",
    "mm.2.bias": "multi_modal_projector.linear_2.bias",
}


def _pixtral_vision_name(name: str):
    """Map an mmproj clip tensor to its mlx-vlm pixtral path.

    Returns ``(target_name, is_patch_conv)`` or ``None`` to skip. The only skipped
    tensor is ``v.token_embd.img_break`` - Pixtral's learned ``[IMG_BREAK]`` row
    separator, which mlx-vlm folds into the text embedding (the processor inserts
    the literal token), so it carries no vision-tower weight.
    """
    hit = _PIXTRAL_TOP_MAP.get(name)
    if hit is not None:
        return hit, False
    if name == "v.patch_embd.weight":
        return f"{_PVM}.patch_conv.weight", True
    m = _VISION_BLK_RE.match(name)
    if m is None:
        return None
    bid, rest = m.group(1), m.group(2)
    sub, _, leaf = rest.rpartition(".")  # leaf = weight (pixtral vision is bias-free)
    tgt = _PIXTRAL_BLK_SUBMAP.get(sub)
    if tgt is None:
        return None
    return f"{_PVM}.transformer.layers.{bid}.{tgt}.{leaf}", False


# gemma-4 (E2B/E4B-it) omni vision tower onto mlx_vlm.models.gemma4. The vision
# attn/MLP linears nest under a ClippableLinear (``...q_proj.linear.weight``);
# use_clipped_linears is False, so the GGUF's per-tensor input/output min/max
# clip scalars (and the whole audio tower) are dropped. The text LLM rides under
# language_model.* via the usual remap; only the vision side lives here.
_GEMMA4V_PROJ_SUBMAP = {        # linears - real weight nests under .linear
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
_GEMMA4V_NORM_SUBMAP = {        # norms - weight sits directly on the module
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ln1": "input_layernorm",
    "attn_post_norm": "post_attention_layernorm",
    "ln2": "pre_feedforward_layernorm",
    "ffn_post_norm": "post_feedforward_layernorm",
}
_GEMMA4V_ENC = "vision_tower.encoder.layers"
_GEMMA4V_TOP_MAP = {
    "mm.input_projection.weight": "embed_vision.embedding_projection.weight",
    "v.position_embd.weight": "vision_tower.patch_embedder.position_embedding_table",
    # Post-pooler standardization (gemma-4-31B's larger SigLIP encoder ships it;
    # E4B does not - llama.cpp gates it on the tensors' presence, as does mlx-vlm
    # via VisionConfig.standardize). Applied as (hidden - std_bias) * std_scale
    # after the pooler, before the multimodal embedder's pre-projection RMS norm.
    "v.std_scale": "vision_tower.std_scale",
    "v.std_bias": "vision_tower.std_bias",
}
_GEMMA4V_CLIP_SCALARS = ("input_max", "input_min", "output_max", "output_min")


def _is_audio_tensor(name: str) -> bool:
    return (name.startswith("a.") or name.startswith("mm.a.")
            or name.startswith("af.") or ".audio" in name)


def _gemma4_vision_name(name: str):
    """Map an mmproj clip tensor to its mlx-vlm gemma4 path.

    Returns ``(target_name, is_patch_conv)`` or ``None`` to skip (clip scalars,
    audio, unrecognized aux). ``is_patch_conv`` flags the conv->linear reshape.
    """
    hit = _GEMMA4V_TOP_MAP.get(name)
    if hit is not None:
        return hit, False
    if name == "v.patch_embd.weight":
        return "vision_tower.patch_embedder.input_proj.weight", True
    m = _VISION_BLK_RE.match(name)
    if m is None:
        return None
    bid, rest = m.group(1), m.group(2)
    sub, _, leaf = rest.rpartition(".")
    if leaf in _GEMMA4V_CLIP_SCALARS:
        return None  # activation-range aux; use_clipped_linears=False drops it
    if leaf != "weight":
        return None
    if sub in _GEMMA4V_PROJ_SUBMAP:
        return f"{_GEMMA4V_ENC}.{bid}.{_GEMMA4V_PROJ_SUBMAP[sub]}.linear.weight", False
    if sub in _GEMMA4V_NORM_SUBMAP:
        return f"{_GEMMA4V_ENC}.{bid}.{_GEMMA4V_NORM_SUBMAP[sub]}.weight", False
    return None


def _patch_conv_to_linear(arr: mx.array) -> mx.array:
    """GGUF patch conv ``[out, C, kH, kW]`` -> Linear ``[out, kH*kW*C]``.

    mlx-vlm gemma4 ``_patchify`` flattens each patch as ``[kH, kW, C]`` (C
    innermost), so transpose ``(0, 2, 3, 1)`` then flatten matches its
    ``input_proj`` column order, making the Linear equal to llama.cpp's conv2d.
    """
    arr = mx.transpose(arr, (0, 2, 3, 1))  # [out, kH, kW, C]
    return arr.reshape(arr.shape[0], -1)


def _patchdim_chw_to_hwc(arr: mx.array, channels: int = 3) -> mx.array:
    """Reorder a flattened patch-vector axis from ``[C, h, w]`` to ``[h, w, C]``.

    The encoder-free patch embedder is a plain Linear over a flattened patch
    vector, but the GGUF stores ``patch_embd`` (and the pre-dense norm) with the
    channel axis outermost - PyTorch conv-kernel order ``[C, h, w]`` (w innermost),
    the same slot every other clip model's conv ``v.patch_embd`` uses - already
    flattened to a single trailing axis. mlx-vlm's image processor emits patches
    as ``[h, w, C]`` (C innermost). The Linear matmul and the pre-dense LayerNorm's
    per-element affine both require column order to match the pixel order, so
    reorder the trailing patch axis here. Works on 1-D (norm) and 2-D (dense
    ``[out, patch_dim]``) arrays alike.
    """
    patch_dim = arr.shape[-1]
    side = round((patch_dim // channels) ** 0.5)
    if channels * side * side != patch_dim:
        raise ValueError(
            f"patch_dim {patch_dim} is not channels*side^2 "
            f"(channels={channels}, side={side})")
    lead = arr.shape[:-1]
    arr = arr.reshape(*lead, channels, side, side)  # [..., C, h, w]
    arr = mx.moveaxis(arr, -3, -1)                   # [..., h, w, C]
    return arr.reshape(*lead, patch_dim)


# gemma-4 omni audio tower (Conformer / Universal Speech Model) onto
# mlx_vlm.models.gemma4. It mirrors the vision side: the attention / FFN /
# light-conv projections are ClippableLinear (real weight nests under ``.linear``,
# with per-tensor input/output min/max clip scalars sitting on the module), while
# norms, the relative-position projection, the depthwise conv, and per_dim_scale
# carry their array directly. Tensor names + the conv-module norm swap are an arch
# convention ported from llama.cpp (``tools/mtmd/clip.cpp`` +
# ``tools/mtmd/models/gemma4a.cpp``), not GGUF metadata. Audio uses
# use_clipped_linears=True (the clip scalars are real bounds the real model
# applies), vs vision's False.
# Both audio towers share this block-name shape; one compiled regex.
_AUDIO_BLK_RE = re.compile(r"^a\.blk\.(\d+)\.(.+)$")
_GEMMA4A_CONV_RE = re.compile(r"^a\.conv1d\.(\d+)\.(.+)$")
_GEMMA4A_ENC = "audio_tower.layers"

# ClippableLinear modules: weight nests under ``.linear``; the four clip scalars
# sit on the module itself.
_GEMMA4A_CLIP_LINEAR = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.post",
    "conv_pw1": "lconv1d.linear_start",
    "conv_pw2": "lconv1d.linear_end",
    "ffn_up": "feed_forward1.ffw_layer_1",
    "ffn_down": "feed_forward1.ffw_layer_2",
    "ffn_up_1": "feed_forward2.ffw_layer_1",
    "ffn_down_1": "feed_forward2.ffw_layer_2",
}
# Norm modules (weight sits directly on the module). conv_norm / norm_conv are
# SWAPPED in the GGUF relative to their graph role (upstream tensor_mapping.py):
# the tensor literally named ``conv_norm`` is the light-conv PRE norm, and
# ``norm_conv`` is the post-depthwise mid norm. clip.cpp loads them reversed and
# the graph (models/gemma4a.cpp: build_norm(residual, norm_conv_w) before the
# pointwise conv, conv_norm_w after the depthwise) confirms the positions.
_GEMMA4A_NORM = {
    "ffn_norm": "feed_forward1.pre_layer_norm",
    "ffn_post_norm": "feed_forward1.post_layer_norm",
    "ffn_norm_1": "feed_forward2.pre_layer_norm",
    "ffn_post_norm_1": "feed_forward2.post_layer_norm",
    "attn_pre_norm": "norm_pre_attn",
    "attn_post_norm": "norm_post_attn",
    "conv_norm": "lconv1d.pre_layer_norm",
    "norm_conv": "lconv1d.conv_norm",
    "ln2": "norm_out",
}
_GEMMA4A_TOP_MAP = {
    "a.input_projection.weight":
        "audio_tower.subsample_conv_projection.input_proj_linear.weight",
    "a.pre_encode.out.weight": "audio_tower.output_proj.weight",
    "a.pre_encode.out.bias": "audio_tower.output_proj.bias",
    "mm.a.input_projection.weight": "embed_audio.embedding_projection.weight",
}


def _gemma4_audio_name(name: str):
    """Map an mmproj ``a.*`` / ``mm.a.*`` tensor to its mlx-vlm gemma4 audio path.

    Returns ``(target_name, transform)`` or ``None`` to skip. ``transform`` is one
    of ``None`` / ``"conv2d"`` (SSCP conv [out,in,kH,kW]->[out,kH,kW,in]) /
    ``"conv_dw"`` (depthwise conv [C,K]->[C,K,1]) / ``"scalar"`` (0-d clip bound).
    """
    hit = _GEMMA4A_TOP_MAP.get(name)
    if hit is not None:
        return hit, None

    m = _GEMMA4A_CONV_RE.match(name)
    if m is not None:
        cid, rest = m.group(1), m.group(2)
        base = f"audio_tower.subsample_conv_projection.layer{cid}"
        if rest == "weight":
            return f"{base}.conv.weight", "conv2d"
        if rest == "norm.weight":
            return f"{base}.norm.weight", None
        return None  # SSCP conv bias: mlx-vlm Conv2d is bias-free

    m = _AUDIO_BLK_RE.match(name)
    if m is None:
        return None
    bid, rest = m.group(1), m.group(2)
    sub, _, leaf = rest.rpartition(".")
    pre = f"{_GEMMA4A_ENC}.{bid}"

    if sub in _GEMMA4A_CLIP_LINEAR:
        base = f"{pre}.{_GEMMA4A_CLIP_LINEAR[sub]}"
        if leaf == "weight":
            return f"{base}.linear.weight", None
        if leaf in _GEMMA4V_CLIP_SCALARS:
            return f"{base}.{leaf}", "scalar"
        return None
    if sub in _GEMMA4A_NORM:
        if leaf == "weight":
            return f"{pre}.{_GEMMA4A_NORM[sub]}.weight", None
        return None
    if leaf != "weight":
        return None
    if sub == "per_dim_scale":
        return f"{pre}.self_attn.per_dim_scale", None  # bare array attr (no .weight)
    if sub == "attn_k_rel":
        return f"{pre}.self_attn.relative_k_proj.weight", None  # plain nn.Linear
    if sub == "conv_dw":
        return f"{pre}.lconv1d.depthwise_conv1d.weight", "conv_dw"
    return None


def _apply_audio_transform(arr: mx.array, transform) -> mx.array:
    if transform is None:
        return arr
    if transform == "conv2d":  # [out, in, kH, kW] -> [out, kH, kW, in]
        return mx.transpose(arr, (0, 2, 3, 1))
    if transform == "conv_dw":  # [C, K] -> [C, K, 1]  (mlx Conv1d depthwise weight)
        return arr.reshape(arr.shape[0], arr.shape[1], 1)
    if transform == "scalar":  # (1,) clip bound -> 0-d, matching mx.array(+/-inf)
        return arr.reshape(())
    raise ValueError(f"unknown audio transform {transform!r}")


# gemma-4 unified (12B) - encoder-FREE omni, onto mlx_vlm.models.gemma4_unified.
# Vision replaces the CLIP tower with a lightweight embedder (patchify -> LayerNorm
# -> a single Linear -> LayerNorm -> add a 2-D position table -> LayerNorm); there are
# no transformer blocks (``clip.vision.block_count`` 0). Audio drops the Conformer
# tower entirely: the raw waveform is chunked and a single Linear projects it into
# text space. So the whole mmproj is 11 tensors with no per-block structure and no
# clip scalars. The three patch norms map by forward order: ``patch_norm.1`` is the
# pre-dense norm (its dim is 3*model_patch^2, the patch vector), ``patch_norm.2`` the
# post-dense norm, ``patch_norm.3`` the post-position-add norm.
_GEMMA4U_TOP_MAP = {
    "v.patch_norm.1.weight": "vision_embedder.patch_ln1.weight",
    "v.patch_norm.1.bias": "vision_embedder.patch_ln1.bias",
    "v.patch_embd.weight": "vision_embedder.patch_dense.weight",
    "v.patch_embd.bias": "vision_embedder.patch_dense.bias",
    "v.patch_norm.2.weight": "vision_embedder.patch_ln2.weight",
    "v.patch_norm.2.bias": "vision_embedder.patch_ln2.bias",
    "v.patch_norm.3.weight": "vision_embedder.pos_norm.weight",
    "v.patch_norm.3.bias": "vision_embedder.pos_norm.bias",
    "v.position_embd.weight": "vision_embedder.pos_embedding",  # transform "posemb"
    "mm.input_projection.weight": "embed_vision.embedding_projection.weight",
    "mm.a.input_projection.weight": "embed_audio.embedding_projection.weight",
}


# Patch-vector tensors the GGUF stores in conv ``[C, h, w]`` order (channel
# outermost) but mlx-vlm consumes in image-processor ``[h, w, C]`` order: the
# Linear weight's columns and the pre-dense norm's per-element affine both index
# the patch vector, so both need the same reorder. patch_norm.2/3 and patch_embd
# .bias live in the output (mm_embed) space, so they pass through untouched.
_GEMMA4U_PATCHDIM = {
    "v.patch_embd.weight",
    "v.patch_norm.1.weight",
    "v.patch_norm.1.bias",
}


def _gemma4_unified_name(name: str):
    """Map a gemma-4 unified mmproj tensor to its mlx-vlm path.

    Returns ``(target, transform)`` or ``None`` to skip. ``transform`` is
    ``"posemb"`` for the 2-D position table - GGUF stores it ``[2, P, D]`` and
    mlx-vlm's ``pos_embedding`` is ``[P, 2, D]`` (indexed ``[positions, axis]``),
    so swap the first two axes - ``"patchchw"`` for the patch-vector tensors that
    need a ``[C, h, w] -> [h, w, C]`` reorder, or ``None`` (passthrough).
    """
    hit = _GEMMA4U_TOP_MAP.get(name)
    if hit is None:
        return None
    if name == "v.position_embd.weight":
        return hit, "posemb"
    if name in _GEMMA4U_PATCHDIM:
        return hit, "patchchw"
    return hit, None


# Qwen3.6 (qwen3_5 / qwen3_5_moe) vision tower onto mlx_vlm.models.qwen3_vl's
# Qwen3VLVisionModel (shared by dense + MoE; deepstack disabled per qwen3.5). A
# plain ViT: a dual temporal patch conv, learned interpolated position
# embeddings, N blocks (fused QKV, two LayerNorms, GELU MLP), then a PatchMerger
# (pre-norm + 2-layer MLP) that also carries the multimodal projection - so
# unlike gemma-4 there is no separate embed_vision; every param lives under
# vision_tower.*. The merger's pre-norm is the GGUF's v.post_ln; mm.0/mm.2 are
# its two Linear layers (mm.1 is the GELU, no weight).
_QWEN35V_BLK_SUBMAP = {
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "ln1": "norm1",
    "ln2": "norm2",
}
_QWEN35V_TOP_MAP = {
    "v.patch_embd.bias": "vision_tower.patch_embed.proj.bias",
    "v.position_embd.weight": "vision_tower.pos_embed.weight",
    "v.post_ln.weight": "vision_tower.merger.norm.weight",
    "v.post_ln.bias": "vision_tower.merger.norm.bias",
    "mm.0.weight": "vision_tower.merger.linear_fc1.weight",
    "mm.0.bias": "vision_tower.merger.linear_fc1.bias",
    "mm.2.weight": "vision_tower.merger.linear_fc2.weight",
    "mm.2.bias": "vision_tower.merger.linear_fc2.bias",
}


def _qwen35_patch_embed_conv3d(w0: mx.array, w1: mx.array) -> mx.array:
    """Two temporal patch-conv slices -> one MLX Conv3d weight.

    Qwen3-VL's patch embed is a Conv3d with temporal kernel 2; the GGUF stores
    the two temporal slices as ``v.patch_embd.weight`` (t=0) and
    ``...weight.1`` (t=1), each ``[out, in, kH, kW]``. Stack to PyTorch
    ``[out, in, t, kH, kW]``, then transpose to MLX's ``[out, t, kH, kW, in]``
    (the same layout mlx-vlm's qwen3_vl sanitize produces for a 5-D conv).
    """
    proj = mx.stack([w0, w1], axis=2)            # [out, in, t, kH, kW]
    return mx.transpose(proj, (0, 2, 3, 4, 1))   # [out, t, kH, kW, in]


# Qwen3-Omni: the vision tower is qwen3.6's (qwen3vl_merger), so the block +
# top maps are reused verbatim under a ``thinker.`` prefix. The audio tower is a
# qwen3a Conformer mapped onto mlx-vlm's qwen3_omni_moe AudioModel.
_QWEN3OMNI_A_CONV_RE = re.compile(r"^a\.conv2d\.(\d+)\.(weight|bias)$")
_QWEN3OMNI_A_BLK_SUBMAP = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ln1": "self_attn_layer_norm",
    "ln2": "final_layer_norm",
    "ffn_up": "fc1",
    "ffn_down": "fc2",
}
_QWEN3OMNI_A_TOP_MAP = {
    "a.conv_out.weight": "audio_tower.conv_out.weight",
    "a.post_ln.weight": "audio_tower.ln_post.weight",
    "a.post_ln.bias": "audio_tower.ln_post.bias",
    "mm.a.mlp.1.weight": "audio_tower.proj1.weight",
    "mm.a.mlp.1.bias": "audio_tower.proj1.bias",
    "mm.a.mlp.2.weight": "audio_tower.proj2.weight",
    "mm.a.mlp.2.bias": "audio_tower.proj2.bias",
}


def remap_vision_arrays(
    arrays: dict[str, mx.array], model_type: str, *, with_audio: bool = False,
    mm_codecs: dict[str, str] | None = None,
) -> tuple[dict[str, mx.array], list[str], dict[str, str]]:
    """Remap a float mmproj's tensors onto the mlx-vlm vision tower + projector.

    With ``with_audio`` (an omni mmproj carrying an audio encoder), the ``a.*`` /
    ``mm.a.*`` audio tensors are also remapped onto the audio tower + audio
    embedder; otherwise they are skipped.

    Most mmproj GGUFs are pure float (vision/projector weights are never K-quant).
    A few (Qwen3-Omni) ship K-quant (Q8_0) vision/audio matmul weights; pass
    ``mm_codecs`` (the mmproj's GGUF tensor->codec map) so those tensors carry
    their packed wire bytes + ``.scales`` placeholder through to the kquant module
    swap. The returned ``vis_kqmeta`` (target-weight-name -> codec) is merged into
    the LLM ``hf_kquant_meta`` so ``install_kquant_modules`` swaps them too.

    Returns ``(hf_vision_weights, skipped_names, vis_kqmeta)``.
    """
    out: dict[str, mx.array] = {}
    skipped: list[str] = []
    vis_kqmeta: dict[str, str] = {}

    if model_type == "llava":
        for name, arr in arrays.items():
            if name.endswith(".scales") or name.endswith(".biases"):
                continue
            hf = _llava_vision_name(name)
            if hf is None:
                skipped.append(name)
                continue
            # patch conv: GGUF [out, in, kH, kW] (NCHW) -> nn.Conv2d [out, kH, kW, in].
            if name == "v.patch_embd.weight":
                arr = mx.transpose(arr, (0, 2, 3, 1))
            out[hf] = arr
        return out, skipped, vis_kqmeta

    if model_type == "pixtral":
        for name, arr in arrays.items():
            if name.endswith(".scales") or name.endswith(".biases"):
                continue
            res = _pixtral_vision_name(name)
            if res is None:
                skipped.append(name)
                continue
            hf, is_patch = res
            # patch conv: GGUF [out, in, kH, kW] (NCHW) -> nn.Conv2d [out, kH, kW, in].
            out[hf] = mx.transpose(arr, (0, 2, 3, 1)) if is_patch else arr
        return out, skipped, vis_kqmeta

    if model_type == "gemma4":
        for name, arr in arrays.items():
            if name.endswith(".scales") or name.endswith(".biases"):
                continue
            if _is_audio_tensor(name):
                if not with_audio:
                    skipped.append(name)
                    continue
                ares = _gemma4_audio_name(name)
                if ares is None:
                    skipped.append(name)
                    continue
                ahf, transform = ares
                out[ahf] = _apply_audio_transform(arr, transform)
                continue
            res = _gemma4_vision_name(name)
            if res is None:
                skipped.append(name)
                continue
            hf, is_patch = res
            out[hf] = _patch_conv_to_linear(arr) if is_patch else arr
        return out, skipped, vis_kqmeta

    if model_type == "gemma4_unified":
        for name, arr in arrays.items():
            if name.endswith(".scales") or name.endswith(".biases"):
                continue
            res = _gemma4_unified_name(name)
            if res is None:
                skipped.append(name)
                continue
            tgt, transform = res
            if transform == "posemb":  # [2, P, D] -> [P, 2, D]
                arr = mx.transpose(arr, (1, 0, 2))
            elif transform == "patchchw":  # [C, h, w] -> [h, w, C]
                arr = _patchdim_chw_to_hwc(arr)
            out[tgt] = arr
        return out, skipped, vis_kqmeta

    if model_type in ("qwen3_5", "qwen3_5_moe"):
        for name, arr in arrays.items():
            if name.endswith(".scales") or name.endswith(".biases"):
                continue
            if name == "v.patch_embd.weight.1":
                continue  # consumed with v.patch_embd.weight below
            if name == "v.patch_embd.weight":
                out["vision_tower.patch_embed.proj.weight"] = (
                    _qwen35_patch_embed_conv3d(arr, arrays["v.patch_embd.weight.1"]))
                continue
            hit = _QWEN35V_TOP_MAP.get(name)
            if hit is not None:
                out[hit] = arr
                continue
            m = _VISION_BLK_RE.match(name)
            if m is None:
                skipped.append(name)
                continue
            bid, rest = m.group(1), m.group(2)
            sub, _, leaf = rest.rpartition(".")  # leaf = weight | bias
            tgt = _QWEN35V_BLK_SUBMAP.get(sub)
            if tgt is None:
                skipped.append(name)
                continue
            out[f"vision_tower.blocks.{bid}.{tgt}.{leaf}"] = arr
        return out, skipped, vis_kqmeta

    if model_type == "qwen3_omni_moe":
        codecs = mm_codecs or {}

        def _emit(tgt: str, arr: mx.array, src: str) -> None:
            """Emit ``thinker.<tgt>``; for a K-quant source weight also carry its
            ``.scales`` placeholder and record the codec in ``vis_kqmeta``."""
            full = "thinker." + tgt
            codec = codecs.get(src)
            if codec is not None and full.endswith(".weight"):
                out[full] = arr
                out[full[: -len(".weight")] + ".scales"] = arrays[
                    src[: -len(".weight")] + ".scales"]
                vis_kqmeta[full] = codec
            else:
                out[full] = arr

        for name, arr in arrays.items():
            if name.endswith(".scales") or name.endswith(".biases"):
                continue

            # - audio tower (a.* / mm.a.*)
            if name.startswith("a.") or name.startswith("mm.a."):
                if not with_audio:
                    skipped.append(name)
                    continue
                if name == "a.position_embd.weight":
                    skipped.append(name)  # recomputed sinusoids, not a parameter
                    continue
                cm = _QWEN3OMNI_A_CONV_RE.match(name)
                if cm is not None:
                    idx, leaf = cm.group(1), cm.group(2)
                    if leaf == "weight":   # [out,in,kH,kW] -> mlx Conv2d [out,kH,kW,in]
                        arr = mx.transpose(arr, (0, 2, 3, 1))
                    else:                  # bias [out,1,1] -> [out]
                        arr = arr.reshape(-1)
                    _emit(f"audio_tower.conv2d{idx}.{leaf}", arr, name)
                    continue
                hit = _QWEN3OMNI_A_TOP_MAP.get(name)
                if hit is not None:
                    _emit(hit, arr, name)
                    continue
                m = _AUDIO_BLK_RE.match(name)
                if m is None:
                    skipped.append(name)
                    continue
                bid, rest = m.group(1), m.group(2)
                sub, _, leaf = rest.rpartition(".")
                tgt = _QWEN3OMNI_A_BLK_SUBMAP.get(sub)
                if tgt is None:
                    skipped.append(name)
                    continue
                _emit(f"audio_tower.layers.{bid}.{tgt}.{leaf}", arr, name)
                continue

            # - vision tower (v.* / mm.*) - qwen3.6's tower under thinker.
            if name == "v.patch_embd.weight.1":
                continue  # consumed with v.patch_embd.weight below
            if name == "v.patch_embd.weight":
                _emit("vision_tower.patch_embed.proj.weight",
                      _qwen35_patch_embed_conv3d(
                          arr, arrays["v.patch_embd.weight.1"]), name)
                continue
            hit = _QWEN35V_TOP_MAP.get(name)
            if hit is not None:
                _emit(hit, arr, name)
                continue
            m = _VISION_BLK_RE.match(name)
            if m is None:
                skipped.append(name)
                continue
            bid, rest = m.group(1), m.group(2)
            sub, _, leaf = rest.rpartition(".")
            tgt = _QWEN35V_BLK_SUBMAP.get(sub)
            if tgt is None:
                skipped.append(name)
                continue
            _emit(f"vision_tower.blocks.{bid}.{tgt}.{leaf}", arr, name)
        return out, skipped, vis_kqmeta

    raise UnsupportedVLMError(
        f"vision remap not implemented for model_type {model_type!r}")


# VLM config synthesis

def _mm(mm_meta: dict, key: str):
    """Read a clip.* metadata value (first element if it's a per-layer list)."""
    v = mm_meta.get(key)
    return v[0] if isinstance(v, list) else v


def _mm_int(mm_meta: dict, key: str) -> int:
    """Required integer clip.* field; a missing key fails naming itself
    instead of int(None)'s anonymous TypeError."""
    v = _mm(mm_meta, key)
    if v is None:
        raise ValueError(f"mmproj GGUF is missing required metadata {key!r}")
    return int(v)


def _mm_floats(mm_meta: dict, key: str) -> list[float] | None:
    """Read a clip.* metadata value as a float list (e.g. image_mean/std)."""
    v = mm_meta.get(key)
    if v is None:
        return None
    return [float(x) for x in v] if isinstance(v, (list, tuple)) else [float(v)]


def _synthesize_qwen35_vision_config(mm_meta: dict, model_type: str) -> dict:
    """Qwen3.6 vision_config from the mmproj's ``clip.vision.*`` metadata.

    Targets mlx-vlm's qwen3_vl ``VisionConfig`` (shared by qwen3_5/qwen3_5_moe).
    ``num_position_embeddings`` is the learned position grid (one embedding per
    patch over a square ``image_size/patch_size`` side), and ``out_hidden_size``
    is the merger's projection width = the text hidden size. deepstack is forced
    empty (qwen3.5 disables it; the GGUF's is_deepstack_layers are all zero)."""
    patch = _mm_int(mm_meta, "clip.vision.patch_size")
    image_size = _mm_int(mm_meta, "clip.vision.image_size")
    grid = image_size // patch
    return {
        "model_type": model_type,
        "depth": _mm_int(mm_meta, "clip.vision.block_count"),
        "hidden_size": _mm_int(mm_meta, "clip.vision.embedding_length"),
        "intermediate_size": _mm_int(mm_meta, "clip.vision.feed_forward_length"),
        "out_hidden_size": _mm_int(mm_meta, "clip.vision.projection_dim"),
        "num_heads": _mm_int(mm_meta, "clip.vision.attention.head_count"),
        "patch_size": patch,
        "spatial_merge_size": _mm_int(mm_meta, "clip.vision.spatial_merge_size"),
        "temporal_patch_size": 2,
        "num_position_embeddings": grid * grid,
        "in_channels": 3,
        "image_size": image_size,
        "deepstack_visual_indexes": [],
    }


def _gguf_token_id(llm_meta: dict, token: str) -> int | None:
    """Look up a token string's id in the GGUF vocab, or None if absent."""
    toks = llm_meta.get("tokenizer.ggml.tokens")
    if toks is not None:
        try:
            return toks.index(token)
        except ValueError:
            return None
    return None


def _synthesize_qwen35_vlm_config(
    text_config: dict, mm_meta: dict, model_type: str, llm_meta: dict
) -> dict:
    """Qwen3.6 VLM config: existing qwen35/qwen35moe text synth + Qwen3-VL vision.

    The text synth already emits every field mlx-vlm's qwen3_5 / qwen3_5_moe
    ``TextConfig`` needs (the ``linear_*`` delta-net params, the mrope
    ``rope_parameters``, ``full_attention_interval``, and for MoE the expert
    widths) because it reuses the mlx-lm qwen3_5 structures - so the text half is
    a passthrough. The marker token ids are resolved from the GGUF vocab, not
    mlx-vlm's ``ModelConfig`` defaults: those default vision_start/end ids
    (248045/6) are actually ``<|im_start|>``/``<|im_end|>`` for this tokenizer,
    while the real ``<|vision_start|>``/``<|vision_end|>`` are 248053/4 - and the
    text model's get_rope_index keys M-RoPE off vision_start, so a wrong id
    miscomputes image position ids."""
    text_config = dict(text_config)
    text_config["model_type"] = model_type
    config = {
        "model_type": model_type,
        "text_config": text_config,
        "vision_config": _synthesize_qwen35_vision_config(mm_meta, model_type),
        "vocab_size": int(text_config.get("vocab_size", 248320)),
    }
    for key, tok in (("image_token_id", "<|image_pad|>"),
                     ("video_token_id", "<|video_pad|>"),
                     ("vision_start_token_id", "<|vision_start|>"),
                     ("vision_end_token_id", "<|vision_end|>")):
        tid = _gguf_token_id(llm_meta, tok)
        if tid is not None:
            config[key] = tid
    return config


def _synthesize_qwen3_omni_config(
    text_config: dict, mm_meta: dict, llm_meta: dict
) -> dict:
    """Qwen3-Omni VLM config: the qwen3_moe text synth + a Qwen3-VL vision tower
    + a qwen3a Conformer audio tower, nested under ``thinker_config``.

    The text synth (read as ``qwen3vlmoe`` -> qwen3_moe) emits every field the
    omni ``TextConfig`` requires. The text rotary is M-RoPE; the per-axis section
    widths live in the LLM GGUF as ``qwen3vlmoe.rope.dimension_sections`` (here
    ``[24, 20, 20, 0]``, summing to ``head_dim/2``) - thread them onto
    ``rope_scaling.mrope_section`` so image position ids rotate correctly (text-
    only collapses to 1-D regardless, but an image makes the 3 axes diverge).

    ``ModelConfig.from_dict`` always builds ``talker_config`` / ``code2wav_config``
    even with ``enable_audio_output=False`` (Thinker-only), and ``TalkerConfig``'s
    nested ``TextConfig`` has 15 no-default fields - so feed the thinker text
    config as a satisfying dummy (the talker is never instantiated)."""
    text_config = dict(text_config)
    text_config["model_type"] = "qwen3_omni_moe_text_encoder"

    sections = llm_meta.get("qwen3vlmoe.rope.dimension_sections")
    if sections:
        mrope = [int(s) for s in sections if int(s) > 0]  # drop the trailing 0
        rope_scaling = dict(text_config.get("rope_scaling") or {})
        rope_scaling.setdefault("type", "default")
        rope_scaling.setdefault("rope_type", "default")
        rope_scaling["mrope_section"] = mrope
        text_config["rope_scaling"] = rope_scaling

    patch = _mm_int(mm_meta, "clip.vision.patch_size")
    grid = _mm_int(mm_meta, "clip.vision.image_size") // patch
    vision_config = {
        "model_type": "qwen3_omni_moe_vision_encoder",
        "depth": _mm_int(mm_meta, "clip.vision.block_count"),
        "hidden_size": _mm_int(mm_meta, "clip.vision.embedding_length"),
        "intermediate_size": _mm_int(mm_meta, "clip.vision.feed_forward_length"),
        "out_hidden_size": _mm_int(mm_meta, "clip.vision.projection_dim"),
        "num_heads": _mm_int(mm_meta, "clip.vision.attention.head_count"),
        "patch_size": patch,
        "spatial_merge_size": _mm_int(mm_meta, "clip.vision.spatial_merge_size"),
        "temporal_patch_size": 2,
        "num_position_embeddings": grid * grid,
        "in_channels": 3,
        "image_size": _mm_int(mm_meta, "clip.vision.image_size"),
        # qwen3.6/omni disable deepstack; the GGUF's is_deepstack_layers are zero.
        "deepstack_visual_indexes": [],
    }

    audio_config = {
        "model_type": "qwen3_omni_moe_audio_encoder",
        "d_model": _mm_int(mm_meta, "clip.audio.embedding_length"),
        "encoder_layers": _mm_int(mm_meta, "clip.audio.block_count"),
        "num_hidden_layers": _mm_int(mm_meta, "clip.audio.block_count"),
        "encoder_attention_heads": _mm_int(mm_meta, "clip.audio.attention.head_count"),
        "encoder_ffn_dim": _mm_int(mm_meta, "clip.audio.feed_forward_length"),
        "num_mel_bins": _mm_int(mm_meta, "clip.audio.num_mel_bins"),
        "output_dim": _mm_int(mm_meta, "clip.audio.projection_dim"),
        # downsample_hidden_size (the a.conv2d width, 480) is a qwen3a arch
        # constant the GGUF doesn't carry; AudioConfig's default (480) is correct.
    }

    thinker_config: dict = {
        "text_config": text_config,
        "vision_config": vision_config,
        "audio_config": audio_config,
    }
    for key, tok in (("image_token_id", "<|image_pad|>"),
                     ("video_token_id", "<|video_pad|>"),
                     ("audio_token_id", "<|audio_pad|>"),
                     ("vision_start_token_id", "<|vision_start|>"),
                     ("vision_end_token_id", "<|vision_end|>"),
                     ("audio_start_token_id", "<|audio_start|>"),
                     ("audio_end_token_id", "<|audio_end|>")):
        tid = _gguf_token_id(llm_meta, tok)
        if tid is not None:
            thinker_config[key] = tid

    return {
        "model_type": "qwen3_omni_moe",
        "enable_audio_output": False,   # Thinker only - no TTS talker / code2wav
        "thinker_config": thinker_config,
        "talker_config": {"text_config": text_config},  # dummy; never built
        "code2wav_config": {},
        "vocab_size": int(text_config.get("vocab_size", 151936)),
    }


def _synthesize_pixtral_vlm_config(
    text_config: dict, mm_meta: dict, llm_meta: dict
) -> dict:
    """Pixtral VLM config: the llama text synth + a Pixtral ViT vision_config.

    The text synth already emits every field mlx-vlm's pixtral ``TextConfig``
    needs (it reuses the llama backbone - Mistral-Nemo's head_dim 128 !=
    hidden/heads comes through from ``llama.rope.dimension_count``). The vision
    side reads the mmproj's ``clip.vision.*`` metadata; ``rope_theta`` (10000) is
    a Pixtral arch constant (the GGUF carries no vision rope base). The image
    placeholder is ``[IMG]`` (id 10 in the Tekken vocab) - resolved from the GGUF
    vocab so ``merge_input_ids_with_image_features`` keys off the right id."""
    head_count = _mm_int(mm_meta, "clip.vision.attention.head_count")
    hidden = _mm_int(mm_meta, "clip.vision.embedding_length")
    vision_config: dict = {
        "model_type": "pixtral",
        "num_hidden_layers": _mm_int(mm_meta, "clip.vision.block_count"),
        "hidden_size": hidden,
        "head_dim": hidden // head_count,
        "intermediate_size": _mm_int(mm_meta, "clip.vision.feed_forward_length"),
        "num_attention_heads": head_count,
        "image_size": _mm_int(mm_meta, "clip.vision.image_size"),
        "patch_size": _mm_int(mm_meta, "clip.vision.patch_size"),
        "num_channels": 3,
        "rope_theta": 10000.0,
    }
    proj_dim = _mm(mm_meta, "clip.vision.projection_dim")
    if proj_dim is not None:
        vision_config["projection_dim"] = int(proj_dim)
    eps = _mm(mm_meta, "clip.vision.attention.layer_norm_epsilon")
    if eps is not None:
        vision_config["rms_norm_eps"] = float(eps)

    config: dict = {
        "model_type": "pixtral",
        "text_config": text_config,
        "vision_config": vision_config,
        # Pixtral keeps every patch feature (no CLS token to drop).
        "vision_feature_select_strategy": "full",
        "vision_feature_layer": -1,
        "vocab_size": int(text_config.get("vocab_size", 131072)),
    }
    img_id = _gguf_token_id(llm_meta, "[IMG]")
    if img_id is not None:
        config["image_token_index"] = img_id
        config["image_token_id"] = img_id
    return config


def synthesize_vlm_config(
    model_type: str, llm_meta: dict, llm_shapes: dict, mm_meta: dict,
    *, mm_tensor_names: set[str] | None = None,
) -> dict:
    """Assemble an mlx-vlm config dict from the two GGUFs.

    ``text_config`` reuses the text-arch synthesizer on the LLM GGUF;
    ``vision_config`` is read from the mmproj's ``clip.vision.*`` metadata.
    ``mm_tensor_names`` (the mmproj's tensor key set) lets the config reflect
    optional tensors that carry no metadata flag - e.g. gemma-4 vision
    standardization, present only on the larger (31B) SigLIP encoder.
    """
    text_config = synthesize_config(llm_meta, llm_shapes)
    names = mm_tensor_names or set()

    if model_type == "gemma4":
        standardize = "v.std_scale" in names and "v.std_bias" in names
        return _synthesize_gemma4_vlm_config(
            text_config, mm_meta, standardize=standardize)
    if model_type == "gemma4_unified":
        return _synthesize_gemma4_unified_vlm_config(text_config, mm_meta)
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        return _synthesize_qwen35_vlm_config(
            text_config, mm_meta, model_type, llm_meta)
    if model_type == "qwen3_omni_moe":
        return _synthesize_qwen3_omni_config(text_config, mm_meta, llm_meta)
    if model_type == "pixtral":
        return _synthesize_pixtral_vlm_config(text_config, mm_meta, llm_meta)
    if model_type != "llava":
        raise UnsupportedVLMError(
            f"config synth not implemented for model_type {model_type!r}")

    vision_config: dict = {
        "model_type": "clip_vision_model",
        "num_hidden_layers": _mm_int(mm_meta, "clip.vision.block_count"),
        "hidden_size": _mm_int(mm_meta, "clip.vision.embedding_length"),
        "intermediate_size": _mm_int(mm_meta, "clip.vision.feed_forward_length"),
        "num_attention_heads": _mm_int(mm_meta, "clip.vision.attention.head_count"),
        "image_size": _mm_int(mm_meta, "clip.vision.image_size"),
        "patch_size": _mm_int(mm_meta, "clip.vision.patch_size"),
        "num_channels": 3,
    }
    proj_dim = _mm(mm_meta, "clip.vision.projection_dim")
    if proj_dim is not None:
        vision_config["projection_dim"] = int(proj_dim)
    eps = _mm(mm_meta, "clip.vision.attention.layer_norm_epsilon")
    if eps is not None:
        vision_config["layer_norm_eps"] = float(eps)

    return {
        "model_type": "llava",
        "text_config": text_config,
        "vision_config": vision_config,
        "image_token_index": 32000,
        "vision_feature_select_strategy": "default",
        # llama.cpp's CLIP conversion drops CLIP-L's unused 24th block, so the
        # mmproj carries block_count (=23) blocks; clip.cpp runs *all* of them
        # (`il < n_layer`) and feeds the projector the final block's output,
        # with no post-layernorm (the file ships none). mlx-vlm collects
        # hidden states as (embeddings, after-block-0, ..., after-block-22), so
        # that same final feature is hidden_states[-1] when block_count blocks
        # are built. (HF's stock -2 assumes the full 24-block tower; we build 23.)
        "vision_feature_layer": -1,
        "vocab_size": int(text_config.get("vocab_size", 32000)),
    }


def _synthesize_gemma4_vlm_config(
    text_config: dict, mm_meta: dict, *, standardize: bool = False
) -> dict:
    """gemma-4 omni config: existing gemma4 text synth + a vision tower from the
    mmproj's ``clip.vision.*`` metadata, plus an audio tower from ``clip.audio.*``
    when the mmproj carries one (``clip.has_audio_encoder``). Vision uses
    ``use_clipped_linears`` False (its clip scalars are dropped); audio uses True
    (its clip scalars are real bounds the real model applies). ``standardize``
    enables the post-pooler ``(h - std_bias) * std_scale`` step - set when the
    mmproj ships ``v.std_scale``/``v.std_bias`` (the 31B SigLIP encoder; E4B
    omits both). Token ids fall to the gemma-4 defaults; the processor drives
    soft-token expansion."""
    head_count = _mm_int(mm_meta, "clip.vision.attention.head_count")
    hidden = _mm_int(mm_meta, "clip.vision.embedding_length")
    # SigLIP vision is MHA, so kv-heads == q-heads unless the GGUF says otherwise.
    # mlx-vlm's VisionConfig.num_key_value_heads default (12) only happens to fit
    # E4B's 12-head encoder; the 31B's 16-head encoder needs it set explicitly or
    # k/v_proj come out GQA-sized (12*head_dim) and the load shape-mismatches.
    kv_heads = _mm(mm_meta, "clip.vision.attention.head_count_kv")
    vision_config: dict = {
        "model_type": "gemma4_vision",
        "hidden_size": hidden,
        "intermediate_size": _mm_int(mm_meta, "clip.vision.feed_forward_length"),
        "num_hidden_layers": _mm_int(mm_meta, "clip.vision.block_count"),
        "num_attention_heads": head_count,
        "num_key_value_heads": int(kv_heads) if kv_heads is not None else head_count,
        "head_dim": hidden // head_count,
        "patch_size": _mm_int(mm_meta, "clip.vision.patch_size"),
        "image_size": _mm_int(mm_meta, "clip.vision.image_size"),
        "use_clipped_linears": False,
        "standardize": standardize,
    }
    eps = _mm(mm_meta, "clip.vision.attention.layer_norm_epsilon")
    if eps is not None:
        vision_config["layer_norm_eps"] = float(eps)

    audio_config = None
    if mm_meta.get("clip.has_audio_encoder"):
        a_head = _mm_int(mm_meta, "clip.audio.attention.head_count")
        audio_config = {
            "hidden_size": _mm_int(mm_meta, "clip.audio.embedding_length"),
            "num_hidden_layers": _mm_int(mm_meta, "clip.audio.block_count"),
            "num_attention_heads": a_head,
            # llama.cpp hardcodes 1e-6 for every gemma4a model (clip.cpp), ignoring
            # the GGUF's clip.audio.attention.layer_norm_epsilon (1e-5 here); match.
            "rms_norm_eps": 1e-6,
            # The per-block projections that ship clip scalars are ClippableLinear
            # in the real model; clamp them (faithful to HF / llama.cpp build_mm).
            "use_clipped_linears": True,
            # Conformer output-projection width: a gemma4a arch constant, not GGUF
            # metadata (``a.pre_encode.out`` is [1536, hidden]).
            "output_proj_dims": 1536,
        }

    return {
        "model_type": "gemma4",
        "text_config": text_config,
        "vision_config": vision_config,
        "audio_config": audio_config,
        "vocab_size": int(text_config.get("vocab_size", 262144)),
    }


def _synthesize_gemma4_unified_vlm_config(text_config: dict, mm_meta: dict) -> dict:
    """gemma-4 unified (12B) config: the gemma4 text synth + an encoder-free
    vision embedder + a bare audio projection.

    The towers carry no encoder (``block_count`` 0), so the config only needs the
    projection widths the GGUF exposes - ``mm_embed_dim`` (the patch-dense output)
    and ``output_proj_dims`` (the soft-token feature width fed to the projector).
    The remaining geometry (``model_patch_size`` 48, ``pooling_kernel_size`` 3,
    ``mm_posemb_size`` 1120, ``num_soft_tokens`` 280) and the whole audio shape
    (raw-waveform ``audio_samples_per_token`` 640 -> ``output_proj_dims`` 640) are
    gemma-4-unified arch constants, not GGUF metadata; mlx-vlm's VisionConfig /
    AudioConfig defaults supply them. The text tower reuses the gemma4 synth (the
    12B's hparams - 48 layers, kv-shared 0, k==v full-attn - come straight from
    its GGUF and match the unified TextConfig defaults)."""
    text_config = dict(text_config)
    text_config["model_type"] = "gemma4_unified_text"

    mm_embed = _mm_int(mm_meta, "clip.vision.embedding_length")
    proj_dim = _mm(mm_meta, "clip.vision.projection_dim")
    vision_config: dict = {
        "model_type": "gemma4_unified_vision",
        "mm_embed_dim": mm_embed,
        "output_proj_dims": int(proj_dim) if proj_dim is not None else mm_embed,
        "patch_size": _mm_int(mm_meta, "clip.vision.patch_size"),
    }
    eps = _mm(mm_meta, "clip.vision.attention.layer_norm_epsilon")
    if eps is not None:
        vision_config["rms_norm_eps"] = float(eps)

    # Audio is a single raw-waveform projection (no Conformer). Its input width
    # (640 samples/token) is an arch constant carried only by the AudioConfig
    # default - the GGUF's clip.audio.embedding_length is the *output* (text)
    # dim, not the projection input - so emit a minimal config and let the
    # default output_proj_dims=640 stand.
    audio_config = None
    if mm_meta.get("clip.has_audio_encoder"):
        audio_config = {"model_type": "gemma4_unified_audio"}
        a_eps = _mm(mm_meta, "clip.audio.attention.layer_norm_epsilon")
        if a_eps is not None:
            audio_config["rms_norm_eps"] = float(a_eps)

    return {
        "model_type": "gemma4_unified",
        "text_config": text_config,
        "vision_config": vision_config,
        "audio_config": audio_config,
        "vocab_size": int(text_config.get("vocab_size", 262144)),
    }


# Build the mlx-vlm Model (bypassing mlx-vlm's nn.quantize)

def build_vlm_model(config_dict: dict):
    """Instantiate the mlx-vlm ``Model`` for ``config_dict['model_type']``.

    Mirrors mlx-vlm's own ``utils.load_model`` (``get_model_and_args`` +
    ``ModelConfig.from_dict`` + ``update_module_configs``) but stops before its
    ``nn.quantize`` - exactly as the text path bypasses mlx-lm's.
    """
    from mlx_vlm.utils import get_model_and_args, update_module_configs

    module, _model_type = get_model_and_args(config_dict)
    cfg = module.ModelConfig.from_dict(config_dict)
    cfg = update_module_configs(
        cfg, module, config_dict, ["text", "vision", "projector", "audio"])
    model = module.Model(cfg)
    return model, config_dict


# GGUF-only processor synthesis (image processor + tokenizer + chat template)

def _synthesize_vlm_processor(model_type: str, tokenizer, mm_meta: dict):
    """Build an mlx-vlm processor from the GGUFs alone - no HF download.

    Pixel-preprocessing params come straight from the mmproj's ``clip.vision.*``
    metadata; the few values llama.cpp keeps as arch constants rather than GGUF
    fields (the pooling factor + soft-token cap, ported from mtmd's gemma4v case)
    are supplied here. The marker tokens + chat template ride on ``tokenizer``,
    which was itself synthesized from the LLM GGUF. The result drives the same
    ``mlx_vlm.generate`` path an HF-built processor would.
    """
    if model_type == "gemma4_unified":
        return _attach_streaming_helpers(
            _synthesize_gemma4_unified_processor(tokenizer, mm_meta), tokenizer)
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        return _synthesize_qwen35_processor(tokenizer, mm_meta)
    if model_type == "qwen3_omni_moe":
        return _synthesize_qwen3_omni_processor(tokenizer, mm_meta)
    if model_type == "pixtral":
        return _synthesize_pixtral_processor(tokenizer, mm_meta)
    if model_type != "gemma4":
        raise UnsupportedVLMError(
            f"processor synth not implemented for model_type {model_type!r}")

    from mlx_vlm.models.gemma4.processing_gemma4 import (
        Gemma4ImageProcessor, Gemma4Processor,
    )

    patch_size = _mm_int(mm_meta, "clip.vision.patch_size")
    image_size = _mm_int(mm_meta, "clip.vision.image_size")
    image_mean = _mm_floats(mm_meta, "clip.vision.image_mean") or [0.0, 0.0, 0.0]
    image_std = _mm_floats(mm_meta, "clip.vision.image_std") or [1.0, 1.0, 1.0]
    # gemma-4 normalizes with mean 0 / std 1 (i.e. plain /255 rescale) - when the
    # GGUF carries exactly that, normalize is a no-op, matching llama.cpp's
    # img_u8_to_f32(mean=[0,0,0], std=[1,1,1]).
    do_normalize = (any(m != 0.0 for m in image_mean)
                    or any(s != 1.0 for s in image_std))

    # Arch constants ported from llama.cpp mtmd (clip.cpp gemma4v hparams):
    # n_merge=3 pooling, set_limit_image_tokens(., 280). Not GGUF metadata.
    # Note parity footgun: mlx-vlm's Gemma4ImageProcessor resize is hardcoded
    # BICUBIC, whereas llama.cpp gemma4v uses BILINEAR - a pixel-level
    # divergence to watch if greedy caption tokens drift.
    pooling_kernel_size = 3
    max_soft_tokens = 280

    image_processor = Gemma4ImageProcessor(
        size={"height": image_size, "width": image_size},
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        patch_size=patch_size,
        max_soft_tokens=max_soft_tokens,
        pooling_kernel_size=pooling_kernel_size,
    )

    # Audio: a gemma-4 omni mmproj carries a Conformer audio tower. The mel
    # frontend params are gemma4a arch constants (only the mel-bin count is in the
    # GGUF, as clip.audio.num_mel_bins); audio_seq_length (750) / audio_ms_per_token
    # (40) are likewise arch constants the Gemma4Processor uses to expand the audio
    # placeholder by waveform duration.
    feature_extractor = None
    audio_seq_length = 750
    if mm_meta.get("clip.has_audio_encoder"):
        from mlx_vlm.models.gemma4.audio_feature_extractor import (
            Gemma4AudioFeatureExtractor,
        )
        feature_extractor = Gemma4AudioFeatureExtractor(
            feature_size=_mm_int(mm_meta, "clip.audio.num_mel_bins"))

    processor = Gemma4Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None),
        image_seq_length=max_soft_tokens,
        audio_seq_length=audio_seq_length,
        feature_extractor=feature_extractor,
    )

    return _attach_streaming_helpers(processor, tokenizer)


def _attach_streaming_helpers(processor, tokenizer):
    """Attach the streaming detokenizer + stopping criteria mlx_vlm.generate
    needs (mlx-vlm's ``load_processor`` adds these on the HF path). EOS ids -
    incl. the inferred turn-end - ride on the GGUF tokenizer;
    ``NaiveStreamingDetokenizer`` is mlx-vlm's universal default."""
    from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
    from mlx_vlm.utils import StoppingCriteria

    eos_ids = getattr(tokenizer, "_gguf_eos_token_ids", None)
    if not eos_ids:
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    # transformers' SpecialTokensMixin intercepts attribute names ending in
    # `_token(_id|_ids)` and rejects non-string values, so set the list via the
    # base setter (StoppingCriteria.reset reads tokenizer.eos_token_ids).
    object.__setattr__(tokenizer, "eos_token_ids", eos_ids)
    processor.detokenizer = NaiveStreamingDetokenizer(tokenizer)
    tokenizer.stopping_criteria = StoppingCriteria(eos_ids, tokenizer)
    return processor


def _synthesize_gemma4_unified_processor(tokenizer, mm_meta: dict):
    """Build the gemma-4 unified (12B) processor from the GGUFs alone.

    The image processor patchifies into ``model_patch_size^2 * 3`` vectors fed
    straight to the patch-dense Linear (no CLIP tower grid); audio is a
    raw-waveform feature extractor (no mel frontend), 640 samples/token. Only
    ``patch_size`` comes from the GGUF; the pooling factor (3), soft-token count
    (280), SigLIP pixel normalization, and audio timing (640 samples = 40
    ms/token, 750-token cap) are gemma-4-unified arch constants.
    ``model_patch_size`` is derived inside the processor as
    ``patch_size * pooling_kernel_size`` (= 48).

    Pixel normalization is effectively a free choice here, unlike the E-series.
    The embedder's *first* op is ``patch_ln1``, a LayerNorm over the patch vector,
    and a LayerNorm is invariant to any global affine ``(x - m)/s`` on its input:
    the offset ``m`` cancels in ``z - mean(z)`` and the scale ``s`` cancels in the
    variance, surviving only through a negligible ``eps*s^2`` term (~1e-7 at eps
    1e-6). So mean = std = 0.5 (the SigLIP [-1, 1] convention gemma-4 vision uses)
    and the GGUF's ``clip.vision.image_mean/std`` = [0]/[1] placeholders produce
    the same embedder output to ~1e-6. We keep mean = std = 0.5 to stay faithful
    to the documented gemma-4 preprocessing; it is not load-bearing for parity.
    """
    from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
        Gemma4UnifiedAudioFeatureExtractor, Gemma4UnifiedImageProcessor,
        Gemma4UnifiedProcessor,
    )

    patch_size = _mm_int(mm_meta, "clip.vision.patch_size")
    image_size = _mm_int(mm_meta, "clip.vision.image_size")
    pooling_kernel_size = 3
    num_soft_tokens = 280

    image_processor = Gemma4UnifiedImageProcessor(
        size={"height": image_size, "width": image_size},
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        patch_size=patch_size,
        pooling_kernel_size=pooling_kernel_size,
        num_soft_tokens=num_soft_tokens,
    )

    feature_extractor = None
    if mm_meta.get("clip.has_audio_encoder"):
        feature_extractor = Gemma4UnifiedAudioFeatureExtractor()

    return Gemma4UnifiedProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None),
        image_seq_length=num_soft_tokens,
        audio_seq_length=750,
        audio_ms_per_token=40,
        feature_extractor=feature_extractor,
    )


def _synthesize_qwen35_processor(tokenizer, mm_meta: dict):
    """Build the Qwen3.6 (qwen3_5 / qwen3_5_moe) processor from the GGUFs alone.

    Qwen3-VL preprocessing is dynamic-resolution: ``smart_resize`` rescales each
    image so its patch count lands in ``[min_pixels, max_pixels]`` with both sides
    a multiple of ``patch_size * spatial_merge_size``, then patchifies into
    ``C * temporal_patch_size * patch_size^2`` vectors plus an ``image_grid_thw``
    the merger + text M-RoPE consume. ``patch_size``, ``spatial_merge_size`` and
    the SigLIP-style ``image_mean``/``image_std`` come from the mmproj metadata;
    ``temporal_patch_size`` (2, the dual-conv stack) is an arch constant. The
    pixel bounds are not in the GGUF - they are llama.cpp's qwen3vl
    ``set_limit_image_tokens(8, 4096)`` (clip.cpp): ``min/max_pixels`` = that
    token count times the per-token patch area ``(patch * merge)^2``, giving 8192
    / 4194304 here. The marker token ids (``<|image_pad|>``/``<|vision_start|>``/
    ``<|vision_end|>``) are resolved by ``Qwen3VLProcessor`` itself via the
    tokenizer's ``convert_tokens_to_ids`` (they live in the GGUF vocab as control
    tokens), so nothing extra rides on the tokenizer."""
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLImageProcessor, Qwen3VLProcessor,
    )

    patch_size = _mm_int(mm_meta, "clip.vision.patch_size")
    merge_size = _mm_int(mm_meta, "clip.vision.spatial_merge_size")
    image_mean = _mm_floats(mm_meta, "clip.vision.image_mean") or [0.5, 0.5, 0.5]
    image_std = _mm_floats(mm_meta, "clip.vision.image_std") or [0.5, 0.5, 0.5]
    # llama.cpp clip.cpp PROJECTOR_TYPE_QWEN3VL: set_limit_image_tokens(8, 4096),
    # patch_area = (patch_size * spatial_merge_size)^2. Temporal patch is the
    # dual-conv stack (constant 2), not GGUF metadata.
    temporal_patch_size = 2
    patch_area = (patch_size * merge_size) ** 2
    min_pixels = 8 * patch_area
    max_pixels = 4096 * patch_area

    image_processor = Qwen3VLImageProcessor(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=image_mean,
        image_std=image_std,
    )

    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=None,
        chat_template=getattr(tokenizer, "chat_template", None),
    )

    return _attach_streaming_helpers(processor, tokenizer)


class _OmniStubVideoProcessor(_BaseVideoProcessor):
    """Minimal ``BaseVideoProcessor`` carrying only the geometry the Qwen3-Omni
    processor reads (``merge_size`` always; ``temporal_patch_size`` only in the
    video branch). The real Qwen3VLVideoProcessor needs torch/torchvision; this
    stand-in lets the GGUF-only image path build without them. We subclass the
    *top-level* ``transformers.BaseVideoProcessor`` (the torchvision dummy when
    those backends are absent, the real class otherwise) so the processor's
    ``isinstance`` type-check passes either way; its ``__init__`` is bypassed
    (the dummy's would raise ``requires_backends``; the real one pulls in
    torch-backed transforms we never run)."""

    def __init__(self, merge_size: int, temporal_patch_size: int):
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size


def _synthesize_qwen3_omni_processor(tokenizer, mm_meta: dict):
    """Build the Qwen3-Omni processor from the GGUFs alone - no HF download.

    The image side is Qwen3-VL's dynamic-resolution pipeline (``smart_resize`` +
    ``image_grid_thw``), identical to qwen3_5. The omni processor reads
    ``video_processor.merge_size`` unconditionally (even with no video) and
    type-checks it as a ``BaseVideoProcessor``, so a tiny stub carrying just the
    merge / temporal-patch sizes stands in (the real Qwen3VLVideoProcessor needs
    torch). The seven multimodal marker token *strings* the processor pulls off
    the tokenizer (``image_token``/``audio_token``/``video_token`` + the
    vision/audio BOS/EOS) are set here from the GGUF vocab.

    Audio is a qwen3a Conformer fed Whisper-style log-mel features. Only the
    mel-bin count (128) is GGUF metadata; the rest of the frontend (16 kHz,
    25 ms / 10 ms window/hop = n_fft 400 / hop 160) are Qwen3-Omni arch
    constants. Audio-preprocessing parity is pending GPU validation; the image
    path is the validated deliverable."""
    from mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        Qwen3OmniMoeProcessor,
    )
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    patch_size = _mm_int(mm_meta, "clip.vision.patch_size")
    merge_size = _mm_int(mm_meta, "clip.vision.spatial_merge_size")
    image_mean = _mm_floats(mm_meta, "clip.vision.image_mean") or [0.5, 0.5, 0.5]
    image_std = _mm_floats(mm_meta, "clip.vision.image_std") or [0.5, 0.5, 0.5]
    temporal_patch_size = 2
    patch_area = (patch_size * merge_size) ** 2
    min_pixels = 8 * patch_area
    max_pixels = 4096 * patch_area

    image_processor = Qwen3VLImageProcessor(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=image_mean,
        image_std=image_std,
    )

    # The marker token strings ride on the tokenizer (the processor reads them
    # directly). object.__setattr__ bypasses SpecialTokensMixin's `_token`-suffix
    # interception (these are plain placeholder strings, not registered specials).
    for attr, tok in (("image_token", "<|image_pad|>"),
                      ("audio_token", "<|audio_pad|>"),
                      ("video_token", "<|video_pad|>"),
                      ("vision_bos_token", "<|vision_start|>"),
                      ("vision_eos_token", "<|vision_end|>"),
                      ("audio_bos_token", "<|audio_start|>"),
                      ("audio_eos_token", "<|audio_end|>")):
        object.__setattr__(tokenizer, attr, tok)

    feature_extractor = None
    if mm_meta.get("clip.has_audio_encoder"):
        from transformers import WhisperFeatureExtractor
        feature_extractor = WhisperFeatureExtractor(
            feature_size=_mm_int(mm_meta, "clip.audio.num_mel_bins"),
            sampling_rate=16000, hop_length=160, n_fft=400, padding_value=0.0)

    processor = Qwen3OmniMoeProcessor(
        image_processor=image_processor,
        video_processor=_OmniStubVideoProcessor(merge_size, temporal_patch_size),
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None),
    )

    return _attach_streaming_helpers(processor, tokenizer)


class _PixtralGgufImageProcessor(ImageProcessingMixin):
    """Torch-free Pixtral image preprocessing (numpy + PIL only).

    mlx-vlm's ``PixtralProcessor`` delegates pixel work to ``transformers``'
    ``PixtralImageProcessor``, which imports ``torch`` - absent here. This
    reimplements its preprocessing faithfully: resize so the longest edge <=
    ``longest_edge`` (aspect preserved, floor on the down-scale), align each side
    up to a ``patch_size`` multiple, BICUBIC resample, rescale by 1/255, then
    normalize by ``image_mean``/``image_std``. Output matches the transformers
    processor's contract - ``{"pixel_values": [N, C, H, W], "image_sizes":
    [(H, W), ...]}`` - so mlx-vlm's existing generate plumbing consumes it
    unchanged. (PIL BICUBIC != torchvision BICUBIC exactly; a sub-pixel resample
    difference that can perturb greedy tokens but not constrained-answer parity.)

    Subclasses ``transformers.ImageProcessingMixin`` (not ``BaseImageProcessor``)
    so ``ProcessorMixin``'s argument type-check accepts it, while staying *out* of
    mlx-vlm ``prepare_inputs``' simple single-soft-token branch (that gate keys off
    mlx-vlm's own unrelated ``BaseImageProcessor``) - Pixtral needs the ``[IMG]``
    grid-expansion path instead.
    """

    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(self, image_mean, image_std, longest_edge=1024, patch_size=16):
        super().__init__()
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.longest_edge = int(longest_edge)
        self.patch_size = int(patch_size)
        self.size = {"longest_edge": self.longest_edge}

    def _target_hw(self, h: int, w: int) -> tuple[int, int]:
        import math
        ratio = max(h / self.longest_edge, w / self.longest_edge)
        if ratio > 1:
            h = int(math.floor(h / ratio))
            w = int(math.floor(w / ratio))
        p = self.patch_size
        nh = (h - 1) // p + 1
        nw = (w - 1) // p + 1
        return nh * p, nw * p

    def _one(self, img):
        import numpy as np
        from PIL import Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        if img.mode != "RGB":
            img = img.convert("RGB")
        h_out, w_out = self._target_hw(img.height, img.width)
        img = img.resize((w_out, h_out), Image.Resampling.BICUBIC)  # PIL: (W, H)
        arr = np.asarray(img, dtype=np.float32) / 255.0             # [H, W, C]
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)
        arr = (arr - mean) / std
        return np.transpose(arr, (2, 0, 1)), (h_out, w_out)          # [C, H, W]

    @staticmethod
    def _flatten(images):
        flat = []
        stack = list(images) if isinstance(images, (list, tuple)) else [images]
        for it in stack:
            if isinstance(it, (list, tuple)):
                flat.extend(_PixtralGgufImageProcessor._flatten(it))
            else:
                flat.append(it)
        return flat

    def __call__(self, images, **kwargs):
        import numpy as np
        flat = self._flatten(images)
        processed, sizes = [], []
        for img in flat:
            chw, hw = self._one(img)
            processed.append(chw)
            sizes.append(hw)
        max_h = max(s[0] for s in sizes)
        max_w = max(s[1] for s in sizes)
        padded = np.zeros((len(processed), 3, max_h, max_w), dtype=np.float32)
        for i, (chw, (h, w)) in enumerate(zip(processed, sizes)):
            padded[i, :, :h, :w] = chw
        return {"pixel_values": padded, "image_sizes": sizes}


def _synthesize_pixtral_processor(tokenizer, mm_meta: dict):
    """Build the Pixtral processor from the GGUFs alone - no HF download.

    Wraps mlx-vlm's ``PixtralProcessor`` (it owns the ``[IMG]`` grid expansion +
    ``[IMG_BREAK]``/``[IMG_END]`` insertion, keyed off the tokenizer's own token
    ids) around a torch-free image processor. Pixel normalization
    (``image_mean``/``image_std``) and ``patch_size`` come from the mmproj's
    ``clip.vision.*`` metadata; the longest-edge resize bound (1024) is a Pixtral
    arch constant (llama.cpp clip.cpp). ``spatial_merge_size`` is 1 - Pixtral
    emits one token per patch (no merge), so the grid math uses ``patch_size``
    alone. The marker token ids ride on the GGUF-synthesized tokenizer."""
    from mlx_vlm.models.pixtral.processing_pixtral import PixtralProcessor

    patch_size = _mm_int(mm_meta, "clip.vision.patch_size")
    image_size = _mm_int(mm_meta, "clip.vision.image_size")
    image_mean = _mm_floats(mm_meta, "clip.vision.image_mean") or [
        0.48145466, 0.4578275, 0.40821073]
    image_std = _mm_floats(mm_meta, "clip.vision.image_std") or [
        0.26862954, 0.26130258, 0.27577711]

    image_processor = _PixtralGgufImageProcessor(
        image_mean=image_mean, image_std=image_std,
        longest_edge=image_size, patch_size=patch_size)

    processor = PixtralProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=patch_size,
        spatial_merge_size=1,
        chat_template=getattr(tokenizer, "chat_template", None),
    )

    return _attach_streaming_helpers(processor, tokenizer)


# Public entry point

@loadlog.seeds
def load_vlm_model(
    gguf_path: str,
    mmproj_path: str,
    *,
    hf_source: str | None = None,
    arch: str | None = None,
    zero_copy: bool = True,
    verbose: bool = False,
    return_tokenizer: bool = False,
):
    """Load a K-quant LLM GGUF + a float mmproj GGUF into an mlx-vlm ``Model``.

    Returns ``(model, config, processor)``. The processor (image preprocessing +
    tokenizer + chat template) is synthesized from the two GGUFs by default - no
    HF download. ``hf_source`` is an optional override (HF repo id or local dir)
    for the rare case a GGUF lacks a usable chat template or processor params.
    The model drives under ``mlx_vlm.generate(model, processor, prompt,
    image=...)``.

    ``return_tokenizer=True`` also returns the raw GGUF text tokenizer (the one
    the processor wraps, carrying ``_gguf_eos_token_ids``) as a 4th element, so
    the VLMxMTP path can wrap it in a ``TokenizerWrapper`` for the text-only
    speculative walk exactly like the text loader does.
    """
    _log = loadlog.verbose_print

    # 1. LLM GGUF - preflight gates the text arch (llama/gemma4/...) as usual.
    loadlog.stage("reading gguf metadata")
    pf = preflight(gguf_path, arch=arch)
    llm_arch = pf.arch
    loadlog.fact("arch", llm_arch)
    _log(f"[vlm] llm arch={llm_arch}")
    loadlog.stage("reading tensors")
    arrays, kquant_meta, _arch, llm_meta, llm_shapes = load_gguf_wire_bytes(
        gguf_path, zero_copy=zero_copy, shards=pf.shards)
    arrays, kquant_meta, n_coalesced = coalesce_split_experts(arrays, kquant_meta)
    if n_coalesced:
        _log(f"[vlm] coalesced {n_coalesced} split-expert groups -> stacked _exps")

    n_head = read_int(llm_meta, f"{llm_arch}.attention.head_count")
    n_head_kv = first_nonzero_int(
        llm_meta, f"{llm_arch}.attention.head_count_kv")

    # 2. mmproj GGUF - plain float metadata, no arch gate (general.architecture=
    #    "clip" is not a buildable text arch). Load it first to resolve the VLM
    #    model_type, which fixes where the text tower nests.
    loadlog.stage("reading mmproj")
    loadlog.fact("mmproj", True)
    mm_arrays, mm_codecs, _mm_arch, mm_meta, _mm_shapes = load_gguf_wire_bytes(
        mmproj_path, zero_copy=zero_copy, expect_quant=False)
    model_type = resolve_vlm_model_type(llm_arch, mm_meta)
    with_audio = bool(mm_meta.get("clip.has_audio_encoder"))
    _log(f"[vlm] model_type={model_type} audio={with_audio}")

    # The text tower nests under `language_model.` for most VLMs, but under
    # `thinker.language_model.` for the omni family (vision+audio+text Thinker).
    lm_prefix = ("thinker.language_model" if model_type == "qwen3_omni_moe"
                 else "language_model")
    loadlog.stage("remapping tensors")
    hf_weights, hf_kquant_meta, _stats = remap_arrays(
        arrays, kquant_meta, llm_arch,
        target_prefix=lm_prefix, n_head=n_head, n_head_kv=n_head_kv)

    # 3. vision/audio remap. Pure-float mmproj weights map straight through; a
    #    K-quant (Q8_0) omni mmproj also threads its codecs into hf_kquant_meta so
    #    install_kquant_modules swaps those vision/audio leaves like LLM leaves.
    vis_weights, skipped, vis_kqmeta = remap_vision_arrays(
        mm_arrays, model_type, with_audio=with_audio, mm_codecs=mm_codecs)
    _log(f"[vlm] mmproj: {len(vis_weights)} mapped, {len(skipped)} skipped, "
         f"{len(vis_kqmeta)} kquant")
    hf_weights.update(vis_weights)
    hf_kquant_meta.update(vis_kqmeta)

    from collections import Counter

    loadlog.fact("codecs", Counter(hf_kquant_meta.values()))

    # 4. synth config + build the mlx-vlm Model (no nn.quantize).
    loadlog.stage("building model")
    config = synthesize_vlm_config(
        model_type, llm_meta, llm_shapes, mm_meta,
        mm_tensor_names=set(mm_arrays))
    text_config = config.get("text_config", {})
    model, config = build_vlm_model(config)
    loadlog.fact("model_type", config.get("model_type"))

    # 3a. GGUF V-head tiling fixup for asymmetric linear-attention K/V heads
    #     (hybrid SSM+attn text towers, e.g. Qwen3.5/3.6). convert_hf_to_gguf
    #     stores gated-delta V heads tiled, but the gated_delta kernels assume
    #     grouped K->V indexing. The mlx-vlm text tower reaches the normal
    #     generate path through mlx_lm.models.gated_delta's gated_delta_kernel/
    #     _ops (it imports them), so the same monkey-patch the text loader uses
    #     fixes prefill and decode here. Without it the recurrent state is built
    #     against the wrong K heads and decode degenerates to token garbage even
    #     though the no-cache forward (and dense towers) look fine.
    if _needs_tiled_v_patch(text_config):
        _patch_gated_delta_tiled_v()
    elif (text_config.get("model_type") == "qwen3_next"
          and _tiled_v_patch_applied()):
        # Same cross-load hazard the text loader guards: once a qwen3.5/3.6
        # hybrid has patched mlx_lm.models.gated_delta in this process, a
        # qwen3_next text tower would silently run the wrong K->V mapping.
        raise RuntimeError(
            "cannot load a qwen3next GGUF after a qwen3.5/3.6 hybrid in the "
            "same process: the qwen3.5 tiled-V runtime patch (already applied) "
            "would corrupt qwen3_next's grouped gated-delta K->V mapping. "
            "Load the qwen3next model in a fresh process.")

    # 5. swap kquant leaves + load all weights. install_kquant swaps every leaf
    #    whose <path>.weight carries a codec - the LLM text tower always, plus the
    #    omni mmproj's Q8_0 vision/audio matmuls; pure-float mmproj weights have no
    #    codec and stay native. The remap already produced final mlx-vlm names
    #    (text under [thinker.]language_model.model.*, vision/audio under their
    #    towers), so model.sanitize must not run - it would re-prefix text keys.
    _install_and_load(model, hf_weights, hf_kquant_meta, log=_log, sanitize=False)

    # 5. processor (image preprocessing + tokenizer + chat template). Synthesized
    #    from the two GGUFs by default (the LLM GGUF carries the tokenizer + chat
    #    template + marker tokens; the mmproj carries the vision preprocessing
    #    params). hf_source is an optional override only.
    loadlog.stage("building processor")
    if hf_source:
        _log(f"[vlm] processor: hf_source override {hf_source!r}")
        from pathlib import Path

        from mlx_vlm.utils import load_processor
        # Resolve to a local dir of *config* files only - never the safetensors
        # weights (the GGUF supplies those), unlike mlx-vlm's get_model_path.
        src = Path(hf_source)
        if not src.exists():
            from huggingface_hub import snapshot_download

            from .hf_cache import network_fetch_allowed
            with network_fetch_allowed():
                src = Path(snapshot_download(
                    repo_id=hf_source,
                    allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken",
                                    "*.jinja", "*.py"]))
        processor = load_processor(src)
        mtp_tokenizer = getattr(processor, "tokenizer", None)
    else:
        from .tokenizer import load_tokenizer_from_gguf
        tokenizer = load_tokenizer_from_gguf(llm_meta, llm_arch)
        processor = _synthesize_vlm_processor(model_type, tokenizer, mm_meta)
        mtp_tokenizer = tokenizer
        _log("[vlm] processor: synthesized from GGUF (no download)")
    if return_tokenizer:
        return model, config, processor, mtp_tokenizer
    return model, config, processor
