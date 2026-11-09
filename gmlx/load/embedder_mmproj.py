"""The media towers of an embedding model, from an mmproj GGUF.

EmbeddingGemma 2 pairs its text GGUF with a Gemma 4 mmproj that carries a
vision tower (``gemma4v``) and an audio tower (``gemma4a``). The embedder
loads the towers on the first media request, into a
``gemma_embedding2.Towers`` module.

Weights stay in their file dtype where the activations allow it:

- With bfloat16 activations (M3 and later), F32 and F16 tensors are cast
  to bfloat16, as the text loader casts them. BF16 tensors stay BF16.
- With float32 activations (M1 and M2), F16 tensors stay F16 and F32
  tensors stay F32; each matmul promotes. Weights are not doubled.

A tensor the remap transforms (a convolution reshape) is dequantized from
a quantized projector to bfloat16 on M3 and later and to float16 on M1 and
M2, the 16-bit type each chip runs natively.
"""

from __future__ import annotations

import mlx.core as mx

from gmlx.models.embedders.gemma_embedding2 import TOWER_PREFIXES

from . import loadlog

__all__ = ["TOWER_PREFIXES", "dequant_dtype", "load_towers", "mmproj_mismatch"]

# Projector types of the Gemma 4 towers, by clip section.
_PROJECTORS = {"vision": "gemma4v", "audio": "gemma4a"}

def mmproj_mismatch(kv: dict, text_hidden: int) -> str | None:
    """Why the mmproj with the header ``kv`` is not a projector for an
    EmbeddingGemma 2 text model of width ``text_hidden``, or None."""
    if kv.get("general.architecture") != "clip":
        return "it is not an mmproj (general.architecture is not clip)"
    sections = [s for s in _PROJECTORS
                if kv.get(f"clip.has_{s}_encoder")]
    if not sections:
        return "it has no vision or audio encoder"
    for s in sections:
        kind = kv.get(f"clip.{s}.projector_type") or kv.get("clip.projector_type")
        if kind != _PROJECTORS[s]:
            return (f"its {s} projector is {kind!r}, not "
                    f"{_PROJECTORS[s]!r}")
        dim = kv.get(f"clip.{s}.projection_dim")
        if dim is not None and int(dim) != text_hidden:
            return (f"its {s} projection_dim is {int(dim)}, and the text "
                    f"model's width is {text_hidden}")
    return None


def dequant_dtype() -> mx.Dtype:
    """The 16-bit type this chip runs natively: bfloat16 on M3 and later,
    float16 on M1 and M2."""
    from .dtypes import has_native_bfloat16

    return mx.bfloat16 if has_native_bfloat16() else mx.float16


def load_towers(mmproj_path: str, *, text_hidden: int,
                act_dtype: mx.Dtype, log=loadlog.verbose_print):
    """The ``Towers`` of the mmproj at ``mmproj_path``, loaded for
    ``act_dtype`` activations. Raises ``ValueError`` with the reason when
    the file is not a projector for this text model."""
    from gmlx.models.embedders.gemma_embedding2 import Towers

    from .loader import _install_and_load, weights_source_key
    from .vlm import (
        gemma4_audio_config,
        gemma4_vision_config,
        remap_vision_arrays,
        unpack_unswappable_tower_weights,
    )
    from .wire import load_gguf_wire_bytes

    arrays, codecs, _arch, meta, _shapes = load_gguf_wire_bytes(
        mmproj_path, zero_copy=True, expect_quant=False)
    reason = mmproj_mismatch(meta, text_hidden)
    if reason:
        raise ValueError(reason)
    with_vision = bool(meta.get("clip.has_vision_encoder"))
    with_audio = bool(meta.get("clip.has_audio_encoder"))
    dq = dequant_dtype()
    weights, skipped, kqmeta = remap_vision_arrays(
        arrays, "gemma4", with_audio=with_audio, mm_codecs=codecs,
        mm_meta=meta, dequant_dtype=dq)
    standardize = "v.std_scale" in arrays and "v.std_bias" in arrays
    towers = Towers(
        text_hidden,
        gemma4_vision_config(meta, standardize=standardize) if with_vision else None,
        gemma4_audio_config(meta) if with_audio else None)
    unpack_unswappable_tower_weights(towers, weights, kqmeta, dq)
    log(f"[mmproj] {len(weights)} tensors mapped, {len(skipped)} skipped, "
        f"{len(kqmeta)} quantized")
    # With float32 activations, F16 tensors keep their dtype and F32 ones
    # need no cast, so the cast rule's f16_keep covers every tower tensor.
    keep = TOWER_PREFIXES if act_dtype == mx.float32 else ()
    _install_and_load(towers, weights, kqmeta, log=log, sanitize=False,
                      f16_keep=keep, act_dtype=act_dtype, text_patches=False,
                      source_key=weights_source_key(mmproj_path))
    if with_vision and act_dtype == mx.float32:
        # The vision tower casts pixels to its patch projection's dtype, so
        # a float16 projection would run the tower in float16. In float32
        # the rest of the tower promotes over its 16-bit weights.
        proj = towers.vision_tower.patch_embedder.input_proj
        if proj.weight.dtype == mx.float16:
            proj.weight = proj.weight.astype(mx.float32)
            mx.eval(proj.weight)
    towers.eval()
    return towers
