# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Black Forest Labs, Inc.
# Ported from BFL's FLUX.2 reference at commit 50fe516: the constants of
# Klein9BParams in src/flux2/model.py, the defaults of the 9B rows of
# FLUX2_MODEL_INFO in src/flux2/util.py and the Qwen3-8B encoder of
# load_qwen3_embedder in src/flux2/text_encoder.py, and modified for gmlx
# (Apache-2.0; see licenses/flux2-LICENSE).
"""FLUX.2 klein 9B registry row.

The 9B checkpoints run the module tree, the LoRA key map, the reference
image rules, the schedule and the sampling loop of the FLUX.2 klein row
at the widths and depths of Klein9BParams, and condition on Qwen3-8B
where the 4B ones condition on Qwen3-4B. This row comes before the 4B
row, takes the 9B files and returns None for every other file, which
the 4B row then reads and refuses by name where it should.

FLUX.2 klein 9B KV, which samples its reference images through a cache
of their keys and values, has the names and shapes of the 9B files and
other weights. Its published GGUFs carry no metadata of their own, so
it is refused by two of its tensors: the identity hash of the image
input projection, and the hash of one q norm scale, which a converter
keeps in float32.
"""

from __future__ import annotations

import dataclasses

from gmlx.image import fingerprint as fp
from gmlx.image.families.flux2_klein import family as klein
from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import ConstantsRow, Family, VariantRow

SIZE = "9B"
TITLE = f"{klein.TITLE} {SIZE}"

# The constants of the 4B row, which the official 9B transformer config
# gives too.
CONSTANTS = ConstantsRow(
    values=klein.CONSTANTS.values,
    repo="black-forest-labs/FLUX.2-klein-9B",
    revision="92196c8e11f7b6cf2b7493e037d8c5345c559216",
)

# The unsloth GGUFs of both 9B checkpoints keep the image input projection
# BF16 in every quant, so every file of a checkpoint gives one hash.
_QUANTS = ("BF16", "F16", "Q2_K", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1",
           "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_1", "Q5_K_M", "Q5_K_S", "Q6_K",
           "Q8_0")
_DISTILLED = VariantRow(
    "distilled", "unsloth/FLUX.2-klein-9B-GGUF",
    "fde8634245fe6b749a221c25b34672b5b8fbd079",
    tuple(f"flux-2-klein-9b-{q}.gguf" for q in _QUANTS))
_BASE = VariantRow(
    "base", "unsloth/FLUX.2-klein-base-9B-GGUF",
    "069a5df5ba8b7fe9f3355c8d4800486983ee8eb6",
    tuple(f"flux-2-klein-base-9b-{q}.gguf" for q in _QUANTS))
VARIANTS = {
    "a0b63c325bcf9b105b2afae179dacda5f1b0836294d808cc05ea056b25d9e3d4":
        _DISTILLED,
    "8df447f8fdd62769e91bc34e322b6e37ac22d12c258c9e93591b54a9dba12f15":
        _BASE,
}

KV_TITLE = f"{TITLE} KV"
KV_REFUSAL = (f"this is a {KV_TITLE} GGUF, which gmlx image does not "
              f"support: it samples its reference images through a cache "
              f"of their keys and values. {TITLE} and base 9B are the 9B "
              f"checkpoints it runs")
# The identity hash of KV's image input projection, BF16 in every GGUF of
# QuantStack/FLUX.2-Klein-9B-KV-GGUF at f4b7b4b0,
# vantagewithai/FLUX.2-klein-9b-kv-GGUF at 28216d4b and
# biali/FLUX.2-klein-9b-kv-GGUF at 994cdaa7 read.
REFUSED = {
    "41fe24e358863ebf78a373b480dce63ca4866a19396cb86f1fc6039c9c219706":
        KV_REFUSAL,
}
# A q norm scale whose bf16 bits differ between KV and both 9B
# checkpoints, F32 in every file read: the hash of KV's.
KV_MARK = "double_blocks.0.img_attn.norm.query_norm.scale"
KV_MARK_HASH = ("c8732255f8c1de5c691671d452cee78cca57e2bbd429f0d7074b72"
                "907d6650c6")

# The Qwen3-8B Q8_0 GGUF of unsloth/Qwen3-8B-GGUF holds the weights of the
# 9B text_encoder of the official repositories, which both 9B
# checkpoints share: its F32 norms equal the bf16 tensors there bit for
# bit, and the Q8_0 blocks of the matrices compared equal those bf16
# matrices quantized to Q8_0.
ENCODER_IDENTITY_SOURCE = (
    "black-forest-labs/FLUX.2-klein-9B text_encoder at "
    "92196c8e11f7b6cf2b7493e037d8c5345c559216")


def encoder():
    """The text encoder of the 9B checkpoints: Qwen3-8B under the rules
    of the 4B row's encoder, three of its 4096 wide states stacked to
    12288 features a token."""
    from gmlx.image.text_encoder import IdentityTensor

    identity = (
        IdentityTensor(
            "blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight",
            "00695bad97c2abc77a9d1ce57e4d4a5e5c2b574387fcb4029ef5ce12239b2530"),
        IdentityTensor(
            "blk.17.ffn_norm.weight",
            "model.layers.17.post_attention_layernorm.weight",
            "e8f207dcf6fb28797a560df4ec0adfddc4e914c8a7d0da39f389df9c35e887a7"),
        IdentityTensor(
            "output_norm.weight", "model.norm.weight",
            "4f4f6cc0467f0cf8516b154f3ec0748ed03d82ae24d435dfe678075e9a8e2070"),
    )
    return klein.qwen3_encoder("qwen3-8b", 4096, identity,
                               ENCODER_IDENTITY_SOURCE)


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    if klein.flux_model(view) != TITLE:
        return None
    refusal = klein.relative_refusal(view, SIZE)
    if refusal is not None:
        return refusal
    if fp.weight_identity(view, (KV_MARK,)) == KV_MARK_HASH:
        return Refusal(KV_REFUSAL)
    return Match()


def lora_map(config: dict):
    return dataclasses.replace(klein.lora_map(config), family="flux2_klein_9b",
                               title=TITLE)


FAMILY = Family(
    name="flux2_klein_9b",
    title=TITLE,
    # Before the FLUX.2 klein row, which then reads every other FLUX file.
    order=24,
    arch_strings=klein.FAMILY.arch_strings,
    shared_arch_strings=klein.FAMILY.shared_arch_strings,
    strip_prefixes=klein.FAMILY.strip_prefixes,
    fingerprint=fingerprint,
    synth_config=klein.synth_config,
    constants=CONSTANTS,
    build=klein.build,
    lora_map=lora_map,
    fp32_keep=klein.FAMILY.fp32_keep,
    encoder=encoder,
    identity_tensors=klein.IDENTITY_TENSORS,
    variants=VARIANTS,
    refused=REFUSED,
    sampling=klein.SAMPLING,
    unguided=klein.UNGUIDED,
    schedule=klein.schedule,
    loop=klein.LOOP,
    edit=klein.FAMILY.edit,
)
