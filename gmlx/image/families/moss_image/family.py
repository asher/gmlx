# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 OpenMOSS Team
# Ported from OpenMOSS-Team/MOSS-Image-SFT at revision fc5de0d5: the
# tensor names and shapes of MOSSImageTransformer in
# pipeline_moss_image.py and the constants of that file and of
# transformer/config.json, and modified for gmlx (Apache-2.0; see
# licenses/MOSS-Image-LICENSE).
"""MOSS-Image fingerprint, constants and config synthesis.

Shapes are GGUF order, input dim first. The file's tensor names are the
training names of the release's MOSSImageTransformer, and the rule takes
a file whose names and shapes are exactly that model's.

SFT and Base share every name and shape, so the variant comes from the
hash of three small float tensors, which every published quant keeps in
F16. A file of another image family given as MOSS-Image is refused by
that family's name.
"""

from __future__ import annotations

import re

from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import ConstantsRow, Family, VariantRow

TITLE = "MOSS-Image"

# The release config: width, feed-forward width, text width, latent
# channels, sinusoid width, head dim and RoPE bands per axis.
_D, _F, _T, _C, _FQ, _HD, _BANDS = 3072, 8192, 4096, 128, 256, 128, 16
_DEPTH, _TEXT_DEPTH = 42, 2

_TOP = {
    "video_embedder.weight": (_C, _D),
    "video_embedder.bias": (_D,),
    "text_embedder.proj.weight": (_T, _D),
    "text_embedder.proj.bias": (_D,),
    "rope.bands": (_BANDS,),
    "time_embedding.0.weight": (_FQ, _D),
    "time_embedding.0.bias": (_D,),
    "time_embedding.2.weight": (_D, _D),
    "time_embedding.2.bias": (_D,),
    "shared_block_modulation.weight": (_D, 6 * _D),
    "shared_block_modulation.bias": (6 * _D,),
    "final_norm_video.weight": (_D,),
    "final_linear_video.weight": (_D, _C),
    "video_head_modulation.weight": (_D, 2 * _D),
    "video_head_modulation.bias": (2 * _D,),
}
_TEXT_BLOCK = {
    "norm1.weight": (_D,),
    "norm2.weight": (_D,),
    "q_norm.weight": (_HD,),
    "k_norm.weight": (_HD,),
    "qkv.weight": (_D, 3 * _D),
    "qkv.bias": (3 * _D,),
    "proj.weight": (_D, _D),
    "proj.bias": (_D,),
    "ffn_gate.weight": (_D, _F),
    "ffn_gate.bias": (_F,),
    "ffn_up.weight": (_D, _F),
    "ffn_up.bias": (_F,),
    "ffn_down.weight": (_F, _D),
    "ffn_down.bias": (_D,),
}
_BLOCK = {
    "norm1.weight": (_D,),
    "norm2.weight": (_D,),
    "q_norm.weight": (_HD,),
    "k_norm.weight": (_HD,),
    "proj.q.weight": (_D, _D),
    "proj.k.weight": (_D, _D),
    "proj.v.weight": (_D, _D),
    "proj.o.weight": (_D, _D),
    "proj.ffn_up_gate.weight": (_D, 2 * _F),
    "proj.ffn_down.weight": (_F, _D),
}
BLOCK_GROUPS = (("layers", _DEPTH, _BLOCK),
                ("text_embedder.blocks", _TEXT_DEPTH, _TEXT_BLOCK))
_GROUP = re.compile(r"(layers|text_embedder\.blocks)\.(\d+)\.")

CONSTANTS = ConstantsRow(
    values={
        "rope_base": 10000.0,
        "norm_eps": 1e-06,
        # The base of the timestep sinusoid's frequencies.
        "time_period": 10000.0,
        "num_train_timesteps": 1000,
        # The pipeline's image processor: 8 pixels a VAE latent, 2 a patch.
        "vae_scale_factor": 16,
        "size_multiple": 16,
    },
    repo="OpenMOSS-Team/MOSS-Image-SFT",
    revision="fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
)

IDENTITY_TENSORS = ("video_embedder.bias", "final_norm_video.weight",
                    "time_embedding.0.bias")
_QUANTS = ("Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S",
           "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0")
_REPO = "realrebelai/MOSS-Image_GGUFs"
_REVISION = "4f21f583c9213eb2230a09e9b33bfa86605aecf2"
_SFT = VariantRow("sft", _REPO, _REVISION,
                  tuple(f"MOSS-Image-SFT-{q}.gguf" for q in _QUANTS))
_BASE = VariantRow("base", _REPO, _REVISION,
                   tuple(f"MOSS-Image-Base-{q}.gguf" for q in _QUANTS))
VARIANTS = {
    "96b09d61487460c3f321abd04a71a2ca989025150226ca1c30be7a57c0d86c60": _SFT,
    "22e89962212b91739f15827592163257e8717b26870e57d446e48846259c2626": _BASE,
}


def expected() -> dict:
    """Every tensor name of the file and its shape."""
    out = dict(_TOP)
    for group, count, block in BLOCK_GROUPS:
        for i in range(count):
            out.update({f"{group}.{i}.{n}": s for n, s in block.items()})
    return out


def _past_count(name: str, view: HeaderView) -> bool:
    """Whether a block tensor sits in a block the counts already name."""
    m = _GROUP.match(name)
    if m is None:
        return False
    want = {g: c for g, c, _ in BLOCK_GROUPS}[m[1]]
    return int(m[2]) >= min(want, view.count(m[1]))


def _other_family(view: HeaderView) -> Refusal | None:
    """The refusal of a file that another family takes or refuses, by
    that family's name, for a file given as MOSS-Image."""
    from gmlx.image.fingerprint import strip_prefix
    from gmlx.image.registry import families

    for fam in families():
        if fam.name == FAMILY.name:
            continue
        try:
            got = fam.fingerprint(strip_prefix(view, fam.strip_prefixes))
        except ValueError:
            continue
        if isinstance(got, Refusal):
            return got
        if got is not None:
            return Refusal(f"this is a {fam.title} GGUF, not a {TITLE} one")
    return None


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    if not (view.has("shared_block_modulation.weight")
            and view.has("video_embedder.weight")):
        return _other_family(view)
    want = expected()
    diffs = [f"{g} has {view.count(g)} blocks, {TITLE} has {c}"
             for g, c, _ in BLOCK_GROUPS if view.count(g) != c]
    diffs += [f"{n} is {view.shape(n)}, {TITLE} has {s}"
              for n, s in want.items()
              if view.has(n) and view.shape(n) != s]
    diffs += [f"{n} is missing" for n in want
              if not view.has(n) and not _past_count(n, view)]
    diffs += [f"{n} is not a {TITLE} tensor" for n in view.tensors
              if n not in want and not _past_count(n, view)]
    if diffs:
        shown = ", ".join(diffs[:8])
        more = f" (+{len(diffs) - 8} more)" if len(diffs) > 8 else ""
        return Refusal(f"a {TITLE} relative that gmlx image does not "
                       f"support: {shown}{more}")
    return Match()


def synth_config(view: HeaderView) -> dict:
    """The keys of the official transformer/config.json, from shapes and
    constants, plus the derived values the module tree needs."""
    c = CONSTANTS.values
    s = view.dims
    latent, dim = s("video_embedder.weight")
    head_dim = s("layers.0.q_norm.weight")[0]
    return {
        "dim": dim,
        "ffn_dim": s("layers.0.proj.ffn_down.weight")[0],
        "freq_dim": s("time_embedding.0.weight")[0],
        "latent_channels": latent,
        "num_heads": dim // head_dim,
        "num_layers": view.count("layers"),
        "rope_base": c["rope_base"],
        "text_dim": s("text_embedder.proj.weight")[0],
        "head_dim": head_dim,
        "num_bands": s("rope.bands")[0],
        "num_text_blocks": view.count("text_embedder.blocks"),
        "norm_eps": c["norm_eps"],
        "time_period": c["time_period"],
        "num_train_timesteps": c["num_train_timesteps"],
    }


def build(config: dict):
    from .model import MossImageDiT

    return MossImageDiT(config)


FAMILY = Family(
    name="moss_image",
    title=TITLE,
    order=40,
    arch_strings=frozenset({"moss_image", None}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=build,
    # The release builds the bands from constants and ignores the stored
    # ones; the file's copy keeps its float32 values.
    fp32_keep=("rope.bands",),
    identity_tensors=IDENTITY_TENSORS,
    variants=VARIANTS,
)
