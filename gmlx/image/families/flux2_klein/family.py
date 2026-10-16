# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Black Forest Labs, Inc.
# Ported from BFL's FLUX.2 reference at commit 50fe516: the constants of
# Klein4BParams, Klein9BParams and Flux2Params in src/flux2/model.py and
# the defaults of FLUX2_MODEL_INFO in src/flux2/util.py, the tensor names
# and shapes of Flux2 in src/flux2/model.py, the prompt rules of
# Qwen3Embedder in src/flux2/text_encoder.py, the rounding of denoise
# and denoise_cfg and the reference image rules of encode_image_refs and
# cap_min_pixels in src/flux2/sampling.py, and modified for gmlx
# (Apache-2.0; see licenses/flux2-LICENSE).
"""FLUX.2 klein fingerprint, constants, config synthesis and the hooks of
its registry row.

Shapes are GGUF order, input dim first. The GGUF tensor names are the
state dict keys of the reference DiT. Converters label the file ``flux``,
as they label FLUX.1 and FLUX.3 files, so the names and shapes decide,
and the label alone names no family. The 4B and 9B sizes differ in their
widths and block counts. The distilled and Base checkpoints of a size
share every name and shape, so the variant comes from the identity hash.
A FLUX.1 or FLUX.2 dev file is refused with the name of its model.
"""

from __future__ import annotations

import re

from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import (ConstantsRow, EditParts, Family,
                                 LoopRules, SamplingRow, VariantRow)

TITLE = "FLUX.2 klein"
# What a refusal of another FLUX model names as the ones gmlx image runs.
RUNS = "FLUX.2 klein and FLUX.3 klein are the FLUX models it runs"

# The block names of BFL's flux and flux2 code, and the modulations that
# tell them apart.
FLUX_DOUBLE = "double_blocks.0.img_attn.proj.weight"
_FLUX2_MOD = "double_stream_modulation_img.lin.weight"
_FLUX1_MOD = "double_blocks.0.img_mod.lin.weight"
IMG_IN = "img_in.weight"
TXT_IN = "txt_in.weight"
_Q_NORM = "double_blocks.0.img_attn.norm.query_norm.scale"

# hidden_size, depth, depth_single_blocks and context_in_dim of
# Klein4BParams and Klein9BParams. The text width is that of three
# stacked layers of the Qwen3 encoder of each size.
SIZES = {"4B": (3072, 5, 20, 7680), "9B": (4096, 8, 24, 12288)}
# The sizes gmlx image runs. A 9B file is told apart and refused, since
# the encoder and the memory rows gmlx image has are the 4B's.
RUN_SIZES = ("4B",)
_HEAD, _LATENT, _T, _MLP_RATIO = 128, 128, 256, 3
# (img_in, txt_in) of FLUX.2 dev (Flux2Params) and of FLUX.1.
_FLUX2_DEV = ((128, 6144), (15360, 6144))
_FLUX1 = ((64, 3072), (4096, 3072))
_GROUP = re.compile(r"(double_blocks|single_blocks)\.(\d+)\.")

CONSTANTS = ConstantsRow(
    values={
        "axes_dims_rope": [32, 32, 32, 32],
        "rope_theta": 2000,
        # The eps of every norm: the LayerNorm of the modulation and the
        # RMSNorm of q and k.
        "eps": 1e-06,
        # A latent pixel is one 128-channel token: the VAE's 8x and its
        # 2x2 patchify. The DiT takes the tokens with no patch of its own.
        "vae_scale_factor": 16,
        "size_multiple": 16,
        "patch_size": 1,
    },
    repo="black-forest-labs/FLUX.2-klein-4B",
    revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
)

# The defaults of FLUX2_MODEL_INFO in src/flux2/util.py: the distilled
# checkpoints at 4 steps and no guidance, Base at 50 steps with guidance
# 4.0, which is gmlx's cfg, as the reference guides by u + guidance *
# (c - u). The schedule is the family's own, from the image's token count
# and the steps, so the shift field is unused.
_BFL = ("github.com/black-forest-labs/flux2",
        "50fe5162777813d869182b139e83b10743caef15")
SAMPLING = {
    "distilled": SamplingRow(4, 1.0, 1.0, *_BFL),
    "base": SamplingRow(50, 4.0, 1.0, *_BFL),
}

# The loop of denoise and denoise_cfg in src/flux2/sampling.py, on the bf16
# noise of scripts/cli.py: the latents, the velocity, its guidance and the
# step's product and sum all in bf16.
LOOP = LoopRules(noise="bfloat16", act_update=True, act_latents=True)

# The variants of a size share every shape, so the variant comes from the
# hash of the image input projection, which the distillation changed.
# Every GGUF of the unsloth repos keeps it BF16.
IDENTITY_TENSORS = (IMG_IN,)
_QUANTS = ("BF16", "F16", "Q2_K", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1",
           "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_1", "Q5_K_M", "Q5_K_S", "Q6_K",
           "Q8_0")
_DISTILLED_4B = VariantRow(
    "distilled", "unsloth/FLUX.2-klein-4B-GGUF",
    "0084d1df98e2e2137fe776d55170bc4792ec1d66",
    tuple(f"flux-2-klein-4b-{q}.gguf" for q in _QUANTS))
_BASE_4B = VariantRow(
    "base", "unsloth/FLUX.2-klein-base-4B-GGUF",
    "5fd6513800b1f13016fab83600b392a35f29d56d",
    tuple(f"flux-2-klein-base-4b-{q}.gguf" for q in _QUANTS))
VARIANTS = {
    "591c63d49ea6d7cb55c1d349febb8736613013aef145462659f19816ed3639a9":
        _DISTILLED_4B,
    "2a60e24acf60f890172dfdf3a887d7f9dc51675c515a566126bb24ddab3d2fce":
        _BASE_4B,
}


# The encoder of Qwen3Embedder in src/flux2/text_encoder.py: Qwen3-4B, the
# hidden states of three layers stacked per token, the prompt in the chat
# template of one user turn with thinking off, right-padded with the pad
# token to 512 rows and cut there, and the pad rows kept. The reference
# masks the pads as keys and gives each row its index as its position.
PROMPT = ConstantsRow(
    values={
        # What the Qwen3 chat template renders for one user turn with the
        # generation prompt and enable_thinking False.
        "template": ("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
                     "<think>\n\n</think>\n\n"),
        # hidden_states indices of the encoder, 0 being the embeddings.
        "tap_layers": (9, 18, 27),
        "max_tokens": 512,
        # The tokenizer's pad token <|endoftext|>.
        "pad_id": 151643,
    },
    repo="github.com/black-forest-labs/flux2",
    revision="50fe5162777813d869182b139e83b10743caef15",
)

# The Qwen3-4B Q8_0 GGUF of unsloth/Qwen3-4B-GGUF, which Z-Image reads too,
# holds the weights of the 4B text_encoder of the official repository: its
# F32 norms equal the bf16 tensors there bit for bit, and the Q8_0 blocks
# of the matrices compared equal those bf16 matrices quantized to Q8_0. So
# the identity rows are Z-Image's.
ENCODER_IDENTITY_SOURCE = (
    "black-forest-labs/FLUX.2-klein-4B text_encoder at "
    "e7b7dc27f91deacad38e78976d1f2b499d76a294")


def encoder():
    """The text encoder of the 4B checkpoints: Qwen3-4B after three of its
    layers and before its final norm, on the chat template of one user
    turn with thinking off. The three states of a token are its features
    side by side, layer by layer, and the 512 rows hold the pad rows the
    reference keeps, each at its own position.

    The residual stream and the norms run in float32, and q, k and v are
    cast to float32 after their products, so the qk norm, the rotary
    embedding and the attention run in float32 too. The products of a
    quantized file return bfloat16: the attention's output product and
    the feed-forward, its activation included, return bfloat16 to the
    float32 residual."""
    from gmlx.image.text_encoder import Z_IMAGE, EncoderSpec

    p = PROMPT.values
    return EncoderSpec(
        name="qwen3-4b",
        arch="qwen3",
        model_type="qwen3",
        hidden=2560,
        layers=36,
        tap_layers=tuple(p["tap_layers"]),
        template=p["template"],
        max_tokens=p["max_tokens"],
        stack=True,
        identity=Z_IMAGE.identity,
        identity_source=ENCODER_IDENTITY_SOURCE,
        activation="float32",
        pad_multiple=p["max_tokens"],
        pad_id=p["pad_id"],
        pad_position=None,
        fp32_qkv=True,
    )


def size_of(view: HeaderView) -> str | None:
    """The klein size whose input widths the file has, or None."""
    shapes = (view.shape(IMG_IN), view.shape(TXT_IN))
    for size, (hidden, _, _, text) in SIZES.items():
        if shapes == ((_LATENT, hidden), (text, hidden)):
            return size
    return None


def flux_model(view: HeaderView) -> str | None:
    """The FLUX model of a file in the block names of BFL's flux and flux2
    code: FLUX.1, FLUX.2 dev, FLUX.2 klein 4B or 9B, or FLUX.2 for another
    FLUX.2 model. None for any other file, such as another model in the
    FLUX.1 block names, which the generic FLUX row of the fingerprint
    refuses."""
    if not view.has(FLUX_DOUBLE):
        return None
    shapes = (view.shape(IMG_IN), view.shape(TXT_IN))
    if view.has(_FLUX2_MOD):
        size = size_of(view)
        if size is not None:
            return f"{TITLE} {size}"
        return "FLUX.2 dev" if shapes == _FLUX2_DEV else "FLUX.2"
    if view.has(_FLUX1_MOD) and shapes == _FLUX1:
        return "FLUX.1"
    return None


def older_refusal(name: str) -> Refusal:
    """The refusal of a FLUX model gmlx image does not run, by name."""
    return Refusal(f"this is a {name} GGUF, which gmlx image does not "
                   f"support; {RUNS}")


def expected(size: str) -> dict:
    """Every tensor name of a file of ``size`` and its shape."""
    hidden, depth, single, text = SIZES[size]
    mlp = _MLP_RATIO * hidden
    norms = {"norm.query_norm.scale": (_HEAD,),
             "norm.key_norm.scale": (_HEAD,)}
    stream = {"attn.qkv.weight": (hidden, 3 * hidden),
              "attn.proj.weight": (hidden, hidden),
              **{f"attn.{n}": s for n, s in norms.items()},
              "mlp.0.weight": (hidden, 2 * mlp),
              "mlp.2.weight": (mlp, hidden)}
    out = {IMG_IN: (_LATENT, hidden),
           TXT_IN: (text, hidden),
           "time_in.in_layer.weight": (_T, hidden),
           "time_in.out_layer.weight": (hidden, hidden),
           "double_stream_modulation_img.lin.weight": (hidden, 6 * hidden),
           "double_stream_modulation_txt.lin.weight": (hidden, 6 * hidden),
           "single_stream_modulation.lin.weight": (hidden, 3 * hidden),
           "final_layer.adaLN_modulation.1.weight": (hidden, 2 * hidden),
           "final_layer.linear.weight": (hidden, _LATENT)}
    for i in range(depth):
        for kind in ("img", "txt"):
            out.update({f"double_blocks.{i}.{kind}_{n}": s
                        for n, s in stream.items()})
    for i in range(single):
        out.update({f"single_blocks.{i}.linear1.weight":
                    (hidden, 3 * hidden + 2 * mlp),
                    f"single_blocks.{i}.linear2.weight":
                    (hidden + mlp, hidden),
                    **{f"single_blocks.{i}.{n}": s
                       for n, s in norms.items()}})
    return out


def _counts(size: str) -> tuple:
    _, depth, single, _ = SIZES[size]
    return (("double_blocks", depth), ("single_blocks", single))


def _past_count(name: str, view: HeaderView, size: str) -> bool:
    """Whether a block tensor sits in a block the counts already name:
    past the file's count of its group, or past the model's."""
    m = _GROUP.match(name)
    if m is None:
        return False
    want = dict(_counts(size))[m[1]]
    return int(m[2]) >= min(want, view.count(m[1]))


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    name = flux_model(view)
    if name is None:
        return None
    size = size_of(view) if view.has(_FLUX2_MOD) else None
    if size is None:
        return older_refusal(name)
    want = expected(size)
    title = f"{TITLE} {size}"
    diffs = [f"{g} has {view.count(g)} blocks, {title} has {c}"
             for g, c in _counts(size) if view.count(g) != c]
    diffs += [f"{n} is {view.shape(n)}, {title} has {s}"
              for n, s in want.items()
              if view.has(n) and view.shape(n) != s]
    diffs += [f"{n} is missing" for n in want
              if not view.has(n) and not _past_count(n, view, size)]
    extra = [n for n in view.tensors
             if n not in want and not _past_count(n, view, size)]
    if any(n.endswith(".bias") for n in extra):
        diffs.append("it has bias tensors")
    diffs += [f"{n} is not a {TITLE} tensor" for n in extra
              if not n.endswith(".bias")]
    if diffs:
        shown = ", ".join(diffs[:8])
        more = f" (+{len(diffs) - 8} more)" if len(diffs) > 8 else ""
        return Refusal(f"a {TITLE} relative that gmlx image does not "
                       f"support: {shown}{more}")
    if size not in RUN_SIZES:
        return Refusal(f"{title} is not supported yet. gmlx image runs "
                       f"{TITLE} 4B and base 4B.")
    return Match()


def schedule(steps: int, *, tokens: int | None, variant: str | None = None):
    """The sigmas of get_schedule for an image of ``tokens`` latent tokens,
    for either variant."""
    from gmlx.image.schedule import flux2_sigmas

    if tokens is None:
        raise ValueError(f"the {TITLE} schedule depends on the image token "
                         f"count; pass tokens")
    return flux2_sigmas(steps, tokens)


def build(config: dict):
    from .model import Flux2KleinDiT

    return Flux2KleinDiT(config)


def synth_config(view: HeaderView) -> dict:
    """The keys of the official transformer/config.json, from shapes and
    constants."""
    c = CONSTANTS.values
    s = view.dims
    in_ch, hidden = s(IMG_IN)
    head_dim = s(_Q_NORM)[0]
    out_ch = s("final_layer.linear.weight")[1]
    return {
        "attention_head_dim": head_dim,
        "axes_dims_rope": list(c["axes_dims_rope"]),
        "eps": c["eps"],
        # Neither size embeds a guidance: the fingerprint takes no
        # guidance_in.
        "guidance_embeds": False,
        "in_channels": in_ch,
        "joint_attention_dim": s(TXT_IN)[0],
        "mlp_ratio": s("double_blocks.0.img_mlp.2.weight")[0] / hidden,
        "num_attention_heads": hidden // head_dim,
        "num_layers": view.count("double_blocks"),
        "num_single_layers": view.count("single_blocks"),
        # The official config says null for the input width.
        "out_channels": None if out_ch == in_ch else out_ch,
        "patch_size": c["patch_size"],
        "rope_theta": c["rope_theta"],
        "timestep_guidance_channels": s("time_in.in_layer.weight")[0],
    }


# The reference images of an edit, by encode_image_refs and default_prep
# in src/flux2/sampling.py: sides of 64 pixels at least and an aspect
# ratio of 8 at most, one reference under 2024 x 2024 pixels and each of
# several under 1024 x 1024, cropped to the grid of one latent token. The
# reference sets no count; gmlx takes 10, as FLUX.3 klein's does. The
# encoder reads the prompt alone, and the references reach the DiT as
# latents only.
REFERENCES = ConstantsRow(
    values={
        "max_refs": 10,
        "one_max_pixels": 2024 ** 2,
        "multi_max_pixels": 1024 ** 2,
        "min_side": 64,
        "max_aspect": 8,
        "grid": 16,
    },
    repo="github.com/black-forest-labs/flux2",
    revision="50fe5162777813d869182b139e83b10743caef15",
)


def _prepare_all(images):
    from .edit import prepare

    return prepare(images)


def _size(refs):
    from .edit import default_size

    return default_size(refs)


FAMILY = Family(
    name="flux2_klein",
    title=TITLE,
    # Before FLUX.3 klein, so a FLUX.2 klein file and the other FLUX models
    # meet this row first.
    order=25,
    # Every FLUX GGUF carries the label flux, so it names no family alone.
    arch_strings=frozenset({None}),
    shared_arch_strings=frozenset({"flux"}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=build,
    # The q and k norm scales, F32 in the file, stay fp32: the norm then
    # runs in fp32 and rounds once with the rotation.
    fp32_keep=("norm.query_norm.scale", "norm.key_norm.scale"),
    encoder=encoder,
    identity_tensors=IDENTITY_TENSORS,
    variants=VARIANTS,
    sampling=SAMPLING,
    schedule=schedule,
    loop=LOOP,
    edit=EditParts(max_refs=REFERENCES.values["max_refs"],
                   prepare_all=_prepare_all, size=_size),
)
