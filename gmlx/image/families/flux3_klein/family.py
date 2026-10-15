# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 Black Forest Labs, Inc.
# Ported from BFL's FLUX.3 klein reference at commit 45e57c9: the
# constants of src/flux3/dit.py, src/flux3/pipeline.py and
# src/flux3/text_encoder.py, and the tensor names and shapes of the DiT of
# src/flux3/dit.py, and modified for gmlx (Apache-2.0; see
# licenses/flux-3-image-klein-LICENSE).
"""FLUX.3 klein fingerprint, constants, config synthesis and the hooks of
its registry row.

Shapes are GGUF order, input dim first. The GGUF tensor names are the
state dict keys of the reference DiT. Converters label the file ``flux``,
as they label FLUX.1 and FLUX.2 files, so the names and shapes decide,
and the label alone names no family. The distilled checkpoint, a flow
map, has the ``ref_time_in`` embedder of the target time and Base has
none, so the tensors name the variant. A FLUX.1 or FLUX.2 file is
refused with the name of its family.
"""

from __future__ import annotations

# The constants are AXES_DIM, THETA and the eps of the norms in
# src/flux3/dit.py, IMAGE_ROPE_SCALE in src/flux3/pipeline.py, and
# OUTPUT_LAYERS, PAD_MULTIPLE, MAX_LENGTH, the pad token, the pad position
# and the chat template of one user turn in src/flux3/text_encoder.py. The
# official config.json and transformer/config.json give the same DiT
# values.

import re

from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import ConstantsRow, Family

TITLE = "FLUX.3 klein"
# The input modes in the order their tokens take inside a sample, and the
# key of each mode's modulations.
MODES = ("image", "image_cond", "text")
MOD_KEYS = {"image": "image", "image_cond": "image_cond", "text": "txt"}
# Each mode's own blocks, then the shared ones.
BLOCK_GROUPS = (("content_mode_blocks.image", 5),
                ("content_mode_blocks.image_cond", 5),
                ("txt_mode_blocks", 5), ("single_blocks", 28))
IMAGE_Q = "content_mode_blocks.image.0.q_proj.weight"
TXT_IN = "txt_in.weight"
REF_TIME = ("ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight")

_HIDDEN, _MLP, _HEAD, _T, _LATENT, _TEXT = 3072, 9216, 128, 256, 128, 20480
_BLOCK = (("q_proj.weight", (_HIDDEN, _HIDDEN)),
          ("k_proj.weight", (_HIDDEN, _HIDDEN)),
          ("v_proj.weight", (_HIDDEN, _HIDDEN)),
          ("attn_out.weight", (_HIDDEN, _HIDDEN)),
          ("mlp_in.weight", (_HIDDEN, 2 * _MLP)),
          ("mlp_out.weight", (_MLP, _HIDDEN)),
          ("norm.query_norm.scale", (_HEAD,)),
          ("norm.key_norm.scale", (_HEAD,)))
_TOP = (("emb_in.image.weight", (_LATENT, _HIDDEN)),
        ("emb_in.image_cond.weight", (_LATENT, _HIDDEN)),
        (TXT_IN, (_TEXT, _HIDDEN)),
        ("time_in.in_layer.weight", (_T, _HIDDEN)),
        ("time_in.out_layer.weight", (_HIDDEN, _HIDDEN)),
        *((f"{g}.{k}.lin.weight", (_HIDDEN, 3 * _HIDDEN))
          for g in ("early_stream_modulations", "single_stream_modulations")
          for k in MOD_KEYS.values()),
        ("final_layer.image.linear.weight", (_HIDDEN, _LATENT)),
        ("final_layer.image.adaLN_modulation.1.weight",
         (_HIDDEN, 2 * _HIDDEN)))
_FLOW_MAP = ((REF_TIME[0], (_T, _HIDDEN)), (REF_TIME[1], (_HIDDEN, _HIDDEN)))
_GROUP = re.compile(r"(content_mode_blocks\.image_cond|"
                    r"content_mode_blocks\.image|txt_mode_blocks|"
                    r"single_blocks)\.(\d+)\.")

CONSTANTS = ConstantsRow(
    values={
        "axes_dims_rope": [32, 32, 32, 32],
        "rope_theta": 10000,
        # The eps of every norm: the LayerNorm of the modulation and the
        # RMSNorm of q and k.
        "eps": 1e-06,
        # The factor of the row and column ids of image tokens.
        "image_rope_scale": 0.5,
        # A latent pixel is one 128-channel token: the VAE's 8x and its
        # 2x2 patchify. The DiT takes the tokens with no patch of its own.
        "vae_scale_factor": 16,
        "size_multiple": 16,
        "patch_size": 1,
        "out_modes": ["image"],
    },
    repo="black-forest-labs/flux-3-image-klein",
    revision="8454017f9ebaada30c998fccd9f67a80c7580f68",
)

# The encoder of the reference: Qwen3-VL-4B-Instruct, the hidden states
# of eight layers stacked per token, the prompt in the chat template and
# right-padded to a multiple of 80 tokens, at most 4096. The official
# config.json says max_tokens 8192, but the reference text_encoder.py
# truncates and pads at MAX_LENGTH 4096 and no reference code reads that
# key, so gmlx follows the code.
PROMPT = ConstantsRow(
    values={
        # What the Qwen3-VL chat template renders for one user turn with
        # the generation prompt.
        "template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        # hidden_states indices of the encoder, 0 being the embeddings.
        "tap_layers": (4, 8, 12, 16, 20, 24, 28, 32),
        "pad_multiple": 80,
        "max_tokens": 4096,
        # The tokenizer's pad token <|endoftext|>, and the position the
        # reference gives every pad row: its masked cumsum minus 1 with
        # the pads filled with 1.
        "pad_id": 151643,
        "pad_position": 1,
    },
    repo="github.com/black-forest-labs/flux-3-image-klein",
    revision="45e57c9fb06f9751cdced29a65c5d2171003be05",
)

# Norm tensors that identify the encoder weights: GGUF name, path in the
# loaded model and sha256 of the bf16 bits. The Qwen3-VL-4B-Instruct
# Q8_0 GGUF of unsloth/Qwen3-VL-4B-Instruct-GGUF, which Krea 2 reads too,
# holds the weights of the reference's text_encoder: these norms equal its
# bf16 tensors bit for bit, and its Q8_0 blocks equal its bf16 matrices
# quantized to Q8_0.
ENCODER_IDENTITY = (
    ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight",
     "a06277a2724034a966ccb331fdd94d5bbff4b3a1e9dd6144fa9a1315a0550034"),
    ("blk.17.ffn_norm.weight",
     "model.layers.17.post_attention_layernorm.weight",
     "34fbce23a08e85df36bd37ddb84c82d2c2dc526d46c1ca89fe45369d8d8adbe8"),
    ("output_norm.weight", "model.norm.weight",
     "59be4927491b995e61ca021ca3d1bddf90bed0f522ec73e9311ebfb5fb8bd9e8"),
)
ENCODER_IDENTITY_SOURCE = (
    "black-forest-labs/flux-3-image-klein text_encoder at "
    "8454017f9ebaada30c998fccd9f67a80c7580f68")

# FLUX.1 and FLUX.2 files under the same label, by the block names of
# BFL's flux and flux2 code. FLUX.2 klein takes its text width from the
# stacked layers of its Qwen3 encoder.
_FLUX_DOUBLE = "double_blocks.0.img_attn.proj.weight"
_FLUX2_MOD = "double_stream_modulation_img.lin.weight"
_FLUX1_MOD = "double_blocks.0.img_mod.lin.weight"
_FLUX2_KLEIN = {((128, 3072), (7680, 3072)): "FLUX.2 klein 4B",
                ((128, 4096), (12288, 4096)): "FLUX.2 klein 9B"}


def _older_flux(view: HeaderView) -> Refusal | None:
    """The refusal of a FLUX.1 or FLUX.2 file, by name, or None."""
    if not view.has(_FLUX_DOUBLE):
        return None
    shapes = (view.shape("img_in.weight"), view.shape(TXT_IN))
    if view.has(_FLUX2_MOD):
        name = _FLUX2_KLEIN.get(shapes, "FLUX.2")
    elif view.has(_FLUX1_MOD) and shapes == ((64, 3072), (4096, 3072)):
        name = "FLUX.1"
    else:
        # Another model in the FLUX.1 block names, which the generic
        # FLUX row of the fingerprint refuses.
        return None
    return Refusal(f"this is a {name} GGUF, which gmlx image does not "
                   f"support; {TITLE} is the FLUX model it runs")


def expected(flow_map: bool) -> dict:
    """Every tensor name of the file and its shape."""
    out = dict(_TOP)
    for group, count in BLOCK_GROUPS:
        for i in range(count):
            out.update({f"{group}.{i}.{n}": s for n, s in _BLOCK})
    if flow_map:
        out.update(_FLOW_MAP)
    return out


def _past_count(name: str, view: HeaderView) -> bool:
    """Whether a block tensor sits in a block the counts already name:
    past the file's count of its group, or past the model's."""
    m = _GROUP.match(name)
    if m is None:
        return False
    want = dict(BLOCK_GROUPS)[m[1]]
    return int(m[2]) >= min(want, view.count(m[1]))


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    older = _older_flux(view)
    if older is not None:
        return older
    if not (view.has(IMAGE_Q) and view.has(TXT_IN)):
        return None
    refs = sum(view.has(n) for n in REF_TIME)
    want = expected(flow_map=refs > 0)
    diffs = [f"{g} has {view.count(g)} blocks, {TITLE} has {c}"
             for g, c in BLOCK_GROUPS if view.count(g) != c]
    if refs == 1:
        diffs.append("it has one of ref_time_in.in_layer and "
                     "ref_time_in.out_layer")
    diffs += [f"{n} is {view.shape(n)}, {TITLE} has {s}"
              for n, s in want.items()
              if view.has(n) and view.shape(n) != s]
    diffs += [f"{n} is missing" for n in want
              if not view.has(n) and not _past_count(n, view)]
    extra = [n for n in view.tensors
             if n not in want and not _past_count(n, view)]
    if any(n.endswith(".bias") for n in extra):
        diffs.append("it has bias tensors")
    diffs += [f"{n} is not a {TITLE} tensor" for n in extra
              if not n.endswith(".bias")]
    if diffs:
        shown = ", ".join(diffs[:8])
        more = f" (+{len(diffs) - 8} more)" if len(diffs) > 8 else ""
        return Refusal(f"a {TITLE} relative that gmlx image does not "
                       f"support: {shown}{more}")
    return Match(variant="distilled" if refs else "base")


def synth_config(view: HeaderView) -> dict:
    """The keys of the official transformer/config.json, from shapes and
    constants, with the ``dit`` block of the official config.json, which
    BFL's code reads, under ``dit``."""
    c = CONSTANTS.values
    s = view.dims
    in_ch, hidden = s("emb_in.image.weight")
    cond_ch = s("emb_in.image_cond.weight")[0]
    text = s(TXT_IN)[0]
    head_dim = s("single_blocks.0.norm.query_norm.scale")[0]
    heads = hidden // head_dim
    mlp_ratio = s("single_blocks.0.mlp_out.weight")[0] / hidden
    depth = view.count("content_mode_blocks.image")
    single = view.count("single_blocks")
    flow_map = view.has(REF_TIME[0])
    out_modes = [m for m in MODES
                 if view.has(f"final_layer.{m}.linear.weight")]
    return {
        "attention_head_dim": head_dim,
        "axes_dims_rope": list(c["axes_dims_rope"]),
        "cond_in_channels": cond_ch,
        "eps": c["eps"],
        "flow_map": flow_map,
        "in_channels": in_ch,
        "joint_attention_dim": text,
        "mlp_ratio": mlp_ratio,
        "num_attention_heads": heads,
        "num_layers": depth,
        "num_single_layers": single,
        "rope_theta": c["rope_theta"],
        "timestep_channels": s("time_in.in_layer.weight")[0],
        "dit": {
            "hidden": hidden,
            "heads": heads,
            "depth": depth,
            "depth_single": single,
            "mlp_ratio": mlp_ratio,
            "axes_dim": list(c["axes_dims_rope"]),
            "theta": c["rope_theta"],
            "in_channels": {"image": in_ch, "image_cond": cond_ch,
                            "text": text},
            "out_modes": out_modes,
            # No vector input: the fingerprint takes no other tensor.
            "vec_in_dim": None,
            "image_rope_scale": c["image_rope_scale"],
            "flow_map": flow_map,
        },
    }


def build(config: dict):
    from .model import Flux3KleinDiT

    return Flux3KleinDiT(config)


def encoder():
    """The text encoder of the family: Qwen3-VL-4B read as a text model,
    after eight of its layers and before its final norm, on the chat
    template of one user turn. The eight states of a token are its
    features side by side, layer by layer, and the pad rows the reference
    keeps follow the prompt's rows. The tap runs in float32."""
    from gmlx.image.text_encoder import EncoderSpec, IdentityTensor

    p = PROMPT.values
    return EncoderSpec(
        name="qwen3-vl-4b",
        arch="qwen3vl",
        model_type="qwen3",
        hidden=2560,
        layers=36,
        tap_layers=tuple(p["tap_layers"]),
        template=p["template"],
        max_tokens=p["max_tokens"],
        stack=True,
        identity=tuple(IdentityTensor(*row) for row in ENCODER_IDENTITY),
        identity_source=ENCODER_IDENTITY_SOURCE,
        activation="float32",
        pad_multiple=p["pad_multiple"],
        pad_id=p["pad_id"],
        pad_position=p["pad_position"],
    )


FAMILY = Family(
    name="flux3_klein",
    title=TITLE,
    order=30,
    # Every FLUX GGUF carries the label flux, so it names no family alone.
    arch_strings=frozenset({None}),
    shared_arch_strings=frozenset({"flux"}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=build,
    # The q and k norms run in fp32, as the reference's do.
    fp32_keep=("norm.query_norm.scale", "norm.key_norm.scale"),
    encoder=encoder,
)
