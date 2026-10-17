"""Krea 2 fingerprint, constants, config synthesis and the hooks of its
registry row.

Shapes are GGUF order, input dim first. Two layouts of one model exist.
One converter writes arch ``qwen_image`` with two extra matrices,
``last.up`` and ``last.down``, that the reference model never reads, and
another writes arch ``krea2`` with or without them and stores
``txtfusion.projector.weight`` flattened to 12 values, with its shape in
``comfy.gguf.orig_shape``. The rule decides by names and shapes, and the
layout says whether the two extra matrices are there, so the module tree
names every tensor the file has.

Turbo and Raw share every shape, so the variant comes from the hash of
the text layer projector and the bias of the patch embedder, which every
converter keeps in float32. gmlx image runs Turbo and refuses Raw, whose
scheduler config is not published.
"""

from __future__ import annotations

# The prompt template, the encoder layers, the timestep shift of each
# variant and the scheduler values are those of diffusers'
# src/diffusers/pipelines/krea2/pipeline_krea2.py and
# src/diffusers/modular_pipelines/krea2/ at commit e0abab83 (Apache-2.0; see
# licenses/diffusers-LICENSE). The upsampler's system prompt is
# docs/expansion.txt of github.com/krea-ai/krea-2 at commit db3984f, with
# its use in docs/prompting.md (Apache-2.0; see licenses/krea-2-LICENSE).

from gmlx.image import upsample
from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import (ConstantsRow, Family, SamplingRow,
                                 UpsampleParts, VariantRow)

_B0 = "blocks.0."
_T0 = "txtfusion.layerwise_blocks.0."

_REQUIRED = (
    ("first.weight", (64, 6144)),
    (_B0 + "attn.wq.weight", (6144, 6144)),
    (_B0 + "attn.wk.weight", (6144, 1536)),
    (_B0 + "attn.wv.weight", (6144, 1536)),
    (_B0 + "attn.gate.weight", (6144, 6144)),
    (_B0 + "attn.qknorm.qnorm.scale", (128,)),
    (_B0 + "mlp.gate.weight", (6144, 16384)),
    (_B0 + "mod.lin", (36864,)),
    ("tmlp.0.weight", (256, 6144)),
    ("tproj.1.weight", (6144, 36864)),
    ("txtmlp.1.weight", (2560, 6144)),
    ("last.linear.weight", (6144, 64)),
    ("last.modulation.lin", (6144, 2)),
    (_T0 + "prenorm.scale", (2560,)),
)
_COUNTS = (("blocks", 28), ("txtfusion.layerwise_blocks", 2),
           ("txtfusion.refiner_blocks", 2))
# The projector over the encoder layers: (layers, 1), or the layers alone
# in a file that flattened it.
PROJECTOR = "txtfusion.projector.weight"
# The two matrices of the final layer that one converter writes and the
# reference model does not read.
LAST_MLP = ("last.up.weight", "last.down.weight")

# The transformer's config values that no tensor shape determines: the
# defaults of Krea2Transformer2DModel, whose rotary axes ComfyUI derives
# from the head dim the same way, and the patch of two latent pixels.
# vae_scale_factor is the original Qwen-Image VAE's.
CONSTANTS = ConstantsRow(
    values={
        "axes_dims_rope": [32, 48, 48],
        "rope_theta": 1000.0,
        "norm_eps": 1e-05,
        "patch_size": 2,
        "num_train_timesteps": 1000,
        "vae_scale_factor": 8,
        "size_multiple": 16,
        # The Turbo checkpoint's fixed timestep shift.
        "turbo_mu": 1.15,
    },
    repo="huggingface/diffusers",
    revision="e0abab83b5df05de9e7abd788643c1a7c1e42e28",
)

# The template of every prompt: a fixed system turn and the opening of
# the user turn, the prompt, and the end of the turn with the open
# assistant turn. The pipeline drops the first 34 rows, the fixed part
# before the prompt, after the encoder ran. It cuts the prompt so that
# what it keeps of the prompt and the five rows after it are at most 512.
PROMPT = ConstantsRow(
    values={
        "template": (
            "<|im_start|>system\nDescribe the image by detailing the color, "
            "shape, size, texture, quantity, text, spatial relationships of "
            "the objects and background:<|im_end|>\n<|im_start|>user\n"
            "{}<|im_end|>\n<|im_start|>assistant\n"),
        "drop_tokens": 34,
        "suffix_tokens": 5,
        "max_sequence_length": 512,
        # hidden_states indices of the encoder, 0 being the embeddings.
        "tap_layers": (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
    },
    repo="huggingface/diffusers",
    revision="e0abab83b5df05de9e7abd788643c1a7c1e42e28",
)

# The model card's own run: Turbo at 8 steps with guidance_scale 0, which
# is no guidance. The schedule is the family's own, so the shift field is
# unused.
SAMPLING = {
    "turbo": SamplingRow(steps=8, cfg=1.0, shift=1.0,
                         repo="krea/Krea-2-Turbo",
                         revision="98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
                         quality=(("low", 4), ("medium", 6), ("high", 8))),
}

IDENTITY_TENSORS = (PROJECTOR, "first.bias")
_TURBO = VariantRow(
    "turbo", "Abiray/Krea-2-Turbo-GGUF",
    "ac1417a9e773b4244b683744b5264f5944d0ff0f",
    tuple(f"Krea-2-Turbo-{q}.gguf" for q in (
        "Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K",
        "Q8_0")))
VARIANTS = {
    "c46a6ba550c75b757a63d3a26ba3a4985c59fec525a73800832bb0b9bc205137": _TURBO,
}
# Krea 2 Raw, as realrebelai/KREA-2_GGUFs at
# 9721c3ec932c43dff6dbb5793a816d9ad105277f stores it under BASE/.
REFUSED = {
    "284b54edd003f36239b1d30e62b64fba898dc0598d4434eb7985639fcaa6112c": (
        "this is a Krea 2 Raw GGUF, which gmlx image does not run; Krea 2 "
        "Turbo is the Krea 2 variant it supports"),
}

# Norm tensors that identify the encoder weights: GGUF name, path in the
# loaded model and sha256 of the bf16 bits.
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
    "unsloth/Qwen3-VL-4B-Instruct-GGUF at "
    "00c00da0690c4b14b5539b02c4ea5d7c9102b35e")


def _projector_layers(view: HeaderView) -> int | None:
    """The encoder layers the projector takes, from its stored shape."""
    s = view.shape(PROJECTOR)
    if s is None:
        return None
    if len(s) == 2 and s[1] == 1:
        return int(s[0])
    if len(s) == 1:
        return int(s[0])
    return None


def projector_shape(view: HeaderView) -> tuple | None:
    """The torch shape a flattened projector is stored with, from its
    ``comfy.gguf.orig_shape`` key, or None when the file keeps the
    shape."""
    raw = view.kv.get("comfy.gguf.orig_shape." + PROJECTOR)
    s = view.shape(PROJECTOR)
    if raw is None or s is None or len(s) != 1:
        return None
    return tuple(int(v) for v in raw)


def layout(view: HeaderView) -> str:
    """``last_mlp`` when the file has the final layer's two extra
    matrices, else ``plain``."""
    return "last_mlp" if all(view.has(n) for n in LAST_MLP) else "plain"


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    if not (view.has(PROJECTOR) and view.has("first.weight")
            and view.count("blocks")):
        return None
    diffs = [f"{n} is {view.shape(n)}, Krea 2 has {s}"
             for n, s in _REQUIRED if view.shape(n) != s]
    n = _projector_layers(view)
    if n != 12:
        diffs.append(f"{PROJECTOR} is {view.shape(PROJECTOR)}, Krea 2 has "
                     "(12, 1)")
    orig = projector_shape(view)
    if len(view.shape(PROJECTOR) or ()) == 1 and orig != (1, 12):
        diffs.append(f"{PROJECTOR} is flattened, and its stored shape is "
                     f"{orig}, not (1, 12)")
    diffs += [f"{g} has {view.count(g)} blocks, Krea 2 has {c}"
              for g, c in _COUNTS if view.count(g) != c]
    if sum(view.has(t) for t in LAST_MLP) == 1:
        diffs.append("it has one of last.up and last.down")
    if diffs:
        return Refusal("a Krea 2 relative that gmlx image does not "
                       "support: " + ", ".join(diffs))
    return Match(layout=layout(view))


def synth_config(view: HeaderView) -> dict:
    """The official transformer config keys, from shapes and constants,
    plus the derived widths the module tree needs."""
    c = CONSTANTS.values
    s = view.dims
    in_ch, dim = s("first.weight")
    head_dim = s(_B0 + "attn.qknorm.qnorm.scale")[0]
    txt_dim = s(_T0 + "prenorm.scale")[0]
    txt_head = s(_T0 + "attn.qknorm.qnorm.scale")[0]
    return {
        "in_channels": in_ch,
        "num_layers": view.count("blocks"),
        "attention_head_dim": head_dim,
        "num_attention_heads": s(_B0 + "attn.wq.weight")[1] // head_dim,
        "num_key_value_heads": s(_B0 + "attn.wk.weight")[1] // head_dim,
        "intermediate_size": s(_B0 + "mlp.gate.weight")[1],
        "timestep_embed_dim": s("tmlp.0.weight")[0],
        "text_hidden_dim": txt_dim,
        "num_text_layers": _projector_layers(view),
        "text_num_attention_heads": s(_T0 + "attn.wq.weight")[1] // txt_head,
        "text_num_key_value_heads": s(_T0 + "attn.wk.weight")[1] // txt_head,
        "text_intermediate_size": s(_T0 + "mlp.gate.weight")[1],
        "num_layerwise_text_blocks": view.count(
            "txtfusion.layerwise_blocks"),
        "num_refiner_text_blocks": view.count("txtfusion.refiner_blocks"),
        "axes_dims_rope": list(c["axes_dims_rope"]),
        "rope_theta": c["rope_theta"],
        "norm_eps": c["norm_eps"],
        "patch_size": c["patch_size"],
        "num_train_timesteps": c["num_train_timesteps"],
        "out_channels": s("last.linear.weight")[1],
        "last_mlp": layout(view) == "last_mlp",
    }


def build(config: dict):
    from .model import Krea2DiT

    return Krea2DiT(config)


def lora_map(config: dict):
    from .lora_map import lora_map as build_map

    return build_map(config)


def encoder():
    """The text encoder of the family: Qwen3-VL-4B read as a text model,
    after twelve of its layers and before its final norm, on the
    reference template without the rows before the prompt. The twelve
    states of a token are its features side by side, layer by layer.
    The tap runs in float32."""
    from gmlx.image.text_encoder import EncoderSpec, IdentityTensor

    p = PROMPT.values
    drop, tail = p["drop_tokens"], p["suffix_tokens"]
    return EncoderSpec(
        name="qwen3-vl-4b",
        arch="qwen3vl",
        model_type="qwen3",
        hidden=2560,
        layers=36,
        tap_layers=tuple(p["tap_layers"]),
        template=p["template"],
        max_tokens=drop + p["max_sequence_length"],
        keep_tail=tail,
        stack=True,
        identity=tuple(IdentityTensor(*row) for row in ENCODER_IDENTITY),
        identity_source=ENCODER_IDENTITY_SOURCE,
        drop=drop,
        activation="float32",
    )


def schedule(steps: int, *, tokens: int | None, variant: str | None = None):
    """Sigmas of ``steps``, shifted by Turbo's fixed mu. Any other variant
    has no schedule."""
    from gmlx.image.schedule import krea2_sigmas

    if variant != "turbo":
        raise ValueError(f"Krea 2 has a schedule for turbo only, not "
                         f"{variant!r}")
    return krea2_sigmas(steps, CONSTANTS.values["turbo_mu"])


# The optional upsampler of docs/prompting.md, on a chat model of the
# user's choice: docs/expansion.txt as the system prompt, shipped
# unmodified, and the typed prompt as the user turn. The prompt asks for
# a thinking block, then one paragraph, so thinking is on and the caption
# is that paragraph. Krea names no sampling; gmlx decodes greedily, so
# the same prompt writes the same caption, with room for the thinking.
UPSAMPLE = UpsampleParts(
    prompts="expansion.txt",
    sha256="2a5c24a3a83e9d415d4679a4e1853ddf056bd0329f3af315b27ed72b6ad9c2f0",
    key=None,
    max_tokens=6144,
    temperature=0.0,
    presence_penalty=0.0,
    thinking="on",
    repo="github.com/krea-ai/krea-2",
    revision="db3984fbc6e13b34c0064990fc2d95ac64d00058",
    package=__package__,
    suggested="Qwen3.5-9B",
    parse=upsample.paragraph,
)


FAMILY = Family(
    name="krea2",
    title="Krea 2",
    # Before Qwen-Image 2.1, whose arch string one Krea 2 layout carries.
    order=15,
    arch_strings=frozenset({"qwen_image", "krea2", "krea-2", None}),
    # Labels that converters also write for other models. A FLUX file keeps
    # its refusal by name, since its tensors never match those of Krea 2.
    shared_arch_strings=frozenset({"flux", "pig"}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=build,
    # The reference keeps every norm weight in float32.
    fp32_keep=("norm.scale", "txtmlp.0.scale"),
    identity_tensors=IDENTITY_TENSORS,
    variants=VARIANTS,
    refused=REFUSED,
    # Only Turbo has a schedule; a merge with new weights may not be Turbo.
    needs_variant=True,
    sampling=SAMPLING,
    lora_map=lora_map,
    encoder=encoder,
    schedule=schedule,
    reshaped=(PROJECTOR,),
    upsample=UPSAMPLE,
)
