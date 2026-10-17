"""Qwen-Image 2.1 fingerprint, constants, config synthesis and the hooks
of its registry row.

Shapes are GGUF order, input dim first. The MLP is stored either as one
fused gate_up matrix or as a gate and projection pair, reported as the
layout and built as the file has it.

The denoise step is given in parts. The conditioning part is the prefix
cache of a conditioning, each block's K and V of the text and reference
image tokens, and the modulation part is what the target tokens derive
from one sigma. The schedule shifts by the image token count of the
request.
"""

from __future__ import annotations

# The prompt template is the text to image template of diffusers'
# src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py at e0abab83
# (Apache-2.0; see licenses/diffusers-LICENSE).

from gmlx.image import upsample
from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import (ConstantsRow, EditParts, Family,
                                 SamplingRow, StepParts, UpsampleParts,
                                 VariantRow)

_B0 = "transformer_blocks.0."

_REQUIRED = (
    ("txt_in.text_norm.weight", (4096,)),
    ("modulation.1.weight", (4096, 16384)),
    (_B0 + "attn.norm_q.weight", (128,)),
    ("img_in.weight", (64, 4096)),
    ("proj_out.weight", (4096, 64)),
)
_N_BLOCKS = 32

# transformer/config.json gives the first five values and
# scheduler/scheduler_config.json the six of the schedule. The VAE factor
# of 16 and the size multiple of 32, two latent pixels, are the reference
# pipeline's own.
CONSTANTS = ConstantsRow(
    values={
        "axes_dims_rope": [16, 56, 56],
        "rope_theta": 10000.0,
        "eps": 1e-06,
        "patch_size": 1,
        "causal_condition": True,
        "base_image_seq_len": 256,
        "max_image_seq_len": 8192,
        "base_shift": 0.5,
        "max_shift": 0.9,
        "shift_terminal": 0.02,
        "num_train_timesteps": 1000,
        "vae_scale_factor": 16,
        "size_multiple": 32,
    },
    repo="Qwen/Qwen-Image-2.1",
    revision="790c92633540aa0cb11d9abf19eb46d861714758",
)

# What the reference pipeline does to a prompt before the text encoder
# and to the encoder's output after it. The template is its text to image
# template: a fixed system turn, the prompt as the user turn and the open
# assistant turn, with no thinking tokens. The pipeline passes it to the
# tokenizer as it stands and never renders the chat template of the
# encoder. It drops the rows of the system turn from the hidden states,
# 14 tokens with the tokenizer of Qwen3-VL-8B, and it encodes an empty
# prompt as one space, since the tokenizer has no token to open with.
PROMPT = ConstantsRow(
    values={
        "template_t2i": (
            "<|im_start|>system\n"
            "Comprehend and analyze the provided prompt.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"),
        "system_turn": (
            "<|im_start|>system\n"
            "Comprehend and analyze the provided prompt.<|im_end|>\n"),
        "drop_tokens": 14,
        "empty_prompt": " ",
        # The template of a prompt with reference images: the user turn
        # opens with one image slot per reference, in order and joined by
        # one space, and the prompt follows the last slot. {n} is the
        # reference's number from 1. The processor expands the pad token
        # of a slot to one token per 32 by 32 pixels of its reference.
        "template_edit": (
            "<|im_start|>system\n"
            "Comprehend and analyze the provided prompt.<|im_end|>\n"
            "<|im_start|>user\n{slots}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"),
        "image_slot": "<image{n}><|vision_start|><|image_pad|><|vision_end|>",
        "image_slot_join": " ",
        "image_pad": "<|image_pad|>",
    },
    repo="huggingface/diffusers",
    revision="e0abab83b5df05de9e7abd788643c1a7c1e42e28",
)

# What the reference pipeline does to a reference image, and how many a
# request takes. The pipeline makes the image RGBA and resizes it with
# Lanczos to the size of calculate_dimensions: the area of
# output_resolution squared, 1024 by default, at the image's aspect ratio,
# each side rounded to a multiple of 32. The encoder sees that image with
# its alpha composited over white, and the VAE all four channels. The
# pipeline sets no count, and the model card at the pinned revision gives
# the count: up to 10 reference images. A reference token holds its cache
# bytes exactly in every measured request, up to four references of
# 1024 pixels under guidance, so the count stays the card's and the area
# the pipeline's, and the memory plan refuses what a machine cannot hold.
REFERENCES = ConstantsRow(
    values={
        "area": 1024 * 1024,
        "side_multiple": 32,
        "resample": "lanczos",
        "encoder_background": [255, 255, 255],
        "slot_px": 32,
        "min_area": 256 * 256,
        "max_refs": 10,
    },
    repo="Qwen/Qwen-Image-2.1",
    revision="790c92633540aa0cb11d9abf19eb46d861714758",
)

# The model card at this revision runs each of its three examples with
# num_inference_steps=40 and names no guidance, so the guidance is the
# reference pipeline's default, true_cfg_scale=1.0. The shift is the
# scheduler config's static shift, which the dynamic shift replaces.
# Turbo is distilled to the eight sigmas of model_index.json's
# sample_sigmas, which its scheduler config (shift 1.0, no dynamic
# shifting) passes as they are, with a final zero; its card samples at
# CFG 1. A file whose identity matches no row samples as base.
SAMPLING = {
    "base": SamplingRow(steps=40, cfg=1.0, shift=1.0,
                        repo="Qwen/Qwen-Image-2.1",
                        revision="790c92633540aa0cb11d9abf19eb46d861714758",
                        quality=(("low", 16), ("medium", 28), ("high", 40))),
    "turbo": SamplingRow(steps=8, cfg=1.0, shift=1.0,
                         repo="Qwen/Qwen-Image-2.1-Turbo",
                         revision="d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
                         quality=(("low", 8), ("medium", 8), ("high", 8)),
                         sigmas=(1.0, 0.978453, 0.95418, 0.926626, 0.89508,
                                 0.845148, 0.704534, 0.414568)),
}

# The variants share every shape, so the variant comes from the hash of
# the text input norm, which Turbo's distillation changed. Every GGUF of
# both repos keeps it BF16, while some quantize the other norms.
IDENTITY_TENSORS = ("txt_in.text_norm.weight",)
_BASE = VariantRow(
    "base", "unsloth/Qwen-Image-2.1-GGUF",
    "2c31ccd392b367a6637841a143813320a02dff55",
    tuple(f"qwen-image-2.1-{q}.gguf" for q in (
        "F16", "Q2_K", "Q3_K_M", "Q3_K_S", "Q3_K_XL", "Q4_K_M", "Q4_K_S",
        "Q5_K_M", "Q5_K_S", "Q6_K", "Q6_K_XL", "Q8_0")))
_TURBO = VariantRow(
    "turbo", "AtomicChat/Qwen-Image-2.1-Turbo-GGUF",
    "bb25d06bc74119c12207243d68917951e6d9c232",
    tuple(f"Qwen-Image-2.1-Turbo-{q}.gguf" for q in (
        "AD-Q2_K", "AD-Q3_K", "AD-Q4_K", "AD-Q5_K", "AD-Q6_K", "BF16",
        "Q8_0")))
VARIANTS = {
    "b496795d142460d2f72d13b560d7cf54fba3081cc895c2d41b8e00b2470f9efe": _BASE,
    "bfd1f6efe5e2f6ccbd2a9c041a08ad8613a69563d1a7927e57c0170d07570613": _TURBO,
}

# Norm tensors that identify the encoder weights: GGUF name, path in the
# loaded model and sha256 of the bf16 bits.
ENCODER_IDENTITY = (
    ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight",
     "da14da9264fcb45df7b76ff86c2cc6947639c162a96edb0e5459c9d2d87eec19"),
    ("blk.17.ffn_norm.weight",
     "model.layers.17.post_attention_layernorm.weight",
     "76ac781c8aaf8ec914fd9fe65a858acaf999b9bf0e783cbd226b1b5e25a1db23"),
    ("output_norm.weight", "model.norm.weight",
     "fe204de7e848b92e8a758ab5e27632e9b698f29571ca56057d13ba73d0b2012a"),
)
ENCODER_IDENTITY_SOURCE = (
    "unsloth/Qwen3-VL-8B-Instruct-GGUF, Qwen3-VL-8B-Instruct-Q4_K_M.gguf "
    "with sha256 108e7ff92b78eefd3db4741885104acba514255c11b617d3c7b197a5"
    "f46efe89")

# Norm tensors that identify the vision tower in the mmproj GGUF, in the
# same form.
MMPROJ_IDENTITY = (
    ("v.blk.0.ln1.weight", "vision_tower.blocks.0.norm1.weight",
     "e6f31c471b644ea73eeddfb1f2e7de3c9920943b691cde08d117ae42574cf723"),
    ("v.blk.26.ln2.weight", "vision_tower.blocks.26.norm2.weight",
     "2a3754a9b47717d80a5907841487308e3af392f0d328b579848a38e943c07047"),
    ("v.post_ln.weight", "vision_tower.merger.norm.weight",
     "c9726ac50eb0344e10ce376368440eff0c56a19cf6f1b8d36399e0e3bf5e5ac8"),
)
MMPROJ_IDENTITY_SOURCE = (
    "unsloth/Qwen3-VL-8B-Instruct-GGUF, mmproj-BF16.gguf with sha256 "
    "6516bb64bae1503a0fcd7ec9fa39655f8c481580be0a0a066397941d9761c9f4")

# A pipeline holds the conditioning states of the request it runs, and
# beside them up to this much of other requests' states less the bytes of
# its own. It is both text states of a guided request twice over. The
# prefix of one megapixel reference is about 2 GiB, so a request with
# references holds its own states alone.
PREFIX_CACHE_BYTES = 1 << 30


def _relative(name: str) -> Refusal:
    return Refusal(f"this is a {name} GGUF, a Qwen-Image relative that gmlx "
                   "image does not support")


_GATE_UP = (4096, 24576)        # the fused gate and up projection
_HALF = (4096, 12288)           # each of the unfused pair
_TO_Q = (4096, 4096)


def _mlp_names(view: HeaderView) -> str | None:
    """The layout the tensor names of block 0 say, at any width."""
    if view.has(_B0 + "img_mlp.gate_up.weight"):
        return "fused"
    if (view.has(_B0 + "img_mlp.gate_layer.weight")
            and view.has(_B0 + "img_mlp.proj.weight")):
        return "unfused"
    return None


def _layout(view: HeaderView) -> str | None:
    """The layout of a file whose image MLP has the widths of Qwen-Image
    2.1, which the fingerprint requires."""
    if view.shape(_B0 + "img_mlp.gate_up.weight") == _GATE_UP:
        return "fused"
    if (view.shape(_B0 + "img_mlp.gate_layer.weight") == _HALF
            and view.shape(_B0 + "img_mlp.proj.weight") == _HALF):
        return "unfused"
    return None


def _to_q_diffs(view: HeaderView) -> list[str]:
    """The blocks whose query projection is absent or not the width of
    Qwen-Image 2.1, the first few by name."""
    bad = [(i, view.shape(f"transformer_blocks.{i}.attn.to_q.weight"))
           for i in range(_N_BLOCKS)]
    bad = [(i, s) for i, s in bad if s != _TO_Q]
    if not bad:
        return []
    named = ", ".join(f"block {i} has {'none' if s is None else s}"
                      for i, s in bad[:3])
    more = f" and {len(bad) - 3} more" if len(bad) > 3 else ""
    return [f"attn.to_q.weight is {_TO_Q} in every block of Qwen-Image "
            f"2.1, and {named}{more}"]


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    if (view.has(_B0 + "attn.norm_added_q.weight")
            and view.has(_B0 + "img_mlp.w1.weight")):
        return _relative("Lens")
    txt_norm = view.shape("txt_norm.weight")
    proj_out = view.shape("proj_out.weight")
    if (txt_norm == (2560,) and proj_out is not None and len(proj_out) == 2
            and proj_out[1] == 128):
        return _relative("Mage-Flow")
    if not (view.has("img_in.weight") and view.count("transformer_blocks")):
        return None
    if txt_norm is not None and view.has(_B0 + "attn.add_q_proj.weight"):
        # The Layered model alone has the extra timestep embedding, with
        # or without the index tensor that marks an Edit 2511 file. An
        # Edit file written without that marker has the header of the
        # 20B model, so the two share a name there.
        if view.has("time_text_embed.addition_t_embedding.weight"):
            return _relative("Qwen-Image-Layered")
        if view.has("__index_timestep_zero__"):
            return _relative("Qwen-Image-Edit 2511")
        return Refusal(
            "this is a Qwen-Image 20B GGUF, or a Qwen-Image-Edit one "
            "written without its index tensor, whose header is the same: "
            "a Qwen-Image relative that gmlx image does not support")
    diffs = [f"{n} is {view.shape(n)}, Qwen-Image 2.1 has {s}"
             for n, s in _REQUIRED if view.shape(n) != s]
    layout = _layout(view)
    if layout is None:
        diffs.append("the image MLP is neither img_mlp.gate_up (4096, 24576) "
                     "nor the img_mlp.gate_layer and img_mlp.proj pair of "
                     "(4096, 12288) each")
    diffs += _to_q_diffs(view)
    if view.any_suffix(".bias"):
        diffs.append("it has bias tensors")
    if txt_norm is not None:
        diffs.append("it has txt_norm.weight")
    if any(".attn.add_q_proj." in n for n in view.tensors):
        diffs.append("it has attn.add_q_proj")
    n = view.count("transformer_blocks")
    if n != _N_BLOCKS:
        diffs.append(f"it has {n} blocks, Qwen-Image 2.1 has {_N_BLOCKS}")
    if diffs:
        return Refusal("a Qwen-Image relative that gmlx image does not "
                       "support: " + ", ".join(diffs))
    return Match(layout=layout or "fused")


def synth_config(view: HeaderView) -> dict:
    """The official transformer config keys, from shapes and constants, plus
    the derived widths and the scheduler's training timesteps, which the
    module tree needs."""
    c = CONSTANTS.values
    s = view.dims
    in_ch, inner = s("img_in.weight")
    head_dim = s(_B0 + "attn.norm_q.weight")[0]
    layout = _mlp_names(view) or "fused"
    if layout == "unfused":
        ffn = s(_B0 + "img_mlp.gate_layer.weight")[1]
    else:
        ffn = s(_B0 + "img_mlp.gate_up.weight")[1] // 2
    return {
        "attention_head_dim": head_dim,
        "axes_dims_rope": list(c["axes_dims_rope"]),
        "context_in_dim": s("txt_in.text_norm.weight")[0],
        "in_channels": in_ch,
        "num_attention_heads": inner // head_dim,
        "num_layers": view.count("transformer_blocks"),
        "out_channels": s("proj_out.weight")[1],
        "patch_size": c["patch_size"],
        "mlp_ratio": ffn // inner,
        "eps": c["eps"],
        "causal_condition": c["causal_condition"],
        "inner_dim": inner,
        "ffn_dim": ffn,
        "modulation_dim": s("modulation.1.weight")[1],
        "rope_theta": c["rope_theta"],
        "num_train_timesteps": c["num_train_timesteps"],
        "mlp_layout": layout,
        "timestep_dim": s(
            "time_text_embed.timestep_embedder.linear_1.weight")[0],
    }


def build(config: dict):
    from .model import QwenImage21DiT

    return QwenImage21DiT(config)


def _modulation(dit, sigma):
    from .model import modulation

    return modulation(dit, sigma)


def _prefix(dit, cond):
    from .model import prefill_prefix

    return prefill_prefix(dit, cond)


def _forward(dit, latents, sigma, cond, prefix, mod):
    from .model import forward_target

    return forward_target(dit, latents, sigma, cond, prefix, mod=mod)


def _prefix_bytes(dit, cond) -> int:
    """Bytes of the prefix state of ``cond``: K and V of every block over
    its prefix tokens."""
    from .model import prefix_bytes

    return prefix_bytes(dit, cond)


def lora_map(config: dict):
    from .lora_map import lora_map as build_map

    return build_map(config)


def encoder():
    """The text encoder of the family: Qwen3-VL-8B read as a text model,
    after its last layer and before its final norm, on the reference
    pipeline's template and without the rows of the system turn. The
    reference sets no token cap.

    The tap runs in float32. The row of each <|im_start|> token amplifies
    rounding: against the float32 features of the reference, the reference
    in bfloat16 is 21 percent off in relative L2 on the first kept row and
    2 percent on the others. On the Q4_K_M file a bfloat16 tap puts that
    row 16 to 71 percent off, by the prompt length, and the features of a
    36 row prompt 10 percent off, which moves the final latents of a 512
    pixel image 6 percent. In float32 over the weights as loaded the row
    is 4 to 13 percent off and the features 1 to 4 percent."""
    from gmlx.image.text_encoder import EncoderSpec, IdentityTensor

    p = PROMPT.values
    return EncoderSpec(
        name="qwen3-vl-8b",
        arch="qwen3vl",
        model_type="qwen3",
        hidden=4096,
        layers=36,
        tap_layers=(36,),
        template=p["template_t2i"],
        max_tokens=None,
        identity=tuple(IdentityTensor(*row) for row in ENCODER_IDENTITY),
        identity_source=ENCODER_IDENTITY_SOURCE,
        drop=p["drop_tokens"],
        empty=p["empty_prompt"],
        activation="float32",
        vision_type="qwen3_vl",
        vision_identity=tuple(IdentityTensor(*row) for row in MMPROJ_IDENTITY),
        vision_identity_source=MMPROJ_IDENTITY_SOURCE,
    )


def _prepare(image, area: int):
    from .edit import prepare

    return prepare(image, area)


def _edit_ids(handle, prompt: str, refs):
    from .edit import token_ids

    return token_ids(handle, prompt, refs)


def _edit_encode(handle, prompt: str, refs):
    from .edit import encode

    return encode(handle, prompt, refs)


def _edit_template_id() -> str:
    from .edit import template_id

    return template_id()


def schedule(steps: int, *, tokens: int | None, variant: str | None = None):
    """Sigmas of ``steps`` for an image of ``tokens`` latent tokens."""
    from gmlx.image.schedule import qwen21_sigmas

    if tokens is None:
        raise ValueError("the Qwen-Image 2.1 schedule shifts by the image "
                         "token count; pass tokens")
    c = CONSTANTS.values
    return qwen21_sigmas(
        steps, tokens, base_len=c["base_image_seq_len"],
        max_len=c["max_image_seq_len"], base_shift=c["base_shift"],
        max_shift=c["max_shift"], terminal=c["shift_terminal"])


# Where to get the rewriter prompt Qwen publishes, which is under a
# license gmlx does not ship.
PROMPT_REPO = "Qwen/Qwen-Image-2.1-PE-T2I"
PROMPT_FILE = "system_prompt.txt"

# The optional upsampler, on the system prompt of a file the user gives:
# gmlx ships none. Qwen's rewriter thinks, then answers with a JSON
# object whose rewritten_prompt is the caption, beside an aspect ratio
# that gmlx does not read, so the image keeps the size the request asks
# for. Thinking is on, with the presence penalty of 1.5 Qwen's text to
# image rewriter samples with, and gmlx decodes greedily, so the same
# prompt writes the same caption.
UPSAMPLE = UpsampleParts(
    prompts=None,
    sha256="",
    key=None,
    max_tokens=8192,
    temperature=0.0,
    presence_penalty=1.5,
    thinking="on",
    repo=f"huggingface.co/{PROMPT_REPO}",
    revision="f3ed7985c788ad75b3ab7223e0c4c51e2a43545b",
    suggested="Qwen3.5-9B",
    parse=upsample.json_field("rewritten_prompt"),
    needs=("Qwen-Image 2.1 ships no upsampler prompt: set upsample_prompt "
           "on its image_models entry, or pass --upsample-prompt, to a text "
           f"file. Qwen publishes one as {PROMPT_FILE} in the Hugging Face "
           f"repository {PROMPT_REPO}, under the Qwen Research License, for "
           "non-commercial use only, and `gmlx pull qwen-image` fetches it"),
)


FAMILY = Family(
    name="qwen_image21",
    title="Qwen-Image 2.1",
    order=20,
    arch_strings=frozenset({"qwen_image", "qwen_image21", None}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=build,
    # diffusers keeps these modules out of its layerwise casting.
    fp32_keep=("norm",),
    identity_tensors=IDENTITY_TENSORS,
    variants=VARIANTS,
    sampling=SAMPLING,
    unknown_variant="base",
    step=StepParts(_modulation, _prefix, _forward, PREFIX_CACHE_BYTES,
                   _prefix_bytes),
    schedule=schedule,
    lora_map=lora_map,
    encoder=encoder,
    edit=EditParts(max_refs=REFERENCES.values["max_refs"],
                   area=REFERENCES.values["area"],
                   min_area=REFERENCES.values["min_area"],
                   prepare=_prepare, token_ids=_edit_ids,
                   encode=_edit_encode, template_id=_edit_template_id),
    upsample=UPSAMPLE,
)
