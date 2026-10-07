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

from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import (ConstantsRow, Family, SamplingRow,
                                 StepParts)

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
    },
    repo="huggingface/diffusers",
    revision="e0abab83b5df05de9e7abd788643c1a7c1e42e28",
)

# The model card at this revision runs each of its three examples with
# num_inference_steps=40 and names no guidance, so the guidance is the
# reference pipeline's default, true_cfg_scale=1.0. The shift is the
# scheduler config's static shift, which the dynamic shift replaces.
SAMPLING = {
    "default": SamplingRow(steps=40, cfg=1.0, shift=1.0,
                           repo="Qwen/Qwen-Image-2.1",
                           revision="790c92633540aa0cb11d9abf19eb46d861714758"),
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

# A pipeline holds the conditioning states of the request it runs, and
# beside them up to this much of other requests' states less the bytes of
# its own. It is both text states of a guided request twice over. The
# prefix of one megapixel reference is about 2 GiB, so a request with
# references holds its own states alone.
PREFIX_CACHE_BYTES = 1 << 30


def _relative(name: str) -> Refusal:
    return Refusal(f"this is a {name} GGUF, a Qwen-Image relative that gmlx "
                   "image does not support")


def _layout(view: HeaderView) -> str | None:
    if view.shape(_B0 + "img_mlp.gate_up.weight") == (4096, 24576):
        return "fused"
    if (view.has(_B0 + "img_mlp.gate_layer.weight")
            and view.has(_B0 + "img_mlp.proj.weight")):
        return "unfused"
    return None


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
        if view.has("__index_timestep_zero__"):
            return _relative("Qwen-Image-Edit 2511")
        if view.has("time_text_embed.addition_t_embedding.weight"):
            return _relative("Qwen-Image-Layered")
        return _relative("Qwen-Image 20B")
    diffs = [f"{n} is {view.shape(n)}, Qwen-Image 2.1 has {s}"
             for n, s in _REQUIRED if view.shape(n) != s]
    layout = _layout(view)
    if layout is None:
        diffs.append("the image MLP is neither img_mlp.gate_up (4096, 24576) "
                     "nor the img_mlp.gate_layer and img_mlp.proj pair")
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
    layout = _layout(view) or "fused"
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
    )


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


FAMILY = Family(
    name="qwen_image21",
    title="Qwen-Image 2.1",
    order=20,
    arch_strings=frozenset({"qwen_image", None}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=build,
    # diffusers keeps these modules out of its layerwise casting.
    fp32_keep=("norm",),
    sampling=SAMPLING,
    step=StepParts(_modulation, _prefix, _forward, PREFIX_CACHE_BYTES,
                   _prefix_bytes),
    schedule=schedule,
    lora_map=lora_map,
    encoder=encoder,
)
