"""Qwen-Image 2.1 fingerprint, constants and config synthesis.

Shapes are GGUF order, input dim first. The MLP is stored either as one
fused gate_up matrix or as a gate and projection pair, reported as the
layout. The module tree is not built yet, so a matching file is recognized
and then refused by the loader.
"""

from __future__ import annotations

from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import ConstantsRow, Family

_B0 = "transformer_blocks.0."

_REQUIRED = (
    ("txt_in.text_norm.weight", (4096,)),
    ("modulation.1.weight", (4096, 16384)),
    (_B0 + "attn.norm_q.weight", (128,)),
    ("img_in.weight", (64, 4096)),
    ("proj_out.weight", (4096, 64)),
)
_N_BLOCKS = 32

CONSTANTS = ConstantsRow(
    values={
        "axes_dims_rope": [16, 56, 56],
        "rope_theta": 10000.0,
        "eps": 1e-06,
        "patch_size": 1,
        "causal_condition": True,
    },
    repo="Qwen/Qwen-Image-2.1",
    revision="790c92633540aa0cb11d9abf19eb46d861714758",
)


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
    the derived widths the module tree needs."""
    c = CONSTANTS.values
    s = view.shape
    in_ch, inner = s("img_in.weight")
    head_dim = s(_B0 + "attn.norm_q.weight")[0]
    if _layout(view) == "unfused":
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
    }


FAMILY = Family(
    name="qwen_image21",
    title="Qwen-Image 2.1",
    order=20,
    arch_strings=frozenset({"qwen_image", None}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=None,
)
