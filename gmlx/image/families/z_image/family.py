"""Z-Image fingerprint, constants and config synthesis.

Shapes are GGUF order, input dim first. Base and Turbo share every name and
shape, so the variant comes from the identity hash.
"""

from __future__ import annotations

from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import ConstantsRow, Family, VariantRow

_REQUIRED = (
    ("cap_embedder.1.weight", (2560, 3840)),
    ("x_embedder.weight", (64, 3840)),
    ("layers.0.attention.qkv.weight", (3840, 11520)),
    ("layers.0.attention.q_norm.weight", (128,)),
    ("layers.0.feed_forward.w1.weight", (3840, 10240)),
    ("layers.0.adaLN_modulation.0.weight", (256, 15360)),
    ("t_embedder.mlp.0.weight", (256, 1024)),
    ("final_layer.linear.weight", (3840, 64)),
)
_PRESENT = (
    "noise_refiner.0.attention.k_norm.weight",
    "context_refiner.0.attention.qkv.weight",
    "cap_pad_token",
    "x_pad_token",
)
_COUNTS = (("layers", 30), ("noise_refiner", 2), ("context_refiner", 2))

# Relatives under arch lumina2, each refused by name.
_RELATIVE_KEYS = (
    ("clip_text_pooled_proj.0.weight", "NewBie"),
    ("siglip_embedder.0.weight", "Z-Image Omni or Edit"),
    ("dec_net.cond_embed.weight", "Z-Image pixel space"),
)

CONSTANTS = ConstantsRow(
    values={
        "all_patch_size": [2],
        "all_f_patch_size": [1],
        "axes_dims": [32, 48, 48],
        "axes_lens": [1536, 512, 512],
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "norm_eps": 1e-05,
        "final_norm_eps": 1e-06,
        "seq_multiple": 32,
        "scheduler_shift": 3.0,
        "num_train_timesteps": 1000,
        "vae_scaling_factor": 0.3611,
        "vae_shift_factor": 0.1159,
    },
    repo="Tongyi-MAI/Z-Image-Turbo",
    revision="f332072aa78be7aecdf3ee76d5c247082da564a6",
)

IDENTITY_TENSORS = ("x_pad_token", "cap_pad_token", "final_layer.linear.bias")

_TURBO = VariantRow(
    "turbo", "unsloth/Z-Image-Turbo-GGUF",
    "6c80814333b7b6a70a2e5b469a7c6437ce65de0f",
    ("z-image-turbo-Q4_K_M.gguf", "z-image-turbo-Q8_0.gguf",
     "z-image-turbo-BF16.gguf"))
_BASE = VariantRow(
    "base", "unsloth/Z-Image-GGUF",
    "c9913e69743c5d9dfa7fdac58a0cc5709a17aa08",
    ("z-image-Q4_K_M.gguf", "z-image-Q8_0.gguf"))
VARIANTS = {
    "67736a82b33f43c468f7a65b2ab28d90eeecbaa3a4dcde9d3abcd4e8524af2ca": _TURBO,
    "b388a0ad8720b3910cebba4868b942d9fc49d2d6383ff6ffa8fd6de94caeada1": _BASE,
}


def _relative(name: str) -> Refusal:
    return Refusal(f"this is a {name} GGUF, a Z-Image relative that gmlx "
                   "image does not support")


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    if not (view.has("layers.0.attention.qkv.weight")
            and view.has("cap_embedder.1.weight")):
        return None
    for key, name in _RELATIVE_KEYS:
        if view.has(key):
            return _relative(name)
    cap = view.shape("cap_embedder.1.weight")
    if cap is not None and len(cap) == 2 and cap[1] == 2304:
        return _relative("Lumina-Image 2.0")
    if not (view.has("x_pad_token") or view.has("cap_pad_token")):
        return _relative("Ming-Image")
    diffs = [f"{n} is {view.shape(n)}, Z-Image has {s}"
             for n, s in _REQUIRED if view.shape(n) != s]
    diffs += [f"{n} is missing" for n in _PRESENT if not view.has(n)]
    diffs += [f"{g} has {view.count(g)} blocks, Z-Image has {c}"
              for g, c in _COUNTS if view.count(g) != c]
    if diffs:
        return Refusal("a Z-Image relative that gmlx image does not support: "
                       + ", ".join(diffs))
    return Match()


def synth_config(view: HeaderView) -> dict:
    """The official transformer config keys, from shapes and constants, plus
    the derived widths the module tree needs."""
    c = CONSTANTS.values
    s = view.shape
    x_in, dim = s("x_embedder.weight")
    head_dim = s("layers.0.attention.q_norm.weight")[0]
    qkv_out = s("layers.0.attention.qkv.weight")[1]
    patch = c["all_patch_size"][0]
    f_patch = c["all_f_patch_size"][0]
    t_freq, t_mid = s("t_embedder.mlp.0.weight")
    return {
        "all_f_patch_size": list(c["all_f_patch_size"]),
        "all_patch_size": list(c["all_patch_size"]),
        "axes_dims": list(c["axes_dims"]),
        "axes_lens": list(c["axes_lens"]),
        "cap_feat_dim": s("cap_embedder.1.weight")[0],
        "dim": dim,
        "in_channels": x_in // (patch * patch * f_patch),
        "n_heads": dim // head_dim,
        "n_kv_heads": (qkv_out - dim) // 2 // head_dim,
        "n_layers": view.count("layers"),
        "n_refiner_layers": view.count("noise_refiner"),
        "norm_eps": c["norm_eps"],
        "qk_norm": view.has("layers.0.attention.q_norm.weight"),
        "rope_theta": c["rope_theta"],
        "t_scale": c["t_scale"],
        "head_dim": head_dim,
        "ffn_dim": s("layers.0.feed_forward.w1.weight")[1],
        "adaln_dim": s("layers.0.adaLN_modulation.0.weight")[0],
        "t_freq_dim": t_freq,
        "t_mid_dim": t_mid,
        "final_norm_eps": c["final_norm_eps"],
        "seq_multiple": c["seq_multiple"],
    }


FAMILY = Family(
    name="z_image",
    title="Z-Image",
    order=10,
    arch_strings=frozenset({"lumina2", None}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=None,
    fp32_keep=("t_embedder.", "cap_embedder.0."),
    identity_tensors=IDENTITY_TENSORS,
    variants=VARIANTS,
)
