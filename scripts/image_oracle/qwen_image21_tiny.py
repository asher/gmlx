"""A tiny Qwen-Image 2.1 transformer for the oracle self-test and the gmlx
CPU tests: its config, tensor names and shapes, weights and inputs.

Weights and inputs follow from integer arithmetic alone, so the oracle
environment and the gmlx tests form the same arrays without a file
between them, and a manifest holds their sha256 to show it. Every weight
is a value that bfloat16 holds, so a load in either dtype keeps it.

This module needs only numpy and the standard library.
"""

from __future__ import annotations

import hashlib

import numpy as np

# The official config keys at a size the CPU runs in milliseconds: two
# blocks, two heads of 16, and rotary axes that sum to the head dim.
TINY = {
    "patch_size": 1, "in_channels": 8, "out_channels": 8, "num_layers": 2,
    "attention_head_dim": 16, "num_attention_heads": 2,
    "context_in_dim": 24, "mlp_ratio": 3, "axes_dims_rope": [4, 6, 6],
    "eps": 1e-6, "causal_condition": True,
}
T_FREQ = 256                    # the reference fixes the sinusoid width
PREFIX = "model.diffusion_model."

# Encoder rows, image slots and reference shapes (height, width) in latent
# tokens. A reference of h x w tokens fills h * w / 4 consecutive slots.
CASES = {
    "text": (7, (), ()),
    "ref1": (9, (3, 4), ((2, 4),)),
    "ref2": (10, (2, 4, 5), ((2, 2), (4, 2))),
    "adjacent": (8, (1, 2, 3), ((2, 2), (4, 2))),
    "wide": (40, (*range(5, 21), *range(30, 36)), ((8, 8), (4, 6))),
}
TARGET = (4, 6)                 # height, width in latent tokens


def shapes(cfg: dict = TINY, fused: bool = True) -> dict:
    """Tensor name to shape in numpy order, output dim first, for the
    official config keys ``cfg``."""
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    ffn = cfg["mlp_ratio"] * dim
    ctx = cfg["context_in_dim"]
    patch = cfg["patch_size"] ** 2
    out = {
        "img_in.weight": (dim, cfg["in_channels"] * patch),
        "txt_in.text_norm.weight": (ctx,),
        "txt_in.in_layer.weight": (dim, ctx),
        "txt_in.out_layer.weight": (dim, dim),
        "time_text_embed.timestep_embedder.linear_1.weight": (dim, T_FREQ),
        "time_text_embed.timestep_embedder.linear_2.weight": (dim, dim),
        "modulation.1.weight": (4 * dim, dim),
        "norm_out.linear.weight": (dim, dim),
        "proj_out.weight": (cfg["out_channels"] * patch, dim),
    }
    for i in range(cfg["num_layers"]):
        b = f"transformer_blocks.{i}."
        for n in ("to_q", "to_k", "to_v", "to_out.0"):
            out[f"{b}attn.{n}.weight"] = (dim, dim)
        for n in ("norm_q", "norm_k"):
            out[f"{b}attn.{n}.weight"] = (cfg["attention_head_dim"],)
        if fused:
            out[f"{b}img_mlp.gate_up.weight"] = (2 * ffn, dim)
        else:
            out[f"{b}img_mlp.gate_layer.weight"] = (ffn, dim)
            out[f"{b}img_mlp.proj.weight"] = (ffn, dim)
        out[f"{b}img_mlp.out.weight"] = (dim, ffn)
    return out


def _mix(label: str, n: int) -> np.ndarray:
    """``n`` uint64 values of the splitmix64 sequence seeded by ``label``."""
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],
                          "little")
    with np.errstate(over="ignore"):
        z = (np.arange(1, n + 1, dtype=np.uint64)
             * np.uint64(0x9E3779B97F4A7C15) + np.uint64(seed))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def uniform(label: str, shape, bits: int = 8) -> np.ndarray:
    """float32 values k / 2 ** bits - 0.5 for integers k of ``bits`` bits,
    which are exact in float32 and, at 8 bits, in bfloat16."""
    n = int(np.prod(shape, dtype=np.int64))
    k = (_mix(label, n) >> np.uint64(64 - bits)).astype(np.float64)
    return (k / float(1 << bits) - 0.5).astype(np.float32).reshape(shape)


def weights(seed: int = 0, fused: bool = True, cfg: dict = TINY) -> dict:
    """float32 weights by tensor name. Matrices lie in [-0.25, 0.25), the
    q and k norm weights in [0.75, 1.25) and the zero-centered text norm
    weight in [-0.25, 0.25)."""
    out = {}
    for name, shape in shapes(cfg, fused=True).items():
        label = f"weight {seed} {name}"
        if name.endswith(("norm_q.weight", "norm_k.weight")):
            a = uniform(label, shape, 6) * np.float32(0.5) + np.float32(1.0)
        else:
            a = uniform(label, shape) * np.float32(0.5)
        out[name] = a.astype(np.float32)
    return out if fused else unfuse(out)


def unfuse(w: dict) -> dict:
    """The same weights with each fused gate_up split into its pair. The
    gate rows come first."""
    out = {}
    for name, a in w.items():
        if name.endswith("img_mlp.gate_up.weight"):
            stem = name[:-len("gate_up.weight")]
            gate, up = np.split(a, 2, axis=0)
            out[stem + "gate_layer.weight"] = gate
            out[stem + "proj.weight"] = up
        else:
            out[name] = a
    return out


def features(case: str, seed: int = 0, cfg: dict = TINY) -> np.ndarray:
    """Encoder rows (L, context dim) of a case, in [-1.5, 1.5)."""
    n_rows = CASES[case][0]
    return uniform(f"features {seed} {case}",
                   (n_rows, cfg["context_in_dim"]), 24) * np.float32(3.0)


def ref_latents(case: str, seed: int = 0, cfg: dict = TINY) -> tuple:
    """The reference latents (C, h, w) of a case."""
    return tuple(
        uniform(f"reference {seed} {case} {i}",
                (cfg["in_channels"], h, w), 24) * np.float32(3.0)
        for i, (h, w) in enumerate(CASES[case][2]))


def latents(seed: int = 0, batch: int = 1, size=TARGET,
            cfg: dict = TINY) -> np.ndarray:
    """Target latents (B, C, H, W)."""
    return uniform(f"latents {seed} {tuple(size)}",
                   (batch, cfg["in_channels"], *size), 24) * np.float32(3.0)


def image_mask(case: str) -> np.ndarray:
    """(L,) bool, true at the image slots of a case's encoder rows."""
    n_rows, slots, _ = CASES[case]
    mask = np.zeros(n_rows, bool)
    mask[list(slots)] = True
    return mask


def sha256(arrays: dict) -> dict:
    """Name to sha256 of each array's little-endian bytes."""
    return {k: hashlib.sha256(np.ascontiguousarray(
        np.asarray(v).astype(np.asarray(v).dtype.newbyteorder("<"))
    ).tobytes()).hexdigest() for k, v in sorted(arrays.items())}
