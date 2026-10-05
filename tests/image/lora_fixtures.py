"""Synthetic LoRA adapters for the image tests, written in each dialect.

An adapter is a mapping of target name, in the diffusers layout, to
(down, up, alpha): down (rank, in) and up (out, rank) as float32 numpy
arrays, and alpha a float or None. ``write_adapter`` spells its keys as
one dialect does and writes a safetensors file."""
from __future__ import annotations

import numpy as np

# name -> (prefix, underscored, down suffix, up suffix, writes alpha)
DIALECTS = {
    "kohya": ("lora_unet_", True, ".lora_down.weight", ".lora_up.weight",
              True),
    "diffusers": ("transformer.", False, ".lora_A.weight", ".lora_B.weight",
                  False),
    "peft": ("base_model.model.", False, ".lora_A.default.weight",
             ".lora_B.default.weight", False),
    "onetrainer": ("transformer.", False, ".lora_down.weight",
                   ".lora_up.weight", True),
}


def make_pair(rng, out: int, inp: int, rank: int, alpha=None,
              scale: float = 0.2):
    down = (rng.standard_normal((rank, inp)) * scale).astype(np.float32)
    up = (rng.standard_normal((out, rank)) * scale).astype(np.float32)
    return down, up, alpha


def tiny_z_adapter(seed: int, alpha=None, wide: bool = False) -> dict:
    """An adapter for the tiny Z-Image DiT of image_fixtures: q, k and v of
    one block at three ranks, and plain leaves. ``wide`` adds the leaves
    that feed the modulation and caption memos and the embedders."""
    from image_fixtures import TINY_Z_CONFIG as c

    dim, ffn = c["dim"], c["ffn_dim"]
    rng = np.random.default_rng(seed)
    out = {
        "layers.1.attention.to_q": make_pair(rng, dim, dim, 4, alpha),
        "layers.1.attention.to_k": make_pair(rng, dim, dim, 2, alpha),
        "layers.1.attention.to_v": make_pair(rng, dim, dim, 8, alpha),
        "layers.1.attention.to_out.0": make_pair(rng, dim, dim, 4, alpha),
        "layers.0.feed_forward.w1": make_pair(rng, ffn, dim, 4, alpha),
        "layers.0.feed_forward.w2": make_pair(rng, dim, ffn, 4, alpha),
        "noise_refiner.0.attention.to_v": make_pair(rng, dim, dim, 2, alpha),
        "context_refiner.0.feed_forward.w3": make_pair(rng, ffn, dim, 2,
                                                       alpha),
    }
    if wide:
        patch_dim = c["in_channels"] * c["all_patch_size"][0] ** 2
        out.update({
            "layers.0.adaLN_modulation.0": make_pair(
                rng, 4 * dim, c["adaln_dim"], 2, alpha),
            "t_embedder.mlp.2": make_pair(
                rng, c["adaln_dim"], c["t_mid_dim"], 2, alpha),
            "cap_embedder.1": make_pair(rng, dim, c["cap_feat_dim"], 2,
                                        alpha),
            "x_embedder": make_pair(rng, dim, patch_dim, 2, alpha),
            "final_layer.linear": make_pair(rng, patch_dim, dim, 2, alpha),
        })
    return out


def dialect_keys(dialect: str, adapter: dict) -> dict:
    """The key to array mapping of ``adapter`` in ``dialect``. A dialect
    that stores no alpha drops it, so its pairs must carry None."""
    prefix, under, down_s, up_s, with_alpha = DIALECTS[dialect]
    out = {}
    for target, (down, up, alpha) in adapter.items():
        stem = prefix + (target.replace(".", "_") if under else target)
        out[stem + down_s] = down
        out[stem + up_s] = up
        if alpha is not None:
            assert with_alpha, f"{dialect} stores no alpha"
            out[stem + ".alpha"] = np.asarray(alpha, np.float32)
    return out


def write_keys(path, keys: dict, metadata: dict | None = None) -> str:
    import mlx.core as mx

    mx.save_safetensors(str(path), {k: mx.array(v) for k, v in keys.items()},
                        metadata=metadata or {"format": "pt"})
    return str(path)


def write_adapter(path, dialect: str, adapter: dict) -> str:
    return write_keys(path, dialect_keys(dialect, adapter))


def without_alpha(adapter: dict) -> dict:
    return {k: (d, u, None) for k, (d, u, _) in adapter.items()}


def reference_delta(parts, dims: tuple) -> np.ndarray:
    """The float64 weight delta of (down, up, alpha, band, strength) rows
    on a weight of ``dims`` (out, in): the sum over rows of strength times
    alpha / rank times up @ down on the band, where a missing alpha gives
    the factor 1.0."""
    delta = np.zeros(dims, np.float64)
    for down, up, alpha, band, strength in parts:
        d = np.asarray(down, np.float64)
        u = np.asarray(up, np.float64)
        f = float(strength) * (1.0 if alpha is None
                               else float(alpha) / d.shape[0])
        prod = f * (u @ d)
        if band is None:
            delta += prod
        elif band[0] == 0:
            delta[band[1]:band[1] + band[2], :] += prod
        else:
            delta[:, band[1]:band[1] + band[2]] += prod
    return delta


def rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    got = np.asarray(got, np.float64)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
