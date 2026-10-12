"""The tiny Qwen-Image 2.1 VAEs of the check 5 self-test.

``shapes`` gives the tensor names and torch shapes of an
AutoencoderKLQwenImage21 from its config, ``weights`` seeded float32 values
for them, and ``write_safetensors`` a file in the layout of the official
one: every tensor F32, names sorted, no metadata. The file's bytes follow
from the config and the seed alone, so the oracle script and the gmlx tests
each write it and compare its sha256.

``CASES`` are further tiny configs with other widths, levels and time
flags, and ``case_inputs`` their seeded latents and RGBA image, so both
sides also compute the same inputs.

This module needs only numpy and the standard library.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib

import numpy as np

# The official structure at widths of a few tens: five levels, the video
# layout's time flags, 64 latent channels and RGBA.
TINY = {
    "base_dim": 4, "decoder_base_dim": 6, "z_dim": 64,
    "dim_mult": [1, 2, 4, 8, 8], "num_res_blocks": 2,
    "temperal_downsample": [False, True, True, True],
    "is_residual": True, "in_channels": 4, "out_channels": 4,
}
SEED = 21
# Gain on decoder.conv_out, so that part of the decode leaves [-1, 1] and
# the clamp acts.
OUT_GAIN = 1.2

# name -> (changes to TINY, (height, width)). The first five keep the
# official levels and time flags, which is all a file can say. In the rest
# the config alone gives the levels or the flags.
CASES = {
    "one_resnet": ({"base_dim": 6, "decoder_base_dim": 4,
                    "num_res_blocks": 1}, (32, 32)),
    "three_resnets": ({"num_res_blocks": 3}, (32, 32)),
    # A doubling where time is not halved: two pixels of a block per group.
    "first_multiplier_2": ({"dim_mult": [2, 2, 4, 8, 8],
                            "decoder_base_dim": 3}, (32, 32)),
    # The last down block has no resample and halves its channels.
    "last_level_narrows": ({"dim_mult": [1, 2, 4, 8, 4]}, (32, 32)),
    # One channel per group on the way down, one repeat on the way up.
    "step_of_8": ({"dim_mult": [1, 1, 1, 8, 8]}, (32, 32)),
    "other_time_flags": ({"temperal_downsample": [True, False, True, False]},
                         (32, 48)),
    "no_time_flags": ({"temperal_downsample": [False] * 4}, (32, 32)),
    "all_time_flags": ({"temperal_downsample": [True] * 4}, (32, 32)),
    "three_levels": ({"dim_mult": [1, 2, 2],
                      "temperal_downsample": [False, True]}, (16, 24)),
    "two_levels": ({"dim_mult": [2, 2], "temperal_downsample": [True],
                    "decoder_base_dim": 3}, (8, 16)),
}


def case_config(name: str) -> dict:
    return {**TINY, **CASES[name][0]}


def case_inputs(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Seeded model-space latents (1, z, h, w) and an RGBA image
    (1, 4, H, W) in [-1, 1] for a case, float32."""
    config, (height, width) = case_config(name), CASES[name][1]
    s = 2 ** (len(config["dim_mult"]) - 1)
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode())])
    latents = rng.standard_normal(
        (1, config["z_dim"], height // s, width // s))
    image = rng.uniform(-1.0, 1.0, (1, config["in_channels"], height, width))
    return latents.astype(np.float32), image.astype(np.float32)


def shapes(config: dict) -> dict:
    """Tensor name to torch shape, time_conv tensors included."""
    out = {}

    def conv(name, cin, cout, k):
        out[f"{name}.weight"] = (cout, cin, k, k)
        out[f"{name}.bias"] = (cout,)

    def resnet(name, cin, cout):
        out[f"{name}.norm1.gamma"] = (cin, 1, 1, 1)
        conv(f"{name}.conv1", cin, cout, 3)
        out[f"{name}.norm2.gamma"] = (cout, 1, 1, 1)
        conv(f"{name}.conv2", cout, cout, 3)
        if cin != cout:
            conv(f"{name}.conv_shortcut", cin, cout, 1)

    def mid(name, c):
        resnet(f"{name}.resnets.0", c, c)
        out[f"{name}.attentions.0.norm.gamma"] = (c, 1, 1)
        conv(f"{name}.attentions.0.to_qkv", c, 3 * c, 1)
        conv(f"{name}.attentions.0.proj", c, c, 1)
        resnet(f"{name}.resnets.1", c, c)

    mult, z = list(config["dim_mult"]), config["z_dim"]
    n, res = len(mult), config["num_res_blocks"]
    down_t = list(config["temperal_downsample"])
    d = [config["base_dim"] * m for m in [1] + mult]
    conv("encoder.conv_in", config["in_channels"], d[0], 3)
    for i in range(n):
        blk = f"encoder.down_blocks.{i}"
        for j in range(res):
            resnet(f"{blk}.resnets.{j}", d[i] if j == 0 else d[i + 1],
                   d[i + 1])
        if i < n - 1:
            conv(f"{blk}.downsampler.resample.1", d[i + 1], d[i + 1], 3)
            if down_t[i]:
                conv(f"{blk}.downsampler.time_conv", d[i + 1], d[i + 1], 1)
    mid("encoder.mid_block", d[-1])
    out["encoder.norm_out.gamma"] = (d[-1], 1, 1, 1)
    conv("encoder.conv_out", d[-1], 2 * z, 3)
    conv("quant_conv", 2 * z, 2 * z, 1)
    conv("post_quant_conv", z, z, 1)

    e = [config["decoder_base_dim"] * m for m in [mult[-1]] + mult[::-1]]
    up_t = down_t[::-1]
    conv("decoder.conv_in", z, e[0], 3)
    mid("decoder.mid_block", e[0])
    for i in range(n):
        blk = f"decoder.up_blocks.{i}"
        for j in range(res + 1):
            resnet(f"{blk}.resnets.{j}", e[i] if j == 0 else e[i + 1],
                   e[i + 1])
        if i < n - 1:
            conv(f"{blk}.upsampler.resample.1", e[i + 1], e[i + 1], 3)
            if up_t[i]:
                conv(f"{blk}.upsampler.time_conv", e[i + 1], 2 * e[i + 1], 1)
    out["decoder.norm_out.gamma"] = (e[-1], 1, 1, 1)
    conv("decoder.conv_out", e[-1], config["out_channels"], 3)
    return out


def weights(config: dict = TINY, seed: int = SEED) -> dict:
    """Seeded float32 values by name: kernels at 1 / sqrt(fan in), gamma
    near 1 and biases near 0. Each tensor has its own stream, keyed by the
    seed and its name."""
    out = {}
    for name, shape in shapes(config).items():
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        v = rng.standard_normal(shape)
        if name.endswith(".gamma"):
            v = 1.0 + 0.1 * v
        elif name.endswith(".bias"):
            v = 0.1 * v
        else:
            v = v / np.sqrt(np.prod(shape[1:]))
            if name == "decoder.conv_out.weight":
                v = v * OUT_GAIN
        out[name] = v.astype(np.float32)
    return out


def digest(arrays) -> str:
    """sha256 over float32 arrays in the order given."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, "<f4").tobytes())
    return h.hexdigest()


def write_safetensors(path, tensors: dict) -> str:
    """``tensors`` as an F32 safetensors file with sorted names."""
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        raw = np.ascontiguousarray(tensors[name], "<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensors[name].shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return str(path)
