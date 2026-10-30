"""The seeded models and inputs of the MiniMax-H3 check 5 self-test.

``values`` gives seeded float32 values for one tensor from its name and
torch shape alone, and ``write_safetensors`` a file in the layout of the
published ones: names sorted, no metadata. The oracle script records each
model's names and shapes from ComfyUI's classes in the manifest, so the
gmlx tests write the same file from the manifest and compare its sha256.

The video VAE keeps the published levels and strides at narrower widths,
with a ViT decoder of two layers and two heads. The audio VAE keeps the
published strides and upsample rates at narrower widths. TAEH3 runs at
its published size, which is small.

``clip`` and ``wave`` are the synthetic inputs of both the self-test and
the check on the published files.

This module needs only numpy and the standard library.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib

import numpy as np

SEED = 55
# MiniMaxH3VideoVAE arguments, and the ViT3DDecoder that replaces the
# published decoder.
VIDEO = {"ch": 32, "ch_mult": (1, 1, 2, 2, 2, 2), "num_res_blocks": 2}
VIDEO_DECODER = {"num_layers": 2, "heads": 2, "dim_head": 64}
# MiniMaxH3AudioVAE arguments.
AUDIO = {"encoder_dim": 4, "latent_dim": 64, "decoder_dim": 256}


def _rng(name: str, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def values(name: str, shape, seed: int = SEED) -> np.ndarray:
    """Seeded float32 values for the tensor ``name`` of torch shape
    ``shape``. Kernels and matrices are scaled by their fan in, norms sit
    near one, filters are positive and sum to one, and the latent
    statistics are a scale near one and a small shift."""
    shape = tuple(int(s) for s in shape)
    rng = _rng(name, seed)
    leaf = name.rsplit(".", 1)[-1]
    n = rng.standard_normal(shape)
    if leaf == "filter":
        v = rng.uniform(0.1, 1.0, shape)
        v = v / v.sum(axis=-1, keepdims=True)
    elif leaf == "latents_std":
        v = rng.uniform(0.5, 1.5, shape)
    elif leaf == "latents_mean":
        v = 0.2 * n
    elif leaf in ("scale1", "scale2"):
        v = 0.3 + 0.1 * n
    elif leaf in ("alpha", "beta"):
        # Snake1d reads alpha as is, SnakeBeta as a log: both stay near 1.
        v = rng.uniform(0.6, 1.4, shape) if "activations" not in name \
            and "activation_post" not in name else 0.1 * n
    elif leaf in ("register_tokens", "mask_token"):
        v = n
    elif "norm" in name.rsplit(".", 2)[-2] and leaf == "weight":
        v = 1.0 + 0.1 * n
    elif leaf in ("bias", "q_bias", "v_bias", "zero_k_bias"):
        v = 0.05 * n
    elif len(shape) >= 2:
        fan_in = int(np.prod(shape[1:]))
        if ".ups." in name:
            # ConvTranspose1d: (in, out, k), stride about k / 2.
            fan_in = max(1, shape[0] * shape[2] // 2)
        v = n / np.sqrt(fan_in)
    else:
        v = 1.0 + 0.1 * n
    return v.astype(np.float32)


def weights(shapes: dict, seed: int = SEED) -> dict:
    return {k: values(k, s, seed) for k, s in shapes.items()}


def write_safetensors(path, tensors: dict, dtype: str = "F32") -> str:
    """Write ``tensors`` with every tensor in ``dtype`` (F32 or F16),
    names sorted, no metadata, and return the file's sha256."""
    np_dt = {"F32": np.float32, "F16": np.float16}[dtype]
    header, blobs, off = {}, [], 0
    for k in sorted(tensors):
        b = np.ascontiguousarray(tensors[k], np_dt).tobytes()
        header[k] = {"dtype": dtype, "shape": list(tensors[k].shape),
                     "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    data = struct.pack("<Q", len(h)) + h + b"".join(blobs)
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def clip(frames: int, height: int, width: int, seed: int) -> np.ndarray:
    """A (1, 3, frames, height, width) float32 clip in [-1, 1]: gradients,
    a disk that moves across the frame, a band of fine stripes and a
    patch of seeded texture."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width),
                       indexing="ij")
    tex = rng.standard_normal((3, height, width))
    iy = np.arange(height)[:, None] + np.zeros((1, width))
    out = np.empty((3, frames, height, width))
    for f in range(frames):
        p = f / max(frames - 1, 1)
        img = np.stack([0.5 + 0.4 * np.sin(2 * np.pi * (1.3 * x + 0.3 * p)),
                        0.2 + 0.6 * y,
                        0.8 - 0.6 * x * y])
        cx, cy = 0.2 + 0.6 * p, 0.4 + 0.1 * np.sin(2 * np.pi * p)
        disk = (x - cx) ** 2 + (y - cy) ** 2 < 0.12 ** 2
        img[:, disk] = np.array([0.9, 0.2, 0.1])[:, None]
        band = (y > 0.08) & (y < 0.2)
        img[:, band] = 0.5 + 0.5 * np.sin(2 * np.pi * (iy[band] + f) / 3.0)
        zone = (x > 0.55) & (y > 0.6)
        img[:, zone] = 0.5 + 0.15 * tex[:, zone]
        out[:, f] = img
    return (np.clip(out, 0, 1) * 2 - 1).astype(np.float32)[None]


def wave(samples: int, seed: int, rate: int = 32000) -> np.ndarray:
    """A (1, 2, samples) float32 stereo waveform in [-1, 1]: a chirp on
    the left, a decaying tone on the right and a little seeded noise on
    both."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / rate
    w = np.stack([0.4 * np.sin(2 * np.pi * (220 + 300 * t) * t),
                  0.3 * np.sin(2 * np.pi * 330 * t) * np.exp(-t)])
    w += 0.01 * rng.standard_normal(w.shape)
    return np.clip(w, -1, 1)[None].astype(np.float32)


def digest(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
