"""Reader rows for the video VAEs, read through the image VAE reader.

MiniMax-H3 decodes with two files from Comfy-Org, read as published: the
video VAE (float16) and the audio VAE (float32, weight norm folded).
Previews decode with madebyollin's TAEH3. Each file is identified by its
tensor names and shapes and read under its own names. The rows say how a file
tensor in torch layout becomes the module's: a Conv3d kernel (O, I, T,
kH, kW) keeps its frames as (O, T, kH, kW, I), a Conv1d kernel (O, I, K)
becomes (O, K, I), a ConvTranspose1d kernel (I, O, K) becomes (O, K, I),
and a Snake alpha stored as (1, C, 1) becomes (C,).
"""

from __future__ import annotations

import mlx.core as mx

from gmlx.image.vae import reader as image_reader
from gmlx.image.vae.reader import VaeRow

from . import h3_audio, h3_video, tiny


def _video_layout(name: str, shape: tuple):
    if len(shape) == 5:
        o, i, t, h, w = shape
        return (0, 2, 3, 4, 1), (o, t, h, w, i)
    return None, tuple(shape)


def _audio_layout(name: str, shape: tuple):
    if len(shape) != 3:
        return None, tuple(shape)
    if name.endswith(".alpha"):
        return None, (shape[1],)
    if name.startswith("decoder.ups."):
        i, o, k = shape
        return (1, 2, 0), (o, k, i)
    o, i, k = shape
    return (0, 2, 1), (o, k, i)


def _video_unused(cfg) -> dict:
    return {"decoder.mask_token": (1, 1, cfg.dim)}


def _audio_unused(cfg) -> dict:
    return {"logs_proj.weight": (cfg.z, cfg.z, 1), "logs_proj.bias": (cfg.z,)}


def _video_refusal(header) -> str | None:
    if header.metadata.get("h3_t1_direct") == "true":
        return (f"{header.path} sets h3_t1_direct in its metadata: its "
                f"decoder is tuned to decode a still from its latent alone, "
                f"and gmlx decodes a still as the first latent of a clip")
    return None


VAES = {
    "h3": VaeRow(
        "the MiniMax-H3 video VAE", h3_video.config_from_shapes,
        h3_video.H3VideoVAE, 24, expected_unused=_video_unused,
        fp32_keep=("latents_mean", "latents_std"), layout=_video_layout,
        dtypes=(mx.float16, mx.float32), own_names=True,
        refusal=_video_refusal),
    "h3_audio": VaeRow(
        "the MiniMax-H3 audio VAE", h3_audio.config_from_shapes,
        h3_audio.H3AudioVAE, 32, expected_unused=_audio_unused,
        layout=_audio_layout, dtypes=(mx.float32,), own_names=True),
    "h3_tiny": VaeRow(
        "TAEH3, the tiny MiniMax-H3 preview decoder", tiny.config_from_shapes,
        tiny.TAEH3, tiny.LATENT, expected_unused=tiny.expected_unused,
        dtypes=(mx.float16, mx.float32), own_names=True),
}


def load_vae(path, *, family: str, dtype):
    """The video or audio VAE module for ``family`` from its safetensors
    file, with weights in ``dtype``."""
    return image_reader.load_vae(path, family=family, dtype=dtype, rows=VAES)
