"""The image primitives over a loaded pipeline.

Latents are ``(B, C, H, W)`` float32 in model space. ``denoise_step``
evaluates the DiT once per row and returns one float32 velocity per row,
such that ``x_next = x + (sigma_next - sigma) * v``. It applies no
guidance and keeps no step counter. ``guide`` combines a positive and a
negative velocity. Rows run as sequential forwards.

``sample`` runs the whole loop inside ``memfit.bounded_cache``, so the
buffers one step frees go back to the system instead of the MLX cache,
and it evaluates the latents after every step. It calls ``on_step(i, x)``
after each update and reads ``should_stop()`` before each step; a stop
raises ``SampleStopped`` with the latents reached.

``vae_decode`` and ``vae_encode`` map between latents and images in
``[0, 1]``, whole or in tiles as ``vae.tiling`` decides.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import memfit
from .registry import Family
from .samplers import (img2img_start, inpaint_blend, make_sampler, noised,
                       noise_from_seed)
from .schedule import check_sigmas, zimage_sigmas


@dataclass(frozen=True)
class Conditioning:
    """One encoded prompt: the encoder tap ``(L, D)`` of its tokens,
    unpadded, and its conditioning cache key."""
    features: mx.array
    key: str = ""


@dataclass(frozen=True)
class SamplerDefaults:
    steps: int
    cfg: float


# diffusers' Z-Image guidance_scale g is cfg = 1 + g.
_DEFAULTS = {
    ("z_image", "turbo"): SamplerDefaults(steps=8, cfg=1.0),
    ("z_image", "base"): SamplerDefaults(steps=50, cfg=6.0),
}


@dataclass(eq=False)
class ImagePipeline:
    dit: nn.Module
    family: Family
    variant: str
    vae: object | None = None
    vae_scale: int = 8                      # image pixels per latent pixel
    patch_tables: dict = field(default_factory=dict)    # id -> table

    @property
    def defaults(self) -> SamplerDefaults:
        try:
            return _DEFAULTS[(self.family.name, self.variant)]
        except KeyError:
            raise ValueError(
                f"no sampler defaults for {self.family.title} "
                f"{self.variant}; pass steps and cfg") from None

    def sigmas(self, steps: int) -> np.ndarray:
        if self.family.name == "z_image":
            shift = self.family.constants.values["scheduler_shift"]
            return zimage_sigmas(steps, shift)
        raise ValueError(f"no schedule for {self.family.title}; pass sigmas")


class SampleStopped(RuntimeError):
    """``should_stop`` ended a run before ``step``. ``latents`` are the
    latents reached, at ``sigmas[step]``."""

    def __init__(self, step: int, latents: mx.array):
        super().__init__(f"sampling stopped before step {step}")
        self.step = step
        self.latents = latents


def _check_table(pipe: ImagePipeline, patch_table) -> None:
    if patch_table is not None and patch_table not in pipe.patch_tables:
        raise ValueError(f"unknown patch table {patch_table!r}")


def denoise_step(pipe: ImagePipeline, latents, sigma, conds, *,
                 patch_table=None) -> mx.array:
    """The velocity of every row of ``latents`` at ``sigma``, a scalar or
    one value per row, under ``conds``, one Conditioning or one per row."""
    _check_table(pipe, patch_table)
    x = mx.array(latents).astype(mx.float32)
    if x.ndim != 4:
        raise ValueError(f"latents must be (B, C, H, W), not {x.shape}")
    b = x.shape[0]
    sig = np.asarray(sigma, np.float32).reshape(-1)
    if sig.size not in (1, b):
        raise ValueError(f"{sig.size} sigmas for {b} rows")
    sig = np.broadcast_to(sig, (b,))
    rows = [conds] * b if isinstance(conds, Conditioning) else list(conds)
    if len(rows) != b:
        raise ValueError(f"{len(rows)} conditionings for {b} rows")
    out = [pipe.dit(x[i:i + 1], sig[i:i + 1], rows[i].features[None])
           for i in range(b)]
    return out[0] if b == 1 else mx.concatenate(out, axis=0)


def guide(v_pos: mx.array, v_neg: mx.array, cfg: float) -> mx.array:
    """``neg + cfg * (pos - neg)`` in float32; 1.0 means no guidance."""
    pos, neg = v_pos.astype(mx.float32), v_neg.astype(mx.float32)
    return neg + mx.array(cfg, mx.float32) * (pos - neg)


def _latent_mask(mask, shape: tuple, scale: int) -> mx.array:
    """A mask at image or latent resolution as (B or 1, 1, h, w) float32
    at latent resolution. Image masks average over each latent pixel."""
    m = np.asarray(mask, np.float32)
    if m.ndim == 2:
        m = m[None, None]
    if m.ndim != 4 or m.shape[1] != 1 or m.shape[0] not in (1, shape[0]):
        raise ValueError(f"mask must be (H, W) or (B, 1, H, W), not {m.shape}")
    h, w = shape[2:]
    if m.shape[2:] == (h * scale, w * scale):
        m = m.reshape(m.shape[0], 1, h, scale, w, scale).mean(axis=(3, 5))
    elif m.shape[2:] != (h, w):
        raise ValueError(f"mask size {m.shape[2]}x{m.shape[3]} is neither "
                         f"the image nor the latent size")
    if m.min() < 0 or m.max() > 1:
        raise ValueError("mask values must lie in [0, 1]")
    return mx.array(m)


def sample(pipe: ImagePipeline, cond: Conditioning, *,
           neg: Conditioning | None = None, cfg: float | None = None,
           height: int, width: int, steps: int | None = None,
           seed: int | None = None, noise=None, sampler: str = "euler",
           sigmas=None, init_latents=None, denoise: float = 1.0, mask=None,
           patch_table=None, on_step=None, should_stop=None) -> mx.array:
    """Sample latents for ``cond``. With ``init_latents`` the run is
    img2img from strength ``denoise``, and a ``mask`` (1 where the image
    changes) makes it inpainting. ``noise`` is (B, C, H, W) float32, else
    it comes from ``seed``, else from a random seed."""
    _check_table(pipe, patch_table)
    if cfg is None:
        cfg = pipe.defaults.cfg
    if cfg != 1.0 and neg is None:
        raise ValueError(f"cfg {cfg} needs a negative conditioning")
    unit = pipe.vae_scale * pipe.dit.cfg.patch
    if height <= 0 or width <= 0 or height % unit or width % unit:
        raise ValueError(f"height and width must be positive multiples of "
                         f"{unit}, not {height}x{width}")
    if not 0.0 < denoise <= 1.0:
        raise ValueError(f"denoise must lie in (0, 1], not {denoise}")
    if init_latents is None and (denoise < 1.0 or mask is not None):
        raise ValueError("denoise below 1 and mask need init_latents")

    if sigmas is None:
        s = pipe.sigmas(pipe.defaults.steps if steps is None else steps)
    else:
        s = check_sigmas(sigmas)
        if steps is not None and steps != len(s) - 1:
            raise ValueError(f"steps {steps} disagrees with "
                             f"{len(s) - 1} steps of sigmas")
    init = None
    if init_latents is not None:
        init = mx.array(init_latents).astype(mx.float32)
        s = s[img2img_start(len(s) - 1, denoise):]
        if len(s) < 2:
            raise ValueError(f"denoise {denoise} leaves no step")

    b = 1 if init is None else init.shape[0]
    shape = (b, pipe.dit.cfg.in_channels, height // pipe.vae_scale,
             width // pipe.vae_scale)
    if init is not None and tuple(init.shape) != shape:
        raise ValueError(f"init_latents {tuple(init.shape)} do not match "
                         f"{shape}")
    if noise is None:
        noise = noise_from_seed(secrets.randbelow(1 << 63)
                                if seed is None else seed, shape)
    elif seed is not None:
        raise ValueError("pass seed or noise, not both")
    noise = mx.array(noise).astype(mx.float32)
    if tuple(noise.shape) != shape:
        raise ValueError(f"noise {tuple(noise.shape)} does not match {shape}")
    m = None if mask is None else _latent_mask(mask, shape, pipe.vae_scale)

    x = noise if init is None else noised(s[0], noise, init)
    smp = make_sampler(sampler, s)
    with memfit.bounded_cache():
        for i in range(smp.steps):
            if should_stop is not None and should_stop():
                raise SampleStopped(i, x)
            v = denoise_step(pipe, x, s[i], cond, patch_table=patch_table)
            if cfg != 1.0:
                v = guide(v, denoise_step(pipe, x, s[i], neg,
                                          patch_table=patch_table), cfg)
            x = smp.step(x, v, i)
            if m is not None:
                x = inpaint_blend(x, m, noised(s[i + 1], noise, init))
            mx.eval(x)
            if on_step is not None:
                on_step(i, x)
    return x


def _need_vae(pipe: ImagePipeline):
    if pipe.vae is None:
        raise ValueError("this pipeline has no VAE")
    return pipe.vae


def vae_decode(pipe: ImagePipeline, latents, *, tiled: bool | None = None,
               alpha: bool = False) -> mx.array:
    """Latents to an image (B, H, W, 3) float32 in [0, 1]."""
    vae = _need_vae(pipe)
    if alpha:
        raise ValueError(f"the {pipe.family.title} VAE has no alpha channel")
    from .vae import tiling

    img = tiling.decode(vae, latents, tiled=tiled)
    return mx.clip(img / 2 + 0.5, 0.0, 1.0)


def vae_encode(pipe: ImagePipeline, image, *, tiled: bool | None = None,
               mode: str = "mode", noise=None) -> mx.array:
    """An image (B, H, W, 3) in [0, 1] to latents (B, C, h, w) float32."""
    vae = _need_vae(pipe)
    from .vae import tiling

    x = mx.array(image).astype(mx.float32) * 2 - 1
    return tiling.encode(vae, x, tiled=tiled, mode=mode, noise=noise)
