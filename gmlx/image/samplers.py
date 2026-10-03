"""Samplers over flow-matching sigmas, and the noise source.

A sampler integrates the velocity a DiT returns from one sigma to the next,
with latents in float32. Its history, for a multistep sampler, lives on
the sampler object, which ``sample`` creates per run. Sampler names follow
ComfyUI's spelling.

Noise comes from numpy's PCG64 on the host, so a seed gives the same
latents on every machine and MLX version. gmlx seeds are not comparable
with diffusers or ComfyUI seeds.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .schedule import check_sigmas


def noise_from_seed(seed: int, shape) -> np.ndarray:
    """float32 standard normal values of ``shape`` in C order, NCHW for
    latents, from ``Generator(PCG64(seed))``."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, not {seed}")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.standard_normal(tuple(shape), dtype=np.float32)


class Sampler:
    """One sampling run over a fixed sigma schedule. ``step(x, v, i)``
    moves float32 latents ``x`` from ``sigmas[i]`` to ``sigmas[i + 1]``
    with the velocity ``v`` the model returned at ``sigmas[i]``."""

    name = ""

    def __init__(self, sigmas):
        self.sigmas = check_sigmas(sigmas)

    @property
    def steps(self) -> int:
        return len(self.sigmas) - 1

    def step(self, x: mx.array, v: mx.array, i: int) -> mx.array:
        raise NotImplementedError


class Euler(Sampler):
    """``x + (sigma_next - sigma) * v``. The difference is formed in
    float32 on the host, then the product and the sum run in float32."""

    name = "euler"

    def step(self, x: mx.array, v: mx.array, i: int) -> mx.array:
        if not 0 <= i < self.steps:
            raise IndexError(f"step {i} outside 0..{self.steps - 1}")
        dt = self.sigmas[i + 1] - self.sigmas[i]
        return x.astype(mx.float32) + mx.array(dt, mx.float32) * v.astype(
            mx.float32)


SAMPLERS = {Euler.name: Euler}


def make_sampler(name: str, sigmas) -> Sampler:
    try:
        cls = SAMPLERS[name]
    except KeyError:
        known = ", ".join(sorted(SAMPLERS))
        raise ValueError(f"unknown sampler {name!r}; known: {known}") from None
    return cls(sigmas)


def img2img_start(steps: int, denoise: float) -> int:
    """The first schedule index of an img2img run over ``steps`` steps, as
    diffusers computes it from the strength ``denoise``."""
    return int(max(steps - min(steps * denoise, steps), 0))


def noised(sigma, noise: mx.array, latents: mx.array) -> mx.array:
    """``sigma * noise + (1 - sigma) * latents`` in float32, the flow
    interpolation of clean latents to ``sigma``."""
    s = mx.array(np.float32(sigma), mx.float32)
    return s * noise.astype(mx.float32) + (1 - s) * latents.astype(mx.float32)


def inpaint_blend(x: mx.array, mask: mx.array, keep: mx.array) -> mx.array:
    """``(1 - mask) * keep + mask * x``: the masked region takes the new
    latents, and the rest keeps ``keep``."""
    return (1 - mask) * keep + mask * x
