"""Sigma schedules of a video DiT, one per stream.

Each stream shifts the same float32 grid ``linspace(1, 0, steps + 1)`` by
its own static shift and drops repeats, as the reference scheduler does,
so ``steps`` counts DiT forwards. The audio stream's sigma at a video
sigma is ``time_shift``.

A variant distilled on fixed timesteps (``rungs``, on the 1000-step clock)
takes them over 1000, then 0, as its grid in place of the even one, and
each stream shifts that grid in float64 and rounds once to float32, as
FastVideo's ``from_dmd_steps`` does.
"""

from __future__ import annotations

import numpy as np

from gmlx.image.schedule import linspace_f32, shift_static

_F32 = np.float32


def sigmas(steps: int, shift: float, rungs: tuple = ()) -> np.ndarray:
    """float32 sigmas from 1 to 0, ``steps`` forwards and a final zero,
    with consecutive repeats from the shift dropped. With ``rungs``, one
    per forward, the sigmas start from the first rung's instead of 1."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1, not {steps}")
    if rungs:
        if len(rungs) != steps:
            raise ValueError(f"{len(rungs)} rungs are not {steps} steps")
        base = np.array([r / 1000.0 for r in rungs] + [0.0], np.float64)
        s = (shift * base / (1.0 + (shift - 1.0) * base)).astype(_F32)
    else:
        s = shift_static(linspace_f32(1.0, 0.0, steps + 1), shift)
    keep = np.concatenate([[True], s[1:] != s[:-1]])
    return s[keep].astype(_F32)


def stream_sigmas(steps: int, shifts: dict, rungs: tuple = ()) -> dict:
    """``sigmas`` per stream, each on the same grid. Every stream has the
    same count, since the shift is monotonic and repeats only where the
    grid does."""
    out = {k: sigmas(steps, s, rungs) for k, s in shifts.items()}
    n = {len(v) for v in out.values()}
    if len(n) != 1:
        raise ValueError(f"streams disagree on the forward count: "
                         f"{ {k: len(v) - 1 for k, v in out.items()} }")
    return out


def time_shift(sigma: float, from_shift: float, to_shift: float) -> float:
    """The sigma under ``to_shift`` of the grid point that ``from_shift``
    takes to ``sigma``, rounded as ComfyUI's forward derives the audio
    sigma from the video one: the shift arithmetic in float64, each op on
    the sigma in float32."""
    s, a, b = _F32(sigma), _F32(from_shift), _F32(to_shift)
    base = s / (a + s * _F32(1.0 - from_shift))
    return float(b * base / (_F32(1) + _F32(to_shift - 1.0) * base))
