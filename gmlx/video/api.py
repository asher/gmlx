"""The video primitives over a loaded DiT: one denoise step, and a run.

A step takes both streams at their own sigmas: the video latent (1, C, T,
H, W) and the audio latent (1, C, 2, T), and returns the velocity of each
(noise minus data). A run steps each stream by Euler on its own schedule,
in float32.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import numpy as np

from gmlx.image import pacing

from . import schedule

_F32 = np.float32


def stream_time(sigma) -> float:
    """The DiT's time of a stream at ``sigma``: 1 - sigma in float32."""
    return float(_F32(1) - _F32(sigma))


def denoise_step(model, video: mx.array, audio: mx.array, sigmas: dict,
                 text: mx.array, layout, *, cond_video=(), cond_audio=(),
                 plan=None, taps: dict | None = None) -> tuple:
    """The velocity of each stream at ``sigmas`` {"video", "audio"}, with
    the text rows of ``model.caption``."""
    tv, ta = stream_time(sigmas["video"]), stream_time(sigmas["audio"])
    return model(video, audio, tv, ta, text, layout, plan=plan,
                 cond_video=cond_video, cond_audio=cond_audio, taps=taps)


def _eval_point(arrays) -> None:
    mx.eval(*arrays)


def sample(model, video: mx.array, audio: mx.array, text: mx.array, layout,
           *, steps: int, shifts: dict, cond_video=(), cond_audio=(),
           on_step: Callable | None = None,
           pacer: Callable | None = _eval_point) -> tuple:
    """Video and audio latents after ``steps`` forwards from the noise
    ``video`` and ``audio``. ``on_step(i, n)`` runs after each forward.
    ``pacer`` runs at each pause point inside a forward; the default
    evaluates there, which bounds the graph to one block."""
    sig = schedule.stream_sigmas(steps, shifts)
    sv, sa = sig["video"], sig["audio"]
    xv, xa = video.astype(mx.float32), audio.astype(mx.float32)
    n = len(sv) - 1
    with pacing.paced(pacer):
        for i in range(n):
            plan = model.plan(layout, stream_time(sv[i]), stream_time(sa[i]))
            vv, va = denoise_step(model, xv, xa,
                                  {"video": sv[i], "audio": sa[i]}, text,
                                  layout, cond_video=cond_video,
                                  cond_audio=cond_audio, plan=plan)
            xv = xv + float(sv[i + 1] - sv[i]) * vv
            xa = xa + float(sa[i + 1] - sa[i]) * va
            mx.eval(xv, xa)
            if on_step is not None:
                on_step(i, n)
    return xv, xa
