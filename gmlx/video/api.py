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


# Each pause point counts the step's rows, not the rows of the arrays it
# carries. The last 4096-row attention chunk of a 37898-row step carries
# 1034 rows, so a budget on array rows would also evaluate every point of
# a 2054-row step. At EVAL_ROWS rows or more every point evaluates, which
# keeps the large shapes and memfit's plan for them as they are. 32768 is
# under the 37710 rows of the default 1344x768 clip of 5 s before its
# caption, so no caption changes that clip. A 512x512 clip of 22 frames
# (2054 rows) evaluates every 16th point: at that size an evaluation at
# every point costs about 6 percent of a step.
EVAL_ROWS = 32768


class RowBudget:
    """The default pacer of a run of ``rows`` rows a step: it runs
    ``on_point`` at every point, then evaluates the arrays of a point when
    the points since its last evaluation, each counted as ``rows``, reach
    ``threshold`` rows. That is every ``ceil(threshold / rows)`` points."""

    def __init__(self, rows: int, threshold: int = EVAL_ROWS,
                 evaluate: Callable | None = None,
                 on_point: Callable | None = None):
        self.rows, self.threshold = int(rows), int(threshold)
        self._eval = mx.eval if evaluate is None else evaluate
        self.on_point = on_point
        self.held = 0

    def __call__(self, arrays) -> None:
        # A server's on_point gives chat its share of the GPU: while chat
        # is active it evaluates the arrays at every point and waits, so
        # evaluation falls back to every point. The count does not start
        # again after it: evaluating arrays already evaluated runs no GPU
        # work, and the first evaluation after chat goes idle comes at
        # most one budget early.
        if self.on_point is not None:
            self.on_point(arrays)
        self.held += self.rows
        if self.held >= self.threshold:
            self.held = 0
            self._eval(*arrays)

    def reset(self) -> None:
        """Start the count again after an evaluation outside the pacer."""
        self.held = 0


class Budget:
    """A pacer for a forward whose rows its caller does not know. The
    forward runs it as a ``RowBudget`` of its layout's rows
    (``forward_pacer``), with ``on_point`` at every point, such as a
    server's stop check and chat's share. Called outside a forward, it runs
    ``on_point`` and evaluates at every point."""

    def __init__(self, on_point: Callable | None = None):
        self.on_point = on_point

    def __call__(self, arrays) -> None:
        if self.on_point is not None:
            self.on_point(arrays)
        if arrays:
            mx.eval(*arrays)


ROW_BUDGET = Budget()


def forward_pacer(pacer, layout):
    """The pacer of one forward over ``layout``: a ``Budget`` becomes a
    ``RowBudget`` of the layout's rows, and any other pacer is kept."""
    if isinstance(pacer, Budget):
        return RowBudget(layout.rows, EVAL_ROWS, on_point=pacer.on_point)
    return pacer


def sample(model, video: mx.array, audio: mx.array, text: mx.array, layout,
           *, steps: int, shifts: dict, sigmas: dict | None = None,
           cond_video=(), cond_audio=(),
           on_step: Callable | None = None,
           pacer: Callable | Budget | None = ROW_BUDGET) -> tuple:
    """Video and audio latents after ``steps`` forwards from the noise
    ``video`` and ``audio``. ``sigmas`` {"video", "audio"}, float32 tables
    of ``steps`` + 1 entries each, replaces the schedule of ``shifts``.
    ``on_step(i, n)`` runs after each forward. ``pacer`` runs at each
    pause point inside a forward. A ``Budget``, the default, runs as a
    ``RowBudget`` of the layout's rows that starts again with each
    forward, and None runs each forward as one graph."""
    sig = (schedule.stream_sigmas(steps, shifts) if sigmas is None
           else sigmas)
    sv, sa = sig["video"], sig["audio"]
    if sigmas is not None and not len(sv) == len(sa) == steps + 1:
        raise ValueError(f"{steps} steps take sigma tables of {steps + 1} "
                         f"entries, not {len(sv)} video and {len(sa)} "
                         f"audio")
    xv, xa = video.astype(mx.float32), audio.astype(mx.float32)
    n = len(sv) - 1
    budget = None
    if isinstance(pacer, Budget):
        budget = pacer = forward_pacer(pacer, layout)
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
            if budget is not None:
                budget.reset()
            if on_step is not None:
                on_step(i, n)
    return xv, xa
