"""Chat decode beside an image job.

The chat engines and the image worker run on threads of their own, each
with its own MLX stream, and nothing orders their GPU work. The GPU serves
the evaluation in flight before a chat token that comes after it, and an
image step is one evaluation that holds the GPU for a second or more at
1024x1024, so a chat decode beside an image gets a token or two per step.

The chat engines mark each prefill and decode step with ``CHAT.begin``
and ``CHAT.end``. While chat is active, the image job's pacer evaluates
the forward at each point between DiT blocks, so the GPU holds at most
one block of image work, and then waits while chat steps run, for a time
in proportion to the image work since the last pause: a chat share of
0.5 gives chat as much GPU time as the image. The pacer always waits for
at least one chat step. With no chat active, it does nothing.

GMLX_IMAGE_CHAT_SHARE sets the share, from 0 to 0.9, and ``off`` turns
the pacer off.
"""

from __future__ import annotations

import functools
import os
import threading
import time

# Chat counts as active for this long after its last step, so the gaps
# between two steps of one decode do not end it.
_IDLE_S = 0.25
# The longest the pacer waits for one chat step before it goes on.
_WAIT_S = 0.1
# The most image time one pause pays for, so the first pause after chat
# starts does not wait for all the image work before it.
_MAX_PIECE_S = 0.25
DEFAULT_SHARE = 0.5
MAX_SHARE = 0.9


class ChatActivity:
    """Steps of the chat engines: how many run now, how many finished and
    when the last one did."""

    def __init__(self, idle_s: float = _IDLE_S, clock=time.monotonic):
        self._cond = threading.Condition()
        self._clock = clock
        self.idle_s = idle_s
        self.busy = 0
        self.ticks = 0
        self.last_end = float("-inf")

    def begin(self) -> None:
        with self._cond:
            self.busy += 1

    def end(self) -> None:
        with self._cond:
            self.busy = max(0, self.busy - 1)
            self.ticks += 1
            self.last_end = self._clock()
            self._cond.notify_all()

    def active(self) -> bool:
        """A chat step runs now or finished within ``idle_s``."""
        return self.busy > 0 or self._clock() - self.last_end < self.idle_s

    def wait_tick(self, timeout: float) -> bool:
        """Wait until a chat step finishes, or ``timeout`` seconds. True
        when one finished."""
        with self._cond:
            seen = self.ticks
            return self._cond.wait_for(lambda: self.ticks != seen, timeout)


CHAT = ChatActivity()


def chat_share(raw: str | None = None) -> float | None:
    """The share of GMLX_IMAGE_CHAT_SHARE, or None for ``off``. A value
    that is no number from 0 to 0.9 warns and gives the default."""
    raw = os.environ.get("GMLX_IMAGE_CHAT_SHARE") if raw is None else raw
    if raw is None or not raw.strip():
        return DEFAULT_SHARE
    val = raw.strip().lower()
    if val in ("off", "none", "false", "no"):
        return None
    try:
        share = float(val)
        if not 0.0 <= share <= MAX_SHARE:
            raise ValueError
        return share
    except ValueError:
        import warnings

        warnings.warn(f"GMLX_IMAGE_CHAT_SHARE={raw!r} is not a number from "
                      f"0 to {MAX_SHARE} or off; using {DEFAULT_SHARE}",
                      stacklevel=2)
        return DEFAULT_SHARE


class ImagePacer:
    """The pacer of one image job (``gmlx.image.pacing``): while chat is
    active, evaluate at each point, then wait while chat steps run for
    ``share / (1 - share)`` times the image time since the last pause, and
    for one chat step at least."""

    def __init__(self, chat: ChatActivity = CHAT, *,
                 share: float = DEFAULT_SHARE, wait_s: float = _WAIT_S,
                 clock=time.monotonic, evaluate=None):
        self.chat = chat
        self.ratio = share / (1.0 - share)
        self.wait_s = wait_s
        self._clock = clock
        self._eval = evaluate
        self.mark: float | None = None
        self.pauses = 0
        self.waited_s = 0.0

    def __call__(self, arrays: tuple) -> None:
        if not self.chat.active():
            self.mark = None
            return
        if arrays:
            if self._eval is None:
                import mlx.core as mx

                self._eval = mx.eval
            self._eval(*arrays)
        now = self._clock()
        piece = 0.0 if self.mark is None else min(now - self.mark,
                                                  _MAX_PIECE_S)
        budget = self.ratio * piece
        while self.chat.wait_tick(self.wait_s):
            if self._clock() - now >= budget:
                break
        self.mark = self._clock()
        self.pauses += 1
        self.waited_s += self.mark - now


def pacer() -> ImagePacer | None:
    """The pacer of a new image job, or None when GMLX_IMAGE_CHAT_SHARE is
    off."""
    share = chat_share()
    return None if share is None else ImagePacer(share=share)


_installed = {"done": False}


def install_chat_ticks() -> bool:
    """Mark each prefill and decode step of the mlx-vlm engine on
    ``CHAT``. Idempotent."""
    if _installed["done"]:
        return True
    from gmlx._vendor.mlx_vlm.generate import ar

    def _wrap(cls, name):
        orig = getattr(cls, name)
        if getattr(orig, "_gmlx_chat_tick", False):
            return

        @functools.wraps(orig)
        def wrapped(self, *args, **kwargs):
            CHAT.begin()
            try:
                return orig(self, *args, **kwargs)
            finally:
                CHAT.end()

        wrapped._gmlx_chat_tick = True  # type: ignore[attr-defined]
        setattr(cls, name, wrapped)

    _wrap(ar.GenerationBatch, "_step")
    _wrap(ar.SpeculativeGenerationBatch, "next")
    _wrap(ar.PromptProcessingBatch, "prompt_step")
    _wrap(ar.PromptProcessingBatch, "generate")
    _installed["done"] = True
    return True
