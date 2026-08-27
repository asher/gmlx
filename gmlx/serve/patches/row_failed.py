"""Client-contract wiring for permanently failed rows (U1a red exit).

The engine side (tick_guard, governor red) fails a row by removing it
and invoking the ``on_row_failed`` callbacks; without a bridge the
row's handler would wait out the token-queue timeout because nothing
ever puts a terminal item on its response queue. This module is the
bridge: a callback collects failures, and a ``ResponseGenerator._step``
wrapper delivers each one to its request queue on the engine thread
right after the tick that produced it, as a typed ``RowShedError``
followed by the close sentinel, and drops the row from the active map.

Wire shapes (the plan's client contract):
- before first byte / blocking: the error propagates out of the token
  iterator and the route's 500 carries str(error), which includes the
  delivered/prompt numbers and the triggering pressure reading, plus
  the retry-after-backoff hint.
- mid-stream: the SSE wrapper (request_flow._keepalive_sse) translates
  the error into a terminal SSE event with finish_reason "shed" and a
  clean close, distinguishable from both a normal stop and a transport
  fault.
"""

from __future__ import annotations

import logging
import threading

_log = logging.getLogger(__name__)

_BRIDGE_FLAG = "_kq_gguf_row_failed_bridge"

_pending: list = []
_pending_lock = threading.Lock()


# The self-owned signature the SSE layer keys the terminal-event
# upgrade on: upstream stream handlers stringify the exception into a
# plain {"error": str} event, and this phrase is how that event is
# recognized as ours (single source of truth; RowShedError builds its
# message from it).
SHED_MARKER = "shed under memory pressure"


class RowShedError(RuntimeError):
    """A request permanently failed by the memory governor or the tick
    guard's containment ladder. Retryable after backoff."""

    def __init__(self, uid, info: dict):
        self.uid = uid
        self.info = dict(info or {})
        delivered = self.info.get("delivered")
        prompt_len = self.info.get("prompt_len")
        why = self.info.get("error") or "memory pressure"
        super().__init__(
            f"request {SHED_MARKER} after "
            f"{delivered if delivered is not None else 0} tokens "
            f"(prompt {prompt_len if prompt_len is not None else '?'} "
            f"tokens): {why}. The request is retryable after backoff.")


def install_row_failed_bridge() -> None:
    """Register the failure collector and wrap ``_step`` to deliver.
    Idempotent."""
    from mlx_vlm.server import generation as gen_mod

    from ..tick_guard import on_row_failed

    if getattr(gen_mod.ResponseGenerator._step, _BRIDGE_FLAG, False):
        return

    def _collect(uid, info):
        with _pending_lock:
            _pending.append((uid, info))

    on_row_failed(_collect)
    _orig_step = gen_mod.ResponseGenerator._step

    def _bridged_step(self, batch_gen, active, gen_kwargs=None):
        try:
            return _orig_step(self, batch_gen, active, gen_kwargs)
        finally:
            with _pending_lock:
                failed = _pending[:]
                _pending.clear()
            for uid, info in failed:
                entry = active.pop(uid, None)
                if entry is None:
                    continue  # already cancelled or finished
                try:
                    entry["rqueue"].put(RowShedError(uid, info))
                    entry["rqueue"].put(None)
                except Exception:
                    _log.warning("[row-failed] queue delivery for uid=%s "
                                 "failed", uid, exc_info=True)

    setattr(_bridged_step, _BRIDGE_FLAG, True)
    gen_mod.ResponseGenerator._step = _bridged_step
    _log.info("row-failed client bridge installed")
