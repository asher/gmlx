"""One submission for the verify of a speculative round.

``GMLX_SPEC_VERIFY_ONE_SUBMIT`` (on by default, 0 turns it off) shapes the command
buffers of a B=1 verify by hand. The stock caps end a buffer every 50 ops,
about one per layer, and mlx holds the host once ten are in flight: the
verify's submit then returns only when the GPU is most of the way through
it. Under the lever the verify graph carries its own boundaries (kq
``cb_commit_mark`` on the stream ahead of the layers listed in
``GMLX_SPEC_VERIFY_MARKS``), the caps are lifted for the one async_eval
that submits it, and the caps are put back. The first buffers are short,
so the GPU starts while the host encodes; the rest of the verify is one
buffer, so the submit returns as soon as it is encoded.

Only the boundaries move: the primitives, their order and their bits are
those of the stock caps. Needs a kq build with ``cb_commit_mark`` and a
model module with a ``_VERIFY_LAYER_HOOK`` slot (qwen4_exp).
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading

import mlx.core as mx

from gmlx.envflags import env_bool

_ON = env_bool("GMLX_SPEC_VERIFY_ONE_SUBMIT", True)
# Caps no verify reaches: the marks are then the only boundaries.
_HIGH = (1 << 20, 1 << 20)


def _parse_marks(text: str) -> frozenset[int]:
    out = set()
    for part in text.split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return frozenset(out)


# A layer runs about 15 times longer on the GPU than its encode takes on
# the host, so one layer in the first buffer and eleven in the second keep
# committed work ahead of the GPU while the last buffer is encoded.
_MARKS = _parse_marks(os.environ.get("GMLX_SPEC_VERIFY_MARKS", "1,12"))

_state = {"kq": None, "made": 0, "submits": 0, "plain": 0}


def _kq():
    if _state["kq"] is None:
        try:
            import mlx_kquant as kq
            ok = hasattr(kq, "cb_commit_mark") and hasattr(kq, "set_cb_caps")
            _state["kq"] = kq if ok else False
        except ImportError:
            _state["kq"] = False
    return _state["kq"]


def _module(lm):
    """The model module of ``lm`` that has the layer hook slot, or None."""
    seen = lm
    for _ in range(3):
        mod = sys.modules.get(type(seen).__module__)
        if mod is not None and hasattr(mod, "_VERIFY_LAYER_HOOK"):
            return mod
        nxt = getattr(seen, "model", None)
        if nxt is None or nxt is seen:
            return None
        seen = nxt
    return None


def enabled(lm) -> bool:
    """The lever is on and this build and model can take it."""
    return bool(_ON and _MARKS and _kq() and _module(lm) is not None
                and mx.default_device() == mx.gpu)


def _hook(li: int, h: mx.array) -> mx.array:
    if li in _MARKS:
        _state["made"] += 1
        return _state["kq"].cb_commit_mark(h)
    return h


# Per model module: the verifies building inside ``marks`` and the hook
# found before the first. Engines of one module on two threads overlap:
# the last one out puts the hook back.
_marks_lock = threading.Lock()
_marks_users: dict = {}


@contextlib.contextmanager
def marks(lm):
    """Verify graphs built inside carry the boundaries."""
    mod = _module(lm)
    with _marks_lock:
        n, prev = _marks_users.get(mod, (0, None))
        if n == 0:
            prev = mod._VERIFY_LAYER_HOOK
            mod._VERIFY_LAYER_HOOK = _hook
        _marks_users[mod] = (n + 1, prev)
    try:
        yield
    finally:
        with _marks_lock:
            n, prev = _marks_users.pop(mod)
            if n > 1:
                _marks_users[mod] = (n - 1, prev)
            else:
                mod._VERIFY_LAYER_HOOK = prev


def submit(*arrays) -> None:
    """async_eval ``arrays`` with the marks as the only boundaries. While an
    image or video job holds the device on fine caps, or the phase caps are
    off, the stock caps stay and the marks only add boundaries."""
    from gmlx.serve import cb_phase
    kq = _state["kq"]
    with cb_phase._LOCK:
        lift = not cb_phase._state["holds"] and cb_phase._live()
        if not lift:
            _state["plain"] += 1
            mx.async_eval(*arrays)
            return
        prev = kq.set_cb_caps(*_HIGH)
        try:
            mx.async_eval(*arrays)
        finally:
            kq.set_cb_caps(int(prev[0]), int(prev[1]))
        _state["submits"] += 1


def stats() -> dict:
    return {k: v for k, v in _state.items() if k != "kq"}
