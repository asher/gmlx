"""Process-wide record of the live expert-streaming installs.

The every-token weight pin and the decode arena are both mlocked, and
mlocked pages are invisible to memory pressure and to jetsam. A second
install therefore adds to the first instead of competing with it, and the
sum can leave a machine with nothing reclaimable: not an out-of-memory
condition the kernel can resolve, but a VM deadlock that ends in a
watchdog panic.

So a released model must give its wired bytes back before the next
install wires anything (``reclaim_dead``), and what survives that must be
charged against the new one (``live_wired_bytes``).
"""

from __future__ import annotations

import weakref

# The attributes the loader hangs a streaming install on. Each is closeable
# and each holds host resources (shard fds, staging pools, mlocked ranges)
# that must not outlive the model.
STREAM_ATTRS = (
    "_kq_prefetcher",
    "_kq_feeder",
    "_kq_decode_feeder",
    "_kq_weights_pin",
)

# (weakref to the owner module, committed wired bytes).
_LIVE: list[tuple[weakref.ref, int]] = []


def streaming_owner(model):
    """The module the loader hung the streaming helpers on. The served
    object is usually a wrapper (the text-only vlm adapter, whose
    ``language_model._model`` is the stock model) that forwards no
    attributes, so reading the helpers off the wrapper finds nothing and
    the feeders' worker threads outlive the model, pinning every expert
    weight through their frames. Descend the wrapper chain to the first
    module that carries a helper; the original object when none does."""
    seen = set()
    cur = model
    while cur is not None and id(cur) not in seen:
        if any(getattr(cur, a, None) is not None for a in STREAM_ATTRS):
            return cur
        seen.add(id(cur))
        nxt = getattr(cur, "language_model", None)
        if nxt is None:
            nxt = getattr(cur, "_model", None)
        cur = nxt
    return model


def record(model, wired_bytes: int) -> None:
    """Add ``wired_bytes`` to ``model``'s committed wired total. The arena
    counts from allocation even though it wires at the first decode: an
    install sized against that window loses the memory as soon as either
    model decodes."""
    if wired_bytes <= 0:
        return
    try:
        ref = weakref.ref(model)
    except TypeError:  # not weak-referenceable
        return
    for i, (r, n) in enumerate(_LIVE):
        if r() is model:
            _LIVE[i] = (r, n + int(wired_bytes))
            return
    _LIVE.append((ref, int(wired_bytes)))


def _sweep() -> list[tuple[weakref.ref, int]]:
    live = [(r, n) for r, n in _LIVE if r() is not None]
    _LIVE[:] = live
    return live


def reclaim_dead() -> int:
    """Unwire the installs whose model is gone. Returns the bytes freed.
    A weakref to a model still in the feeder/module cycle resolves, so the
    count only settles after a collection."""
    if not _LIVE:
        return 0
    import gc

    before = sum(n for _, n in _LIVE)
    gc.collect()
    return before - sum(n for _, n in _sweep())


def live_wired_bytes() -> int:
    """Wired bytes committed by streaming installs still held elsewhere."""
    return sum(n for _, n in _sweep())


def release(model) -> None:
    """Tear one model's streaming install down now: close the prefetcher
    and the feeders (shard fds, staging pools, the mlocked arena) and
    unlock the weight pin. Call it before loading a second model into one
    process. The MoE modules keep their own feeder reference, so the model
    is not usable after this. Best-effort per helper: one failing close
    must not leak the others."""
    import gc

    owner = streaming_owner(model)
    for attr in STREAM_ATTRS:
        helper = getattr(owner, attr, None)
        if helper is None:
            continue
        close = getattr(helper, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        try:
            object.__setattr__(owner, attr, None)
        except Exception:
            pass
    _LIVE[:] = [(r, n) for r, n in _LIVE if r() is not model]
    del owner, model
    gc.collect()
