"""Process-wide record of the live expert-streaming installs.

An install wires memory that the usual accounting cannot undo. The
every-token weight pin (``pin_weights``) and the decode arena
(``decode_feeder``) are both mlocked, and mlocked pages are invisible to
memory pressure and to jetsam: they are never compressed, swapped, or
evicted, and no process is ever selected for kill on their account. So a
second install does not compete with the first for that memory. It adds to
it, and the sum can take a machine past the point where any page is
reclaimable. The result is not an out-of-memory condition the kernel can
resolve; it is a VM deadlock that ends in a watchdog panic.

Two rules follow, and this module supplies both.

A released model must give its wired bytes back before the next install
wires anything. It does not do so on its own: the decode feeder holds
every MoE module and every MoE module holds the feeder, so a dropped model
sits in a reference cycle that only a generational collection breaks, and
until then its arena stays wired. ``reclaim_dead`` runs that collection.
The server does the same at eviction (``gmlx.serve.residency._teardown``);
nothing did on the CLI path, where one process can load two models.

What survives that collection is genuinely held, and must be charged
against the new install rather than ignored. ``live_wired_bytes`` is that
charge. The weight pin and the arena sizer both take it as a reservation,
so a second streaming model in a process sizes itself against what is
left, and degrades to the page-cache path instead of wiring a machine
solid.
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
    """The module the loader hung the streaming helpers on.

    The served object is usually a wrapper (the text-only vlm adapter,
    whose ``language_model._model`` is the stock model) that forwards no
    attributes, so reading the helpers off the wrapper finds nothing and
    the feeders' worker threads outlive the model, pinning every expert
    weight through their frames. Descend the wrapper chain to the first
    module that carries a helper; the original object when none does.
    """
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
    """Add ``wired_bytes`` to ``model``'s committed wired total.

    Called once for the weight pin and once for the decode arena, which
    the loader sizes at different points. The arena counts from the moment
    it is allocated even though it wires at the first decode: a second
    install that sized itself against the unwired window would find the
    memory gone the moment either model decoded.
    """
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

    A weakref cannot answer this on its own: while the model sits in the
    feeder/module cycle it is unreachable but not yet finalized, so its
    weakref still resolves and its arena is still wired. Only the
    collection settles it, which is why this collects first and counts
    after.
    """
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
    """Tear one model's streaming install down now.

    The deterministic form of what dropping the model plus a collection
    eventually does: close the prefetcher and the feeders (shard fds,
    staging pools, the mlocked arena) and unlock the weight pin. Call it
    before loading a second model into one process - two wired installs is
    how a machine earns a watchdog panic.

    The closes are what free the wired bytes; the collection that follows
    only reclaims the cycles a caller who has already dropped the model
    left behind. The MoE modules keep their own feeder reference, so the
    model is not usable after this - release it and load again.

    Best-effort per helper: one failing close must not leak the others.
    """
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
