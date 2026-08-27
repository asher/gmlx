"""Memory-headroom gate on serve admission (GMLX_ADMIT_HEADROOM).

No knob setting should be able to kill the server. Under decode-heavy
pacing a burst of concurrent requests on a near-RAM-size model can run
the box past the Metal working set mid-decode and the process dies with
an Insufficient Memory abort; the queued requests themselves hold almost
nothing, so the exposure is the moment a prompt batch is formed and its
KV plus prefill transient are committed on top of the live batch.

The gate sits exactly there: before the stock admission arm of
``BatchGenerator._next`` forms a prompt batch, it projects the bytes the
candidate rows would commit (server_memory.project_admission) against
the measured free headroom (prefill_decay.headroom_bytes). While the
projection does not fit, the pending list is hidden for that tick with
the same stash-and-restore the pacer uses, so the stock body runs decode
and never forms the batch. The request is never failed: it keeps its
queue position and is retried next tick; from the client it is a longer
time to first token, which the SSE keepalive already covers.

Two rules keep it from deadlocking. With no live decode rows and no
prompt batch in flight, admitting is the only way to make progress, so
the gate never declines an idle server. And a request deferred longer
than GMLX_ADMIT_DEFER_MAX_S seconds is admitted anyway, loudly: a gate
that silently holds a request forever is a worse failure than the one it
prevents. That blind admission is one row per tick, not a full prompt
batch: the ceiling tick trims the pending list to its head row, and the
rest re-project next tick, so past-ceiling admission stays serialized
for as long as the projection keeps failing.

The projection is conservative in one direction by design: a finished
batch that has not yet released its footprint makes measured headroom
look smaller than it will be, which biases toward deferring. That is the
safe side, not a bug.

State lives on the generator under ``_kq_admit_`` attributes (the
``_kq_`` convention), read by auto pacing through getattr defaults:
``_kq_admit_deferred_s`` maps uid to cumulative seconds declined by the
gate, and ``_kq_admit_last_decline`` stamps the most recent declined
tick.

Install after the pacer (install_decode_priority_sched) so this wrapper
runs outside it: on a declined tick the pacer sees no prefill work and
passes straight through, decode runs unpaced while admission waits. Both
wrappers merge-never-clobber the pending list on restore, so an insert
from a handler thread mid-call survives one stash nested in the other.

Knobs:
    GMLX_ADMIT_HEADROOM=0       kill switch, checked at install
    GMLX_ADMIT_RESERVE_GB       headroom held back beyond the projection
                                    (server_memory; default
                                    max(2, 0.05 x working set))
    GMLX_ADMIT_DEFER_MAX_S      defer ceiling, admit one row blind per
                                    tick past it (default 60)
"""

from __future__ import annotations

import logging
import os
import time

from .batch_rows import batch_rows

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_admit_headroom"
_LOG_EVERY_S = 5.0
_MAX_TICK_CREDIT_S = 1.0

# Server-wide gate counters for /v1/metrics. A deferral counts once per
# request group entering the deferred state, not per declined tick.
_DEFERRALS = 0
_LAST_DEFER = ""


def admit_stats() -> dict:
    return {"deferrals": _DEFERRALS,
            "last_defer_reason": _LAST_DEFER or None}


def _defer_max_s() -> float:
    try:
        return float(os.environ.get("GMLX_ADMIT_DEFER_MAX_S", "60"))
    except ValueError:
        return 60.0


def _candidate_uids(gen) -> list:
    n = min(gen.prefill_batch_size, len(gen._unprocessed_sequences))
    return [s[0] for s in gen._unprocessed_sequences[:n]]


def _prune_state(gen, pending_uids) -> None:
    deferred = getattr(gen, "_kq_admit_deferred_s", None)
    if deferred:
        for uid in [u for u in deferred if u not in pending_uids]:
            del deferred[uid]


def _note_decline(gen, uids, now: float) -> None:
    deferred = getattr(gen, "_kq_admit_deferred_s", None)
    if deferred is None:
        deferred = gen._kq_admit_deferred_s = {}
    last = getattr(gen, "_kq_admit_last_decline", 0.0)
    credit = min(max(now - last, 0.0), _MAX_TICK_CREDIT_S) if last else 0.0
    for uid in uids:
        deferred[uid] = deferred.get(uid, 0.0) + credit
    gen._kq_admit_last_decline = now


def _should_decline(gen) -> bool:
    """Decide this tick. Runs only when the stock body could actually form
    a prompt batch; otherwise the gate is not the reason anyone waits and
    it must not charge deferred time."""
    pending = gen._unprocessed_sequences
    if not pending or gen._prompt_batch is not None:
        return False
    num_to_add = gen.completion_batch_size - batch_rows(gen)
    if num_to_add < gen.prefill_batch_size:
        return False
    # Nothing to wait for: admitting is the only way to make progress.
    if batch_rows(gen) == 0:
        return False

    uids = _candidate_uids(gen)
    _prune_state(gen, {s[0] for s in pending})
    now = time.perf_counter()
    deferred = getattr(gen, "_kq_admit_deferred_s", {})

    from .governor import admission_hold_reason
    from .server_memory import project_admission

    hold = admission_hold_reason(gen)
    if hold:
        projected, headroom, parts = 0.0, 0.0, hold
    else:
        verdict = project_admission(gen, pending[: len(uids)])
        if verdict is None:
            # No basis to project (nothing measured yet): admit. The
            # first request on a freshly loaded model cannot be the one
            # that exhausts a box sized for the model.
            _log_admit(gen, uids, deferred)
            return False
        projected, headroom, parts = verdict
        if projected <= headroom:
            _log_admit(gen, uids, deferred)
            return False

    waited = max((deferred.get(u, 0.0) for u in uids), default=0.0)
    if waited > _defer_max_s():
        # Past the ceiling the gate no longer admits blind: the governor
        # sheds first (registered-cache evict + pool reclaim), then the
        # admit lands into recovered headroom. Admission is still one
        # row, not a full prompt batch: flag the tick so the wrapper
        # trims pending to the head row, and the rest re-project next
        # tick (admitting one by one only while the projection fails).
        from .governor import make_room_for_admission

        try:
            make_room_for_admission(gen)
        except Exception:
            _log.warning("[admit] ceiling handoff shed failed",
                         exc_info=True)
        gen._kq_admit_ceiling_tick = True
        _log.warning(
            "[admit] defer ceiling %.0fs hit: admitting one row past "
            "ceiling uid=%s (%s; projected %.1f GB > headroom %.1f GB; "
            "%d more candidates re-project next tick)",
            _defer_max_s(), uids[0] if uids else None, parts,
            projected / 1e9, headroom / 1e9, max(0, len(uids) - 1))
        return False

    first = any(u not in deferred for u in uids)
    _note_decline(gen, uids, now)
    global _DEFERRALS, _LAST_DEFER
    if hold:
        _LAST_DEFER = hold
    else:
        _LAST_DEFER = (f"projected {projected / 1e9:.1f} GB ({parts}) > "
                       f"headroom {headroom / 1e9:.1f} GB")
    if first:
        _DEFERRALS += 1
    last_log = getattr(gen, "_kq_admit_last_log", 0.0)
    if first or now - last_log > _LOG_EVERY_S:
        gen._kq_admit_last_log = now
        _log.info(
            "[admit] deferred uid=%s: %s; waiting=%d, decoding=%d",
            uids, _LAST_DEFER, len(pending), batch_rows(gen))
    return True


def _log_admit(gen, uids, deferred) -> None:
    waited = [deferred.get(u, 0.0) for u in uids if u in deferred]
    if waited:
        _log.info("[admit] admitting uid=%s after %.1fs deferred",
                  uids, max(waited))


def _one_row_next(gen, orig_next, kwargs):
    """Run one tick with pending trimmed to the head row (defer-ceiling
    blind admission). The stock body sees a one-row pending list, so the
    blind prompt batch is one row instead of up to prefill_batch_size
    rows; the trimmed rows keep their queue positions and re-project
    next tick. Same merge-never-clobber restore as the decline path:
    an insert() from a handler thread mid-call lands at the tail."""
    stash_pending = gen._unprocessed_sequences
    head = stash_pending[0]
    gen._unprocessed_sequences = [head]
    try:
        return orig_next(gen, **kwargs)
    finally:
        leftover = gen._unprocessed_sequences
        gen._unprocessed_sequences = stash_pending
        if not any(s is head for s in leftover):
            stash_pending.pop(0)  # the stock body consumed the head row
        arrived = [s for s in leftover if s is not head]
        if arrived:
            stash_pending.extend(arrived)


def install_admit_headroom_gate() -> None:
    """Gate prompt-batch formation on projected memory headroom.

    Late-bound monkeypatch on ``BatchGenerator._next``, same pattern as
    the apc_pooling gates: idempotent via a flag attribute, env kill
    switch checked at install, and the per-tick decision wrapped so a
    probe failure degrades to stock admission rather than a crash. Must
    install after install_decode_priority_sched (see module docstring).
    """
    from mlx_vlm.generate import ar as _ar

    if getattr(_ar.BatchGenerator._next, _INSTALLED_FLAG, False):
        return
    if os.environ.get("GMLX_ADMIT_HEADROOM", "1") == "0":
        return

    _orig_next = _ar.BatchGenerator._next

    def _gated_next(self, **kwargs):
        try:
            decline = _should_decline(self)
        except Exception:
            _log.warning("admit gate decision failed; admitting",
                         exc_info=True)
            decline = False
        ceiling = getattr(self, "_kq_admit_ceiling_tick", False)
        if ceiling:
            self._kq_admit_ceiling_tick = False
        if not decline:
            if ceiling and len(self._unprocessed_sequences) > 1:
                return _one_row_next(self, _orig_next, kwargs)
            return _orig_next(self, **kwargs)
        stash_pending = self._unprocessed_sequences
        self._unprocessed_sequences = []
        try:
            return _orig_next(self, **kwargs)
        finally:
            # insert() may have appended to (or rebound) the temp list
            # from a handler thread mid-call; merge, never clobber.
            arrived = self._unprocessed_sequences
            self._unprocessed_sequences = stash_pending
            if arrived:
                stash_pending.extend(arrived)

    setattr(_gated_next, _INSTALLED_FLAG, True)
    _ar.BatchGenerator._next = _gated_next
    _log.info("admission headroom gate installed")
