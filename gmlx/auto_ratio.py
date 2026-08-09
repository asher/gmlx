"""Dynamic decode-prefill pacing (``decode_prefill_ratio: auto``).

The static pacing ratio cannot be right at both ends of the depth range.
Pacing exists to keep a live decoding stream from being starved by a deep
prompt's prefill, and how much stock scheduling costs that stream is
decided by one measurable quantity: C, the wall cost of one prefill chunk
expressed in decode steps. At depth C is large and an unpaced prefill
puts the live stream near 1/(1+C) of its rate, so pacing rescues it. On
cheap chunks (shallow prompts, warm prefix hits, small models) stock
already leaves the stream above any reasonable floor and pacing only
delays admission, narrowing the decode batch that aggregate throughput
comes from.

Auto mode enforces one user-facing constraint, the retention floor rho
(default 0.5): a live decoding row is not pushed below rho of its
contemporaneous no-prefill batched rate. The floor is about sustained
starvation, so pacing shapes multi-chunk prefill trains; C is measured,
not predicted, so a single-chunk admission (a warm-prefix suffix, a
short prompt) completes before its cost is observable, and its worst
case for an incumbent is one chunk of stall, bounded by the prefill
step size. Steady-interleave algebra gives
retention r/(1+r) at ratio r independent of depth, so the floor fixes the
paced ratio at rho/(1-rho) and the chunk-cost threshold at (1-rho)/rho;
there is no useful middle ratio (a mid ratio pays for both a narrow
decode batch and a stretched prefill). Auto therefore only ever selects
the paced ratio or 0, per tick:

    r = rho/(1-rho)  if an incumbent row exists (admitted at least
                        grace_ms before the oldest waiter arrived)
                     and C > (1-rho)/rho (hysteresis + dwell)
                     and no queued waiter's paced wait exceeds deadline_s
        0            otherwise

A waiter is a queued sequence or a row of the active prompt batch: a
prompt being chunked is still competing prefill work, and dropping
pacing at promotion would unpace the whole chunk train one tick after
it starts. Only queued waiters age toward the deadline; a prefilling
waiter's TTFT is already bounded multiplicatively by the paced ratio
(its chunk train stretches by at most (1+r)x), while a queued waiter's
wait is what pacing cannot bound.

The deadline ages pacing-attributable seconds, not wall time since
arrival: a queued waiter accrues age only on ticks where a prompt
train is live and the previous tick resolved paced, bounded at 1 s per
tick. Time blocked by capacity is deliberately excluded. A waiter
behind a full decode batch or a memory-gate decline is not waiting on
pacing, and abandoning the floor for it cannot admit it any sooner:
ratio 0 creates neither batch slots nor headroom, so an arrival-age
deadline would starve every incumbent to buy nothing. The sustained
regime C run measured exactly this shape: four waiters capacity-held
~50 s at saturation whose final admission was correctly paced, while
waiters genuinely queued behind paced trains aged to the deadline and
were shed unpaced.

Only the C conjunct enforces the floor; incumbency decides whether anyone
is owed it. The deadline conjunct knowingly abandons the floor in favor
of waiters, and every tick that does so logs the abandonment with the
incumbent's projected retention.

The deadline is the only abandonment policy; there is deliberately no
queue-depth term. Waiters drain into the prompt batch within a tick or
two of arriving, so a count of queued sequences is a race against
promotion, not a measure of pressure (the same burst can read 4 or 0
depending on tick phase). Real pressure is waiters that cannot promote,
and those age in the queue until the deadline abandons the floor for
them. Standing down on count alone was measured to freeze the incumbent
to ~0.02 retention during a burst's batched prefill while also finishing
the incumbent later than pacing would have; the paced alternative holds
the floor and bounds every burst waiter's TTFT at (1+r)x unpaced.

Signals are measured in the scheduler wrapper with no new sync: the
decode step cost s is an exponentially weighted mean per decode width
(width changes exactly at admission), the first decode tick after a chunk
is skipped (its bracket reads the chunk's sync, not a step), and C is
computed residue-corrected as max(0, last_chunk - s) / s because the
chunk bracket absorbs up to one in-flight decode step. Waiter arrival and
row admission stamps are kept per uid, bounded by one tick.

Admission-gate coupling: on a declined tick the gate hides the pending
list from the stock body, so this resolver sees no queued waiters and
accrues nothing, which is the exclusion the design wants (the gate, not
pacing, is why they wait). The one thing the resolver must do is keep
stamps and accruals alive for hidden waiters instead of pruning them as
departed, so it unions the gate's deferred-uid dict
(``_kq_admit_deferred_s``, read through getattr defaults so either
feature works without the other) into the waiter set for retention.

State lives on the generator under ``_kq_auto_`` attributes. The kill
switch GMLX_DECODE_PREFILL_AUTO=0 resolves auto to the static paced ratio
without a restart; any numeric ratio at any precedence layer bypasses
auto entirely.

Calibration knobs (documented here, not in the public config reference):

    GMLX_DECODE_PREFILL_FLOOR     retention floor rho (default 0.5)
    GMLX_DECODE_PREFILL_HYST      C hysteresis band (default 1.5)
    GMLX_DECODE_PREFILL_DWELL_MS  min time in a C state (default 1000)
    GMLX_DECODE_PREFILL_DEADLINE_S queued paced-wait forcing ratio 0
                                      (default 10)
    GMLX_DECODE_PREFILL_GRACE_MS  incumbency grace (default 500)
    GMLX_DECODE_PREFILL_ALPHA     step-cost smoothing (default 0.2)
    GMLX_DECODE_PREFILL_LOG_S     transition log rate limit (default 1;
                                      0 logs every transition, for
                                      debugging sub-second episodes)
"""

from __future__ import annotations

import logging
import os
import time

_log = logging.getLogger(__name__)

_LOG_EVERY_S = 1.0


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def floor_rho() -> float:
    rho = _envf("GMLX_DECODE_PREFILL_FLOOR", 0.5)
    return min(max(rho, 0.05), 0.95)


def paced_ratio() -> float:
    rho = floor_rho()
    return rho / (1.0 - rho)


def c_threshold() -> float:
    rho = floor_rho()
    return (1.0 - rho) / rho


def deadline_s() -> float:
    return _envf("GMLX_DECODE_PREFILL_DEADLINE_S", 10.0)


class _AutoState:
    """Per-generator resolver state (held under gen._kq_auto)."""

    def __init__(self):
        self.s_by_width: dict[int, float] = {}
        self.s_samples: dict[int, int] = {}
        self.skip_next_step = False
        self.first_seen: dict = {}
        self.admitted_at: dict = {}
        self.paced_wait: dict = {}
        self.c_on: bool | None = None
        self.c_since = 0.0
        self.last_resolve_t: float | None = None
        self.last_log = 0.0
        self.last_logged: float | None = None
        self.last_resolved: float | None = None


def _state(gen) -> _AutoState:
    st = getattr(gen, "_kq_auto", None)
    if st is None:
        st = gen._kq_auto = _AutoState()
    return st


def observe(gen, dt: float, prompt_delta: float, rows: int) -> None:
    """Fold one tick's bracket into the step-cost estimate. A chunk tick
    poisons the next decode bracket (dispatched pre-chunk, completed in
    the chunk's sync), so it is skipped."""
    st = _state(gen)
    if prompt_delta > 0.0:
        st.skip_next_step = True
        return
    if rows <= 0 or dt <= 0.0:
        return
    if st.skip_next_step:
        st.skip_next_step = False
        return
    alpha = min(max(_envf("GMLX_DECODE_PREFILL_ALPHA", 0.2), 0.01), 1.0)
    prev = st.s_by_width.get(rows)
    real = st.s_samples.get(rows, 0)
    st.s_by_width[rows] = dt if (prev is None or real == 0) else (
        (1 - alpha) * prev + alpha * dt)
    st.s_samples[rows] = real + 1


def _step_cost(st: _AutoState, width: int) -> tuple[float | None, bool]:
    """(s for this width, frozen). An unfed bucket seeds from the nearest
    populated width; the seed is exactly the stale value at the moment of
    admission, so the C state holds (frozen) until real samples land."""
    if st.s_samples.get(width, 0) > 0:
        return st.s_by_width[width], False
    fed = [w for w, n in st.s_samples.items() if n > 0]
    if not fed:
        return None, True
    nearest = min(fed, key=lambda w: abs(w - width))
    st.s_by_width[width] = st.s_by_width[nearest]
    return st.s_by_width[nearest], True


def _pb_uids(gen) -> list:
    """Rows of the active prompt batch: promoted waiters mid-prefill."""
    pb = gen._prompt_batch
    return list(getattr(pb, "uids", ())) if pb is not None else []


def _stamp(gen, st: _AutoState, now: float) -> None:
    # A waiter keeps its arrival stamp across promotion into the prompt
    # batch; the stamp dies when its rows reach the decode batch (or the
    # request is gone). Dropping it at promotion would both end pacing a
    # tick after the chunk train starts and reset incumbency comparisons.
    # Gate-deferred waiters are hidden from the pending list on declined
    # ticks; their stamps and accruals are retained, not pruned.
    pending_uids = [s[0] for s in gen._unprocessed_sequences]
    gate_hidden = list(getattr(gen, "_kq_admit_deferred_s", None) or {})
    waiter_uids = pending_uids + _pb_uids(gen) + gate_hidden
    for uid in waiter_uids:
        st.first_seen.setdefault(uid, now)
    keep = set(waiter_uids)
    for uid in list(st.first_seen):
        if uid not in keep:
            del st.first_seen[uid]
    for uid in list(st.paced_wait):
        if uid not in keep:
            del st.paced_wait[uid]
    live = list(getattr(gen._generation_batch, "uids", ()))
    for uid in live:
        st.admitted_at.setdefault(uid, now)
    for uid in list(st.admitted_at):
        if uid not in set(live):
            del st.admitted_at[uid]


def _c_term(gen, st: _AutoState, now: float) -> tuple[bool, float]:
    """C-threshold conjunct with hysteresis, dwell, and the width-change
    freeze. Returns (pacing_wanted, C)."""
    width = len(gen._generation_batch)
    s, frozen = _step_cost(st, width)
    last_chunk = getattr(gen, "_kq_last_chunk_time", 0.0)
    if s is None or s <= 0.0 or last_chunk <= 0.0:
        return (bool(st.c_on), 0.0)
    c = max(0.0, last_chunk - s) / s
    if frozen and st.c_on is not None:
        return st.c_on, c
    c_on = c_threshold()
    hyst = max(_envf("GMLX_DECODE_PREFILL_HYST", 1.5), 1.0)
    dwell = _envf("GMLX_DECODE_PREFILL_DWELL_MS", 1000.0) / 1e3
    if st.c_on is None:
        st.c_on = c > c_on
        st.c_since = now
        return st.c_on, c
    dwelled = (now - st.c_since) >= dwell
    if not st.c_on and c > c_on and dwelled:
        st.c_on, st.c_since = True, now
    elif st.c_on and c < c_on / hyst and dwelled:
        st.c_on, st.c_since = False, now
    return st.c_on, c


def resolve(gen, now: float | None = None) -> float:
    """The effective ratio for this tick under auto mode."""
    if os.environ.get("GMLX_DECODE_PREFILL_AUTO", "1") == "0":
        return paced_ratio()
    st = _state(gen)
    if now is None:
        now = time.perf_counter()
    _stamp(gen, st, now)
    pending = gen._unprocessed_sequences
    prefilling = _pb_uids(gen)

    # Queued waiters age only while behind a live paced train; capacity
    # waits (no train, or gate-hidden pending) accrue nothing, since
    # ratio 0 cannot admit them sooner. The per-tick bound keeps a
    # stalled tick from charging its gap to pacing.
    dt = min(max(now - st.last_resolve_t, 0.0), 1.0) \
        if st.last_resolve_t is not None else 0.0
    st.last_resolve_t = now
    if dt > 0.0 and prefilling and (st.last_resolved or 0.0) > 0.0:
        for s in pending:
            st.paced_wait[s[0]] = st.paced_wait.get(s[0], 0.0) + dt

    if not pending and not prefilling:
        return _resolved(gen, st, 0.0, now, "no waiters")

    # Incumbency compares against the oldest waiter of either kind;
    # the deadline (below) ages only the queued ones.
    oldest_seen = None
    queued_uid, queued_wait = None, 0.0
    for uid in [s[0] for s in pending] + prefilling:
        seen = st.first_seen.get(uid, now)
        if oldest_seen is None or seen < oldest_seen:
            oldest_seen = seen
    for s in pending:
        wait = st.paced_wait.get(s[0], 0.0)
        if queued_uid is None or wait > queued_wait:
            queued_uid, queued_wait = s[0], wait

    grace = _envf("GMLX_DECODE_PREFILL_GRACE_MS", 500.0) / 1e3
    incumbent = None
    for uid, adm in st.admitted_at.items():
        if adm + grace < (oldest_seen if oldest_seen is not None else now):
            incumbent = uid
            break
    if incumbent is None:
        return _resolved(
            gen, st, 0.0, now,
            f"no incumbent (waiting={len(pending)}, "
            f"width={len(gen._generation_batch)})")

    c_wants, c = _c_term(gen, st, now)
    if not c_wants:
        return _resolved(gen, st, 0.0, now,
                         f"chunk {c:.1f} steps <= {c_threshold():.1f}")

    # Deadline: queued waiters only. A prefilling waiter's TTFT is
    # bounded by the paced ratio itself (chunk train stretches at most
    # (1+r)x); a queued waiter's paced wait has no such bound, so it is
    # the one the floor is abandoned for.
    deadline = deadline_s()
    if queued_uid is not None and queued_wait > deadline:
        return _resolved(
            gen, st, 0.0, now,
            f"queued waiter paced {queued_wait:.1f}s > deadline "
            f"{deadline:.1f}s (floor abandoned: incumbent uid={incumbent} "
            f"projected retention {1.0 / (1.0 + c):.2f})")

    return _resolved(
        gen, st, paced_ratio(), now,
        f"incumbent uid={incumbent} (admitted "
        f"{now - st.admitted_at[incumbent]:.1f}s ago), chunk {c:.1f} "
        f"steps, waiting={len(pending)}+{len(prefilling)}")


def _resolved(gen, st: _AutoState, ratio: float, now: float,
              reason: str) -> float:
    # Compare against last_logged, not last_resolved: a rate-limited
    # transition must still log once the window opens, or a persisted
    # state stays unlogged.
    every = _envf("GMLX_DECODE_PREFILL_LOG_S", _LOG_EVERY_S)
    if ratio != st.last_logged and now - st.last_log >= every:
        st.last_log = now
        st.last_logged = ratio
        _log.info("[sched] pacing %s: %s",
                  "on" if ratio > 0 else "off", reason)
    st.last_resolved = ratio
    return ratio
