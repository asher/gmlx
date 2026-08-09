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
contemporaneous no-prefill batched rate. Steady-interleave algebra gives
retention r/(1+r) at ratio r independent of depth, so the floor fixes the
paced ratio at rho/(1-rho) and the chunk-cost threshold at (1-rho)/rho;
there is no useful middle ratio (a mid ratio pays for both a narrow
decode batch and a stretched prefill). Auto therefore only ever selects
the paced ratio or 0, per tick:

    r = rho/(1-rho)  if an incumbent row exists (admitted at least
                        grace_ms before the oldest waiter arrived)
                     and C > (1-rho)/rho (hysteresis + dwell)
                     and pending count <= queue_max (own hysteresis)
                     and the oldest waiter is inside deadline_s
        0            otherwise

Only the C conjunct enforces the floor; incumbency decides whether anyone
is owed it. The queue and deadline conjuncts knowingly abandon the floor
in favor of waiters, and every tick that does so logs the abandonment
with the incumbent's projected retention.

Signals are measured in the scheduler wrapper with no new sync: the
decode step cost s is an exponentially weighted mean per decode width
(width changes exactly at admission), the first decode tick after a chunk
is skipped (its bracket reads the chunk's sync, not a step), and C is
computed residue-corrected as max(0, last_chunk - s) / s because the
chunk bracket absorbs up to one in-flight decode step. Waiter arrival and
row admission stamps are kept per uid, bounded by one tick. Time a waiter
spends declined by the admission gate does not count against the deadline
(the gate, not pacing, is why it waits), read through getattr defaults so
either feature works without the other.

State lives on the generator under ``_kq_auto_`` attributes. The kill
switch GMLX_DECODE_PREFILL_AUTO=0 resolves auto to the static paced ratio
without a restart; any numeric ratio at any precedence layer bypasses
auto entirely.

Calibration knobs (documented here, not in the public config reference):

    GMLX_DECODE_PREFILL_FLOOR     retention floor rho (default 0.5)
    GMLX_DECODE_PREFILL_HYST      C hysteresis band (default 1.5)
    GMLX_DECODE_PREFILL_DWELL_MS  min time in a C state (default 1000)
    GMLX_DECODE_PREFILL_QUEUE_MAX pending above which pacing stands down
                                      (default 1, one-count hysteresis)
    GMLX_DECODE_PREFILL_DEADLINE_S oldest-waiter age forcing ratio 0
                                      (default 10)
    GMLX_DECODE_PREFILL_GRACE_MS  incumbency grace (default 500)
    GMLX_DECODE_PREFILL_ALPHA     step-cost smoothing (default 0.2)
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


def queue_max() -> int:
    return int(_envf("GMLX_DECODE_PREFILL_QUEUE_MAX", 1))


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
        self.c_on: bool | None = None
        self.c_since = 0.0
        self.queue_ok = True     # permissive init: first contact at
        # pending == queue_max is the hold case and must pace
        self.last_tick = 0.0
        self.last_log = 0.0
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


def _stamp(gen, st: _AutoState, now: float) -> None:
    pending_uids = [s[0] for s in gen._unprocessed_sequences]
    for uid in pending_uids:
        st.first_seen.setdefault(uid, now)
    for uid in list(st.first_seen):
        if uid not in set(pending_uids):
            del st.first_seen[uid]
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
    prev_tick, st.last_tick = st.last_tick, now
    _stamp(gen, st, now)
    pending = gen._unprocessed_sequences
    if not pending:
        return _resolved(gen, st, 0.0, now, "no waiters")

    # Admission-gate coupling: time declined by the gate is not pacing's
    # doing. Gate-declined waiters leave the pending count while the last
    # decline was the previous tick, and their declined seconds do not
    # age them toward the deadline.
    gate_deferred = getattr(gen, "_kq_admit_deferred_s", {}) or {}
    last_decline = getattr(gen, "_kq_admit_last_decline", 0.0)
    gate_active = bool(last_decline) and last_decline >= prev_tick > 0.0
    pending_count = 0 if gate_active else len(pending)

    oldest_uid, oldest_seen = None, None
    for s in pending:
        seen = st.first_seen.get(s[0], now)
        if oldest_seen is None or seen < oldest_seen:
            oldest_uid, oldest_seen = s[0], seen

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

    qmax = queue_max()
    if pending_count > qmax:
        st.queue_ok = False
    elif pending_count < qmax:
        st.queue_ok = True
    if not st.queue_ok:
        return _resolved(
            gen, st, 0.0, now,
            f"queue depth {pending_count} > {qmax} (floor abandoned: "
            f"incumbent uid={incumbent} projected retention "
            f"{1.0 / (1.0 + c):.2f})")

    deadline = deadline_s()
    age = now - (oldest_seen if oldest_seen is not None else now)
    age -= float(gate_deferred.get(oldest_uid, 0.0))
    if age > deadline:
        return _resolved(
            gen, st, 0.0, now,
            f"oldest waiter {age:.1f}s > deadline {deadline:.1f}s "
            f"(floor abandoned: incumbent uid={incumbent} projected "
            f"retention {1.0 / (1.0 + c):.2f})")

    return _resolved(
        gen, st, paced_ratio(), now,
        f"incumbent uid={incumbent} (admitted "
        f"{now - st.admitted_at[incumbent]:.1f}s ago), chunk {c:.1f} "
        f"steps, waiting={len(pending)}")


def _resolved(gen, st: _AutoState, ratio: float, now: float,
              reason: str) -> float:
    if ratio != st.last_resolved and now - st.last_log >= _LOG_EVERY_S:
        st.last_log = now
        _log.info("[sched] pacing %s: %s",
                  "on" if ratio > 0 else "off", reason)
    st.last_resolved = ratio
    return ratio
