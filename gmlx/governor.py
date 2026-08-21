"""Runtime memory governor for the batched serve loop (GMLX_GOVERNOR).

Owns the decode tail between the admission gate (which prices joins)
and the tick guard (which contains the error that means everything
else failed). Per tick it computes ticks-to-collision from the one
shared accounting and walks a band ladder whose rungs each declare
the byte or peak class they move and are measured before/after:

    headroom(n) = WS x (1 - M) - tracked(n)     [tracked = active
                  minus zero-copy weights; prefill_decay.headroom
                  minus the margin M x WS]
    demand(n)   = declared next-tick peak: per-token KV rates x rows
                  x measured tokens/tick, plus the last admission
                  projection while a join is pending or its prompt
                  batch is in flight, plus the priced transient of a
                  scheduled shed
    ttc(n)      = headroom(n) / max(demand(n), 1)

Bands are rates, not levels: a deep batch at flat headroom is green;
a shallow one growing fast is not.

  green   ttc > K_y for d_y consecutive ticks (dwell inside the
          trigger, so entry and exit share hysteresis)
  yellow  ttc <= K_y: stop admitting; arm the throttle
          (mx.set_memory_limit on band entry only, restored on exit,
          never at boot: armed value and reclaim threshold are
          coupled) and shrink the MLX buffer cache; then demand
          rungs on measured miss: halve the prefill chunk, clamp
          speculative width. Demand rungs declare a peak class:
          success is a lower get_peak_memory delta across the
          following ticks (reset on band entry), never freed bytes.
  orange  headroom - k_o x demand < shed_transient(largest row)
          + reserve (the threshold includes orange's own cost so the
          shed can still afford itself): evict registered caches by
          fraction, then retire the largest row through the tick
          guard's ledger (remove + requeue same uid; APC turns the
          replay into a prefix hit), gated on the APC disk tier
          being armed; without it a retire is a cold re-prefill,
          worse for the victim than red's contained failure.
  red     orange measured-failed or next-tick headroom negative:
          contained-fail the largest removable row (U1a path,
          on_row_failed callbacks carry the client contract).
          Largest, deterministically; the tiebreak never victimizes
          the same uid twice in a row and victim repeats are counted
          in /v1/metrics so starvation is visible.

Anti-thrash: minimum dwell per band and a cap on shed actions per
minute, or orange fights a decode loop that immediately
re-allocates. Every shed reports measured recovery (before/after
active bytes).

Registered-cache protocol: anything holding resident GPU bytes
registers ``bytes()`` (backed by the _BaseCache.nbytes contract) and
``evict(fraction) -> freed bytes``. The governor trusts neither
beyond the counters. The serve APC manager self-registers at first
governed tick. Registered bytes are sampled on a slow cadence
(every 16 ticks and at band transitions); the per-tick trajectory
rides the free counters and the admission-path rate stash, never a
vars() walk.

Threading: everything here runs on the engine thread inside the
``_next`` wrapper; remove() is engine-stream-scoped by upstream and
the tick guard's ledger is harvested on the same thread.

Knobs (all calib values are U5's to tighten):
    GMLX_GOVERNOR=0          kill switch, checked at install
    GMLX_GOV_MARGIN          M, margin fraction of WS (default 0.05,
                             D-4's decision; mirrors 1 - wired frac)
    GMLX_GOV_KY              yellow entry ttc in ticks (default 16)
    GMLX_GOV_DY              green dwell ticks to de-escalate
                             (default 8)
    GMLX_GOV_KO              orange lookahead ticks (default 4)
    GMLX_GOV_RUNG_TICKS      measured-miss window per demand rung
                             (default 4)
    GMLX_GOV_SHEDS_PER_MIN   shed rate cap (default 4)
    GMLX_GOV_MIN_DWELL_S     min seconds in a band before
                             de-escalating (default 2.0)
"""

from __future__ import annotations

import logging
import os
import time
import weakref

import mlx.core as mx

from .batch_rows import batch_rows

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_governor"

GREEN, YELLOW, ORANGE, RED = 0, 1, 2, 3
_BAND_NAMES = ("green", "yellow", "orange", "red")

# Registered caches: name -> (bytes_fn, evict_fn). Values may be dead
# weakref-bound callables; a registrant that raises is dropped.
_REG: dict = {}

# Server-wide counters for /v1/metrics.
_STATS = {
    "band": "green",
    "yellow_entries": 0,
    "orange_evictions": 0,
    "orange_retires": 0,
    "red_failures": 0,
    "victim_repeats": 0,
    "sheds_suppressed": 0,
    "last_action": None,
    "last_recovered_bytes": 0,
}


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


_inject_memo: tuple[str, tuple] = ("", ())


def _inject_directives() -> tuple:
    """Parsed GMLX_OOM_INJECT directives (U5 fault injection): a
    comma-separated list of kind@tick, e.g. ``yellow@40,orange@120,
    red@200``. The governor forces the named band's condition at that
    tick, once; ``throw@N`` and ``kill@N`` belong to the tick guard.
    Never set in production."""
    global _inject_memo
    raw = os.environ.get("GMLX_OOM_INJECT", "")
    if raw == _inject_memo[0]:
        return _inject_memo[1]
    out = []
    for part in raw.split(","):
        if "@" not in part:
            continue
        kind, _, t = part.partition("@")
        try:
            out.append((kind.strip(), int(t)))
        except ValueError:
            _log.warning("GMLX_OOM_INJECT: unparsable directive %r", part)
    _inject_memo = (raw, tuple(out))
    return _inject_memo[1]


def _fire_injections(st: "_GovState") -> set:
    fired = set()
    for kind, tick in _inject_directives():
        key = (kind, tick)
        if st.tick_no >= tick and key not in st.injected:
            st.injected.add(key)
            if kind in ("yellow", "orange", "red"):
                fired.add(kind)
                _log.warning("[governor] INJECT: forcing %s at tick %d",
                             kind, st.tick_no)
    return fired


def governor_enabled() -> bool:
    return os.environ.get("GMLX_GOVERNOR", "1") != "0"


def register_cache(name: str, bytes_fn, evict_fn) -> None:
    """Register a resident-byte holder: bytes() and evict(fraction)."""
    _REG[name] = (bytes_fn, evict_fn)


def unregister_cache(name: str) -> None:
    _REG.pop(name, None)


def governor_stats() -> dict:
    return dict(_STATS)


class _GovState:
    def __init__(self):
        self.band = GREEN
        self.band_entered = time.perf_counter()
        self.tick_no = 0
        self.green_streak = 0
        self.tok_ema = 1.0
        self.rows_prev = 0
        # yellow arm/restore
        self.saved_mem_limit = None
        self.saved_cache_limit = None
        self.saved_prefill_step = None
        self.width_clamped = False
        # demand-rung measurement (peak rate per tick)
        self.rung = 0
        self.rung_tick = 0
        self.rung_peak = 0.0
        self.rung_rate_before = None
        # orange
        self.evict_fraction = 0.5
        self.orange_failed = False
        self.shed_times: list = []
        self.pending_shed_bytes = 0.0
        self.shed_measure_from = None
        # cached slow samples
        self.reg_bytes = 0.0
        self.reg_sampled_tick = -999
        self.last_victim_uid = None
        # U5 fault injection: (kind, tick) directives already fired
        self.injected: set = set()


def _state(gen) -> _GovState:
    st = getattr(gen, "_kq_governor", None)
    if st is None:
        st = gen._kq_governor = _GovState()
    return st


def admission_hold_reason(gen) -> str | None:
    """Non-green bands stop admitting; the admit gate consults this."""
    st = getattr(gen, "_kq_governor", None)
    if st is None or st.band == GREEN:
        return None
    return f"governor {_BAND_NAMES[st.band]}"


def make_room_for_admission(gen) -> None:
    """The 60 s admit-ceiling handoff: shed, then the gate admits.

    Supersedes blind admission: before the one-row past-ceiling admit,
    evict registered caches outright and reclaim the buffer pool so
    the admit lands into recovered headroom. No-op when disabled."""
    if not governor_enabled():
        return
    st = _state(gen)
    freed = _evict_registered(1.0)
    mx.clear_cache()
    _STATS["last_action"] = "ceiling-handoff evict"
    _log.warning("[governor] admit-ceiling handoff: evicted %.2f GB of "
                 "registered caches before past-ceiling admit",
                 freed / 1e9)
    st.pending_shed_bytes = 0.0


# ---------------------------------------------------------------- sampling

def _headroom_and_ws(margin: float):
    from .prefill_decay import headroom_bytes

    head = headroom_bytes()
    if head is None:
        return None, 0.0
    try:
        ws = float(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        return None, 0.0
    return head - margin * ws, ws


def _sample_registered(st: _GovState) -> float:
    if st.tick_no - st.reg_sampled_tick < 16:
        return st.reg_bytes
    st.reg_sampled_tick = st.tick_no
    total = 0.0
    for name in list(_REG):
        bytes_fn, _ = _REG[name]
        try:
            total += float(bytes_fn())
        except Exception:
            _log.warning("[governor] registered cache %r bytes() failed; "
                         "dropping registrant", name, exc_info=True)
            _REG.pop(name, None)
    st.reg_bytes = total
    return total


def _demand_bytes(gen, st: _GovState) -> float:
    rates = getattr(gen, "_kq_admit_kv_rates", None) or {}
    rate_sum = sum(k.get("rate", 0.0) for k in rates.values())
    rows = batch_rows(gen)
    demand = rate_sum * rows * max(st.tok_ema, 1.0)
    proj = getattr(gen, "_kq_admit_last_projection", None)
    if proj is not None:
        stamp, projected = proj
        join_relevant = (gen._unprocessed_sequences
                         or gen._prompt_batch is not None)
        if join_relevant and time.perf_counter() - stamp < 30.0:
            demand += projected
    demand += st.pending_shed_bytes
    return demand


def _shed_transient_bytes(gen, st: _GovState) -> float:
    """Priced cost of orange's own action: extract of the largest row
    plus the per-cache filter gather, summed (extract runs first)."""
    live = float(getattr(gen, "_kq_admit_live_bytes", 0.0))
    rows = max(1, batch_rows(gen))
    n_caches = max(1, int(getattr(gen, "_kq_admit_n_caches", 1)))
    row_bytes = 2.0 * live / rows  # largest row bounded by 2x the mean
    gather = live / (rows * n_caches)
    return row_bytes + gather


# ---------------------------------------------------------------- rungs

def _pause_admission(gen, st: _GovState) -> None:
    pass  # the hold is the band itself; admit_gate reads admission_hold_reason


def _arm_throttle(gen, st: _GovState, ws: float, margin: float) -> None:
    from .prefill_decay import untracked_weight_bytes

    if st.saved_mem_limit is None:
        limit = int(ws * (1.0 - margin) + untracked_weight_bytes())
        try:
            st.saved_mem_limit = mx.set_memory_limit(limit)
        except Exception:
            st.saved_mem_limit = None
            _log.warning("[governor] set_memory_limit failed", exc_info=True)
    if st.saved_cache_limit is None:
        try:
            st.saved_cache_limit = mx.set_cache_limit(0)
        except Exception:
            st.saved_cache_limit = None


def _disarm_throttle(gen, st: _GovState) -> None:
    if st.saved_mem_limit is not None:
        try:
            mx.set_memory_limit(st.saved_mem_limit)
        except Exception:
            pass
        st.saved_mem_limit = None
    if st.saved_cache_limit is not None:
        try:
            mx.set_cache_limit(st.saved_cache_limit)
        except Exception:
            pass
        st.saved_cache_limit = None


def _arm_demand_rung(gen, st: _GovState) -> None:
    """Advance the yellow demand ladder one rung. Rung 1 halves the
    prefill chunk (the serve peak is the last chunk's score transient,
    linear in step); rung 2 clamps speculative width (verify transient
    is width x block x vocab). Client cost is latency only."""
    if st.rung == 0:
        step = getattr(gen, "prefill_step_size", None)
        if isinstance(step, int) and step > 512:
            st.saved_prefill_step = step
            gen.prefill_step_size = max(512, step // 2)
            _STATS["last_action"] = f"prefill step {step}->{step // 2}"
            _log.warning("[governor] yellow rung: prefill step %d -> %d",
                         step, gen.prefill_step_size)
        st.rung = 1
        return
    if st.rung == 1:
        if not st.width_clamped:
            from .speculative import set_governor_width_clamp

            clamp = max(1, batch_rows(gen) // 2)
            set_governor_width_clamp(clamp)
            st.width_clamped = True
            _STATS["last_action"] = f"spec width clamp {clamp}"
            _log.warning("[governor] yellow rung: speculative width "
                         "clamped to %d", clamp)
        st.rung = 2


def _restore_demand_rungs(gen, st: _GovState) -> None:
    if st.saved_prefill_step is not None:
        gen.prefill_step_size = st.saved_prefill_step
        st.saved_prefill_step = None
    if st.width_clamped:
        try:
            from .speculative import set_governor_width_clamp

            set_governor_width_clamp(0)
        except Exception:
            pass
        st.width_clamped = False
    st.rung = 0
    st.rung_rate_before = None


def _shed_allowed(st: _GovState) -> bool:
    now = time.perf_counter()
    cap = int(_env_f("GMLX_GOV_SHEDS_PER_MIN", 4))
    st.shed_times = [t for t in st.shed_times if now - t < 60.0]
    if len(st.shed_times) >= cap:
        _STATS["sheds_suppressed"] += 1
        return False
    return True


def _evict_registered(fraction: float) -> float:
    freed = 0.0
    for name in list(_REG):
        _, evict_fn = _REG[name]
        try:
            freed += float(evict_fn(fraction) or 0)
        except Exception:
            _log.warning("[governor] registered cache %r evict() failed; "
                         "dropping registrant", name, exc_info=True)
            _REG.pop(name, None)
    return freed


def _apc_disk_armed(gen) -> bool:
    mgr = getattr(gen, "apc_manager", None)
    return mgr is not None and getattr(mgr, "disk", None) is not None


def _retire_largest(gen, st: _GovState) -> bool:
    """Orange's width lever: remove the largest decode row and requeue
    it as itself through the tick guard's ledger (replay = prompt +
    delivered; APC makes the re-prefill warm)."""
    from . import tick_guard as tg

    tg_st = tg._state(gen)
    victim = tg._pick_victim(gen, tg_st)
    if victim is None:
        return False
    if victim == st.last_victim_uid:
        _STATS["victim_repeats"] += 1
    st.last_victim_uid = victim
    tg_st.last_victim = victim
    before = mx.get_active_memory()
    st.pending_shed_bytes = _shed_transient_bytes(gen, st)
    tg._rebuild_row(gen, tg_st, victim)
    st.shed_times.append(time.perf_counter())
    st.shed_measure_from = float(before)
    _STATS["orange_retires"] += 1
    _STATS["last_action"] = f"retire uid={victim}"
    return True


def _fail_largest(gen, st: _GovState, why: str) -> bool:
    from . import tick_guard as tg

    tg_st = tg._state(gen)
    victim = tg._pick_victim(gen, tg_st)
    if victim is None:
        return False
    if victim == st.last_victim_uid:
        _STATS["victim_repeats"] += 1
    st.last_victim_uid = victim
    tg_st.last_victim = victim
    before = mx.get_active_memory()
    tg._fail_row_permanently(gen, tg_st, victim, why)
    st.shed_times.append(time.perf_counter())
    st.shed_measure_from = float(before)
    _STATS["red_failures"] += 1
    _STATS["last_action"] = f"red fail uid={victim}"
    return True


# ---------------------------------------------------------------- bands

def _enter(gen, st: _GovState, band: int, ws: float, margin: float) -> None:
    if band == st.band:
        return
    up = band > st.band
    _log.warning("[governor] band %s -> %s", _BAND_NAMES[st.band],
                 _BAND_NAMES[band])
    st.band = band
    st.band_entered = time.perf_counter()
    _STATS["band"] = _BAND_NAMES[band]
    if band >= YELLOW and up:
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
        st.rung_peak = 0.0
        st.rung_tick = st.tick_no
        st.rung_rate_before = None
    if band >= YELLOW:
        if st.saved_mem_limit is None and st.saved_cache_limit is None:
            _STATS["yellow_entries"] += 1
        _arm_throttle(gen, st, ws, margin)
    if band == GREEN:
        _disarm_throttle(gen, st)
        _restore_demand_rungs(gen, st)
        st.evict_fraction = 0.5
        st.orange_failed = False


def _measure_rung(gen, st: _GovState) -> None:
    """Escalate the yellow demand ladder only on a measured miss: the
    peak-growth rate over the rung window did not fall."""
    window = max(2, int(_env_f("GMLX_GOV_RUNG_TICKS", 4)))
    if st.tick_no - st.rung_tick < window:
        return
    try:
        peak = float(mx.get_peak_memory())
    except Exception:
        return
    rate = (peak - st.rung_peak) / max(1, st.tick_no - st.rung_tick)
    st.rung_peak = peak
    st.rung_tick = st.tick_no
    if st.rung_rate_before is None:
        st.rung_rate_before = rate
        return
    if rate >= st.rung_rate_before and st.rung < 2:
        _arm_demand_rung(gen, st)
    st.rung_rate_before = rate


def _governor_tick(gen) -> None:
    st = _state(gen)
    st.tick_no += 1
    margin = _env_f("GMLX_GOV_MARGIN", 0.05)
    ky = _env_f("GMLX_GOV_KY", 16.0)
    dy = int(_env_f("GMLX_GOV_DY", 8))
    ko = _env_f("GMLX_GOV_KO", 4.0)
    min_dwell = _env_f("GMLX_GOV_MIN_DWELL_S", 2.0)

    _maybe_register_apc(gen)
    head, ws = _headroom_and_ws(margin)
    if head is None:
        return
    _sample_registered(st)
    demand = _demand_bytes(gen, st)
    ttc = head / max(demand, 1.0)

    # measured recovery report for the last shed
    if st.shed_measure_from is not None:
        recovered = st.shed_measure_from - float(mx.get_active_memory())
        _STATS["last_recovered_bytes"] = int(recovered)
        if recovered <= 0 and st.band >= ORANGE:
            st.orange_failed = True
        st.shed_measure_from = None
        st.pending_shed_bytes = 0.0

    from .server_memory import admit_reserve_bytes

    reserve = admit_reserve_bytes(ws, gen)
    orange_now = (head - ko * demand
                  < _shed_transient_bytes(gen, st) + reserve)
    red_now = (head - demand < 0.0) or st.orange_failed

    forced = _fire_injections(st)
    if "yellow" in forced:
        ttc = 0.0
        st.green_streak = 0
    if "orange" in forced:
        orange_now = True
    if "red" in forced:
        red_now = True

    dwelled = time.perf_counter() - st.band_entered >= min_dwell
    if ttc > ky:
        st.green_streak += 1
    else:
        st.green_streak = 0

    if red_now and batch_rows(gen) > 0:
        _enter(gen, st, RED, ws, margin)
        if _shed_allowed(st):
            _fail_largest(
                gen, st,
                f"governor red: headroom {head / 1e9:.2f} GB below "
                f"next-tick demand {demand / 1e9:.2f} GB")
        st.orange_failed = False
        return
    if orange_now and batch_rows(gen) > 0:
        _enter(gen, st, ORANGE, ws, margin)
        if _shed_allowed(st):
            before = mx.get_active_memory()
            freed = _evict_registered(st.evict_fraction)
            mx.clear_cache()
            _STATS["orange_evictions"] += 1
            _STATS["last_action"] = (
                f"evict {st.evict_fraction:.1f} freed {freed / 1e9:.2f} GB")
            st.shed_times.append(time.perf_counter())
            st.shed_measure_from = float(before)
            if st.evict_fraction >= 1.0 and freed < demand:
                # registered caches are dry; the width lever is next
                if _apc_disk_armed(gen):
                    _retire_largest(gen, st)
                else:
                    st.orange_failed = True  # fall through to red next tick
            st.evict_fraction = min(1.0, st.evict_fraction + 0.5)
        return
    if ttc <= ky:
        _enter(gen, st, YELLOW, ws, margin)
        if st.rung_rate_before is None and st.tick_no - st.rung_tick >= 1:
            try:
                st.rung_peak = float(mx.get_peak_memory())
            except Exception:
                pass
        _measure_rung(gen, st)
        return
    if st.band != GREEN and st.green_streak >= dy and dwelled:
        _enter(gen, st, GREEN, ws, margin)


def _harvest_tick(gen, st: _GovState, out) -> None:
    """Post-tick: fold emitted tokens per row into the tokens/tick EMA
    the demand model uses."""
    try:
        responses = out[1]
    except Exception:
        return
    rows = max(1, batch_rows(gen))
    per_row = len(responses) / rows if responses else 0.0
    st.tok_ema = 0.8 * st.tok_ema + 0.2 * per_row if per_row else st.tok_ema


_APC_REGISTERED = False


def _maybe_register_apc(gen) -> None:
    """Self-register the serve APC manager the first time a governed
    tick sees one."""
    global _APC_REGISTERED
    if _APC_REGISTERED:
        return
    mgr = getattr(gen, "apc_manager", None)
    if mgr is None or not hasattr(mgr, "governor_bytes"):
        return
    ref = weakref.ref(mgr)

    def _bytes():
        m = ref()
        return m.governor_bytes() if m is not None else 0

    def _evict(fraction):
        m = ref()
        return m.governor_evict(fraction) if m is not None else 0

    register_cache("apc", _bytes, _evict)
    _APC_REGISTERED = True


def install_governor() -> bool:
    """Wrap BatchGenerator._next with the band ladder. Install after
    the trace wrapper and before the tick guard, so the guard contains
    any memory error a governor action itself trips. Idempotent."""
    if not governor_enabled():
        return False
    from mlx_vlm.generate import ar as _ar

    if getattr(_ar.BatchGenerator._next, _INSTALLED_FLAG, False):
        return True
    _orig = _ar.BatchGenerator._next

    def _governed_next(self, **kwargs):
        try:
            _governor_tick(self)
        except Exception:
            _log.warning("[governor] tick decision failed; running "
                         "ungoverned", exc_info=True)
        out = _orig(self, **kwargs)
        try:
            _harvest_tick(self, _state(self), out)
        except Exception:
            pass
        return out

    setattr(_governed_next, _INSTALLED_FLAG, True)
    _ar.BatchGenerator._next = _governed_next
    _log.info("memory governor installed")
    return True
