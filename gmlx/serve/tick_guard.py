"""Engine-tick OOM containment for the batched serve loop.

A memory error inside ``BatchGenerator._next`` (an allocator refusal,
or a command-buffer execution failure resurfacing at a wait) aborts
the whole serve process today. The guard wraps the tick outermost.
On a memory error it drains the streams in the measured-safe order
(eval_guard), reclaims the buffer pool, and lets the loop retry; a
repeat failure removes the largest decoding row, rebuilds it from
committed state (prompt ids plus every token already delivered), and
requeues it at the head of the admission queue under the same uid, so
the request resumes as itself after a re-prefill (APC turns the
replay into a prefix hit). A row that fails again after its own
rebuild is failed permanently through the ``on_row_failed``
callbacks. Rows the throw did not land in keep their committed state
untouched; nothing is ever rebuilt from arrays that survived a
failed tape.

Bookkeeping: per live uid, the original prompt ids and every
committed token, harvested from the responses of successful ticks.
This is the per-row delivered-token ledger any retire-and-requeue
action needs, kept engine-side because response objects are consumed
by the HTTP layer and gone.

Only the memory-error class is contained; every other exception
propagates unchanged. GMLX_TICK_GUARD=0 disables.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_tick_guard"

_MEM_MARKS = ("metal::malloc", "Insufficient Memory",
              "Command buffer execution failed", "kIOGPUCommand",
              "Attempting to allocate")

_row_failed_callbacks: list = []


def is_memory_error(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and any(
        m in str(e) for m in _MEM_MARKS)


def on_row_failed(fn) -> None:
    """Register ``fn(uid, info: dict)`` for permanently failed rows.
    info carries prompt_len, delivered, and the triggering error text;
    the server layer maps it to the client-contract shape."""
    _row_failed_callbacks.append(fn)


class _Row:
    __slots__ = ("prompt_ids", "max_tokens", "kwargs", "lp", "tc",
                 "committed")

    def __init__(self, prompt_ids, max_tokens, kwargs, lp, tc):
        self.prompt_ids = list(prompt_ids)
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.lp = lp
        self.tc = tc
        self.committed: list = []


class _TickState:
    def __init__(self):
        self.ledger: dict = {}
        self.fail_streak = 0
        self.rebuilt: set = set()
        self.last_victim = None
        self.ticks = 0
        self.injected: set = set()


_INJECT_FIRED: set = set()


def _maybe_inject(st: _TickState) -> None:
    """U5 fault injection: GMLX_OOM_INJECT directives ``throw@N`` (a
    synthetic allocator error at tick N, exercising the containment
    ladder end to end) and ``kill@N`` (SIGKILL the worker, exercising
    the supervisor path). Band directives belong to the governor.
    Never set in production."""
    raw = os.environ.get("GMLX_OOM_INJECT", "")
    if not raw:
        return
    for part in raw.split(","):
        kind, _, t = part.partition("@")
        kind = kind.strip()
        if kind not in ("throw", "kill"):
            continue
        try:
            tick = int(t)
        except ValueError:
            continue
        # once per process: tick counters restart per batch cycle
        if st.ticks < tick or (kind, tick) in _INJECT_FIRED:
            continue
        _INJECT_FIRED.add((kind, tick))
        st.injected.add((kind, tick))
        if kind == "kill":
            _log.error("[tick-guard] INJECT: SIGKILL at tick %d", st.ticks)
            import signal

            os.kill(os.getpid(), signal.SIGKILL)
        _log.warning("[tick-guard] INJECT: synthetic allocator throw at "
                     "tick %d", st.ticks)
        raise RuntimeError(
            "[metal::malloc] Attempting to allocate 1 bytes "
            "(GMLX_OOM_INJECT synthetic)")


def _state(gen) -> _TickState:
    st = getattr(gen, "_kq_tick_guard", None)
    if st is None:
        st = gen._kq_tick_guard = _TickState()
    return st


def _harvest_pending(gen, st: _TickState) -> None:
    for uid, ids, m, kw, lp, tc in gen._unprocessed_sequences:
        if uid not in st.ledger:
            st.ledger[uid] = _Row(ids, m, kw, lp, tc)


def _live_decode_rows(gen) -> list:
    gb = gen._generation_batch
    uids = list(getattr(gb, "uids", ()))
    toks = getattr(gb, "_num_tokens", None)
    if toks is None:
        return [(u, 0) for u in uids]
    all_uids = getattr(gb, "_all_uids", uids)
    per = dict(zip(all_uids, toks))
    return [(u, per.get(u, 0)) for u in uids]


def _pick_victim(gen, st: _TickState):
    """Largest live decode row by resident tokens (prompt + emitted);
    deterministic, avoiding an immediate repeat victim on ties."""
    rows = []
    for uid, ntok in _live_decode_rows(gen):
        row = st.ledger.get(uid)
        plen = len(row.prompt_ids) if row else 0
        rows.append((plen + ntok, uid))
    if not rows:
        return None
    rows.sort(reverse=True)
    if len(rows) > 1 and rows[0][1] == st.last_victim \
            and rows[1][0] == rows[0][0]:
        return rows[1][1]
    return rows[0][1]


def _fail_row_permanently(gen, st: _TickState, uid, err: str) -> None:
    row = st.ledger.pop(uid, None)
    gen.remove(uid)
    info = {
        "prompt_len": len(row.prompt_ids) if row else None,
        "delivered": len(row.committed) if row else None,
        "error": err[:300],
    }
    _log.error("[tick-guard] row %s failed permanently after rebuild "
               "(%s delivered)", uid, info["delivered"])
    for fn in _row_failed_callbacks:
        try:
            fn(uid, info)
        except Exception:
            _log.warning("[tick-guard] row-failed callback error",
                         exc_info=True)


def _rebuild_row(gen, st: _TickState, uid) -> None:
    row = st.ledger.get(uid)
    gen.remove(uid)
    if row is None:
        _log.error("[tick-guard] no ledger entry for %s; removed without "
                   "requeue", uid)
        return
    remaining = None
    if isinstance(row.max_tokens, int):
        remaining = max(row.max_tokens - len(row.committed), 0)
        if remaining == 0:
            # the row had already delivered its budget; it is done
            st.ledger.pop(uid, None)
            return
    replay = row.prompt_ids + row.committed
    gen._unprocessed_sequences.insert(
        0, (uid, replay, remaining, row.kwargs, row.lp, row.tc))
    st.rebuilt.add(uid)
    _log.warning("[tick-guard] row %s retired and requeued: replaying "
                 "%d prompt + %d delivered tokens", uid,
                 len(row.prompt_ids), len(row.committed))


def _contain(gen, st: _TickState, e: RuntimeError) -> None:
    from gmlx.eval_guard import drain_for

    st.fail_streak += 1
    _log.warning("[tick-guard] engine tick memory error (streak %d): %s",
                 st.fail_streak, str(e).splitlines()[0][:200])
    drain_for("engine-tick")
    mx.clear_cache()
    if st.fail_streak < 2:
        return  # pool reclaimed; the serve loop retries the tick
    victim = _pick_victim(gen, st)
    if victim is None:
        _log.error("[tick-guard] repeat failure with no removable row")
        return
    st.last_victim = victim
    if victim in st.rebuilt:
        _fail_row_permanently(gen, st, victim, str(e))
    else:
        _rebuild_row(gen, st, victim)


def _harvest_responses(st: _TickState, generation_responses) -> None:
    for r in generation_responses:
        row = st.ledger.get(getattr(r, "uid", None))
        if row is None:
            continue
        tok = getattr(r, "token", None)
        if tok is not None:
            row.committed.append(tok)
        if getattr(r, "finish_reason", None):
            st.ledger.pop(r.uid, None)
            st.rebuilt.discard(r.uid)


def install_tick_guard() -> bool:
    """Wrap BatchGenerator._next outermost. Idempotent."""
    if os.environ.get("GMLX_TICK_GUARD", "1") == "0":
        return False
    from mlx_vlm.generate import ar as _ar

    if getattr(_ar.BatchGenerator._next, _INSTALLED_FLAG, False):
        return True
    _orig = _ar.BatchGenerator._next

    def _guarded_next(self, **kwargs):
        st = _state(self)
        st.ticks += 1
        _harvest_pending(self, st)
        try:
            _maybe_inject(st)
            out = _orig(self, **kwargs)
        except RuntimeError as e:
            if not is_memory_error(e):
                raise
            _contain(self, st, e)
            return [], []
        st.fail_streak = 0
        _harvest_responses(st, out[1])
        return out

    setattr(_guarded_next, _INSTALLED_FLAG, True)
    _ar.BatchGenerator._next = _guarded_next
    _log.info("engine tick guard installed")
    return True
