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

A command buffer that failed for a reason other than memory is a
different animal: a timeout or a page fault says the submitted work
was wrong, not that the box is full. Those get one contained retry
so a driver hiccup cannot take the server down, and then propagate
with the Metal error in the log, because no amount of reclaim fixes
them and shedding a row would file a kernel bug under capacity.
A command buffer failure of either kind inside a row's removal (the
guard's victim, the governor's retire, a cancel) drops the engine and
requeues every row, and a fault there counts toward the same one retry.
Every other exception propagates unchanged. GMLX_TICK_GUARD=0
disables.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

from gmlx.upstream.patching import link, wrapped_with

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_tick_guard"

# Running out of memory is what the ladder below can actually fix, so the
# memory class holds only the allocator's own words. Every other command
# buffer failure is a fault in the work that was submitted: a timeout means
# a kernel hung or the launch was oversized, a page fault means it read out
# of bounds. Reclaiming buffers cannot fix either one, and answering them
# with a re-prefill throws away a request and files the bug under capacity.
_MEM_MARKS = ("metal::malloc", "Insufficient Memory",
              "Attempting to allocate",
              "kIOGPUCommandBufferCallbackErrorOutOfMemory")

_GPU_MARKS = ("Command buffer execution failed", "kIOGPUCommand")

_row_failed_callbacks: list = []


def is_memory_error(e: BaseException) -> bool:
    """The allocator refused, or Metal reported it ran out of memory."""
    return isinstance(e, RuntimeError) and any(
        m in str(e) for m in _MEM_MARKS)


def is_gpu_fault(e: BaseException) -> bool:
    """A command buffer failed for a reason other than memory."""
    return (isinstance(e, RuntimeError)
            and not is_memory_error(e)
            and any(m in str(e) for m in _GPU_MARKS))


def is_command_buffer_failure(e: BaseException) -> bool:
    """A command buffer failed, for memory or any other reason. Its
    outputs are not valid, and the wait that reports it can come a step
    after the failed work."""
    return isinstance(e, RuntimeError) and any(m in str(e) for m in _GPU_MARKS)


def is_allocator_refusal(e: BaseException) -> bool:
    """The allocator refused before any work ran. Only then are the
    arrays the tick computed valid. A command buffer that ran out of
    memory reports it at a wait, after its outputs were spoiled."""
    return is_memory_error(e) and not is_command_buffer_failure(e)


def salvage(exc: BaseException, responses, *, ahead: bool = False) -> None:
    """Responses a failed tick computed. They ride the exception, so the
    guard can deliver them and nothing stale outlives the tick. ``ahead``
    puts them before what an inner frame salvaged."""
    if not responses:
        return
    got = getattr(exc, "_kq_salvage", None)
    if got is None:
        got = []
        exc._kq_salvage = got
    if ahead:
        got[:0] = list(responses)
    else:
        got.extend(responses)


def mark_broken(exc: BaseException, uids) -> None:
    """Decode rows whose engine state a retry cannot reuse. The guard
    removes them and rebuilds them as a group."""
    got = getattr(exc, "_kq_broken", None)
    if got is None:
        got = []
        exc._kq_broken = got
    got.extend(u for u in uids if u not in got)


def mark_prefill_broken(exc: BaseException, uids, apc_meta=None) -> None:
    """Rows of a prompt batch, or rows taken from the queue for one, that a
    retry cannot reuse. ``apc_meta`` is the prefix-block metadata of rows
    whose prompt batch was never built. The guard drops the prompt batch
    and rebuilds the rows as a group."""
    got = getattr(exc, "_kq_prefill_broken", None)
    if got is None:
        got = []
        exc._kq_prefill_broken = got
        exc._kq_prefill_meta = []
    got.extend(u for u in uids if u not in got)
    if apc_meta:
        exc._kq_prefill_meta.extend(apc_meta)


def on_row_failed(fn) -> None:
    """Register ``fn(uid, info: dict)`` for permanently failed rows.
    info carries prompt_len, delivered, and the triggering error text;
    the server layer maps it to the client-contract shape."""
    _row_failed_callbacks.append(fn)


def forget(gen, uid) -> None:
    """Drop ``uid``'s ledger entry: the request ended without a finish
    response, so no requeue may bring it back."""
    st = getattr(gen, "_kq_tick_guard", None)
    if st is None:
        return
    st.ledger.pop(uid, None)
    st.rebuilt.discard(uid)


def drop_prompt_batch(gen) -> None:
    """Take the prompt batch out of the engine and give back the prefix
    blocks it holds."""
    _drop_prompt_batch(gen, None)


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
        self.fault_streak = 0
        self.rebuilt: set = set()
        self.last_victim = None
        self.ticks = 0
        self.injected: set = set()


_INJECT_FIRED: set = set()
_INJECT_TICK = [0]
_REFUSAL = ("[metal::malloc] Attempting to allocate 1 bytes "
            "(GMLX_OOM_INJECT synthetic)")
_CB_OOM = ("[METAL] Command buffer execution failed: Insufficient Memory "
           "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory) "
           "(GMLX_OOM_INJECT synthetic)")
_SITES = ("", "decode", "prefill", "round")


def _inject_directives() -> list:
    """The tick guard's GMLX_OOM_INJECT directives, as (kind, tick, site).
    Band directives belong to the governor."""
    out = []
    for part in os.environ.get("GMLX_OOM_INJECT", "").split(","):
        kind, _, rest = part.partition("@")
        kind = kind.strip()
        if kind not in ("throw", "cbfail", "kill"):
            continue
        t, _, site = rest.partition(":")
        site = site.strip()
        try:
            tick = int(t)
        except ValueError:
            continue
        if site in _SITES and not (kind == "kill" and site):
            out.append((kind, tick, site))
    return out


def _inject_due(site: str):
    """The directive due at ``site`` now, marked fired, or None. Each fires
    once per process: tick counters restart per batch cycle."""
    if not os.environ.get("GMLX_OOM_INJECT"):
        return None
    for d in _inject_directives():
        kind, tick, at = d
        if at == site and _INJECT_TICK[0] >= tick and d not in _INJECT_FIRED:
            _INJECT_FIRED.add(d)
            return d
    return None


def _inject_error(d) -> RuntimeError:
    kind, tick, site = d
    where = f" at {site}" if site else ""
    if kind == "cbfail":
        _log.warning("[tick-guard] INJECT: synthetic command buffer OOM%s, "
                     "tick %d", where, _INJECT_TICK[0])
        return RuntimeError(_CB_OOM)
    _log.warning("[tick-guard] INJECT: synthetic allocator throw%s, tick %d",
                 where, _INJECT_TICK[0])
    return RuntimeError(_REFUSAL)


class _FailingRound:
    """A rounds iterator whose next round raises ``err`` inside the rounds
    generator, as a failure in the round would."""

    def __init__(self, inner, err):
        self.inner, self.err = inner, err

    def __iter__(self):
        return self

    def __next__(self):
        err, self.err = self.err, None
        if err is not None:
            return self.inner.throw(err)
        return next(self.inner)

    def close(self):
        self.inner.close()


def _maybe_inject(gen, st: _TickState) -> None:
    """Fault injection, never set in production. GMLX_OOM_INJECT takes
    ``kind@N`` or ``kind@N:site``. ``throw`` raises an allocator refusal,
    ``cbfail`` a command buffer out of memory, and ``kill`` sends SIGKILL
    to the worker. With no site the error comes at the start of tick N.
    The sites fire the first time they are reached at or after tick N:
    ``decode`` at the end of a tick that decoded, ``prefill`` as a prompt
    batch's ``prompt_step`` returns, after the chunk's forward, and
    ``round`` inside the next round of a speculative batch. Band directives belong to the
    governor."""
    if not os.environ.get("GMLX_OOM_INJECT"):
        return
    _INJECT_TICK[0] = st.ticks
    d = _inject_due("")
    if d is not None:
        st.injected.add(d)
        if d[0] == "kill":
            _log.error("[tick-guard] INJECT: SIGKILL at tick %d", st.ticks)
            import signal

            os.kill(os.getpid(), signal.SIGKILL)
        raise _inject_error(d)
    gb = getattr(gen, "_generation_batch", None)
    rounds = getattr(gb, "_rounds_iter", None)
    if rounds is not None and not isinstance(rounds, _FailingRound):
        d = _inject_due("round")
        if d is not None:
            st.injected.add(d)
            gb._rounds_iter = _FailingRound(rounds, _inject_error(d))


def _inject_after_decode(st: _TickState, out) -> None:
    if not os.environ.get("GMLX_OOM_INJECT") or not out[1]:
        return
    d = _inject_due("decode")
    if d is not None:
        st.injected.add(d)
        e = _inject_error(d)
        salvage(e, out[1])
        raise e


def _inject_in_prefill() -> None:
    if not os.environ.get("GMLX_OOM_INJECT"):
        return
    d = _inject_due("prefill")
    if d is not None:
        raise _inject_error(d)


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


def mid_close(row) -> bool:
    """Whether a row's thinking-budget close has an id out: its MTP hook's
    or its criteria's. A stood-down hook is never closing."""
    tc = getattr(row, "tc", None)
    if tc is None:
        return False
    hook = getattr(tc, "_kq_mtp_hook", None)
    if hook is not None and hook.closing():
        return True
    closing = getattr(tc, "closing", None)
    return bool(closing is not None and closing())


def speaks(gen, uid) -> bool:
    """Whether ``uid`` is a speaking row. It never requeues: the orange
    retire passes over it, and every other requeue fails it."""
    from . import speech_rows

    return speech_rows.speaks(gen, uid)


def _pick_victim(gen, st: _TickState, spare=None, spare_uid=None):
    """Largest live decode row by resident tokens (prompt + emitted);
    deterministic, avoiding an immediate repeat victim on ties. A row
    ``spare(row)`` or ``spare_uid(uid)`` holds is passed over."""
    rows = []
    for uid, ntok in _live_decode_rows(gen):
        row = st.ledger.get(uid)
        if spare is not None and row is not None and spare(row):
            continue
        if spare_uid is not None and spare_uid(uid):
            continue
        plen = len(row.prompt_ids) if row else 0
        rows.append((plen + ntok, uid))
    if not rows:
        return None
    rows.sort(reverse=True)
    if len(rows) > 1 and rows[0][1] == st.last_victim \
            and rows[1][0] == rows[0][0]:
        return rows[1][1]
    return rows[0][1]


_REMOVE_TRIES = 3
_STAYS = object()     # a removal refused through every retry
_DROPPED = object()   # a command buffer failed in the removal; engine dropped


def _take_out(gen, uid):
    """``gen.remove(uid)`` under an allocator refusal at a host read inside
    it. Returns ``(removed, swept)``: what remove returned, and the other
    decode rows taken out with the row, which the caller requeues.

    A refusal before the filter changed the batch leaves the row in it, so
    the pool is reclaimed and the removal tried again, up to
    ``_REMOVE_TRIES`` times; after that the row stays and ``removed`` is
    ``_STAYS``. A refusal inside one layer's own filter leaves the layers
    disagreeing, and no filter can finish that batch: it is dropped whole,
    and the rows left in it are swept. Any other error, a command buffer
    failure among them, propagates."""
    from gmlx.eval_guard import drain_for

    for _ in range(_REMOVE_TRIES):
        try:
            return gen.remove(uid), []
        except RuntimeError as e:
            if not is_allocator_refusal(e):
                raise
            gb = gen._generation_batch
            if getattr(gb, "_kq_torn", False):
                # A row that finished on a tick whose filter failed has
                # delivered its finish already.
                done = getattr(gb, "_kq_pending_drop", None) or ()
                swept = [u for u in getattr(gb, "uids", ())
                         if u != uid and u not in done]
                _log.warning("[tick-guard] removing row %s stopped a layer's "
                             "filter half way; dropping the batch and "
                             "requeueing its %d other row(s): %s", uid,
                             len(swept), str(e).splitlines()[0][:200])
                gen._kq_empty_generation_batch()
                mx.clear_cache()
                _close_speech(gen, uid)
                return True, swept
            _log.warning("[tick-guard] removing row %s was refused before "
                         "the batch changed; reclaiming and retrying: %s",
                         uid, str(e).splitlines()[0][:200])
            drain_for("engine-remove")
            mx.clear_cache()
    _log.error("[tick-guard] row %s stays in the batch: its removal was "
               "refused %d times", uid, _REMOVE_TRIES)
    return _STAYS, []


def _close_speech(gen, uid) -> None:
    from . import speech_rows

    speech_rows.close_row(gen, uid)


def _take_out_or_drop(gen, st: _TickState, uid, *, keep_out: bool,
                      carry: BaseException | None = None, also=()):
    """``_take_out``, with a command buffer failure inside the removal sent
    to the command buffer recovery: nothing the engine holds is valid, so
    it is dropped and every row requeues, except ``uid`` when ``keep_out``.
    ``carry`` is the error being contained, whose prefill rows go too,
    and ``also`` rows the caller took out already. A fault that is not a
    memory error counts toward the guard's one contained fault. Returns
    ``(_DROPPED, [])`` then."""
    from gmlx.eval_guard import drain_for

    try:
        return _take_out(gen, uid)
    except RuntimeError as err:
        if not is_command_buffer_failure(err):
            raise
        if is_gpu_fault(err):
            st.fault_streak += 1
        _log.error("[tick-guard] removing row %s hit a command buffer "
                   "failure; dropping the engine: %s", uid,
                   str(err).splitlines()[0][:200])
        if carry is not None:
            for name in ("_kq_prefill_broken", "_kq_prefill_meta"):
                if getattr(carry, name, None) and not getattr(err, name, None):
                    setattr(err, name, list(getattr(carry, name)))
        drain_for("engine-remove")
        mx.clear_cache()
        # The filter can have taken the row out before the failure.
        if keep_out:
            _drop_engine(gen, st, err, leave=(uid,), also=also)
        else:
            _drop_engine(gen, st, err, also=[*also, uid])
        if keep_out:
            _close_speech(gen, uid)
        return _DROPPED, []


def remove_row(gen, uid) -> bool:
    """Take ``uid`` out of the engine, for a caller outside the tick (a
    cancel). A refusal or a command buffer failure inside the removal
    never tears the batch or fails the rows beside it: they requeue as
    themselves. False when the row is not out, so the caller retries it."""
    st = getattr(gen, "_kq_tick_guard", None)
    if st is None:
        return gen.remove(uid)
    removed, swept = _take_out_or_drop(gen, st, uid, keep_out=True)
    if removed is _DROPPED:
        return True
    _requeue_rows(gen, st, swept)
    return removed is not _STAYS and bool(removed)


def _fail_row_permanently(gen, st: _TickState, uid, err: str, *,
                          remove: bool = True, strike: bool = False) -> None:
    """Fail ``uid`` for good. ``strike`` is the guard's own repeat
    failure, which sheds the row even when its removal stays refused; any
    other caller (the governor) leaves the row decoding then."""
    swept = []
    if remove:
        removed, swept = _take_out_or_drop(gen, st, uid, keep_out=True)
        if removed is _STAYS:
            if not strike:
                return
            # The pool refuses even the filter: the batch goes without
            # one, so the strike still sheds this row.
            swept = [u for u in _drop_decode_batch(gen) if u != uid]
    row = st.ledger.pop(uid, None)
    _requeue_rows(gen, st, swept)
    # A spoken reply's record closes on every path that ends its row.
    from . import speech_rows

    speech_rows.close_row(gen, uid)
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


def _resumed_processors(lp, committed):
    """The row's processors for its replay. The replay's prompt ends with
    the delivered tokens, so the presence and frequency penalties start
    afresh from them, and a grammar resumes after them."""
    if not lp:
        return lp
    return [p.resumed(committed) if hasattr(p, "resumed") else p
            for p in lp]


def _requeue(gen, st: _TickState, uids) -> list:
    """Put rows the caller took out of the engine back at the head of the
    queue, in order. Each replays its prompt plus every token it
    delivered, with its criteria rewound and its processors resumed to
    that point. Every requeue goes through here, so none skips a step.
    Returns the uids requeued."""
    entries, done = [], []
    for uid in uids:
        row = st.ledger.get(uid)
        if row is None:
            _log.error("[tick-guard] no ledger entry for %s; removed without "
                       "requeue", uid)
            continue
        remaining = None
        if isinstance(row.max_tokens, int):
            remaining = max(row.max_tokens - len(row.committed), 0)
            if remaining == 0:
                # the row had already delivered its budget; it is done
                st.ledger.pop(uid, None)
                continue
        try:
            if speaks(gen, uid):
                # A frame's token does not carry its codes, and a prompt
                # without embeddings would replay with no error.
                raise ValueError("a spoken reply cannot replay its frames")
            kwargs = _replay_kwargs(gen, uid, row)
        except Exception as e:
            # A wrong replay is worse than a failed row: the client gets an
            # error, not a corrupt answer.
            _log.error("[tick-guard] row %s cannot replay: %s", uid, e)
            _fail_row_permanently(gen, st, uid, f"replay unsupported: {e}",
                                  remove=False)
            continue
        rewind = getattr(row.tc, "rewind", None)
        if rewind is not None:
            # This runs inside the memory-error handler: a failed rewind
            # still requeues the row.
            try:
                rewind(list(row.committed))
            except Exception:
                _log.warning("[tick-guard] row %s: criteria rewind failed",
                             uid, exc_info=True)
        entries.append((uid, row.prompt_ids + row.committed, remaining,
                        kwargs, _resumed_processors(row.lp, row.committed),
                        row.tc))
        st.rebuilt.add(uid)
        done.append(uid)
    gen._unprocessed_sequences[:0] = entries
    for uid in done:
        row = st.ledger[uid]
        _log.warning("[tick-guard] row %s retired and requeued: replaying "
                     "%d prompt + %d delivered tokens", uid,
                     len(row.prompt_ids), len(row.committed))
    return done


def _replay_kwargs(gen, uid, row):
    """The row's prompt kwargs for its replay. The ledger keeps the
    prompt's, so a later requeue extends from them again."""
    from .replay_prompt import replay_prompt_kwargs

    if not row.committed or not row.kwargs \
            or row.kwargs.get("inputs_embeds") is None:
        return row.kwargs
    embed = getattr(gen, "_kq_embed_text", None)
    if embed is None:
        raise ValueError("no text embedder on the generator")
    settings = (getattr(gen, "_row_settings", None) or {}).get(uid)
    return replay_prompt_kwargs(
        row.kwargs, len(row.prompt_ids), row.committed,
        lambda ids: embed(ids, settings), getattr(gen, "model", None))


def _requeue_rows(gen, st: _TickState, uids) -> None:
    """Requeue rows the caller took out of the engine, failing each when
    the requeue itself throws."""
    if not uids:
        return
    try:
        _requeue(gen, st, uids)
    except Exception as e:
        # The rows are out of the engine, and the governor drops whatever
        # escapes its tick: fail them here, so their clients hear of it.
        _log.error("[tick-guard] requeue of %s failed", list(uids),
                   exc_info=True)
        gone = set(uids)
        gen._unprocessed_sequences[:] = [
            q for q in gen._unprocessed_sequences if q[0] not in gone]
        for uid in uids:
            st.rebuilt.discard(uid)
            _fail_row_permanently(gen, st, uid, f"requeue failed: {e}",
                                  remove=False)


def _rebuild_row(gen, st: _TickState, uid, err: str = "", *,
                 strike: bool = False) -> None:
    """Take ``uid`` out and requeue it as itself. ``strike`` is the
    guard's own repeat failure: a removal that stays refused then drops
    the batch and rebuilds its rows together. Any other caller (the
    governor's retire, which passes over rows that cannot replay) leaves
    every row decoding then, and samples again."""
    removed, swept = _take_out_or_drop(gen, st, uid, keep_out=False)
    if removed is _DROPPED:
        return  # every row, this one too, requeued with the engine
    if removed is _STAYS and not strike:
        return
    if removed is _STAYS:
        # The pool refuses even the filter: the batch goes without one,
        # and its rows are rebuilt together, so a repeat sheds a row.
        rest = [u for u in _drop_decode_batch(gen) if u != uid]
        _rebuild_group(gen, st, [uid] + rest, err)
        return
    _requeue_rows(gen, st, [uid] + swept)


def _largest(st: _TickState, uids):
    """The largest of ``uids`` by prompt plus delivered tokens, ranked as
    ``_pick_victim`` ranks."""
    rows = []
    for uid in uids:
        row = st.ledger.get(uid)
        size = len(row.prompt_ids) + len(row.committed) if row else 0
        rows.append((size, uid))
    rows.sort(reverse=True)
    if len(rows) > 1 and rows[0][1] == st.last_victim \
            and rows[1][0] == rows[0][0]:
        return rows[1][1]
    return rows[0][1]


def _rebuild_group(gen, st: _TickState, uids, err: str) -> list:
    """Requeue rows the caller took out of the engine together. When some
    were rebuilt before, the largest of those fails for good and the rest
    requeue, so each repeat failure sheds one row. Returns the uids
    requeued."""
    uids = list(dict.fromkeys(uids))
    again = [u for u in uids if u in st.rebuilt]
    if again:
        victim = _largest(st, again)
        st.last_victim = victim
        _fail_row_permanently(gen, st, victim, err, remove=False)
        uids.remove(victim)
    return _requeue(gen, st, uids)


def _drop_prompt_batch(gen, e: BaseException) -> None:
    """Take the prompt batch out of the engine and give back the prefix
    blocks it held, as its own error path does."""
    pb = getattr(gen, "_prompt_batch", None)
    if pb is not None:
        pb._apc_harvest_enabled = False
        try:
            pb._release_apc_meta_blocks()
        except Exception:
            _log.warning("[tick-guard] prompt batch APC release failed",
                         exc_info=True)
        pb._apc_meta = []
        gen._prompt_batch = None
    meta = getattr(e, "_kq_prefill_meta", None) or []
    apc = getattr(gen, "apc_manager", None)
    if apc is not None:
        for m in meta:
            if m is not None:
                try:
                    apc.release(m.get("apc_blocks", []))
                except Exception:
                    _log.warning("[tick-guard] APC release failed",
                                 exc_info=True)


def _rebuild_broken(gen, st: _TickState, e: BaseException) -> bool:
    """Rebuild the rows the failed tick marked broken: decode rows whose
    state a retry would reuse wrongly, and the rows of the prefill arm.
    Returns True when the tick marked any."""
    decode = list(getattr(e, "_kq_broken", ()))
    prefill = getattr(e, "_kq_prefill_broken", None)
    if not decode and prefill is None:
        return False
    gb = gen._generation_batch
    swept = []
    if getattr(gb, "_kq_extending", False) or getattr(gb, "_kq_torn", False):
        # extend threw half way, or a layer's filter did: the batch's
        # fields disagree, so no filter can run on it. Every row in it is
        # marked; drop it whole.
        gen._kq_empty_generation_batch()
    else:
        for i, uid in enumerate(decode):
            removed, out = _take_out_or_drop(gen, st, uid, keep_out=False,
                                             carry=e, also=decode[:i])
            if removed is _DROPPED:
                return True  # every row requeued with the engine
            if removed is _STAYS:
                # Refused through every retry: the row and every marked
                # row after it leave with the batch.
                out = _drop_decode_batch(gen)
            if out:
                # The batch went whole; the marked rows still in it are
                # rebuilt with the group, the rest requeue as themselves.
                swept = [u for u in out if u not in decode]
                break
    if prefill is not None:
        _drop_prompt_batch(gen, e)
    done = _rebuild_group(gen, st, decode + list(prefill or ()), str(e))
    _requeue_rows(gen, st, swept)
    _log.warning("[tick-guard] rebuilt %d row(s) whose engine state cannot "
                 "resume", len(done))
    return True


def _drop_decode_batch(gen) -> list:
    """Take the decode batch out of the engine without evaluating or
    filtering it. Returns the uids it served that had not finished."""
    gb = gen._generation_batch
    if hasattr(gb, "_finished") and hasattr(gb, "_all_uids"):
        uids = [u for u, done in zip(gb._all_uids, gb._finished) if not done]
        for other in getattr(gb, "_pending_injections", None) or ():
            uids += list(getattr(other, "_all_uids", ()))
        gb._pending_injections = []
        injections = getattr(getattr(gb, "model", None),
                             "_generator_injections", None)
        if injections:
            injections.clear()
        gb._finished = [True] * len(gb._finished)
        gb._refresh_uids()
        release = getattr(gb, "_kq_release_heavy_state", None)
        if release is not None:
            release()
        else:
            rounds, gb._rounds_iter = getattr(gb, "_rounds_iter", None), None
            if rounds is not None:
                try:
                    rounds.close()
                except Exception:
                    _log.warning("[tick-guard] rounds close failed",
                                 exc_info=True)
    else:
        uids = list(getattr(gb, "uids", ()))
    gen._kq_empty_generation_batch()
    return uids


def _drop_engine(gen, st: _TickState, e: BaseException, leave=(),
                 also=()) -> None:
    """A command buffer failed: nothing the engine holds is valid, nor
    what the window's stores wrote. Drop the decode and prompt batches
    without evaluating them, take back the window's stores, and requeue
    every row from what it delivered, with the rows ``also`` names and
    except the rows ``leave`` names."""
    from gmlx.cache.snapshot import retirement_blocked

    pb = getattr(gen, "_prompt_batch", None)
    with retirement_blocked():
        uids = _drop_decode_batch(gen)
        if pb is not None:
            uids += list(getattr(pb, "uids", ()))
        uids += list(getattr(e, "_kq_prefill_broken", ()))
        # A row the failed tick finished has left the batch, and its finish
        # is dropped with the salvage.
        uids += [r.uid for r in getattr(e, "_kq_salvage", ())
                 if getattr(r, "uid", None) in st.ledger]
        _drop_prompt_batch(gen, e)
    evicted = 0
    evict = getattr(getattr(gen, "apc_manager", None), "kq_evict", None)
    if evict is not None:
        try:
            evicted = evict(st.ticks - 1, owner=pb)
        except Exception:
            _log.warning("[tick-guard] APC window eviction failed",
                         exc_info=True)
    uids = [u for u in dict.fromkeys(uids + list(also)) if u not in leave]
    done = _rebuild_group(gen, st, uids, str(e))
    _log.warning("[tick-guard] command buffer failure: requeued %d row(s), "
                 "evicted %d APC entries stored since tick %d", len(done),
                 evicted, st.ticks - 1)


def _contain_fault(st: _TickState, e: RuntimeError) -> bool:
    """A non-memory command buffer failure. Contain the first one, since a
    driver hiccup should not take the server down, and let a second one
    through: reclaiming buffers cannot fix a fault in the work itself, so
    retrying past that only hides the kernel bug behind a wedged loop.
    Returns True when the tick was contained."""
    from gmlx.eval_guard import drain_for

    st.fault_streak += 1
    if st.fault_streak < 2:
        _log.error("[tick-guard] GPU fault, not a memory error, containing "
                   "one retry: %s", str(e).splitlines()[0][:200])
        drain_for("engine-tick")
        mx.clear_cache()
        return True
    _log.error("[tick-guard] GPU fault repeated; not containing it, because "
               "reclaiming memory cannot fix a fault in the submitted work: "
               "%s", str(e).splitlines()[0][:200])
    return False


def _contain(gen, st: _TickState, e: RuntimeError) -> None:
    from gmlx.eval_guard import drain_for

    st.fail_streak += 1
    _log.warning("[tick-guard] engine tick memory error (streak %d): %s",
                 st.fail_streak, str(e).splitlines()[0][:200])
    drain_for("engine-tick")
    mx.clear_cache()
    if is_command_buffer_failure(e):
        _drop_engine(gen, st, e)
        return  # every row is requeued; no victim on this strike
    if _rebuild_broken(gen, st, e):
        return  # those rows' memory is freed; no victim on this strike
    if st.fail_streak < 2:
        return  # pool reclaimed; the serve loop retries the tick
    victim = _pick_victim(gen, st)
    if victim is None:
        _log.error("[tick-guard] repeat failure with no removable row")
        return
    st.last_victim = victim
    if victim in st.rebuilt:
        _fail_row_permanently(gen, st, victim, str(e), strike=True)
    else:
        _rebuild_row(gen, st, victim, str(e), strike=True)


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


def _install_store_owner(cls) -> None:
    """A prompt batch's APC stores belong to it, so dropping it after a
    command buffer failure takes them back whatever their tick."""
    for name in ("prompt_step", "generate"):
        orig = getattr(cls, name)
        if wrapped_with(orig, _INSTALLED_FLAG):
            continue

        def owned(self, *a, _orig=orig, _name=name, **k):
            scope = getattr(getattr(self, "_apc_manager", None), "kq_scope",
                            None)
            if scope is None:
                out = _orig(self, *a, **k)
            else:
                with scope(self):
                    out = _orig(self, *a, **k)
            if _name == "prompt_step":
                # After the chunk's forward: the batch still holds its rows.
                _inject_in_prefill()
            return out

        setattr(owned, _INSTALLED_FLAG, True)
        setattr(cls, name, link(owned, orig))


def install_tick_guard() -> bool:
    """Wrap BatchGenerator._next outermost. Idempotent."""
    if os.environ.get("GMLX_TICK_GUARD", "1") == "0":
        return False
    from gmlx._vendor.mlx_vlm.generate import ar as _ar

    if wrapped_with(_ar.BatchGenerator._next, _INSTALLED_FLAG):
        return True
    _orig = _ar.BatchGenerator._next

    def _guarded_next(self, **kwargs):
        st = _state(self)
        st.ticks += 1
        _harvest_pending(self, st)
        mark = getattr(getattr(self, "apc_manager", None), "kq_mark_tick", None)
        if mark is not None:
            mark(st.ticks, getattr(self, "_prompt_batch", None))
        try:
            _maybe_inject(self, st)
            out = _orig(self, **kwargs)
            _inject_after_decode(st, out)
        except RuntimeError as e:
            if is_gpu_fault(e):
                if not _contain_fault(st, e):
                    raise
                _drop_engine(self, st, e)
                return [], []
            if not is_memory_error(e):
                raise
            # A refusal ran no work, so what the tick computed goes out,
            # and a rebuild below replays after it.
            saved = (list(getattr(e, "_kq_salvage", ()))
                     if is_allocator_refusal(e) else [])
            _harvest_responses(st, saved)
            _contain(self, st, e)
            return [], saved
        st.fail_streak = 0
        st.fault_streak = 0
        _harvest_responses(st, out[1])
        return out

    setattr(_guarded_next, _INSTALLED_FLAG, True)
    _ar.BatchGenerator._next = link(_guarded_next, _orig)
    _install_store_owner(_ar.PromptProcessingBatch)
    _log.info("engine tick guard installed")
    return True
