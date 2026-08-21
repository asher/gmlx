"""Cache-freshness admission gate (GMLX_APC_FRESH_WAIT_MS).

Sibling fan-out requests that arrive together co-admit into one mixed
prompt batch. Each row's APC pick runs at batch formation, before any
row stores, so every sibling prefills the shared prefix cold and the
batch does that work once per row. The other freshness windows are
already closed: checkpoint-tier and pooling models form prompt batches
one request at a time, and every store commits on the engine thread
before the finish response leaves the generator, so a request fired on
another's completion always sees its records.

The gate sits on ``BatchGenerator._next``, the admit_gate pattern: on a
tick that would form a prompt batch, it walks the candidates in FCFS
order and cuts the list before the first follower whose shared prefix
with an earlier candidate is not yet covered by any tier (block chain,
exact entries, anchor LRU -- a stats-neutral peek of the same indexes
the pick reads). The held follower keeps its queue position; the next
formation runs after the leader's prefill has published its stores, and
the follower admits warm. Nothing overtakes a held request and running
rows never wait.

A follower held longer than GMLX_APC_FRESH_WAIT_MS admits cold anyway:
a leader that never covers the shared prefix must not hold its siblings
forever. The timeout is the deadlock kill, not tunable politeness.

Knobs:
    GMLX_APC_FRESH_WAIT_MS   hold ceiling in ms (default 500; 0 = off)
    GMLX_APC_FRESH_MIN       minimum uncovered shared tokens before a
                                 follower is held (default 256)
"""

from __future__ import annotations

import logging
import os
import time

from .batch_rows import batch_rows

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_fresh_gate"
_LCP_CHUNK = 4096

# Server-wide counters for /v1/metrics. A hold counts once per request
# entering the held state, not per held tick.
_HOLDS = 0
_LAST_HOLD = ""


def fresh_stats() -> dict:
    return {"holds": _HOLDS, "last_hold_reason": _LAST_HOLD or None}


def _wait_ms() -> float:
    try:
        return float(os.environ.get("GMLX_APC_FRESH_WAIT_MS", "500"))
    except ValueError:
        return 500.0


def _hold_min() -> int:
    try:
        return int(os.environ.get("GMLX_APC_FRESH_MIN", "256"))
    except ValueError:
        return 256


def _lcp(a, b) -> int:
    """Longest common prefix of two token lists. Chunked slice compares
    keep the loop in C for deep prompts."""
    n = min(len(a), len(b))
    p = 0
    while p < n:
        step = min(_LCP_CHUNK, n - p)
        if a[p:p + step] == b[p:p + step]:
            p += step
            continue
        while p < n and a[p] == b[p]:
            p += 1
        break
    return p


def _covered_len(manager, ids, extra_hash: int) -> int:
    """Longest stored prefix of ``ids`` across every in-memory tier.

    Reads the same indexes the admission pick reads (block hash chain,
    exact entries, anchor LRU) without acquiring blocks, cloning, or
    touching hit counters, so a peek never distorts stats or LRU order.
    """
    from mlx_vlm import apc as _apc

    tt = tuple(int(t) for t in ids)
    best = 0
    with manager.lock:
        bs = int(getattr(manager, "block_size", 0) or 0)
        table = getattr(manager, "hash_table", None)
        if bs > 0 and table:
            parent = _apc.SEED_PARENT_HASH
            for i in range(len(tt) // bs):
                chunk = tt[i * bs:(i + 1) * bs]
                h = _apc._hash_tokens(parent, chunk, extra_hash)
                blk = table.get(h)
                if blk is None or blk.token_ids != chunk:
                    break
                parent = h
                best = (i + 1) * bs
        for entry in (getattr(manager, "_exact_cache", None) or {}).values():
            p = len(entry.token_ids)
            if (entry.extra_hash == extra_hash and best < p < len(tt)
                    and tt[:p] == tuple(entry.token_ids)):
                best = p
        for kids, kh in (getattr(manager, "_kq_anchor_cache", None) or {}):
            p = len(kids)
            if kh == extra_hash and best < p < len(tt) and tt[:p] == kids:
                best = p
    return best


def _note_hold(gen, uid, reason: str) -> bool:
    """Stamp the follower's first-held time; True while under the hold
    ceiling."""
    held = getattr(gen, "_kq_fresh_held", None)
    if held is None:
        held = gen._kq_fresh_held = {}
    now = time.perf_counter()
    first = held.get(uid)
    if first is None:
        held[uid] = now
        global _HOLDS, _LAST_HOLD
        _HOLDS += 1
        _LAST_HOLD = reason
        _log.info("APC fresh hold: %s", reason)
        return True
    if (now - first) * 1000.0 > _wait_ms():
        _log.warning(
            "APC fresh hold ceiling %.0fms hit: admitting uid=%s cold",
            _wait_ms(), uid)
        return False
    return True


def _keep_count(gen):
    """Candidates to admit this tick, or None for the full stock cut.

    Runs only when the stock body could form a prompt batch. The head is
    never held; the cut lands before the first follower whose uncovered
    shared prefix clears the floor, so FCFS order survives truncation.
    """
    manager = getattr(gen, "apc_manager", None)
    pending = gen._unprocessed_sequences
    held = getattr(gen, "_kq_fresh_held", None)
    if held:
        alive = {s[0] for s in pending}
        for uid in [u for u in held if u not in alive]:
            del held[uid]
    if manager is None or len(pending) < 2 or gen._prompt_batch is not None:
        return None
    num_to_add = gen.completion_batch_size - batch_rows(gen)
    if num_to_add < gen.prefill_batch_size:
        return None
    wait = _wait_ms()
    if wait <= 0:
        return None
    n = min(gen.prefill_batch_size, len(pending))
    if n < 2:
        return None
    floor = _hold_min()
    kept = [list(pending[0][1] or ())]
    for i in range(1, n):
        ids = list(pending[i][1] or ())
        shared = max((_lcp(ids, k) for k in kept), default=0)
        if shared < floor:
            kept.append(ids)
            continue
        extra_hash = gen._apc_extra_hash(pending[i][3] or {})
        covered = _covered_len(manager, ids, extra_hash)
        uncovered = shared - covered
        if uncovered < floor:
            kept.append(ids)
            continue
        uid = pending[i][0]
        reason = (f"uid={uid} shares {shared} tokens with a co-admitted "
                  f"candidate, {covered} covered; held for the leader's "
                  f"stores")
        if _note_hold(gen, uid, reason):
            return i
        kept.append(ids)
    return None


def install_fresh_admission_gate() -> None:
    """Hold sibling co-admission until the shared prefix is stored.

    Late-bound monkeypatch on ``BatchGenerator._next``, the admit_gate
    pattern: idempotent flag, env kill switch at install, decision
    failure degrades to stock admission. Installs after the headroom
    gate so a truncated tick still projects memory for the rows it
    admits.
    """
    from mlx_vlm.generate import ar as _ar

    if getattr(_ar.BatchGenerator._next, _INSTALLED_FLAG, False):
        return
    if _wait_ms() <= 0:
        return

    _orig_next = _ar.BatchGenerator._next

    def _gated_next(self, **kwargs):
        try:
            k = _keep_count(self)
        except Exception:
            _log.warning("fresh gate decision failed; admitting",
                         exc_info=True)
            k = None
        if k is None:
            return _orig_next(self, **kwargs)
        pending = self._unprocessed_sequences
        kept, tail = pending[:k], pending[k:]
        self._unprocessed_sequences = kept
        try:
            return _orig_next(self, **kwargs)
        finally:
            # The stock body rebinds the list, and a handler thread may
            # append mid-call; splice unconsumed candidates back ahead of
            # the held tail, arrivals behind it. Merge, never clobber.
            current = self._unprocessed_sequences
            kept_ids = {id(s) for s in kept}
            leftover = [s for s in current if id(s) in kept_ids]
            arrivals = [s for s in current if id(s) not in kept_ids]
            self._unprocessed_sequences = leftover + tail + arrivals

    setattr(_gated_next, _INSTALLED_FLAG, True)
    _ar.BatchGenerator._next = _gated_next
    _log.info("freshness admission gate installed")
