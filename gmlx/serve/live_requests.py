"""Live per-request view for /v1/metrics (``server.requests[]``).

A dispatcher sizing a fan-out, or an operator watching a busy server,
wants to see every request the server holds - waiting, prefilling, or
decoding - with its progress, not just per-model busy counts and a
completion-time ``[req]`` log line. This module publishes, once per
engine tick (rate-limited), one row per request the serve path knows
about:

    id, uid, model, state (queued | prefill | decode), position (queued
    rows), prompt_tokens, generated, max_tokens, elapsed_s, ttft_s,
    decode_tok_s, cache {tier: exact | block | ckpt | anchor | miss | hit,
    warm_tokens},
    speculative {rounds, accepted, drafted, accept_rate} | None

Sources, all read-only and all already in scope on the seam this wraps
(``ResponseGenerator._step(batch_gen, active)``):

* the server-side request queue (``rg.requests.queue``): submitted, not
  yet handed to the engine;
* the engine's unadmitted prompts (``batch_gen._unprocessed_sequences``);
* the prefill batch (``batch_gen._prompt_batch``: uids, per-row warm
  token counts, APC pick metadata for the tier);
* the decode batch (via ``tick_guard._live_decode_rows`` so speculative
  batches are read without the promoting ``__len__``);
* upstream's per-request registry ``active[uid]`` (request id, queue and
  phase timestamps, warm tokens after prefill, the speculative counter
  snapshot the acceptance rate is diffed against);
* the tick guard's ledger for prompt length and max_tokens of any uid.

gmlx's drafters (DFlash, MTP, DeepSeek) run inside the batch generator
(``spec_engine`` admits their ``SpeculativeGenerationBatch`` to the
continuous-batching loop), so their rows come from the same tick; the
acceptance numbers fall back to the drafter's per-generation
``accept_lens`` / ``draft_lens`` when stock's lifetime counters have not
moved. Stock mlx-vlm's non-MTP drafters instead take
``ResponseGenerator._run_speculative``, which prefills a static batch and
runs the round loop without ever calling ``_step``; for that path the
rows come from the per-request logging hooks the loop does call -
``_log_prefill_started`` (one per admitted request) and
``_log_decode_progress`` (one per emitted token, with the finish reason)
- tracked per engine thread, with the batch's speculative stats and no
cache tier.

One snapshot per engine (keyed by the ``ResponseGenerator`` identity):
several resident models tick concurrently, and a shared snapshot would
show whichever engine ticked last. The view merges every fresh one;
``position`` is within that model's queue.

The publisher never raises into the tick: any failure leaves the previous
snapshot in place. Readers get rows only while the snapshot is fresh
(``_STALE_S``); an idle engine stops ticking, and the last tick with an
empty ``active`` publishes an empty list.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import weakref

_log = logging.getLogger(__name__)

_STEP_FLAG = "_kq_gguf_live_requests"
_MIN_INTERVAL_S = 0.25
_STALE_S = 10.0

_LOCK = threading.Lock()
# id(rg) -> {"at": perf_counter, "rows": [...], "ref": weakref | None}
_SNAPS: dict = {}
_TIER: dict = {}          # (id(rg), uid) -> cache tier, captured during prefill
_WARM: dict = {}          # (id(rg), uid) -> prefix tokens a gmlx-owned tier restored
# Speculative engines: id(rg) -> {"ref", "snap", "rows": {uid: row state}}
_SPEC: dict = {}
_THREAD_RG: dict = {}     # engine thread ident -> id(rg) of its speculative loop
_MODEL_ID: dict = {}      # id(rg) -> (weakref | None, model id): fixed per engine
_SPEC_FLAG = "_kq_gguf_live_requests_spec"


def _snap_key(rg) -> int:
    return id(rg)


def _model_id(rg) -> str | None:
    """The configured id the engine ``rg`` serves. Memoized per engine:
    the mapping is fixed for the engine's lifetime, and resolving it walks
    the residency pool under its lock - which an eviction holds across a
    whole teardown, so an unmemoized tick would stall every other resident
    model's decode for that long."""
    key = _snap_key(rg)
    memo = _MODEL_ID.get(key)
    if memo is not None:
        ref, mid = memo
        if ref is None or ref() is rg:
            return mid
        _MODEL_ID.pop(key, None)         # id reused by a new engine
    mid = _resolve_model_id(rg)
    if mid is not None:
        try:
            ref = weakref.ref(rg)
        except TypeError:
            ref = None
        _MODEL_ID[key] = (ref, mid)
    return mid


def _resolve_model_id(rg) -> str | None:
    try:
        import importlib

        pkg = importlib.import_module("mlx_vlm.server")
        pool = getattr(pkg, "_kq_residency_pool", None)
        if pool is None:
            return None
        path = pool.model_path_for_generator(rg)
        if not path:
            return None
        from . import bridge_vlm as serving

        ids = getattr(serving, "_PATH_TO_IDS", {}).get(path) or []
        return ids[0] if ids else os.path.basename(path)
    except Exception:
        return None


def _ledger(batch_gen) -> dict:
    st = getattr(batch_gen, "_kq_tick_guard", None)
    return getattr(st, "ledger", None) or {}


def _tier_of(meta) -> str:
    if not isinstance(meta, dict):
        return "miss"
    if meta.get("apc_blocks"):
        return "block"
    return "exact" if int(meta.get("prefix_len") or 0) > 0 else "miss"


def _restored_of(pb, idx: int):
    """``(prefix_tokens, tier)`` when a gmlx-owned tier (the MTP L1 pick,
    the checkpoint restore) warmed this row. Those leave stock's
    ``prefix_len`` at 0 and trim the prompt in place, so the batch
    records the restore separately (``_kq_apc_restored``; single-row
    batches only)."""
    rec = getattr(pb, "_kq_apc_restored", None)
    if not rec or idx != 0 or len(list(getattr(pb, "uids", ()))) != 1:
        return None
    try:
        n, tier = int(rec[0]), str(rec[1])
    except Exception:
        return None
    return (n, tier) if n > 0 else None


def _row_tier(pb, idx: int, metas: list):
    """``(tier, warm)`` for a prefill row: the gmlx restore record wins,
    else stock's pick metadata (warm None: use the batch's per-row count)."""
    rest = _restored_of(pb, idx)
    if rest is not None:
        return rest[1], rest[0]
    return (_tier_of(metas[idx]) if idx < len(metas) else None), None


def _memo_tiers(rg, batch_gen, active) -> None:
    """Record each prefilling row's APC tier. Called every tick, ahead of
    the rate limit: a short prompt prefills inside a single tick, and the
    metadata is gone once the row moves to decode."""
    try:
        pb = getattr(batch_gen, "_prompt_batch", None)
        if pb is None:
            return
        key = _snap_key(rg)
        metas = list(getattr(pb, "_apc_meta", ()) or ())
        for idx, uid in enumerate(list(getattr(pb, "uids", ()))):
            if active is not None and uid not in active:
                continue
            tier, warm = _row_tier(pb, idx, metas)
            if tier is not None:
                _TIER[(key, uid)] = tier
            if warm:
                _WARM[(key, uid)] = warm
    except Exception:
        pass


def _tier_fallback(rg, uid, warm) -> str | None:
    """The memoized tier, else what the warm-token count alone can say
    (``hit`` / ``miss``; exact vs block needs the prefill metadata)."""
    tier = _TIER.get((_snap_key(rg), uid))
    if tier is not None:
        return tier
    if isinstance(warm, (int, float)):
        return "hit" if warm > 0 else "miss"
    return None


def _spec_of(rg, info) -> dict | None:
    draft = getattr(rg, "draft_model", None)
    snap = (info or {}).get("spec_snapshot")
    if draft is None or snap is None:
        return None
    try:
        from mlx_vlm.speculative.common import speculative_stats_since

        rounds, accepted, drafted = speculative_stats_since(draft, snap)
        if rounds is None:
            # gmlx's own drafters keep per-generation round lists (reset
            # at each batch start) rather than the lifetime totals stock
            # diffs; read those while the batch runs.
            al = list(getattr(draft, "accept_lens", None) or ())
            if not al:
                return None
            dl = list(getattr(draft, "draft_lens", None) or ())
            rounds, accepted = len(al), sum(al)
            drafted = sum(dl) if len(dl) == len(al) else 0
        rate = (float(accepted) / float(drafted)) if drafted else None
        return {"rounds": int(rounds), "accepted": float(accepted or 0),
                "drafted": int(drafted or 0),
                "accept_rate": round(rate, 3) if rate is not None else None}
    except Exception:
        return None


def _row(uid, info, *, model, state, now, prompt_tokens, generated,
         max_tokens, warm, tier, spec=None, position=None) -> dict:
    info = info or {}
    queued_at = info.get("queued_at")
    decode_at = info.get("decode_started_at")
    elapsed = (now - queued_at) if isinstance(queued_at, (int, float)) else None
    ttft = ((decode_at - queued_at)
            if isinstance(decode_at, (int, float))
            and isinstance(queued_at, (int, float)) else None)
    tok_s = None
    if isinstance(decode_at, (int, float)) and generated and now > decode_at:
        tok_s = generated / (now - decode_at)
    rid = info.get("request_id")
    row = {
        "id": rid if rid else (f"uid{uid}" if uid is not None else None),
        "uid": uid,
        "model": model,
        "state": state,
        "prompt_tokens": prompt_tokens,
        "generated": int(generated or 0),
        "max_tokens": max_tokens,
        "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "decode_tok_s": round(tok_s, 1) if tok_s is not None else None,
        "cache": {"tier": tier, "warm_tokens": warm},
        "speculative": spec,
    }
    if position is not None:
        row["position"] = position
    return row


def build_rows(rg, batch_gen, active, now=None) -> list:
    """One dict per known request, queued rows first in queue order.
    Pure over its inputs; every read is best-effort."""
    now = time.perf_counter() if now is None else now
    active = active or {}
    model = _model_id(rg)
    ledger = _ledger(batch_gen)
    key = _snap_key(rg)
    rows: list = []
    position = 0

    # server-side queue: submitted, not yet handed to the engine
    try:
        for req in list(getattr(getattr(rg, "requests", None), "queue", ())):
            args = getattr(req, "args", None)
            info = {"request_id": getattr(req, "request_id", None),
                    "queued_at": getattr(req, "queued_at", None)}
            rows.append(_row(None, info, model=model, state="queued", now=now,
                             prompt_tokens=getattr(req, "prompt_tokens", None),
                             generated=0,
                             max_tokens=getattr(args, "max_tokens", None),
                             warm=None, tier=None, position=position))
            position += 1
    except Exception:
        pass

    # engine-side: unadmitted prompts
    try:
        for item in list(getattr(batch_gen, "_unprocessed_sequences", ())):
            uid, ids, m = item[0], item[1], item[2]
            rows.append(_row(uid, active.get(uid), model=model, state="queued",
                             now=now, prompt_tokens=len(ids), generated=0,
                             max_tokens=m, warm=None, tier=None,
                             position=position))
            position += 1
    except Exception:
        pass

    live_uids = set()

    # prefill batch
    try:
        pb = getattr(batch_gen, "_prompt_batch", None)
        if pb is not None:
            uids = list(getattr(pb, "uids", ()))
            warm_per = list(getattr(pb, "_cached_tokens_per_row", ()) or ())
            metas = list(getattr(pb, "_apc_meta", ()) or ())
            for idx, uid in enumerate(uids):
                if uid not in active:
                    continue
                live_uids.add(uid)
                led = ledger.get(uid)
                tier, restored = _row_tier(pb, idx, metas)
                if tier is not None:
                    _TIER[(key, uid)] = tier
                if restored:
                    _WARM[(key, uid)] = restored
                warm = restored or (int(warm_per[idx]) if idx < len(warm_per) else None)
                rows.append(_row(
                    uid, active.get(uid), model=model, state="prefill", now=now,
                    prompt_tokens=len(led.prompt_ids) if led else None,
                    generated=0,
                    max_tokens=getattr(led, "max_tokens", None),
                    warm=warm, tier=_TIER.get((key, uid))))
    except Exception:
        pass

    # decode batch
    try:
        from .tick_guard import _live_decode_rows

        for uid, ntok in _live_decode_rows(batch_gen):
            if uid not in active or uid in live_uids:
                continue
            live_uids.add(uid)
            info = active.get(uid)
            led = ledger.get(uid)
            try:
                warm = (info or {}).get("cached_tokens")
                if not warm and (key, uid) in _WARM:
                    warm = _WARM[(key, uid)]
                rows.append(_row(
                    uid, info, model=model, state="decode", now=now,
                    prompt_tokens=len(led.prompt_ids) if led else None,
                    generated=ntok,
                    max_tokens=getattr(led, "max_tokens", None),
                    warm=warm,
                    tier=_tier_fallback(rg, uid, warm),
                    spec=_spec_of(rg, info)))
            except Exception:
                # one unreadable row must not hide the rest of the batch
                rows.append(_row(uid, info, model=model, state="decode", now=now,
                                 prompt_tokens=None, generated=ntok, max_tokens=None,
                                 warm=None, tier=None))
    except Exception:
        pass

    # speculative engine: rows tracked from the loop's logging hooks
    try:
        table = _SPEC.get(key)
        if table and table["rows"]:
            rows.extend(_spec_rows(rg, table, now, model))
    except Exception:
        pass

    return rows


def _prune_memos(key, active) -> None:
    """Drop the tier/warm memos of ``key``'s finished rows. Runs from
    ``publish`` on every tick, not from ``build_rows``: an engine that
    drains to empty skips the row build, which is exactly when its last
    rows need forgetting (else the memos grow by one entry per request
    for the process lifetime, and a reused ``id(rg)`` could read a dead
    engine's tier)."""
    for memo in (_TIER, _WARM):
        for k in list(memo):
            if k[0] == key and k[1] not in active:
                memo.pop(k, None)


# --- speculative engines -------------------------------------------------

def _spec_table(rg, create: bool = False) -> dict | None:
    key = _snap_key(rg)
    table = _SPEC.get(key)
    if table is None and create:
        try:
            ref = weakref.ref(rg)
        except TypeError:
            ref = None
        table = _SPEC[key] = {"ref": ref, "snap": None, "rows": {}}
    return table


def spec_prefill_started(rg, request, state: dict) -> None:
    """A request admitted to a speculative batch: track it from the
    prefill hook's log state (the dict the loop keeps per uid)."""
    try:
        table = _spec_table(rg, create=True)
        rows = table["rows"]
        # A new batch only starts after the previous one finished; rows
        # still in decode belong to a batch that died on the error path.
        if any(r["state"] == "decode" for r in rows.values()):
            rows.clear()
        if not rows:
            try:
                from mlx_vlm.speculative.common import speculative_stats_snapshot

                table["snap"] = speculative_stats_snapshot(getattr(rg, "draft_model", None))
            except Exception:
                table["snap"] = None
        uid = id(getattr(request, "rqueue", None))
        rows[uid] = {"state": "prefill", "info": state,
                     "prompt_tokens": getattr(request, "prompt_tokens", None),
                     "max_tokens": getattr(getattr(request, "args", None), "max_tokens", None)}
        _THREAD_RG[threading.get_ident()] = _snap_key(rg)
        publish(rg, None, None, force=True)
    except Exception:
        _log.debug("speculative prefill hook failed", exc_info=True)


def spec_decode_progress(uid, info: dict, finish_reason) -> None:
    """A token emitted by a speculative round loop (same engine thread as
    the prefill hook). No-op for uids the loop does not track."""
    try:
        key = _THREAD_RG.get(threading.get_ident())
        table = _SPEC.get(key) if key is not None else None
        if table is None:
            return
        row = table["rows"].get(uid)
        if row is None:
            return
        row["state"] = "decode"
        if finish_reason:
            table["rows"].pop(uid, None)
        rg = table["ref"]() if table["ref"] is not None else None
        if rg is None:
            return
        publish(rg, None, None, force=bool(finish_reason))
    except Exception:
        _log.debug("speculative decode hook failed", exc_info=True)


def _spec_rows(rg, table: dict, now: float, model) -> list:
    rows = []
    snap = table.get("snap")
    for uid, r in list(table["rows"].items()):
        info = r["info"]
        spec = _spec_of(rg, {"spec_snapshot": snap}) if snap is not None else None
        rows.append(_row(uid, info, model=model, state=r["state"], now=now,
                         prompt_tokens=r["prompt_tokens"],
                         generated=int(info.get("generated_tokens", 0) or 0),
                         max_tokens=r["max_tokens"], warm=None, tier=None,
                         spec=spec))
    return rows


def _has_spec_rows(rg) -> bool:
    table = _SPEC.get(_snap_key(rg))
    return bool(table and table["rows"])


def publish(rg, batch_gen, active, *, force: bool = False) -> None:
    """Refresh the snapshot, at most every ``_MIN_INTERVAL_S`` unless
    forced. Never raises."""
    now = time.perf_counter()
    try:
        key = _snap_key(rg)
        _memo_tiers(rg, batch_gen, active)
        _prune_memos(key, active or {})
        if batch_gen is not None:
            try:
                from .queue_cap import note_engine

                note_engine(rg, batch_gen)
            except Exception:
                pass
        with _LOCK:
            snap = _SNAPS.get(key)
        if not force and snap is not None and now - snap["at"] < _MIN_INTERVAL_S:
            return
        rows = [] if not active and not _has_queue(rg, batch_gen) \
            and not _has_spec_rows(rg) \
            else build_rows(rg, batch_gen, active, now=now)
        try:
            ref = weakref.ref(rg)
        except TypeError:
            ref = None
        with _LOCK:
            _SNAPS[key] = {"at": now, "rows": rows, "ref": ref}
    except Exception:
        _log.debug("live request snapshot failed", exc_info=True)


def _has_queue(rg, batch_gen) -> bool:
    try:
        q = getattr(getattr(rg, "requests", None), "qsize", None)
        if callable(q) and q() > 0:
            return True
        return bool(getattr(batch_gen, "_unprocessed_sequences", ()))
    except Exception:
        return False


def live_requests_view() -> list:
    """The ``requests`` section of /v1/metrics: the last published rows,
    or an empty list once the snapshot is older than ``_STALE_S`` (an
    idle engine stops ticking; nothing is live)."""
    now = time.perf_counter()
    out: list = []
    with _LOCK:
        for key, snap in list(_SNAPS.items()):
            ref = snap.get("ref")
            if ref is not None and ref() is None:      # engine torn down
                _SNAPS.pop(key, None)
                _SPEC.pop(key, None)
                _MODEL_ID.pop(key, None)
                for memo in (_TIER, _WARM):
                    for k in list(memo):
                        if k[0] == key:
                            memo.pop(k, None)
                continue
            if snap["rows"] and now - snap["at"] <= _STALE_S:
                out.extend(snap["rows"])
    return out


def _reset() -> None:
    """Test hook."""
    with _LOCK:
        _SNAPS.clear()
    _TIER.clear()
    _WARM.clear()
    _SPEC.clear()
    _THREAD_RG.clear()
    _MODEL_ID.clear()


def install_live_requests() -> None:
    """Wrap ``ResponseGenerator._step`` to publish after each tick.
    Passthrough: the wrapped step's result and exceptions are untouched.
    Idempotent."""
    from mlx_vlm.server import generation as gen_mod

    if getattr(gen_mod.ResponseGenerator._step, _STEP_FLAG, False):
        return
    _orig_step = gen_mod.ResponseGenerator._step

    def _publishing_step(self, batch_gen, active, gen_kwargs=None):
        try:
            return _orig_step(self, batch_gen, active, gen_kwargs)
        finally:
            publish(self, batch_gen, active, force=not active)

    setattr(_publishing_step, _STEP_FLAG, True)
    gen_mod.ResponseGenerator._step = _publishing_step
    _install_speculative_hooks(gen_mod)
    _log.info("live request view installed")


def _install_speculative_hooks(gen_mod) -> None:
    """Track speculative batches from the per-request logging hooks the
    round loop calls (it never steps a BatchGenerator). Passthrough on
    both; ``_log_decode_progress`` stays a staticmethod. Idempotent."""
    cls = gen_mod.ResponseGenerator
    if getattr(cls._log_prefill_started, _SPEC_FLAG, False):
        return
    _orig_prefill = cls._log_prefill_started
    _orig_progress = cls.__dict__["_log_decode_progress"]
    _orig_progress_fn = getattr(_orig_progress, "__func__", _orig_progress)

    def _prefill_started(self, request, *, backend):
        state = _orig_prefill(self, request, backend=backend)
        if isinstance(backend, str) and backend.startswith("speculative"):
            spec_prefill_started(self, request, state)
        return state

    def _decode_progress(uid, info, *, token, text, finish_reason, token_count=1):
        try:
            return _orig_progress_fn(uid, info, token=token, text=text,
                                     finish_reason=finish_reason,
                                     token_count=token_count)
        finally:
            spec_decode_progress(uid, info, finish_reason)

    setattr(_prefill_started, _SPEC_FLAG, True)
    setattr(_decode_progress, _SPEC_FLAG, True)
    cls._log_prefill_started = _prefill_started
    cls._log_decode_progress = staticmethod(_decode_progress)
