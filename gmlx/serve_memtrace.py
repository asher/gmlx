"""Per-tick memory trace for the batched serve loop (GMLX_SERVE_MEMSTATS).

Serve memory diagnosis needs per-owner attribution, not just totals: to
size an admission decision you have to know which bytes belong to the
live decode batch, the in-flight prompt batch, and a finished batch's
parked speculative state, and whether batched caches allocate padded to
a shared length. This wraps ``BatchGenerator._next`` outermost and
appends one JSON line per tick:

- MLX counters (active, buffer cache, peak) and the estimated free
  headroom (prefill_decay). All byte values.
- Tick wall time and the generator's prompt-time counter, so decode
  ticks and prefill-chunk ticks separate downstream.
- Decode batch rows, per-row emitted tokens, uids; prompt-batch uids and
  processed/total columns; pending uids awaiting admission.
- Per-owner cache attribution (``gen`` decode batch, ``pb`` prompt
  batch, ``spec`` parked speculative attrs): allocated bytes grouped by
  cache class, with offset and allocated-length ranges. Attribute
  arrays (keys, values, pools) are walked rather than ``c.state``:
  state slices report logical bytes, attribute arrays report the true
  allocation, and the difference is the padding and block-growth
  geometry the trace exists to measure.
- A per-owner allocation-shape signature; the record carries a full
  per-cache shape dump for an owner whenever its signature changed
  (block growth, rows joining or leaving), so growth-boundary crossings
  are the marked ticks.

Off unless GMLX_SERVE_MEMSTATS names a writable JSONL path. The file is
opened line-buffered append, so a hard process abort loses at most the
current line. The walk is pure Python attribute access: no evals, no
syncs, no lazy-slice construction on the tick path.
"""

from __future__ import annotations

import json
import logging
import os
import time

import mlx.core as mx

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_serve_memtrace"

_writer = None


def _arrays(v, depth: int = 2):
    """Yield mx.array leaves from v, recursing ``depth`` container levels
    (pools hold lists; quantized storage holds (data, scales, biases)
    tuples)."""
    if isinstance(v, mx.array):
        yield v
    elif depth > 0 and isinstance(v, (list, tuple)):
        for item in v:
            yield from _arrays(item, depth - 1)
    elif depth > 0 and isinstance(v, dict):
        for item in v.values():
            yield from _arrays(item, depth - 1)


def _leaf_caches(prompt_cache):
    """Flatten one CacheList level, same shape as prefill_decay's walk."""
    for entry in prompt_cache or ():
        subs = getattr(entry, "caches", None)
        for c in subs or (entry,):
            yield c


def _cache_report(prompt_cache):
    """(total_bytes, per-kind summary, shape signature, per-cache shapes).

    Per kind: cache count, allocated bytes, [min, max] integer offset,
    [min, max] allocated time-axis length (dim -2 of >=3-dim arrays).
    The signature is hashable and changes exactly when any allocation
    shape changes."""
    kinds: dict = {}
    shapes = []
    sig = []
    total = 0
    for i, c in enumerate(_leaf_caches(prompt_cache)):
        kind = type(c).__name__
        cbytes = 0
        cshapes = {}
        alens = []
        for name, v in sorted(vars(c).items()):
            arrs = list(_arrays(v))
            if not arrs:
                continue
            cbytes += sum(a.nbytes for a in arrs)
            shp = [list(a.shape) for a in arrs]
            cshapes[name] = shp[0] if len(shp) == 1 else shp
            alens.extend(a.shape[-2] for a in arrs if a.ndim >= 3)
        total += cbytes
        k = kinds.setdefault(
            kind, {"n": 0, "bytes": 0, "off": [], "alen": []})
        k["n"] += 1
        k["bytes"] += cbytes
        off = getattr(c, "offset", None)
        if isinstance(off, int):
            k["off"].append(off)
        k["alen"].extend(alens)
        lp = getattr(c, "left_padding", None)
        if isinstance(lp, (list, tuple)) and all(
                isinstance(x, int) for x in lp):
            k["lpad"] = list(lp)
        shapes.append({"i": i, "kind": kind, **cshapes})
        sig.append((kind, tuple((n, str(s)) for n, s in cshapes.items())))
    for k in kinds.values():
        for key in ("off", "alen"):
            k[key] = [min(k[key]), max(k[key])] if k[key] else None
    return total, kinds, tuple(sig), shapes


def _spec_bytes(batch):
    """Bytes parked on speculative-batch attrs (None when absent/empty)."""
    out = {}
    for name in ("hidden", "shared_kv_states", "prompt_tokens",
                 "first_tokens"):
        nb = sum(a.nbytes for a in _arrays(getattr(batch, name, None), 3))
        if nb:
            out[name] = nb
    return out or None


def _headroom():
    try:
        from .prefill_decay import _headroom_bytes

        head = _headroom_bytes()
        return None if head is None else int(head)
    except Exception:
        return None


def _record(gen, dt: float) -> dict:
    """One tick's trace record for a BatchGenerator. State (tick counter,
    shape signatures) lives on the generator under _kq_ attrs so
    multi-model serving never crosses signals."""
    tick = getattr(gen, "_kq_memtrace_tick", 0) + 1
    gen._kq_memtrace_tick = tick
    sigs = getattr(gen, "_kq_memtrace_sig", None)
    if sigs is None:
        sigs = gen._kq_memtrace_sig = {}
    gb = gen._generation_batch
    rec = {
        "t": round(time.time(), 3),
        "tick": tick,
        "dt_ms": round(dt * 1e3, 2),
        "ptime": round(gen._prompt_time_counter, 3),
        "act": mx.get_active_memory(),
        "cachemem": mx.get_cache_memory(),
        "peak": mx.get_peak_memory(),
        "head": _headroom(),
        "rows": len(gb),
        "uids": list(getattr(gb, "uids", ())),
        "pend": [s[0] for s in gen._unprocessed_sequences],
    }
    row_tok = getattr(gb, "_num_tokens", None)
    if row_tok:
        rec["row_tok"] = list(row_tok)
    pb = gen._prompt_batch
    if pb is not None:
        rec["pb"] = {
            "uids": list(getattr(pb, "uids", ())),
            "done": getattr(pb, "_processed_prompt_columns", None),
            "total": getattr(pb, "_total_prompt_tokens", None),
        }
    owners = {"gen": getattr(gb, "prompt_cache", None)}
    if pb is not None:
        owners["pb"] = getattr(pb, "prompt_cache", None)
    att = {}
    dumps = {}
    for owner, pc in owners.items():
        total, kinds, sig, shapes = _cache_report(pc)
        att[owner] = {"bytes": total, "kinds": kinds}
        if sig != sigs.get(owner):
            sigs[owner] = sig
            dumps[owner] = shapes
    spec = _spec_bytes(gb)
    if spec:
        att["spec"] = spec
    rec["own"] = att
    if dumps:
        rec["shapes"] = dumps
    return rec


def _emit(rec: dict) -> None:
    global _writer
    if _writer is None:
        return
    try:
        _writer.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception:
        _log.warning("serve memtrace write failed; trace disabled",
                     exc_info=True)
        _writer = None


def install_serve_memtrace() -> bool:
    """Wrap BatchGenerator._next with the per-tick trace when
    GMLX_SERVE_MEMSTATS names an output path. Must install after every
    other _next wrapper (pacing, admission) so the bracket times the full
    tick. Idempotent. Returns True when the trace is active."""
    global _writer
    path = os.environ.get("GMLX_SERVE_MEMSTATS", "")
    if not path:
        return False
    from mlx_vlm.generate import ar as _ar

    if getattr(_ar.BatchGenerator._next, _INSTALLED_FLAG, False):
        return True
    try:
        _writer = open(os.path.expanduser(path), "a", buffering=1)
    except OSError:
        _log.warning("GMLX_SERVE_MEMSTATS=%r is not writable; trace off",
                     path)
        return False
    _writer.write(json.dumps(
        {"meta": {"pid": os.getpid(),
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}}) + "\n")
    _orig_next = _ar.BatchGenerator._next

    def _traced_next(self, **kwargs):
        tic = time.perf_counter()
        try:
            return _orig_next(self, **kwargs)
        finally:
            try:
                _emit(_record(self, time.perf_counter() - tic))
            except Exception:
                _log.warning("serve memtrace sample failed", exc_info=True)

    setattr(_traced_next, _INSTALLED_FLAG, True)
    _ar.BatchGenerator._next = _traced_next
    _log.info("serve memtrace -> %s", path)
    return True
