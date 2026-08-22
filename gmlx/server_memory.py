"""Serve-side MLX buffer-cache policy.

The MLX buffer cache retains freed GPU buffers (wired) for reuse. Unbounded,
a deep-context request on a near-RAM-size model accumulates multi-GB prefill
transients in the cache and walks the box toward free-page exhaustion; the
fix is a bounded cache limit, large enough to recycle the biggest legitimate
transient, small enough to leave the slack to KV.

``resolve_cache_limit`` decides the limit once at server start:

  explicit env ``GMLX_CACHE_LIMIT_GB`` > config ``server.cache_limit_gb`` >
  auto policy > unlimited.

Explicit values: GiB float; ``0`` disables the cache entirely (MLX semantics);
``off``/``none``/``unlimited`` (or any negative number) force unlimited and
suppress the auto policy.

Auto policy: when the largest configured model's weight bytes exceed
``_AUTO_PRESSURE`` of the recommended working set, cap the cache at a quarter
of the remaining slack, clamped to [4, 12] GiB. Models with ample slack are
untouched (an unbounded cache is strictly good there), so small/medium-model
receipts are unchanged by this policy.
"""

from __future__ import annotations

import os

from .batch_rows import batch_rows

GIB = 1 << 30
_AUTO_PRESSURE = 0.6      # engage when weights > 60% of the working set
_AUTO_SLACK_FRACTION = 0.25
_AUTO_FLOOR = 4 * GIB
_AUTO_CEIL = 12 * GIB
_UNLIMITED_WORDS = ("off", "none", "unlimited")


def model_weight_bytes(path: str) -> int:
    """Total on-disk bytes of a GGUF (all shards). 0 when unresolvable
    (hf: refs not yet pulled, missing files) - auto policy then treats the
    model as weightless rather than guessing."""
    if not path or str(path).startswith("hf:"):
        return 0
    try:
        from .preflight import find_split_shards

        return sum(os.path.getsize(p) for p in find_split_shards(str(path)))
    except Exception:
        return 0


def auto_cache_limit_bytes(ws_bytes: float, weight_bytes: float) -> int | None:
    """The auto policy value, or None when pressure is low."""
    if ws_bytes <= 0 or weight_bytes <= _AUTO_PRESSURE * ws_bytes:
        return None
    slack = ws_bytes - weight_bytes
    return int(min(max(_AUTO_SLACK_FRACTION * slack, _AUTO_FLOOR), _AUTO_CEIL))


def _parse_explicit(raw: str) -> tuple[bool, int | None]:
    """(handled, bytes|None). None with handled=True means force-unlimited."""
    raw = raw.strip().lower()
    if not raw:
        return False, None
    if raw in _UNLIMITED_WORDS:
        return True, None
    try:
        gb = float(raw)
    except ValueError:
        return False, None
    if gb < 0:
        return True, None
    return True, int(gb * GIB)


def resolve_cache_limit(cfg_gb, model_paths, ws_bytes) -> tuple[int | None, str]:
    """(cache limit bytes | None, human-readable source). None => leave the
    MLX default (unlimited) in place."""
    handled, val = _parse_explicit(os.environ.get("GMLX_CACHE_LIMIT_GB", ""))
    if handled:
        return val, "env"
    if cfg_gb is not None:
        if float(cfg_gb) < 0:
            return None, "config"
        return int(float(cfg_gb) * GIB), "config"
    weights = max((model_weight_bytes(p) for p in model_paths), default=0)
    auto = auto_cache_limit_bytes(ws_bytes, weights)
    if auto is not None:
        return auto, (f"auto: weights {weights / GIB:.1f} GiB of "
                      f"{ws_bytes / GIB:.1f} GiB working set")
    return None, "unlimited"


# ---- Admission headroom projection ----------------------------------------
#
# Byte arithmetic for the admission gate (admit_gate): how much would
# admitting the candidate rows commit, against the measured free headroom.
# The KV term is measured, never derived: per cache kind, bytes per row
# token from a walk of the live decode batch's allocations, folded into an
# exponentially weighted mean on the generator. It self-corrects as
# kv_bits, pooling, or quantized storage change underneath.
#
# The projection is the padded form: every row is priced at the batch's
# maximum row length rounded up to the allocation block, because batched
# caches allocate rows to a shared padded length and under-projection is
# the direction that fails to prevent an abort. Rotating-window kinds are
# capped at their window so deep prompts do not price linear growth a ring
# will never hold.

_KV_EWM_ALPHA = 0.3
_STEP_BLOCK = 256


def _round_block(n: float) -> int:
    return -(-int(n) // _STEP_BLOCK) * _STEP_BLOCK


def admit_reserve_bytes(ws_bytes: float, gen=None) -> float:
    """Headroom held back beyond the projection, geometry-derived when a
    measured batch walk is available: one cache's transient (a filter
    gather or a join holds one layer's old and new arrays together) plus
    the whole-shed term (a single-row drop under per-cache eval holds
    live / (rows x caches) extra). The old constant (max of 2 GB and 5
    percent of the working set) stands in until the first walk, and
    GMLX_ADMIT_RESERVE_GB overrides everything. Decimal GB (1e9)
    throughout; GiB consumers convert at the edge."""
    env = os.environ.get("GMLX_ADMIT_RESERVE_GB", "")
    if env:
        try:
            return max(0.0, float(env)) * 1e9
        except ValueError:
            pass
    max_cache = getattr(gen, "_kq_admit_max_cache_bytes", 0.0) if gen \
        else 0.0
    if max_cache > 0:
        live = getattr(gen, "_kq_admit_live_bytes", 0.0)
        rows = max(1, batch_rows(gen))
        n_caches = max(1, getattr(gen, "_kq_admit_n_caches", 1))
        shed_term = live / (rows * n_caches)
        return max(1e9, max_cache + shed_term)
    return max(2e9, 0.05 * ws_bytes)


class cache_release_gate:
    """Pre-armed cache gate for a mutation seam that frees big arrays:
    pin the cache limit at its current fill immediately before the
    mutation, so every byte the mutation frees releases straight to the
    OS instead of growing the pool; synchronize the seam's stream on
    exit (a bare synchronize covers only the default stream), then
    restore the previous limit. Used after the fact the same call can
    no-op or evict the wrong end, so arm it before, never after."""

    def __init__(self, stream=None):
        self._stream = stream
        self._old = None

    def __enter__(self):
        import mlx.core as mx

        self._old = mx.set_cache_limit(int(mx.get_cache_memory()))
        return self

    def __exit__(self, *exc):
        import mlx.core as mx

        try:
            if self._stream is not None:
                mx.synchronize(self._stream)
            else:
                mx.synchronize()
        finally:
            if self._old is not None:
                mx.set_cache_limit(self._old)
        return False


def spec_state_bytes(gen):
    """Speculative-batch resident bytes the cache walk cannot see:
    the parked batch attrs (hidden, shared KV snapshot, prompt tokens,
    first tokens) and the drafter's own head KV. Returns
    ``(depth_scaled, row_const)``: bytes that grow with context depth
    (the shared KV snapshot and the drafter cache) and bytes that are
    per-row constants. Both zero when the batch is not speculative."""
    from .serve_memtrace import _arrays

    batch = gen._generation_batch
    depth_scaled = 0.0
    row_const = 0.0
    for name in ("shared_kv_states",):
        depth_scaled += sum(a.nbytes for a in _arrays(
            getattr(batch, name, None), 3))
    for name in ("hidden", "prompt_tokens", "first_tokens"):
        row_const += sum(a.nbytes for a in _arrays(
            getattr(batch, name, None), 3))
    draft = getattr(gen, "draft_model", None)
    dcache = getattr(draft, "_cache", None)
    if dcache:
        for c in dcache:
            for v in vars(c).values():
                depth_scaled += sum(a.nbytes for a in _arrays(v))
    return depth_scaled, row_const


def update_kv_rates(gen) -> None:
    """Fold a fresh per-kind KV bytes-per-row-token measurement of the live
    decode batch into the generator's running estimate (``_kq_admit_``
    attrs, the same convention the gate's defer state uses). Speculative
    state is folded in the same pass: depth-scaled spec bytes (shared KV
    snapshot, drafter head KV) become a synthetic per-kind rate so the
    projection prices their growth, and per-row constants land in
    ``_kq_admit_spec_row_const``. Offset-less caches (recurrent and conv
    state) are constant-size: they join the per-row constant, never the
    rate map."""
    batch = gen._generation_batch
    rows = batch_rows(gen)
    pc = getattr(batch, "prompt_cache", None)
    if rows <= 0 or not pc:
        return
    from .serve_memtrace import _arrays, _leaf_caches

    fresh: dict = {}
    live_bytes = 0.0
    live_depth = 0
    state_row_bytes = 0.0
    max_cache_bytes = 0.0
    n_caches = 0
    for c in _leaf_caches(pc):
        nbytes = 0
        alen = 0
        for v in vars(c).values():
            for a in _arrays(v):
                nbytes += a.nbytes
                if a.ndim >= 3:
                    alen = max(alen, int(a.shape[-2]))
        if not nbytes:
            continue
        live_bytes += nbytes
        max_cache_bytes = max(max_cache_bytes, float(nbytes))
        n_caches += 1
        off = getattr(c, "offset", None)
        off = off if isinstance(off, int) else 0
        if off <= 0:
            # No token offset means constant-size state, not KV: charge
            # per row, never per token. A bytes/state_dim rate would
            # scale constant state with depth.
            state_row_bytes += nbytes
            continue
        live_depth = max(live_depth, off)
        tokens = min(off, alen) if alen else off
        kind = fresh.setdefault(type(c).__name__,
                                {"rate": 0.0, "window": None})
        kind["rate"] += nbytes / tokens / rows
        window = getattr(c, "max_size", None)
        if isinstance(window, int) and window > 0:
            kind["window"] = (window if kind["window"] is None
                              else min(kind["window"], window))
    if not fresh:
        return
    spec_depth_bytes, spec_row_const = spec_state_bytes(gen)
    if spec_depth_bytes and live_depth > 0:
        fresh["_spec_state"] = {
            "rate": spec_depth_bytes / live_depth / rows, "window": None}
    live_bytes += spec_depth_bytes + spec_row_const
    prev = getattr(gen, "_kq_admit_kv_rates", None) or {}
    merged = {}
    for name, k in fresh.items():
        old = prev.get(name)
        rate = (k["rate"] if old is None else
                (1 - _KV_EWM_ALPHA) * old["rate"] + _KV_EWM_ALPHA * k["rate"])
        merged[name] = {"rate": rate, "window": k["window"]}
    gen._kq_admit_kv_rates = merged
    gen._kq_admit_live_bytes = live_bytes
    gen._kq_admit_live_depth = live_depth
    row_const = spec_row_const + state_row_bytes
    gen._kq_admit_spec_row_const = row_const / rows if row_const else 0.0
    gen._kq_admit_max_cache_bytes = max_cache_bytes
    gen._kq_admit_n_caches = n_caches


def project_admission(gen, candidates):
    """Projected bytes committing ``candidates`` on top of the live batch,
    against measured headroom.

    Returns ``(projected, headroom, parts)`` with parts a human-readable
    breakdown, or None when there is no measured basis to project (fresh
    model, empty batch, probe failure): the gate must admit then.
    ``candidates`` are pending-queue tuples (uid, prompt, max_tokens, ...).
    """
    import mlx.core as mx

    from .prefill_decay import headroom_bytes, score_transient_bytes

    update_kv_rates(gen)
    rates = getattr(gen, "_kq_admit_kv_rates", None)
    if not rates:
        return None
    head = headroom_bytes()
    if head is None:
        return None
    cand_tokens = []
    for s in candidates:
        try:
            prompt_toks = len(s[1])
        except TypeError:
            prompt_toks = 0
        max_toks = s[2] if isinstance(s[2], int) else 0
        cand_tokens.append(prompt_toks + max_toks)
    if not cand_tokens:
        return None
    width = batch_rows(gen) + len(cand_tokens)
    depth = _round_block(max([getattr(gen, "_kq_admit_live_depth", 0)]
                             + cand_tokens))
    kv_total = 0.0
    for k in rates.values():
        capped = depth if k["window"] is None else min(
            depth, _round_block(k["window"]))
        kv_total += k["rate"] * width * capped
    kv_total += getattr(gen, "_kq_admit_spec_row_const", 0.0) * width
    kv_new = max(0.0, kv_total - getattr(gen, "_kq_admit_live_bytes", 0.0))
    transient = score_transient_bytes(
        gen.model, getattr(gen._generation_batch, "prompt_cache", None),
        max(cand_tokens))
    try:
        ws = float(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        ws = 0.0
    reserve = admit_reserve_bytes(ws, gen)
    projected = kv_new + transient + reserve
    parts = (f"kv {kv_new / 1e9:.1f} + transient {transient / 1e9:.1f}"
             f" + reserve {reserve / 1e9:.1f}")
    # Stash for the governor's per-tick demand model: while a join is
    # pending (or its prompt batch is in flight) the declared next-tick
    # peak includes this projection, without re-walking anything.
    import time

    gen._kq_admit_last_projection = (time.perf_counter(), projected)
    return projected, head, parts


def apply_cache_limit(cfg) -> None:
    """Resolve and apply the server's cache limit; called once at startup.
    Working-set source: capacity (the one accounting, U4)."""
    import mlx.core as mx

    from .capacity import working_set_bytes

    ws = working_set_bytes() or 0.0
    paths = [str(mc.path) for mc in getattr(cfg, "models", {}).values()]
    limit, source = resolve_cache_limit(
        getattr(cfg, "cache_limit_gb", None), paths, ws)
    if limit is None:
        if source != "unlimited":
            print(f"[serve] MLX cache limit: unlimited ({source})")
        return
    mx.set_cache_limit(limit)
    from .prefill_decay import note_cache_limit

    note_cache_limit(limit)
    print(f"[serve] MLX cache limit: {limit / GIB:.1f} GiB ({source})")
