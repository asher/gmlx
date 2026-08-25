"""Boot capacity derivation: one accounting for what fits (U4).

At model build time the serve path derives a capacity table from the
same admit-side cost model requests are priced with (kv_layer_costs
windows and MLA latents included, the prefill score transient, the
admission reserve) against the working budget WS x (1 - M):

- max context at width 1, max width at representative depths, and the
  depth-width frontier across widths;
- the two non-byte ceilings: max_buffer_length (a single cache tensor
  or transient larger than it can never allocate, regardless of free
  bytes) and the Metal resource (buffer-count) limit, surfaced with
  the byte numbers;
- a boot refusal with the numbers for a configuration that cannot fit
  at width 1, instead of the box's biggest allocation aborting the
  process. GMLX_OVERCOMMIT=1 disables the refusal and the derived
  ceilings, keeps the trace, and logs that it is armed (the over-RAM
  decode program runs there deliberately).

The table is the shared base the split accountings derive from: the
residency weight budget and the cache auto-policy read working-set
bytes from here, memfit's share bars live here, decode concurrency is
min(GMLX_DECODE_BATCH, frontier width) and the queue cap follows it,
and /v1/metrics surfaces the table. Model load and swap take the same
headroom check a request takes (residency's build gate): a load that
does not fit refuses with numbers.

The derivation is header-based (headerscan + config synthesis via
tool_preflight), so it prices GGUF-served models; an HF fall-through
load logs that no table is available and keeps stock behavior.
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

GB = 1e9

# memfit's share bars, moved here so the user-facing accounting and the
# boot accounting read one source (values preserved).
TIGHT_SHARE = 0.65   # above this share of RAM, KV headroom gets scarce
OVER_SHARE = 0.85    # above this, the GPU path can't hold weights+cache

_WIDTHS = (1, 2, 4, 8, 16, 32)
_DEPTHS = (4096, 16384, 65536)
_FRONTIER_MIN_CTX = 4096

_TABLE: dict | None = None


def overcommit() -> bool:
    return os.environ.get("GMLX_OVERCOMMIT", "0") == "1"


def margin() -> float:
    try:
        return float(os.environ.get("GMLX_GOV_MARGIN", "0.05"))
    except ValueError:
        return 0.05


def working_set_bytes() -> float | None:
    import mlx.core as mx

    try:
        ws = float(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        return None
    return ws if ws > 0 else None


def reserve_bytes(mem_size: float) -> float:
    """Bytes left to the kernel and every other process on the box
    (GMLX_GOV_RESERVE_GB, default max(8 GB, 10% of RAM))."""
    default = max(8e9, 0.10 * mem_size)
    try:
        return float(os.environ.get("GMLX_GOV_RESERVE_GB",
                                    str(default / 1e9))) * 1e9
    except ValueError:
        return default


def ceiling_bytes(ws: float, m: float | None = None) -> float:
    """The one ceiling on tracked bytes: Metal's recommended working
    set less the margin, but never closer to physical RAM than the
    reserve. Metal's recommendation assumes the GPU owns the box; a
    serve process sharing it with the kernel and the user does not
    (2026-08-24: WS x 0.95 = 114 GB on a 128 GB box, box panicked)."""
    import mlx.core as mx

    cap = ws * (1.0 - (margin() if m is None else m))
    try:
        mem = float(mx.device_info()["memory_size"])
    except Exception:
        return cap
    return min(cap, mem - reserve_bytes(mem))


def working_budget_bytes() -> float | None:
    ws = working_set_bytes()
    return None if ws is None else ceiling_bytes(ws)


def classify_weight_share(size_bytes: int, ram_bytes: int) -> str:
    """memfit's fit classes, derived here: fits / tight / over
    (boundaries inclusive on the safe side, matching memfit)."""
    share = size_bytes / ram_bytes if ram_bytes else 0.0
    if share <= TIGHT_SHARE:
        return "fits"
    if share <= OVER_SHARE:
        return "tight"
    return "over"


def get_table() -> dict | None:
    return _TABLE


def frontier_width(min_ctx: int = _FRONTIER_MIN_CTX) -> int | None:
    """Widest decode batch whose max context still reaches ``min_ctx``,
    from the installed table; None without one (or with overcommit
    armed, which disables derived ceilings)."""
    t = _TABLE
    if t is None or overcommit():
        return None
    best = None
    for w in _WIDTHS:
        if t["max_ctx"].get(w, 0) >= min_ctx:
            best = w
    return best


def _transient_bytes(heads: int | None, depth: int) -> float:
    from .tool_preflight import _SCORE_CHUNK

    if not heads:
        return 0.0
    step = min(_SCORE_CHUNK, max(1, depth))
    return heads * step * (depth + step) * 2.0


def derive_table(gguf_path: str, weight_bytes: float | None = None
                 ) -> dict | None:
    """The capacity table for serving ``gguf_path`` on this box, or None
    when the header or the device cannot be read (callers keep stock
    behavior then)."""
    import mlx.core as mx

    from .server_memory import admit_reserve_bytes
    from .tool_preflight import _kv_costs, _shards, _synth_config

    try:
        shards = _shards(gguf_path)
        weights = (float(weight_bytes) if weight_bytes else
                   float(sum(os.path.getsize(p) for p in shards)))
        cfg = _synth_config(shards[0])
        costs = _kv_costs(cfg) if cfg else None
        ws = working_set_bytes()
        if not (cfg and costs and ws):
            return None
        info = mx.device_info()
        max_buffer = float(info.get("max_buffer_length", 0) or 0)
        resource_limit = int(info.get("resource_limit", 0) or 0)
    except Exception:
        _log.debug("capacity derivation skipped", exc_info=True)
        return None

    heads = cfg.get("num_attention_heads")
    heads = heads if isinstance(heads, int) and heads > 0 else None
    budget = ceiling_bytes(ws)
    reserve = admit_reserve_bytes(ws)

    from .mem_preflight import prompt_kv_bytes

    def fits(w: int, d: int) -> bool:
        kv = prompt_kv_bytes(costs, d)
        tr = _transient_bytes(heads, d)
        if weights + w * kv + tr + reserve > budget:
            return False
        if max_buffer > 0:
            if tr > max_buffer:
                return False
            for window, bpt in costs:
                span = d if window is None else min(d, int(window))
                if w * bpt * span > max_buffer:
                    return False
        return True

    def max_ctx(w: int) -> int:
        lo, hi = 0, 1 << 22
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(w, mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    ctx_by_width = {w: max_ctx(w) for w in _WIDTHS}
    width_at_depth = {}
    for d in _DEPTHS:
        best = 0
        for w in _WIDTHS:
            if ctx_by_width[w] >= d:
                best = w
        width_at_depth[d] = best

    trained = cfg.get("max_position_embeddings")
    return {
        "path": gguf_path,
        "weight_bytes": int(weights),
        "working_set_bytes": int(ws),
        "margin": margin(),
        "budget_bytes": int(budget),
        "reserve_bytes": int(reserve),
        "max_buffer_length": int(max_buffer),
        "resource_limit": resource_limit,
        "trained_ctx": trained if isinstance(trained, int) else None,
        "max_ctx": ctx_by_width,
        "max_width_at_depth": width_at_depth,
        "overcommit": overcommit(),
    }


def _log_table(t: dict) -> None:
    ctx = ", ".join(f"w{w}={t['max_ctx'][w]}" for w in _WIDTHS)
    _log.info(
        "[capacity] weights %.1f GB, budget %.1f GB (WS %.1f GB, margin "
        "%.2f), reserve %.1f GB, max single buffer %.1f GB, resource "
        "limit %d; max context by width: %s; trained %s",
        t["weight_bytes"] / GB, t["budget_bytes"] / GB,
        t["working_set_bytes"] / GB, t["margin"], t["reserve_bytes"] / GB,
        t["max_buffer_length"] / GB, t["resource_limit"], ctx,
        t["trained_ctx"])


def streamed_expert_bytes(gguf_path: str) -> int:
    """A-priori bytes of the routed-expert stacks (the tensors
    ``stream: experts`` serves from disk instead of wiring), summed from
    the GGUF header's tensor records without touching data. 0 when the
    header cannot be read; the caller then gates on the full size."""
    try:
        from gguf import GGUFReader

        from .preflight import find_split_shards

        total = 0
        for shard in find_split_shards(gguf_path):
            for t in GGUFReader(shard).tensors:
                name = str(t.name)
                if "_exps." in name or name.endswith("_exps"):
                    total += int(t.n_bytes)
        return total
    except Exception:
        return 0


def preload_gate_bytes(footprint: int, stream, expert_bytes: int) -> int:
    """Resident bytes the preload gate should judge. ``stream: experts``
    never wires the routed-expert stacks (they decode through the disk
    arena), so gating such a model on its full on-disk size refuses
    exactly the over-RAM serving the mode exists for; ``stream: cpu``
    still occupies unified RAM, so it gets no discount."""
    if stream == "experts" and expert_bytes > 0:
        return max(0, int(footprint) - int(expert_bytes))
    return int(footprint)


def preload_gate(weight_bytes: float, model_id: str) -> None:
    """The same headroom check a request takes, at model load and swap:
    a build whose weights cannot fit the measured free working set (the
    pool has already evicted what it may) refuses with numbers instead
    of aborting the box's biggest allocation. GMLX_OVERCOMMIT=1 skips."""
    if overcommit() or weight_bytes <= 0:
        return
    from .prefill_decay import headroom_bytes

    head = headroom_bytes()
    budget = working_budget_bytes()
    if budget is not None and weight_bytes > budget:
        raise RuntimeError(
            f"model does not fit: {model_id} weights "
            f"{weight_bytes / GB:.1f} GB exceed this box's working "
            f"budget {budget / GB:.1f} GB (working set x "
            f"{1 - margin():.2f}). GMLX_OVERCOMMIT=1 overrides; for MoE "
            f"models --stream-experts serves the experts from disk.")
    if head is not None and weight_bytes > head:
        raise RuntimeError(
            f"model load deferred: {model_id} weights "
            f"{weight_bytes / GB:.1f} GB exceed the measured free "
            f"working set {head / GB:.1f} GB (resident models are "
            f"pinned or busy). Retry when a slot frees, or "
            f"GMLX_OVERCOMMIT=1 overrides.")


def install_boot_table(gguf_path: str, weight_bytes: float | None,
                       model_id: str) -> dict | None:
    """Derive, log, install, and gate the capacity table after a build.
    Raises with the numbers when the configuration cannot fit at width
    1 (unless GMLX_OVERCOMMIT=1, which logs that it is armed)."""
    global _TABLE
    t = derive_table(gguf_path, weight_bytes)
    if t is None:
        _log.info("[capacity] no table for %s (header or device "
                  "unreadable); stock behavior kept", model_id)
        return None
    _TABLE = t
    _log_table(t)
    if t["overcommit"]:
        _log.warning("[capacity] GMLX_OVERCOMMIT=1 armed: boot refusal "
                     "and derived ceilings disabled")
        return t
    if t["max_ctx"].get(1, 0) <= 0:
        raise RuntimeError(
            f"model cannot fit at width 1: {model_id} weights "
            f"{t['weight_bytes'] / GB:.1f} GB + reserve "
            f"{t['reserve_bytes'] / GB:.1f} GB leave no context inside "
            f"the {t['budget_bytes'] / GB:.1f} GB working budget "
            f"(working set {t['working_set_bytes'] / GB:.1f} GB, margin "
            f"{t['margin']:.2f}). GMLX_OVERCOMMIT=1 overrides; for MoE "
            f"models --stream-experts serves the experts from disk.")
    return t


def clear_table() -> None:
    global _TABLE
    _TABLE = None


# GGUF path -> (mtime, trained context) for /v1/models; a header scan per
# configured model per listing would otherwise be 50 mmaps a call.
_CTX_CACHE: dict = {}


def trained_context_length(gguf_path) -> int | None:
    """The GGUF's trained context (``<arch>.context_length``), or None
    when the header cannot be read. Cached by path + mtime."""
    if not gguf_path:
        return None
    try:
        mtime = os.path.getmtime(gguf_path)
    except OSError:
        return None
    hit = _CTX_CACHE.get(gguf_path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    value = None
    try:
        from .headerscan import scan_gguf

        kv = scan_gguf(gguf_path, include_tensors=False).kv
        arch = kv.get("general.architecture")
        raw = kv.get(f"{arch}.context_length") if arch else None
        if raw is None:
            raw = next((v for k, v in kv.items()
                        if k.endswith(".context_length")), None)
        if isinstance(raw, (int, float)) and int(raw) > 0:
            value = int(raw)
    except Exception:
        _log.debug("context_length scan failed for %s", gguf_path, exc_info=True)
    _CTX_CACHE[gguf_path] = (mtime, value)
    return value


def max_context_at_width_1(gguf_path) -> int | None:
    """What the installed capacity table says fits at width 1 - only for
    the model it was derived from (the boot model); None otherwise, or
    with overcommit armed."""
    t = _TABLE
    if t is None or overcommit() or not gguf_path:
        return None
    if str(t.get("path")) != str(gguf_path):
        return None
    v = t["max_ctx"].get(1)
    return int(v) if isinstance(v, int) and v > 0 else None
