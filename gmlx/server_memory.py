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


def apply_cache_limit(cfg) -> None:
    """Resolve and apply the server's cache limit; called once at startup."""
    import mlx.core as mx

    try:
        ws = float(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        ws = 0.0
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
