"""One memory budget for a streaming install.

The decode arena, the prefill ring it lends to and the KV cache are all
MLX-tracked, and the serve governor enforces one ceiling over them
(``gmlx.serve.capacity.ceiling_bytes``). The arena is sized as what that
ceiling leaves after the every-token weights and a priced KV room, so at
decode the governor's headroom is the room minus live KV on any box and
any quant, not the gap between two unrelated estimates.

The room prices ``GMLX_STREAM_KV_CTX`` tokens (default 32768, clamped to
the trained context) at ``GMLX_STREAM_KV_WIDTH`` rows (default 1) with
the model's own per-token KV cost, plus the prefill score transient the
decay policy allows and the admission reserve. ``GMLX_DECODE_KV_RESERVE_GB``
replaces the priced room with a flat value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from gmlx.envflags import env_int

_DEFAULT_CTX = 32768
_LEGACY_RESERVE_GB = 8.0


@dataclass(frozen=True)
class KvRoom:
    bytes: int
    depth: int
    width: int
    kv_bytes: int
    transient_bytes: int
    reserve_bytes: int
    priced: bool


def ceiling_bytes() -> float | None:
    """The serve governor's ceiling on tracked bytes, or None when the
    device cannot be read."""
    from gmlx.serve.capacity import ceiling_bytes as _ceiling, working_set_bytes

    ws = working_set_bytes()
    return None if ws is None else _ceiling(ws)


def governor_headroom_bytes() -> float | None:
    """What the governor reads at its tick: ``prefill_decay.headroom_bytes``
    shifted from the full working set down to the governor ceiling."""
    from gmlx.serve.capacity import margin
    from gmlx.serve.governor import _headroom_and_ws

    return _headroom_and_ws(margin())[0]


def legacy_room_bytes() -> int:
    raw = os.environ.get("GMLX_DECODE_KV_RESERVE_GB", "")
    try:
        gb = float(raw or _LEGACY_RESERVE_GB)
    except ValueError:
        gb = _LEGACY_RESERVE_GB
    return int(gb * (1 << 30))


def transient_bytes(ws: float) -> float:
    """The prefill score transient the decay policy allows on a box with
    working set ``ws``."""
    from gmlx.gen.prefill_decay import cap_bytes_for

    return cap_bytes_for(ws)


def price_room(costs, trained_ctx, ws: float, transient: float) -> KvRoom:
    """The KV room from priced per-layer costs. A flat legacy reserve
    when ``costs`` is None or ``GMLX_DECODE_KV_RESERVE_GB`` is set."""
    from gmlx.serve.mem_preflight import prompt_kv_bytes
    from gmlx.serve.memory import admit_reserve_bytes

    depth = max(1, env_int("GMLX_STREAM_KV_CTX", _DEFAULT_CTX))
    width = max(1, env_int("GMLX_STREAM_KV_WIDTH", 1))
    flat = KvRoom(legacy_room_bytes(), depth, width, 0, 0, 0, priced=False)
    if os.environ.get("GMLX_DECODE_KV_RESERVE_GB", "") or not costs or not ws:
        return flat
    if isinstance(trained_ctx, int) and trained_ctx > 0:
        depth = min(depth, trained_ctx)
    kv = int(prompt_kv_bytes(costs, depth) * width)
    reserve = int(admit_reserve_bytes(ws))
    return KvRoom(kv + int(transient) + reserve, depth, width, kv,
                  int(transient), reserve, priced=True)


def kv_room_bytes(gguf_path: str | None, env: dict | None = None) -> KvRoom:
    """The KV room to leave outside the arena. A flat legacy reserve when
    the header cannot be priced or ``GMLX_DECODE_KV_RESERVE_GB`` is set."""
    if os.environ.get("GMLX_DECODE_KV_RESERVE_GB", "") or not gguf_path:
        return price_room(None, None, 0, 0)
    try:
        from gmlx.gen.prefill_decay import _cap_bytes
        from gmlx.serve.capacity import boot_costs, working_set_bytes

        priced = boot_costs(gguf_path, env)
        ws = working_set_bytes()
        if priced is None or not ws:
            return price_room(None, None, 0, 0)
        cfg, costs, _ = priced
        return price_room(costs, cfg.get("max_position_embeddings"), ws,
                          _cap_bytes())
    except Exception:
        return price_room(None, None, 0, 0)
