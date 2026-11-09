"""What a resident model costs: its priced footprint and its streams' rooms."""

from __future__ import annotations

import os
from typing import Callable

# The descent to the module the loader hung the streaming helpers on. Shared
# with the loader, which reads the same helper set to charge a live install's
# wired bytes against the next one.
from gmlx.stream.installs import streaming_owner as _streaming_owner

from .residency_types import _log


def _gguf_footprint_bytes(model_path: str) -> int:
    """On-disk size of a GGUF (summed over split shards) - a precise a-priori
    estimate of a model's resident weight bytes, since the loader mmaps the wire
    bytes zero-copy. Returns 0 if the path can't be sized (e.g. a non-GGUF
    mlx-vlm source), so such a model simply doesn't count against the budget."""
    try:
        from gmlx.load.preflight import find_split_shards

        return sum(os.path.getsize(s) for s in find_split_shards(model_path))
    except Exception:
        return 0


def _stamp_apc_mode(rg) -> None:
    """Set ``rg.apc_mode`` the way ``ResponseGenerator.__init__`` would have
    had the manager existed at construction. gmlx wires the manager in
    post-load, so without this the generator's mode stays None, the server
    skips the semantic-salt precompute, and the engine's fallback folds a
    content hash of the prompt's embedding matrix into every cache key: a
    per-request hashing cost, and a salt nothing outside the engine (the
    dry-run's warm-prefix probe) can reproduce."""
    if getattr(rg, "apc_mode", None) is not None:
        return
    model = getattr(rg, "model", None)
    if model is None:
        return
    mode = getattr(model, "_kq_apc_mode", None)
    if mode is None:
        try:
            from gmlx._vendor.mlx_vlm import apc as _apc
            lm = getattr(model, "language_model", None) or model
            mode = _apc.model_apc_mode(lm)
        except Exception:
            _log.warning("apc_mode probe skipped", exc_info=True)
            return
    if mode is not None:
        try:
            rg.apc_mode = mode
        except Exception:
            _log.warning("apc_mode stamp failed; the generator keeps its "
                         "default mode", exc_info=True)


def _stamp_boot_kv_costs(rg, gguf_path: str) -> None:
    """Hang the boot table's KV costs on the served model, so a fresh
    model's admission and governor demand price this model, not the
    table installed last."""
    from gmlx.stream.installs import wrapper_chain

    from .capacity import boot_kv_costs

    model = getattr(rg, "model", None)
    if model is None:
        return
    costs = boot_kv_costs(gguf_path)
    # Every module of the wrapper descent: the batch generator reads the
    # stamp off whichever of them it was built on (the text tower).
    for target in wrapper_chain(model):
        try:
            object.__setattr__(target, "_kq_boot_kv_costs", costs)
        except Exception:                                  # noqa: BLE001, S110 - stamp only; an unsettable module reads as unstamped
            pass


def _env_arena_cap(model, box, env, held_bytes: int) -> int | None:
    """The clamp the loader puts on an explicit ``GMLX_DECODE_ARENA_GB``
    (``budget._decode_arena_bytes``): the memory not wired, counting what
    the resident models hold as free to evict, less the incoming every-token
    weights (pinned before the arena is sized), the host floor, the ring,
    the KV room and the host overhead. None under
    ``GMLX_DECODE_ARENA_FORCE=1`` or without a priced box."""
    force = ((env or {}).get("GMLX_DECODE_ARENA_FORCE")
             or os.environ.get("GMLX_DECODE_ARENA_FORCE", ""))
    room = getattr(box, "room", None)
    if (str(force).strip().lower() in ("1", "true", "yes", "on")
            or room is None or not getattr(box, "ram_bytes", 0)):
        return None
    from gmlx.stream.budget import host_floor_bytes, unwired_ram_bytes

    avail = unwired_ram_bytes()
    if avail is None:
        return None
    ring = int(model.ring_bytes) if box.ring_fits else 0
    return max(0, int(avail) + int(held_bytes)
               - int(model.every_token_bytes)
               - host_floor_bytes(int(box.ram_bytes)) - ring
               - int(room.bytes) - int(getattr(box, "host_bytes", 0)))


def _drafter_bytes(model_path: str, build_spec) -> int:
    """Bytes of the MTP drafter a serve build loads beside the model: its
    ``draft_gguf`` file, else the model's own NextN blocks. 0 without
    speculation."""
    if not getattr(build_spec, "speculative", False):
        return 0
    draft = getattr(build_spec, "draft_gguf", None) or ""
    if draft:
        return os.path.getsize(draft) if os.path.isfile(draft) else 0
    from gmlx.load.discovery import mtp_block_bytes

    return mtp_block_bytes(model_path)


def text_adapter_bytes(build_spec) -> int:
    """The resident bytes of the text adapters a chat entry loads, each
    file of the entry's adapter union priced by ``adapter_resident_bytes``.
    A missing file prices at 0; its load fails."""
    from gmlx.load.adapter import adapter_resident_bytes

    effective: Callable[[], tuple[str, ...]] | None = getattr(
        build_spec, "effective_adapters", None)
    if not callable(effective):
        return 0
    return sum(adapter_resident_bytes(os.path.getsize(p))
               for p in effective() if os.path.isfile(p))


def _plain_footprint(model_path, file_bytes: int, env=None) -> int:
    """Resident bytes of a model loaded without a stream placement: its
    file, less the lookup tables the load reads from the file instead of
    mapping (``capacity.deferred_table_bytes``). A model over the budget
    whose tables alone bring it under loads that way, with its experts
    resident."""
    from .capacity import deferred_table_bytes

    if not isinstance(model_path, str) or not model_path.endswith(".gguf"):
        return int(file_bytes)
    return max(0, int(file_bytes) - deferred_table_bytes(model_path, env=env))


def _streaming_footprint(model_path, file_bytes: int, env=None,
                         held_bytes: int = 0) -> int:
    """Resident bytes a ``stream: experts`` entry will hold: the every-token
    weights, the decode arena the loader sizes at install (the env override
    when set, clamped as the loader clamps it with ``held_bytes`` of
    residents counted as free to evict, else the fit planner's arena) and
    the prefill ring's room the arena does not hold and lend at
    prefill. The routed experts and
    the arch's streamable tables stay on disk, so the file size overstates
    the entry by their bytes and would evict every other model for room
    they never take. The build streams what
    :func:`capacity.predicted_stream_bytes` predicts, and so the entry is
    priced on it."""
    from .capacity import predicted_stream_bytes, preload_gate_bytes

    experts_stream = True
    try:
        shares = predicted_stream_bytes(str(model_path), env)
        every = preload_gate_bytes(file_bytes, "experts", sum(shares))
        experts_stream = bool(shares[1])
    except Exception:
        every = int(file_bytes)
    if not experts_stream:
        return every
    raw = ((env or {}).get("GMLX_DECODE_ARENA_GB")
           or os.environ.get("GMLX_DECODE_ARENA_GB"))
    arena = None
    if raw:
        try:
            arena = int(float(raw) * (1 << 30))
        except (ValueError, OverflowError):
            arena = None
    ring = 0
    try:
        from gmlx.stream.plan import plan_path

        plan = plan_path(str(model_path))
        if plan is None:
            _log.warning("streaming footprint: no plan for %s; arena and "
                         "ring priced at zero", model_path)
        else:
            model, box = plan
            if box is not None:
                if arena is None:
                    arena = int(box.arena_bytes)
                else:
                    cap = _env_arena_cap(model, box, env, held_bytes)
                    if cap is not None:
                        arena = min(arena, cap)
                if box.ring_fits:
                    # An env-sized arena keeps the whole ring apart.
                    shared = 0 if raw else int(
                        getattr(box, "ring_shared_bytes", 0) or 0)
                    ring = int(model.ring_bytes) - shared
    except Exception:
        _log.warning("streaming footprint: plan for %s failed; arena and "
                     "ring priced at zero", model_path, exc_info=True)
    return every + max(0, arena or 0) + ring


def _prefill_feeder_of(rg):
    """The prefill ring behind a response generator's model, or None."""
    model = getattr(rg, "model", None)
    if model is None:
        return None
    return getattr(_streaming_owner(model), "_kq_feeder", None)


def _streamed_shares(model, predicted, gguf_path) -> tuple[int, int] | None:
    """``(table bytes, expert bytes)`` the install really streamed, priced
    at the header figures; None when no install verdict is found."""
    from gmlx.stream.installs import wrapper_chain
    from gmlx.stream.table_stream import table_streaming_active

    from .capacity import streamed_expert_bytes, streamed_table_bytes

    for cur in wrapper_chain(model):
        if hasattr(cur, "_kq_streaming"):
            table = (predicted[0] or streamed_table_bytes(gguf_path)
                     if table_streaming_active(cur) else 0)
            experts = (predicted[1] or streamed_expert_bytes(gguf_path)
                       if cur._kq_streaming else 0)
            return table, experts
    return None


def _prefill_ring_of(rg) -> int:
    """Bytes of the prefill ring's room behind a response generator's
    model: both slots, kept for the process lifetime, less the share the
    decode arena holds and lends at each prefill. 0 without a ring."""
    feeder = _prefill_feeder_of(rg)
    share = int(getattr(_decode_feeder_of(rg), "_ring_share", 0) or 0)
    return max(0, _ring_room(feeder) - share)


def _ring_room(feeder) -> int:
    """Bytes of all of a prefill ring's slots; 0 for None."""
    return (int(getattr(feeder, "n_slots", 2) or 0)
            * int(getattr(feeder, "slot_bytes", 0) or 0))


def _decode_feeder_of(rg):
    """The decode feeder behind a response generator's model, or None."""
    model = getattr(rg, "model", None)
    if model is None:
        return None
    return getattr(_streaming_owner(model), "_kq_decode_feeder", None)
