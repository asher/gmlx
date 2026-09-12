"""One memory budget for a streaming install.

The decode arena, the prefill ring beside it and the KV cache are all
MLX-tracked, and the serve governor enforces one ceiling over them
(``gmlx.serve.capacity.ceiling_bytes``). The arena is sized as what that
ceiling leaves after the every-token weights, a priced KV room, the ring's
own room and the host floor, so at decode the governor's headroom is the
room minus live KV on any box and any quant, not the gap between two
unrelated estimates.

The room prices ``GMLX_STREAM_KV_CTX`` tokens (default 32768, clamped to
the trained context) at ``GMLX_STREAM_KV_WIDTH`` rows (default 1) with
the model's own per-token KV cost, plus the prefill score transient the
decay policy allows and the admission reserve. ``GMLX_DECODE_KV_RESERVE_GB``
replaces the priced room with a flat value.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import mlx.core as mx

from gmlx.envflags import env_bool, env_int

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


def reclaimable_ram_bytes() -> int | None:
    """RAM the kernel can hand back without swapping anyone: free,
    purgeable, speculative and file-backed pages. The governor's floor
    measure; the loader's vm_stat sum when the mach counters are
    unavailable."""
    try:
        from gmlx.serve.kernel_vm import reclaimable_bytes

        v = reclaimable_bytes()
    except Exception:
        v = None
    if v is not None:
        return int(v)
    return _available_ram_bytes()


def kernel_floor_bytes() -> float:
    """The governor's kernel reclaimable floor."""
    from gmlx.serve.governor import _kernel_floor_bytes

    return float(_kernel_floor_bytes())


def host_floor_bytes(ram: int | None) -> int:
    """Breathing margin left to the system whenever the arena takes RAM:
    under the ceiling at sizing, against reclaimable RAM at sizing and on
    every pressure-driven regrow (``GMLX_DECODE_RAM_FLOOR_GB`` overrides).
    On top of the base margin, ``GMLX_DECODE_PAGECACHE_GB`` reserves room
    for the page cache specifically: the prefill feeder and the CPU-mmap
    fallback read through it, and starving it collapses buffered pread
    throughput far below the SSD's sequential rate. The margin is what
    keeps the rest of the box out of swap while the arena is wired; a
    swap storm under a wired arena is a watchdog panic."""
    from gmlx.envflags import env_float

    gb = float(
        os.environ.get("GMLX_DECODE_RAM_FLOOR_GB", "")
        or max(4.0, 0.05 * (ram or 0) / (1 << 30))
    )
    gb += env_float("GMLX_DECODE_PAGECACHE_GB", 2.5)
    return int(gb * (1 << 30))


def governor_headroom_bytes() -> float | None:
    """What the governor reads at its tick: ``prefill_decay.headroom_bytes``
    shifted from the full working set down to the governor ceiling."""
    from gmlx.serve.capacity import margin
    from gmlx.serve.governor import _headroom_and_ws

    return _headroom_and_ws(margin())[0]


def legacy_room_bytes() -> int:
    raw = os.environ.get("GMLX_DECODE_KV_RESERVE_GB", "")
    try:
        return int(float(raw or _LEGACY_RESERVE_GB) * (1 << 30))
    except (ValueError, OverflowError):
        return int(_LEGACY_RESERVE_GB * (1 << 30))


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


def _available_ram_bytes(include_inactive: bool = True) -> int | None:
    """RAM this process can take without swapping anyone's anonymous memory:
    free + purgeable + the file-backed page cache (macOS ``vm_stat``). A
    load-time snapshot of the machine's offer - a machine already
    half-occupied by other workloads offers the arena half a machine,
    whatever the hardware total says. File-backed pages drop without IO
    whatever queue they sit on; counting only the *inactive* queue (the old
    formula) missed the tens of GB of recently-read GGUF cache still on the
    active queue and made a mostly-cache machine look nearly full. A
    ``vm_stat`` without the ``File-backed pages`` line falls back to
    inactive + speculative.

    ``include_inactive=False`` is the stricter set (free + purgeable +
    speculative only) for a caller that must not take the page cache."""
    from gmlx.serve import kernel_vm

    s = kernel_vm.snapshot()
    if s is not None:
        return s["free"] + s["purgeable"] + (
            s["filebacked"] if include_inactive else s["speculative"])
    import subprocess

    try:
        # posix_spawn (absolute path, close_fds=False): a fork beside a
        # Metal-mapped buffer copies the buffer first.
        out = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=5,
            close_fds=False,
        ).stdout
    except Exception:
        return None
    m = re.search(r"page size of (\d+)", out)
    if not m:
        return None
    keys = ["free", "purgeable"]
    pages = 0
    found = False
    if include_inactive:
        mm = re.search(r"File-backed pages:\s+(\d+)\.", out)
        if mm:
            pages += int(mm.group(1))
            found = True
        else:
            keys += ["speculative", "inactive"]
    else:
        keys.append("speculative")
    for key in keys:
        mm = re.search(rf"Pages {key}:\s+(\d+)\.", out)
        if mm:
            pages += int(mm.group(1))
            found = True
    return pages * int(m.group(1)) if found else None


def _ram_floor_bytes(ram: int | None) -> int:
    """``gmlx.stream.budget.host_floor_bytes``."""
    return host_floor_bytes(ram)


def _decode_arena_bytes(
    total_bytes: int, offsets, budget: int | None, room_bytes: int | None = None,
    pinned_bytes: int = 0, streamable_bytes: int = 0,
    cast_dead_bytes: int = 0, ring_bytes: int = 0,
) -> int:
    """Arena budget for the decode feeder: what the memory ceiling leaves
    after the non-expert weights, the KV room, the prefill ring and the
    host floor, clamped to the RAM reclaimable right now, and capped at
    the expert bytes themselves (a model whose experts fit goes fully
    resident).

    The ceiling is the serve governor's (``gmlx.stream.budget``), so the
    arena, the prefill ring and the KV cache share one budget: at decode
    the governor's headroom is the room minus live KV, whatever the box's
    working-set ratio or the quant's ring size. ``ring_bytes`` keeps the
    ring's room out of the arena for good: a ring rebuilt on top of a
    full wired arena, or lent out of it with a copy of every layer, is a
    transient the kernel has to swap for on a box the arena has already
    filled. ``room_bytes`` is the priced KV room
    (``budget.kv_room_bytes``); None keeps the flat legacy reserve.
    ``GMLX_DECODE_ARENA_RAM_FRAC`` caps the ceiling at a fraction of
    physical RAM when set. ``GMLX_DECODE_ARENA_GB`` overrides the
    ceilings but is still clamped to what is reclaimable minus the floor
    - an arena wired past that starves the page cache every buffered
    read path depends on (``GMLX_DECODE_ARENA_FORCE=1`` restores the
    unclamped behavior).

    A second live streaming install needs no term here: mlock moves a page
    out of the file-backed count and an arena is anonymous, so the
    reclaimable snapshot already excludes both."""
    env = os.environ.get("GMLX_DECODE_ARENA_GB")
    if env:
        want = int(float(env) * (1 << 30))
        if env_bool("GMLX_DECODE_ARENA_FORCE", False):
            return want
        avail = _available_ram_bytes()
        if avail is None:
            return want
        try:
            ram = int(mx.device_info()["memory_size"])
        except Exception:
            ram = avail
        cap = max(0, avail - _ram_floor_bytes(ram))
        if want > cap:
            print(
                f"[stream] GMLX_DECODE_ARENA_GB={env} exceeds reclaimable"
                f" RAM minus the floor; clamping the arena to"
                f" {cap / (1 << 30):.1f}GB (GMLX_DECODE_ARENA_FORCE=1"
                f" overrides)"
            )
            return cap
        return want
    if budget is None:
        return 0
    ceiling = int(ceiling_bytes() or budget)
    ram = None
    try:
        ram = int(mx.device_info()["memory_size"])
    except Exception:
        pass
    frac = os.environ.get("GMLX_DECODE_ARENA_RAM_FRAC", "")
    if frac and ram:
        try:
            ceiling = min(ceiling, int(float(frac) * ram))
        except (ValueError, OverflowError):
            pass
    expert_bytes = sum(r[2] for ranges in offsets.values() for r in ranges)
    # Streamable components are page-cache citizens like the experts;
    # charging them as non-expert would zero the arena. Cast tensors cost
    # what their converted copy weighs, not what the wire does: the wire
    # range is unpinned and never read again (gmlx.stream.pin_weights
    # .cast_copies), so charging it would cancel the pin it just freed.
    non_expert_bytes = max(
        0, total_bytes - expert_bytes - streamable_bytes - cast_dead_bytes)
    room = int(room_bytes) if room_bytes is not None else legacy_room_bytes()
    # The floor on both measures: the ceiling is a share of the Metal
    # working set, and the OS side (the page cache, other processes) is
    # not in it.
    arena = (ceiling - non_expert_bytes - room - int(ring_bytes)
             - _ram_floor_bytes(ram))
    # Second ceiling: what is reclaimable right now. The governor ceiling
    # assumes an otherwise idle machine; co-resident workloads shrink the
    # offer, and a wired arena sized past it would evict them to swap. The
    # floor keeps a breathing margin for the system. This is a live
    # post-pin snapshot: already-wired weights are out of it, so only the
    # still-unwired share of the non-expert set is charged (charging all
    # of it double-counted the pin and zeroed the arena on exactly the
    # models that need it).
    avail = _available_ram_bytes()
    if avail is not None:
        unpinned = max(0, non_expert_bytes - pinned_bytes)
        arena = min(
            arena,
            avail - _ram_floor_bytes(ram or avail) - room - unpinned
            - int(ring_bytes),
        )
    return min(max(0, arena), expert_bytes)


def _prefill_ring_reason(offsets, left: int | None) -> str | None:
    """Why the prefill ring must not be built, or None. The ring (two
    slots of the largest layer's expert stacks, sized by the model) takes
    its room under the memory ceiling before the decode arena; ``left``
    is what the ceiling leaves after the every-token weights and the KV
    room. A ring larger than that would sit on top of them and take the
    ceiling with it at the first prefill. An explicit GMLX_DECODE_ARENA_GB
    is the user's budget and the ring is not judged against it."""
    if left is None or os.environ.get("GMLX_DECODE_ARENA_GB"):
        return None
    from gmlx.stream.prefill_feeder import ring_bytes

    ring = ring_bytes(offsets)
    if ring <= left:
        return None
    return (f"ring 2 x {ring / 2e9:.1f} GB exceeds the {left / 1e9:.1f} GB "
            "left under the memory ceiling after the every-token weights "
            "and the KV room")
