"""One memory budget for a streaming install.

The decode arena, the prefill ring beside it and the KV cache are all
MLX-tracked, and the serve governor enforces one ceiling over them
(``gmlx.serve.capacity.ceiling_bytes``). The arena is sized as what that
ceiling leaves after the every-token weights, a priced KV room, the ring's
own room, the host floor and the host overhead, so at decode the
governor's headroom is the room minus live KV on any box and any quant,
not the gap between two unrelated estimates.

The room prices ``GMLX_STREAM_KV_CTX`` tokens (default 32768, clamped to
the trained context) at ``GMLX_STREAM_KV_WIDTH`` rows (default 1) with
the model's own per-token KV cost, plus the prefill score transient the
decay policy allows, the live activations of one streamed prefill chunk,
the MLX buffer pool that recycles them, and the admission reserve for a
batch at the decode batch cap.
``GMLX_DECODE_KV_RESERVE_GB`` replaces the priced room with a flat value.

The host overhead is RAM the arena leaves outside MLX: this process's
own memory and the file pages that stay resident under load.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace

from gmlx.device import metal_value
from gmlx.envflags import env_bool, env_int

_DEFAULT_CTX = 32768
_LEGACY_RESERVE_GB = 8.0
_POOL_BYTES = 2 << 30
_UNPRICED_POOL_BYTES = 4 << 30
_PROCESS_GB = 2.0
_RESIDENT_CACHE_RAM_SHARE = 1 / 16
_RESIDENT_CACHE_MAX_GB = 8.0


@dataclass(frozen=True)
class KvRoom:
    bytes: int
    depth: int
    width: int
    kv_bytes: int
    transient_bytes: int
    reserve_bytes: int
    priced: bool
    # One streamed prefill chunk's live activations, and the MLX buffer
    # pool (the allocator's cache limit) that keeps freed buffers for the
    # next layer. The pool fills to its limit and the live set sits on
    # top of it.
    activation_bytes: int = 0
    pool_bytes: int = 0

    @property
    def fit_bytes(self) -> int:
        """The room less the prefill chunk and its pool: what a model
        needs under the ceiling to stream at all. The chunk comes out of
        the arena, and with no arena it takes RAM the page cache gives
        back. A flat room is all fit."""
        if not self.priced:
            return self.bytes
        return self.bytes - self.activation_bytes - self.pool_bytes


def ceiling_bytes() -> float | None:
    """The serve governor's ceiling on tracked bytes, or None when the
    device cannot be read."""
    from gmlx.serve.capacity import ceiling_bytes as _ceiling, working_set_bytes

    ws = working_set_bytes()
    return None if ws is None else _ceiling(ws)


def reclaimable_ram_bytes() -> int | None:
    """RAM the kernel can hand back without swapping anyone: free,
    purgeable and file-backed pages, the speculative pages counted once.
    The governor's floor measure; the loader's read when the mach
    counters are unavailable."""
    try:
        from gmlx.serve.kernel_vm import reclaimable_bytes

        v = reclaimable_bytes()
    except Exception:
        v = None
    if v is not None:
        return int(v)
    return _available_ram_bytes()


def live_reclaimable_ram_bytes() -> int | None:
    """``reclaimable_ram_bytes`` read at the moment of the call
    (``kernel_vm.live_reclaimable_bytes``), for the resident wire's check
    and its chunk guard: the counters under ``reclaimable_ram_bytes``
    can be a second old, longer than a wire takes. That reader when the
    live one fails."""
    try:
        from gmlx.serve.kernel_vm import live_reclaimable_bytes

        v = live_reclaimable_bytes()
    except Exception:
        v = None
    if v is not None:
        return int(v)
    return reclaimable_ram_bytes()


def kernel_floor_bytes() -> float:
    """The governor's kernel reclaimable floor."""
    from gmlx.serve.governor import _kernel_floor_bytes

    return float(_kernel_floor_bytes())


def host_floor_bytes(ram: int | None) -> int:
    """Breathing margin left to the system whenever the arena takes RAM:
    under the ceiling at sizing, against reclaimable RAM at sizing and on
    every pressure-driven regrow (``GMLX_DECODE_RAM_FLOOR_GB`` overrides).
    On top of the base margin, ``GMLX_DECODE_PAGECACHE_GB`` reserves room
    for the page cache specifically: the CPU-mmap fallback and the
    buffered prefill reads (a ring out of service, or
    ``GMLX_PREFILL_NOCACHE=0``) go through it, and starving it collapses
    buffered pread throughput far below the SSD's sequential rate. The margin is what
    keeps the rest of the box out of swap while the arena is wired; a
    swap storm under a wired arena is a watchdog panic."""
    from gmlx.envflags import env_float

    gb = float(
        os.environ.get("GMLX_DECODE_RAM_FLOOR_GB", "")
        or max(4.0, 0.05 * (ram or 0) / (1 << 30))
    )
    gb += env_float("GMLX_DECODE_PAGECACHE_GB", 2.5)
    return int(gb * (1 << 30))


def host_overhead_bytes(ram: int | None) -> int:
    """RAM the arena leaves outside MLX beyond the host floor: this
    process's own memory (``GMLX_DECODE_PROCESS_GB``, default 2: the
    Python heap, the tokenizer and the server) and the file pages that
    stay resident under load (the table rows, the libraries), which the
    reclaimable snapshot counts as free to take.
    ``GMLX_DECODE_RESIDENT_CACHE_GB`` prices those pages, default 1/16 of
    RAM and at most 8. A fixed price, not the page cache seen at sizing,
    so a cold and a warm load size the same arena."""
    from gmlx.envflags import env_float

    process = env_float("GMLX_DECODE_PROCESS_GB", _PROCESS_GB)
    cap_gb = env_float("GMLX_DECODE_RESIDENT_CACHE_GB", min(
        _RESIDENT_CACHE_MAX_GB,
        _RESIDENT_CACHE_RAM_SHARE * (ram or 0) / (1 << 30)))
    return int((max(0.0, process) + max(0.0, cap_gb)) * (1 << 30))


def arena_ceiling_bytes(budget: float, ram: int | None) -> int:
    """The ceiling the decode arena is sized under: the memory ceiling, or
    ``budget`` without one, capped at ``GMLX_DECODE_ARENA_RAM_FRAC`` of
    ``ram`` when that is set."""
    ceiling = int(ceiling_bytes() or budget)
    frac = os.environ.get("GMLX_DECODE_ARENA_RAM_FRAC", "")
    if frac and ram:
        try:
            ceiling = min(ceiling, int(float(frac) * ram))
        except (ValueError, OverflowError):
            pass
    return ceiling


def outside_ceiling_bytes(ram: int | None, ceiling: float,
                          host: int) -> int:
    """What the host floor and the host overhead need past the RAM the
    memory ceiling leaves outside it. Both live outside MLX, so only the
    excess comes out of an arena sized under the ceiling."""
    if not ram:
        return 0
    return max(0, _ram_floor_bytes(ram) + int(host) - int(ram - ceiling))


def governor_headroom_bytes() -> float | None:
    """What the governor reads at its tick: ``prefill_decay.headroom_bytes``
    shifted from the full working set down to the governor ceiling."""
    from gmlx.serve.capacity import margin
    from gmlx.serve.governor import _headroom_and_ws

    return _headroom_and_ws(margin())[0]


def ring_share_bytes(full_bytes: int, held_bytes: int, ring_bytes: int) -> int:
    """Bytes of the prefill ring's room the decode arena holds while
    decoding and lends back for each prefill pass, instead of leaving the
    room idle under the ceiling. ``full_bytes`` is the arena sized with
    the room, ``held_bytes`` the arena with the room set aside; the share
    is what the room adds. Lends free whole layers, so the arena must be
    at least twice the ring. ``GMLX_RING_TIMESHARE=0`` keeps the room
    apart for the process lifetime."""
    if (not env_bool("GMLX_RING_TIMESHARE", True) or ring_bytes <= 0
            or full_bytes < 2 * ring_bytes):
        return 0
    return max(0, min(int(ring_bytes), int(full_bytes) - int(held_bytes)))


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


# The arch prefill steps a streamed prefill takes over the loader's
# default (the models' _*_base_step); 0 or unparseable keeps the default.
_ARCH_STEP_ENV = {"deepseek_v41": "GMLX_DS41_PREFILL_STEP",
                  "deepseek_v4": "GMLX_V4_PREFILL_STEP",
                  "qwen4_exp": "GMLX_Q4_PREFILL_STEP"}


def prefill_rows(model_type: str | None, step: int | None = None) -> int:
    """Rows of the widest streamed prefill chunk: the requested ``step``
    (``--prefill-step-size``) or ``PREFILL_STEP_SIZE`` when set, else
    the arch's own step env, else the loader's streaming step for this
    model type, and 9/8 of it while the short tail merges into the chunk
    before (``expert_streaming.merge_prefill_tail``)."""
    if not (isinstance(step, int) and step > 0):
        step = env_int("PREFILL_STEP_SIZE", 0)
    if step <= 0 and model_type in _ARCH_STEP_ENV:
        step = env_int(_ARCH_STEP_ENV[model_type], 0)
    if step <= 0:
        from gmlx.load.loader import (_STREAMING_PREFILL_STEP,
                                      _STREAMING_PREFILL_STEP_BY_MODEL_TYPE)

        step = _STREAMING_PREFILL_STEP_BY_MODEL_TYPE.get(
            model_type or "", _STREAMING_PREFILL_STEP)
    if env_bool("GMLX_STREAM_PREFILL_TAIL_MERGE", True):
        step += step // 8
    return step


def _width(c, *names) -> int:
    """The widest positive int under any of ``names``: a per-layer list
    gives its widest layer."""
    from gmlx.serve.mem_preflight import _get

    best = 0
    for name in names:
        v = _get(c, name)
        for x in (v if isinstance(v, (list, tuple)) else (v,)):
            if isinstance(x, int) and not isinstance(x, bool) and x > best:
                best = x
    return best


# The synth's names for the same widths across archs.
_TOP_K = ("num_experts_per_tok", "moe_k", "moe_topk", "top_k_experts")
_EXPERT_FF = ("moe_intermediate_size", "expert_hidden_dim")
_STREAMS = ("hc_mult", "hc_count")
_HEAD_DIM = ("head_dim", "v_head_dim", "global_head_dim")


def prefill_activation_bytes(cfg, step: int | None = None) -> int:
    """Live activations of one streamed prefill chunk: one layer's
    working set at fp32 for ``prefill_rows`` rows (``step`` the requested
    chunk width). Per row: four
    hidden-wide copies (residual, norm, attention and mixer outputs) on
    each residual stream (``hc_mult``), the attention output (heads x
    head_dim), and per routed expert the row takes its input and output
    (2 x the expert's input width: ``routed_expert_hidden_size`` behind a
    latent projection, else hidden) and its gate, up and GLU product
    (3 x the expert width). Per-layer widths price the widest layer. 0
    when the config lacks the widths."""
    from gmlx.serve.mem_preflight import _get

    if not cfg:
        return 0
    text = _get(cfg, "text_config")
    c = text if text is not None else cfg
    hidden = _width(c, "hidden_size")
    heads = _width(c, "num_attention_heads")
    if not (hidden and heads):
        return 0
    head_dim = _width(c, *_HEAD_DIM) or hidden // heads
    k = _width(c, *_TOP_K)
    ff = _width(c, *_EXPERT_FF) or _width(c, "intermediate_size")
    expert_in = _width(c, "routed_expert_hidden_size") or hidden
    streams = max(1, _width(c, *_STREAMS))
    per_row = 4 * (4 * hidden * streams + heads * head_dim
                   + k * (2 * expert_in + 3 * ff))
    mt = _get(c, "model_type") or _get(cfg, "model_type")
    return per_row * prefill_rows(mt if isinstance(mt, str) else None, step)


def pool_override() -> int | None:
    """``GMLX_STREAM_CACHE_GB`` in bytes, a negative value as 0 and a value
    past physical RAM as physical RAM, so the allocator takes it. None
    when it is unset or not a number."""
    raw = os.environ.get("GMLX_STREAM_CACHE_GB", "")
    if not raw:
        return None
    try:
        n = max(0, int(float(raw) * (1 << 30)))
    except (ValueError, OverflowError):
        return None
    ram = ram_bytes()
    return n if ram is None else min(n, ram)


def pool_bytes(activation: int, fallback: int = _UNPRICED_POOL_BYTES) -> int:
    """The MLX buffer pool a streamed install sets: 2 GiB, or
    ``fallback`` when the activations are not priced. The pool holds the
    buffers one layer frees for the next. On DeepSeek V4.1 Flash a pool
    of one chunk's activations (8.0 GB) prefilled a 32k prompt about 3%
    faster than 2 GiB and an 8k prompt no faster, and the 6 GB it took
    from the decode arena cost about 5.5% of the decode speed.
    ``GMLX_STREAM_CACHE_GB`` sizes it."""
    override = pool_override()
    if override is not None:
        return override
    if activation <= 0:
        return int(fallback)
    return _POOL_BYTES


def price_room(costs, trained_ctx, ws: float, transient: float,
               activation: int = 0, depth: int | None = None) -> KvRoom:
    """The KV room from priced per-layer costs, the prefill transients
    (the score cap, one chunk's ``activation`` bytes and the buffer pool
    that keeps them) and the admission reserve, priced from the same
    costs (``memory.priced_admit_reserve_bytes``). A flat legacy reserve
    when ``costs`` is None or ``GMLX_DECODE_KV_RESERVE_GB`` is set; the
    pool then fills the flat room. ``depth`` prices that many tokens in
    place of ``GMLX_STREAM_KV_CTX``."""
    from gmlx.serve.decode_batch import decode_batch
    from gmlx.serve.mem_preflight import prompt_kv_bytes
    from gmlx.serve.memory import priced_admit_reserve_bytes

    depth = max(1, int(depth) if depth else env_int(
        "GMLX_STREAM_KV_CTX", _DEFAULT_CTX))
    width = max(1, env_int("GMLX_STREAM_KV_WIDTH", 1))
    legacy = legacy_room_bytes()
    flat = KvRoom(legacy, depth, width, 0, 0, 0, priced=False,
                  pool_bytes=pool_bytes(0, fallback=legacy))
    if os.environ.get("GMLX_DECODE_KV_RESERVE_GB", "") or not costs or not ws:
        return flat
    if isinstance(trained_ctx, int) and trained_ctx > 0:
        depth = min(depth, trained_ctx)
    kv = int(prompt_kv_bytes(costs, depth) * width)
    # Serve decodes a batch up to the decode batch cap on a streamed model
    # too, so the reserve covers that many rows at the room's depth.
    reserve = int(priced_admit_reserve_bytes(
        costs, depth, max(width, decode_batch())))
    act = max(0, int(activation))
    pool = pool_bytes(act)
    return KvRoom(kv + int(transient) + act + pool + reserve, depth, width,
                  kv, int(transient), reserve, priced=True,
                  activation_bytes=act, pool_bytes=pool)


def kv_room_bytes(gguf_path: str | None, env: dict | None = None,
                  step: int | None = None,
                  depth: int | None = None) -> KvRoom:
    """The KV room to leave outside the arena, its prefill chunk at the
    requested ``step`` and its KV at ``depth`` tokens (None:
    ``GMLX_STREAM_KV_CTX``). A flat legacy reserve when the header cannot
    be priced or ``GMLX_DECODE_KV_RESERVE_GB`` is set."""
    if os.environ.get("GMLX_DECODE_KV_RESERVE_GB", "") or not gguf_path:
        return price_room(None, None, 0, 0, depth=depth)
    try:
        from gmlx.gen.prefill_decay import _cap_bytes
        from gmlx.serve.capacity import boot_costs, working_set_bytes

        priced = boot_costs(gguf_path, env)
        ws = working_set_bytes()
        if priced is None or not ws:
            return price_room(None, None, 0, 0, depth=depth)
        cfg, costs, _ = priced
        return price_room(costs, cfg.get("max_position_embeddings"), ws,
                          _cap_bytes(), prefill_activation_bytes(cfg, step),
                          depth=depth)
    except Exception:
        return price_room(None, None, 0, 0, depth=depth)


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

    The sum is read live (``kernel_vm.live_reclaimable_bytes``): the
    mach snapshot can be a second old, and a sizing read right after a
    wire of tens of GB would count the wired pages as free. The snapshot
    is the fallback, then ``vm_stat``. Mach's free and file-backed counts
    both hold the speculative pages, so they count once.

    ``include_inactive=False`` is the stricter set (free + purgeable;
    Mach's free holds the speculative pages) for a caller that must not
    take the page cache."""
    from gmlx.serve import kernel_vm

    if include_inactive:
        live = kernel_vm.live_reclaimable_bytes()
        if live is not None:
            return int(live)
    s = kernel_vm.snapshot()
    if s is not None:
        if include_inactive:
            return (s["free"] - s["speculative"] + s["purgeable"]
                    + s["filebacked"])
        return s["free"] + s["purgeable"]
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


# The share of the expert bytes a forced stream's arena may hold when the
# model fits under the budget. An arena of all of them would wire every
# expert, which is the resident load the flag asks to avoid.
FORCED_ARENA_SHARE = 0.5


def _decode_arena_bytes(
    total_bytes: int, offsets, budget: int | None, room_bytes: int | None = None,
    pinned_bytes: int = 0, streamable_bytes: int = 0,
    cast_dead_bytes: int = 0, ring_bytes: int = 0, forced: bool = False,
    avail_bytes: int | None = None, host_bytes: int = 0,
) -> int:
    """Arena budget for the decode feeder: what the memory ceiling leaves
    after the non-expert weights, the KV room and the prefill ring,
    clamped to the RAM reclaimable right now less the same, the host
    floor and the host overhead (``host_bytes``, ``host_overhead_bytes``),
    and capped at the expert bytes themselves. The floor and the host
    overhead are not MLX memory: the RAM the ceiling leaves outside it
    holds them, so the ceiling leg charges only what they need past that
    RAM (``outside_ceiling_bytes``, as the plan does), and the reclaimable
    leg, which measures RAM, charges them whole. ``forced`` is a stream the
    flag asked for on a model under the budget: its cap is
    ``FORCED_ARENA_SHARE`` of the expert bytes, so the experts stay
    streamed rather than all wired.

    The ceiling is the serve governor's (``gmlx.stream.budget``), so the
    arena, the prefill ring and the KV cache share one budget: at decode
    the governor's headroom is the room minus live KV, whatever the box's
    working-set ratio or the quant's ring size. ``ring_bytes`` keeps the
    ring's room out of the arena: a ring rebuilt on top of a full wired
    arena is a transient the kernel has to swap for on a box the arena
    has already filled. The loader sizes the arena before the ring is
    wired, with and without the ring's room, and ``ring_share_bytes``
    decides whether the arena holds the room and lends it at prefill; an
    explicit ``GMLX_DECODE_ARENA_GB`` keeps the whole ring apart and is
    sized once, against ``avail_bytes`` read before the ring was wired.
    ``room_bytes`` is the priced KV room
    (``budget.kv_room_bytes``); None keeps the flat legacy reserve.
    ``GMLX_DECODE_ARENA_RAM_FRAC`` caps the ceiling at a fraction of
    physical RAM when set. ``GMLX_DECODE_ARENA_GB`` overrides the
    ceilings but is still clamped to what is reclaimable minus the floor,
    the ring, the KV room and the host overhead: an arena wired past that
    leaves the next prefill's transients nowhere but swap and starves the
    page cache every buffered read path depends on
    (``GMLX_DECODE_ARENA_FORCE=1`` restores the unclamped behavior).

    A second live streaming install needs no term here: mlock moves a page
    out of the file-backed count and an arena is anonymous, so the
    reclaimable snapshot already excludes both."""
    env = os.environ.get("GMLX_DECODE_ARENA_GB")
    room = int(room_bytes) if room_bytes is not None else legacy_room_bytes()
    host = max(0, int(host_bytes))
    if env:
        want = int(float(env) * (1 << 30))
        if env_bool("GMLX_DECODE_ARENA_FORCE", False):
            return want
        avail = (_available_ram_bytes() if avail_bytes is None
                 else avail_bytes)
        if avail is None:
            return want
        try:
            ram = int(metal_value("memory_size"))
        except Exception:
            ram = avail
        cap = max(0, avail - _ram_floor_bytes(ram) - int(ring_bytes)
                  - room - host)
        if want > cap:
            print(
                f"[stream] GMLX_DECODE_ARENA_GB={env} exceeds reclaimable"
                f" RAM minus the floor, the KV room, the host overhead"
                f"{' and the prefill ring' if ring_bytes else ''};"
                f" clamping the arena to"
                f" {cap / (1 << 30):.1f}GB (GMLX_DECODE_ARENA_FORCE=1"
                f" overrides)"
            )
            return cap
        return want
    if budget is None:
        return 0
    ram = None
    try:
        ram = int(metal_value("memory_size"))
    except Exception:  # noqa: S110 - no device info (CPU-only build); the RAM fraction lever is skipped
        pass
    ceiling = arena_ceiling_bytes(budget, ram)
    expert_bytes = sum(r[2] for ranges in offsets.values() for r in ranges)
    # Streamable components are page-cache citizens like the experts;
    # charging them as non-expert would zero the arena. Cast tensors cost
    # what their converted copy weighs, not what the wire does: the wire
    # range is unpinned and never read again (gmlx.stream.pin_weights
    # .cast_copies), so charging it would cancel the pin it just freed.
    non_expert_bytes = max(
        0, total_bytes - expert_bytes - streamable_bytes - cast_dead_bytes)
    outside = (outside_ceiling_bytes(ram, ceiling, host) if ram
               else _ram_floor_bytes(None))
    arena = ceiling - non_expert_bytes - room - int(ring_bytes) - outside
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
        reclaimable = (avail - _ram_floor_bytes(ram or avail) - room
                       - unpinned - int(ring_bytes) - host)
        if reclaimable < arena:
            from gmlx.load import loadlog

            loadlog.info(
                f"[stream] arena clamped to reclaimable RAM: "
                f"{max(0, reclaimable) / 1e9:.1f} GB, not the ceiling's "
                f"{arena / 1e9:.1f} GB (reclaimable now {avail / 1e9:.1f} GB "
                "of RAM; a co-resident workload or a cold page cache is the "
                "usual cause)")
        arena = min(arena, reclaimable)
    cap = int(FORCED_ARENA_SHARE * expert_bytes) if forced else expert_bytes
    return min(max(0, arena), cap)


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
    from gmlx.stream.prefill_feeder import ring_bytes, ring_slots

    ring = ring_bytes(offsets)
    if ring <= left:
        return None
    n = ring_slots()
    return (f"ring {n} x {ring / n / 1e9:.1f} GB exceeds the "
            f"{left / 1e9:.1f} GB left under the memory ceiling after the "
            "every-token weights and the KV room")


def keep_bytes(ram: int | None, room_bytes: int, host_bytes: int) -> int:
    """RAM a resident wire leaves free beside it: the host floor, the
    host overhead and the KV room. What other live installs hold wired
    is out of reclaimable RAM already, so the keep does not count it."""
    return (_ram_floor_bytes(ram) + max(0, int(host_bytes))
            + max(0, int(room_bytes)))


# The most a resident wire check keeps beside the wire, as a share of
# RAM. The host floor and the host overhead have fixed parts (4 GiB,
# 2.5 GiB of page cache, 2 GiB for the process) sized for the arena on a
# large box. Uncapped, they keep 8.9 GiB of a 7 GiB box and 9.5 GiB of a
# 16 GiB one, so a MoE of half the RAM would always stream there.
_WIRE_KEEP_RAM_SHARE = 1 / 5


def wire_keep_bytes(ram: int | None, room_bytes: int,
                    host_bytes: int) -> tuple[int, bool]:
    """``(keep, capped)``: ``keep_bytes`` for a resident wire check, at
    most a fifth of RAM. ``capped`` says the cap applied."""
    keep = keep_bytes(ram, room_bytes, host_bytes)
    if ram and keep > ram * _WIRE_KEEP_RAM_SHARE:
        return int(ram * _WIRE_KEEP_RAM_SHARE), True
    return keep, False


# A resident wire no larger is judged on the budget only, whatever the
# chunk size, so a small MoE loads on a box with little reclaimable
# memory, such as a CI runner.
WIRE_EXEMPT_BYTES = 2 << 30


def wire_chunk_bytes() -> int:
    """GMLX_WIRE_CHUNK_GB: the bytes a resident expert wire commits
    between reclaimable-memory reads (default 2). The exemption from the
    reclaimable check stays ``WIRE_EXEMPT_BYTES`` whatever it says."""
    raw = os.environ.get("GMLX_WIRE_CHUNK_GB", "")
    try:
        gb = float(raw) if raw else 2.0
    except ValueError:
        gb = 2.0
    if gb <= 0:
        gb = 2.0
    return max(1, int(gb * (1 << 30)))


def overcommit() -> bool:
    """GMLX_OVERCOMMIT=1: a resident wire that fits in RAM skips the
    pre-load check, and each chunk keeps only the kernel floor beside
    it."""
    return os.environ.get("GMLX_OVERCOMMIT", "0") == "1"


# The KV depth the resident wire check prices, set by a caller that
# knows its context (``gmlx run``: prompt and max tokens); None prices
# the serve context.
_WIRE_DEPTH: int | None = None


def set_wire_room_depth(tokens: int | None) -> None:
    """Price the resident wire check's KV room at ``tokens`` (None: the
    serve context) for the loads that follow."""
    global _WIRE_DEPTH
    _WIRE_DEPTH = int(tokens) if tokens and tokens > 0 else None


def wire_room_depth() -> int | None:
    return _WIRE_DEPTH


def wire_room_label(depth: int | None) -> str:
    """How the check priced its KV room, for the lines that cite it."""
    return ("the run's prompt and max tokens" if depth
            else "the serve context")


def ram_bytes() -> int | None:
    """Physical RAM: the Metal device's, else the OS page count."""
    try:
        return int(metal_value("memory_size"))
    except Exception:
        try:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            return None


# The sentences a MoE load prints when its experts stream for want of
# memory. Fixed text: the troubleshooting docs are filed under them.
STREAM_FREE_NOW = ("the experts stream because they do not fit in the "
                   "memory free now")
STREAM_OVER_BUDGET = ("the experts stream because the model is over its "
                      "memory budget")
# What GMLX_OVERCOMMIT=1 does for a MoE's experts, in the lines that
# offer it: a load that wires them in chunks beside a streamed table
# (``expert_streaming._wire_resident_experts``) still stops the wire,
# and any other load holds them.
OVERCOMMIT_LEVER = ("GMLX_OVERCOMMIT=1 keeps them in memory unless the "
                    "memory the kernel can reclaim runs out while they wire.")
OVERCOMMIT_KEEPS = "GMLX_OVERCOMMIT=1 keeps them in memory."


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def _kept(room: KvRoom | None, depth: int | None, capped: bool) -> str:
    """What the keep holds, for the lines that cite it."""
    kv = "" if room is None else f" of {_gb(room.bytes)}"
    if room is not None and room.priced:
        kv += f" at {room.depth} tokens ({wire_room_label(depth)})"
    cap = ", capped at a fifth of RAM" if capped else ""
    return f"kept for the system, this process and the KV room{kv}{cap}"


@dataclass(frozen=True)
class WireCheck:
    """A resident wire judged against live memory. ``reason`` names the
    first leg it fails: ``budget`` (past the resident budget) or
    ``reclaimable`` (reclaimable RAM less ``keep`` is short); empty when
    it fits, the memory cannot be read, or GMLX_OVERCOMMIT=1 on a set
    that fits in RAM. ``ram_bytes`` is physical RAM, None when unread.
    ``dropped_bytes`` is an auto MTP drafter the load leaves out
    (``check_wire_beside_drafter``): to keep the experts in memory when
    it fits, else because they stream."""
    wire_bytes: int
    reclaimable_bytes: int | None
    keep_bytes: int
    budget_bytes: int | None
    reason: str = ""
    capped: bool = False
    dropped_bytes: int = 0
    ram_bytes: int | None = None

    @property
    def past_ram(self) -> bool:
        """The set is larger than physical RAM: no lever keeps it."""
        return self.ram_bytes is not None and self.wire_bytes > self.ram_bytes

    @property
    def fits(self) -> bool:
        return not self.reason

    @property
    def mtp_off(self) -> bool:
        return self.dropped_bytes > 0

    def mtp_off_line(self) -> str:
        """The line of a check that fits with auto MTP off."""
        return (f"auto MTP is off: the experts stay in memory, and the "
                f"{_gb(self.dropped_bytes)} MTP drafter does not fit beside "
                "them. --speculative streams the experts to keep it.")

    def line(self, room: KvRoom | None = None,
             depth: int | None = None, *, chunked: bool = False) -> str:
        """The stream line of a failed check, in GB: its fixed sentence,
        the leg that failed, and the ways to keep the experts in memory.
        ``room`` is the KV room the keep holds, priced at ``depth``
        (``wire_room_label``) unless it is flat. ``chunked`` is a load
        that wires the experts in chunks, which GMLX_OVERCOMMIT=1 still
        stops when reclaimable memory runs out. A set larger than RAM
        gets the reason and the numbers, with no lever."""
        if self.past_ram:
            lever = ""
        else:
            lever = " " + (OVERCOMMIT_LEVER if chunked else OVERCOMMIT_KEEPS)
        if self.reason == "budget":
            past = " and more than RAM" if self.past_ram else ""
            return (f"{STREAM_OVER_BUDGET}: holding them takes "
                    f"{_gb(self.wire_bytes)}, past the "
                    f"{_gb(self.budget_bytes or 0)} budget{past}.{lever}")
        tail = ("They are more than RAM." if self.past_ram
                else "To keep them in memory, close other apps.")
        return (f"{STREAM_FREE_NOW}: holding them takes "
                f"{_gb(self.wire_bytes)}, against "
                f"{_gb(self.reclaimable_bytes)} of reclaimable memory less "
                f"{_gb(self.keep_bytes)} {_kept(room, depth, self.capped)}. "
                f"{tail}{lever}")

    def stop_line(self, *, wired: int, reclaimable: int, need: int,
                  room: KvRoom | None = None,
                  depth: int | None = None) -> str:
        """The stream line of a chunked wire that stopped: reclaimable
        memory fell under the ``need`` of the next chunk with ``wired``
        bytes of it wired, and did not rise. Under GMLX_OVERCOMMIT=1
        each chunk kept only the kernel floor."""
        if overcommit():
            beside = f"beside the {_gb(kernel_floor_bytes())} kernel floor"
            lever = ""
        else:
            beside = (f"beside the {_gb(self.keep_bytes)} "
                      f"{_kept(room, depth, self.capped)}")
            lever = f" {OVERCOMMIT_LEVER}"
        return (f"{STREAM_FREE_NOW}: holding them takes "
                f"{_gb(self.wire_bytes)}, and after {_gb(wired)} of it "
                f"wired, reclaimable memory stopped at {_gb(reclaimable)}, "
                f"under the {_gb(need)} the next chunk needs {beside}. To "
                f"keep them in memory, close other apps.{lever}")


def check_resident_wire(wire_bytes: int, *, room_bytes: int,
                        budget: int | None = None,
                        host_bytes: int | None = None) -> WireCheck:
    """Whether wiring ``wire_bytes`` resident leaves the box room: under
    ``budget``, and under reclaimable RAM less the kept bytes
    (``wire_keep_bytes``), read live as the wire's chunk guard reads it
    (``live_reclaimable_ram_bytes``). A wired page leaves the file-backed
    count whether or not it was cached, so the model's own cached pages
    earn no credit. A wire of at most ``WIRE_EXEMPT_BYTES`` is judged on the
    budget only. GMLX_OVERCOMMIT=1 or an unreadable box admits, except a
    set larger than RAM, which fails on the budget leg."""
    ram = ram_bytes()
    host = host_overhead_bytes(ram) if host_bytes is None else host_bytes
    keep, capped = wire_keep_bytes(ram, room_bytes, host)
    w = max(0, int(wire_bytes))
    rec = live_reclaimable_ram_bytes() if w > WIRE_EXEMPT_BYTES else None
    reason = ""
    if not overcommit():
        if budget is not None and w > budget:
            reason = "budget"
        elif rec is not None and w > rec - keep:
            reason = "reclaimable"
    elif ram is not None and w > ram:
        # Overcommit keeps no set larger than RAM: it streams.
        reason = "budget"
    return WireCheck(w, rec, keep, budget, reason, capped, ram_bytes=ram)


def check_wire_beside_drafter(set_bytes: int, drafter_bytes: int, *,
                              auto: bool, room_bytes: int,
                              alone_room_bytes: int | None = None,
                              kept_streamed: bool = False,
                              budget: int | None = None) -> WireCheck:
    """``check_resident_wire`` for a resident set that an MTP drafter of
    ``drafter_bytes`` loads beside. An ``auto`` drafter (no flag asked
    for it) that tips the set over is dropped: the check of the set
    without it, at ``alone_room_bytes`` (the KV room of the load without
    the drafter, else ``room_bytes``), with ``dropped_bytes`` set. When
    the set fits alone, the load keeps its experts in memory and turns
    auto MTP off. When it does not, the experts stream, and a streamed
    load takes no auto drafter, unless it is ``kept_streamed`` (a VLM's
    text-only MTP), which the check then counts, as a drafter a flag
    asked for."""
    check = check_resident_wire(set_bytes + drafter_bytes,
                                room_bytes=room_bytes, budget=budget)
    if check.fits or not (auto and drafter_bytes):
        return check
    alone = check_resident_wire(
        set_bytes, budget=budget,
        room_bytes=room_bytes if alone_room_bytes is None else alone_room_bytes)
    if alone.fits or not kept_streamed:
        return replace(alone, dropped_bytes=drafter_bytes)
    return check
