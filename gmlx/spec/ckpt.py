"""Checkpoint-tier arming, exact anchors, and the plain ckpt decode path.

Split out of ``gmlx.spec.engine``; the engine keeps the L1 view, layout
probes, and env flags these helpers read.
"""

from __future__ import annotations

import logging
import os

from gmlx.envflags import env_int
from gmlx.spec.engine import (
    _L1View,
    _SPEC_APC_DISABLED,
    _SPEC_APC_RETIRE_DISABLED,
    _SPEC_APC_SIDECAR_DISABLED,
    _ckpt_active,
    _ckpt_layout_for,
    _ckpt_layout_live,
    _live_kv_quant_config,
)

_log = logging.getLogger(__name__)


def _l1_lookup_and_arm_store(batch, manager, mode, l0_prefix) -> int:
    """Consult the shared APCManager below L0 and arm the stock post-prefill
    store (mid-prefill exact checkpoints + post-prefill exact store / block
    harvest, all owned by stock ``PromptProcessingBatch.generate``) by
    populating ``_apc_manager`` / ``_apc_mode`` / ``_apc_meta``.

    Returns the restored L1 prefix length (0 on miss, or when L0 already
    restored -- L0 carries full-prompt hidden and is always preferred).

    ``meta["prefix_len"]`` stays 0 by design: the owned prefill keeps
    ``_processed_prompt_columns`` in absolute token space (it trims
    ``_input_ids`` in place, unlike stock warm batches which are constructed
    with suffix-only rows), and ``_row_real_tokens_processed`` -- which gates
    the mid-prefill checkpoint store -- is only correct in that space with a
    zero meta prefix. The one cost is that a block-tier harvest re-walks
    restored prefix blocks, but ``store_kv_blocks`` dedups by hash chain
    (acquire+release of existing blocks, no data copies).
    """
    view = _L1View(batch.model, manager, mode)
    ids_list = [int(t) for t in batch._mtp_full_input_ids[0].tolist()]
    prompt_kwargs = batch._prompt_kwargs or {}
    extra_hash = view._apc_extra_hash(prompt_kwargs)
    ckpt = _ckpt_active(batch.model, mode, int(manager.block_size))
    held_blocks = []
    l1_prefix = 0
    if l0_prefix == 0 and len(ids_list) >= 2:
        warm = None
        blocks = []
        prefix_len = 0
        tier = "exact"
        pick = view._apc_pick_for((0, ids_list, 0, prompt_kwargs, None, None))
        # Same trivial-pick floor as the admission wrapper: a sub-block
        # exact restore saves nothing and its nonzero l1_prefix would skip
        # the L0 hidden store for this request.
        if (pick is not None and not pick.get("matched_blocks")
                and 0 < int(pick.get("prefix_len") or 0)
                < int(manager.block_size)):
            pick = None
        if pick is not None:
            warm = pick.get("warm_cache")
            blocks = list(pick.get("matched_blocks") or ())
            prefix_len = int(pick.get("prefix_len") or 0)
            extra_hash = int(pick.get("extra_hash", extra_hash))
            if warm is None and blocks:
                from mlx_vlm import apc as _apc

                warm = _apc.make_warm_kv_cache(
                    blocks, min_capacity_tokens=len(ids_list) + 1
                )
                tier = "block"
        if ckpt:
            # Checkpoint tier: the longest salted sidecar + block chain
            # wins only when strictly longer than the exact-tier pick.
            # Media guards mirror the stock exact probe.
            from gmlx.cache.snapshot import ckpt_lookup
            min_p = max(prefix_len,
                        view._apc_safe_prefix_lookup_min(ids_list))
            cw, cp = ckpt_lookup(
                manager,
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=min_p,
                layout=_ckpt_layout_live(batch, int(manager.block_size)),
            )
            if (
                cw is not None
                and cp > prefix_len
                and view._apc_suffix_is_text_only(ids_list, cp)
            ):
                if blocks:
                    manager.release(blocks)
                    blocks = []
                warm, prefix_len, tier = cw, cp, "ckpt"
        elif mode == "exact":
            # Exact-tier anchor: the shared-system-prefix clone in the
            # gmlx anchor LRU wins only when strictly longer than the
            # stock exact pick. Media guards mirror the stock probe.
            from gmlx.cache.snapshot import anchor_exact_lookup
            min_p = max(prefix_len,
                        view._apc_safe_prefix_lookup_min(ids_list))
            aw, ap = anchor_exact_lookup(
                manager, ids_list, extra_hash=extra_hash,
                min_prefix_tokens=min_p)
            if (aw is not None and ap > prefix_len
                    and view._apc_suffix_is_text_only(ids_list, ap)):
                if blocks:
                    manager.release(blocks)
                    blocks = []
                warm, prefix_len, tier = aw, ap, "anchor"
        if warm and 0 < prefix_len < len(ids_list) and tier in (
                "exact", "anchor"):
            # Same batch-aware merge admission applies to its picks: raw
            # exact/anchor clones carry single-row leaves (left_padding
            # None, scalar offsets) and crash the batch cache classes'
            # update path (mx.depends on a None) when the suffix forwards.
            # kv_quant_config re-quantizes the float snapshot to the live
            # _make_cache layer types under serve kv_bits (stored exact
            # entries stay float; a float row joining a quantized batch
            # breaks the update path).
            from mlx_vlm import apc as _apc
            warm, _ = _apc.make_warm_batch_exact_cache_multi(
                [warm], prefix_lens=[prefix_len],
                kv_quant_config=_live_kv_quant_config(batch.model))
        if warm and 0 < prefix_len < len(ids_list):
            batch.prompt_cache = warm
            # Matched blocks stay acquired until the stock post-prefill
            # harvest releases them (the warm-cache concatenation is
            # lazy; the pool must not recycle these blocks before it
            # materializes).
            held_blocks = blocks
            l1_prefix = prefix_len
            # Observability only: the live request view reads the
            # restored prefix from here (meta keeps prefix_len 0 so the
            # stock machinery does not account it twice).
            batch._kq_apc_restored = (int(prefix_len), str(tier))
            _log.info(
                "APC L1 hit: prefix=%d suffix=%d tier=%s",
                prefix_len,
                len(ids_list) - prefix_len,
                tier,
            )
            # Drafter-KV sidecar: a plain L1 hit restores target KV but
            # not hidden, so the drafter would re-seed from suffix-only
            # hidden at the wrong positions (acceptance erodes at depth).
            # A sidecar covering exactly the restored prefix hands the
            # owned round a warm drafter start. Stash rides the first
            # cache entry, same discipline as the retirement context.
            if not _SPEC_APC_SIDECAR_DISABLED:
                from gmlx.cache.snapshot import drafter_sidecar_lookup
                side = drafter_sidecar_lookup(
                    manager, ids_list, prefix_len, extra_hash)
                if side:
                    batch.prompt_cache[0]._kq_apc_drafter_warm = side
                    _log.info("APC sidecar hit: prefix=%d", prefix_len)
        elif blocks:
            manager.release(blocks)
    batch._mtp_l1_prefix_len = l1_prefix
    batch._apc_manager = manager
    batch._apc_mode = mode
    guard = int(view._apc_exact_checkpoint_len(ids_list) or 0)
    meta = {
        "full_input_ids": ids_list,
        "prefix_len": 0,
        "extra_hash": extra_hash,
        "apc_blocks": held_blocks,
        "checkpoint_len": guard,
    }
    batch._apc_meta = [meta]
    if ckpt:
        # The checkpoint tier replaces the stock exact-tier stores: the
        # post-prefill full-cache clone is suppressed here, and the
        # mid-prefill checkpoint store is superseded by the cursor riding
        # the wrapped stock store (_install_ckpt_checkpoint_store; the
        # stock body is suppressed by the cursor's advance). Column
        # alignment itself still runs on the stock machinery, which
        # requires _apc_mode == "exact".
        _ckpt_arm_schedule(batch, meta, guard,
                           max(l0_prefix, l1_prefix),
                           int(manager.block_size))
        batch._apc_harvest_enabled = False
        batch._kq_ckpt_armed = True
        from gmlx.cache.snapshot import ckpt_note_armed
        ckpt_note_armed(manager)
    elif mode == "exact":
        _exact_anchor_arm(batch, meta, guard,
                          max(l0_prefix, l1_prefix))
    return l1_prefix


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _ckpt_unit(batch, block_size: int) -> int:
    """The natural chunk grid: lcm(prefill_step_size, block_size)."""
    step = int(getattr(batch, "prefill_step_size", 0) or 0)
    return block_size if step <= 0 else \
        step * block_size // _gcd(step, block_size)


def _ckpt_cursor_init(batch, guard: int, restored: int,
                      block_size: int) -> tuple[list, int, int]:
    """Boundary schedule for the checkpoint cursor: an ordered
    ``[(position, kind), ...]`` list plus ``(terminal, interval)``.

    Boundaries sit on the natural chunk grid, lcm(prefill_step_size,
    block_size) -- an off-grid boundary truncates a chunk, and gated-delta
    state is chunk-shape sensitive (certified: any grid change drifts).
    Interval points above the restored prefix, then the terminal (the
    grid point at or below the stock guard column). GMLX_APC_CKPT_INTERVAL
    tokens, default 4096, snapped up to the grid; 0 = terminal-only.
    Later stages add store positions by appending boundaries here, never
    by new store mechanisms.
    """
    unit = _ckpt_unit(batch, block_size)
    terminal = (guard // unit) * unit
    if terminal <= max(0, restored):
        return [], 0, 0
    raw = env_int("GMLX_APC_CKPT_INTERVAL", 4096)
    interval = 0 if raw <= 0 else max(unit, (raw // unit) * unit)
    bounds = []
    if interval:
        b = ((max(0, restored) // interval) + 1) * interval
        while b < terminal:
            bounds.append((b, "boundary"))
            b += interval
    bounds.append((terminal, "boundary"))
    return bounds, terminal, interval


def _ckpt_replay_boundary(batch, meta, restored: int,
                          block_size: int) -> int | None:
    """N-1 replay boundary, or None when it cannot earn its pause.

    An identical resend can only adopt a record strictly below the
    query, and the interval/terminal schedule never places one there
    for prompts under one interval (the depth e2e's bug 1); N-1 is the
    deepest position that is both adoptable and drift-free (the warm
    turn forwards exactly one token), and the pause is free on the cold
    side -- both prefill loops already stop at N-1 to feed the first
    decode step, so the boundary lands on a natural chunk edge and
    perturbs no chunk shape. arr layouts gate on a minimum N:
    recurrent state is prompt-length-independent (>100 MB per record on
    27B-class), and short-prompt records would churn the LRU out of the
    deep-conversation records it exists to protect. Rotating layouts
    need N-1 at or past the window (below it the store's grid gate
    declines). Kill switch: GMLX_APC_CKPT_REPLAY=0.
    """
    if env_int("GMLX_APC_CKPT_REPLAY", 1) == 0:
        return None
    n = len(meta.get("full_input_ids") or ())
    replay = n - 1
    if replay < 2 or replay <= max(0, restored):
        return None
    tags = _ckpt_layout_live(batch, block_size) or ()
    if "arr" in tags and n < env_int("GMLX_APC_CKPT_REPLAY_MIN", 1024):
        return None
    for t in tags:
        if t.startswith("rot") and replay < int(t.split(":")[1]):
            return None
    return replay


def _ckpt_turn_boundaries(batch, meta, restored: int,
                          block_size: int) -> list[int]:
    """Render-stable turn boundary positions for the schedule.

    p_stable is the deepest prompt position a next-turn re-render keeps;
    the gen-prompt/think tail past it is re-rendered away, so records
    stored only above it can never serve turn 2 (how multi-turn adoption
    silently died). Every layout gets the grid point at or below
    p_stable (drift-free for chunk-shape-sensitive state); rot-only
    layouts also pause exactly at p_stable (attention splits exactly;
    needs the window wrapped). GMLX_APC_CKPT_TURN=0 disables these and
    with them the p=N drop gate.
    """
    if env_int("GMLX_APC_CKPT_TURN", 1) == 0:
        return []
    ids = meta.get("full_input_ids") or ()
    unit = _ckpt_unit(batch, block_size)
    tags = _ckpt_layout_live(batch, block_size) or ()
    ws = [int(t.split(":")[1]) for t in tags if t.startswith("rot")]
    # Cheapest boundary this layout could arm: the grid needs one unit
    # of stable prefix; rot-only layouts can also pause exactly at
    # p_stable once the window wraps. Below that no boundary can land,
    # so skip the render+tokenize prediction entirely.
    need = unit if ("arr" in tags or not ws) else min(unit, max(ws))
    if len(ids) - 1 < need:
        return []
    from gmlx.cache.retire_key import lookup_render_ctx, prompt_stable_lcp
    ctx = lookup_render_ctx(ids)
    p_stable = prompt_stable_lcp(ctx, ids) if ctx else None
    if not p_stable or p_stable < 2:
        return []
    p_stable = min(int(p_stable), len(ids) - 1)
    meta["ckpt_p_stable"] = p_stable
    floor = max(0, restored)
    out = []
    grid = (p_stable // unit) * unit
    if grid > floor:
        out.append(grid)
    if ws and "arr" not in tags and p_stable != grid \
            and p_stable > floor and p_stable >= max(ws):
        out.append(p_stable)
    return out


def _ckpt_sys_boundary(batch, meta, restored: int,
                       block_size: int) -> int | None:
    """Anchor stop at the end of the shared system prefix.

    Sibling fan-out requests share the system prompt and tool schemas
    and diverge at the first user message, generally between grid
    points, so the interval schedule alone wastes up to one interval of
    sibling recompute, and strip-on-extend removes the early boundary
    the siblings need as the chain deepens (the anchor exemption in
    _record_insert keeps this one). arr layouts snap the stop down to
    the chunk grid (off-grid chunking drifts GDN state) and keep the
    replay byte floor (recurrent state is prompt-length-independent, so
    a tiny anchor costs the same >100 MB clone as a deep one);
    attention layouts snap to the block grid, which also satisfies the
    rotating store's below-window grid gate. GMLX_APC_CKPT_SYS=0
    disables; GMLX_APC_CKPT_SYS_MIN floors the position (a sub-floor
    shared prefix re-prefills in milliseconds and is not worth a
    record).
    """
    if env_int("GMLX_APC_CKPT_SYS", 1) == 0:
        return None
    ids = meta.get("full_input_ids") or ()
    tags = _ckpt_layout_for(getattr(batch, "model", None), block_size) or ()
    floor_min = max(block_size, env_int("GMLX_APC_CKPT_SYS_MIN", 256))
    if "arr" in tags:
        floor_min = max(floor_min,
                        env_int("GMLX_APC_CKPT_REPLAY_MIN", 1024))
    # Below the floor no anchor can land; skip the render+tokenize
    # prediction entirely (same rule as the turn boundaries).
    if len(ids) - 1 < floor_min:
        return None
    from gmlx.cache.retire_key import lookup_render_ctx, system_prefix_lcp
    ctx = lookup_render_ctx(ids)
    lcp = system_prefix_lcp(ctx, ids) if ctx else None
    if not lcp:
        return None
    unit = _ckpt_unit(batch, block_size) if "arr" in tags else block_size
    pos = (min(int(lcp), len(ids) - 1) // unit) * unit
    if pos < floor_min or pos <= max(0, restored):
        return None
    meta["ckpt_sys_bound"] = pos
    return pos


def _exact_anchor_boundary(batch, meta, guard: int,
                           restored: int) -> int | None:
    """Anchor position for an exact-tier (non-ckpt) model: the sibling
    divergence point, ungridded (exact clones restore at any position).
    Clamped to the stock guard column, so the prefill pauses at most
    twice: once for the anchor, once for the stock guard store.
    GMLX_APC_CKPT_SYS=0 disables (one switch for both tiers);
    GMLX_APC_CKPT_SYS_MIN floors the position (a sub-floor shared
    prefix re-prefills in milliseconds and is not worth a clone).
    """
    if env_int("GMLX_APC_CKPT_SYS", 1) == 0:
        return None
    ids = meta.get("full_input_ids") or ()
    floor_min = max(2, env_int("GMLX_APC_CKPT_SYS_MIN", 256))
    if len(ids) - 1 < floor_min:
        return None
    from gmlx.cache.retire_key import lookup_render_ctx, system_prefix_lcp
    ctx = lookup_render_ctx(ids)
    lcp = system_prefix_lcp(ctx, ids) if ctx else None
    if not lcp:
        _log.info("APC anchor declined: no measurable system prefix "
                  "(render ctx %s)", "present" if ctx else "missing")
        return None
    pos = min(int(lcp), len(ids) - 1)
    if guard > 0:
        pos = min(pos, guard)
    if pos < floor_min or pos <= max(0, restored):
        return None
    return pos


def _exact_anchor_arm(batch, meta, guard: int, restored: int) -> None:
    """Schedule the anchor pause by mirroring its position into
    ``checkpoint_len`` (the key the stock column truncation reads).
    ``_exact_anchor_store`` hands the column back to the stock guard
    after firing, so the stock store still runs exactly as unarmed."""
    pos = _exact_anchor_boundary(batch, meta, guard, restored)
    if pos is None:
        return
    meta["anchor_len"] = pos
    meta["anchor_guard"] = guard
    if pos != guard:
        meta["checkpoint_len"] = pos
    batch._kq_anchor_armed = True
    _log.info("APC anchor armed: pos=%d guard=%d", pos, guard)


def _exact_anchor_store(batch) -> None:
    """Anchor store for exact-tier models: one whole-prefix clone at the
    sibling divergence, into the gmlx anchor LRU. Runs from the wrapped
    stock store immediately before the stock body; after firing it
    restores ``checkpoint_len`` to the stock guard column without
    latching ``checkpoint_done``, so the stock guard store (and its
    latch) fire untouched."""
    manager = getattr(batch, "_apc_manager", None)
    meta_list = getattr(batch, "_apc_meta", None) or []
    if manager is None or not meta_list or meta_list[0] is None:
        return
    meta = meta_list[0]
    pos = int(meta.get("anchor_len") or 0)
    if pos <= 0 or meta.get("anchor_done"):
        return
    if batch._row_real_tokens_processed(0) != pos:
        return
    meta["anchor_done"] = True
    guard = int(meta.get("anchor_guard") or 0)
    if int(meta.get("checkpoint_len") or 0) == pos and pos != guard:
        meta["checkpoint_len"] = guard
    cache = batch._apc_prompt_cache_for_store(0)
    if cache is None:
        return
    from gmlx.cache.snapshot import anchor_exact_store
    anchor_exact_store(manager, meta["full_input_ids"][:pos], cache,
                       extra_hash=int(meta.get("extra_hash", 0)))


def _sched_insert(bounds: list, pos: int, kind: str, *,
                  upgrade: bool = False) -> None:
    """Insert (pos, kind) keeping order. On collision the existing entry
    keeps its kind: a colliding position is always grid-aligned or an
    exact turn boundary, where a plain boundary record adopts freely --
    identical resend included -- while flipping it to replay would gate
    turn-2 and branch adoption out on recurrent layouts (and satisfy the
    p=N drop with a record turn 2 cannot use). ``upgrade`` lets an
    anchor replace a plain boundary at the same position (strictly more
    retention, same free adoption), never a replay."""
    import bisect

    pts = [b for b, _ in bounds]
    i = bisect.bisect_left(pts, pos)
    if i < len(pts) and pts[i] == pos:
        if upgrade and bounds[i][1] == "boundary":
            bounds[i] = (pos, kind)
        return
    bounds.insert(i, (pos, kind))


def _ckpt_arm_schedule(batch, meta, guard: int, restored: int,
                       block_size: int) -> None:
    """Publish the boundary schedule into the request meta. The head
    mirrors into ``checkpoint_len`` (an int) because the stock
    checkpoint-column truncation and store reads exactly that key.
    ``ckpt_stored_boundaries`` collects every boundary whose store
    landed (record verified in the index) -- the settled variable the
    post-prefill p=N decision and the sidecar key set both read;
    ``ckpt_p_stable_bounds`` is the qualifying set for the p=N drop."""
    bounds, terminal, interval = _ckpt_cursor_init(
        batch, guard, restored, block_size)
    turn = _ckpt_turn_boundaries(batch, meta, restored, block_size)
    for pos in turn:
        _sched_insert(bounds, pos, "boundary")
    sysb = _ckpt_sys_boundary(batch, meta, restored, block_size)
    if sysb is not None:
        _sched_insert(bounds, sysb, "anchor", upgrade=True)
    replay = _ckpt_replay_boundary(batch, meta, restored, block_size)
    if replay is not None:
        # Colliding with the anchor keeps the anchor (default no-upgrade):
        # it adopts identical resends freely, replay semantics add nothing.
        _sched_insert(bounds, replay, "replay")
    meta["ckpt_boundaries"] = bounds
    meta["checkpoint_len"] = int(bounds[0][0]) if bounds else 0
    meta["ckpt_terminal"] = terminal
    meta["ckpt_interval"] = interval
    meta["ckpt_last_stored"] = 0
    meta["ckpt_stored_boundaries"] = []
    meta["ckpt_p_stable_bounds"] = turn


def _ckpt_mid_prefill_store(batch) -> None:
    """Checkpoint-tier replacement for the stock mid-prefill exact store.

    Fires at the schedule head, pops it, and mirrors the next head into
    ``checkpoint_len``, latching ``checkpoint_done`` when the schedule
    empties. The advance is what suppresses the stock store;
    ``_install_ckpt_checkpoint_store`` wraps the stock method so the
    cursor always runs immediately before it -- the ordering is
    structural, not positional. Advances past failed stores;
    ``ckpt_last_stored`` records only boundaries that landed.
    """
    if not getattr(batch, "_kq_ckpt_armed", False):
        return
    manager = getattr(batch, "_apc_manager", None)
    meta_list = getattr(batch, "_apc_meta", None) or []
    if manager is None or not meta_list or meta_list[0] is None:
        return
    meta = meta_list[0]
    if meta.get("checkpoint_done"):
        return
    checkpoint_len = int(meta.get("checkpoint_len") or 0)
    if checkpoint_len <= 0:
        return
    if batch._row_real_tokens_processed(0) != checkpoint_len:
        return
    terminal = int(meta.get("ckpt_terminal") or 0)
    bounds = meta.get("ckpt_boundaries") or []
    kind = "boundary"
    if bounds and int(bounds[0][0]) == checkpoint_len:
        kind = str(bounds.pop(0)[1])
    # Inline-heavy skeletons (GDN state >100 MB; kvarn state scales with p
    # across every attention layer) earn disk only at the terminal --
    # boundaries superseded within the same prefill do not, and a replay
    # skeleton would buy restart-repair of an identical resend only,
    # which does not earn it either.
    layout = _ckpt_layout_live(batch, int(manager.block_size)) or ()
    heavy = "arr" in layout or any(t.startswith("kvarn") for t in layout)
    skel = not heavy or (kind != "replay"
                         and checkpoint_len >= terminal)
    from gmlx.cache.snapshot import ckpt_store

    if ckpt_store(
            manager, meta["full_input_ids"][:checkpoint_len],
            batch.prompt_cache, extra_hash=int(meta.get("extra_hash", 0)),
            skeleton_disk=skel, kind=kind):
        meta["ckpt_last_stored"] = checkpoint_len
        meta.setdefault("ckpt_stored_boundaries", []).append(checkpoint_len)
    if bounds:
        meta["checkpoint_len"] = int(bounds[0][0])
    else:
        meta["checkpoint_done"] = True


_CKPT_STORE_FLAG = "_kq_ckpt_cursor_store"


def _install_ckpt_checkpoint_store() -> None:
    """Wrap the stock mid-prefill checkpoint store so the cursor runs
    immediately before it on armed batches (both the owned MTP prefill
    and the stock prompt_step call the stock method, so one wrap covers
    both paths). The cursor's advance of ``checkpoint_len`` is what
    suppresses the stock store -- wrapping makes that ordering
    structural. Exact-tier anchor batches ride the same wrap with their
    own single-stop hook. Idempotent."""
    from mlx_vlm.generate.ar import PromptProcessingBatch

    if getattr(
        PromptProcessingBatch._store_apc_exact_checkpoints, _CKPT_STORE_FLAG, False
    ):
        return
    _orig = PromptProcessingBatch._store_apc_exact_checkpoints

    def _store_with_ckpt_cursor(self):
        if getattr(self, "_kq_ckpt_armed", False):
            _ckpt_mid_prefill_store(self)
        elif getattr(self, "_kq_anchor_armed", False):
            _exact_anchor_store(self)
        _orig(self)

    _store_with_ckpt_cursor.__dict__[_CKPT_STORE_FLAG] = True
    PromptProcessingBatch._store_apc_exact_checkpoints = _store_with_ckpt_cursor


def _snap_fields(batch, manager) -> dict:
    """Decode-time snapshot ring parameters for a retirement stash.

    ``snap_grid`` anchors snapshot positions to the prefill chunk grid
    (lcm of step and block size), so a restore replays chunk-exact --
    but only while one grid unit fits inside the snapshot interval; a
    serve-sized step (2048) would otherwise push the first snapshot far
    past prompt end + interval, so it falls back to the block size (the
    off-grid restore is the scoped-benign case). ``snap_align`` is the
    block alignment a rotating window store requires below the window;
    ``snap_offgrid_min`` (= W) is where the store gate stops caring --
    a wrapped window is whole blocks at any p.
    """
    import math
    from gmlx.cache.snapshot import _DECODE_CKPT_DEFAULT
    bs = int(manager.block_size)
    tags = _ckpt_layout_live(batch, bs) or ()
    step = int(getattr(batch, "prefill_step_size", 0) or 0)
    grid = math.lcm(step, bs) if step > 0 else bs
    if grid > env_int("GMLX_APC_DECODE_CKPT", _DECODE_CKPT_DEFAULT):
        grid = bs
    rot_w = 0
    for t in tags:
        if t.startswith("rot"):
            rot_w = int(t.split(":")[1])
            break
    return {
        "snap_ok": bool(tags),
        "snap_grid": grid,
        "snap_align": bs if rot_w else 1,
        "snap_offgrid_min": rot_w,
    }


def _plain_ckpt_init(batch) -> None:
    """Checkpoint-tier lookup + arming for a stock (non-speculative)
    prompt batch.

    The stock path reaches the tier only here: exact-tier stores are
    suppressed on ckpt models, so admission's own lookup ladder misses
    and every ckpt-tier request arrives as a cold single-request batch.
    Lookup and in-place prefix trim mirror the owned MTP prefill
    (single-row caches throughout; the batched warm-merge machinery
    never runs). B=1 unbatched batches only; anything else stays stock.
    """
    manager = getattr(batch, "_apc_manager", None)
    mode = getattr(batch, "_apc_mode", None)
    meta_list = getattr(batch, "_apc_meta", None) or []
    if (
        manager is None
        or mode != "exact"
        or len(meta_list) != 1
        or meta_list[0] is None
        or len(batch.uids) != 1
        or batch._right_pad_per_row is not None
        or batch._inputs_embeds is None
    ):
        return
    bs = int(manager.block_size)
    if not _ckpt_active(batch.model, mode, bs):
        return
    meta = meta_list[0]
    if int(meta.get("prefix_len") or 0):
        return  # stock warm row: leave it stock
    ids_list = [int(t) for t in meta["full_input_ids"]]
    if len(ids_list) < 2:
        return
    extra_hash = int(meta.get("extra_hash", 0))
    view = _L1View(batch.model, manager, mode)
    restored = 0
    from gmlx.cache.snapshot import ckpt_lookup
    warm, cp = ckpt_lookup(
        manager,
        ids_list,
        extra_hash=extra_hash,
        min_prefix_tokens=view._apc_safe_prefix_lookup_min(ids_list),
        layout=_ckpt_layout_live(batch, bs),
    )
    if (
        warm is not None
        and 0 < cp < len(ids_list)
        and view._apc_suffix_is_text_only(ids_list, cp)
    ):
        batch.prompt_cache = warm
        batch._input_ids = batch._input_ids[:, cp:]
        batch._inputs_embeds = batch._inputs_embeds[:, cp:]
        batch._processed_prompt_columns = cp
        for k in batch._prompt_length_aware_keys:
            batch._prompt_kwargs[k] = batch._prompt_kwargs[k][:, cp:, ...]
        restored = cp
        batch._kq_apc_restored = (int(cp), "ckpt")     # live request view
        _log.info("APC L1 hit: prefix=%d suffix=%d tier=ckpt",
                  cp, len(ids_list) - cp)
    guard = int(meta.get("checkpoint_len") or 0)
    _ckpt_arm_schedule(batch, meta, guard, restored, bs)
    batch._apc_harvest_enabled = False
    batch._kq_ckpt_armed = True
    from gmlx.cache.snapshot import ckpt_note_armed
    ckpt_note_armed(manager)
    if not _SPEC_APC_RETIRE_DISABLED and batch.prompt_cache:
        from gmlx.cache.retire_key import lookup_render_ctx
        batch.prompt_cache[0]._kq_apc_retire = {
            "full_ids": ids_list,
            "extra_hash": extra_hash,
            "mode": "ckpt",
            "checkpoint_len": int(meta.get("checkpoint_len") or 0),
            "apc_meta": meta,
            "render_ctx": lookup_render_ctx(ids_list),
            "manager": manager,
            "gen": [],
            **_snap_fields(batch, manager),
        }


def _plain_anchor_init(batch) -> None:
    """Arm the exact-tier anchor stop on a stock prompt batch (non-ckpt
    exact models: DeepSeek-V4-class pooling stacks).

    Restores come from the admission pick (_install_exact_anchor_pick),
    so this only schedules the store. Warm and right-padded rows are
    included: a restored prefix is usually far short of the divergence
    (a bare bos match off some unrelated request), and upstream's
    checkpoint column and row extraction handle both shapes. Refusing
    them would skip every row that rides a warm batch, which on a busy
    server is nearly all of them. The restored prefix becomes the
    boundary floor, so a row already past the divergence arms nothing.
    """
    manager = getattr(batch, "_apc_manager", None)
    mode = getattr(batch, "_apc_mode", None)
    meta_list = getattr(batch, "_apc_meta", None) or []
    if (manager is None or mode != "exact" or len(meta_list) != 1
            or meta_list[0] is None or len(batch.uids) != 1
            or batch._inputs_embeds is None):
        return
    if _ckpt_active(batch.model, mode, int(manager.block_size)):
        return                          # ckpt tier owns these models
    meta = meta_list[0]
    if len(meta.get("full_input_ids") or ()) < 2:
        return
    _exact_anchor_arm(batch, meta, int(meta.get("checkpoint_len") or 0),
                      int(meta.get("prefix_len") or 0))
    # Retirement stash, independent of the anchor outcome: exact-tier
    # rows retire their full post-decode row at filter (the per-turn
    # store the post-prefill exact store cannot cover), warm rows
    # included -- the decode cache holds the full sequence either way.
    if not _SPEC_APC_RETIRE_DISABLED and batch.prompt_cache:
        from gmlx.cache.retire_key import lookup_render_ctx
        ids_list = [int(t) for t in meta["full_input_ids"]]
        batch.prompt_cache[0]._kq_apc_retire = {
            "full_ids": ids_list,
            "extra_hash": int(meta.get("extra_hash", 0)),
            "mode": "exact",
            "manager": manager,
            "render_ctx": lookup_render_ctx(ids_list),
            "gen": [],
        }


_ANCHOR_PICK_FLAG = "_kq_exact_anchor_pick"


def _install_exact_anchor_pick() -> None:
    """Consult the anchor LRU inside the stock admission pick.

    The pick is where a warm prefix belongs: admission builds the batch
    from it (suffix rows, right padding, warm-cache merge) and every
    downstream path treats an anchor restore exactly like a stock exact
    one. The anchor wins only when strictly longer than the stock pick,
    so it never shortens a restore. Idempotent.
    """
    from mlx_vlm.generate.ar import BatchGenerator
    if getattr(BatchGenerator._apc_pick_for, _ANCHOR_PICK_FLAG, False):
        return
    _orig = BatchGenerator._apc_pick_for

    def _pick_with_anchor(self, sequence):
        pick = _orig(self, sequence)
        try:
            if _SPEC_APC_DISABLED or getattr(self, "apc_mode", None) != "exact":
                return pick
            manager = getattr(self, "apc_manager", None)
            if manager is None or _ckpt_active(
                    getattr(self, "model", None), "exact",
                    int(manager.block_size)):
                return pick
            _uid, ids_list, _mt, prompt_kwargs, _lps, _crit = sequence
            if not ids_list or len(ids_list) < 2:
                return pick
            # Floor trivial exact picks: a sub-block restore (a bare-BOS
            # match off an unrelated request) saves nothing but suffix-
            # constructs the batch, knocking the spec path's ids out of
            # render space (anchor + retirement keys). Real warm picks are
            # thousands of tokens and pass untouched.
            if (pick is not None and not pick.get("matched_blocks")
                    and 0 < int(pick.get("prefix_len") or 0)
                    < int(manager.block_size)):
                pick = None
            have = int((pick or {}).get("prefix_len") or 0)
            extra_hash = self._apc_extra_hash(prompt_kwargs or {})
            floor = max(have, self._apc_safe_prefix_lookup_min(ids_list))
            from gmlx.cache.snapshot import anchor_exact_lookup
            warm, ap = anchor_exact_lookup(
                manager, ids_list, extra_hash=extra_hash,
                min_prefix_tokens=floor)
            if warm is None or ap <= have or ap >= len(ids_list):
                return pick
            if not self._apc_suffix_is_text_only(ids_list, ap):
                return pick
            if pick and pick.get("matched_blocks"):
                manager.release(pick["matched_blocks"])
            _log.info("APC L1 hit: prefix=%d suffix=%d tier=anchor",
                      ap, len(ids_list) - ap)
            return {
                "matched_blocks": [],
                "warm_cache": warm,
                "prefix_len": ap,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        except Exception:
            _log.warning("APC anchor pick failed; continuing",
                         exc_info=True)
            return pick

    _pick_with_anchor.__dict__[_ANCHOR_PICK_FLAG] = True
    BatchGenerator._apc_pick_for = _pick_with_anchor


_PLAIN_DECODE_FLAG = "_kq_ckpt_plain_decode"


def _retire_rows(gb) -> dict:
    """uid -> retire-stash registry on a generation batch.

    Stashes arm on the B=1 prompt batch's cache object (the only stable
    home before the decode batch exists); the first decode-side touch
    lifts them here so they survive ``extend`` rebuilding the cache
    objects at continuous-batch injection."""
    reg = getattr(gb, "_kq_apc_retire_rows", None)
    if reg is None:
        reg = {}
        gb._kq_apc_retire_rows = reg
    return reg


def _lift_cache_stash(gb) -> None:
    if not getattr(gb, "prompt_cache", None) or len(gb.uids) != 1:
        return
    stash = getattr(gb.prompt_cache[0], "_kq_apc_retire", None)
    if stash is not None:
        gb.prompt_cache[0]._kq_apc_retire = None
        _retire_rows(gb)[gb.uids[0]] = stash


def _plain_step_tick(gb, out) -> None:
    """Per-token accounting + snapshot tick for stock-path retire rows.

    Rows are tracked per uid so accounting survives ``extend`` merges.
    Runs per step, so a deterministic failure disables the hook for that
    row on first strike instead of emitting a traceback per token;
    dropping ``gen`` also quiets retirement (its offset check would skip
    anyway on a broken count). The decode-time snapshot ring stays B=1
    (its clones ride the live single-row caches); rows in a B>1 batch
    retire snapshot-free, under their verbatim key or an LCP cap the
    tier arm can serve without a ring."""
    try:
        _lift_cache_stash(gb)
        reg = getattr(gb, "_kq_apc_retire_rows", None)
    except Exception:
        _log.warning("APC plain decode hook failed; continuing",
                     exc_info=True)
        return
    if not reg:
        return
    # _step returns (tokens, lps, top_idx, top_lp); slot 0 is the flat
    # per-row token list.
    rows = out[0] if isinstance(out, tuple) else out
    if rows is None:
        return
    solo = len(gb.uids) == 1
    for i, uid in enumerate(gb.uids):
        stash = reg.get(uid)
        if stash is None or "gen" not in stash:
            continue
        tok = rows[i] if i < len(rows) else None
        if tok is None:
            continue                    # no emission for this row this tick
        try:
            if isinstance(tok, (list, tuple)):
                tok = tok[0]
            stash["gen"].append(int(tok))
            if solo and stash.get("mode") == "ckpt":
                from gmlx.cache.snapshot import decode_ckpt_tick
                decode_ckpt_tick(stash, gb.prompt_cache, stash["gen"])
        except Exception:
            stash.pop("gen", None)
            stash["snap_ok"] = False
            _log.warning("APC plain decode hook failed; disabled for "
                         "this request", exc_info=True)


def _plain_retire(stash: dict, prompt_cache: list) -> None:
    """Retire a finished stock-path row off a single-row cache list.

    Offset invariants mirror ``speculative._retire_b1``: the stock step
    loop forwards each token as it emits it, so a clean finish leaves
    ``offset == len(seq)`` (an abort between steps leaves the same).
    ``stash["mode"]`` picks the tier arm: "ckpt" stores blocks +
    sidecar, "exact" a whole-row snapshot (DeepSeek-V4-class pooling
    stacks).
    """
    try:
        manager = stash.get("manager")
        if manager is None:
            return
        gen = [int(t) for t in stash.get("gen") or ()]
        if not gen:
            return
        seq = [int(t) for t in stash["full_ids"]] + gen
        from gmlx.cache.snapshot import _cache_offset_max, retirement_store
        offset = _cache_offset_max(prompt_cache)
        if offset == len(seq) - 1:
            seq = seq[:-1]
        elif offset != len(seq):
            _log.info(
                "APC retire skipped: cache offset %d != tokens %d", offset, len(seq)
            )
            return
        lcp = None
        if os.environ.get("GMLX_APC_RETIRE_LCP") != "0":
            from gmlx.cache.retire_key import next_turn_lcp
            lcp = next_turn_lcp(stash.get("render_ctx"), seq, gen)
        max_len = lcp if lcp is not None and lcp < len(seq) else None
        _log.info("APC retire: seq=%d ctx=%s lcp=%s cap=%s",
                  len(seq), stash.get("render_ctx") is not None,
                  lcp, max_len)
        ok = retirement_store(
            manager, stash.get("mode") or "ckpt", seq, prompt_cache,
            row=0,
            extra_hash=int(stash.get("extra_hash", 0)), max_len=max_len,
            decode_snaps=stash.get("snaps"))
        if ok:
            _log.info("APC retire store: tokens=%d", ok)
    except Exception:
        _log.warning("APC retire failed; continuing", exc_info=True)


def _install_plain_ckpt_decode() -> None:
    """Stock-path decode hooks for the retirement store (ckpt + exact).

    Token accounting rides ``_step``; retirement fires from ``filter``
    for every leaving row (finish or client abort). A lone row retires
    off its live single-row caches; a row leaving a B>1 batch is first
    extracted via ``row_snapshot`` (padding-trimmed clones with row-true
    offsets), so retirement survives concurrency instead of firing only
    when the batch happens to drain to one row. Stashes live in a
    uid-keyed registry lifted across ``extend`` (the seam that rebuilds
    cache objects at continuous-batch injection).
    GMLX_APC_RETIRE_BATCH=0 restores the lone-row-only v1 scope.
    Idempotent."""
    from mlx_vlm.generate.ar import GenerationBatch

    if getattr(GenerationBatch._step, _PLAIN_DECODE_FLAG, False):
        return
    _orig_step = GenerationBatch._step
    _orig_filter = GenerationBatch.filter
    _orig_extend = GenerationBatch.extend

    def _step_with_ckpt(self):
        out = _orig_step(self)
        _plain_step_tick(self, out)
        return out

    def _filter_with_ckpt(self, keep):
        try:
            _lift_cache_stash(self)
            reg = getattr(self, "_kq_apc_retire_rows", None)
            if reg and self.prompt_cache:
                keep_set = set(keep)
                solo = len(self.uids) == 1
                batched_ok = os.environ.get(
                    "GMLX_APC_RETIRE_BATCH") != "0"
                for i, uid in enumerate(self.uids):
                    if i in keep_set:
                        continue
                    stash = reg.pop(uid, None)
                    if stash is None:
                        continue
                    if solo:
                        _plain_retire(stash, self.prompt_cache)
                    elif batched_ok:
                        from gmlx.cache.snapshot import row_snapshot
                        rows = row_snapshot(self.prompt_cache, i)
                        if rows is None:
                            _log.info("APC retire skipped: row %d "
                                      "extract unavailable", i)
                        else:
                            _plain_retire(stash, rows)
        except Exception:
            _log.warning("APC plain retire hook failed; continuing", exc_info=True)
        _orig_filter(self, keep)

    def _extend_with_ckpt(self, other):
        try:
            _lift_cache_stash(self)
            _lift_cache_stash(other)
            other_reg = getattr(other, "_kq_apc_retire_rows", None)
            if other_reg:
                _retire_rows(self).update(other_reg)
                other._kq_apc_retire_rows = {}
        except Exception:
            _log.warning("APC retire stash carry failed; continuing",
                         exc_info=True)
        _orig_extend(self, other)

    _step_with_ckpt.__dict__[_PLAIN_DECODE_FLAG] = True
    _filter_with_ckpt.__dict__[_PLAIN_DECODE_FLAG] = True
    _extend_with_ckpt.__dict__[_PLAIN_DECODE_FLAG] = True
    GenerationBatch._step = _step_with_ckpt
    GenerationBatch.filter = _filter_with_ckpt
    GenerationBatch.extend = _extend_with_ckpt
