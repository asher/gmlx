"""Single-row cache extraction + retirement store for the owned MTP engine.

``row_snapshot`` is the inverse of the batch cache classes' ``merge``: it slices
one row out of a (possibly batched) prompt cache into a list of single-row
caches, exactly the shape ``APCManager.store_exact_cache`` consumes and the shape
``merge``/``extend`` reconstruct a batch from. ``retirement_store`` uses it to
persist a finished request's full context (prompt + generated tokens) into the
shared APC, so a follow-up turn that repeats this text as a prefix warm-starts.

Stock mlx-vlm only harvests KV at prefill and never stores generated tokens, so
retirement is a beyond-stock win for multi-turn conversations. It runs between
generations, off the per-round hot path.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from .envflags import env_int

_log = logging.getLogger(__name__)


def _layer_has_content(snap: Any) -> bool:
    """True when a snapshotted layer carries reusable state.

    A KV layer needs a non-empty offset with materialized keys; a recurrent
    (ArraysCache) layer needs non-empty state; a CacheList needs all of its
    sub-caches populated. An incomplete layer means the row cannot be stored
    under a full-sequence key without lying about what it covers.
    """
    subs = getattr(snap, "caches", None)
    if subs is not None:  # CacheList
        return all(_layer_has_content(s) for s in subs)
    if hasattr(snap, "accumulate_windows"):  # PoolingCache: no `keys`;
        # content lives in pooled rows and/or the staging remainder
        return snap.size() > 0 or getattr(snap, "remainder", 0) > 0
    off = getattr(snap, "offset", None)
    if isinstance(off, int):
        return off > 0 and getattr(snap, "keys", None) is not None
    empty = getattr(snap, "empty", None)
    if callable(empty):
        try:
            return not empty()
        except Exception:
            return True  # unprobeable cache -> assume it holds state
    return True


def _clone_single_row(cache: Any) -> Any | None:
    """Deep-copy an already-single-row cache, preserving its concrete kind.

    Reuses upstream's APC clone so the snapshot matches what
    ``store_exact_cache`` would produce for the non-speculative path.
    """
    from mlx_vlm import apc as _apc

    eval_targets: list[Any] = []
    out = _apc._clone_cache_entry_for_apc(
        cache, min_capacity_tokens=None, eval_targets=eval_targets)
    if out is None:
        # Upstream isinstance-gates on the mlx_vlm cache classes; gmlx text
        # models carry the mlx_lm twins. Mirror the same per-kind copy for
        # any class cache_types recognizes.
        out = _clone_lm_twin(cache, eval_targets)
    if eval_targets:
        import mlx.core as mx
        mx.eval(*eval_targets)
    return out


def _clone_lm_twin(cache: Any, eval_targets: list[Any]) -> Any | None:
    from mlx_vlm import apc as _apc
    from .cache_compat import cache_types

    copy = _apc._copy_mlx_array
    if isinstance(cache, cache_types("RotatingKVCache")):
        out = type(cache)(max_size=int(cache.max_size),
                          keep=int(getattr(cache, "keep", 0)))
        out.offset = int(getattr(cache, "offset", 0) or 0)
        out._idx = int(getattr(cache, "_idx", 0) or 0)
        if cache.keys is not None and cache.values is not None:
            out.keys = copy(cache.keys)
            out.values = copy(cache.values)
            eval_targets.extend([out.keys, out.values])
        return out
    if isinstance(cache, cache_types("KVCache")):
        out = type(cache)()
        off = int(getattr(cache, "offset", 0) or 0)
        if cache.keys is not None and cache.values is not None and off > 0:
            out.keys = copy(cache.keys[..., :off, :])
            out.values = copy(cache.values[..., :off, :])
            out.offset = off
            eval_targets.extend([out.keys, out.values])
        return out
    if isinstance(cache, cache_types("ArraysCache")):
        out = type(cache)(len(cache.cache))
        out.cache = []
        for state in cache.cache:
            if state is None:
                out.cache.append(None)
                continue
            copied = copy(state)
            out.cache.append(copied)
            eval_targets.append(copied)
        for attr in ("left_padding", "lengths"):
            v = getattr(cache, attr, None)
            if v is not None:
                copied = copy(v)
                setattr(out, attr, copied)
                eval_targets.append(copied)
        return out
    return None


def row_snapshot(prompt_cache: list[Any], row: int = 0) -> list[Any] | None:
    """Extract ``row`` from a prompt cache as a list of single-row caches.

    Returns caches that ``store_exact_cache`` can persist and ``merge``/``extend``
    can splice back into a live batch (the round-trip inverse of ``merge``).
    Returns None when any layer lacks content for this row -- an incomplete
    snapshot must never be stored under a full-sequence key. Rotating (sliding
    window) layers are snapshot-able even after they wrap: the window they retain
    is exactly what sliding-window attention re-attends to on continuation.
    """
    if not prompt_cache:
        return None
    snaps: list[Any] = []
    for cache in prompt_cache:
        extract = getattr(cache, "extract", None)
        if callable(extract):
            try:
                snap = extract(row)
            except AttributeError:
                # CacheList.extract delegates to sub-caches, and single-row
                # sub-caches (e.g. RotatingKVCache in an unbatched deepseek4
                # stack) don't implement it. Row 0 of a single-row stack is
                # the whole stack; any other row doesn't exist.
                if row != 0:
                    return None
                snap = _clone_single_row(cache)
        else:
            # Already single-row (B=1 without a batch wrapper); clone so the
            # stored copy is decoupled from the live decode cache.
            snap = _clone_single_row(cache)
        if snap is None or not _layer_has_content(snap):
            return None
        snaps.append(snap)
    return snaps


# Mixed into extra_hash for drafter-KV sidecar entries on the disk tier so
# they can never collide with (or be loaded as) real full-cache exact shards
# under the same token key. Any fixed wide constant works; changing it
# invalidates previously persisted sidecars, nothing else.
_SIDECAR_SALT = 0x5D_CA_9E_11_3F_2B_71

# Sidecars get their own in-memory LRU (attached to the manager instance),
# never the manager's exact-entry LRU: that one defaults to 2 slots, and the
# 2-3 tiny sidecars a request stores would evict the multi-GB real entries
# they exist to accompany.
_SIDECAR_ENTRIES = max(
    1, env_int("GMLX_SPEC_APC_SIDECAR_ENTRIES", 8))


def sidecar_extra_hash(extra_hash: int) -> int:
    return int(extra_hash) ^ _SIDECAR_SALT


def _sidecar_index(manager: Any) -> "OrderedDict":
    with manager.lock:
        idx = getattr(manager, "_kq_sidecar_cache", None)
        if idx is None:
            idx = OrderedDict()
            manager._kq_sidecar_cache = idx
        return idx


def drafter_sidecar_store(
    manager: Any,
    drafter: Any,
    token_ids,
    store_len: int,
    extra_hash: int = 0,
) -> bool:
    """Store the drafter's own KV, trimmed to ``store_len`` positions, under
    the target entry's token key (side index + salted disk shard).

    Alignment invariant: drafter KV row ``p`` holds (token_{p+1}, hidden_p), so
    a sidecar covering exactly ``store_len`` rows pairs with a target entry of
    ``store_len`` tokens -- on restore, the warm turn's suffix hidden (rows
    ``store_len``..) teacher-forces at exactly the right positions. A drafter
    whose KV covers fewer rows than ``store_len`` cannot be stored faithfully
    and is skipped. Best-effort; never raises.
    """
    if manager is None or drafter is None or store_len < 1:
        return False
    if not getattr(drafter, "supports_kv_sidecar", False):
        return False
    try:
        caches = drafter.export_kv()
        if not caches:
            return False
        clones: list[Any] = []
        for c in caches:
            offset = int(getattr(c, "offset", 0) or 0)
            if offset < store_len:
                _log.info(
                    "APC sidecar store skipped: head offset %d < %d",
                    offset, store_len)
                return False
            clone = _clone_single_row(c)
            if clone is None:
                return False
            clone.trim(offset - store_len)
            clones.append(clone)
        ids = tuple(int(t) for t in token_ids)[:store_len]
        idx = _sidecar_index(manager)
        with manager.lock:
            idx[(ids, int(extra_hash))] = clones
            idx.move_to_end((ids, int(extra_hash)))
            while len(idx) > _SIDECAR_ENTRIES:
                idx.popitem(last=False)
        disk = getattr(manager, "disk", None)
        if disk is not None:
            try:
                from mlx_vlm import apc as _apc
                salted = sidecar_extra_hash(extra_hash)
                khash = _apc._sequence_hash(ids, salted, manager.block_size)
                disk.save_exact_cache(khash, ids, salted, clones)
            except Exception:
                _log.debug("APC sidecar disk save failed", exc_info=True)
        return True
    except Exception:
        _log.warning("APC sidecar store failed; continuing", exc_info=True)
        return False


def drafter_sidecar_lookup(
    manager: Any,
    token_ids,
    prefix_len: int,
    extra_hash: int = 0,
) -> list[Any] | None:
    """Fetch a drafter-KV sidecar covering exactly ``prefix_len`` tokens.

    The exact-length gate matters: a shorter sidecar would leave a positional
    hole between its rows and the warm turn's suffix hidden (drafter positions
    come from its own cache offset), so a near-miss is worse than a cold
    drafter start. Memory side index first (O(1): the exact-length
    requirement fully determines the key), then the disk tier's salted exact
    shards. Returns a restore-safe clone list or None.
    """
    if manager is None or prefix_len < 1:
        return None
    try:
        ids = tuple(int(t) for t in token_ids)
        if len(ids) < prefix_len:
            return None
        key = (ids[:prefix_len], int(extra_hash))
        idx = _sidecar_index(manager)
        with manager.lock:
            entry = idx.get(key)
            if entry is not None:
                idx.move_to_end(key)
        if entry is not None:
            return [_clone_single_row(c) for c in entry]
        disk = getattr(manager, "disk", None)
        if disk is None:
            return None
        salted = sidecar_extra_hash(extra_hash)
        match = disk.find_exact_prefix(
            ids, extra_hash=salted,
            max_prefix_tokens=prefix_len,
            min_prefix_tokens=prefix_len - 1,
        )
        if match is None:
            return None
        cache_hash, plen = match
        if plen != prefix_len:
            return None
        loaded = disk.load_exact_cache(
            cache_hash, min_capacity_tokens=len(ids) + 1)
        if loaded is None:
            return None
        stored_tokens, stored_extra, caches = loaded
        if (stored_extra != salted or len(stored_tokens) != prefix_len
                or stored_tokens != ids[:prefix_len]):
            return None
        with manager.lock:
            idx[key] = caches
            idx.move_to_end(key)
            while len(idx) > _SIDECAR_ENTRIES:
                idx.popitem(last=False)
        return [_clone_single_row(c) for c in caches]
    except Exception:
        _log.warning("APC sidecar lookup failed; continuing", exc_info=True)
        return None


# Rotating (sliding-window) snapshot/restore inverse. A live RotatingKVCache
# buffer has three regimes (contiguous concat-mode, padded in-place growth,
# rotated ring); the snapshot canonicalizes all three into temporal order at
# exactly L = min(offset, max_size) tokens: [0, keep) plus the trailing
# window. Restore rebuilds a cache whose continuation is bit-identical to an
# uninterrupted run at p: _update_concat trims any longer live buffer to the
# same trailing set on the next chunk, and a decode step rotates both to the
# same write position, so the canonical form is the unique length that is
# correct for both (the L == min(p, max_size), _idx == L invariant).
# Geometry is never inferred: restore requires stored meta and rejects on
# either invariant half, naming which broke.


def rotating_geometry(prompt_cache, block_size: int):
    """(window, keep) when every rotating layer shares one block-aligned
    geometry, else None (exact tier). One distinct (max_size, keep) pair
    across layers; window and keep whole numbers of blocks, so checkpoint
    boundaries are rotation boundaries and the keep region restores as a
    prefix slice of whole blocks."""
    from .cache_compat import cache_types

    rot_types = cache_types("RotatingKVCache")
    geoms = {(int(c.max_size), int(c.keep))
             for c in prompt_cache if isinstance(c, rot_types)}
    if len(geoms) != 1:
        if len(geoms) > 1:
            _log.info("APC rotating: mixed window geometries %s", geoms)
        return None
    (window, keep), = geoms
    if window <= 0 or window % block_size or keep % block_size:
        _log.info(
            "APC rotating: geometry (W=%d keep=%d) not on the %d-token "
            "block grid", window, keep, block_size)
        return None
    return window, keep


def rotating_canonical_window(cache):
    """Canonical temporal snapshot of a live rotating layer.

    Returns ``(keys, values, meta)`` where the arrays cover ``[0, keep)``
    plus the trailing window in temporal order with length
    ``L == min(offset, max_size)``, and ``meta = (keep, max_size, offset,
    L)``. Arrays are lazy views into the live buffer; callers that persist
    them must deep-copy (the store paths already do). Returns None when the
    buffer cannot faithfully produce the canonical form.
    """
    keys, values = cache.keys, cache.values
    if keys is None:
        return None
    keep = int(cache.keep)
    max_size = int(cache.max_size)
    offset = int(cache.offset)
    idx = int(cache._idx)
    buflen = keys.shape[2]
    # Temporal order, mirroring RotatingKVCache._temporal_order's three
    # regimes: contiguous (concat mode), rotated ring, partial in-place fill.
    if idx == buflen:
        tk, tv = keys, values
    elif idx < offset:
        import mlx.core as mx
        tk = mx.concatenate(
            [keys[..., :keep, :], keys[..., idx:, :],
             keys[..., keep:idx, :]], axis=2)
        tv = mx.concatenate(
            [values[..., :keep, :], values[..., idx:, :],
             values[..., keep:idx, :]], axis=2)
    else:
        tk, tv = keys[..., :idx, :], values[..., :idx, :]
    t_len = tk.shape[2]
    t_len = min(t_len, offset)          # concat/partial buffers never exceed
    tk, tv = tk[..., :t_len, :], tv[..., :t_len, :]
    L = min(offset, max_size)
    if t_len < L or L <= keep and offset > keep:
        return None
    if t_len > L:
        import mlx.core as mx
        tk = mx.concatenate(
            [tk[..., :keep, :], tk[..., t_len - (L - keep):, :]], axis=2)
        tv = mx.concatenate(
            [tv[..., :keep, :], tv[..., t_len - (L - keep):, :]], axis=2)
    return tk, tv, (keep, max_size, offset, L)


def rotating_restore(keys, values, meta):
    """Rebuild a RotatingKVCache from a canonical window plus stored meta.

    ``meta`` is ``(keep, max_size, offset, idx)`` and is required: geometry
    is never inferred from buffer shape (a guessed offset or max_size
    silently shifts RoPE positions and shrinks the window). The two
    invariant halves are checked separately so a failure names which broke:
    the assembled length must equal ``min(offset, max_size)`` (coverage),
    and the stored ``idx`` must equal the assembled length (the ring
    pointer is checked against what the chain actually returned, never
    trusted). Returns the cache or None.
    """
    from .cache_compat import construction_cache_module

    if meta is None:
        _log.warning("APC rotating restore rejected: no stored meta")
        return None
    keep, max_size, offset, idx = (int(v) for v in meta)
    L = int(keys.shape[2])
    if L != min(offset, max_size):
        _log.warning(
            "APC rotating restore rejected: assembled length %d != "
            "min(offset=%d, max_size=%d)", L, offset, max_size)
        return None
    if idx != L:
        _log.warning(
            "APC rotating restore rejected: stored _idx %d != assembled "
            "length %d", idx, L)
        return None
    cache = construction_cache_module().RotatingKVCache(
        max_size=max_size, keep=keep)
    cache.keys = keys
    cache.values = values
    cache.offset = offset
    cache._idx = L
    return cache


def rotating_invariant(cache):
    """The two invariant halves for a restored rotating cache, separately:
    ``(L == min(offset, max_size), _idx == L)``."""
    L = 0 if cache.keys is None else int(cache.keys.shape[2])
    return (L == min(int(cache.offset), int(cache.max_size)),
            int(cache._idx) == L)


# Hybrid checkpoint tier, three-way (plan section 3): plain KVCache
# attention layers ride the shared block pool under a salted keyspace
# ([0, b_full) whole blocks); RotatingKVCache layers ride per-checkpoint
# position-salted window chains (the canonical window of section 4 is
# exactly W/B whole blocks at an aligned p, so there is no ring arithmetic
# and no tail); ArraysCache recurrent state (GDN) is the only per-checkpoint
# flat payload. The checkpoint record itself is a gmlx-owned LRU entry that
# holds refcounts on its block chains (pin rather than repair: eviction
# releases them), with a best-effort disk skeleton written through for
# restart repair. At an aligned p the on-disk skeleton's KV entries are
# zero-width placeholders stamped offset=p; only an unaligned GDN store
# (retirement at an arbitrary length) still carries a 1..block_size-1 token
# KV tail. Both keyspaces are salted so these subset chains can never
# satisfy (or be satisfied by) a full-layer store.

_CKPT_SALT = 0x7C_4B_D2_0A_86_E5_93
_BOUNDED_SALT = 0x3A_91_C7_55_0E_D4_26

_CKPT_RECORD_ENTRIES = max(2, env_int("GMLX_APC_CKPT_RECORDS", 32))
# Strip-on-extend: newest N restorable checkpoints per chain (section 7).
_CKPT_HEAVY_PER_CHAIN = max(1, env_int("GMLX_APC_CKPT_HEAVY", 2))


def ckpt_extra_hash(extra_hash: int) -> int:
    return int(extra_hash) ^ _CKPT_SALT


def bounded_extra_hash(extra_hash: int, p: int) -> int:
    """Per-checkpoint salt for a rotating window chain: mixing ``p`` makes
    each checkpoint's window an independent chain, so identical window text
    at different absolute positions can never cross-match, and releasing a
    record releases exactly its own window blocks."""
    return (int(extra_hash) ^ _BOUNDED_SALT) ^ (int(p) * 0x9E3779B1)


def _rot_tag(cache) -> str:
    """Rotating layout tag carries geometry: a stored entry restored into a
    model with a different window must miss, never be inferred around."""
    return (f"rot:{int(cache.max_size)}:"
            f"{int(getattr(cache, 'keep', 0) or 0)}")


def _is_rot(tag: str) -> bool:
    return tag.startswith("rot")


def ckpt_layout(prompt_cache, block_size: int = 16):
    """Per-layer tags ("kv" | "rot:W:keep" | "arr") for a supported hybrid
    cache list, else None. Supported: every layer one of the three classes,
    at least one non-plain-KV layer (pure-KV models belong to the stock
    block tier), and rotating layers passing the block-grid geometry gate."""
    from .cache_compat import cache_types

    if not prompt_cache:
        return None
    kv_types = cache_types("KVCache")
    rot_types = cache_types("RotatingKVCache")
    arr_types = cache_types("ArraysCache")
    tags = []
    for c in prompt_cache:
        if isinstance(c, rot_types):
            tags.append(_rot_tag(c))
        elif isinstance(c, kv_types):
            tags.append("kv")
        elif isinstance(c, arr_types):
            tags.append("arr")
        else:
            return None
    has_rot = any(_is_rot(t) for t in tags)
    if not has_rot and "arr" not in tags:
        return None                       # pure-KV: stock block tier
    if "kv" not in tags and not has_rot:
        return None                       # no chain-backed layer: exact tier
    if has_rot and rotating_geometry(prompt_cache, block_size) is None:
        return None
    return tags


def ckpt_supported(prompt_cache, block_size: int = 16) -> bool:
    """True for cache shapes the checkpoint tier serves (see ckpt_layout)."""
    return ckpt_layout(prompt_cache, block_size) is not None


def _ckpt_block_prefix(p: int, block_size: int) -> int:
    """Tokens covered by whole blocks below a checkpoint at ``p``. At an
    aligned p this equals p: the tail is zero-width and the skeleton's KV
    placeholders carry only the offset stamp (the serializer writes offset
    before the empty check, so this costs nothing on disk)."""
    return (p // block_size) * block_size


class _CkptRecord:
    __slots__ = ("ids", "extra_hash", "p", "b_full", "layout",
                 "main_blocks", "bounded_blocks", "rot_meta", "states",
                 "tails")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _ckpt_records(manager) -> "OrderedDict":
    with manager.lock:
        idx = getattr(manager, "_kq_ckpt_records", None)
        if idx is None:
            idx = OrderedDict()
            manager._kq_ckpt_records = idx
        return idx


def _release_record(manager, rec) -> None:
    for blocks in (rec.main_blocks, rec.bounded_blocks):
        if blocks:
            try:
                manager.release(blocks)
            except Exception:
                _log.debug("APC ckpt record release failed", exc_info=True)
    rec.main_blocks = rec.bounded_blocks = []
    rec.states = rec.tails = None


def _record_insert(manager, rec) -> None:
    """Insert a checkpoint record: strip-on-extend superseded records on the
    same chain immediately (LRU would keep exactly the wrong ones - the
    second-newest on a just-grown chain is among the most recently touched),
    then bound the index, releasing refs on everything dropped."""
    idx = _ckpt_records(manager)
    key = (rec.ids, rec.extra_hash)
    with manager.lock:
        old = idx.pop(key, None)
        if old is not None:
            _release_record(manager, old)
        # chain = records whose ids are a strict prefix of this one
        chain = [k for k, r in idx.items()
                 if r.extra_hash == rec.extra_hash and r.p < rec.p
                 and rec.ids[:r.p] == r.ids]
        chain.sort(key=lambda k: idx[k].p, reverse=True)
        for k in chain[_CKPT_HEAVY_PER_CHAIN - 1:]:
            _release_record(manager, idx.pop(k))
        idx[key] = rec
        idx.move_to_end(key)
        while len(idx) > _CKPT_RECORD_ENTRIES:
            _, victim = idx.popitem(last=False)
            _release_record(manager, victim)


def ckpt_store(
    manager: Any,
    token_ids,
    prompt_cache: list[Any],
    *,
    extra_hash: int = 0,
) -> bool:
    """Store a hybrid checkpoint at ``p = len(token_ids)``.

    Single-row cache list, KV/rotating offsets == p. Plain KV rides the
    salted main chain, rotating layers a per-checkpoint window chain
    (grid-aligned p required), states and unaligned-GDN tails land in the
    pinned record; disk skeleton written best-effort. Never raises.
    """
    if manager is None or token_ids is None:
        return False
    from .cache_compat import cache_types, runtime_cache_module

    kv_types = cache_types("KVCache")
    rot_types = cache_types("RotatingKVCache")
    main_blocks: list[Any] = []
    bounded_blocks: list[Any] = []
    try:
        ids = [int(t) for t in token_ids]
        p = len(ids)
        bs = int(manager.block_size)
        layout = ckpt_layout(prompt_cache, bs)
        if p < 2 or layout is None:
            return False
        for c in prompt_cache:
            off = getattr(c, "offset", None)
            if off is not None and not isinstance(c, rot_types) \
                    and isinstance(c, kv_types) and int(off) != p:
                _log.info("APC ckpt store skipped: KV offset %d != %d",
                          int(off), p)
                return False
            if isinstance(c, rot_types) and int(c.offset) != p:
                _log.info("APC ckpt store skipped: rot offset %d != %d",
                          int(c.offset), p)
                return False
        has_rot = any(_is_rot(t) for t in layout)
        b_full = _ckpt_block_prefix(p, bs)
        if has_rot and b_full != p:
            _log.info(
                "APC ckpt store declined: rotating store needs grid-aligned "
                "p (%d %% %d != 0)", p, bs)
            return False
        tail_len = p - b_full
        salted = ckpt_extra_hash(extra_hash)
        kv_caches = [c for c in prompt_cache if isinstance(c, kv_types)
                     and not isinstance(c, rot_types)]
        rot_caches = [c for c in prompt_cache if isinstance(c, rot_types)]
        arr_caches = [c for c in prompt_cache
                      if not isinstance(c, (kv_types, rot_types))]

        store_blocks = getattr(manager, "store_ckpt_blocks", None)
        if b_full > 0 and kv_caches:
            lk = [c.keys[..., :b_full, :] for c in kv_caches]
            lv = [c.values[..., :b_full, :] for c in kv_caches]
            if store_blocks is not None:
                main_blocks = store_blocks(
                    ids[:b_full], lk, lv, extra_hash=salted)
            else:
                main_blocks = manager.store_kv_blocks(
                    ids[:b_full], lk, lv, extra_hash=salted)
            if len(main_blocks) * bs < b_full:
                # An unpinnable record is a tombstone that displaces
                # restorable ones through strip-on-extend: decline.
                _log.info(
                    "APC ckpt store declined: main chain short (%d/%d "
                    "blocks)", len(main_blocks), b_full // bs)
                manager.release(main_blocks)
                return False

        rot_meta = None
        if rot_caches:
            canon = [rotating_canonical_window(c) for c in rot_caches]
            if any(cw is None for cw in canon):
                manager.release(main_blocks)
                return False
            metas = {cw[2] for cw in canon}
            if len(metas) != 1:
                manager.release(main_blocks)
                return False
            rot_meta = canon[0][2]
            keep, _w, _off, L = rot_meta
            canon_ids = ids[:keep] + ids[p - (L - keep):p]
            bsalt = bounded_extra_hash(extra_hash, p)
            if store_blocks is not None:
                # Memory-only: the position salt makes window shards
                # undedupable on disk.
                bounded_blocks = store_blocks(
                    canon_ids, [cw[0] for cw in canon],
                    [cw[1] for cw in canon], extra_hash=bsalt, disk=False)
            else:
                bounded_blocks = manager.store_kv_blocks(
                    canon_ids, [cw[0] for cw in canon],
                    [cw[1] for cw in canon], extra_hash=bsalt)
            if len(bounded_blocks) * bs < L:
                _log.info(
                    "APC ckpt store: window chain short (%d/%d blocks); "
                    "store declined", len(bounded_blocks), L // bs)
                manager.release(main_blocks)
                manager.release(bounded_blocks)
                return False

        states = [_clone_single_row(c) for c in arr_caches]
        if any(s is None for s in states):
            manager.release(main_blocks)
            manager.release(bounded_blocks)
            return False
        tails = None
        if tail_len and kv_caches:
            tails = []
            for c in kv_caches:
                t = runtime_cache_module().KVCache()
                t.state = (c.keys[..., b_full:p, :],
                           c.values[..., b_full:p, :])
                t = _clone_single_row(t)
                if t is None:
                    manager.release(main_blocks)
                    manager.release(bounded_blocks)
                    return False
                tails.append(t)

        rec = _CkptRecord(
            ids=tuple(ids), extra_hash=int(extra_hash), p=p, b_full=b_full,
            layout=tuple(layout), main_blocks=main_blocks,
            bounded_blocks=bounded_blocks, rot_meta=rot_meta,
            states=states, tails=tails)
        _record_insert(manager, rec)

        _ckpt_disk_write(manager, ids, prompt_cache, layout, p, b_full,
                         tail_len, states, tails, salted)
        _log.info(
            "APC ckpt store: tokens=%d main=%d window=%d tail=%d states=%d",
            p, len(main_blocks), len(bounded_blocks), tail_len, len(states))
        return True
    except Exception:
        try:
            manager.release(main_blocks)
            manager.release(bounded_blocks)
        except Exception:
            pass  # best-effort release on the failure path
        _log.warning("APC ckpt store failed; continuing", exc_info=True)
        return False


def _ckpt_disk_write(manager, ids, prompt_cache, layout, p, b_full,
                     tail_len, states, tails, salted) -> None:
    """Best-effort disk skeleton: placeholders for chain-backed layers
    (offset stamped, rotating meta native), real tails/states inline."""
    disk = getattr(manager, "disk", None)
    if disk is None:
        return
    from .cache_compat import runtime_cache_module
    try:
        rcm = runtime_cache_module()
        entries: list[Any] = []
        kv_i = arr_i = 0
        for tag, c in zip(layout, prompt_cache):
            if tag == "kv":
                if tail_len:
                    entries.append(tails[kv_i])
                else:
                    ph = rcm.KVCache()
                    ph.offset = p
                    entries.append(ph)
                kv_i += 1
            elif _is_rot(tag):
                ph = rcm.RotatingKVCache(
                    max_size=int(c.max_size), keep=int(c.keep))
                ph.offset = p
                ph._idx = min(p, int(c.max_size))
                entries.append(ph)
            else:
                entries.append(states[arr_i])
                arr_i += 1
        # The upstream writer drops zero-array shards; the sentinel keeps
        # an all-placeholder skeleton persistable. Stripped on lookup.
        import mlx.core as mx
        sent = rcm.ArraysCache(size=1)
        sent.cache[0] = mx.array([float(p)], dtype=mx.float32)
        entries.append(sent)
        from mlx_vlm import apc as _apc
        tid = tuple(ids)
        khash = _apc._sequence_hash(tid, salted, manager.block_size)
        disk.save_exact_cache(khash, tid, salted, entries)
    except Exception:
        _log.debug("APC ckpt disk skeleton save failed", exc_info=True)


def _assemble_from_record(manager, rec, geometry_check=None):
    """Warm cache list from a pinned record, or None (conjunction fail)."""
    import mlx.core as mx
    from .cache_compat import runtime_cache_module

    rcm = runtime_cache_module()
    n_kv = sum(1 for t in rec.layout if t == "kv")
    n_rot = sum(1 for t in rec.layout if _is_rot(t))
    if rec.b_full > 0 and n_kv and not rec.main_blocks:
        return None
    if n_rot and not rec.bounded_blocks:
        return None
    layer_kv = None
    if n_kv:
        if rec.b_full > 0:
            if len(rec.main_blocks[0].keys) != n_kv:
                return None
            layer_kv = [
                (mx.concatenate([b.keys[j] for b in rec.main_blocks],
                                axis=2),
                 mx.concatenate([b.values[j] for b in rec.main_blocks],
                                axis=2))
                for j in range(n_kv)
            ]
        if rec.tails:
            tails = [(t.keys[..., :t.offset, :], t.values[..., :t.offset, :])
                     for t in rec.tails]
            if layer_kv is None:
                layer_kv = tails
            else:
                layer_kv = [
                    (mx.concatenate([layer_kv[j][0], tails[j][0]], axis=2),
                     mx.concatenate([layer_kv[j][1], tails[j][1]], axis=2))
                    for j in range(n_kv)
                ]
    layer_rot = None
    if n_rot:
        if len(rec.bounded_blocks[0].keys) != n_rot:
            return None
        keep, w, off, L = rec.rot_meta
        layer_rot = []
        for j in range(n_rot):
            k = mx.concatenate([b.keys[j] for b in rec.bounded_blocks],
                               axis=2)
            v = mx.concatenate([b.values[j] for b in rec.bounded_blocks],
                               axis=2)
            rc = rotating_restore(k, v, (keep, w, rec.p, k.shape[2]))
            if rc is None or k.shape[2] != L:
                return None
            layer_rot.append(rc)
    warm: list[Any] = []
    kv_i = rot_i = arr_i = 0
    for tag in rec.layout:
        if tag == "kv":
            kc = rcm.KVCache()
            kc.state = layer_kv[kv_i]
            warm.append(kc)
            kv_i += 1
        elif _is_rot(tag):
            warm.append(layer_rot[rot_i])
            rot_i += 1
        else:
            clone = _clone_single_row(rec.states[arr_i])
            if clone is None:
                return None
            warm.append(clone)
            arr_i += 1
    targets: list[Any] = []
    for c in warm:
        if getattr(c, "keys", None) is not None:
            targets.extend([c.keys, c.values])
        elif hasattr(c, "cache"):
            targets.extend(s for s in c.cache if s is not None)
    mx.eval(*targets)
    return warm


def ckpt_lookup(
    manager: Any,
    token_ids,
    *,
    extra_hash: int = 0,
    min_prefix_tokens: int = 0,
    layout: tuple | None = None,
) -> tuple:
    """Longest checkpoint-tier warm start for ``token_ids``.

    Walks pinned records p-descending testing the whole conjunction, then
    falls back to the disk skeleton (restart repair). ``layout`` rejects
    records from a different per-layer layout. Returns
    ``(warm_prompt_cache, p)`` or ``(None, 0)``. Never raises.
    """
    if manager is None or token_ids is None:
        return None, 0
    try:
        ids = [int(t) for t in token_ids]
        tid = tuple(ids)
        idx = _ckpt_records(manager)
        with manager.lock:
            cands = [
                rec for rec in idx.values()
                if rec.extra_hash == int(extra_hash)
                and min_prefix_tokens < rec.p < len(ids)
                and tid[:rec.p] == rec.ids
                and (layout is None or tuple(layout) == rec.layout)
            ]
        cands.sort(key=lambda r: r.p, reverse=True)
        for rec in cands:
            warm = _assemble_from_record(manager, rec)
            if warm is not None:
                with manager.lock:
                    if (rec.ids, rec.extra_hash) in idx:
                        idx.move_to_end((rec.ids, rec.extra_hash))
                _log.info("APC ckpt hit: prefix=%d (pinned record)", rec.p)
                return warm, rec.p
            _log.info("APC ckpt walk-back past %d (assembly failed)", rec.p)
        return _ckpt_disk_lookup(
            manager, ids, extra_hash=extra_hash,
            min_prefix_tokens=min_prefix_tokens, layout=layout)
    except Exception:
        _log.warning("APC ckpt lookup failed; continuing", exc_info=True)
        return None, 0


def _ckpt_disk_lookup(manager, ids, *, extra_hash, min_prefix_tokens,
                      layout):
    """Restart-repair path: skeleton entry via the upstream exact
    machinery, chains from the salted block keyspaces. p comes from the
    lookup's matched length, never from a placeholder."""
    import mlx.core as mx
    from .cache_compat import cache_types, runtime_cache_module

    kv_types = cache_types("KVCache")
    rot_types = cache_types("RotatingKVCache")
    salted = ckpt_extra_hash(extra_hash)
    entries, p = manager.lookup_exact_cache(
        ids, extra_hash=salted, min_prefix_tokens=min_prefix_tokens)
    if entries is None or p <= 0:
        return None, 0
    # Strip the skeleton writer's sentinel: exact by length when the live
    # layout is known, by shape otherwise.
    if layout is not None:
        if len(entries) == len(layout) + 1:
            entries = entries[:-1]
    else:
        last = entries[-1] if entries else None
        if (last is not None and not isinstance(last, kv_types + rot_types)
                and hasattr(last, "cache") and len(last.cache) == 1
                and last.cache[0] is not None
                and getattr(last.cache[0], "size", 0) == 1):
            entries = entries[:-1]
    tags = []
    for e in entries:
        if isinstance(e, rot_types):
            tags.append(_rot_tag(e))
        elif isinstance(e, kv_types):
            tags.append("kv")
        else:
            tags.append("arr")
    if layout is not None and tuple(layout) != tuple(tags):
        _log.info("APC ckpt disk miss: layout %s != model %s",
                  tags, list(layout))
        return None, 0
    bs = int(manager.block_size)
    b_full = _ckpt_block_prefix(p, bs)
    n_kv = tags.count("kv")
    blocks = []
    wblocks = []
    try:
        layer_kv = None
        if n_kv and b_full > 0:
            blocks, matched = manager.lookup_prefix(
                ids[:b_full], extra_hash=salted)
            if matched >= b_full and blocks \
                    and len(blocks[0].keys) == n_kv:
                layer_kv = [
                    (mx.concatenate([b.keys[j] for b in blocks], axis=2),
                     mx.concatenate([b.values[j] for b in blocks], axis=2))
                    for j in range(n_kv)
                ]
            else:
                manager.release(blocks)
                blocks = []
                disk_caches, dmatched = manager.lookup_prefix_disk_cache(
                    ids[:b_full], extra_hash=salted,
                    max_prefix_tokens=b_full,
                    min_prefix_tokens=b_full - 1,
                    allow_memory_overlap=True)
                if (disk_caches is None or dmatched < b_full
                        or len(disk_caches) != n_kv):
                    _log.info("APC ckpt disk miss: main chain %d/%d",
                              dmatched if disk_caches else 0, b_full)
                    return None, 0
                layer_kv = [
                    (c.keys[..., :b_full, :], c.values[..., :b_full, :])
                    for c in disk_caches
                ]
        rot_entries = [e for e in entries if isinstance(e, rot_types)]
        layer_rot = None
        if rot_entries:
            keep = int(rot_entries[0].keep)
            w = int(rot_entries[0].max_size)
            L = min(p, w)
            canon_ids = ids[:keep] + ids[p - (L - keep):p]
            bsalt = bounded_extra_hash(extra_hash, p)
            wblocks, wmatched = manager.lookup_prefix(
                canon_ids, extra_hash=bsalt)
            src = None
            if wmatched >= L and wblocks \
                    and len(wblocks[0].keys) == len(rot_entries):
                src = [
                    (mx.concatenate([b.keys[j] for b in wblocks], axis=2),
                     mx.concatenate([b.values[j] for b in wblocks], axis=2))
                    for j in range(len(rot_entries))
                ]
            else:
                manager.release(wblocks)
                wblocks = []
                wcaches, wdm = manager.lookup_prefix_disk_cache(
                    canon_ids, extra_hash=bsalt,
                    max_prefix_tokens=L, min_prefix_tokens=L - 1,
                    allow_memory_overlap=True)
                if wcaches is None or wdm < L \
                        or len(wcaches) != len(rot_entries):
                    _log.info("APC ckpt disk miss: window chain")
                    return None, 0
                src = [(c.keys[..., :L, :], c.values[..., :L, :])
                       for c in wcaches]
            layer_rot = []
            for j in range(len(rot_entries)):
                rc = rotating_restore(
                    src[j][0], src[j][1],
                    (keep, w, p, src[j][0].shape[2]))
                if rc is None:
                    return None, 0
                layer_rot.append(rc)
        warm: list[Any] = []
        kv_i = rot_i = 0
        for tag, e in zip(tags, entries):
            if tag == "kv":
                has_tail = getattr(e, "keys", None) is not None
                t_len = int(e.offset) if has_tail else 0
                kc = runtime_cache_module().KVCache()
                base = layer_kv[kv_i] if layer_kv is not None else None
                if t_len:
                    tk = e.keys[..., :t_len, :]
                    tv = e.values[..., :t_len, :]
                    if base is not None:
                        kc.state = (mx.concatenate([base[0], tk], axis=2),
                                    mx.concatenate([base[1], tv], axis=2))
                    else:
                        kc.state = (tk, tv)
                elif base is not None:
                    kc.state = base
                else:
                    return None, 0
                if kc.offset != p:
                    _log.info("APC ckpt disk miss: kv assembled %d != %d",
                              kc.offset, p)
                    return None, 0
                warm.append(kc)
                kv_i += 1
            elif _is_rot(tag):
                warm.append(layer_rot[rot_i])
                rot_i += 1
            else:
                warm.append(e)
        targets = []
        for c in warm:
            if getattr(c, "keys", None) is not None:
                targets.extend([c.keys, c.values])
            elif hasattr(c, "cache"):
                targets.extend(s for s in c.cache if s is not None)
        mx.eval(*targets)
        manager.release(blocks)
        manager.release(wblocks)
        _log.info("APC ckpt hit: prefix=%d (disk skeleton)", p)
        return warm, p
    except Exception:
        try:
            manager.release(blocks)
            manager.release(wblocks)
        except Exception:
            pass  # best-effort release on the failure path
        _log.warning("APC ckpt disk lookup failed; continuing",
                     exc_info=True)
        return None, 0


def _truncate_kv_snapshot(snap: list[Any], n: int) -> list[Any] | None:
    """Trim a row snapshot to its first ``n`` tokens, or None if any layer
    cannot be truncated faithfully (rotation, recurrent state, pooling)."""
    from .cache_compat import cache_types

    kv_types = cache_types("KVCache")
    for c in snap:
        if not isinstance(c, kv_types) or int(getattr(c, "offset", 0)) < n:
            return None
    for c in snap:
        c.keys = c.keys[..., :n, :]
        c.values = c.values[..., :n, :]
        c.offset = n
    return snap


def retirement_store(
    manager: Any,
    mode: str | None,
    token_ids,
    prompt_cache: list[Any],
    *,
    row: int = 0,
    extra_hash: int = 0,
    max_len: int | None = None,
) -> bool:
    """Persist a finished row's full KV into the shared APC.

    ``token_ids`` is the full sequence (prompt + generated); it must be the
    request's original full_input_ids plus the emitted tokens, never a
    suffix-only serve-layer ``prompt_tokens`` (which is trimmed on a warm turn).
    Exact mode snapshots the row and stores it whole (the rotating chat
    fleet); ckpt mode stores blocks + sidecar (gated-delta hybrids, B=1
    rows only in v1); block mode harvests the row's blocks into the shared
    pool. ``max_len`` caps the stored key at a shorter prefix (the next-turn
    LCP): blocks are prefix-causal and truncate freely; an exact snapshot
    truncates only when every layer is a plain KVCache; a ckpt store is
    skipped outright -- recurrent state cannot rewind, and an entry keyed
    past the replayable prefix can never match. Best-effort: a failure
    never breaks generation.
    """
    if manager is None or token_ids is None:
        return False
    ids = [int(t) for t in token_ids]
    if max_len is not None and max_len < len(ids):
        if mode == "ckpt":
            _log.info(
                "APC retirement skipped: ckpt state at %d cannot rewind to "
                "the replayable prefix %d", len(ids), max_len)
            return False
        if mode != "exact":
            ids = ids[:max_len]
    if len(ids) < 2:
        return False
    try:
        if mode == "ckpt":
            # The row is already single-row on the B=1 path; ckpt_store
            # slices it directly (its own stores copy internally), so no
            # full-cache clone happens -- the exact tier's whole sin.
            if ckpt_store(manager, ids, prompt_cache,
                          extra_hash=extra_hash):
                return True
            # A rotating store declines off the block grid (retirement p
            # is arbitrary; the evicted window front cannot be rewound to
            # a lower grid point). Fall back to the exact tier's verbatim
            # row store, which carries rotation via meta -- today's
            # shipped behavior for the sliding-window fleet. GDN rows
            # never take this branch (unaligned p stores with a KV tail).
            from .cache_compat import cache_types
            if not any(isinstance(c, cache_types("RotatingKVCache"))
                       for c in prompt_cache):
                return False
            mode = "exact"
        if mode == "exact":
            snap = row_snapshot(prompt_cache, row)
            if snap is None:
                return False
            if max_len is not None and max_len < len(ids):
                if max_len < 2:
                    return False
                snap = _truncate_kv_snapshot(snap, max_len)
                if snap is None:
                    _log.info(
                        "APC retirement skipped: snapshot not truncatable "
                        "to the replayable prefix %d", max_len)
                    return False
                ids = ids[:max_len]
            return bool(manager.store_exact_cache(
                ids, snap, extra_hash=extra_hash))
        from mlx_vlm import apc as _apc
        blocks = _apc.harvest_blocks_from_batch_cache(
            manager, prompt_cache, row, ids, extra_hash=extra_hash)
        manager.release(blocks)
        return bool(blocks)
    except Exception:
        _log.warning("APC retirement store failed; continuing", exc_info=True)
        return False
