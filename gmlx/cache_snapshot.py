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
import os
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


def _buffered_types() -> tuple:
    """BufferedRotatingKVCache classes (spec-path swap-in). It subclasses
    RotatingKVCache but carries ``start_position`` and a slack buffer no
    rotating clone or canonical window preserves -- every snapshot path
    declines it loudly until dedicated support lands."""
    from .cache_compat import cache_types

    try:
        return cache_types("BufferedRotatingKVCache")
    except AttributeError:
        return ()


def _clone_single_row(cache: Any) -> Any | None:
    """Deep-copy an already-single-row cache, preserving its concrete kind.

    Reuses upstream's APC clone so the snapshot matches what
    ``store_exact_cache`` would produce for the non-speculative path.
    Rotating layers canonicalize first: min(offset, W) tokens in temporal
    order instead of the untrimmed ring (max_size + prefill_step - 1
    columns) -- every consumer (exact store, decode-ring slot, splice)
    needs only the canonical form, which is the shape the lookup paths
    already restore to.
    """
    from mlx_vlm import apc as _apc
    from .cache_compat import cache_types

    if isinstance(cache, _buffered_types()):
        _log.info("APC clone declined: BufferedRotatingKVCache carries "
                  "start_position no rotating clone preserves")
        return None
    if isinstance(cache, cache_types("RotatingKVCache")):
        return _clone_rot_canonical(cache)
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


def _clone_rot_canonical(cache: Any) -> Any | None:
    """Canonical rotating clone: same concrete class, buffer trimmed to
    the canonical window, ring pointer at its end (the restored form)."""
    import mlx.core as mx
    from mlx_vlm import apc as _apc

    out = type(cache)(max_size=int(cache.max_size),
                      keep=int(getattr(cache, "keep", 0) or 0))
    if cache.keys is None or cache.values is None:
        out.offset = int(getattr(cache, "offset", 0) or 0)
        out._idx = int(getattr(cache, "_idx", 0) or 0)
        return out
    cw = rotating_canonical_window(cache)
    if cw is None:
        _log.info("APC clone declined: rotating buffer has no canonical "
                  "window (offset=%s, idx=%s)",
                  getattr(cache, "offset", None),
                  getattr(cache, "_idx", None))
        return None
    tk, tv, (keep, max_size, offset, length) = cw
    copy = _apc._copy_mlx_array
    out.keys = copy(tk)
    out.values = copy(tv)
    out.offset = int(offset)
    out._idx = int(length)
    mx.eval(out.keys, out.values)
    return out


def _clone_lm_twin(cache: Any, eval_targets: list[Any]) -> Any | None:
    from mlx_vlm import apc as _apc
    from .cache_compat import cache_types

    copy = _apc._copy_mlx_array
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
    1, env_int("GMLX_SPEC_APC_SIDECAR_ENTRIES", 12))
# Deep-context drafter KV is not tiny (a 32k single-layer sidecar runs to
# ~100 MB), so the side index is byte-bounded too; newest always survives.
_SIDECAR_BUDGET_BYTES = max(
    1, env_int("GMLX_SPEC_APC_SIDECAR_BUDGET_MB", 512)) << 20


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
            total = sum(_caches_nbytes(v) for v in idx.values())
            while total > _SIDECAR_BUDGET_BYTES and len(idx) > 1:
                _, victim = idx.popitem(last=False)
                total -= _caches_nbytes(victim)
        disk = getattr(manager, "disk", None)
        if disk is not None:
            try:
                from mlx_vlm import apc as _apc
                salted = sidecar_extra_hash(extra_hash)
                khash = _apc._sequence_hash(ids, salted, manager.block_size)
                disk.save_exact_cache(khash, ids, salted, clones)
                with manager.lock:
                    manager.stats.disk_writes += 1
            except Exception:
                _log.debug("APC sidecar disk save failed", exc_info=True)
        _ckpt_bump(manager, "sidecar_writes")
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


# Exact-tier anchor: one whole-prefix clone per system-prompt chain, held in
# a gmlx-owned side LRU. The upstream exact LRU is count-capped (2 slots by
# default) and every request writes its guard-column entry there, so sibling
# fan-out churns out the early shared-prefix entry the siblings need; this
# index holds nothing but anchors.
_ANCHOR_ENTRIES = max(1, env_int("GMLX_APC_ANCHOR_ENTRIES", 4))
# Whole-prefix clones of pooling/MLA stacks run to GBs at deep prefixes, so
# the index is byte-bounded too; newest always survives.
_ANCHOR_BUDGET_BYTES = max(
    1, env_int("GMLX_APC_ANCHOR_BUDGET_MB", 4096)) << 20


def _anchor_index(manager: Any) -> "OrderedDict":
    with manager.lock:
        idx = getattr(manager, "_kq_anchor_cache", None)
        if idx is None:
            idx = OrderedDict()
            manager._kq_anchor_cache = idx
        return idx


def anchor_exact_store(
    manager: Any,
    token_ids,
    prompt_cache: list,
    extra_hash: int = 0,
) -> bool:
    """Store a whole-prefix clone of ``prompt_cache`` under the anchor key.

    Cloning goes through the upstream exact-clone path (the pooling arms are
    installed there), so any stack the exact tier serves anchors identically.
    A re-store under the same key replaces the entry in place, keeping one
    anchor per chain. Best-effort; never raises.
    """
    if manager is None or not prompt_cache:
        return False
    try:
        ids = tuple(int(t) for t in token_ids)
        if not ids:
            return False
        from mlx_vlm import apc as _apc
        clones = _apc._clone_prompt_cache_for_apc(prompt_cache)
        if clones is None:
            _ckpt_decline(manager, "anchor_clone")
            return False
        nbytes = _caches_nbytes(clones)
        key = (ids, int(extra_hash))
        idx = _anchor_index(manager)
        with manager.lock:
            idx[key] = (clones, nbytes)
            idx.move_to_end(key)
            while len(idx) > _ANCHOR_ENTRIES:
                idx.popitem(last=False)
            total = sum(n for _, n in idx.values())
            while total > _ANCHOR_BUDGET_BYTES and len(idx) > 1:
                _, (_, n) = idx.popitem(last=False)
                total -= n
        _ckpt_bump(manager, "anchor_stores")
        _log.info("APC anchor store: tokens=%d", len(ids))
        return True
    except Exception:
        _log.warning("APC anchor store failed; continuing", exc_info=True)
        return False


def anchor_exact_lookup(
    manager: Any,
    token_ids,
    extra_hash: int = 0,
    min_prefix_tokens: int = 0,
) -> tuple:
    """Longest anchor strictly prefixing ``token_ids`` on the same chain.

    Returns ``(warm_prompt_cache, p)`` or ``(None, 0)``. The warm list is a
    fresh clone with capacity for the full query, decoupled from the stored
    entry. Never raises.
    """
    if manager is None or token_ids is None:
        return None, 0
    try:
        ids = tuple(int(t) for t in token_ids)
        n = len(ids)
        idx = _anchor_index(manager)
        best_key = None
        with manager.lock:
            for key in idx:
                kids, kh = key
                p = len(kids)
                if (kh != int(extra_hash)
                        or not min_prefix_tokens < p < n
                        or ids[:p] != kids):
                    continue
                if best_key is None or p > len(best_key[0]):
                    best_key = key
            if best_key is None:
                return None, 0
            clones = idx[best_key][0]
            idx.move_to_end(best_key)
        from mlx_vlm import apc as _apc
        warm = _apc._clone_prompt_cache_for_apc(
            clones, min_capacity_tokens=n + 1)
        if warm is None:
            return None, 0
        p = len(best_key[0])
        with manager.lock:
            # Anchor hits are cache-served tokens the upstream ledger never
            # sees; bumping both keeps token_hit_rate honest.
            manager.stats.hits += 1
            manager.stats.matched_tokens += p
        _ckpt_bump(manager, "anchor_hits")
        _log.info("APC anchor hit: prefix=%d", p)
        return warm, p
    except Exception:
        _log.warning("APC anchor lookup failed; continuing", exc_info=True)
        return None, 0


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


# Hybrid checkpoint tier, three-way: plain KVCache attention layers ride
# the shared block pool under a salted keyspace ([0, b_full) whole
# blocks); RotatingKVCache layers ride per-checkpoint position-salted
# window chains (the canonical temporal window is exactly W/B whole
# blocks at an aligned p, so there is no ring arithmetic
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
# Strip-on-extend: newest N restorable checkpoints per chain, plus the
# chain's anchor (see _record_insert).
_CKPT_HEAVY_PER_CHAIN = max(1, env_int("GMLX_APC_CKPT_HEAVY", 2))
# Byte budget for record-owned payload (recurrent states + KV tails; chain
# blocks are bounded by the manager pool). A GDN record can carry >100 MB
# of state, so a count bound alone silently pins gigabytes.
_CKPT_BUDGET_BYTES = max(1, env_int("GMLX_APC_CKPT_BUDGET_MB", 4096)) << 20


def _iter_arrays(obj):
    if obj is None:
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_arrays(x)
    elif hasattr(obj, "nbytes"):
        yield obj


def _caches_nbytes(caches) -> int:
    total = 0
    for c in caches or ():
        for a in _iter_arrays(getattr(c, "state", None)):
            total += int(a.nbytes)
    return total


def _rec_nbytes(rec) -> int:
    return _caches_nbytes(rec.states) + _caches_nbytes(rec.tails)


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
    # kind in {"boundary", "anchor", "replay", "retire"}: prefill-cursor
    # boundaries, the chain's pinned early boundary (sibling fan-out
    # reuse), the N-1 identical-replay record, and retirement stores.
    # Retention differs only in _record_insert's strip-on-extend
    # exemptions and eviction order; adoption gates only ever test for
    # "replay".
    __slots__ = ("ids", "extra_hash", "p", "b_full", "layout",
                 "main_blocks", "bounded_blocks", "rot_meta", "states",
                 "tails", "nbytes", "kind")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        if self.kind is None:
            self.kind = "boundary"


def _ckpt_records(manager) -> "OrderedDict":
    with manager.lock:
        idx = getattr(manager, "_kq_ckpt_records", None)
        if idx is None:
            idx = OrderedDict()
            manager._kq_ckpt_records = idx
        return idx


# Ckpt-tier counters live in a gmlx-owned side dict on the manager (same
# pattern as _kq_ckpt_records): the upstream APCStats dataclass stays
# untouched, and GmlxAPCManager's stats_snapshot wrap merges these keys
# into /v1/cache/stats. Keys prefixed "_" are internal (tripwire state),
# never exported.
_CKPT_STAT_INTS = (
    "ckpt_stores", "ckpt_hits", "ckpt_matched_tokens",
    "ckpt_missed_adoptions", "ckpt_skeleton_writes", "sidecar_writes",
    "retire_fallback_full", "retire_fallback_skipped",
    "ckpt_pool_evictions",
    "ckpt_grid_truncate", "anchor_stores", "anchor_hits",
)


def _ckpt_stats(manager) -> dict:
    with manager.lock:
        st = getattr(manager, "_kq_ckpt_stats", None)
        if st is None:
            st = {k: 0 for k in _CKPT_STAT_INTS}
            st["ckpt_declines"] = {}
            manager._kq_ckpt_stats = st
        return st


def _ckpt_bump(manager, key: str, n: int = 1) -> None:
    st = _ckpt_stats(manager)
    with manager.lock:
        st[key] = st.get(key, 0) + n


def _ckpt_decline(manager, reason: str) -> None:
    st = _ckpt_stats(manager)
    with manager.lock:
        d = st["ckpt_declines"]
        d[reason] = d.get(reason, 0) + 1


def ckpt_stats_snapshot(manager) -> dict:
    st = _ckpt_stats(manager)
    with manager.lock:
        out = {k: int(st.get(k, 0)) for k in _CKPT_STAT_INTS}
        out["ckpt_declines"] = dict(st["ckpt_declines"])
        return out


def ckpt_stats_clear(manager) -> None:
    with manager.lock:
        manager._kq_ckpt_stats = None


def ckpt_reset(manager) -> None:
    """Drop ckpt records, sidecars, and counters after a manager.clear().

    The pool clear already zeroed every block's tensors and refcount, so
    the pinned records' chains are gone either way; references are
    dropped, never released (a release into the reset pool would push
    blocks onto the free list twice). The generation bump tells an
    in-flight lookup that pinned a record before the clear to drop its
    held refs the same way."""
    with manager.lock:
        manager._kq_ckpt_gen = int(getattr(manager, "_kq_ckpt_gen", 0)) + 1
        idx = getattr(manager, "_kq_ckpt_records", None)
        if idx:
            for rec in idx.values():
                rec.main_blocks = rec.bounded_blocks = []
                rec.states = rec.tails = None
            idx.clear()
        side = getattr(manager, "_kq_sidecar_cache", None)
        if side:
            side.clear()
        anchors = getattr(manager, "_kq_anchor_cache", None)
        if anchors:
            anchors.clear()
        manager._kq_ckpt_stats = None


def ckpt_note_armed(manager) -> None:
    """Per-request arming tick feeding tripwire 1: a ckpt-armed model
    that keeps completing requests with zero checkpoint stores is
    broken, not idle -- correctness is unaffected by design, so this
    warning is the only runtime signal. Fires once."""
    thresh = env_int("GMLX_APC_CKPT_TRIPWIRE", 5)
    if manager is None or thresh <= 0:
        return
    st = _ckpt_stats(manager)
    with manager.lock:
        st["_armed_requests"] = st.get("_armed_requests", 0) + 1
        armed = st["_armed_requests"]
        if (st.get("_tripwire_stores_fired") or st["ckpt_stores"] > 0
                or armed <= thresh):
            return
        st["_tripwire_stores_fired"] = True
    _log.warning(
        "APC ckpt tripwire: %d requests armed with zero checkpoint "
        "stores -- the cache tier is not working for this model/config "
        "and every request prefills cold (GMLX_APC_CKPT_TRIPWIRE=0 "
        "silences)", armed)


def _ckpt_note_miss(manager, matched_refused) -> None:
    """Missed-adoption accounting, empty-return path only: a record whose
    full stored ids prefix the query was refused (p-bound, layout,
    replay gate, or assembly failure). The candidate walk supplies the
    verdict -- no index rescan on the miss path. This is tripwire 2's
    signal; unlike bare stores-without-hits it never fires on unrelated
    one-shot traffic."""
    if not matched_refused:
        return
    thresh = env_int("GMLX_APC_CKPT_TRIPWIRE", 5)
    st = _ckpt_stats(manager)
    with manager.lock:
        st["ckpt_missed_adoptions"] += 1
        n = st["ckpt_missed_adoptions"]
        if (thresh <= 0 or st.get("_tripwire_adopt_fired")
                or st["ckpt_hits"] > 0 or n < thresh):
            return
        st["_tripwire_adopt_fired"] = True
    _log.warning(
        "APC ckpt tripwire: %d lookups found a prefix-matching record "
        "and adopted nothing (zero hits) -- stores and lookups are not "
        "intersecting for this model/config "
        "(GMLX_APC_CKPT_TRIPWIRE=0 silences)", n)


def ckpt_full_store_redundant(meta) -> bool:
    """Whether the p=N post-prefill store may be dropped: a render-stable
    boundary LANDED (armed is not enough -- the store can decline for
    every reason the counters track, and arm-then-decline must keep p=N).
    With that record live, N-1 covers identical resend, p_stable covers
    turn-2 and branch prefixes, and retirement covers continuation --
    the near-identical adjacent records at N-1/N are exactly what
    strip-on-extend was fighting over."""
    if not meta:
        return False
    stored = meta.get("ckpt_stored_boundaries") or ()
    return any(b in stored for b in meta.get("ckpt_p_stable_bounds") or ())


def _release_record(manager, rec) -> None:
    for blocks in (rec.main_blocks, rec.bounded_blocks):
        if blocks:
            try:
                manager.release(blocks)
            except Exception:
                _log.debug("APC ckpt record release failed", exc_info=True)
    rec.main_blocks = rec.bounded_blocks = []
    rec.states = rec.tails = None


def _free_block_count(manager) -> int:
    n = 0
    b = manager._free_head
    while b is not None:
        n += 1
        b = b.next
    return n


def _evict_for_pool(manager, deficit: int) -> int:
    """Release least-recently-used records until ``deficit`` blocks sit on
    the pool free list, always at least one when the deficit is
    reachable (the caller loops on retry, so each call must make
    progress). Window chains are position-salted (no dedup), so on
    large-window models pinned records exhaust the pool long before the
    entry or byte bounds trip; insert-time eviction then never runs
    again because no store can complete. Blocks a record shares with an
    in-flight lookup pin or another request's chain do not free on
    release, so an unreachable deficit is detected upfront and evicts
    nothing rather than draining the whole index for a store that still
    declines. Plain LRU order on purpose: anchor records get no
    protection here, since an anchor pinning window blocks on an
    exhausted pool would starve every future store. Returns the number
    of records released."""
    if not hasattr(manager, "_free_head"):
        return 0
    idx = _ckpt_records(manager)
    evicted = 0
    with manager.lock:
        # Census stays inside the call: each retry round's evictions
        # change both the ref counts and the deficit, so a hoisted copy
        # would judge reachability from stale state.
        held: dict[int, tuple[int, Any]] = {}
        for rec in idx.values():
            for blocks in (rec.main_blocks, rec.bounded_blocks):
                for b in blocks or ():
                    n, _ = held.get(id(b), (0, b))
                    held[id(b)] = (n + 1, b)
        reclaimable = sum(1 for n, b in held.values() if b.ref_cnt <= n)
        if reclaimable < deficit:
            return 0
        # The caller's shortfall is measured with today's free list already
        # consumed, so the target is growth by ``deficit``, not an
        # absolute free count.
        target = _free_block_count(manager) + deficit
        while idx and (evicted == 0
                       or _free_block_count(manager) < target):
            _, victim = idx.popitem(last=False)
            _release_record(manager, victim)
            evicted += 1
    return evicted


def _evict_lru_record(manager, idx, keep):
    """Release and return the eviction victim: the least-recently-used
    non-anchor record, else the least-recently-used anchor, never the
    ``keep`` key (the record being inserted always survives). Lookup
    hits move_to_end, so anchor order is LRU by last hit: an anchor
    that never serves a sibling ages out, one that does stays hot.
    Caller holds the lock and guarantees a non-keep record exists."""
    key = next((k for k, r in idx.items()
                if r.kind != "anchor" and k != keep), None)
    if key is None:
        key = next(k for k in idx if k != keep)
    victim = idx.pop(key)
    _release_record(manager, victim)
    return victim


def _record_insert(manager, rec) -> None:
    """Insert a checkpoint record: strip-on-extend superseded records on the
    same chain immediately (LRU would keep exactly the wrong ones - the
    second-newest on a just-grown chain is among the most recently touched),
    then bound the index by count and payload bytes, releasing refs on
    everything dropped. The newest record always survives.

    Two strip exemptions. Replay: growth (a longer boundary or
    retirement insert) is exactly the moment an identical resend still
    needs the N-1 record, so a longer non-replay insert never releases
    one; a newer replay on the same chain supersedes it. Anchor: the
    chain's early boundary is what sibling fan-out requests (shared
    system prompt, disjoint user turns) can adopt, and it is exactly
    the record strip-on-extend removes first as the chain deepens. The
    schedule tags the system-prefix stop "anchor"; when no tagged stop
    exists (no render ctx, completions API), the first restorable
    boundary on a fresh chain is promoted instead. One anchor per
    chain: an anchor insert supersedes tagged anchors below it.

    Against real memory pressure the exemptions pin little: the count
    and byte bounds evict anchors after non-anchors (LRU by last hit),
    and pool-pressure eviction (_evict_for_pool) gives anchors no
    protection at all: position-salted window chains must never sit
    pinned on an exhausted pool."""
    idx = _ckpt_records(manager)
    key = (rec.ids, rec.extra_hash)
    rec.nbytes = _rec_nbytes(rec)
    with manager.lock:
        old = idx.pop(key, None)
        if old is not None:
            if old.kind == "anchor" and rec.kind == "boundary":
                rec.kind = "anchor"     # a re-store keeps the anchor tag
            _release_record(manager, old)
        # chain = records whose ids are a strict prefix of this one
        chain = [k for k, r in idx.items()
                 if r.extra_hash == rec.extra_hash and r.p < rec.p
                 and rec.ids[:r.p] == r.ids]
        if rec.kind == "replay":
            # One replay per chain: the newer one supersedes outright.
            for k in [k for k in chain if idx[k].kind == "replay"]:
                _release_record(manager, idx.pop(k))
        elif rec.kind == "anchor":
            for k in [k for k in chain if idx[k].kind == "anchor"]:
                _release_record(manager, idx.pop(k))
        elif rec.kind == "boundary" and not any(
                idx[k].kind != "replay" for k in chain if k in idx):
            rec.kind = "anchor"         # first restorable boundary
        chain = [k for k in chain
                 if k in idx and idx[k].kind not in ("replay", "anchor")]
        chain.sort(key=lambda k: idx[k].p, reverse=True)
        for k in chain[_CKPT_HEAVY_PER_CHAIN - 1:]:
            _release_record(manager, idx.pop(k))
        idx[key] = rec
        idx.move_to_end(key)
        while len(idx) > _CKPT_RECORD_ENTRIES:
            _evict_lru_record(manager, idx, key)
        total = sum(int(getattr(r, "nbytes", 0) or 0) for r in idx.values())
        while total > _CKPT_BUDGET_BYTES and len(idx) > 1:
            victim = _evict_lru_record(manager, idx, key)
            total -= int(getattr(victim, "nbytes", 0) or 0)


def ckpt_store(
    manager: Any,
    token_ids,
    prompt_cache: list[Any],
    *,
    extra_hash: int = 0,
    skeleton_disk: bool = True,
    kind: str = "boundary",
    grid_truncate: bool = False,
) -> int:
    """Store a hybrid checkpoint at ``p = len(token_ids)``.

    Single-row cache list, KV/rotating offsets == p. Plain KV rides the
    salted main chain, rotating layers a per-checkpoint window chain
    (block-grid p below the window, any p at or beyond it), states and
    unaligned-GDN tails land in the pinned record; disk skeleton written
    best-effort. ``skeleton_disk=False`` skips the skeleton write (the
    skeleton inlines recurrent state, >100 MB per GDN checkpoint --
    interval boundaries superseded minutes later do not earn that).
    ``kind`` stamps the record's retention class (see _CkptRecord).
    ``grid_truncate`` turns the below-window off-grid rotating decline
    into a terminal store at the largest block-aligned prefix: pre-wrap
    the buffer is a temporal prefix, so a slice is a faithful shorter
    run. Non-recurrent layouts only (state cannot rewind), memory-only
    (the live cache's offset would stamp a mismatched skeleton).
    Returns the stored length in tokens, 0 when nothing stored. Never
    raises.
    """
    if manager is None or token_ids is None:
        return 0
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
            _ckpt_decline(manager, "layout")
            return 0
        if kind == "replay" and "arr" in layout:
            # The disk path knows no kinds, so a skeleton here would let
            # a restart serve this record past the replay adopt gate --
            # enforce at the tier boundary, not per call site.
            skeleton_disk = False
        # Window chains are position-salted (no dedup across checkpoints),
        # so they earn disk only where restart repair actually reads them:
        # the replay and retirement records. Boundary chains stay
        # memory-only, but their skeletons still land -- within the
        # process a skeleton re-indexes a record whose blocks survived
        # strip-on-extend (the divergent-suffix recovery path); across a
        # restart it misses on the window chain, loudly.
        rot_disk = skeleton_disk and kind in ("replay", "retire")
        if any(isinstance(c, _buffered_types()) for c in prompt_cache):
            _log.info(
                "APC ckpt store declined: BufferedRotatingKVCache rows "
                "cannot snapshot (support deferred)")
            _ckpt_decline(manager, "buffered")
            return 0
        for c in prompt_cache:
            off = getattr(c, "offset", None)
            if off is not None and not isinstance(c, rot_types) \
                    and isinstance(c, kv_types) and int(off) != p:
                _log.info("APC ckpt store skipped: KV offset %d != %d",
                          int(off), p)
                _ckpt_decline(manager, "offset")
                return 0
            if isinstance(c, rot_types) and int(c.offset) != p:
                _log.info("APC ckpt store skipped: rot offset %d != %d",
                          int(c.offset), p)
                _ckpt_decline(manager, "offset")
                return 0
        has_rot = any(_is_rot(t) for t in layout)
        b_full = _ckpt_block_prefix(p, bs)
        trunc_from = None
        if has_rot and b_full != p:
            # Off-grid p is storable once the window has wrapped: the
            # canonical window is then exactly W tokens -- whole blocks
            # regardless of p (rotating_geometry enforces the single
            # block-aligned (W, keep) that ckpt_layout already passed).
            # Below the wrap the window chain would need a partial
            # block, and there is no rot tail mechanism.
            geom = rotating_geometry(prompt_cache, bs)
            if geom is None:
                # ckpt_layout validated the geometry already; reaching
                # here means the cache mutated since. Decline, don't die.
                _log.warning("APC ckpt store declined: rotating geometry "
                             "unavailable at grid gate")
                _ckpt_decline(manager, "layout")
                return 0
            if p < geom[0]:
                has_arr = any(not isinstance(c, (kv_types, rot_types))
                              for c in prompt_cache)
                if not grid_truncate or has_arr or b_full < 2:
                    _log.info(
                        "APC ckpt store declined: off-grid rotating store "
                        "below the window (p=%d < W=%d, %d %% %d != 0)",
                        p, geom[0], p, bs)
                    _ckpt_decline(manager, "grid")
                    return 0
                # Terminal grid store: pre-wrap the buffer is a temporal
                # prefix, so the block-aligned slice is a faithful
                # shorter run. Memory-only: a skeleton would stamp the
                # live cache's deeper offset.
                trunc_from = p
                p = b_full
                ids = ids[:p]
                # No skeleton and no window-chain disk blocks: without
                # the skeleton nothing re-indexes them after a restart.
                skeleton_disk = False
                rot_disk = False
                _ckpt_bump(manager, "ckpt_grid_truncate")
                _log.info(
                    "APC ckpt store: terminal grid store at %d (prompt "
                    "%d below window %d)", p, trunc_from, geom[0])
        tail_len = p - b_full
        salted = ckpt_extra_hash(extra_hash)
        kv_caches = [c for c in prompt_cache if isinstance(c, kv_types)
                     and not isinstance(c, rot_types)]
        rot_caches = [c for c in prompt_cache if isinstance(c, rot_types)]
        arr_caches = [c for c in prompt_cache
                      if not isinstance(c, (kv_types, rot_types))]

        store_blocks = getattr(manager, "store_ckpt_blocks", None)

        def _chained(span_ids, ks, vs, *, extra, disk, need, what):
            # A short chain means the pool ran out of evictable blocks
            # (an unpinnable record is a tombstone that displaces
            # restorable ones through strip-on-extend). Evict records
            # and retry; each round releases at least one record, so
            # the loop is bounded by the record count. A span the pool
            # can never hold declines upfront without sacrificing live
            # records.
            pool = getattr(manager, "pool", None)
            if pool is not None and need > len(pool):
                _log.info(
                    "APC ckpt store declined: %s chain needs %d blocks, "
                    "pool holds %d", what, need, len(pool))
                _ckpt_decline(manager, "short_chain")
                return None

            def _once():
                if store_blocks is not None:
                    return store_blocks(
                        span_ids, ks, vs, extra_hash=extra, disk=disk)
                return manager.store_kv_blocks(
                    span_ids, ks, vs, extra_hash=extra)

            blocks = _once()
            evicted_total = 0
            while len(blocks) < need:
                got = len(blocks)
                manager.release(blocks)
                evicted = _evict_for_pool(manager, need - got)
                if not evicted:
                    _log.info(
                        "APC ckpt store declined: %s chain short (%d/%d "
                        "blocks)", what, got, need)
                    _ckpt_decline(manager, "short_chain")
                    return None
                evicted_total += evicted
                blocks = _once()
            if evicted_total:
                _ckpt_bump(manager, "ckpt_pool_evictions", evicted_total)
                _log.info(
                    "APC ckpt pool pressure: evicted %d record(s) for a "
                    "%s chain store", evicted_total, what)
            return blocks

        if b_full > 0 and kv_caches:
            lk = [c.keys[..., :b_full, :] for c in kv_caches]
            lv = [c.values[..., :b_full, :] for c in kv_caches]
            # Evaluate before storing: the shard payload crosses to the
            # disk writer thread, and evaluating arrays that share
            # unevaluated inputs from two threads is undefined in mlx.
            # Owned survivors: lk/lv slice the live prompt cache, whose
            # own pending graph rides the same tape; the guard drains
            # before the store's except path releases and declines.
            from .eval_guard import guard
            guard.eval(*(lk + lv), site="ckpt-store-main", owner="owned")
            got_main = _chained(
                ids[:b_full], lk, lv, extra=salted, disk=True,
                need=b_full // bs, what="main")
            if got_main is None:
                return 0
            main_blocks = got_main

        rot_meta = None
        if rot_caches:
            canon = [rotating_canonical_window(c) for c in rot_caches]
            if trunc_from is not None:
                # Slice each canonical window to the terminal grid p and
                # restamp its meta as the shorter run's canonical form
                # (pre-wrap: L == offset == p, idx == L).
                canon = [None if cw is None else
                         (cw[0][..., :p, :], cw[1][..., :p, :],
                          (cw[2][0], cw[2][1], p, p))
                         for cw in canon]
            if any(cw is None for cw in canon):
                _ckpt_decline(manager, "canon")
                manager.release(main_blocks)
                return 0
            metas = {cw[2] for cw in canon}
            if len(metas) != 1:
                _ckpt_decline(manager, "canon")
                manager.release(main_blocks)
                return 0
            rot_meta = canon[0][2]
            keep, _w, _off, L = rot_meta
            canon_ids = ids[:keep] + ids[p - (L - keep):p]
            bsalt = bounded_extra_hash(extra_hash, p)
            canon_k = [cw[0] for cw in canon]
            canon_v = [cw[1] for cw in canon]
            # Same writer-thread rule and ownership as the main chain.
            from .eval_guard import guard
            guard.eval(*(canon_k + canon_v), site="ckpt-store-window",
                       owner="owned")
            got_win = _chained(
                canon_ids, canon_k, canon_v, extra=bsalt, disk=rot_disk,
                need=L // bs, what="window")
            if got_win is None:
                manager.release(main_blocks)
                return 0
            bounded_blocks = got_win

        states = [_clone_single_row(c) for c in arr_caches]
        if any(s is None for s in states):
            _ckpt_decline(manager, "clone")
            manager.release(main_blocks)
            manager.release(bounded_blocks)
            return 0
        tails = None
        if tail_len and kv_caches:
            tails = []
            for c in kv_caches:
                t = runtime_cache_module().KVCache()
                t.state = (c.keys[..., b_full:p, :],
                           c.values[..., b_full:p, :])
                t = _clone_single_row(t)
                if t is None:
                    _ckpt_decline(manager, "clone")
                    manager.release(main_blocks)
                    manager.release(bounded_blocks)
                    return 0
                tails.append(t)

        rec = _CkptRecord(
            ids=tuple(ids), extra_hash=int(extra_hash), p=p, b_full=b_full,
            layout=tuple(layout), main_blocks=main_blocks,
            bounded_blocks=bounded_blocks, rot_meta=rot_meta,
            states=states, tails=tails, kind=str(kind))
        _record_insert(manager, rec)
        # Ownership moved to the record; the except path must not release.
        main_blocks = bounded_blocks = []

        if skeleton_disk:
            _ckpt_disk_write(manager, ids, prompt_cache, layout, p, b_full,
                             tail_len, states, tails, salted)
        _ckpt_bump(manager, "ckpt_stores")
        _log.info(
            "APC ckpt store: tokens=%d main=%d window=%d tail=%d states=%d",
            p, len(rec.main_blocks), len(rec.bounded_blocks), tail_len,
            len(states))
        return p
    except Exception:
        try:
            _ckpt_decline(manager, "exception")
            manager.release(main_blocks)
            manager.release(bounded_blocks)
        except Exception:
            pass  # best-effort release on the failure path
        _log.warning("APC ckpt store failed; continuing", exc_info=True)
        return 0


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
        _ckpt_bump(manager, "ckpt_skeleton_writes")
        with manager.lock:
            manager.stats.disk_writes += 1
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
    falls back to the disk skeleton (restart repair). When the deepest
    pinned candidate is the chain's anchor, the disk tier is consulted
    first for anything strictly deeper. ``layout`` rejects records from a
    different per-layer layout. Returns ``(warm_prompt_cache, p)`` or
    ``(None, 0)``. Never raises.
    """
    if manager is None or token_ids is None:
        return None, 0
    try:
        ids = [int(t) for t in token_ids]
        tid = tuple(ids)
        idx = _ckpt_records(manager)
        n = len(ids)
        with manager.lock:
            gen = int(getattr(manager, "_kq_ckpt_gen", 0))
            cands, gated, refused = [], 0, 0
            salted = diverged = 0
            bs = int(getattr(manager, "block_size", 0) or 1)
            for rec in idx.values():
                # Last-token sentinel before the full slice compare:
                # unrelated queries reject in O(1), so tallying refused
                # matches here costs the hot path nothing and the miss
                # path never rescans the index.
                if rec.extra_hash != int(extra_hash):
                    if rec.ids and tid[:bs] == rec.ids[:bs]:
                        salted += 1
                    continue
                if (not rec.ids or rec.p > n
                        or tid[rec.p - 1] != rec.ids[-1]
                        or tid[:rec.p] != rec.ids):
                    # Same-chain records that diverge before their own p
                    # (first block matches, full slice does not) were
                    # invisible: the 9B probe showed adoption dying here
                    # with zero counters moving.
                    if (rec.ids and len(tid) >= bs
                            and tid[:bs] == rec.ids[:bs]):
                        diverged += 1
                        if os.environ.get("GMLX_APC_CKPT_DEBUG"):
                            m = 0
                            lim = min(len(tid), rec.p)
                            while m < lim and tid[m] == rec.ids[m]:
                                m += 1
                            _log.info(
                                "APC ckpt diverged: rec.p=%d first "
                                "mismatch at %d (query %r vs rec %r)",
                                rec.p, m, tid[m:m + 4],
                                rec.ids[m:m + 4])
                    continue
                if (not min_prefix_tokens < rec.p < n
                        or (layout is not None
                            and tuple(layout) != rec.layout)):
                    refused += 1
                    continue
                # Recurrent-state replay records serve identical resends
                # only: at exactly one token past the record, the warm
                # turn forwards a single token and stays bit-identical to
                # armed cold. A longer suffix would chunk off the grid
                # the record was built on and drift. Attention-only
                # replay records split exactly and adopt freely.
                if (rec.kind == "replay" and "arr" in (rec.layout or ())
                        and rec.p != n - 1):
                    gated += 1
                    continue
                cands.append(rec)
        if gated:
            _ckpt_decline(manager, "replay_gate")
        if salted:
            _ckpt_decline(manager, "salt")
        if diverged:
            _ckpt_decline(manager, "ids_diverged")
        cands.sort(key=lambda r: r.p, reverse=True)
        if cands and cands[0].kind == "anchor":
            # The anchor is a retention floor, not a depth ceiling. It
            # sits early on the chain by construction, and the pinned
            # walk below returns on first success, so without this the
            # anchor would cap every divergent query at its own p while
            # a deeper skeleton sits on disk (the position strip-on-
            # extend drops from memory but the skeleton keeps).
            warm, p = _ckpt_disk_lookup(
                manager, ids, extra_hash=extra_hash,
                min_prefix_tokens=max(min_prefix_tokens, cands[0].p),
                layout=layout)
            if warm is not None:
                return warm, p
        for rec in cands:
            # Assembly runs unlocked (it concatenates and evals block
            # tensors), so the record's chains must be pinned against a
            # concurrent _record_insert releasing them mid-assembly: +1
            # ref per block under the lock, dropped after the final eval
            # decouples the warm arrays. Fields are snapshotted under
            # the same lock -- a release rebinds rec.states to None, but
            # cache objects are never pool-recycled, so captured
            # references stay valid.
            with manager.lock:
                if (rec.ids, rec.extra_hash) not in idx:
                    continue          # released since candidate selection
                snap = _CkptRecord(
                    **{k: getattr(rec, k) for k in _CkptRecord.__slots__})
                held = (list(snap.main_blocks or ())
                        + list(snap.bounded_blocks or ()))
                for b in held:
                    manager._acquire_existing(b)
            try:
                warm = _assemble_from_record(manager, snap)
            finally:
                with manager.lock:
                    stale = int(getattr(manager, "_kq_ckpt_gen", 0)) != gen
                    if not stale:
                        manager.release(held)
            if stale:
                # A clear() ran while the record was pinned: the pool
                # free list was rebuilt under the held refs (releasing
                # them would push already-free blocks twice) and the
                # record index died with it. The warm result is
                # pre-clear state -- discard it.
                continue
            if warm is not None:
                with manager.lock:
                    if (rec.ids, rec.extra_hash) in idx:
                        idx.move_to_end((rec.ids, rec.extra_hash))
                    # Memory-record hits are cache-served tokens the
                    # upstream ledger never sees (the ckpt tier bypasses
                    # lookup_exact_cache); bumping both keeps
                    # token_hit_rate honest.
                    manager.stats.hits += 1
                    manager.stats.matched_tokens += rec.p
                _ckpt_bump(manager, "ckpt_hits")
                _ckpt_bump(manager, "ckpt_matched_tokens", rec.p)
                _log.info("APC ckpt hit: prefix=%d (pinned record)", rec.p)
                return warm, rec.p
            _log.info("APC ckpt walk-back past %d (assembly failed)", rec.p)
        warm, p = _ckpt_disk_lookup(
            manager, ids, extra_hash=extra_hash,
            min_prefix_tokens=min_prefix_tokens, layout=layout)
        if warm is not None:
            return warm, p
        _ckpt_note_miss(manager,
                        bool(cands) or bool(gated) or bool(refused))
        return None, 0
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
        # Scratch survivors: the warm-cache arrays were built by this
        # restore from pool blocks; the except path's release-and-decline
        # is the right exit and now runs post-drain.
        from .eval_guard import guard
        guard.eval(*targets, site="ckpt-disk-lookup", owner="scratch")
        manager.release(blocks)
        manager.release(wblocks)
        # ckpt_* only: lookup_exact_cache above already fed the upstream
        # hit/matched ledger for this entry.
        _ckpt_bump(manager, "ckpt_hits")
        _ckpt_bump(manager, "ckpt_matched_tokens", p)
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


# Decode-time checkpoints (the predicted-LCP path). A ckpt-tier entry
# keyed past the next turn's replayable prefix can never match, and
# recurrent state cannot rewind at retirement, so whatever the tier
# retains of the generated tokens must be captured while decode passes
# it. At interval boundaries (anchored to the prompt end and snapped to
# the prefill chunk grid, so the interval is a floor on generated tokens
# retained and a restore replays chunk-exact) the predicted next-turn
# render is compared against the sequence so far: while it still replays
# (LCP within a retokenization margin of the live length), recurrent
# states and rotating windows are cloned into a two-slot ring; the first
# structural divergence (thinking strip, tool-call re-serialization)
# freezes the ring and drops the unreachable slots. Retirement assembles
# a checkpoint from the newest snapshot at or below the actual LCP:
# plain KV truncates freely, everything else comes from the snapshot.
# Rotating windows cannot rewind, so their snapshots are usable only at
# block-aligned positions -- the tick waits for one.

_DECODE_CKPT_DEFAULT = 512
_DECODE_CKPT_MARGIN = 64


def _grid_ceil(n: int, g: int) -> int:
    g = max(1, int(g))
    return -(-int(n) // g) * g


def _cache_offset_max(prompt_cache) -> int:
    p = 0
    for c in prompt_cache or ():
        off = getattr(c, "offset", None)
        if off is not None:
            try:
                p = max(p, int(off))
            except (TypeError, ValueError):
                return 0
    return p


def decode_ckpt_tick(stash: dict, prompt_cache: list[Any],
                     generated: list[int]) -> None:
    """Advance the decode-time snapshot ring for a live B=1 ckpt row.

    ``stash`` is the request's retirement context (``snaps`` /
    ``snap_next`` / ``snap_frozen`` live here and die with it). Cheap
    off-boundary; never raises into the decode loop, and a failure
    disables the ring for the request rather than retrying per token.
    """
    try:
        if stash.get("snap_frozen") or not stash.get("snap_ok"):
            return
        interval = env_int("GMLX_APC_DECODE_CKPT", _DECODE_CKPT_DEFAULT)
        ctx = stash.get("render_ctx")
        if interval <= 0 or ctx is None:
            return
        full_ids = stash["full_ids"]
        p = _cache_offset_max(prompt_cache)
        # The render retokenizes the whole sequence, so its cost grows
        # with p; widening the interval with p keeps prediction a bounded
        # fraction of decode time.
        eff = max(interval, p >> 6)
        grid = int(stash.get("snap_grid") or 1)
        nxt = stash.get("snap_next")
        if nxt is None:
            nxt = _grid_ceil(len(full_ids) + eff, grid)
            stash["snap_next"] = nxt
        if p < nxt:
            return
        align = int(stash.get("snap_align") or 1)
        if align > 1 and p % align:
            # Mirror of the store gate: off-grid rotating stores are
            # valid once the window has wrapped (p >= W); below it the
            # ring still waits for a block-aligned p.
            off_min = int(stash.get("snap_offgrid_min") or 0)
            if off_min <= 0 or p < off_min:
                return
        gen = [int(t) for t in generated]
        seq = list(full_ids) + gen
        if p > len(seq):
            return
        from .retire_key import next_turn_lcp
        pred = next_turn_lcp(ctx, seq, gen, partial=True)
        if pred is None:
            stash["snap_next"] = _grid_ceil(p + eff, grid)
            return
        if pred + _DECODE_CKPT_MARGIN < p:
            # Structural divergence behind us: no future snapshot can sit
            # at or below the final LCP. Keep only slots that still can.
            stash["snaps"] = [s for s in stash.get("snaps") or ()
                              if s[0] <= pred]
            stash["snap_frozen"] = True
            _log.info("APC decode ckpt frozen: render diverges at %d "
                      "(live %d)", pred, p)
            return
        from .cache_compat import cache_types
        kv_types = cache_types("KVCache")
        rot_types = cache_types("RotatingKVCache")
        states = []
        for c in prompt_cache:
            if isinstance(c, kv_types) and not isinstance(c, rot_types):
                continue
            clone = _clone_single_row(c)
            if clone is None:
                stash["snap_ok"] = False
                _log.info("APC decode ckpt disabled: uncloneable cache %s",
                          type(c).__name__)
                return
            states.append(clone)
        snaps = stash.setdefault("snaps", [])
        snaps.append((p, states))
        del snaps[:-2]
        stash["snap_next"] = _grid_ceil(p + eff, grid)
        _log.info("APC decode ckpt snapshot: p=%d pred=%d", p, pred)
    except Exception:
        stash["snap_ok"] = False
        _log.warning("APC decode ckpt tick failed; ring disabled for "
                     "this request", exc_info=True)


def _snap_assemble(prompt_cache: list[Any], states: list[Any],
                   p: int) -> list[Any] | None:
    """Rebuild a single-row cache list at snapshot position ``p``: plain
    KV sliced from the live row, rotating windows and recurrent states
    from the snapshot clones."""
    from .cache_compat import cache_types, construction_cache_module

    kv_types = cache_types("KVCache")
    rot_types = cache_types("RotatingKVCache")
    out = []
    si = 0
    for c in prompt_cache:
        if isinstance(c, kv_types) and not isinstance(c, rot_types):
            if c.keys is None or int(getattr(c, "offset", 0) or 0) < p:
                return None
            k = construction_cache_module().KVCache()
            k.keys = c.keys[..., :p, :]
            k.values = c.values[..., :p, :]
            k.offset = p
            out.append(k)
        else:
            if si >= len(states):
                return None
            out.append(states[si])
            si += 1
    if si != len(states):
        return None
    return out


def _ckpt_retirement(manager, ids, prompt_cache, *, extra_hash,
                     max_len, decode_snaps) -> int:
    """Ckpt-mode retirement: the full sequence when it can store whole
    (a short rotating prompt truncates to the block grid), else the
    newest decode-time snapshot at or below the replayable prefix. Never spills to the exact tier -- on ckpt models the exact
    tier stays empty, so the stock warm path never bypasses arming.
    Returns the stored length (0 = nothing)."""
    try:
        cap = len(ids)
        if max_len is not None and max_len < len(ids):
            cap = max_len
        elif len(ids) >= 2:
            # The row is already single-row on the B=1 path; ckpt_store
            # slices it directly (its own stores copy internally), so no
            # full-cache clone happens -- the exact tier's whole sin. A
            # rotating layer below the window stores its block-grid
            # prefix (grid_truncate); the ring below holds aligned
            # clones for the post-wrap off-grid cases.
            stored = ckpt_store(manager, ids, prompt_cache,
                                extra_hash=extra_hash,
                                grid_truncate=True, kind="retire")
            if stored:
                return stored
        for p, states in sorted(decode_snaps or (),
                                key=lambda s: s[0], reverse=True):
            if not 2 <= p <= cap:
                continue
            assembled = _snap_assemble(prompt_cache, states, p)
            # skeleton_disk stays on: this is the entry a post-restart
            # turn restores from, unlike interval boundaries.
            if assembled is not None and ckpt_store(
                    manager, ids[:p], assembled, extra_hash=extra_hash,
                    skeleton_disk=True, kind="retire"):
                _log.info("APC retirement: decode ckpt stored at %d "
                          "(cap %d, full %d)", p, cap, len(ids))
                return p
        # No snapshot at or below the replayable prefix: fall back to the
        # full sequence under its verbatim key rather than storing
        # nothing. A re-rendered turn will not adopt it, but a raw
        # continuation client replays prompt+gen exactly and does.
        # Reason-counted so fallback traffic is distinguishable from
        # turn reuse. Only when the whole-sequence branch above did not
        # already try (cap == len means it ran and declined).
        if cap < len(ids) and len(ids) >= 2:
            # The fallback serves raw-continuation clients only (a
            # re-rendered turn diverges at cap). On a pressured pool it
            # evicts adoptable prefixes to store unmatchable chains --
            # the 122B cert churn spiral -- so it yields at 90 percent
            # occupancy.
            used = int(getattr(getattr(manager, "stats", None),
                               "pool_used", 0) or 0)
            total = int(getattr(manager, "num_blocks", 0) or 0)
            if total and used * 10 >= total * 9:
                _ckpt_bump(manager, "retire_fallback_skipped")
                _log.info("APC retirement: fallback skipped, pool "
                          "%d/%d (replayable prefix %d)",
                          used, total, cap)
                return 0
            stored = ckpt_store(manager, ids, prompt_cache,
                                extra_hash=extra_hash,
                                grid_truncate=True, kind="retire")
            if stored:
                _ckpt_bump(manager, "retire_fallback_full")
                _log.info("APC retirement: full-sequence fallback stored "
                          "at %d (replayable prefix %d)", stored, cap)
                return stored
        _log.info("APC retirement skipped: no decode snapshot at or "
                  "below the replayable prefix %d (full %d)",
                  cap, len(ids))
        return 0
    except Exception:
        _log.warning("APC ckpt retirement failed; continuing",
                     exc_info=True)
        return 0


def retirement_store(
    manager: Any,
    mode: str | None,
    token_ids,
    prompt_cache: list[Any],
    *,
    row: int = 0,
    extra_hash: int = 0,
    max_len: int | None = None,
    decode_snaps: list | None = None,
) -> int:
    """Persist a finished row's full KV into the shared APC.

    ``token_ids`` is the full sequence (prompt + generated); it must be the
    request's original full_input_ids plus the emitted tokens, never a
    suffix-only serve-layer ``prompt_tokens`` (which is trimmed on a warm turn).
    Exact mode snapshots the row and stores it whole (plain-KV fleets);
    ckpt mode stores blocks + sidecar (gated-delta and sliding-window
    hybrids, B=1 rows only in v1); block mode harvests the row's blocks
    into the shared pool. ``max_len`` caps the stored key at a shorter
    prefix (the next-turn LCP): blocks are prefix-causal and truncate
    freely; an exact snapshot truncates only when every layer is a plain
    KVCache; a ckpt store uses the newest decode-time snapshot at or
    below the LCP (``decode_snaps``, from ``decode_ckpt_tick``), falling
    back to the full sequence under its verbatim key -- neither recurrent
    state nor a rotating window can rewind, and nothing spills to the
    exact tier. Best effort: a failure never breaks generation.

    Returns the stored prefix length in tokens (0 = nothing stored) --
    the fallback can store past ``max_len``, so callers must log the
    return, not the cap.
    """
    if manager is None or token_ids is None:
        return 0
    ids = [int(t) for t in token_ids]
    if mode == "ckpt":
        return _ckpt_retirement(manager, ids, prompt_cache,
                                extra_hash=extra_hash, max_len=max_len,
                                decode_snaps=decode_snaps)
    if max_len is not None and max_len < len(ids) and mode != "exact":
        ids = ids[:max_len]
    if len(ids) < 2:
        return 0
    try:
        if mode == "exact":
            snap = row_snapshot(prompt_cache, row)
            if snap is None:
                return 0
            if max_len is not None and max_len < len(ids):
                if max_len < 2:
                    return 0
                snap = _truncate_kv_snapshot(snap, max_len)
                if snap is None:
                    _log.info(
                        "APC retirement skipped: snapshot not truncatable "
                        "to the replayable prefix %d", max_len)
                    return 0
                ids = ids[:max_len]
            if manager.store_exact_cache(ids, snap, extra_hash=extra_hash):
                return len(ids)
            return 0
        from mlx_vlm import apc as _apc
        blocks = _apc.harvest_blocks_from_batch_cache(
            manager, prompt_cache, row, ids, extra_hash=extra_hash)
        manager.release(blocks)
        if not blocks:
            return 0
        bs = int(getattr(manager, "block_size", 0) or 0)
        return len(blocks) * bs if bs else len(ids)
    except Exception:
        _log.warning("APC retirement store failed; continuing", exc_info=True)
        return 0
