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

import itertools
import logging
import math
import os
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from gmlx.envflags import env_int

from .media_salt import as_salt, salt_at

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
    if hasattr(snap, "accumulate_windows"):  # DS4PoolingCache: no `keys`;
        # content lives in pooled rows and/or the staging remainder
        return snap.size() > 0 or getattr(snap, "remainder", 0) > 0
    if getattr(snap, "kv_quant_scheme", None) == "kvarn":  # no `keys` either
        return snap.size() > 0
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
    from .kv_cache import BufferedRotatingKVCache

    return (BufferedRotatingKVCache,)


def _clone_single_row(cache: Any, eval_targets: list | None = None) -> Any | None:
    """Deep-copy an already-single-row cache, preserving its concrete kind.

    Rotating layers canonicalize: min(offset, W) tokens in temporal order
    instead of the untrimmed ring. For CROSS-REQUEST store consumers
    (ckpt-tier chained blocks and their decode snapshots) that is the
    required form -- the chain key is content-addressed over the canonical
    window and a future request can never reproduce this stream's ring
    phase. Same-stream resumes must use ``_clone_row_faithful`` instead.

    With ``eval_targets``, the copies go into that list and the caller
    evaluates them, so a many-layer clone syncs once and a store can make
    room before its copy materializes.
    """
    return _clone_row(cache, rot_canonical=True, eval_targets=eval_targets)


def _held_state(cache: Any, eval_targets: list) -> Any:
    """An inline-state layer for a record that holds the live arrays rather
    than copies. A chunk replaces these arrays and never writes them, so
    they stay as they were at the store. mx.contiguous copies only a view
    of a larger buffer, which a record must not pin."""
    import mlx.core as mx

    held = [None if a is None else mx.contiguous(a) for a in cache.state]
    eval_targets.extend(a for a in held if a is not None)
    return type(cache).from_state(held, cache.meta_state)


def _clone_row_faithful(cache: Any,
                        eval_targets: list | None = None) -> Any | None:
    """Deep-copy a single-row cache bitwise, rotating rings included.

    For SAME-STREAM consumers (exact retirement, the rung ring, drafter
    KV sidecars): the restored state must replay bitwise against the live
    row. The canonical rotating reorder holds the same window but permutes
    fp16 summation order inside it, which MoE routing amplifies into
    visible logit swings. ``eval_targets`` as in ``_clone_single_row``.
    """
    return _clone_row(cache, rot_canonical=False, eval_targets=eval_targets)


def _clone_row(cache: Any, *, rot_canonical: bool,
               eval_targets: list | None = None) -> Any | None:
    from gmlx._vendor.mlx_vlm import apc as _apc

    from .kv_cache import RotatingKVCache

    if isinstance(cache, _buffered_types()):
        return _clone_buffered_window(cache, eval_targets)
    if rot_canonical and isinstance(cache, RotatingKVCache):
        return _clone_rot_canonical(cache, eval_targets)
    targets: list[Any] = [] if eval_targets is None else eval_targets
    out = _apc._clone_cache_entry_for_apc(
        cache, min_capacity_tokens=None, eval_targets=targets)
    if eval_targets is None and targets:
        import mlx.core as mx
        mx.eval(*targets)
    return out


def _clone_rot_canonical(cache: Any,
                         eval_targets: list | None = None) -> Any | None:
    """Canonical rotating clone: same concrete class, buffer trimmed to
    the canonical window, ring pointer at its end (the restored form)."""
    from gmlx._vendor.mlx_vlm import apc as _apc

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
    _eval_or_defer([out.keys, out.values], eval_targets)
    return out


def _clone_buffered_window(cache: Any,
                           eval_targets: list | None = None) -> Any | None:
    """Clone of a BufferedRotatingKVCache (the spec-path swap-in) as the
    plain RotatingKVCache the model's make_cache produces.

    The buffered cache keeps its window as a linear temporal buffer
    ``[0, _idx)`` plus rollback slack, so the canonical window (the
    trailing ``min(offset, max_size)`` tokens in temporal order) is also
    its faithful state: no ring phase to permute, hence one clone serves
    both consumers. The result is what a stored entry restores into (the
    spec path re-swaps on its next prefill). ``keep > 0`` never reaches
    the swap; declined."""
    from gmlx._vendor.mlx_vlm import apc as _apc

    from .kv_cache import RotatingKVCache

    keep = int(getattr(cache, "keep", 0) or 0)
    if keep:
        _log.info("APC clone declined: BufferedRotatingKVCache with keep=%d",
                  keep)
        return None
    max_size = int(cache.max_size)
    offset = int(getattr(cache, "offset", 0) or 0)
    out = RotatingKVCache(max_size=max_size, keep=0)
    out.offset = offset
    if cache.keys is None or cache.values is None:
        out._idx = 0
        return out
    idx = int(cache._idx)
    L = min(offset, max_size)
    if idx < L or L <= 0:
        _log.info("APC clone declined: buffered window covers %d of %d "
                  "tokens", idx, L)
        return None
    # The trailing slice is the canonical window only while the buffer's
    # linear layout holds (``start_position == offset - _idx``, which
    # from_cache establishes and compact/update preserve). Decline loudly
    # rather than store a window that starts somewhere else.
    start = getattr(cache, "start_position", None)
    if start is not None and int(start) != offset - idx:
        _log.warning("APC clone declined: buffered window start %s != "
                     "offset %d - idx %d", start, offset, idx)
        return None
    copy = _apc._copy_mlx_array
    out.keys = copy(cache.keys[..., idx - L:idx, :])
    out.values = copy(cache.values[..., idx - L:idx, :])
    out._idx = L
    _eval_or_defer([out.keys, out.values], eval_targets)
    return out


def _eval_or_defer(arrays: list, eval_targets: list | None) -> None:
    """Evaluate ``arrays`` now, or hand them to the caller's list."""
    if eval_targets is None:
        import mlx.core as mx
        mx.eval(*arrays)
    else:
        eval_targets.extend(arrays)


def row_snapshot(prompt_cache: list[Any], row: int = 0) -> list[Any] | None:
    """Extract ``row`` from a prompt cache as a list of single-row caches.

    Returns caches that ``store_exact_cache`` can persist and ``merge``/``extend``
    can splice back into a live batch (the round-trip inverse of ``merge``).
    Returns None when any layer lacks content for this row -- an incomplete
    snapshot must never be stored under a full-sequence key. Rotating (sliding
    window) layers are snapshot-able even after they wrap: the window they retain
    is exactly what sliding-window attention re-attends to on continuation.
    """
    return _row_snapshot(prompt_cache, row, lazy=False)


def _row_view(cache: Any):
    """The cache's ``_row_view`` when its ``extract`` is the copy of that
    view (both defined on the same class), else None."""
    for klass in type(cache).__mro__:
        if "extract" in klass.__dict__:
            if "_row_view" in klass.__dict__:
                return cache._row_view
            return None
    return None


def _row_snapshot(prompt_cache: list[Any], row: int, *,
                  lazy: bool) -> list[Any] | None:
    """``row_snapshot``; with ``lazy`` nothing is copied yet, for a caller
    that copies the snapshot once and drops it before that copy
    materializes. Plain KV layers then give views of the live row, and
    the other layers their extract or clone unevaluated."""
    if not prompt_cache:
        return None
    snaps: list[Any] = []
    for cache in prompt_cache:
        extract = getattr(cache, "extract", None)
        if lazy and callable(extract):
            extract = _row_view(cache) or extract
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
                snap = _clone_row_faithful(cache, [] if lazy else None)
        else:
            # Already single-row (B=1 without a batch wrapper); clone so the
            # stored copy is decoupled from the live decode cache.
            snap = _clone_row_faithful(cache, [] if lazy else None)
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


def _side_room(idx: "OrderedDict", key: Any, nbytes: int, entries: int,
               budget: int, size_of) -> None:
    """Make room in a side index (drafter sidecars, anchors) for one entry
    of ``nbytes`` under ``key`` before its copy materializes: drop ``key``,
    then the least recently used entries until the new one fits
    ``entries`` and ``budget``. The newest always survives, so an entry
    over the whole budget empties the index and is still kept. The entries
    kept are the ones insert-then-trim kept. Caller holds the lock."""
    idx.pop(key, None)
    total = sum(size_of(v) for v in idx.values())
    while idx and (len(idx) + 1 > entries or total + nbytes > budget):
        _, victim = idx.popitem(last=False)
        total -= size_of(victim)


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
    extra_hash = int(salt_at(extra_hash, store_len))
    try:
        caches = drafter.export_kv()
        if not caches:
            return False
        # The clones stay lazy until the index has made room for them.
        clones: list[Any] = []
        pending: list[Any] = []
        for c in caches:
            offset = int(getattr(c, "offset", 0) or 0)
            if offset < store_len:
                _log.info(
                    "APC sidecar store skipped: head offset %d < %d",
                    offset, store_len)
                return False
            clone = _clone_row_faithful(c, pending)
            if clone is None:
                return False
            excess = offset - store_len
            if excess and int(clone.trim(excess) or 0) != excess:
                # kvarn trim returns 0 on refusal without raising; an
                # untrimmed clone must not be stored under a shorter key.
                _log.info(
                    "APC sidecar store skipped: trim refused %d rows",
                    excess)
                return False
            clones.append(clone)
        ids = tuple(int(t) for t in token_ids)[:store_len]
        key = (ids, int(extra_hash))
        idx = _sidecar_index(manager)
        record = getattr(manager, "kq_record", None)
        with manager.lock:
            _side_room(idx, key, _caches_nbytes(clones), _SIDECAR_ENTRIES,
                       _SIDECAR_BUDGET_BYTES, _caches_nbytes)
        if pending:
            from gmlx.eval_guard import guard
            guard.eval(*pending, site="sidecar-store", owner="owned")
        with manager.lock:
            idx[key] = clones
            if record is not None:
                record("sidecar", key)
            idx.move_to_end(key)
        disk = getattr(manager, "disk", None)
        if disk is not None:
            try:
                from gmlx._vendor.mlx_vlm import apc as _apc
                # Both sidecar sides bypass the manager wrappers, so the
                # wire salt is mixed here and in the lookup twin -- a
                # drafter cache written under one KV wire config must not
                # restore under another.
                salted = sidecar_extra_hash(extra_hash) ^ int(
                    getattr(manager, "_exact_extra_salt", 0) or 0)
                khash = _apc._sequence_hash(ids, extra_hash=salted,
                                            block_size=manager.block_size)
                # The disk writer counts disk_writes when the write lands.
                disk.save_exact_cache(khash, ids, salted, clones)
                if record is not None:
                    record("disk", khash)
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
    extra_hash = int(salt_at(extra_hash, prefix_len))
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
            return [_clone_row_faithful(c) for c in entry]
        disk = getattr(manager, "disk", None)
        if disk is None:
            return None
        salted = sidecar_extra_hash(extra_hash) ^ int(
            getattr(manager, "_exact_extra_salt", 0) or 0)
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
        # The load serves this restore either way; the index keeps it
        # within both of its bounds.
        with manager.lock:
            _side_room(idx, key, _caches_nbytes(caches), _SIDECAR_ENTRIES,
                       _SIDECAR_BUDGET_BYTES, _caches_nbytes)
            idx[key] = caches
        return [_clone_row_faithful(c) for c in caches]
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
        from gmlx._vendor.mlx_vlm import apc as _apc
        # The upstream exact clone, with its one eval held back until the
        # index has made room for the copy.
        pending: list[Any] = []
        clones = []
        for c in prompt_cache:
            copied = _apc._clone_cache_entry_for_apc(
                c, min_capacity_tokens=None, eval_targets=pending)
            if copied is None:
                _ckpt_decline(manager, "anchor_clone")
                return False
            clones.append(copied)
        nbytes = _caches_nbytes(clones)
        key = (ids, int(salt_at(extra_hash, len(ids))))
        idx = _anchor_index(manager)
        with manager.lock:
            _side_room(idx, key, nbytes, _ANCHOR_ENTRIES,
                       _ANCHOR_BUDGET_BYTES, lambda v: v[1])
        if pending:
            from gmlx.eval_guard import guard
            guard.eval(*pending, site="anchor-store", owner="owned")
        with manager.lock:
            idx[key] = (clones, nbytes)
            idx.move_to_end(key)
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
                if (kh != int(salt_at(extra_hash, p))
                        or not min_prefix_tokens < p < n
                        or ids[:p] != kids):
                    continue
                if best_key is None or p > len(best_key[0]):
                    best_key = key
            if best_key is None:
                return None, 0
            clones = idx[best_key][0]
            idx.move_to_end(best_key)
        from gmlx._vendor.mlx_vlm import apc as _apc
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
    from .kv_cache import RotatingKVCache

    geoms = {(int(c.max_size), int(c.keep))
             for c in prompt_cache if isinstance(c, RotatingKVCache)}
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
    from .kv_cache import RotatingKVCache

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
    cache = RotatingKVCache(
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
_QSA_SKEL_SALT = 0x2E_B4_19_C6_57_A0_D3

_CKPT_RECORD_ENTRIES = max(2, env_int("GMLX_APC_CKPT_RECORDS", 32))
# Strip-on-extend: newest N restorable checkpoints per chain, plus the
# chain's anchor (see _record_insert).
_CKPT_HEAVY_PER_CHAIN = max(1, env_int("GMLX_APC_CKPT_HEAVY", 2))
# Decision records a manager keeps, so a stream of decisions on new states
# cannot push the chat records out.
_CKPT_DECISION_ENTRIES = 8
# Records-only byte budget for record-owned payload (recurrent states, KV
# tails, indexer keys). A GDN record can carry >100 MB of state, so a count
# bound alone silently pins gigabytes. It bounds the records of a manager
# autosize never sized; on an autosized one the records share the pool's
# room with its blocks (_record_room), unless GMLX_APC_CKPT_BUDGET_MB sets
# this budget in its place.
_CKPT_BUDGET_BYTES = max(1, env_int("GMLX_APC_CKPT_BUDGET_MB", 4096)) << 20
_CKPT_BUDGET_SET = bool(os.environ.get("GMLX_APC_CKPT_BUDGET_MB", "").strip())


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
    total = _caches_nbytes(rec.states) + _caches_nbytes(rec.tails)
    for a in _iter_arrays(getattr(rec, "qsa", None)):
        total += int(a.nbytes)
    return total


def ckpt_extra_hash(extra_hash: int) -> int:
    # A MediaSalt stays one: the disk lookup resolves it per window.
    return as_salt(extra_hash) ^ _CKPT_SALT


def bounded_extra_hash(extra_hash: int, p: int) -> int:
    """Per-checkpoint salt for a rotating window chain: mixing ``p`` makes
    each checkpoint's window an independent chain, so identical window text
    at different absolute positions can never cross-match, and releasing a
    record releases exactly its own window blocks."""
    return ((int(salt_at(extra_hash, p)) ^ _BOUNDED_SALT)
            ^ (int(p) * 0x9E3779B1))


def _rot_tag(cache) -> str:
    """Rotating layout tag carries geometry: a stored entry restored into a
    model with a different window must miss, never be inferred around."""
    return (f"rot:{int(cache.max_size)}:"
            f"{int(getattr(cache, 'keep', 0) or 0)}")


def _is_rot(tag: str) -> bool:
    return tag.startswith("rot")


def _is_qsa(tag: str) -> bool:
    return tag.startswith("qsa")


def _skeleton_salt(salted, layout, block_size: int):
    """Disk skeleton salt. A QSA skeleton's tail starts at its writer's
    block boundary and the disk namespace has no block size, so QSA
    skeletons key on it: one written at another block size, or by a release
    without the b_full sentinel, is unreachable rather than loaded and
    refused."""
    if layout and any(_is_qsa(t) for t in layout):
        return salted ^ (_QSA_SKEL_SALT + int(block_size))
    return salted


def _is_chain(tag: str) -> bool:
    """Chain-backed attention layer: k/v ride the shared block pool. Kvarn
    records are blockless, so a kvarn tag is never chain-backed."""
    return tag == "kv" or _is_qsa(tag)


def _qsa_tag(cache) -> str | None:
    """QSA layers (KVCache plus the indexer key stream) get their own tag:
    treating them as plain "kv" restores a cache without ``ik``, which the
    model silently serves with dense attention from then on."""
    if hasattr(cache, "update_and_fetch_qsa"):
        return f"qsa:{int(cache.ratio)}"
    return None


def _qsa_cache_cls():
    from gmlx.models.qwen4_exp.model import QSAKVCache
    return QSAKVCache


def kvarn_layout_tag(k_bits: int, v_bits: int, tail: int) -> str:
    """Kvarn layout tag carries the wire config: a record restored into a
    different width/tail (or a stock boot) must miss, never adopt."""
    return f"kvarn:{int(k_bits)}:{int(v_bits)}:{int(tail)}"


def _kvarn_tag(cache) -> str:
    return kvarn_layout_tag(cache.k_bits, cache.v_bits, cache.tail_cap)


def _is_kvarn(tag: str) -> bool:
    return tag.startswith("kvarn")


def ckpt_layout(prompt_cache, block_size: int = 16):
    """Per-layer tags ("kv" | "qsa:R" | "rot:W:keep" | "arr" |
    "kvarn:k:v:tail") for a supported hybrid cache list, else None.
    Supported: every layer one of the classes, at least one non-plain-KV
    layer (pure-KV and pure-kvarn models belong to the block/exact tiers),
    and rotating layers passing the block-grid geometry gate. Kvarn matches
    the concrete single-stream class only: the rotating subclass loses
    window geometry under the plain tag, and batch classes belong to the
    batch engine -- both decline until dedicated support lands."""
    from .kv_cache import ArraysCache, KVCache, RotatingKVCache
    from .kvarn_cache import KVarNKVCache

    if not prompt_cache:
        return None
    tags = []
    for c in prompt_cache:
        qsa = _qsa_tag(c)
        if isinstance(c, RotatingKVCache):
            tags.append(_rot_tag(c))
        elif qsa is not None:
            tags.append(qsa)
        elif isinstance(c, KVCache):
            tags.append("kv")
        elif isinstance(c, ArraysCache):
            tags.append("arr")
        elif type(c) is KVarNKVCache:
            tags.append(_kvarn_tag(c))
        else:
            return None
    has_rot = any(_is_rot(t) for t in tags)
    has_kvarn = any(_is_kvarn(t) for t in tags)
    if not has_rot and "arr" not in tags:
        return None       # pure-KV/pure-kvarn: block/exact tier
    if not any(_is_chain(t) for t in tags) and not has_rot and not has_kvarn:
        # No chain-backed layer: exact tier. A chainless kvarn+arr record
        # still earns the ckpt tier -- B=1 direct-assign adoption (the
        # exact tier's merge machinery returns None on mixed stacks), the
        # checkpoint boundary schedule, and disk skeletons.
        return None
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


_CKPT_KINDS = ("boundary", "anchor", "replay", "retire", "decision")


class _CkptRecord:
    # kind in _CKPT_KINDS: prefill-cursor boundaries, the chain's pinned
    # early boundary (sibling fan-out reuse), the N-1 identical-replay
    # record, retirement stores, and the state prefixes of letter
    # decisions (gmlx/systemone/prefixes.py). Retention differs only in
    # _record_insert's strip-on-extend rules and eviction order;
    # adoption gates only ever test for "replay". spared: a record
    # strip-on-extend kept for the next turn (see _record_insert).
    # stood_in: a replay that stands in beside its prompt's render-stable
    # boundary, a replay again once a record past it lands.
    __slots__ = ("ids", "extra_hash", "p", "b_full", "layout",
                 "main_blocks", "bounded_blocks", "rot_meta", "states",
                 "tails", "qsa", "nbytes", "kind", "spared", "stood_in")

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
    "retire_fallback_suppressed",
    "ckpt_pool_evictions", "ckpt_governor_released",
    "ckpt_evict_own_chain", "ckpt_budget_evictions", "ckpt_budget_evicted_mb",
    "ckpt_grid_truncate", "anchor_stores", "anchor_hits",
)


def _ckpt_stats(manager) -> dict:
    with manager.lock:
        st = getattr(manager, "_kq_ckpt_stats", None)
        if st is None:
            fresh: dict[str, Any] = {k: 0 for k in _CKPT_STAT_INTS}
            fresh["ckpt_declines"] = {}
            manager._kq_ckpt_stats = st = fresh
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
        out: dict[str, Any] = {k: int(st.get(k, 0)) for k in _CKPT_STAT_INTS}
        out["ckpt_declines"] = dict(st["ckpt_declines"])
        return out


def ckpt_stats_clear(manager) -> None:
    with manager.lock:
        manager._kq_ckpt_stats = None


def ckpt_reset(manager) -> None:
    """Drop ckpt records, sidecars, and counters, as manager.clear() does
    before it wipes the pool. Each record gives its chains back
    (``_release_record``), so the pool frees what only records held and
    leaves what a lookup or a store still leases to that holder. The
    generation bump tells a store in flight to insert no record and a
    lookup in flight to discard what it assembled from a record pinned
    before the clear; each still releases what it holds."""
    with manager.lock:
        manager._kq_ckpt_gen = int(getattr(manager, "_kq_ckpt_gen", 0)) + 1
        idx = getattr(manager, "_kq_ckpt_records", None)
        if idx:
            for rec in idx.values():
                _release_record(manager, rec)
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
    strip-on-extend was fighting over. A replay that landed stands in
    first all the same (``ckpt_replay_stands_in``): it covers the
    continuation a retirement leaves unstored, and adds no record."""
    if not meta:
        return False
    stored = meta.get("ckpt_stored_boundaries") or ()
    return any(b in stored for b in meta.get("ckpt_p_stable_bounds") or ())


def ckpt_replay_stands_in(manager, token_ids, prompt_cache=None, *,
                          extra_hash: int = 0, keep: int | None = None,
                          beside_turn: bool = False) -> bool:
    """Whether the request's N-1 replay record stands in for the p=N
    post-prefill store, which then is skipped. The two sit one token
    apart and differ only in the lookups they serve: the replay the
    identical resend (a record at N cannot serve an N-token query), the
    p=N record the next turn or a branch past the prompt. A replay on a
    recurrent layout serves only the resend (the replay gate in
    ``_ckpt_scan``), so it becomes a boundary here and serves both: the
    p=N record is off the grid as well, so a next turn restored from
    either chunks off the grid alike, and the resend still forwards one
    token. It goes back in as the p=N record would have, strip-on-extend
    and ``keep`` (the request's p_stable) included, so the boundaries of
    earlier turns on its chain still go. It takes the disk skeleton the
    p=N store would have written, and on a kvarn layout its window chain
    goes to disk as the p=N chain would have (``rot_disk``), so a restart
    restores the prompt record as before. A replay on an attention-only
    layout already adopts on extension and keeps its kind.
    ``beside_turn``: the request's render-stable boundary landed. That
    boundary sits on the grid below p_stable, up to a unit short of the
    prompt, so where no retirement stores, the next turn restores from the
    replay rather than from it. A boundary at or below ``keep`` on the
    chain counts the same, for a request that restored past the boundary
    an earlier one with its prompt stored. Once a record past the replay
    lands, the replay is a replay again (``_record_insert``): it serves
    the resend, and strip-on-extend keeps the boundary for the branches
    it serves.
    False when no replay landed: the p=N store runs unless the boundary
    makes it redundant."""
    if manager is None or token_ids is None:
        return False
    try:
        ids = tuple(int(t) for t in token_ids)[:-1]
        p = len(ids)
        if p < 2:
            return False
        key = (ids, int(salt_at(extra_hash, p)))
        idx = _ckpt_records(manager)
        win: list[Any] = []
        with manager.lock:
            rec = idx.get(key)
            if rec is None or rec.kind != "replay":
                return False
            promote = "arr" in rec.layout
            if promote:
                # Out of the index unreleased, then in again as a boundary.
                idx.pop(key)
                rec.kind = "boundary"
                # A boundary at or below keep on the chain is the turn
                # boundary too when an earlier request with this prompt
                # stored it and this one restored past it.
                rec.stood_in = bool(beside_turn) or (
                    keep is not None and any(
                        r.kind == "boundary" and r.p <= keep
                        and r.extra_hash == rec.extra_hash
                        and r.p < p and ids[:r.p] == r.ids
                        for r in idx.values()))
                _record_insert(manager, rec, keep)
            snap = _CkptRecord(
                **{k: getattr(rec, k) for k in _CkptRecord.__slots__})
            if (promote and key in idx and rec.bounded_blocks
                    and getattr(manager, "disk", None) is not None
                    and any(_is_kvarn(t) for t in rec.layout)):
                # Pinned for the unlocked join, as a lookup pins them.
                win = list(rec.bounded_blocks)
                for b in win:
                    manager._acquire_existing(b)
        _log.info("APC ckpt post-prefill store skipped: the replay at %d "
                  "stands in%s", p, " as a boundary" if promote else "")
        if win:
            _window_to_disk(manager, snap, win)
        if promote and prompt_cache is not None and snap.states is not None:
            _ckpt_disk_write(manager, list(snap.ids), prompt_cache,
                             snap.layout, snap.p, snap.b_full,
                             snap.p - snap.b_full, snap.states, snap.tails,
                             ckpt_extra_hash(snap.extra_hash),
                             qsa_streams=snap.qsa)
        return True
    except Exception:
        _log.warning("APC ckpt replay stand-in failed; storing p=N",
                     exc_info=True)
        return False


def _window_to_disk(manager, rec, held) -> None:
    """Write a record's window chain to disk under the salt its store
    used, from the blocks ``held`` (pinned by the caller, released here).
    An arr replay keeps its chain memory-only."""
    from .apc_blocks import join_blocks

    try:
        sink, _w, _off, L = rec.rot_meta
        ids = list(rec.ids)
        layers = join_blocks(held)
        got = manager.store_ckpt_blocks(
            ids[:sink] + ids[rec.p - (L - sink):rec.p],
            [k for k, _ in layers], [v for _, v in layers],
            extra_hash=bounded_extra_hash(rec.extra_hash, rec.p), disk=True)
        with manager.lock:
            manager.release(got)
    except Exception:
        _log.debug("APC ckpt window chain disk write failed", exc_info=True)
    finally:
        with manager.lock:
            manager.release(held)


def _release_blocks(manager, blocks) -> None:
    """Give ``blocks`` back to the pool. A clear() since they were taken
    left them to this holder (``GmlxAPCManager.clear``), so the release
    frees them."""
    if not blocks:
        return
    with manager.lock:
        manager.release(blocks)


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


def ckpt_governor_release(manager, fraction: float) -> int:
    """Governor rung: release ``fraction`` of ckpt records LRU-first and
    return the bytes freed. Records are retention, never correctness --
    dropping one costs the next lookup a warm start. Releasing unpins
    the record's blocks, so the manager's pool eviction running in the
    same governor pass can actually reclaim them; without this the
    pinned share is invisible to red-band reclaim (the 122B run-3 shed:
    11.2 GB recovered, shortfall anyway). Plain LRU, anchors included:
    on red band the cheapest global order is the point; fairness across
    chains is the pool path's concern, not the emergency one's."""
    idx = _ckpt_records(manager)
    if not idx:
        return 0
    fraction = min(1.0, max(0.0, float(fraction)))
    freed = 0
    released = 0
    with manager.lock:
        n = len(idx)
        take = max(1, round(n * fraction)) if n and fraction > 0 else 0
        for _ in range(take):
            if not idx:
                break
            _, rec = idx.popitem(last=False)
            freed += int(getattr(rec, "nbytes", 0) or 0)
            _release_record(manager, rec)
            released += 1
    if released:
        _ckpt_bump(manager, "ckpt_governor_released", released)
    return freed


def ckpt_release_coldest(manager) -> int | None:
    """Release the least recently used ckpt record and return its own
    bytes, or None when no record is left. The blocks it pinned go to the
    tail of the free queue for the caller to evict."""
    idx = _ckpt_records(manager)
    with manager.lock:
        if not idx:
            return None
        _, rec = idx.popitem(last=False)
        own = int(getattr(rec, "nbytes", 0) or 0)
        _release_record(manager, rec)
    _ckpt_bump(manager, "ckpt_governor_released", 1)
    return own


def _fatchain_victim(idx, bs: int):
    """Key of the eviction victim: the deepest non-anchor record of the
    chain group holding the most blocks, the group's anchor only once
    nothing else in it is left. Groups are keyed by the first block of
    ids: turns of one session share it, and shared-system-prefix
    siblings land together, which matches how their blocks pool."""
    tally: dict[Any, list] = {}
    for k, r in idx.items():
        g = r.ids[:bs]
        held = len(r.main_blocks or ()) + len(r.bounded_blocks or ())
        t = tally.setdefault(g, [0, []])
        t[0] += held
        t[1].append(k)
    keys = tally[max(tally, key=lambda g: tally[g][0])][1]
    non_anchor = [k for k in keys if idx[k].kind != "anchor"]
    return max(non_anchor or keys, key=lambda k: idx[k].p)


def _evict_for_pool(manager, deficit: int, store_ids=None) -> int:
    """Release records until ``deficit`` blocks sit on the pool free
    list, always at least one when the deficit is reachable (the caller
    loops on retry, so each call must make progress). Window chains are
    position-salted (no dedup), so on large-window models pinned
    records exhaust the pool long before the entry or byte bounds trip;
    insert-time eviction then never runs again because no store can
    complete. Blocks a record shares with an in-flight lookup pin or
    another request's chain do not free on release, so an unreachable
    deficit is detected upfront and evicts nothing rather than draining
    the whole index for a store that still declines.

    Victim order is fattest-chain-deepest-first, not LRU. Under
    round-robin sessions on a saturated pool, LRU's victim is exactly
    the session due to run next: every store evicts a record about to
    be reused, everyone pays the store and nobody collects the hit (the
    122B run-4 starvation: 1519 stores, 182 hits). Charging the chain
    group holding the most blocks makes chains equilibrate toward a
    fair share, and taking its deepest record first degrades that
    session to a shallower warm start instead of zeroing anyone.
    Anchors still go last within their group and get no absolute
    protection, since an anchor pinning window blocks on an exhausted
    pool would starve every future store. ``store_ids`` marks the
    storing chain so self-recycled evictions are visible in the stats.
    Returns the number of records released."""
    if not hasattr(manager, "_free_head"):
        return 0
    idx = _ckpt_records(manager)
    bs = int(getattr(manager, "block_size", 0) or 1)
    own_group = tuple(store_ids[:bs]) if store_ids is not None else None
    evicted = own = 0
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
            victim = idx.pop(_fatchain_victim(idx, bs))
            if own_group is not None and victim.ids[:bs] == own_group:
                own += 1
            _release_record(manager, victim)
            evicted += 1
    if own:
        _ckpt_bump(manager, "ckpt_evict_own_chain", own)
    return evicted


def _ckpt_extends(o, r) -> bool:
    """Whether record ``o`` continues record ``r``'s prompt. The pool
    keeps one block per hashed span, the salt included, so sharing
    ``r``'s last chain block proves the prefix up to ``r.b_full``; the
    tail ids settle the rest. A record with no chain (kvarn) compares its
    salt, as ``_record_insert`` does, then its ids, the last one first,
    so a pair of different prompts costs one compare and not a slice of
    the whole prompt."""
    if o.p <= r.p:
        return False
    nb = len(r.main_blocks or ())
    if nb:
        ob = o.main_blocks or ()
        return (len(ob) >= nb and ob[nb - 1] is r.main_blocks[nb - 1]
                and o.ids[r.b_full:r.p] == r.ids[r.b_full:])
    if o.extra_hash != r.extra_hash or o.ids[r.p - 1] != r.ids[-1]:
        return False
    return o.ids[:r.p] == r.ids


def _chain_heads(idx) -> set:
    """Keys of the records a next turn restores: per chain, the deepest
    record no deeper one continues, and each fork point, a record that
    two or more branches continue, unless it is the anchor. A client that
    forks a parent's context into children leaves the parent's record
    continued by every child, and the parent's next turn restores it. A
    shared anchor stays a non-head, as decided with the heads rule: it
    goes before any conversation's head, last among the rest, trading a
    future sibling's restore for a running conversation's next turn.
    Records on one line below a deeper record (the anchor, the record the
    last turn restored from) are not heads. A replay serves only its
    identical resend and a decision record belongs to no conversation, so
    neither is a head nor hides one."""
    recs = [(k, r) for k, r in idx.items()
            if r.kind not in ("replay", "decision")]
    n = len(recs)
    # ext[i][j]: record j continues record i.
    ext = [[_ckpt_extends(recs[j][1], recs[i][1]) for j in range(n)]
           for i in range(n)]
    heads = set()
    for i in range(n):
        kids = [j for j in range(n) if ext[i][j]]
        # A branch's first record continues no other record of i's.
        firsts = [j for j in kids if not any(ext[c][j] for c in kids)]
        if not firsts or (len(firsts) > 1
                          and recs[i][1].kind != "anchor"):
            heads.add(recs[i][0])
    return heads


def _lru_victim_key(idx, keep):
    """Key of the eviction victim, never the ``keep`` key (the record
    being inserted always survives). A chain's head (``_chain_heads``)
    goes only when nothing else is left: under round robin the least
    recently used conversation is the one due next, and its head is the
    record that turn restores. Within each group: the least recently used
    spared record, else non-anchor, else anchor. Lookup hits move_to_end,
    so anchor order is LRU by last hit: an anchor that never serves a
    sibling ages out, one that does stays hot. Caller holds the lock and
    guarantees a non-keep record exists."""
    heads = _chain_heads(idx)
    for group in ([k for k in idx if k != keep and k not in heads],
                  [k for k in idx if k != keep]):
        for tier in (lambda r: r.spared, lambda r: r.kind != "anchor",
                     lambda r: True):
            key = next((k for k in group if tier(idx[k])), None)
            if key is not None:
                return key
    raise KeyError("no eviction victim besides the record kept")


def _evict_lru_record(manager, idx, keep):
    """Drop and return the eviction victim (``_lru_victim_key``), with the
    blocks only it held (``_drop_record``)."""
    victim = idx.pop(_lru_victim_key(idx, keep))
    _drop_record(manager, victim)
    return victim


def _record_room(manager) -> tuple[int, bool]:
    """The bytes checkpoint records may hold, and whether the pool's
    committed blocks count against the same figure. autosize stashes the
    pool's share of the prompt cache's room (``_kq_ckpt_share_bytes``),
    and records and blocks live within it together.
    GMLX_APC_CKPT_BUDGET_MB, when set, is a records-only budget in its
    place, and a manager autosize never sized keeps the records-only
    default."""
    share = getattr(manager, "_kq_ckpt_share_bytes", None)
    if share and not _CKPT_BUDGET_SET:
        return int(share), True
    return int(_CKPT_BUDGET_BYTES), False


def _committed_block_bytes(manager) -> int:
    tally = getattr(manager, "_pool_bytes", None)
    if tally is not None:
        return int(tally.bytes)
    return sum(int(b.resident_bytes())
               for b in getattr(manager, "pool", None) or ())


def _free_unheld(manager, blocks) -> int:
    """Free each of ``blocks`` that nothing holds any more: on a
    checkpoint model a chain of blocks is useless without its record. The
    block leaves the hash table and releases its slabs, and stays empty
    in the free queue, as the pool's own eviction leaves one. A block a
    live record or a lookup still holds stays. Returns the bytes freed."""
    freed = 0
    table = getattr(manager, "hash_table", None)
    seen: set[int] = set()
    for b in blocks:
        if id(b) in seen or b.ref_cnt > 0:
            continue
        seen.add(id(b))
        nb = int(b.resident_bytes())
        if not nb:
            continue
        if (table is not None and b.block_hash is not None
                and table.get(b.block_hash) is b):
            del table[b.block_hash]
            manager.stats.evictions += 1
        b.block_hash = None
        b.token_ids = ()
        b.release_components()
        freed += nb
    return freed


def _drop_record(manager, rec) -> int:
    """Release a record leaving the index and free the blocks only it
    held: a record is never dropped while its blocks stay, and blocks
    whose record is gone are freed, on every path that drops one. The
    pool path (``_evict_for_pool``) and the governor release only, since
    their callers take or free those blocks at once. Returns the bytes
    freed. Caller holds the lock."""
    held = list(rec.main_blocks or ()) + list(rec.bounded_blocks or ())
    _release_record(manager, rec)
    return _free_unheld(manager, held)


def _fit_record_room(manager, idx, keep) -> tuple[list, dict]:
    """Bring the records, and the pool's committed blocks when they share
    the room, within the room (``_record_room``). What goes first: blocks
    no record holds, then records in eviction order (``_lru_victim_key``),
    each with the blocks only it held, so a record is never dropped while
    its blocks stay. The record ``keep`` always stays. Caller holds the
    lock. Returns each dropped record with the bytes it freed, and the
    figures the log line reports."""
    room, shared = _record_room(manager)
    total = sum(int(getattr(r, "nbytes", 0) or 0) for r in idx.values())
    blocks = _committed_block_bytes(manager) if shared else 0
    note = {"room": room, "shared": shared, "before": total + blocks,
            "idle": 0}
    victims: list = []
    if total + blocks <= room:
        return victims, note
    evict_idle = getattr(manager, "_evict_idle_blocks", None)
    if shared and evict_idle is not None:
        note["idle"] = int(evict_idle(total + blocks - room))
        blocks -= note["idle"]
    while total + blocks > room and len(idx) > 1:
        victim = idx.pop(_lru_victim_key(idx, keep))
        own = int(getattr(victim, "nbytes", 0) or 0)
        freed = _drop_record(manager, victim)
        total -= own
        if shared:
            blocks -= freed
        victims.append((victim, own + freed))
    note["after"] = total + blocks
    return victims, note


def _chain_fresh_bytes(manager, chain, block_bytes: int) -> int:
    """The bytes a chain's block store adds to the pool's committed
    blocks. ``chain`` is the chain's ``(chunk, hash)`` pairs
    (``apc_manager.block_chain``). A block the pool holds already dedups
    and adds nothing; each other block takes the next block of the free
    queue, in order, and adds ``block_bytes`` less what that block held,
    all of it for an empty block. Caller holds the lock."""
    table = getattr(manager, "hash_table", None) or {}
    held: set[int] = set()
    new = 0
    for chunk, h in chain:
        b = table.get(h)
        if b is not None and b.token_ids == chunk:
            held.add(id(b))
        else:
            new += 1
    fresh = 0
    b = getattr(manager, "_free_head", None)
    while b is not None and new > 0:
        if id(b) not in held:
            new -= 1
            fresh += max(0, int(block_bytes) - int(b.resident_bytes()))
        b = b.next
    return fresh


def _fit_chain_room(manager, span_ids, ks, vs, extra, n_blocks: int,
                    keep) -> tuple[list, dict | None]:
    """Before a checkpoint chain's block copy, bring the records and the
    pool's committed blocks, with the bytes the chain adds
    (``_chain_fresh_bytes``), within the shared room (``_record_room``),
    in ``_fit_record_room``'s order: blocks no record holds, then records
    in eviction order, each with the blocks only it held. Blocks the
    chain dedups are not priced, so no record goes for them. The record
    ``keep`` stays; the new record's own payload is priced at its insert.
    Caller holds the lock. Returns the dropped records with the bytes
    each freed, and the figures the log line reports (None when the room
    is not shared)."""
    room, shared = _record_room(manager)
    if not shared or n_blocks <= 0 or not ks:
        return [], None
    from .apc_manager import block_chain

    bs = int(manager.block_size)
    tokens = max(1, int(ks[0].shape[-2]))
    block = int(sum(int(a.nbytes) for a in list(ks) + list(vs))
                * bs // tokens)
    chain = list(block_chain(span_ids, extra, n_blocks, bs))
    idx = _ckpt_records(manager)

    def size() -> int:
        return (sum(int(getattr(r, "nbytes", 0) or 0) for r in idx.values())
                + _committed_block_bytes(manager)
                + _chain_fresh_bytes(manager, chain, block))

    total = size()
    note = {"room": room, "shared": True, "before": total, "idle": 0}
    victims: list = []
    evict_idle = getattr(manager, "_evict_idle_blocks", None)
    while total > room:
        idle = int(evict_idle(total - room)) if evict_idle else 0
        if idle:
            note["idle"] += idle
        elif any(k != keep for k in idx):
            victim = idx.pop(_lru_victim_key(idx, keep))
            own = int(getattr(victim, "nbytes", 0) or 0)
            victims.append((victim, own + _drop_record(manager, victim)))
        else:
            break
        total = size()
    note["after"] = total
    return victims, note


def _note_budget_evictions(manager, rec, victims, note) -> None:
    """The budget's counters and its one log line for an insert that
    dropped records."""
    mib = float(1 << 20)
    freed = sum(b for _, b in victims)
    st = _ckpt_stats(manager)
    with manager.lock:
        st["ckpt_budget_evictions"] = (st.get("ckpt_budget_evictions", 0)
                                       + len(victims))
        st["_budget_evicted_bytes"] = (st.get("_budget_evicted_bytes", 0)
                                       + int(freed))
        st["ckpt_budget_evicted_mb"] = st["_budget_evicted_bytes"] >> 20
    _log.info(
        "APC ckpt budget: dropped %d record(s), %.1f MiB, for the p=%d %s "
        "store: %s; %s %.1f MiB, now %.1f MiB, of %.1f MiB%s",
        len(victims), freed / mib, rec.p, rec.kind,
        ", ".join(f"p={v.p} {v.kind} {b / mib:.1f} MiB"
                  for v, b in victims),
        "records and pool blocks" if note["shared"] else "records",
        note["before"] / mib, note["after"] / mib, note["room"] / mib,
        f"; blocks no record held {note['idle'] / mib:.1f} MiB"
        if note["idle"] else "")


def _record_insert(manager, rec, keep: int | None = None,
                   materialize=None, gen: int | None = None) -> bool:
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

    Decision records keep to themselves: one is never promoted to anchor,
    a decision insert strips only decision records on its chain, and no
    other insert strips a decision record. A decision insert past
    _CKPT_DECISION_ENTRIES releases the least recently used decision
    record. Eviction ranks them with the non-anchors.

    Bytes: the records, with the pool's committed blocks on an autosized
    manager, stay within the room (``_fit_record_room``).

    Against real memory pressure the exemptions pin little: the count
    and byte bounds keep each chain's head to the last and evict anchors
    after non-anchors among the rest (LRU by last hit), and
    pool-pressure eviction (_evict_for_pool) gives anchors no protection
    at all: position-salted window chains must never sit pinned on an
    exhausted pool.

    ``keep``: the storing request's p_stable. Where a record past it would
    strip the deepest record at or below it, the one a next turn
    restores, that record stays, marked spared: the count and byte
    bounds evict it first, so it never costs another chain a record.

    ``materialize``: the eval of a new record's payload. The bounds run
    first with the record counted, from its shapes, so the payload lands
    in the room they made. The record sits out of the index while it
    materializes, so no lookup adopts arrays not yet evaluated, and comes
    back as the newest.

    ``gen``: the clear generation the record's blocks were taken at. Once
    a clear() has run since, the reset dropped what was cached: nothing
    goes in and the call returns False, and the caller releases the
    record's chains."""
    idx = _ckpt_records(manager)
    key = (rec.ids, rec.extra_hash)
    rec.nbytes = _rec_nbytes(rec)

    def cleared() -> bool:
        return (gen is not None
                and int(getattr(manager, "_kq_ckpt_gen", 0)) != gen)

    with manager.lock:
        if cleared():
            return False
        old = idx.pop(key, None)
        if old is not None:
            if old.kind == "anchor" and rec.kind == "boundary":
                rec.kind = "anchor"     # a re-store keeps the anchor tag
            _drop_record(manager, old)
        # chain = records whose ids are a strict prefix of this one
        chain = [k for k, r in idx.items()
                 if r.extra_hash == rec.extra_hash and r.p < rec.p
                 and rec.ids[:r.p] == r.ids]
        if rec.kind != "decision":
            for k in chain:
                if idx[k].stood_in:
                    # Its continuation is past it now: back to the resend.
                    idx[k].kind, idx[k].stood_in = "replay", False
        if rec.kind == "replay":
            # One replay per chain: the newer one supersedes outright.
            for k in [k for k in chain if idx[k].kind == "replay"]:
                _drop_record(manager, idx.pop(k))
        elif rec.kind == "anchor":
            for k in [k for k in chain if idx[k].kind == "anchor"]:
                _drop_record(manager, idx.pop(k))
        elif rec.kind == "boundary" and not any(
                idx[k].kind not in ("replay", "decision")
                for k in chain if k in idx):
            rec.kind = "anchor"         # first restorable boundary
        held = None
        if keep is not None and rec.p > keep and rec.kind != "decision":
            below = [k for k in chain if k in idx and idx[k].p <= keep
                     and idx[k].kind != "decision"]
            held = max(below, key=lambda k: idx[k].p, default=None)
        if rec.kind == "decision":
            chain = [k for k in chain if k in idx and idx[k].kind == "decision"]
        else:
            chain = [k for k in chain if k in idx
                     and idx[k].kind not in ("replay", "anchor", "decision")]
        # Strip-on-extend frees superseded records' main chains for the
        # pool; sub-prefix adoption survives it because those chains are
        # content-deduped and skeleton re-index re-cuts them. A record
        # that carries a kvarn layer, or no main chain at all, is the only
        # carrier of its boundary (inline codes; the fp16 carve-out layer
        # beside them has a chain but cannot restore the kvarn rows), so
        # stripping it kills divergent-suffix and turn adoption outright.
        # The count and byte bounds below govern those records, and
        # _evict_for_pool reclaims the blocks they pin.
        chain = [k for k in chain
                 if idx[k].main_blocks
                 and not any(_is_kvarn(t) for t in idx[k].layout)]
        chain.sort(key=lambda k: idx[k].p, reverse=True)
        for k in chain[_CKPT_HEAVY_PER_CHAIN - 1:]:
            if k != held:
                _drop_record(manager, idx.pop(k))
            else:
                idx[k].spared = True
        idx[key] = rec
        idx.move_to_end(key)
        if rec.kind == "decision":
            decisions = [k for k, r in idx.items() if r.kind == "decision"]
            for k in decisions[:-_CKPT_DECISION_ENTRIES]:
                _drop_record(manager, idx.pop(k))
        while len(idx) > _CKPT_RECORD_ENTRIES:
            _evict_lru_record(manager, idx, key)
        victims, note = _fit_record_room(manager, idx, key)
        if materialize is not None:
            idx.pop(key)
    if victims:
        _note_budget_evictions(manager, rec, victims, note)
    if materialize is not None:
        materialize()
        with manager.lock:
            if cleared():
                return False
            idx[key] = rec
            idx.move_to_end(key)
    return True


def ckpt_store(
    manager: Any,
    token_ids,
    prompt_cache: list[Any],
    *,
    extra_hash: int = 0,
    skeleton_disk: bool = True,
    kind: str = "boundary",
    grid_truncate: bool = False,
    keep: int | None = None,
    eval_async: bool = False,
) -> int:
    """Store a hybrid checkpoint at ``p = len(token_ids)``.

    Single-row cache list, KV/rotating offsets == p. Plain KV rides the
    salted main chain, rotating layers a per-checkpoint window chain
    (block-grid p below the window, any p at or beyond it), states and
    unaligned-GDN tails land in the pinned record; disk skeleton written
    best-effort. ``skeleton_disk=False`` skips the skeleton write (the
    skeleton inlines recurrent state, >100 MB per GDN checkpoint --
    interval boundaries superseded minutes later do not earn that).
    ``kind`` stamps the record's retention class (see _CkptRecord); an
    unknown kind stores nothing. ``grid_truncate`` turns the below-window off-grid rotating decline
    into a terminal store at the largest block-aligned prefix: pre-wrap
    the buffer is a temporal prefix, so a slice is a faithful shorter
    run. Non-recurrent layouts only (state cannot rewind), memory-only
    (the live cache's offset would stamp a mismatched skeleton).
    ``keep``: the request's p_stable, which the record's strip-on-extend
    honors (see _record_insert).
    ``eval_async``: the copies are submitted, not waited for, where the
    manager is memory-only and every K/V layer fences its next write.
    The next chunk then queues behind them.
    Returns the stored length in tokens, 0 when nothing stored. Never
    raises.
    """
    if manager is None or token_ids is None:
        return 0
    from .kv_cache import ArraysCache, KVCache, RotatingKVCache
    from .kvarn_cache import KVarNKVCache

    main_blocks: list[Any] = []
    bounded_blocks: list[Any] = []
    gen = None
    try:
        ids = [int(t) for t in token_ids]
        p = len(ids)
        bs = int(manager.block_size)
        if kind not in _CKPT_KINDS:
            _ckpt_decline(manager, "kind")
            return 0
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
        # restart it misses on the window chain, loudly. Kvarn layouts
        # widen the set: a kvarn prefill boundary gets a disk skeleton
        # only where the arm lists it (``ckpt_skeletons``, the replay
        # included), so each one is restart-restorable, and its window
        # chain (bounded by W, not p) must persist with it or the skeleton
        # is dead weight.
        rot_disk = skeleton_disk and (
            kind in ("replay", "retire")
            or any(_is_kvarn(str(t)) for t in layout))
        if any(isinstance(c, _buffered_types()) for c in prompt_cache):
            _log.info(
                "APC ckpt store declined: BufferedRotatingKVCache rows "
                "cannot snapshot (support deferred)")
            _ckpt_decline(manager, "buffered")
            return 0
        for c in prompt_cache:
            off = getattr(c, "offset", None)
            if off is not None and not isinstance(c, RotatingKVCache) \
                    and isinstance(c, KVCache) and int(off) != p:
                _log.info("APC ckpt store skipped: KV offset %d != %d",
                          int(off), p)
                _ckpt_decline(manager, "offset")
                return 0
            if isinstance(c, RotatingKVCache) and int(c.offset) != p:
                _log.info("APC ckpt store skipped: rot offset %d != %d",
                          int(c.offset), p)
                _ckpt_decline(manager, "offset")
                return 0
            if type(c) is KVarNKVCache and int(c.offset) != p:
                _log.info("APC ckpt store skipped: kvarn offset %d != %d",
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
                has_arr = any(not isinstance(c, (KVCache, RotatingKVCache))
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
        # A record keys on the images inside its own p.
        extra_hash = int(salt_at(extra_hash, p))
        salted = ckpt_extra_hash(extra_hash)
        # Any: QSA layers subclass KVCache and carry their indexer stream.
        kv_caches: list[Any] = [c for c in prompt_cache
                                if isinstance(c, KVCache)
                                and not isinstance(c, RotatingKVCache)]
        rot_caches = [c for c in prompt_cache
                      if isinstance(c, RotatingKVCache)]
        # Inline-state layers (arr and kvarn), in prompt order: the record
        # stores full clones, and assembly pulls them positionally by tag.
        inline_caches = [c for c in prompt_cache
                         if not isinstance(c, (KVCache, RotatingKVCache))]

        store_blocks = getattr(manager, "store_ckpt_blocks", None)
        # The disk writer thread reads evaluated arrays only. A copy in
        # flight holds the buffer it reads, so a layer the next chunk
        # writes in place waits for the copies first (hold_writes), or
        # MLX copies its whole buffer instead.
        if eval_async and (
                store_blocks is None
                or getattr(manager, "disk", None) is not None
                or rot_caches or not kv_caches
                or not all(hasattr(c, "hold_writes") for c in kv_caches)
                or not all(isinstance(c, ArraysCache)
                           for c in inline_caches)):
            eval_async = False
        fence: list[Any] = []
        if eval_async:
            for c in kv_caches:
                c.hold_writes(fence)

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
                # The chain's new blocks are priced against the shared
                # room before the copy, under the lock the copy runs in.
                with manager.lock:
                    victims, note = _fit_chain_room(
                        manager, span_ids, ks, vs, extra, need,
                        (tuple(ids), int(extra_hash)))
                    if eval_async:
                        got = store_blocks(span_ids, ks, vs, extra_hash=extra,
                                           disk=disk, eval_async=True)
                    elif store_blocks is not None:
                        got = store_blocks(
                            span_ids, ks, vs, extra_hash=extra, disk=disk)
                    else:
                        got = manager.store_kv_blocks(
                            span_ids, ks, vs, extra_hash=extra)
                if victims:
                    _note_budget_evictions(
                        manager, SimpleNamespace(p=p, kind=kind), victims,
                        note)
                return got

            blocks = _once()
            evicted_total = 0
            while len(blocks) < need:
                got = len(blocks)
                _release_blocks(manager, blocks)
                evicted = _evict_for_pool(manager, need - got,
                                          store_ids=ids)
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

        # The clear generation the chains are taken at: after a clear()
        # the insert puts no record in (_record_insert).
        with manager.lock:
            gen = int(getattr(manager, "_kq_ckpt_gen", 0))
        if b_full > 0 and kv_caches:
            lk = [c.keys[..., :b_full, :] for c in kv_caches]
            lv = [c.values[..., :b_full, :] for c in kv_caches]
            # Evaluate before storing: the shard payload crosses to the
            # disk writer thread, and evaluating arrays that share
            # unevaluated inputs from two threads is undefined in mlx.
            # Owned survivors: lk/lv slice the live prompt cache, whose
            # own pending graph rides the same tape; the guard drains
            # before the store's except path releases and declines.
            from gmlx.eval_guard import guard
            if not eval_async:
                guard.eval(*(lk + lv), site="ckpt-store-main",
                           owner="owned")
            got_main = _chained(
                ids[:b_full], lk, lv, extra=salted, disk=True,
                need=b_full // bs, what="main")
            if got_main is None:
                return 0
            main_blocks = got_main
            if eval_async:
                fence.extend(a for b in main_blocks
                             for a in (b.kv_handle().parts or ()))

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
                _release_blocks(manager, main_blocks)
                return 0
            metas = {cw[2] for cw in canon}
            if len(metas) != 1:
                _ckpt_decline(manager, "canon")
                _release_blocks(manager, main_blocks)
                return 0
            rot_meta = canon[0][2]
            sink, _w, _off, L = rot_meta
            canon_ids = ids[:sink] + ids[p - (L - sink):p]
            bsalt = bounded_extra_hash(extra_hash, p)
            canon_k = [cw[0] for cw in canon]
            canon_v = [cw[1] for cw in canon]
            # Same writer-thread rule and ownership as the main chain.
            from gmlx.eval_guard import guard
            guard.eval(*(canon_k + canon_v), site="ckpt-store-window",
                       owner="owned")
            got_win = _chained(
                canon_ids, canon_k, canon_v, extra=bsalt, disk=rot_disk,
                need=L // bs, what="window")
            if got_win is None:
                _release_blocks(manager, main_blocks)
                return 0
            bounded_blocks = got_win

        # The payload clones stay lazy and sync once, inside the insert,
        # after the bounds have made room for the record.
        pending: list[Any] = []
        # The async path holds the recurrent states (_held_state), which
        # its gate keeps to ArraysCache layers.
        states = [_held_state(c, pending) if eval_async
                  else _clone_single_row(c, pending) for c in inline_caches]
        if any(s is None for s in states):
            _ckpt_decline(manager, "clone")
            _release_blocks(manager, main_blocks)
            _release_blocks(manager, bounded_blocks)
            return 0
        # QSA layers: the chain carries only k/v, so the indexer key
        # stream (and mrope positions) ride the record whole. ik is one
        # raw key per token; a 3k-token 12-layer store runs ~10 MB.
        qsa_streams = None
        if any(hasattr(c, "update_and_fetch_qsa") for c in kv_caches):
            import mlx.core as mx
            qsa_streams = []
            for c in kv_caches:
                if not hasattr(c, "update_and_fetch_qsa"):
                    continue
                if c.ik is None or c.ik.shape[1] < p:
                    qsa_streams = None
                    break
                ik = mx.contiguous(c.ik[:, :p])
                qpos = None if c.pos is None else mx.contiguous(
                    c.pos[:, :, :p])
                qsa_streams.append((ik, qpos, int(c.ratio)))
            if qsa_streams is None:
                _ckpt_decline(manager, "clone")
                _release_blocks(manager, main_blocks)
                _release_blocks(manager, bounded_blocks)
                return 0
            pending.extend(a for t in qsa_streams for a in t[:2]
                           if a is not None)
        tails = None
        if tail_len and kv_caches:
            tails = []
            for c in kv_caches:
                t = KVCache()
                t.state = (c.keys[..., b_full:p, :],
                           c.values[..., b_full:p, :])
                t = _clone_single_row(t, pending)
                if t is None:
                    _ckpt_decline(manager, "clone")
                    _release_blocks(manager, main_blocks)
                    _release_blocks(manager, bounded_blocks)
                    return 0
                tails.append(t)

        def materialize():
            if pending:
                # Owned survivors, as on the main chain: the clones read
                # the live prompt cache and its pending graph.
                from gmlx.eval_guard import guard
                if eval_async:
                    from gmlx.serve.cb_phase import one_buffer
                    with one_buffer():
                        guard.async_eval(*pending, site="ckpt-store-clones",
                                         owner="owned")
                    fence.extend(pending)
                else:
                    guard.eval(*pending, site="ckpt-store-clones",
                               owner="owned")

        rec = _CkptRecord(
            ids=tuple(ids), extra_hash=int(extra_hash), p=p, b_full=b_full,
            layout=tuple(layout), main_blocks=main_blocks,
            bounded_blocks=bounded_blocks, rot_meta=rot_meta,
            states=states, tails=tails, qsa=qsa_streams, kind=str(kind))
        inserted = _record_insert(manager, rec, keep, materialize, gen)
        # Ownership moved to the record; the except path must not release.
        main_blocks = bounded_blocks = []
        if not inserted:
            # A clear() ran since the chains were taken: the reset dropped
            # what was cached and left the chains to this store, whose
            # release frees the blocks taken before it.
            _release_record(manager, rec)
            _ckpt_decline(manager, "reset")
            _log.info("APC ckpt store dropped: the cache was reset during "
                      "the store (p=%d)", p)
            return 0

        if skeleton_disk:
            _ckpt_disk_write(manager, ids, prompt_cache, layout, p, b_full,
                             tail_len, states, tails, salted,
                             qsa_streams=qsa_streams)
        _ckpt_bump(manager, "ckpt_stores")
        _log.info(
            "APC ckpt store: tokens=%d main=%d window=%d tail=%d states=%d "
            "rec_mb=%.1f",
            p, len(rec.main_blocks), len(rec.bounded_blocks), tail_len,
            len(states), rec.nbytes / (1 << 20))
        return p
    except Exception:
        # One swallow per call: a failing decline (telemetry) or first
        # release must not leak the other block set out of the manager.
        try:
            _ckpt_decline(manager, "exception")
        except Exception:  # noqa: S110 - decline is telemetry only
            pass
        for blks in (main_blocks, bounded_blocks):
            try:
                _release_blocks(manager, blks)
            except Exception:  # noqa: S110 - best-effort release on the failure path
                pass
        _log.warning("APC ckpt store failed; continuing", exc_info=True)
        return 0


def _ckpt_disk_write(manager, ids, prompt_cache, layout, p, b_full,
                     tail_len, states, tails, salted,
                     qsa_streams=None) -> None:
    """Best-effort disk skeleton: placeholders for chain-backed layers
    (offset stamped, rotating meta native), real tails/states inline.
    ``qsa_streams`` are the record's already-evaluated (ik, pos, ratio)
    clones: the shard writer thread cannot evaluate GPU-stream graphs,
    so the skeleton must only carry materialized arrays."""
    disk = getattr(manager, "disk", None)
    if disk is None:
        return
    from .kv_cache import ArraysCache, KVCache, RotatingKVCache
    try:
        entries: list[Any] = []
        kv_i = arr_i = qsa_i = 0
        for tag, c in zip(layout, prompt_cache):
            if _is_qsa(tag):
                # QSA skeleton: k/v are chain-backed like "kv" (tail-only
                # payload, offset stamped), but the full indexer stream
                # rides the entry so a restart can rebuild the real cache.
                if not qsa_streams or qsa_i >= len(qsa_streams):
                    return
                ik, qpos, ratio = qsa_streams[qsa_i]
                sk = _qsa_cache_cls()(ratio)
                sk.ik = ik
                sk.pos = qpos
                if tail_len:
                    # Evaluated clone arrays, handed over whole: the
                    # writer thread cannot evaluate main-thread slices,
                    # and the serializer trims to stored width itself.
                    t = tails[kv_i]
                    sk.keys = t.keys
                    sk.values = t.values
                sk.offset = p
                entries.append(sk)
                kv_i += 1
                qsa_i += 1
            elif tag == "kv":
                if tail_len:
                    entries.append(tails[kv_i])
                else:
                    ph = KVCache()
                    ph.offset = p
                    entries.append(ph)
                kv_i += 1
            elif _is_rot(tag):
                ph = RotatingKVCache(
                    max_size=int(c.max_size), keep=int(c.keep))
                ph.offset = p
                ph._idx = min(p, int(c.max_size))
                entries.append(ph)
            else:
                entries.append(states[arr_i])
                arr_i += 1
        # The upstream writer drops zero-array shards; the sentinel keeps
        # an all-placeholder skeleton persistable. Stripped on lookup. It
        # also records b_full, where a QSA tail starts.
        import mlx.core as mx
        sent = ArraysCache(size=1)
        sent.cache[0] = mx.array([float(p), float(b_full)], dtype=mx.float32)
        entries.append(sent)
        from gmlx._vendor.mlx_vlm import apc as _apc
        tid = tuple(ids)
        # Mirror the manager's wire salt: the read side goes through
        # lookup_exact_cache, which XORs _exact_extra_salt into the value
        # that feeds both the sequence hash and the persisted-metadata
        # check -- a write missing either half is a permanent restart
        # miss on every salted (kvarn) boot. One variable salts both.
        salted = int(_skeleton_salt(
            salted, layout, manager.block_size)) ^ int(
            getattr(manager, "_exact_extra_salt", 0) or 0)
        khash = _apc._sequence_hash(tid, extra_hash=salted,
                                    block_size=manager.block_size)
        # The disk writer counts disk_writes when the write lands.
        disk.save_exact_cache(khash, tid, salted, entries)
        _ckpt_bump(manager, "ckpt_skeleton_writes")
    except Exception:
        _log.debug("APC ckpt disk skeleton save failed", exc_info=True)


def _assemble_from_record(manager, rec, geometry_check=None):
    """Warm cache list from a pinned record, or None (conjunction fail)."""
    import mlx.core as mx
    from .apc_blocks import block_layers, join_blocks
    from .kv_cache import KVCache

    n_kv = sum(1 for t in rec.layout if _is_chain(t))
    n_rot = sum(1 for t in rec.layout if _is_rot(t))
    if any(_is_qsa(t) for t in rec.layout) and not rec.qsa:
        return None
    if rec.b_full > 0 and n_kv and not rec.main_blocks:
        return None
    if n_rot and not rec.bounded_blocks:
        return None
    layer_kv = None
    if n_kv:
        if rec.b_full > 0:
            if block_layers(rec.main_blocks[0]) != n_kv:
                return None
            layer_kv = join_blocks(rec.main_blocks)
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
        if block_layers(rec.bounded_blocks[0]) != n_rot:
            return None
        keep, w, off, L = rec.rot_meta
        layer_rot = []
        for k, v in join_blocks(rec.bounded_blocks):
            rc = rotating_restore(k, v, (keep, w, rec.p, k.shape[2]))
            if rc is None or k.shape[2] != L:
                return None
            layer_rot.append(rc)
    warm: list[Any] = []
    # The array clones join the one eval below.
    targets: list[Any] = []
    kv_i = rot_i = arr_i = qsa_i = 0
    for tag in rec.layout:
        if tag == "kv":
            kc = KVCache()
            kc.state = layer_kv[kv_i]
            warm.append(kc)
            kv_i += 1
        elif _is_qsa(tag):
            ik, qpos, ratio = rec.qsa[qsa_i]
            st = layer_kv[kv_i] + ((ik,) if qpos is None else (ik, qpos))
            qc = _qsa_cache_cls()(ratio)
            qc.state = st
            if qc.offset != rec.p:
                return None
            warm.append(qc)
            kv_i += 1
            qsa_i += 1
        elif _is_rot(tag):
            warm.append(layer_rot[rot_i])
            rot_i += 1
        else:
            clone = _clone_single_row(rec.states[arr_i], targets)
            if clone is None:
                return None
            warm.append(clone)
            arr_i += 1
    for c in warm:
        if getattr(c, "keys", None) is not None:
            targets.extend([c.keys, c.values])
        elif hasattr(c, "cache"):
            targets.extend(s for s in c.cache if s is not None)
        if getattr(c, "ik", None) is not None:
            targets.append(c.ik)
    mx.eval(*targets)
    return warm


def _ckpt_scan(manager, tid: tuple, extra_hash: int, min_prefix_tokens: int,
               layout) -> tuple:
    """Candidate records for ``tid`` under the manager lock: the pinned
    records whose ids are a strict prefix of the query at the same salt
    and layout, plus the tallies the miss path reports (``gated``,
    ``refused``, ``salted``, ``diverged``) and the index generation the
    caller checks its assembly against. Pure: bumps no counters."""
    n = len(tid)
    idx = _ckpt_records(manager)
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
            if rec.extra_hash != int(salt_at(extra_hash, rec.p)):
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
                            "APC ckpt diverged: rec.p=%d kind=%s "
                            "layout=%s first mismatch at %d "
                            "(query %r vs rec %r)",
                            rec.p, rec.kind, rec.layout, m,
                            tid[m:m + 4], rec.ids[m:m + 4])
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
    return cands, gated, refused, salted, diverged, gen


def ckpt_peek(manager, token_ids, *, extra_hash: int = 0,
              min_prefix_tokens: int = 0, layout=None) -> int:
    """How deep the checkpoint tier could warm-start ``token_ids``: the
    deepest pinned record's ``p``, or 0. The dry-run's probe: it neither
    assembles a cache nor consults the disk skeleton, and moves no
    counter. Never raises."""
    if manager is None or token_ids is None:
        return 0
    try:
        tid = tuple(int(t) for t in token_ids)
        cands = _ckpt_scan(manager, tid, extra_hash, min_prefix_tokens,
                           layout)[0]
        return max((int(r.p) for r in cands), default=0)
    except Exception:
        _log.debug("APC ckpt peek failed", exc_info=True)
        return 0


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
        cands, gated, refused, salted, diverged, gen = _ckpt_scan(
            manager, tid, extra_hash, min_prefix_tokens, layout)
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
                    manager.release(held)
            if stale:
                # A clear() ran while the record was pinned: the record
                # index died with it, and the warm result is pre-clear
                # state, so it is discarded. The clear left the pinned
                # blocks to this lookup, and the release above frees them.
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
    from .apc_blocks import block_layers, join_blocks
    from .kv_cache import KVCache, RotatingKVCache

    salted = _skeleton_salt(ckpt_extra_hash(extra_hash), layout,
                            manager.block_size)
    entries, p = manager.lookup_exact_cache(
        ids, extra_hash=salted, min_prefix_tokens=min_prefix_tokens)
    if entries is None or p <= 0:
        return None, 0
    # Strip the skeleton writer's sentinel: exact by length when the live
    # layout is known, by shape otherwise.
    sent = None
    if layout is not None:
        if len(entries) == len(layout) + 1:
            sent = entries[-1]
            entries = entries[:-1]
    else:
        last = entries[-1] if entries else None
        if (last is not None
                and not isinstance(last, (KVCache, RotatingKVCache))
                and hasattr(last, "cache") and len(last.cache) == 1
                and last.cache[0] is not None
                and getattr(last.cache[0], "size", 0) in (1, 2)):
            sent = last
            entries = entries[:-1]
    from .kvarn_cache import KVarNKVCache
    tags = []
    for e in entries:
        qsa = _qsa_tag(e)
        if isinstance(e, RotatingKVCache):
            tags.append(_rot_tag(e))
        elif qsa is not None:
            tags.append(qsa)
        elif isinstance(e, KVCache):
            tags.append("kv")
        elif type(e) is KVarNKVCache:
            tags.append(_kvarn_tag(e))
        else:
            tags.append("arr")
    if layout is not None and tuple(layout) != tuple(tags):
        _log.info("APC ckpt disk miss: layout %s != model %s",
                  tags, list(layout))
        return None, 0
    # The skeleton's chains key on the images inside its p.
    salted = ckpt_extra_hash(int(salt_at(extra_hash, p)))
    bs = int(manager.block_size)
    b_full = _ckpt_block_prefix(p, bs)
    if any(_is_qsa(t) for t in tags):
        # A QSA tail starts at the writer's b_full, and the disk namespace
        # does not key on block size.
        sv = getattr(sent, "cache", [None])[0]
        b_w = (int(sv[1].item())
               if sv is not None and getattr(sv, "size", 0) == 2 else -1)
        if b_w != b_full:
            _log.info("APC ckpt disk miss: qsa block prefix %d != %d",
                      b_w, b_full)
            return None, 0
    n_kv = sum(1 for t in tags if _is_chain(t))
    blocks = []
    wblocks = []
    try:
        layer_kv = None
        if n_kv and b_full > 0:
            blocks, matched = manager.lookup_prefix(
                ids[:b_full], extra_hash=salted)
            if matched >= b_full and blocks \
                    and block_layers(blocks[0]) == n_kv:
                layer_kv = join_blocks(blocks)
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
        rot_entries = [e for e in entries if isinstance(e, RotatingKVCache)]
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
                    and block_layers(wblocks[0]) == len(rot_entries):
                src = join_blocks(wblocks)
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
            if _is_chain(tag):
                qsa_c = _is_qsa(tag)
                if qsa_c:
                    # QSA skeleton entries stamp offset=p and carry only
                    # the unaligned tail, p - b_full rows, at the head of
                    # keys. Neither the offset nor the width says so: the
                    # restore grows the buffer for the next prefill.
                    ek = getattr(e, "keys", None)
                    t_len = p - b_full if ek is not None else 0
                    if ek is not None and int(ek.shape[2]) < t_len:
                        _log.info("APC ckpt disk miss: qsa tail %d < %d",
                                  int(ek.shape[2]), t_len)
                        return None, 0
                else:
                    has_tail = getattr(e, "keys", None) is not None
                    t_len = int(e.offset) if has_tail else 0
                base = layer_kv[kv_i] if layer_kv is not None else None
                if t_len:
                    tk = e.keys[..., :t_len, :]
                    tv = e.values[..., :t_len, :]
                    if base is not None:
                        kv_state = (
                            mx.concatenate([base[0], tk], axis=2),
                            mx.concatenate([base[1], tv], axis=2))
                    else:
                        kv_state = (tk, tv)
                elif base is not None:
                    kv_state = base
                else:
                    return None, 0
                if qsa_c:
                    ik = getattr(e, "ik", None)
                    if ik is None or ik.shape[1] < p:
                        _log.info(
                            "APC ckpt disk miss: qsa ik %d != %d",
                            0 if ik is None else int(ik.shape[1]), p)
                        return None, 0
                    st = kv_state + (ik[:, :p],)
                    if getattr(e, "pos", None) is not None:
                        st = st + (e.pos[:, :, :p],)
                    kc = _qsa_cache_cls()(int(tag.split(":")[1]))
                else:
                    kc = KVCache()
                    st = kv_state
                kc.state = st
                if kc.offset != p:
                    _log.info("APC ckpt disk miss: kv assembled %d != %d",
                              kc.offset, p)
                    return None, 0
                warm.append(kc)
                kv_i += 1
            elif _is_rot(tag):
                warm.append(layer_rot[rot_i])
                rot_i += 1
            elif _is_kvarn(tag):
                if int(getattr(e, "offset", -1)) != p:
                    _log.info(
                        "APC ckpt disk miss: kvarn offset %d != %d",
                        int(getattr(e, "offset", -1)), p)
                    return None, 0
                # Clone like the record path: the exact tier may serve
                # this entry list again, and adopted kvarn buffers
                # mutate in place.
                clone = _clone_single_row(e)
                if clone is None:
                    return None, 0
                warm.append(clone)
            else:
                warm.append(e)
        targets = []
        for c in warm:
            if getattr(c, "keys", None) is not None:
                targets.extend([c.keys, c.values])
            elif hasattr(c, "cache"):
                targets.extend(s for s in c.cache if s is not None)
            else:
                targets.extend(_iter_arrays(getattr(c, "state", None)))
            if getattr(c, "ik", None) is not None:
                targets.append(c.ik)
        # Scratch survivors: the warm-cache arrays were built by this
        # restore from pool blocks; the except path's release-and-decline
        # is the right exit and now runs post-drain.
        from gmlx.eval_guard import guard
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
        # One swallow per release: a failing first release must not leak
        # the second block set out of the manager.
        for blks in (blocks, wblocks):
            try:
                manager.release(blks)
            except Exception:  # noqa: S110 - best-effort release on the failure path
                pass
        _log.warning("APC ckpt disk lookup failed; continuing",
                     exc_info=True)
        return None, 0


def _truncate_kv_snapshot(snap: list[Any], n: int) -> list[Any] | None:
    """Trim a row snapshot to its first ``n`` tokens, or None if any layer
    cannot be truncated faithfully (rotation, recurrent state, pooling)."""
    from .kv_cache import KVCache

    for c in snap:
        if not isinstance(c, KVCache) or int(getattr(c, "offset", 0)) < n:
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


def row_kv_len(prompt_cache, row: int = 0) -> int | None:
    """Tokens of KV the cache holds for ``row``: the max per-row offset
    across layers (batch classes carry one offset per row; single-row
    classes a scalar; CacheList members are read through). None when no
    layer reports one. Cheap - no clone - so a store can pick its key
    from what the row actually covers."""
    best = None
    for c in prompt_cache or ():
        members = getattr(c, "caches", None)
        if isinstance(members, (tuple, list)):
            n = row_kv_len(list(members), row)
        else:
            off = getattr(c, "offset", None)
            if off is None:
                continue
            try:
                if hasattr(off, "ndim") and off.ndim >= 1:
                    if row >= off.shape[0]:
                        continue
                    n = int(off[row].item())
                else:
                    n = int(off)
            except (TypeError, ValueError, AttributeError):
                continue
        if n is not None and (best is None or n > best):
            best = n
    return best


class _Snap(tuple):
    """A ring entry, ``(p, states)``, carrying the order it was taken in
    (``seq``) and its bytes (``size``) for the governor."""

    seq: int
    size: int


class _SnapRing(list):
    """A request's decode snapshot ring. Every live ring is listed, so the
    governor can free the snapshots under memory pressure, oldest first;
    the ring takes its next snapshot at its next boundary."""

    __slots__ = ("__weakref__",)


# Live rings by registration number: a list is not hashable, so a weak
# set cannot hold one.
_RINGS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_RING_LOCK = threading.Lock()
_RING_SEQ = itertools.count()


def _snap_ring(stash: dict) -> _SnapRing:
    ring = stash.get("snaps")
    if not isinstance(ring, _SnapRing):
        ring = _SnapRing(ring or ())
        stash["snaps"] = ring
        with _RING_LOCK:
            _RINGS[next(_RING_SEQ)] = ring
    return ring


def snapshot_ring_bytes() -> int:
    """Bytes the live decode snapshot rings hold. Attribute reads only."""
    with _RING_LOCK:
        return sum(getattr(s, "size", 0)
                   for r in list(_RINGS.values()) for s in list(r))


def evict_snapshot_rings(fraction: float) -> int:
    """Free ``fraction`` of the bytes the live decode snapshot rings hold,
    oldest snapshot first, for the governor. Returns the bytes freed. A
    request whose snapshots go stores its retirement from the snapshots
    it takes after, or stores none."""
    fraction = min(1.0, max(0.0, float(fraction)))
    if fraction <= 0.0:
        return 0
    with _RING_LOCK:
        held = sorted(((getattr(s, "seq", -1), id(s), r, s)
                       for r in list(_RINGS.values()) for s in list(r)),
                      key=lambda e: e[:2])
        total = sum(getattr(e[3], "size", 0) for e in held)
        want = total if fraction >= 1.0 else math.ceil(total * fraction)
        drop: dict = {}
        freed = 0
        for _, _, r, s in held:
            if fraction < 1.0 and freed >= want:
                break
            drop.setdefault(id(r), (r, set()))[1].add(id(s))
            freed += getattr(s, "size", 0)
        for r, ids in drop.values():
            r[:] = [s for s in r if id(s) not in ids]
    if drop:
        _log.info("APC decode ckpt: evicted %d snapshots, %.2f GB, for "
                  "memory", sum(len(v[1]) for v in drop.values()),
                  freed / 1e9)
    return freed


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
                     generated: list[int], row: int | None = None) -> None:
    """Advance the decode-time snapshot ring for a live ckpt row.

    ``stash`` is the request's retirement context (``snaps`` /
    ``snap_next`` / ``snap_frozen`` live here and die with it). ``row``
    is the request's row in a batch of two or more, whose states the
    snapshot takes from that row alone; None reads ``prompt_cache`` as
    the row's own. Cheap off-boundary; never raises into the decode
    loop, and a failure disables the ring for the request rather than
    retrying per token.
    """
    try:
        if stash.get("snap_frozen") or not stash.get("snap_ok"):
            return
        interval = env_int("GMLX_APC_DECODE_CKPT", _DECODE_CKPT_DEFAULT)
        ctx = stash.get("render_ctx")
        if interval <= 0 or ctx is None:
            return
        full_ids = stash["full_ids"]
        nxt = stash.get("snap_next")
        if nxt is not None and len(full_ids) + len(generated) < nxt:
            # The row's cache holds at most its tokens so far. A batch
            # class keeps its offsets in arrays, and reading one waits
            # for the step in flight, so they are read once a boundary
            # is due.
            return
        if row is not None and any(
                _is_kvarn(str(t)) for t in stash.get("layout") or ()):
            # A kvarn record carries its whole KV inline, so the snapshot
            # of a row in a batch would copy that KV at every boundary,
            # and requantize it when the row is padded. The row takes its
            # snapshots once it decodes alone.
            return
        if row is None:
            p = _cache_offset_max(prompt_cache)
        else:
            p = row_kv_len(prompt_cache, row) or 0
        # The render retokenizes the whole sequence, so its cost grows
        # with p; widening the interval with p keeps prediction a bounded
        # fraction of decode time.
        eff = max(interval, p >> 6)
        grid = int(stash.get("snap_grid") or 1)
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
            ring = _snap_ring(stash)
            with _RING_LOCK:
                ring[:] = [s for s in ring if s[0] <= pred]
            stash["snap_frozen"] = True
            _log.info("APC decode ckpt frozen: render diverges at %d "
                      "(live %d)", pred, p)
            return
        from .kv_cache import BatchKVCache, KVCache, RotatingKVCache
        states = []
        pending: list[Any] = []
        for c in prompt_cache:
            # Plain KV is sliced from the row at retirement. A row that
            # was joined holds it in the batch class, or in a cache of one
            # cache per row (qwen4exp's QSA layers), which the retirement
            # extracts.
            if isinstance(c, BatchKVCache) or getattr(
                    c, "holds_row_caches", False) or (
                    isinstance(c, KVCache)
                    and not isinstance(c, RotatingKVCache)):
                continue
            if row is not None:
                # The row's view, copied once by the clone below.
                c = (_row_view(c) or c.extract)(row)
            clone = _clone_single_row(c, pending)
            if clone is None:
                stash["snap_ok"] = False
                _log.info("APC decode ckpt disabled: uncloneable cache %s",
                          type(c).__name__)
                return
            states.append(clone)
        # The ring holds two. The oldest makes room before the new
        # snapshot materializes, so it never holds three.
        ring = _snap_ring(stash)
        with _RING_LOCK:
            del ring[:-1]
        if pending:
            from gmlx.eval_guard import guard
            guard.eval(*pending, site="decode-ckpt-tick", owner="owned")
        from .kv_cache import cache_nbytes
        snap = _Snap((p, states))
        snap.seq, snap.size = next(_RING_SEQ), cache_nbytes(states)
        with _RING_LOCK:
            ring.append(snap)
        stash["snap_next"] = _grid_ceil(p + eff, grid)
        _log.info("APC decode ckpt snapshot: p=%d pred=%d", p, pred)
    except Exception:
        stash["snap_ok"] = False
        _log.warning("APC decode ckpt tick failed; ring disabled for "
                     "this request", exc_info=True)


def _snap_assemble(prompt_cache: list[Any], states: list[Any],
                   p: int) -> list[Any] | None:
    """Rebuild a single-row cache list at snapshot position ``p``: plain
    KV and QSA layers sliced from the live row, rotating windows and
    recurrent states from the snapshot clones."""
    from .kv_cache import KVCache, RotatingKVCache

    out = []
    si = 0
    for c in prompt_cache:
        if getattr(c, "holds_row_caches", False):
            # A batched QSA row: the tick took no snapshot of it.
            return None
        if _qsa_tag(c) is not None:
            # K, V, indexer keys and positions to p, as the prefix arm of
            # the layer's merge takes them, so the record tags "qsa:R" as
            # the lookups sign. Views of the live row, set past the state
            # setter, which copies and evaluates each stream: the store
            # copies what it keeps, so its copies are the only ones.
            if int(c.offset) < p:
                return None
            st = c.state
            if st[0] is None or st[2] is None:
                return None
            q = type(c)(c.ratio)
            q._kb, q._vb = st[0][..., :p, :], st[1][..., :p, :]
            q._ib = st[2][:, :p]
            q.pos = None if len(st) == 3 else st[3][:, :, :p]
            q.offset = p
            out.append(q)
            continue
        if isinstance(c, KVCache) and not isinstance(c, RotatingKVCache):
            if c.keys is None or int(getattr(c, "offset", 0) or 0) < p:
                return None
            k = KVCache()
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


def _retire_usable(manager, caches, layout, refused) -> bool:
    """Whether a retirement record of ``caches`` can serve a lookup: its
    layout is the one the request's lookup signed with (``layout``; None
    when the arming recorded none). A row whose QSA layers come back as
    plain KV, as the padded batch class leaves them, tags "kv" where every
    lookup signs "qsa:R": such a record would cost the room and strip its
    chain for nothing, so it is not stored. A refused layout goes on
    ``refused``."""
    if layout is None:
        return True
    got = ckpt_layout(caches, int(manager.block_size))
    if got is None or tuple(got) == tuple(layout):
        return True
    refused.append(list(got))
    return False


def _ckpt_retirement(manager, ids, prompt_cache, *, extra_hash,
                     max_len, decode_snaps, layout=None) -> int:
    """Ckpt-mode retirement: the full sequence when it can store whole
    (a short rotating prompt truncates to the block grid), else the
    newest decode-time snapshot at or below the replayable prefix. Never spills to the exact tier -- on ckpt models the exact
    tier stays empty, so the stock warm path never bypasses arming.
    Nothing a lookup signed ``layout`` cannot use (``_retire_usable``):
    one retire_layout decline and one line per retirement, however many
    candidates it refused.
    Returns the stored length (0 = nothing)."""
    refused: list = []
    try:
        cap = len(ids)
        if max_len is not None and max_len < len(ids):
            cap = max_len
        elif len(ids) >= 2 and _retire_usable(manager, prompt_cache, layout,
                                              refused):
            # A row still on its single-row caches is sliced directly:
            # the stores copy per block, with no clone of the row. A row
            # taken out of batch classes arrives as one copy of its KV. A
            # rotating layer below the window stores its block-grid
            # prefix (grid_truncate); the ring below holds aligned
            # clones for the post-wrap off-grid cases.
            stored = ckpt_store(manager, ids, prompt_cache,
                                extra_hash=extra_hash,
                                grid_truncate=True, kind="retire")
            if stored:
                return stored
        snap_refused = False
        for p, states in sorted(decode_snaps or [],
                                key=lambda s: s[0], reverse=True):
            if not 2 <= p <= cap:
                continue
            assembled = _snap_assemble(prompt_cache, states, p)
            if assembled is None:
                continue
            if not _retire_usable(manager, assembled, layout, refused):
                snap_refused = True
                continue
            # skeleton_disk stays on: this is the entry a post-restart
            # turn restores from, unlike interval boundaries.
            if ckpt_store(
                    manager, ids[:p], assembled, extra_hash=extra_hash,
                    skeleton_disk=True, kind="retire"):
                _log.info("APC retirement: decode ckpt stored at %d "
                          "(cap %d, full %d)", p, cap, len(ids))
                return p
        # No snapshot at or below the replayable prefix. cap < len only
        # when the next-turn render prediction SUCCEEDED: the client came
        # through the chat template and provably re-renders, so a
        # full-sequence store under the verbatim key can never match its
        # next turn -- it only ages in the pool and burns an
        # ids_diverged decline on every later same-chain lookup (the
        # gemma-4/gpt-oss cert arms: ~410 declines each, every one a
        # retire-kind record past the strip boundary). The prefill-side
        # p_stable turn boundary of this same request covers turn-2, so
        # storing nothing here loses nothing reachable. Raw-continuation
        # clients have no render ctx (lcp None -> cap == len) and keep
        # the whole-sequence branch above. Counter kept for visibility;
        # the 90-percent-occupancy yield (122B churn spiral) is subsumed.
        if cap < len(ids) and len(ids) >= 2:
            _ckpt_bump(manager, "retire_fallback_suppressed")
            _log.info(
                "APC retirement: predicted-diverged fallback suppressed "
                "(replayable prefix %d, full %d)", cap, len(ids))
            return 0
        if not snap_refused:
            _log.info("APC retirement skipped: no decode snapshot at or "
                      "below the replayable prefix %d (full %d)",
                      cap, len(ids))
        return 0
    except Exception:
        _log.warning("APC ckpt retirement failed; continuing",
                     exc_info=True)
        return 0
    finally:
        if refused:
            _ckpt_decline(manager, "retire_layout")
            _log.info("APC retirement: %d candidate(s) skipped, layout %s "
                      "matches no lookup (%s)", len(refused), refused[0],
                      list(layout or ()))


_RETIRE_BLOCKED = [0]


@contextmanager
def retirement_blocked():
    """No retirement store runs in the scope. The tick guard drops the
    engine's state after a command buffer failure: that state is not
    valid, and closing a round generator would retire it."""
    _RETIRE_BLOCKED[0] += 1
    try:
        yield
    finally:
        _RETIRE_BLOCKED[0] -= 1


def retirement_is_blocked() -> bool:
    """Whether a ``retirement_blocked`` scope is open: a caller skips the
    work it does for a retirement store, which would store nothing."""
    return _RETIRE_BLOCKED[0] > 0


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
    layout: tuple | None = None,
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
    exact tier. ``layout``: the signature the request's lookup signed
    with; a ckpt record that cannot match it is not stored. Best effort:
    a failure never breaks generation.

    Returns the stored prefix length in tokens (0 = nothing stored) --
    the fallback can store past ``max_len``, so callers must log the
    return, not the cap.
    """
    if manager is None or token_ids is None or _RETIRE_BLOCKED[0]:
        return 0
    ids = [int(t) for t in token_ids]
    if mode == "ckpt":
        return _ckpt_retirement(manager, ids, prompt_cache,
                                extra_hash=extra_hash, max_len=max_len,
                                decode_snaps=decode_snaps, layout=layout)
    if max_len is not None and max_len < len(ids) and mode != "exact":
        ids = ids[:max_len]
    if len(ids) < 2:
        return 0
    try:
        if mode == "exact":
            return _exact_retirement(manager, ids, prompt_cache, row=row,
                                     extra_hash=extra_hash, max_len=max_len)
        from gmlx._vendor.mlx_vlm import apc as _apc
        blocks = _apc.harvest_blocks_from_batch_cache(
            manager, prompt_cache, ids, batch_idx=row, extra_hash=extra_hash)
        manager.release(blocks)
        if not blocks:
            return 0
        bs = int(getattr(manager, "block_size", 0) or 0)
        return len(blocks) * bs if bs else len(ids)
    except Exception:
        _log.warning("APC retirement store failed; continuing", exc_info=True)
        return 0


def _exact_retirement(manager: Any, ids: list[int], prompt_cache: list[Any],
                      *, row: int, extra_hash: int,
                      max_len: int | None) -> int:
    """Store ``row`` in the exact tier as one copy of its KV.

    The snapshot copies nothing yet (``_row_snapshot`` lazy), and it is
    cloned the way the tier clones, unevaluated too. Once the snapshot is
    dropped only the clone's graph holds it, so the store makes room for
    the clone's bytes and then materializes the clone as the row's only
    copy. A truncated plain KV row slices views, so the copy holds only
    the kept prefix.
    """
    from gmlx._vendor.mlx_vlm import apc as _apc

    snap = _row_snapshot(prompt_cache, row, lazy=True)
    if snap is None:
        return 0
    covered = _cache_offset_max(snap)
    if covered and covered != len(ids):
        # An exact entry replays bitwise from its key; a row whose KV count
        # disagrees with the key (a batch row with a pending or stale tail)
        # must not be stored under it.
        _log.warning("APC retirement skipped: row covers %d tokens, key %d",
                     covered, len(ids))
        return 0
    if max_len is not None and max_len < len(ids):
        if max_len < 2:
            return 0
        snap = _truncate_kv_snapshot(snap, max_len)
        if snap is None:
            _log.info("APC retirement skipped: snapshot not truncatable to "
                      "the replayable prefix %d", max_len)
            return 0
        ids = ids[:max_len]
    pending: list = []
    entry = [_apc._clone_cache_entry_for_apc(
        c, min_capacity_tokens=None, eval_targets=pending) for c in snap]
    if any(c is None for c in entry):
        # The tier's own clone declines the row too, and books the reject.
        stored = manager.store_exact_cache(ids, snap, extra_hash=extra_hash)
        return len(ids) if stored else 0
    del snap
    if manager.store_exact_cache(ids, entry, extra_hash=extra_hash,
                                 materialize=pending):
        return len(ids)
    return 0
