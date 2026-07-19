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
    if eval_targets:
        import mlx.core as mx
        mx.eval(*eval_targets)
    return out


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


# Hybrid checkpoint tier: for gated-delta hybrids (plain KVCache attention
# layers + ArraysCache recurrent layers), a checkpoint at position p stores
# the attention layers' KV through the shared block pool (incremental,
# deduped, disk-persistent) and the recurrent states + a 1..block_size-token
# attention-KV tail as a small exact-tier "sidecar" entry keyed on
# tokens[:p]. This replaces the exact tier's full-cache clones (GBs per
# entry at depth, duplicated per turn) with shared blocks plus ~10s-of-MB
# sidecars. Both keyspaces are salted: block hashes ignore layer count, so
# the salt is what guarantees these attn-subset blocks can never satisfy (or
# be satisfied by) any full-layer store.

_CKPT_SALT = 0x7C_4B_D2_0A_86_E5_93


def ckpt_extra_hash(extra_hash: int) -> int:
    return int(extra_hash) ^ _CKPT_SALT


def ckpt_supported(prompt_cache) -> bool:
    """True for the gated-delta hybrid cache shape: every layer a plain
    KVCache or an ArraysCache, at least one of each. Rotating/chunked/
    quantized layers (gemma-4 et al) stay on the exact tier."""
    from .cache_compat import cache_types

    if not prompt_cache:
        return False
    kv_types = cache_types("KVCache")
    arr_types = cache_types("ArraysCache")
    has_kv = has_arr = False
    for c in prompt_cache:
        if isinstance(c, kv_types):
            has_kv = True
        elif isinstance(c, arr_types):
            has_arr = True
        else:
            return False
    return has_kv and has_arr


def _ckpt_block_prefix(p: int, block_size: int) -> int:
    """Tokens covered by whole blocks below a checkpoint at ``p``. Always
    leaves a 1..block_size-token tail for the sidecar: recurrent state is
    captured at exactly ``p`` and cannot rewind, so the tail bridges the
    block grain -- and a never-empty tail KVCache keeps the sidecar entry
    round-trippable through the exact serializer."""
    return ((p - 1) // block_size) * block_size


def ckpt_store(
    manager: Any,
    token_ids,
    prompt_cache: list[Any],
    *,
    extra_hash: int = 0,
) -> bool:
    """Store a hybrid checkpoint at ``p = len(token_ids)``.

    ``prompt_cache`` must be a single-row cache list whose KV offsets equal
    ``p`` exactly (mid-prefill at the aligned checkpoint column,
    post-prefill, or a guarded retirement row); a mismatch means the cache
    does not faithfully cover the key and the store is skipped. The block
    store dedups against the existing chain, so re-walking an already-stored
    prefix copies nothing. Best-effort; never raises.
    """
    if manager is None or token_ids is None:
        return False
    from .cache_compat import cache_types, runtime_cache_module

    kv_types = cache_types("KVCache")
    try:
        ids = [int(t) for t in token_ids]
        p = len(ids)
        if p < 2 or not ckpt_supported(prompt_cache):
            return False
        for c in prompt_cache:
            if isinstance(c, kv_types) and int(c.offset) != p:
                _log.info(
                    "APC ckpt store skipped: KV offset %d != %d",
                    int(c.offset), p)
                return False
        bs = int(manager.block_size)
        b_full = _ckpt_block_prefix(p, bs)
        salted = ckpt_extra_hash(extra_hash)
        n_blocks = 0
        if b_full > 0:
            lk = [c.keys[..., :b_full, :]
                  for c in prompt_cache if isinstance(c, kv_types)]
            lv = [c.values[..., :b_full, :]
                  for c in prompt_cache if isinstance(c, kv_types)]
            blocks = manager.store_kv_blocks(
                ids[:b_full], lk, lv, extra_hash=salted)
            n_blocks = len(blocks)
            manager.release(blocks)
        sidecar: list[Any] = []
        for c in prompt_cache:
            if isinstance(c, kv_types):
                # Runtime-origin tail: store_exact_cache's clone/serialize
                # gates isinstance on the mlx-vlm runtime's classes.
                tail = runtime_cache_module().KVCache()
                # Lazy views into the live cache; store_exact_cache deep-
                # clones (and evals) before anything can mutate them.
                tail.state = (c.keys[..., b_full:p, :],
                              c.values[..., b_full:p, :])
                sidecar.append(tail)
            else:
                sidecar.append(c)
        ok = bool(manager.store_exact_cache(ids, sidecar, extra_hash=salted))
        if ok:
            _log.info(
                "APC ckpt store: tokens=%d blocks=%d tail=%d",
                p, n_blocks, p - b_full)
        return ok
    except Exception:
        _log.warning("APC ckpt store failed; continuing", exc_info=True)
        return False


def ckpt_lookup(
    manager: Any,
    token_ids,
    *,
    extra_hash: int = 0,
    min_prefix_tokens: int = 0,
) -> tuple:
    """Longest checkpoint-tier warm start for ``token_ids``.

    Finds the longest salted sidecar at some ``p`` (memory LRU then disk,
    all upstream exact machinery -- never serves ``p == len(token_ids)``),
    then assembles the attention KV from the salted block chain below ``p``
    plus the sidecar tail, and the recurrent layers from the sidecar states.
    Returns ``(warm_prompt_cache, p)`` with KV offsets exactly ``p``, or
    ``(None, 0)``. A sidecar whose block chain is incomplete (evicted from
    both memory and disk) is a miss: recurrent state cannot bridge a KV
    hole. Best-effort; never raises.
    """
    if manager is None or token_ids is None:
        return None, 0
    import mlx.core as mx

    from .cache_compat import cache_types, runtime_cache_module

    kv_types = cache_types("KVCache")
    blocks: list[Any] = []
    try:
        ids = [int(t) for t in token_ids]
        salted = ckpt_extra_hash(extra_hash)
        sidecar, p = manager.lookup_exact_cache(
            ids, extra_hash=salted, min_prefix_tokens=min_prefix_tokens)
        if sidecar is None or p <= 0:
            return None, 0
        bs = int(manager.block_size)
        b_full = _ckpt_block_prefix(p, bs)
        n_attn = sum(1 for e in sidecar if isinstance(e, kv_types))
        layer_kv = None
        if b_full > 0:
            blocks, matched = manager.lookup_prefix(
                ids[:b_full], extra_hash=salted)
            if matched >= b_full and blocks and len(blocks[0].keys) == n_attn:
                layer_kv = [
                    (mx.concatenate([b.keys[j] for b in blocks], axis=2),
                     mx.concatenate([b.values[j] for b in blocks], axis=2))
                    for j in range(n_attn)
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
                        or len(disk_caches) != n_attn):
                    _log.info(
                        "APC ckpt miss: sidecar at %d, block chain %d/%d",
                        p, max(matched, dmatched), b_full)
                    return None, 0
                layer_kv = [
                    (c.keys[..., :b_full, :], c.values[..., :b_full, :])
                    for c in disk_caches
                ]
        warm: list[Any] = []
        j = 0
        for entry in sidecar:
            if isinstance(entry, kv_types):
                t_len = int(entry.offset)
                if t_len != p - b_full:
                    manager.release(blocks)
                    _log.warning(
                        "APC ckpt miss: sidecar tail %d != %d", t_len,
                        p - b_full)
                    return None, 0
                tk = entry.keys[..., :t_len, :]
                tv = entry.values[..., :t_len, :]
                kc = runtime_cache_module().KVCache()
                if layer_kv is not None:
                    kc.state = (
                        mx.concatenate([layer_kv[j][0], tk], axis=2),
                        mx.concatenate([layer_kv[j][1], tv], axis=2))
                else:
                    kc.state = (tk, tv)
                j += 1
                warm.append(kc)
            else:
                # ArraysCache: lookup_exact_cache returned a decoupled clone;
                # it is ours to hand to the live batch.
                warm.append(entry)
        targets: list[Any] = []
        for c in warm:
            if isinstance(c, kv_types):
                targets.extend([c.keys, c.values])
            else:
                targets.extend(s for s in c.cache if s is not None)
        mx.eval(*targets)
        # Concats are materialized copies now; the pool may recycle.
        manager.release(blocks)
        blocks = []
        _log.info(
            "APC ckpt hit: prefix=%d blocks=%d tail=%d",
            p, b_full // bs, p - b_full)
        return warm, p
    except Exception:
        try:
            manager.release(blocks)
        except Exception:
            pass  # best-effort release on the failure path
        _log.warning("APC ckpt lookup failed; continuing", exc_info=True)
        return None, 0


def retirement_store(
    manager: Any,
    mode: str | None,
    token_ids,
    prompt_cache: list[Any],
    *,
    row: int = 0,
    extra_hash: int = 0,
) -> bool:
    """Persist a finished row's full KV into the shared APC.

    ``token_ids`` is the full sequence (prompt + generated); it must be the
    request's original full_input_ids plus the emitted tokens, never a
    suffix-only serve-layer ``prompt_tokens`` (which is trimmed on a warm turn).
    Exact mode snapshots the row and stores it whole (the rotating chat
    fleet); ckpt mode stores blocks + sidecar (gated-delta hybrids, B=1
    rows only in v1); block mode harvests the row's blocks into the shared
    pool. Best-effort: a failure never breaks generation.
    """
    if manager is None or token_ids is None:
        return False
    ids = [int(t) for t in token_ids]
    if len(ids) < 2:
        return False
    try:
        if mode == "ckpt":
            # The row is already single-row on the B=1 path; ckpt_store
            # slices it directly (its own stores copy internally), so no
            # full-cache clone happens -- the exact tier's whole sin.
            return ckpt_store(manager, ids, prompt_cache,
                              extra_hash=extra_hash)
        if mode == "exact":
            snap = row_snapshot(prompt_cache, row)
            if snap is None:
                return False
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
