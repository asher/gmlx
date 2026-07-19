"""LRU prefix cache for speculative-path automatic prefix caching (APC).

Caches target KV state and hidden after prefill so that subsequent requests
sharing a token prefix (system prompt + conversation history) skip re-prefill
of the shared prefix through the target model's 28+ transformer layers.

Architecture-safe: handles KVCache (standard), RotatingKVCache (gemma-4
sliding-window layers, whose .state setter loses offset/_idx), ArraysCache
(Qwen3.5/3.6 gated-delta conv/SSM state), and CacheList (hybrids with
KV+GDN sub-caches).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import mlx.core as mx

_log = logging.getLogger(__name__)

_CACHELIST_TAG = "_CacheList"
_ARRAYS_TAG = "_ArraysCache"
_POOLING_TAG = "_PoolingCache"


def _trim_rotating_state(c: Any, keys: mx.array, values: mx.array):
    """Trim a sliding-window cache snapshot to its window.

    During chunked prefill the rotating cache accumulates the full prompt
    (trim is deferred to the first decode step for chunked/unchunked output
    parity), so a post-prefill snapshot would pin O(prompt_len) per sliding
    layer -- tens of GB at 32k+ depth on window-heavy models. Only the last
    ``max_size`` positions (plus the ``keep`` head) can ever be attended
    again, so the snapshot keeps exactly the state the first decode step's
    trim would produce.
    """
    window = int(getattr(c, "max_size", 0) or 0)
    n = int(keys.shape[2])
    if window <= 0:
        return keys, values
    # Reorder before any early return: a ring-wrapped buffer at exactly the
    # window length (post-wrap prefill ending on _update_in_place) is in ring
    # order, and the snapshot is tagged temporally ordered.
    order = getattr(c, "_temporal_order", None)
    if callable(order) and int(getattr(c, "_idx", n)) != n:
        keys = order(keys)
        values = order(values)
    if n <= window:
        return keys, values
    keep = min(max(int(getattr(c, "keep", 0) or 0), 0), window)
    tail = window - keep
    if keep > 0:
        keys = mx.concatenate([keys[..., :keep, :], keys[..., -tail:, :]], axis=2)
        values = mx.concatenate(
            [values[..., :keep, :], values[..., -tail:, :]], axis=2)
    else:
        keys = keys[..., -tail:, :]
        values = values[..., -tail:, :]
    return keys, values


def _owned_pair(keys: mx.array, values: mx.array):
    """Copies that share no mutable buffer with the live cache. In-place
    cache updates (RotatingKVCache._update_in_place) __setitem__-mutate the
    assigned object, so snapshot and live cache must never hold the same
    array. K-eq-V layers keep their single shared array.

    ``mx.contiguous`` allocates an owned buffer even for already-contiguous
    input; that is an mlx implementation detail, not API contract, so the
    ownership invariant is pinned by test_restore_does_not_alias_stored_entry
    and test_stored_entry_survives_continued_live_decode - if an mlx upgrade
    makes contiguous() alias, those fail and this needs a true copy op."""
    k = mx.contiguous(keys)
    v = k if values is keys else mx.contiguous(values)
    return k, v


def _snapshot_entry(c: Any) -> Any:
    """Snapshot one prompt_cache entry, preserving offset for all cache types."""
    from .cache_compat import cache_types
    from .deepseek_v4_cache import BatchPoolingCache, PoolingCache
    if isinstance(c, cache_types("CacheList")):
        return (_CACHELIST_TAG, [_snapshot_entry(sub) for sub in c.caches])
    if isinstance(c, (PoolingCache, BatchPoolingCache)):
        # DeepSeek-V4 pooled-KV cache: no keys/values attrs, so the generic
        # branch would silently corrupt it on restore. state slices are lazy
        # views over the pre-mutation buffer node; _eval_snapshot pins them.
        return (_POOLING_TAG, c.state, c.meta_state)
    if isinstance(c, cache_types("ArraysCache")):
        return (_ARRAYS_TAG, list(c.cache), c.left_padding, c.lengths)
    state = c.state
    offset = getattr(c, "offset", state[0].shape[2])
    _idx = getattr(c, "_idx", 0)
    if isinstance(c, cache_types("RotatingKVCache")):
        # When no trim/reorder applies, the returned arrays are the live ring
        # buffers, which decode mutates in place; _owned_pair decouples them.
        keys, values = _owned_pair(*_trim_rotating_state(c, state[0], state[1]))
        # _idx == buffer length marks the buffer temporal-ordered, which the
        # trimmed snapshot is by construction; offset keeps the true position.
        return (keys, values, int(offset), int(keys.shape[2]))
    return (state[0], state[1], int(offset), int(_idx))


def _restore_entry(c: Any, snap: Any) -> None:
    """Restore one prompt_cache entry from a snapshot."""
    from .cache_compat import cache_types
    if isinstance(snap, tuple) and snap[0] == _CACHELIST_TAG:
        if isinstance(c, cache_types("CacheList")):
            for sub, sub_snap in zip(c.caches, snap[1]):
                _restore_entry(sub, sub_snap)
        return
    if isinstance(snap, tuple) and snap[0] == _POOLING_TAG:
        _, state, meta_state = snap
        c.meta_state = meta_state  # ratio first: the state setter re-pools
        c.state = state
        return
    if isinstance(snap, tuple) and snap[0] == _ARRAYS_TAG:
        _, cache_list, left_padding, lengths = snap
        c.cache = [mx.contiguous(a) if isinstance(a, mx.array) else a
                   for a in cache_list]
        c.left_padding = left_padding
        c.lengths = lengths
        return
    keys, values, offset, _idx = snap
    c.keys, c.values = _owned_pair(keys, values)
    c.offset = offset
    if hasattr(c, "_idx"):
        c._idx = _idx


def _eval_snapshot(snaps: list[Any]) -> None:
    """Materialize all arrays in a snapshot list so they survive cache mutation.

    Shares the tag/length dispatch with :func:`_collect_snapshot_arrays` (one
    batched ``mx.eval`` instead of one per CacheList nesting level - identical
    materialization)."""
    arrays: list[mx.array] = []
    _collect_snapshot_arrays(snaps, arrays)
    if arrays:
        mx.eval(arrays)


def _collect_snapshot_arrays(snaps: list[Any], out: list[mx.array]) -> None:
    for snap in snaps:
        if isinstance(snap, tuple) and len(snap) >= 2 and snap[0] == _CACHELIST_TAG:
            _collect_snapshot_arrays(snap[1], out)
        elif isinstance(snap, tuple) and len(snap) == 3 and snap[0] == _POOLING_TAG:
            out.extend(a for a in snap[1] if isinstance(a, mx.array))
        elif isinstance(snap, tuple) and len(snap) == 4 and snap[0] == _ARRAYS_TAG:
            _, cache_list, left_padding, lengths = snap
            out.extend(a for a in cache_list if a is not None)
            if left_padding is not None:
                out.append(left_padding)
            if lengths is not None:
                out.append(lengths)
        elif isinstance(snap, tuple) and len(snap) == 4:
            out.extend([snap[0], snap[1]])


def _entry_nbytes(kv_snaps: list[Any], hidden: mx.array) -> int:
    """Bytes pinned by one entry. Dedup by array identity: K-eq-V layers
    store the same array as both K and V and must be counted once."""
    arrays: list[mx.array] = [hidden]
    _collect_snapshot_arrays(kv_snaps, arrays)
    seen = set()
    total = 0
    for a in arrays:
        if id(a) in seen:
            continue
        seen.add(id(a))
        total += a.nbytes
    return total


class _PrefixEntry:
    __slots__ = ("token_ids", "kv_snaps", "hidden", "nbytes")

    def __init__(
        self,
        token_ids: tuple[int, ...],
        kv_snaps: list[Any],
        hidden: mx.array,
    ):
        self.token_ids = token_ids
        self.kv_snaps = kv_snaps
        self.hidden = hidden
        self.nbytes = _entry_nbytes(kv_snaps, hidden)


class SpecPrefixCache:
    """LRU cache of target KV + hidden snapshots keyed by token prefix.

    Designed for the chat use case where each turn shares system_prompt +
    conversation_history as a token prefix with the previous turn.

    Parameters
    ----------
    max_entries : int
        Maximum number of cached prefixes (default 4). Each entry stores
        per-layer KV arrays + the full hidden tensor, so memory scales with
        model size and prefix length (~240 MB per 2K-prefix entry on a 27B
        model, less on gemma-4 due to KV-shared layers).
    min_prefix : int
        Minimum prefix length worth caching (default 32). Shorter prefixes
        are not stored and never returned from lookup.
    max_bytes : int
        Total byte budget for pinned snapshots (default 8 GiB). Entries are
        evicted LRU until under budget; a single entry larger than the whole
        budget is never stored. Full-attention KV at deep context is the
        dominant term (a 32k-prefix entry on a large model runs to GBs), so
        an entry count alone does not bound memory.
    """

    def __init__(self, max_entries: int = 4, min_prefix: int = 32,
                 max_bytes: int = 8 << 30):
        self._entries: OrderedDict[tuple[int, ...], _PrefixEntry] = OrderedDict()
        self._max = max_entries
        self._min_prefix = min_prefix
        self._max_bytes = max_bytes
        self._total_bytes = 0

    def lookup(
        self, token_ids: mx.array
    ) -> tuple[int, _PrefixEntry] | None:
        """Find the longest cached prefix matching token_ids.

        Returns (prefix_len, entry) on hit, None on miss.  Requires at least
        one suffix token (``prefix_len < len(query)``) so the caller always
        has tokens to forward through the model for logits + shared_kv.
        Moves the matched entry to the head of the LRU on access.
        """
        if token_ids.ndim == 2:
            ids = tuple(int(x) for x in token_ids[0].tolist())
        else:
            ids = tuple(int(x) for x in token_ids.tolist())

        best: tuple[int, _PrefixEntry] | None = None
        best_len = 0
        for key, entry in self._entries.items():
            n = len(key)
            if n >= len(ids) or n <= best_len:
                continue
            if ids[:n] == key:
                best = (n, entry)
                best_len = n

        if best is not None and best_len >= self._min_prefix:
            self._entries.move_to_end(best[1].token_ids)
            return best
        return None

    def store(
        self,
        token_ids: mx.array,
        prompt_cache: list,
        hidden: mx.array,
    ) -> None:
        """Cache target KV state + hidden for this token sequence.

        Snapshots every cache entry in prompt_cache (architecture-agnostic:
        handles KVCache, RotatingKVCache, ArraysCache, CacheList) and the
        target hidden. Arrays are evaluated/materialized so they survive
        subsequent in-place cache mutation during decode.
        """
        if token_ids.ndim == 2:
            ids = tuple(int(x) for x in token_ids[0].tolist())
        else:
            ids = tuple(int(x) for x in token_ids.tolist())

        if len(ids) < self._min_prefix:
            return

        kv_snaps = [_snapshot_entry(c) for c in prompt_cache]
        entry = _PrefixEntry(ids, kv_snaps, hidden)

        # Budget check before materializing: nbytes needs only shapes, and
        # skipping the eval avoids a doomed multi-GB copy of deep-context KV.
        if entry.nbytes > self._max_bytes:
            _log.info(
                "APC store skipped: entry %.1f MB exceeds budget %.1f MB "
                "(prefix len=%d)",
                entry.nbytes / 2**20, self._max_bytes / 2**20, len(ids),
            )
            return

        _eval_snapshot(kv_snaps)
        mx.eval(hidden)

        if ids in self._entries:
            old = self._entries.pop(ids)
            self._total_bytes -= old.nbytes
        self._entries[ids] = entry
        self._entries.move_to_end(ids)
        self._total_bytes += entry.nbytes

        while self._entries and (
            len(self._entries) > self._max
            or self._total_bytes > self._max_bytes
        ):
            evicted_key, evicted = self._entries.popitem(last=False)
            self._total_bytes -= evicted.nbytes
            _log.debug(
                "APC evict prefix len=%d (%.1f MB)",
                len(evicted_key), evicted.nbytes / 2**20,
            )

    def restore(self, entry: _PrefixEntry, prompt_cache: list) -> None:
        """Restore target KV from a cached entry into prompt_cache."""
        for c, snap in zip(prompt_cache, entry.kv_snaps):
            _restore_entry(c, snap)

    def clear(self) -> None:
        self._entries.clear()
        self._total_bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __repr__(self) -> str:
        lens = [len(e.token_ids) for e in self._entries.values()]
        return (f"SpecPrefixCache(entries={len(self._entries)}, "
                f"prefix_lens={lens}, "
                f"bytes={self._total_bytes / 2**20:.1f}MB)")
