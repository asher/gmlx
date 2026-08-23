"""gmlx-owned APC manager: a subclass of mlx-vlm's APCManager plus its builder.

Ownership posture: gmlx behavior lands as subclass overrides here, never as
method assignments on the upstream class. The server's manager is constructed
by the load bridge (``server_bridge_vlm.load_model_resources``) from the same
env vars ``apc.from_env`` reads; ``from_env`` itself stays unpatched and
unused for gmlx-served loads - the residency build window pins
``APC_ENABLED=0`` around the stock load (after capturing the effective value
into ``GMLX_APC_ENABLED``), so the stock call returns ``None`` and never
builds a manager to discard.

Override rule: an override may wrap, pre/post-process, or replace an
inherited method outright, but does not duplicate an inherited body to edit
part of it; when a change would need a copied body, vendor instead.
``store_kv_blocks`` is the one standing copied body (the chunked eval is
mid-loop, unwrappable) - carried from the retired
``install_apc_batched_store_eval`` method patch, not repeated. It now also
carries the ckpt-store policy (``_ckpt_disk``: suppress layer-major,
optionally memory-only), a second in-body divergence - one more divergence
in this body is the agreed trigger to vendor the upstream file instead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mlx_vlm import apc as _apc

from .envflags import env_int

_log = logging.getLogger(__name__)

# Private upstream helpers the store override needs, resolved once. None means
# the pinned mlx-vlm drifted: the override falls back to the stock store (the
# cache still works, slower) and warns once.
_HELPERS = None
_HELPERS_FAILED = False


def _store_helpers():
    global _HELPERS, _HELPERS_FAILED
    if _HELPERS is not None or _HELPERS_FAILED:
        return _HELPERS
    try:
        _HELPERS = {
            "clone_lm": _apc._clone_layer_major_kv_cache_for_apc,
            "seq_hash": _apc._sequence_hash,
            "entry_cls": _apc.APCExactCacheEntry,
            "seed_parent": _apc.SEED_PARENT_HASH,
            "hash_tokens": _apc._hash_tokens,
            "disk_block_cls": _apc._DiskLayerMajorBlock,
            "copy_arr": _apc._copy_mlx_array,
            "time": _apc.time,
        }
    except AttributeError as e:
        _HELPERS_FAILED = True
        print(f"[apc] prompt-cache fast path not installed ({e}); the cache "
              "still works, slower - reinstall the pinned mlx-vlm "
              "(pip install mlx-vlm==0.6.4)")
    return _HELPERS


class GmlxAPCManager(_apc.APCManager):
    """APCManager with the per-block store eval batched into chunks.

    The stock store loop deep-copies each 16-token block's K/V slices and
    evaluates them one block at a time - ~0.4-0.5 ms of dispatch/sync per
    block regardless of data size, which compounds to ~1 s of synchronous
    prefill-thread stall for a 32k-token store on a 27B model. The copies
    are pure materialization (nothing in the loop consumes their values), so
    they can be evaluated in chunks: same data volume, ~30x fewer syncs.

    ``store_kv_blocks`` is a verbatim copy of the stock method with the eval
    hoisted into ``GMLX_APC_STORE_EVAL_CHUNK``-block batches (default 32;
    <=0 means a single eval before return). All evals still complete before
    the method returns, preserving the stock guarantee that block tensors
    are decoupled from the caller's cache before ``mx.clear_cache`` can
    release it. If the private helpers it references drift upstream, the
    override defers to the stock store instead.
    """

    def autosize(self, model, budget_fraction: float = None) -> None:
        """Size the caches to the box post-load. Pool: raise the block
        cap to a working-budget share when APC_NUM_BLOCKS is unset
        (blocks allocate lazily, so the cap costs nothing until
        committed; never shrinks, committed blocks would dangle).
        Exact tier: when APC_EXACT_CACHE_ENTRIES is unset, replace the
        2-entry count cap with a byte budget (entries are full KV
        snapshots; counting them says nothing about their cost)."""
        if model is None:
            return
        if budget_fraction is None:
            budget_fraction = _POOL_BUDGET_FRACTION
        import mlx.core as mx

        from .capacity import working_budget_bytes
        from .mem_preflight import kv_layer_costs
        from .prefill_decay import untracked_weight_bytes

        costs = kv_layer_costs(model)
        budget = working_budget_bytes()
        if not costs or not budget:
            return
        per_tok = sum(bpt for _w, bpt in costs)
        if per_tok <= 0:
            return
        # Stashed for stats: lets /metrics report committed pool bytes so
        # harnesses can separate evictable cache growth from real residue.
        self._pool_per_token_bytes = per_tok
        weights = float(mx.get_active_memory()) + untracked_weight_bytes()
        free = max(0.0, budget - weights)
        if not os.environ.get("APC_EXACT_CACHE_ENTRIES"):
            self._exact_budget_bytes = free * _EXACT_BUDGET_FRACTION
            self._exact_cache_max = 64
            _log.info("APC exact tier budget: %.1f GB (count cap 64)",
                      self._exact_budget_bytes / 1e9)
        if os.environ.get("APC_NUM_BLOCKS"):
            return
        target = int(free * budget_fraction
                     // (self.block_size * per_tok))
        # A committed block holds K+V arrays per layer; the pool competes
        # with everything else against the Metal resource limit, which is
        # what APC_MAX_POOL_TENSORS guards.
        tensors_per_block = 2 * len(costs)
        max_tensors = int(os.environ.get("APC_MAX_POOL_TENSORS", "450000"))
        target = min(target, max_tensors // max(1, tensors_per_block))
        if target <= self.num_blocks:
            return
        with self.lock:
            for i in range(len(self.pool), target):
                b = _apc.APCBlock(block_id=i)
                self.pool.append(b)
                self._free_push(b)
            self.num_blocks = target
        _log.info(
            "APC pool auto-sized: %d blocks (%.1f GB cap, %.1f KB/token)",
            target, target * self.block_size * per_tok / 1e9, per_tok / 1e3)

    def _trim_exact_to_budget(self) -> None:
        """Evict oldest exact entries until their bytes fit the budget
        autosize stashed. No-op without a budget (explicit
        APC_EXACT_CACHE_ENTRIES keeps stock count semantics)."""
        budget = getattr(self, "_exact_budget_bytes", None)
        if not budget:
            return
        from .serve_memtrace import _arrays, _leaf_caches

        def entry_bytes(e):
            # _leaf_caches unwraps CacheList layers (deepseek_v4 wraps
            # local + pools per layer); walking the wrapper's vars sees
            # cache objects, not arrays, and counts zero.
            total = 0
            for c in _leaf_caches(e.prompt_cache):
                for v in vars(c).values():
                    for a in _arrays(v):
                        total += a.nbytes
            return total

        with self.lock:
            sizes = {k: entry_bytes(e) for k, e in self._exact_cache.items()}
            total = sum(sizes.values())
            while total > budget and len(self._exact_cache) > 1:
                k, _ = self._exact_cache.popitem(last=False)
                total -= sizes.pop(k, 0)
            # Mirror of pool_bytes for the exact tier: lets harnesses
            # separate budgeted, evictable retention from real residue.
            self.stats.exact_bytes = int(total)

    def lookup_exact_cache(self, *args, **kwargs):
        self._trim_exact_to_budget()
        return super().lookup_exact_cache(*args, **kwargs)

    def store_exact_cache(self, *args, **kwargs):
        out = super().store_exact_cache(*args, **kwargs)
        self._trim_exact_to_budget()
        return out

    def stats_snapshot(self) -> dict:
        """Stock snapshot plus the gmlx ckpt-tier side counters (pure
        wrap: super() + merge). Visible at /v1/cache/stats -- a ckpt
        model with zeroed ckpt_* keys is broken, not idle."""
        from .cache_snapshot import ckpt_stats_snapshot
        from .prefix_cache import spec_prefix_stats
        snap = super().stats_snapshot()
        snap.update(ckpt_stats_snapshot(self))
        snap.update(spec_prefix_stats())
        # Upstream's APCStats.snapshot is a fixed whitelist; the byte
        # gauges the budget trims maintain have to be merged here or
        # they never reach /metrics.
        for key in ("pool_bytes", "exact_bytes"):
            val = getattr(self.stats, key, None)
            if val is not None:
                snap[key] = int(val)
        return snap

    def reset_stats(self) -> None:
        from .cache_snapshot import ckpt_stats_clear
        from .prefix_cache import spec_prefix_stats_clear
        super().reset_stats()
        ckpt_stats_clear(self)
        spec_prefix_stats_clear()

    def clear(self) -> None:
        """Stock clear plus the ckpt tier: the pool wipe zeroes the block
        tensors pinned records reference, so records/sidecars/counters
        must not survive it. One lock span (RLock) so a concurrent
        lookup can never pin a record between the pool wipe and the
        record drop."""
        from .cache_snapshot import ckpt_reset
        from .prefix_cache import clear_all_spec_prefix_caches
        with self.lock:
            super().clear()
            ckpt_reset(self)
        # Outside the manager lock: the spec prefix cache has its own
        # lifetime and its pinned snapshots must not survive a reset.
        clear_all_spec_prefix_caches()

    def governor_bytes(self) -> int:
        """Resident GPU bytes this manager holds: committed pool blocks
        plus exact-prefix clones. Attribute math only, no device sync;
        the governor samples this on a slow cadence."""
        total = 0
        with self.lock:
            for b in self.pool:
                for arrs in (b.keys, b.values):
                    if arrs:
                        total += sum(a.nbytes for a in arrs)
            for e in self._exact_cache.values():
                for c in e.prompt_cache:
                    nb = getattr(c, "nbytes", 0)
                    total += int(nb) if isinstance(nb, int) else 0
            idx = getattr(self, "_kq_ckpt_records", None)
            for rec in (idx or {}).values():
                total += int(getattr(rec, "nbytes", 0) or 0)
        return total

    def governor_evict(self, fraction: float) -> int:
        """Evict ``fraction`` of reclaimable bytes for the governor's
        orange rung: exact-prefix clones oldest-first, then unreferenced
        pool blocks through the stock LRU eviction (hash entry removed,
        slabs nulled, block returned empty to the free queue). Rows in
        flight never lose blocks: only ref_cnt 0 blocks sit in the free
        queue."""
        fraction = min(1.0, max(0.0, float(fraction)))
        freed = 0
        # Ckpt records first: releasing them unpins their blocks, so the
        # pool eviction below can reclaim those in the same pass. Without
        # this the pinned share is invisible to red-band reclaim.
        from .cache_snapshot import ckpt_governor_release
        freed += ckpt_governor_release(self, fraction)
        with self.lock:
            n = len(self._exact_cache)
            for _ in range(max(1, round(n * fraction)) if n else 0):
                if not self._exact_cache:
                    break
                _, e = self._exact_cache.popitem(last=False)
                for c in e.prompt_cache:
                    nb = getattr(c, "nbytes", 0)
                    freed += int(nb) if isinstance(nb, int) else 0
            committed = sum(
                1 for b in self.pool if b.keys and b.ref_cnt == 0)
            target = int(committed * fraction)
            evicted = 0
            guard = len(self.pool)
            while evicted < target and guard > 0:
                guard -= 1
                head = self._free_head
                if head is None:
                    break
                had_slabs = bool(head.keys)
                nb = (sum(a.nbytes for a in head.keys or ())
                      + sum(a.nbytes for a in head.values or ()))
                b = self._evict_lru()
                if b is None:
                    break
                # Return the emptied block to the queue tail; empty
                # blocks cycling there is harmless (order matters only
                # among committed blocks).
                self._free_push(b)
                if had_slabs:
                    freed += nb
                    evicted += 1
        return freed

    def store_ckpt_blocks(self, token_ids, layer_keys, layer_values,
                          *, extra_hash=0, disk=True):
        """Block store for checkpoint chains: always per-block -- the
        layer-major branch would return no blocks (an unpinnable record)
        and clone the whole prefix into the 2-slot exact LRU. disk=False
        keeps a chain memory-only (the window chain's position salt makes
        its disk shards undedupable)."""
        return self.store_kv_blocks(
            token_ids, layer_keys, layer_values, extra_hash=extra_hash,
            _ckpt_disk=bool(disk))

    def store_kv_blocks(self, token_ids, layer_keys, layer_values,
                        *, extra_hash=0, skip_first_n_tokens=0,
                        _ckpt_disk=None):
        import mlx.core as mx

        h = _store_helpers()
        if h is None:
            out = super().store_kv_blocks(
                token_ids, layer_keys, layer_values,
                extra_hash=extra_hash,
                skip_first_n_tokens=skip_first_n_tokens)
            # The upstream base store has the same inline exact branch.
            self._trim_exact_to_budget()
            return out
        is_ckpt = _ckpt_disk is not None
        allow_disk = self.disk is not None and (_ckpt_disk is None
                                                or _ckpt_disk)
        _clone_lm = h["clone_lm"]
        _seq_hash = h["seq_hash"]
        _entry_cls = h["entry_cls"]
        _seed_parent = h["seed_parent"]
        _hash_tokens = h["hash_tokens"]
        _disk_block_cls = h["disk_block_cls"]
        _copy_arr = h["copy_arr"]
        _time = h["time"]
        eval_chunk_blocks = env_int("GMLX_APC_STORE_EVAL_CHUNK", 32)

        with self.lock:
            n_full = len(token_ids) // self.block_size
            skip_full = skip_first_n_tokens // self.block_size
            full_prefix_tokens = n_full * self.block_size
            guarded_prefix_tokens = max(
                0, len(token_ids) - self.exact_cache_guard_tokens
            )
            layer_major_prefix_tokens = min(
                full_prefix_tokens,
                (guarded_prefix_tokens // self.block_size) * self.block_size,
            )
            new_blocks = []
            disk_blocks = []
            per_block_tensors = len(layer_keys) + len(layer_values)
            token_tuple = tuple(
                int(t) for t in token_ids[:layer_major_prefix_tokens])
            layer_major_stored = False
            if (
                not is_ckpt
                and self._layer_major_memory_min_tokens > 0
                and self._exact_cache_max > 0
                and layer_major_prefix_tokens
                >= self._layer_major_memory_min_tokens
            ):
                copied = _clone_lm(
                    layer_keys,
                    layer_values,
                    layer_major_prefix_tokens,
                )
                if copied is not None:
                    key = _seq_hash(token_tuple, extra_hash, self.block_size)
                    self._exact_cache[key] = _entry_cls(
                        token_ids=token_tuple,
                        extra_hash=int(extra_hash),
                        prompt_cache=copied,
                        last_used=_time.time(),
                    )
                    self._exact_cache.move_to_end(key)
                    while len(self._exact_cache) > self._exact_cache_max:
                        self._exact_cache.popitem(last=False)
                    self.stats.exact_stores += 1
                    layer_major_stored = True
                    # Inline exact entry: enforce the byte budget here
                    # too, not just in store_exact_cache (lock is an
                    # RLock, safe to re-enter).
                    self._trim_exact_to_budget()
            parent = _seed_parent
            for i in range(skip_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[
                        i * self.block_size:(i + 1) * self.block_size]
                )
                parent = _hash_tokens(parent, chunk, extra_hash)

            # Deferred-eval accumulator (the only change vs stock).
            pending = []

            def _flush_pending(force=False):
                if not pending:
                    return
                if force or (eval_chunk_blocks > 0 and len(pending)
                             >= eval_chunk_blocks * per_block_tensors):
                    mx.eval(pending)
                    pending.clear()

            for i in range(skip_full, n_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[
                        i * self.block_size:(i + 1) * self.block_size]
                )
                h2 = _hash_tokens(parent, chunk, extra_hash)
                if allow_disk and not self.disk.has(h2):
                    disk_blocks.append(
                        _disk_block_cls(
                            block_hash=int(h2),
                            parent_hash=int(parent),
                            extra_hash=int(extra_hash),
                            token_ids=chunk,
                            source_block_idx=i,
                        )
                    )
                if layer_major_stored:
                    parent = h2
                    continue
                existing = self.hash_table.get(h2)
                if existing is not None and existing.token_ids == chunk:
                    acquired = self._acquire_existing(existing)
                    new_blocks.append(acquired)
                    parent = h2
                    continue
                if (
                    self._max_pool_tensors > 0
                    and per_block_tensors > 0
                    and (len(self.hash_table) + 1) * per_block_tensors
                    > self._max_pool_tensors
                ):
                    _apc.logger.debug(
                        "APC pool tensor limit reached; skipping memory "
                        "store at block %d/%d",
                        i,
                        n_full,
                    )
                    if not allow_disk:
                        break
                    parent = h2
                    continue
                b = self._evict_lru()
                if b is None:
                    _apc.logger.debug(
                        "APC pool exhausted; skipping memory store at "
                        "block %d/%d",
                        i,
                        n_full,
                    )
                    if not allow_disk:
                        break
                    parent = h2
                    continue
                start = i * self.block_size
                end = start + self.block_size
                # Deep-copy each slice into its own buffer so the block
                # tensor is decoupled from the caller's cache, which
                # mlx.clear_cache may release after generation. The copies
                # are lazy here; _flush_pending materializes them in chunks
                # and always before return.
                k_slabs = [_copy_arr(k[..., start:end, :]) for k in layer_keys]
                v_slabs = [_copy_arr(v[..., start:end, :]) for v in layer_values]
                pending.extend(k_slabs)
                pending.extend(v_slabs)
                _flush_pending()
                b.block_hash = h2
                b.parent_hash = parent
                b.token_ids = chunk
                b.extra_hash = extra_hash
                b.keys = k_slabs
                b.values = v_slabs
                b.ref_cnt = 1
                self.hash_table[h2] = b
                new_blocks.append(b)
                self.stats.stores += 1
                self.stats.served_tokens += self.block_size
                parent = h2
            _flush_pending(force=True)
            if allow_disk and disk_blocks:
                try:
                    # The shard crosses to the disk writer thread; hand it
                    # evaluated arrays only (loop paths that skip the slab
                    # copies leave these lazy). Owned survivors: the slices
                    # are the live generation's KV, retained by the prompt
                    # cache; the guard drains before this handler continues.
                    from .eval_guard import guard
                    guard.eval(*(list(layer_keys) + list(layer_values)),
                               site="apc-disk-save", owner="owned")
                    self.disk.save_layer_major_blocks(
                        disk_blocks, layer_keys, layer_values, self.block_size
                    )
                    self.stats.disk_writes += len(disk_blocks)
                except Exception as e:
                    _apc.logger.warning(
                        "APC disk save scheduling failed: %s", e)
            self.stats.pool_used = sum(
                1 for x in self.pool if x.block_hash is not None)
            self.stats.pool_bytes = int(
                self.stats.pool_used * self.block_size
                * getattr(self, "_pool_per_token_bytes", 0))
            return new_blocks


# The pool is the cheapest use of free RAM (recompute avoided) and it
# self-registers with the governor as evictable, so it can take a large
# share: pressure reclaims it before any request is shed.
_POOL_BUDGET_FRACTION = 0.5
_EXACT_BUDGET_FRACTION = 0.15
_BLOCK_SIZES = (16, 32, 64, 128, 256)


def _auto_block_size(model_path):
    """Smallest block size whose tensor-capped pool covers the byte
    budget autosize will ask for. Committed blocks cost K+V arrays per
    layer, so at block_size 16 the Metal resource limit (proxied by
    APC_MAX_POOL_TENSORS) binds far below the byte budget on deep
    workloads. None keeps the stock default (non-GGUF path, unreadable
    header, or 16 already suffices)."""
    if not model_path:
        return None
    try:
        from .capacity import working_budget_bytes
        from .tool_preflight import _kv_costs, _shards, _synth_config

        shards = _shards(str(model_path))
        cfg = _synth_config(shards[0])
        costs = _kv_costs(cfg) if cfg else None
        budget = working_budget_bytes()
        if not (costs and budget):
            return None
        per_tok = sum(bpt for _w, bpt in costs)
        if per_tok <= 0:
            return None
        weights = float(sum(os.path.getsize(p) for p in shards))
        budget_tokens = (max(0.0, budget - weights)
                         * _POOL_BUDGET_FRACTION / per_tok)
        max_tensors = int(os.environ.get("APC_MAX_POOL_TENSORS", "450000"))
        need = 2 * len(costs) * budget_tokens / max(1, max_tensors)
        for bs in _BLOCK_SIZES:
            if bs >= need:
                return bs if bs > _apc.DEFAULT_BLOCK_SIZE else None
        return _BLOCK_SIZES[-1]
    except Exception:
        _log.debug("APC block-size derivation skipped", exc_info=True)
        return None


def build_apc_manager(model_namespace=None):
    """Build the gmlx APC manager from the env vars ``from_env`` reads.

    Gated on ``GMLX_APC_ENABLED``, which the residency build window sets to
    the effective ``APC_ENABLED`` value (config-over-ambient) before pinning
    the stock var to ``0``. Absent gate (a non-pooled embedding of the
    bridge) builds nothing - stock wiring applies there untouched. The knob
    vars keep their stock names and parsing: ``APC_BLOCK_SIZE``,
    ``APC_NUM_BLOCKS``, and the ``APC_DISK_*`` family, with the disk
    namespace defaulting to the model path.
    """
    if os.environ.get("GMLX_APC_ENABLED") != "1":
        return None
    env_bs = os.environ.get("APC_BLOCK_SIZE")
    block_size = (int(env_bs) if env_bs else
                  _auto_block_size(model_namespace)
                  or _apc.DEFAULT_BLOCK_SIZE)
    num_blocks = int(os.environ.get("APC_NUM_BLOCKS", _apc.DEFAULT_NUM_BLOCKS))

    disk = None
    disk_path = os.environ.get("APC_DISK_PATH")
    if disk_path:
        ns = model_namespace or os.environ.get("APC_DISK_NAMESPACE", "default")
        max_gb = float(os.environ.get("APC_DISK_MAX_GB", 0))
        max_bytes = int(max_gb * (1 << 30)) if max_gb > 0 else None
        workers = int(os.environ.get("APC_DISK_WORKERS", "1"))
        try:
            disk = _apc.DiskBlockStore(
                Path(disk_path).expanduser(),
                namespace=ns,
                num_workers=workers,
                max_bytes=max_bytes,
            )
            cap_str = f"{max_gb:.1f} GB" if max_bytes else "unbounded"
            _log.info(
                "APC disk tier at %s (ns=%s, cap=%s, read_mode=%s)",
                disk.dir, ns, cap_str, disk._read_mode,
            )
        except Exception as e:
            _log.warning("APC disk tier disabled (init failed): %s", e)

    _log.info(
        "APC enabled (block_size=%d, num_blocks=%d, disk=%s, gmlx manager)",
        block_size, num_blocks, bool(disk),
    )
    return GmlxAPCManager(
        num_blocks=num_blocks, block_size=block_size, disk=disk)
