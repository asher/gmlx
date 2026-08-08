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

    def stats_snapshot(self) -> dict:
        """Stock snapshot plus the gmlx ckpt-tier side counters (pure
        wrap: super() + merge). Visible at /v1/cache/stats -- a ckpt
        model with zeroed ckpt_* keys is broken, not idle."""
        from .cache_snapshot import ckpt_stats_snapshot
        snap = super().stats_snapshot()
        snap.update(ckpt_stats_snapshot(self))
        return snap

    def reset_stats(self) -> None:
        from .cache_snapshot import ckpt_stats_clear
        super().reset_stats()
        ckpt_stats_clear(self)

    def clear(self) -> None:
        """Stock clear plus the ckpt tier: the pool wipe zeroes the block
        tensors pinned records reference, so records/sidecars/counters
        must not survive it."""
        from .cache_snapshot import ckpt_reset
        super().clear()
        ckpt_reset(self)

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
            return super().store_kv_blocks(
                token_ids, layer_keys, layer_values,
                extra_hash=extra_hash,
                skip_first_n_tokens=skip_first_n_tokens)
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
                    # copies leave these lazy).
                    mx.eval(list(layer_keys) + list(layer_values))
                    self.disk.save_layer_major_blocks(
                        disk_blocks, layer_keys, layer_values, self.block_size
                    )
                    self.stats.disk_writes += len(disk_blocks)
                except Exception as e:
                    _apc.logger.warning(
                        "APC disk save scheduling failed: %s", e)
            self.stats.pool_used = sum(
                1 for x in self.pool if x.block_hash is not None)
            return new_blocks


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
    block_size = int(os.environ.get("APC_BLOCK_SIZE", _apc.DEFAULT_BLOCK_SIZE))
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
