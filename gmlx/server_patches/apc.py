"""APC (prefix-cache) engine patches: lone-sequence harvest and batched
store eval."""

from __future__ import annotations

import importlib


from ..envflags import env_int


def install_apc_lone_harvest() -> None:
    """Teach mlx-vlm's APC block harvest to read a lone request's plain ``KVCache``.

    The continuous-batching engine harvests prefill K/V into APC prefix blocks via
    ``apc.harvest_blocks_from_batch_cache``, which slices a row out of a batched
    cache by its ``_idx`` attribute. A single, non-concurrent request takes a fast
    path that builds a plain ``KVCache`` - which exposes ``offset`` (valid length),
    not ``_idx`` - so the stock harvest hits its ``idx is None`` guard, returns
    ``[]``, and the request stores nothing in memory or on the SSD tier. The result
    is that a purely sequential single-client session (the common coding-assistant
    case) never populates the prefix cache, while concurrent traffic does.

    This generalizes the harvest: when ``_idx`` is absent, fall back to ``offset``
    over the single row (no left padding), which is exactly the slice the stock
    code would take on a batched cache. A quantized KV cache keeps ``keys`` as a
    tuple, which neither path can slice - skip it as before. ``ar.py`` calls the
    function by module attribute (``_apc.harvest_blocks_from_batch_cache``), so
    replacing it on the module is picked up at the call site without a fork.

    mlx-vlm 0.6.4 absorbed the offset fallback into the stock harvest, but
    without the quantized (tuple ``keys``) guard: stock raises where this
    declines. The replacement stays installed on every version -- it is a
    strict superset of both stock behaviors."""
    import mlx.core as mx

    apc = importlib.import_module("mlx_vlm.apc")
    if getattr(apc, "_kq_lone_harvest", False):
        return

    def harvest_blocks_from_batch_cache(
        apc_manager, batch_caches, batch_idx, full_token_ids,
        *, extra_hash=0, skip_first_n_tokens=0):
        layer_keys = []
        layer_values = []
        for c in batch_caches:
            keys = getattr(c, "keys", None)
            values = getattr(c, "values", None)
            if keys is None or values is None:
                return []
            idx = getattr(c, "_idx", None)
            left_padding = getattr(c, "left_padding", None)
            if idx is None:
                # Lone-request fast path: a plain KVCache has a scalar `offset`
                # and no `_idx`/`left_padding`. Harvest its one row over
                # [0, offset). Skip a quantized cache (tuple `keys`).
                offset = getattr(c, "offset", None)
                if offset is None or not isinstance(keys, mx.array):
                    return []
                idx = int(offset)
                left_padding = None
            if left_padding is not None:
                try:
                    lp = int(left_padding[batch_idx].item())
                except Exception:
                    lp = 0
            else:
                lp = 0
            layer_keys.append(keys[batch_idx:batch_idx + 1, :, lp:idx, :])
            layer_values.append(values[batch_idx:batch_idx + 1, :, lp:idx, :])
        return apc_manager.store_kv_blocks(
            full_token_ids, layer_keys, layer_values,
            extra_hash=extra_hash, skip_first_n_tokens=skip_first_n_tokens)

    apc.harvest_blocks_from_batch_cache = harvest_blocks_from_batch_cache
    apc._kq_lone_harvest = True


def install_apc_batched_store_eval() -> None:
    """Batch the per-block ``mx.eval`` inside ``APCManager.store_kv_blocks``.

    The stock store loop deep-copies each 16-token block's K/V slices and
    evaluates them one block at a time -- ~0.4-0.5 ms of dispatch/sync per
    block regardless of data size, which compounds to ~1 s of synchronous
    prefill-thread stall for a 32k-token store on a 27B model. The copies
    are pure materialization (nothing in the loop consumes their values), so
    they can be evaluated in chunks: same data volume, ~30x fewer syncs.

    This is a verbatim copy of the stock method with the eval hoisted into
    ``GMLX_APC_STORE_EVAL_CHUNK``-block batches (default 32; <=0 means a
    single eval before return). All evals still complete before the method
    returns, preserving the stock guarantee that block tensors are decoupled
    from the caller's cache before ``mx.clear_cache`` can release it.

    Installs only if every private helper it references still exists
    upstream; otherwise logs and keeps the stock method. Idempotent."""
    import mlx.core as mx

    apc = importlib.import_module("mlx_vlm.apc")
    if getattr(apc, "_kq_batched_store_eval", False):
        return
    try:
        _clone_lm = apc._clone_layer_major_kv_cache_for_apc
        _seq_hash = apc._sequence_hash
        _entry_cls = apc.APCExactCacheEntry
        _seed_parent = apc.SEED_PARENT_HASH
        _hash_tokens = apc._hash_tokens
        _disk_block_cls = apc._DiskLayerMajorBlock
        _copy_arr = apc._copy_mlx_array
        _time = apc.time
        _logger = apc.logger
    except AttributeError as e:
        print(f"[apc] prompt-cache fast path not installed ({e}); the cache "
              "still works, slower - reinstall the pinned mlx-vlm "
              "(pip install mlx-vlm==0.6.4)")
        return

    eval_chunk_blocks = env_int("GMLX_APC_STORE_EVAL_CHUNK", 32)

    def store_kv_blocks(self, token_ids, layer_keys, layer_values,
                        *, extra_hash=0, skip_first_n_tokens=0):
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
                self._layer_major_memory_min_tokens > 0
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
                if force or eval_chunk_blocks <= 0:
                    mx.eval(pending)
                    pending.clear()
                elif len(pending) >= eval_chunk_blocks * per_block_tensors:
                    mx.eval(pending)
                    pending.clear()

            for i in range(skip_full, n_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[
                        i * self.block_size:(i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                if self.disk is not None and not self.disk.has(h):
                    disk_blocks.append(
                        _disk_block_cls(
                            block_hash=int(h),
                            parent_hash=int(parent),
                            extra_hash=int(extra_hash),
                            token_ids=chunk,
                            source_block_idx=i,
                        )
                    )
                if layer_major_stored:
                    parent = h
                    continue
                existing = self.hash_table.get(h)
                if existing is not None and existing.token_ids == chunk:
                    acquired = self._acquire_existing(existing)
                    new_blocks.append(acquired)
                    parent = h
                    continue
                if (
                    self._max_pool_tensors > 0
                    and per_block_tensors > 0
                    and (len(self.hash_table) + 1) * per_block_tensors
                    > self._max_pool_tensors
                ):
                    _logger.debug(
                        "APC pool tensor limit reached; skipping memory "
                        "store at block %d/%d",
                        i,
                        n_full,
                    )
                    if self.disk is None:
                        break
                    parent = h
                    continue
                b = self._evict_lru()
                if b is None:
                    _logger.debug(
                        "APC pool exhausted; skipping memory store at "
                        "block %d/%d",
                        i,
                        n_full,
                    )
                    if self.disk is None:
                        break
                    parent = h
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
                b.block_hash = h
                b.parent_hash = parent
                b.token_ids = chunk
                b.extra_hash = extra_hash
                b.keys = k_slabs
                b.values = v_slabs
                b.ref_cnt = 1
                self.hash_table[h] = b
                new_blocks.append(b)
                self.stats.stores += 1
                self.stats.served_tokens += self.block_size
                parent = h
            _flush_pending(force=True)
            if self.disk is not None and disk_blocks:
                try:
                    self.disk.save_layer_major_blocks(
                        disk_blocks, layer_keys, layer_values, self.block_size
                    )
                    self.stats.disk_writes += len(disk_blocks)
                except Exception as e:
                    _logger.warning("APC disk save scheduling failed: %s", e)
            self.stats.pool_used = sum(
                1 for x in self.pool if x.block_hash is not None)
            return new_blocks

    apc.APCManager.store_kv_blocks = store_kv_blocks
    apc._kq_batched_store_eval = True
