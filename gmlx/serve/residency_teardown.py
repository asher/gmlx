"""The residency pool's teardown of the entries it took out."""

from __future__ import annotations

import contextlib

from gmlx.stream.installs import streaming_owner as _streaming_owner

from .residency_build import _clear_table_step
from .residency_types import _Entry, _Scratch, _build_scratch, _log
from .residency_workers import _holds_worker


class _Teardown:
    """Teardown of the entries the pool took out, with no _lock held."""

    def _teardown_all(self, torn: list, released: list) -> None:
        """Tear down the entries taken out of the pool, with no _lock held:
        every one is tried, and the first failure raises after. Each one's
        bytes leave the tearing count once its teardown ends, an image
        entry's once its pipeline's release ends."""
        first = None
        while torn:
            entry = torn.pop(0)
            try:
                self._teardown(entry, released)
            except Exception as e:  # noqa: BLE001 - raised after the rest
                _log.warning("teardown of %s failed", entry.model_path,
                             exc_info=True)
                first = first or e
            finally:
                # An image entry's bytes went with its pipeline.
                with self._lock:
                    n, entry.tearing = entry.tearing, 0
                    self._torn(entry.kind if entry.is_image else None, n)
        if first is not None:
            raise first

    def _discard_failed_build(self, cache_key, model_path, scratch) -> None:
        """Tear down what a failed build left in its scratch: the stock
        cache entry, the feeders, the pin. A build that fails after the
        stock load (a kv policy refusal) otherwise leaves a whole model
        resident behind the exception. Best-effort."""
        mc = scratch.model_cache
        if not hasattr(mc, "get") or mc.get("model") is None:
            return
        entry = _Entry(cache_key=cache_key, model_path=model_path,
                       model_cache=scratch.model_cache,
                       response_generator=scratch.response_generator,
                       apc_manager=scratch.apc_manager)
        scratch.model_cache = {}
        scratch.response_generator = None
        scratch.apc_manager = None
        try:
            self._teardown(entry)
        except Exception:
            _log.warning("failed build teardown skipped", exc_info=True)

    def _teardown(self, entry: _Entry, released=None):
        """Tear ``entry`` down. An image entry's pipeline goes to
        ``released``, which the caller hands to :meth:`_release_images`
        once it drops ``_lock``: the release waits on its worker, and
        a collection there can finalize a hold whose release takes
        ``_lock``. Called with no _lock held."""
        self._hit("teardown")
        if entry.is_image:
            pipe, entry.pipeline = entry.pipeline, None
            # The bytes it counts as tearing go with the pipeline.
            item, entry.tearing = (entry.kind, pipe, entry.tearing), 0
            if released is None:
                self._release_images([item])
            else:
                released.append(item)
            return
        self._teardown_chat(entry)

    @contextlib.contextmanager
    def _teardown_window(self, released: list, which=None):
        """The worker reservation and the build lock for a teardown of the
        idle entries ``which(entry)`` picks, under _lock (every idle one
        when None): only their workers are held. An
        image entry busy when the reservation was decided can be idle and
        torn down under the lock: without the worker held, its pipeline is
        released after the lock drops, as a release would queue on the
        worker behind a reservation that waits on the lock."""
        try:
            with self._worker_for(teardown=which), self._build_lock:
                try:
                    yield
                finally:
                    held = [r for r in released if _holds_worker(r[0])]
                    released[:] = [r for r in released
                                   if not _holds_worker(r[0])]
                    self._release_images(held)
        finally:
            self._release_images(released)

    def _teardown_chat(self, entry: _Entry):
        from . import bridge_vlm as _serving

        # Drop any stashed in-memory MTP drafter for this path, or its weights
        # stay referenced after the model is gone.
        _serving.drop_mtp_stash(entry.model_path)
        _clear_table_step(entry.model_path)
        # Capture the expert prefetcher before the stock unload empties
        # model_cache; it holds one open fd per GGUF shard and nothing else
        # closes it (repeated load/unload cycles would creep toward EMFILE).
        owner = _streaming_owner(entry.model_cache.get("model"))
        _prefetcher = getattr(owner, "_kq_prefetcher", None)
        # Same for the prefill feeder: shard fds + staging pools + ~GBs of
        # host ring slots that must not outlive the model.
        _feeder = getattr(owner, "_kq_feeder", None)
        # And the decode feeder: its mlocked wired arena (a large fraction of
        # RAM on hybrid over-RAM MoE models), read pool, and shard fds sit in
        # a feeder<->module reference cycle, so refcounting alone won't
        # reclaim them before the next model sizes its own arena.
        _decode_feeder = getattr(owner, "_kq_decode_feeder", None)
        if _decode_feeder is not None:
            from .governor import unregister_arena

            unregister_arena(_decode_feeder)
        # And the every-token weight pin: mlocked file-backed ranges, which
        # stay file-backed and so still read as reclaimable to vm_stat. The
        # next model's arena sizes against that snapshot, so an unreleased
        # pin is budget the next load believes it has.
        _weights_pin = getattr(owner, "_kq_weights_pin", None)
        _resident_holders = [
            m for m in (entry.model_cache.get("model"), owner)
            if getattr(m, "_kq_resident_arrays", None)
        ]
        scratch = _Scratch()
        scratch.response_generator = entry.response_generator
        scratch.model_cache = entry.model_cache
        scratch.apc_manager = entry.apc_manager
        token = _build_scratch.set(scratch)
        try:
            self._stock_unload()
        finally:
            _build_scratch.reset(token)
        # The stock unload only clear()s the APC's in-memory blocks; close() drains
        # and stops the disk writer thread so a reaped model doesn't leak a daemon
        # writer. The on-disk SSD-KV shards persist either way (close never deletes
        # them) - a later reload of this model warm-restores from them.
        # Best-effort: teardown runs every closer to completion; one failing
        # close must not leak the others' fds/arenas.
        for owner in (entry.apc_manager, _prefetcher, _feeder, _decode_feeder,
                      _weights_pin):
            close = getattr(owner, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception:
                _log.warning("teardown: %s close failed",
                             type(owner).__name__, exc_info=True)
        # A larger-than-RAM model leaves a page-cache remnant that taxes
        # whoever faults next (gmlx.stream.pagecache). Process exit sweeps it for
        # CLI runs; a long-lived server sweeps at eviction, before the next
        # model loads against the stale cache. No-op for in-RAM models.
        try:
            from gmlx.stream.pagecache import release_streaming_for
            release_streaming_for(entry.model_path)
        except Exception:  # noqa: S110 - page-cache sweep is advisory
            pass
        # Every eviction and reap funnels through here: drop this entry's
        # untracked-weights attributions so an evicted model stops taxing
        # headroom_bytes(). The registry forgets exactly the keys this
        # entry's build registered; bytes leave the estimate only at zero
        # owners, so a resident sibling sharing shards keeps them counted.
        from gmlx.gen.prefill_decay import forget_untracked_weights
        forget_untracked_weights(entry.cache_key)
        # The model modules sit in reference cycles (feeder<->module among
        # others) that refcounting cannot reclaim, and an idle server may not
        # run a gen-2 GC for a long time. The next load's preload_gate
        # re-measures headroom immediately, so a swap on a near-full box
        # defers against hundreds of GB of already-dead arrays unless the
        # cycles are collected here. A collection while this frame or the
        # entry still points at the model frees nothing (the closer loop's
        # last bound method alone pins the feeder and, through it, every
        # expert weight): drop every reference first, then collect.
        for m in _resident_holders:
            try:
                import mlx_kquant as kq

                if getattr(kq, "residency_erase", None):
                    for a in m._kq_resident_arrays:
                        kq.residency_erase(a)
                    kq.residency_commit()
            except Exception:
                _log.warning("teardown: residency erase failed; the wired "
                             "accounting may be stale", exc_info=True)
            m._kq_resident_arrays = None
        entry.model_cache = {}
        entry.response_generator = None
        entry.apc_manager = None
        del scratch, owner, close, _prefetcher, _feeder, _decode_feeder
        del _weights_pin
        m = None
        del m, _resident_holders
        import gc
        gc.collect()
