"""The residency pool's core: entries, ledgers, load, unload and the idle reaper."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

import gmlx.load.loadlog as loadlog

from .capacity import _defer
from .residency_build import _clear_table_step
from .residency_footprint import (_decode_feeder_of, _gguf_footprint_bytes,
                                  _plain_footprint, _prefill_feeder_of,
                                  _ring_room, _streaming_footprint)
from .residency_types import (IMAGE_KIND, STORE, VIDEO_KIND, ModelBusyError,
                              _STOCK_IMAGE_KINDS, _WORKER_KINDS, _WORKER_ORDER,
                              _Entry, _Figures, _Ledger, _log, _mine,
                              profile_label)
from .residency_workers import _holds_worker


class _PoolCore:
    """Entries, ledgers, cold load, eviction, unload and the idle reaper."""

    def __init__(
        self,
        proxy,
        stock_get,
        stock_unload,
        budget_bytes,
        pinned_paths,
        *,
        max_models=None,
        footprint_fn=_gguf_footprint_bytes,
        time_fn=None,
        in_flight_fn=None,
        image_fns=None,
        video_fns=None,
    ):
        self._proxy = proxy
        self._stock_get = stock_get
        self._stock_unload = stock_unload
        # (footprint, load, release) of an image entry, each taking what
        # the pool holds: the spec, the spec, a one-item list with the
        # pipeline. Injectable so eviction across kinds tests on the CPU.
        self._image_fns = image_fns
        self._video_fns = video_fns
        # Primary bound: total resident weight bytes. Secondary (optional) bound:
        # model count. ``max_models=None`` means bytes alone govern.
        self._budget = max(0, int(budget_bytes))
        self._max = None if max_models is None else max(1, int(max_models))
        self._footprint_fn = footprint_fn
        # Wall clock for idle-TTL accounting + the server-wide in-flight reader the
        # reaper gates on (both injectable so the reaper is testable without a GPU).
        self._time_fn = time_fn or time.monotonic
        self._in_flight_fn = in_flight_fn
        self._pinned_paths = set(pinned_paths or ())
        # A softer tier than a pin: paths kept resident through the idle-TTL reaper
        # but still LRU-evictable under memory pressure (see reap_idle vs
        # _evict_for_room). Mutated at runtime by set_keep (the /v1/keep route).
        self._keep_paths: set = set()
        self._entries: "OrderedDict[tuple, _Entry]" = OrderedDict()
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._clock = 0
        # The footprint of the cold build in flight, counted resident from
        # its room check to its insert, and the model it loads.
        self._building = 0
        self._loading: str | None = None
        # A turn that waits for a chat model in use to go idle waits here;
        # release() notifies it.
        self._idle = threading.Condition(self._lock)
        self._releases = 0
        # The bytes of the victims taken out of the pool whose teardown,
        # or pipeline release, runs, and of the upload bodies admitted and
        # not put yet.
        self._tearing = 0
        self._incoming = 0
        # The tearing bytes of the pipelines each worker kind releases.
        self._tearing_kind = dict.fromkeys(_WORKER_KINDS, 0)
        # The running count of the tearing bytes each worker kind freed.
        self._freed_kind = dict.fromkeys(_WORKER_KINDS, 0)
        # Injectable for tests: the measured room; an entry's LoRA figures
        # ({request key: bytes}, own bytes) from its caches; the shed of an
        # entry's request LoRA bytes and of the idle uploads, each returning
        # the bytes it let go; the uploads' bytes, one int read with no
        # lock; the Retry-After of a turn's 503, which bounds its waits.
        self._headroom_fn: Callable[[], Any] | None = None
        self._lora_figures: Callable[[Any], Any] | None = None
        self._lora_shed: Callable[..., int] | None = None
        self._store_shed: Callable[[int, tuple], int] | None = None
        self._store_live: Callable[[], int] | None = None
        self._retry_fn: Callable[[], int] | None = None
        # A test's hook on a ledger's bytes past its booking.
        self.overrun_hook: Callable[[str, int], None] | None = None
        # A test's pause at a named point of a turn or a cold build, and
        # its check at each teardown and measure.
        self._seam = None
        self.figures = _Figures(budget=self._budget)

    # public
    def acquire(self, model_path, adapter_path, model_kind, *,
                ttl=None, cache_key_extra=(), env=None, build_spec=None,
                loaded_as=None) -> _Entry:
        # ``cache_key_extra`` (a model's load_signature) distinguishes two ids
        # backed by the same GGUF but loaded with different params (kv bits,
        # mmproj, drafter) - they become separate resident entries.
        cache_key = (model_path, adapter_path, model_kind) + tuple(cache_key_extra)
        # Every acquire returns with the entry's in-flight refcount incremented;
        # the caller pairs it with release() when the request finishes (the
        # pooled get_cached_model wraps it in a _BusyHold).
        # Fast path: already resident - never blocks on a cold build.
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                if ttl is not None:
                    entry.ttl = ttl          # latest acquire's TTL wins
                self._touch(entry)
                entry.busy += 1
                self._publish()
                return entry
        if model_kind in _STOCK_IMAGE_KINDS:
            raise ValueError(f"model kind {model_kind!r} loads mlx-vlm's own "
                             f"image pipelines; gmlx loads {IMAGE_KIND!r}")
        # Cold path: one build at a time (bounds peak memory).
        file_bytes, incoming = self._price(model_path, model_kind, build_spec,
                                           env)
        if getattr(build_spec, "stream", None) == "experts":
            _log.info("streaming entry priced at %.1f GB resident of a %.1f GB file",
                      incoming / 1e9, file_bytes / 1e9)
        # Refuse an unloadable GGUF *before* evicting healthy residents for
        # room - a corrupt/unsupported file would otherwise cost every LRU
        # victim a full cold reload for a build that was never going to work.
        # Header-only, so the cost is milliseconds; the loader re-validates.
        if (model_kind not in _WORKER_KINDS and isinstance(model_path, str)
                and model_path.endswith(".gguf")
                and os.path.isfile(model_path)):
            from gmlx.load.preflight import preflight
            preflight(model_path)
        # Image pipelines a post-build refit evicts. The worker reservation
        # was decided before the build, which may have seen them busy, so
        # they are released after the build lock drops: a reservation taken
        # since waits on that lock, and a release would queue behind it.
        late: list = []
        try:
            # A worker entry busy when the worker reservations are decided
            # can be idle at the eviction. Its release would queue on its
            # worker behind a reservation that waits on the build lock, so
            # that attempt evicts nothing and the next takes the
            # reservation of that entry's worker, and no other.
            force: frozenset = frozenset()
            for _attempt in range(len(_WORKER_ORDER) + 1):
                with self._worker_for(model_kind, incoming=incoming,
                                      force=force), \
                        self._build_lock:
                    released: list = []
                    torn: list = []
                    try:
                        with self._lock:
                            entry = self._entries.get(cache_key)
                            if entry is not None:
                                if ttl is not None:
                                    entry.ttl = ttl
                                self._touch(entry)
                                entry.busy += 1
                                self._publish()
                                return entry
                            missing = frozenset(
                                k for k in self._victim_kinds(incoming)
                                if not _holds_worker(k))
                            if missing:
                                force = force | missing
                                continue
                            try:
                                self._evict_for_room(
                                    incoming, model_path=model_path,
                                    streaming=getattr(build_spec, "stream", None) == "experts",
                                    torn=torn)
                            finally:
                                self._publish()
                            self._hit("room")
                            # The build's room, in the hold that found it, so
                            # a LoRA turn cannot take it before the insert.
                            self._building = incoming
                            self._loading = (loaded_as or
                                             os.path.basename(str(model_path)))
                            self._publish()
                        self._hit("roomed")
                    finally:
                        # The victims go after _lock drops, under the
                        # build lock; a teardown that raises releases the
                        # pipelines all the same and ends the build.
                        try:
                            self._teardown_all(torn, released)
                        except BaseException:
                            with self._lock:
                                self._building = 0
                                self._loading = None
                                self._publish()
                            raise
                        finally:
                            self._release_images(released)
                    inserted = False
                    try:
                        try:
                            entry = self._build(cache_key, model_path, adapter_path,
                                                model_kind, file_bytes, ttl=ttl, env=env,
                                                build_spec=build_spec)
                        except BaseException:
                            # Free the failed build before the lock drops: a waiting
                            # acquire gates on live memory next.
                            _collect_failed_build()
                            _clear_table_step(model_path)
                            raise
                        if entry.priced_up:
                            self._fit_repriced(entry, model_path, late)
                        # The id that built the entry names it for the process lifetime
                        # (a metrics label wants a name fixed per entry; aliases sharing
                        # the signature reuse this entry and keep the first name).
                        entry.loaded_as = loaded_as
                        # The LoRA figures are read before the hold: no
                        # cache lock is taken inside it.
                        try:
                            self._hit("inserting")
                            lora = self._new_ledger(entry)
                        except BaseException:
                            self._teardown(entry, late)
                            raise
                        with self._lock:
                            self._hit("insert")
                            entry.lora = lora
                            self._entries[cache_key] = entry
                            self._building = 0
                            self._loading = None
                            inserted = True
                            self._touch(entry)
                            entry.busy += 1
                            self._publish()
                        return entry
                    finally:
                        if not inserted:
                            with self._lock:
                                self._building = 0
                                self._loading = None
                                self._publish()
            # Without a byte budget, only the count cap evicts.
            bound = (f"the {self._budget / 1e9:.1f} GB budget"
                     if self._budget else "the model count cap")
            _defer(
                f"model load deferred: {model_path} priced at "
                f"{incoming / 1e9:.1f} GB resident does not fit {bound} "
                "unless an idle image or video model is evicted, and its "
                "worker could not be held to release it. Retry when a slot "
                "frees.")
        finally:
            self._release_images(late)

    def _price(self, model_path, model_kind, build_spec, env,
               held: int | None = None) -> tuple:
        """(file bytes, resident bytes) of a cold load of ``model_path``: a
        worker kind's weights and activation reserve, a streamed model's
        every-token weights and arena beside ``held`` resident bytes (the
        pool's now when None), else the file less the tables read from
        it."""
        if model_kind in _WORKER_KINDS:
            # From the headers: a file that is no model of the kind is
            # refused here, before any eviction.
            file_bytes = incoming = int(self._kind_fn(model_kind, 0)(build_spec))
        else:
            file_bytes = incoming = self._footprint_fn(model_path)
        stream = getattr(build_spec, "stream", None)
        if stream == "experts":
            incoming = _streaming_footprint(
                model_path, file_bytes, env, held_bytes=(
                    self._resident_bytes() if held is None else held))
        elif stream is None and model_kind not in _WORKER_KINDS:
            incoming = _plain_footprint(model_path, file_bytes, env)
        return file_bytes, incoming

    def load_entry(self, model_path, spec, env=None) -> _Entry:
        """For a route's check, from ``figures`` with no lock: the resident
        chat entry of ``model_path``, or one outside the pool at the bytes
        a cold load of it books."""
        f = self.figures
        for e in f.entries:
            if not e.is_image and str(e.model_path) == str(model_path):
                return e
        _file, price = self._price(model_path, "auto", spec, env, f.resident)
        return _Entry(cache_key=(model_path,), model_path=model_path,
                      footprint=int(price), loaded_as=getattr(spec, "id", None))

    def _fit_repriced(self, entry: _Entry, model_path, released) -> None:
        """The install streamed less than the header predicted, so the
        entry holds more than the budget admitted. Evict for the new price
        as for any build; when only pinned or busy models remain and it
        cannot fit, unload the new model and defer the request. Evicted
        image pipelines go to ``released`` for the caller to release
        outside the build lock."""
        torn: list = []
        try:
            try:
                with self._lock:
                    # The build counts at its new price once, in this hold.
                    self._building = 0
                    self._evict_for_room(entry.footprint,
                                         model_path=model_path,
                                         streaming=True, torn=torn)
                    self._building = entry.footprint
                    self._publish()
            finally:
                self._teardown_all(torn, released)
        except BaseException:
            try:
                self._teardown(entry)
            except Exception:
                _log.warning("repriced build teardown skipped",
                             exc_info=True)
            _collect_failed_build()
            raise

    def _hit(self, name: str) -> None:
        if self._seam is not None:
            self._seam(name)

    def _publish(self) -> None:
        """Rebind ``figures`` to what the pool holds now. Under _lock, at
        the end of each hold that changes the entries, a ledger, a booking,
        the build in flight or a pin."""
        es = tuple(self._entries.values())
        held = tuple(0 if e.lora is None else e.lora.held for e in es)
        pending = sum(e.lora.pending for e in es if e.lora is not None)
        self.figures = _Figures(
            budget=self._budget, entries=es, held=held,
            footprints=sum(e.footprint for e in es),
            lora=sum(held) + pending, building=self._building,
            tearing=self._tearing, incoming=self._incoming)

    def _read_lora(self, entry):
        """({request key: bytes}, own bytes) the LoRA caches of ``entry``
        hold now, or None for an entry without them. Each read takes its
        cache's lock: call with no pool lock."""
        if entry is STORE or not getattr(entry, "is_image", False):
            return None
        return (self._lora_figures or _default_lora_figures)(entry)

    def _new_ledger(self, entry):
        """The ledger of a new image or video entry, set from its caches,
        or None. Before the insert's hold."""
        figs = self._read_lora(entry)
        if figs is None:
            return None
        led = _Ledger(int(getattr(entry.pipeline, "lora_own_term", 0) or 0))
        led.set(figs)
        led.base = led.held
        return led

    def _in_pool(self, entry) -> bool:
        return self._entries.get(entry.cache_key) is entry

    def _entry_bytes(self, e) -> int:
        led = e.lora
        return e.footprint + (0 if led is None else led.held + led.pending)

    def release(self, entry: _Entry) -> None:
        """Drop one in-flight reference taken by :meth:`acquire`. At zero the
        entry becomes LRU-evictable again."""
        with self._lock:
            entry.busy = max(0, entry.busy - 1)
            self._releases += 1
            self._idle.notify_all()

    def resident_adapter(self, model_path):
        """Adapter path of a resident entry for ``model_path`` (adapter
        inheritance), or None."""
        with self._lock:
            for entry in self._entries.values():
                if entry.model_path == model_path:
                    return entry.cache_key[1]
        return None

    @staticmethod
    def _live_count(entry: _Entry, ignore_retained: bool) -> int:
        return max(0, entry.busy - entry.retained) if ignore_retained else entry.busy

    def entry_resident(self, entry) -> bool:
        """Whether ``entry`` (a hold's ``_entry``) is still in the pool -
        false once an eviction tore it down."""
        with self._lock:
            return any(e is entry for e in self._entries.values())

    def clear(self, *, ignore_retained: bool = False) -> bool:
        """Tear down every resident entry that has no in-flight request. A busy
        entry (a client is mid-stream on it) is skipped, not stopped - tearing
        it down would kill that client's running generation; the TTL reaper or
        a later unload collects it once released. Returns whether anything was
        torn down. ``ignore_retained`` as in :meth:`evict`."""
        released: list = []
        torn: list = []
        with self._teardown_window(released):
            try:
                with self._lock:
                    for entry in list(self._entries.values()):
                        if self._live_count(entry, ignore_retained) > 0:
                            loadlog.verbose_print(
                                f"[residency] unload skipping busy model "
                                f"{os.path.basename(str(entry.model_path))} "
                                f"({entry.busy} in-flight)")
                            continue
                        self._detach(entry, torn)
                    cleared = bool(torn)
                    self._publish()
            finally:
                self._teardown_all(torn, released)
            return cleared

    def busy_paths(self) -> list:
        """Model paths that currently have in-flight requests."""
        with self._lock:
            return sorted({str(e.model_path) for e in self._entries.values()
                           if e.busy > 0})

    def mark_retained(self, hold) -> None:
        """Count ``hold`` (a ``_BusyHold`` the caller keeps for the process
        lifetime, e.g. the primary preload) as retained, so ``stats()`` can
        report ``in_flight`` = busy minus retained: the streams actually
        generating, which is what a dispatcher or the menu bar wants."""
        entry = getattr(hold, "_entry", None)
        if entry is None:
            return
        with self._lock:
            entry.retained += 1

    def unmark_retained(self, hold) -> None:
        """Undo :meth:`mark_retained` before the hold is released (an
        explicit unload drops the preload's lifetime hold)."""
        entry = getattr(hold, "_entry", None)
        if entry is None:
            return
        with self._lock:
            entry.retained = max(0, entry.retained - 1)

    def resident_entry(self, model_path):
        """The resident entry backing ``model_path`` (the one carrying the
        resident adapter when several profiles are loaded), or None. Never
        loads: a dry-run must not pull a model in."""
        with self._lock:
            hits = [e for e in self._entries.values()
                    if str(e.model_path) == str(model_path)]
        if not hits:
            return None
        return max(hits, key=lambda e: e.last_access)

    def model_path_for_generator(self, rg) -> str | None:
        """The resident model path whose ``response_generator`` is ``rg``
        (identity match), or None. The live-request view labels engine
        rows with a model id this way; the engine itself has no notion
        of which configured id it serves."""
        with self._lock:
            for e in self._entries.values():
                if e.response_generator is rg:
                    return str(e.model_path)
        return None

    def apc_managers(self, model_path=None) -> list:
        """``(model_path, apc_manager)`` for every resident entry that has
        one, or just the entries at ``model_path``. The per-model cache
        reset walks this; ``runtime.apc_manager`` alone resolves to the
        current request context's entry, which is one model at most."""
        with self._lock:
            return [(str(e.model_path), e.apc_manager)
                    for e in self._entries.values()
                    if e.apc_manager is not None
                    and (model_path is None
                         or str(e.model_path) == str(model_path))]

    def response_generators(self) -> list:
        """``(model_path, response_generator)`` for every resident entry
        that has an engine. The waiting census sums every engine's
        queue; ``runtime.response_generator`` outside a request context
        resolves to the last-used entry only."""
        with self._lock:
            return [(str(e.model_path), e.response_generator)
                    for e in self._entries.values()
                    if e.response_generator is not None]

    def evict(self, model_path, *, ignore_retained: bool = False) -> bool:
        """Tear down every resident entry backing ``model_path`` (a GGUF abspath),
        across all of its load profiles. Returns whether anything was evicted;
        raises :class:`ModelBusyError` if the model has in-flight requests (an
        unload must never stop another client's running generation).
        ``ignore_retained`` discounts process-lifetime holds (the preload's,
        see :meth:`mark_retained`) from that count, so an explicit unload can
        judge busy-ness on real streams alone without dropping the hold first
        - a 409 must leave the preload exactly as pinned as it found it.

        Also drops any keep mark - an explicit unload is a full release, so the
        path won't be silently re-kept (and TTL-exempted) on a later reload."""
        released: list = []
        torn: list = []
        with self._teardown_window(
                released, lambda e: e.model_path == model_path):
            with self._lock:
                self._keep_paths.discard(model_path)
                keys = [k for k, e in self._entries.items()
                        if e.model_path == model_path]
                in_flight = sum(
                    self._live_count(self._entries[k], ignore_retained)
                    for k in keys)
                if in_flight:
                    raise ModelBusyError(model_path, in_flight)
                for k in keys:
                    self._detach(self._entries[k], torn)
                self._publish()
            self._teardown_all(torn, released)
            return bool(keys)

    def set_keep(self, model_path, keep: bool) -> None:
        """Mark/unmark a path as kept resident through the idle-TTL reaper. A kept
        model is not pinned: it stays LRU-evictable under memory pressure. Path-based
        so it survives a re-acquire that would otherwise re-arm the entry's TTL."""
        with self._lock:
            if keep:
                self._keep_paths.add(model_path)
            else:
                self._keep_paths.discard(model_path)

    def flush_all(self) -> int:
        """Drain + stop every resident model's APC disk writer so shard writes
        still queued are landed before the process exits. Eviction already
        flushes per-model (see _teardown); this covers models that are still
        resident at graceful shutdown, where nothing else calls close(). The
        on-disk shards persist regardless - close() never deletes them. Models
        are left in the table (the process is exiting); a manager closed here is
        never closed again from this path, so there is no double-close. Returns
        the number of managers flushed."""
        with self._lock:
            managers = [e.apc_manager for e in self._entries.values()
                        if e.apc_manager is not None]
        n = 0
        for mgr in managers:
            _close = getattr(mgr, "close", None)
            if not callable(_close):
                continue
            try:
                _close()
                n += 1
            except Exception:  # noqa: S110 - best-effort sweep; count only the successful closes
                pass
        return n

    def stats(self) -> dict:
        with self._lock:
            now = self._time_fn()
            return {
                "budget_bytes": self._budget,
                "resident_bytes": self._resident_bytes(),
                "max_models": self._max,
                "resident": [
                    {
                        "model_path": e.model_path,
                        "kind": ("video" if e.kind == VIDEO_KIND else
                                 "image" if e.is_image else "chat"),
                        "loaded_as": e.loaded_as,
                        "profile": profile_label(e.cache_key),
                        "pinned": e.pinned,
                        "kept": e.model_path in self._keep_paths,
                        "seq": e.seq,
                        "busy": e.busy,
                        "retained": e.retained,
                        "in_flight": max(0, e.busy - e.retained),
                        "footprint_bytes": e.footprint,
                        "ttl_s": e.ttl,
                        "idle_s": max(0.0, now - e.last_access),
                        "kv_quant": (e.kv_policy.to_json()
                                     if e.kv_policy is not None else None),
                    }
                    for e in self._entries.values()
                ],
            }

    def reap_idle(self) -> list:
        """Tear down non-pinned entries idle past their TTL - but only when the
        server is **drained** (zero in-flight requests), so an idle reap can never
        race a live generation (in-flight 0 <=> no entry's generator is mid-stream).
        Returns the unloaded model paths. Called on a bounded tick by the daemon
        the installer starts; directly callable (with an injected clock) in tests."""
        in_flight = self._server_in_flight()
        if in_flight is None or in_flight != 0:
            return []                    # busy or unknown -> skip this tick
        now = self._time_fn()
        reaped = []
        released: list = []
        torn: list = []
        with self._teardown_window(released,
                                   lambda e: self._reapable(e, now)):
            try:
                with self._lock:
                    self._reap_locked(now, torn, reaped)
                    self._publish()
            finally:
                self._teardown_all(torn, released)
        return reaped

    def _reapable(self, entry, now) -> bool:
        """Under _lock: whether the reaper takes ``entry``, idle past its
        TTL and neither pinned, kept nor held past a preload's lifetime
        hold."""
        if (entry.pinned or entry.model_path in self._keep_paths
                or not entry.ttl or entry.ttl <= 0):
            return False
        if now - entry.last_access < entry.ttl:
            return False
        # A job is not in the server's in-flight count; its hold is: an
        # image or video job, and the chat model one rewrites its prompt
        # or encodes through.
        return entry.busy <= entry.retained

    def _reap_locked(self, now, torn: list, reaped: list) -> None:
        """Under _lock: take the entries idle past their TTL out of the
        pool into ``torn``."""
        for entry in list(self._entries.values()):
            if not self._reapable(entry, now):
                continue
            _log.info("idle TTL unload (%.0fs): %s",
                      entry.ttl, entry.model_path)
            self._detach(entry, torn)
            reaped.append(entry.model_path)

    def _server_in_flight(self):
        """The server-wide in-flight request count (mlx-vlm's metrics store), or
        ``None`` if it can't be read - in which case the reaper conservatively
        skips (never reaps when it can't confirm the server is drained)."""
        if self._in_flight_fn is not None:
            return self._in_flight_fn()
        try:
            return int(self._proxy.metrics._in_flight)
        except Exception:
            return None

    # internals
    def _touch(self, entry: _Entry):
        self._clock += 1
        entry.seq = self._clock
        entry.last_access = self._time_fn()   # resets the idle-TTL timer
        if entry.model_path in self._pinned_paths:
            entry.pinned = True
        self._entries.move_to_end(entry.cache_key)
        if not entry.is_image:
            # The proxy's fallback outside a request is the last chat
            # entry; an image entry has no stock state to read.
            self._proxy._last_entry = entry

    def _resident_bytes(self) -> int:
        """The footprints, the LoRA ledgers and bookings and the cold
        build in flight."""
        with self._lock:
            return sum(self._entry_bytes(e)
                       for e in self._entries.values()) + self._building

    def plan_beside(self, owner) -> int:
        """What the pool holds beside a request plan on entry ``owner``,
        from ``figures`` with no lock: the resident bytes less its weights
        and its held LoRA bytes, which the plan counts itself."""
        f = self.figures
        mine = next((h for e, h in zip(f.entries, f.held) if e is owner), 0)
        return f.resident - owner.footprint - mine

    def _stream_reservations(self, entries=None) -> tuple[int, int, int, int]:
        """``(room, ceiling margin, host overhead, host floor)`` the
        resident streamed models keep but have not filled. The room is a
        released prefill ring and the priced KV room. The host floor and
        the host overhead are what each arena was sized to leave in RAM,
        and the ceiling margin the part of them its ceiling leg charged,
        past the RAM outside the ceiling. The margins are the machine's
        and the process's, so they count once."""
        room = ceiling = host = floor = 0
        for e in (self._entries.values() if entries is None else entries):
            rg = e.response_generator
            feeder = _decode_feeder_of(rg)
            if feeder is None:
                continue
            room += int(getattr(feeder, "_room_bytes", 0) or 0)
            ceiling = max(ceiling,
                          int(getattr(feeder, "_outside_bytes", 0) or 0))
            host = max(host, int(getattr(feeder, "_host_bytes", 0) or 0))
            floor = max(floor, int(getattr(feeder, "_floor_bytes", 0) or 0))
            ring = _prefill_feeder_of(rg)
            if ring is not None and not getattr(ring, "_slots", None):
                room += max(0, _ring_room(ring)
                            - int(getattr(feeder, "_ring_share", 0) or 0))
        return room, ceiling, host, floor

    def _reserved_room_bytes(self) -> int:
        """What the resident streams keep under the ceiling: their room
        and the margin their ceiling leg charged. The headroom measure
        reads it as free; a load into it sheds the stream's arena at its
        next prefill."""
        room, ceiling, _, _ = self._stream_reservations()
        return room + ceiling

    def _over_capacity(self, incoming: int, held: int | None = None,
                       count: int | None = None) -> bool:
        """True if admitting ``incoming`` more weight bytes would exceed the
        byte budget or the optional count cap. ``held`` and ``count`` stand
        for the resident bytes and entries when given."""
        if held is None:
            held = self._resident_bytes()
        if count is None:
            count = len(self._entries)
        if self._budget and held + incoming > self._budget:
            return True
        if self._max is not None and count >= self._max:
            return True
        return False

    def _victim_kinds(self, incoming: int) -> frozenset:
        """Under _lock: the worker kinds of the entries
        :meth:`_evict_for_room` would evict to fit ``incoming``, taking
        victims in its order. Evicts nothing."""
        held = self._resident_bytes()
        count = len(self._entries)
        kinds = set()
        for e in sorted((e for e in self._entries.values()
                         if not e.pinned and e.busy == 0),
                        key=lambda e: e.seq):
            if not self._over_capacity(incoming, held, count):
                break
            if e.is_image:
                kinds.add(e.kind)
            held -= self._entry_bytes(e)
            count -= 1
        return frozenset(kinds)

    def _evict_for_room(self, incoming: int, *, model_path=None,
                        streaming: bool = False, torn: list,
                        strict: bool = False, extra: int = 0,
                        kind: str | None = None) -> bool:
        """Under _lock: take LRU idle unpinned entries out of the pool, into
        ``torn``, until ``incoming`` more bytes fit. The caller tears them
        down after _lock drops, under the build lock. Returns whether it
        fits.

        A cold load (not ``strict``) fits the budget and the optional count
        cap. It never evicts a pinned or busy (in-flight generation) model;
        if only those remain, it exceeds the budget rather than refuse the
        request or kill a live generation. A streaming model is the
        exception: its arena is sized to the ceiling, so an overcommit
        takes the room the live generation's KV needs and the governor
        sheds it. That one defers with the typed 503 instead.

        A LoRA turn (``strict``) fits the budget alone, with ``extra``
        upload bytes, and takes no entry unless the entries it takes make
        the room. It takes chat entries and those of its own worker
        ``kind`` only, whose release runs inline on that worker; the other
        worker's idle entries are named by its 503."""
        if strict:
            if not self._budget:
                return True
            short = self._lora_resident(extra) + incoming - self._budget
            victims, freed = [], 0
            for e in sorted((e for e in self._entries.values()
                             if not e.pinned and e.busy == 0
                             and (kind is None or _mine(e, kind))),
                            key=lambda e: e.seq):
                if freed >= short:
                    break
                victims.append(e)
                freed += self._entry_bytes(e)
            if freed < short:
                return False
            for v in victims:
                _log.info("evicting LRU model (%.1f GB) for LoRA bytes: %s",
                          v.footprint / 1e9, v.model_path)
                self._detach(v, torn)
            return True
        while self._entries and self._over_capacity(incoming):
            victim = min(
                (e for e in self._entries.values() if not e.pinned and e.busy == 0),
                key=lambda e: e.seq,
                default=None,
            )
            if victim is None:
                held = self._resident_bytes()
                if (streaming and self._budget
                        and held + incoming > self._budget):
                    _defer(
                        f"model load deferred: {model_path} priced at "
                        f"{incoming / 1e9:.1f} GB resident (every-token weights "
                        f"plus decode arena and prefill ring) does not fit the "
                        f"{self._budget / 1e9:.1f} GB budget beside "
                        f"{held / 1e9:.1f} GB of pinned or busy models. Retry "
                        "when a slot frees, or cap the arena with "
                        "GMLX_DECODE_ARENA_GB.")
                _log.warning(
                    "all %d resident models pinned or busy (%.1f GB); incoming "
                    "%.1f GB will exceed the %.1f GB budget",
                    len(self._entries), self._resident_bytes() / 1e9,
                    incoming / 1e9, self._budget / 1e9,
                )
                return False
            _log.info(
                "evicting LRU model (%.1f GB) to fit %.1f GB: %s",
                victim.footprint / 1e9, incoming / 1e9, victim.model_path,
            )
            self._detach(victim, torn)
        return True

    def _detach(self, entry: _Entry, torn: list) -> None:
        """Under _lock: take ``entry`` out of the pool, with its ledger,
        for a teardown after _lock drops. Its bytes count as tearing until
        that teardown ends, so no grant takes them first."""
        del self._entries[entry.cache_key]
        if self._proxy._last_entry is entry:
            self._proxy._last_entry = None
        entry.tearing = self._entry_bytes(entry)
        self._tearing += entry.tearing
        if entry.is_image:
            self._tearing_kind[entry.kind] += entry.tearing
        entry.lora = None
        torn.append(entry)

    def _build(self, cache_key, model_path, adapter_path, model_kind, footprint,
               *, ttl=None, env=None, build_spec=None) -> _Entry:
        """Build one entry of ``model_kind``: the shared gate and
        bookkeeping, around the build of its kind."""
        if model_kind in _WORKER_KINDS:
            return self._build_image(cache_key, model_path, footprint,
                                     ttl=ttl, build_spec=build_spec,
                                     kind=model_kind)
        return self._build_chat(cache_key, model_path, adapter_path,
                                model_kind, footprint, ttl=ttl, env=env,
                                build_spec=build_spec)

    def _new_entry(self, cache_key, model_path, footprint, ttl,
                   **fields) -> _Entry:
        return _Entry(cache_key=cache_key, model_path=model_path,
                      pinned=model_path in self._pinned_paths,
                      footprint=footprint, ttl=ttl,
                      last_access=self._time_fn(), **fields)


def _default_lora_figures(entry):
    from . import image_service

    return image_service.lora_figures(entry)


def _collect_failed_build() -> None:
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
    except Exception:  # noqa: S110 - cache release is advisory
        pass
