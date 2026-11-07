"""Multi-model residency for mlx-vlm's continuous-batching server.

Stock mlx-vlm holds exactly one model: a request for a different model evicts
and reloads the previous one (``get_cached_model`` -> ``unload_model_sync`` ->
rebuild). This installs a small **pinned + LRU pool** so several models stay
resident at once and switching between them is a pointer swap, not a multi-second
reload.

Seams (both late-bound by mlx-vlm, so a monkeypatch suffices - no fork):

* ``get_cached_model`` - the per-request model resolver. Every protocol handler
  reaches it through ``_server_package_attr("get_cached_model")`` and the
  lifespan preload calls ``app.get_cached_model`` directly, so we patch both. The
  pooled version resolves (or loads) the entry for the requested model, records
  it on a :class:`contextvars.ContextVar`, and returns it.
* the shared ``runtime`` singleton - the handlers read ``runtime.response_generator``
  / ``runtime.model_cache`` / ``runtime.apc_manager`` immediately after
  ``get_cached_model``. We replace ``runtime`` (bound in five server modules)
  with a :class:`_RuntimeProxy` whose three pooled fields resolve to the entry
  bound to the *current request's* context. Concurrent requests for different
  models therefore each see their own generator instead of racing on one global;
  the handler resolves ``runtime.response_generator.generate`` in its own context
  before handing it to the executor.

Loading and teardown delegate to stock ``get_cached_model`` /
``unload_model_sync`` through a context-isolated scratch object, so the pool
inherits mlx-vlm's exact load path (vision cache, KV-quant config,
``ResponseGenerator`` with its speculative drafter) rather than duplicating it.
The APC manager is the exception: the load bridge builds the gmlx-owned one
(``apc_manager.build_apc_manager``) and ``_build`` wires it in post-load,
with stock ``from_env`` pinned dead for the load's duration.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any, Callable, NamedTuple

import gmlx.load.loadlog as loadlog
from .capacity import LoadDeferred, _defer
from gmlx.envflags import env_float
# The descent to the module the loader hung the streaming helpers on. Shared
# with the loader, which reads the same helper set to charge a live install's
# wired bytes against the next one.
from gmlx.stream.installs import streaming_owner as _streaming_owner

from .residency_types import (  # noqa: F401 - re-exported
    _build_scratch,
    _Entry,
    _Figures,
    gb,
    IMAGE_KIND,
    _Ledger,
    LoraRefused,
    _mine,
    ModelBusyError,
    _owner_kind,
    profile_label,
    _Scratch,
    _STOCK_IMAGE_KINDS,
    STORE,
    _Store,
    VIDEO_KIND,
    _WORKER_KINDS,
    _WORKER_NAMES,
    _WORKER_ORDER,
    _WORKER_WORDS,
)
from .residency_footprint import (  # noqa: F401 - re-exported
    _decode_feeder_of,
    _drafter_bytes,
    _env_arena_cap,
    _gguf_footprint_bytes,
    _plain_footprint,
    _prefill_feeder_of,
    _prefill_ring_of,
    _ring_room,
    _stamp_apc_mode,
    _stamp_boot_kv_costs,
    _streamed_shares,
    _streaming_footprint,
)
from .residency_build import (  # noqa: F401 - re-exported
    _BATCH_UNSAFE_ARCHES,
    _batch_unsafe_warned,
    _ChatBuild,
    _clear_table_step,
    _latch_guarded_wire,
    _say_tables_offloaded,
    _warn_if_batch_unsafe,
)

_log = logging.getLogger(__name__)

# The entry bound to the current request (set by the pooled get_cached_model).
_active_entry: ContextVar["_Entry | None"] = ContextVar("_kq_active_entry", default=None)
# The current request's one-shot busy hold (set beside _active_entry); the
# generate seam transfers it to the generation's token iterator.
_active_hold: ContextVar["_BusyHold | None"] = ContextVar("_kq_active_hold", default=None)


_INSTALL_FLAG = "_kq_gguf_residency_installed"
# Resident models are bounded by a weight-byte budget (the primary mechanism),
# with an optional secondary count cap. The default budget is this fraction of
# the GPU's recommended working set, leaving headroom for the active model's KV
# cache + activations; override with MLX_VLM_RESIDENT_BUDGET_GB.
_DEFAULT_BUDGET_FRACTION = 0.8
# Fallback working set if the device can't report one (kept conservative).
_FALLBACK_WORKING_SET = 32 * 1024**3

_RUNTIME_MODULES = (
    "gmlx._vendor.mlx_vlm.server.runtime",
    "gmlx._vendor.mlx_vlm.server.app",
    "gmlx._vendor.mlx_vlm.server.anthropic",
    "gmlx._vendor.mlx_vlm.server.generation",
    "gmlx._vendor.mlx_vlm.server.openai",
    "gmlx._vendor.mlx_vlm.server",
)


def _default_budget_bytes() -> int:
    """``_DEFAULT_BUDGET_FRACTION`` of the GPU's recommended working set
    (working-set source: capacity, the one accounting)."""
    from .capacity import working_set_bytes

    working_set = working_set_bytes() or _FALLBACK_WORKING_SET
    return int(_DEFAULT_BUDGET_FRACTION * working_set)


class _Aside(NamedTuple):
    """A turn's answer that it waits on the releases queued on the other
    worker: the bytes it is short of, and the running count of the bytes
    their releases freed as it saw it."""

    short: int
    freed: int


class _BusyHold:
    """One-shot holder of an entry's in-flight refcount (taken at acquire).

    Released deterministically when the generation it covers finishes (the
    generate seam hands it to the token iterator); ``__del__`` backstops requests
    that never reach generate (token-count endpoints, pre-generate 4xx paths) -
    the hold dies with the request context, so the refcount can't leak."""

    __slots__ = ("_pool", "_entry")

    def __init__(self, pool, entry):
        self._pool = pool
        self._entry = entry

    def release(self):
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.release(self._entry)
            self._entry = None

    def __del__(self):
        try:
            self.release()
        except Exception:  # noqa: S110 - GC-time cleanup must never raise
            pass


class _ReleasingTokenIterator:
    """Wraps the engine's token iterator to release a busy hold exactly once when
    the generation ends - exhaustion, error, ``close()``, or GC."""

    def __init__(self, inner, hold):
        self._inner = inner
        self._hold = hold

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._inner)
        except BaseException:
            self._hold.release()
            raise

    def close(self):
        try:
            close = getattr(self._inner, "close", None)
            if close is not None:
                close()
        finally:
            self._hold.release()

    def __getattr__(self, name):
        if name in ("_inner", "_hold"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __del__(self):
        try:
            self._hold.release()
        except Exception:  # noqa: S110 - GC-time cleanup must never raise
            pass


class _GenerationGuard:
    """Per-request view of an entry's ``ResponseGenerator``: ``generate`` hands
    the request's busy hold to the returned token iterator, so the entry stays
    eviction-proof until the token stream finishes (not just until the next
    acquire bumps another entry's LRU seq). Everything else delegates."""

    __slots__ = ("_rg", "_hold")

    def __init__(self, rg, hold):
        self._rg = rg
        self._hold = hold

    def __getattr__(self, name):
        if name in ("_rg", "_hold"):
            raise AttributeError(name)
        return getattr(self._rg, name)

    def generate(self, *args, **kwargs):
        hold, self._hold = self._hold, None
        try:
            ctx, token_iter = self._rg.generate(*args, **kwargs)
        except BaseException:
            if hold is not None:
                hold.release()
            raise
        if hold is None:
            return ctx, token_iter
        return ctx, _ReleasingTokenIterator(token_iter, hold)


class _RuntimeProxy:
    """Context-aware view over the residency pool.

    ``response_generator`` / ``model_cache`` / ``apc_manager`` resolve, in order,
    to: the entry bound to the current request, the scratch of an in-flight stock
    load/teardown, then the most recently used entry (for context-free status
    reads such as ``/health``). Every other attribute (e.g. ``metrics``)
    passes through to the original runtime singleton.
    """

    def __init__(self, original):
        self._original = original
        self._last_entry: _Entry | None = None

    # pooled fields
    def _resolve(self):
        # A build/teardown sets the scratch in its own context and routes all
        # proxy access there (it must read an empty cache and write the loading
        # model's state) - so it takes precedence over any request entry that
        # may already be bound in the same context. Other contexts never see
        # another's scratch, so this stays concurrency-safe.
        scratch = _build_scratch.get()
        if scratch is not None:
            return scratch
        entry = _active_entry.get()
        if entry is not None:
            return entry
        return self._last_entry

    @property
    def response_generator(self):
        target = self._resolve()
        if target is None:
            return None
        rg = target.response_generator
        if isinstance(target, _Entry) and callable(getattr(rg, "generate", None)):
            # Resolved in the request's context (handlers bind the attribute
            # before any executor hop), so the busy hold is capturable here.
            return _GenerationGuard(rg, _active_hold.get())
        return rg

    @response_generator.setter
    def response_generator(self, value):
        target = self._resolve()
        if target is not None:
            target.response_generator = value

    @property
    def model_cache(self):
        target = self._resolve()
        return target.model_cache if target is not None else {}

    @model_cache.setter
    def model_cache(self, value):
        target = self._resolve()
        if target is not None:
            target.model_cache = value

    @property
    def apc_manager(self):
        target = self._resolve()
        return target.apc_manager if target is not None else None

    @apc_manager.setter
    def apc_manager(self, value):
        target = self._resolve()
        if target is not None:
            target.apc_manager = value

    # everything else (metrics, ...)
    def __getattr__(self, name):
        if name in ("_original", "_last_entry"):
            raise AttributeError(name)
        return getattr(self._original, name)


class _ResidencyPool(_ChatBuild):
    """Pinned + LRU pool of resident models, backed by stock load/teardown."""

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

    # The LoRA turn (docs/internals/image-steps.md). A worker's job books
    # the LoRA bytes it allocates next at its turn, under the build lock a
    # cold load holds, with _lock inside it for each read or write of the
    # entries and ledgers. No hold of _lock spans a teardown, a pipeline
    # release, a collection, a measure or a cache lock.

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

    def _reconcile(self, entry, figs) -> int:
        """Under _lock: set the ledger of ``entry``, the turn's owner, to
        ``figs`` and end its booking. Returns the bytes it holds past what
        it booked."""
        led = entry.lora if entry is not STORE else None
        if led is None or figs is None or not self._in_pool(entry):
            return 0
        led.set(figs)
        over = led.held - (led.base + led.pending)
        led.pending = 0
        led.base = led.held
        return max(0, over)

    def _in_pool(self, entry) -> bool:
        return self._entries.get(entry.cache_key) is entry

    def _overrun(self, subject: str, over: int) -> None:
        if over <= 0:
            return
        _log.info("%s hold %.1f MB of LoRA memory past their turn's "
                  "booking", subject, over / 1e6)
        if self.overrun_hook is not None:
            self.overrun_hook(subject, over)

    def _uploads(self) -> int:
        """Under _lock: the uploads' bytes, held and admitted. The store's
        count is one int read with no lock: a put counts its bytes there
        before they leave ``_incoming``, so the sum never reads short."""
        live = (self._store_live or _default_store_live)()
        return int(live) + self._incoming

    def _retry(self) -> int:
        if self._retry_fn is not None:
            return int(self._retry_fn())
        from .patches.request_flow import deferred_retry_after

        return int(deferred_retry_after())

    def _lora_resident(self, live: int) -> int:
        """Under _lock: what the budget counts with ``live`` upload bytes,
        the victims whose teardown runs included."""
        return self._resident_bytes() + self._tearing + live

    def _measured_room(self, owner, head, live: int):
        """Under _lock: the measured room of a turn, or None unmeasured:
        the measure less what the streams keep, the bookings of other
        owners, the build in flight and the uploads, host memory the
        measure misses."""
        if head is None:
            return None
        other = sum(e.lora.pending for e in self._entries.values()
                    if e.lora is not None and e is not owner)
        return (head - self._reserved_room_bytes() - other - self._building
                - live)

    def _fits(self, owner, need: int, head, live: int) -> bool:
        """Under _lock: whether ``need`` more bytes fit the budget and the
        measured room."""
        if self._budget and self._lora_resident(live) + need > self._budget:
            return False
        room = self._measured_room(owner, head, live)
        return room is None or need <= room

    def _book(self, owner, need: int) -> None:
        """Under _lock: book ``need`` bytes on the owner's ledger, or count
        them as admitted for the upload store."""
        if owner is STORE:
            self._incoming += int(need)
            return
        led = owner.lora
        if led is not None:
            led.base = led.held
            led.pending = int(need)

    def upload_admit(self, n: int, head) -> bool:
        """Count the ``n`` bytes of an upload body in the pool's incoming
        figure when they fit the budget and the measured room ``head``
        (read with no lock) beside what the pool holds. Runs off the event
        loop and takes the pool lock for the count. It books, claims and
        sheds no model or LoRA memory."""
        from .capacity import overcommit

        over = overcommit()
        with self._lock:
            ok = over or self._fits(STORE, n, head, self._uploads())
            if ok:
                self._book(STORE, n)
                self._publish()
        return ok

    def upload_done(self, n: int) -> None:
        """An admitted body was put, or failed: its bytes leave the
        admitted count."""
        with self._lock:
            self._incoming = max(0, self._incoming - int(n))
            self._publish()

    def _worker_guard(self, owner) -> None:
        kind = _owner_kind(owner)
        if not threading.current_thread().name.startswith(
                _WORKER_NAMES[kind]):
            raise RuntimeError(f"a LoRA turn runs on the {_WORKER_WORDS[kind]}"
                               " worker, where its job allocates")

    def lora_turn(self, owner, need: int, *, subject: str, hold=(),
                  held: int = 0, store=(), beside=()) -> None:
        """The turn of a job on its owner's worker: room for ``need`` bytes
        it allocates next, booked on the ledger of ``owner``, a pool entry,
        or admitted into the upload store. ``hold`` names the owner's keys
        its job keeps, ``held`` their bytes held now, ``store`` the uploads
        it keeps and ``beside`` the other entries its job holds, such as a
        borrowed encoder, which count as its own. Sheds idle LoRA bytes,
        evicts idle models as a cold load does, measures and books, or
        waits for chat models in use within the deadline. A release
        queued on the other worker frees its bytes first: after it sheds
        and before it evicts, the turn waits on it until the bytes freed
        cover its shortfall, within one window of half the deadline left
        when it first waits, which it may wait in more than once; the
        rest keeps the wait for the build lock.
        Raises LoraRefused: 413 when no unload makes room, else 503 with
        Retry-After."""
        from .capacity import overcommit

        self._worker_guard(owner)
        need = max(0, int(need))
        beside = tuple(beside)
        over = overcommit()
        retry = self._retry()
        deadline = time.monotonic() + retry
        figs = self._read_lora(owner)
        head = None if over else self._measured_lora_room()
        with self._lock:
            late = self._reconcile(owner, figs)
            # A turn that allocates nothing needs no room.
            ok = owner is not STORE and (
                need == 0 or self._fits(owner, need, head, self._uploads()))
            if ok:
                self._book(owner, need)
                self._hit("booked")
            self._publish()
        self._overrun(subject, late)
        if ok:
            return
        err = None
        aside_end = None
        while True:
            now = time.monotonic()
            left = deadline - now
            if err is not None and left <= 0:
                raise err
            if not self._build_lock.acquire(timeout=max(0.0, left)):
                head = (None if over or err is not None
                        else self._measured_lora_room())
                with self._lock:
                    if err is not None and not self._building:
                        raise err
                    raise self._late_503(owner, need, subject, retry, head,
                                         beside)
            try:
                got = self._turn_room(
                    owner, need, subject, hold, held, store, retry, over,
                    beside, aside=left > 0 and (aside_end is None
                                                or now < aside_end))
            finally:
                self._build_lock.release()
            if got is None:
                return
            if isinstance(got, _Aside):
                # Off the build lock, which the release may wait on. One
                # window, half the deadline left as the turn first waits:
                # past it, the turn evicts as before, with the rest to
                # wait for the build lock.
                if aside_end is None:
                    t = time.monotonic()
                    aside_end = t + (deadline - t) / 2
                with self._lock:
                    while (self._aside(owner) > 0 and self._aside_freed(owner)
                           - got.freed < got.short):
                        left = aside_end - time.monotonic()
                        if left <= 0:
                            break
                        self._idle.wait(left)
                continue
            seen, err = got
            with self._lock:
                while self._releases == seen:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise err
                    self._idle.wait(left)

    def _turn_room(self, owner, need, subject, hold, held, store, retry,
                   over, beside, aside: bool = True):
        """Under the build lock: shed, evict, measure, then check and book
        in one hold. Returns None when booked, (the release count, the
        503) to wait on, or, with ``aside``, an _Aside when the releases
        queued on the other worker free the bytes the turn is still short
        of after it sheds: the turn waits on them before it picks a
        victim."""
        self._hit("turn")
        head = None if over else self._measured_lora_room()
        with self._lock:
            short = self._shortfall(owner, need, head, self._uploads())
        if short > 0:
            self._shed_for(owner, short, hold, store)
            if aside and not over and self._has_aside(owner):
                head = self._measured_lora_room()
                with self._lock:
                    short = self._shortfall(owner, need, head,
                                            self._uploads())
                    if 0 < short <= self._aside(owner):
                        return _Aside(short, self._aside_freed(owner))
        torn: list = []
        released: list = []
        try:
            with self._lock:
                if not over:
                    self._evict_for_room(need, torn=torn, strict=True,
                                         extra=self._uploads(),
                                         kind=_owner_kind(owner))
                self._publish()
        finally:
            try:
                self._teardown_all(torn, released)
            finally:
                self._release_images(released)
        head = None if over else self._measured_lora_room()
        with self._lock:
            live = self._uploads()
            if over or self._fits(owner, need, head, live):
                self._book(owner, need)
                self._publish()
                return None
            return self._refusal(owner, need, head, live, held, subject,
                                 retry, beside)

    def _shortfall(self, owner, need: int, head, live: int) -> int:
        """Under _lock: the bytes past the budget or the measured room."""
        short = 0
        if self._budget:
            short = self._lora_resident(live) + need - self._budget
        room = self._measured_room(owner, head, live)
        if room is not None:
            short = max(short, need - room)
        return max(0, short)

    def _shed_for(self, owner, nbytes: int, hold, store) -> None:
        """Shed ``nbytes`` of idle LoRA bytes, after _lock dropped: the
        owner's own request bytes (``hold`` kept), its worker's other
        entries' by their last use, then the idle uploads (``store``
        kept). The store's own turn sheds the uploads first. The other
        worker's caches are never shed here, where its job may read them;
        an upload is its body's bytes, no worker's arrays, so any turn
        sheds it."""
        shed = self._lora_shed or _default_lora_shed
        store_shed = self._store_shed or _default_store_shed
        left = int(nbytes)
        kind = _owner_kind(owner)
        with self._lock:
            others = sorted((e for e in self._entries.values()
                             if e.lora is not None and e is not owner
                             and _mine(e, kind)),
                            key=lambda e: e.seq)
        order = ([STORE] + others if owner is STORE
                 else [owner] + others + [STORE])
        for src in order:
            if left <= 0:
                return
            if src is STORE:
                left -= int(store_shed(left, tuple(store)) or 0)
                continue
            mine = src is owner
            left -= int(shed(src, left, tuple(hold) if mine else (), mine)
                        or 0)
            figs = self._read_lora(src)
            with self._lock:
                if figs is not None and src.lora is not None \
                        and self._in_pool(src):
                    src.lora.set(figs)
                    src.lora.base = src.lora.held
                self._publish()

    @staticmethod
    def _own(owner, beside) -> list:
        """The entries a turn's job holds as its own: its owner, past the
        upload store, and ``beside``."""
        return ([] if owner is STORE else [owner]) + list(beside)

    def _hopeless_413(self, budget, entries, owner, beside, n: int,
                      subject: str) -> None:
        """The 413 of ``n`` bytes no unload makes room for: over the budget
        less the footprints of the job's own entries and the pinned
        models."""
        if not budget:
            return
        own = self._own(owner, beside)
        ids = {id(e) for e in own}
        kept = (sum(e.footprint for e in own)
                + sum(e.footprint for e in entries
                      if e.pinned and id(e) not in ids))
        if n <= budget - kept:
            return
        names = ", ".join(repr(self._lora_name(e)) for e in own)
        where = f"{names} and the pinned models" if names else \
            "the pinned models"
        raise LoraRefused(
            413, f"{subject} need {gb(n, True)} GB of memory, over the "
            f"{gb(budget - kept, False)} GB the server's memory budget "
            f"leaves beside {where}")

    def _lora_503(self, subject: str, n: int, room: int, retry: int,
                  own=(), measured: bool = False,
                  held_up: str | None = None) -> "LoraRefused":
        """Under _lock: the 503 of a turn left short. It names the cold
        load in flight, else what held up a turn with room (``held_up``),
        else the room the budget leaves beside the models in use whose
        bytes go once idle, past the job's own entries ``own``, or,
        ``measured``, the working set's."""
        lead = f"{subject} need {gb(n, True)} GB of memory"
        if self._building:
            return LoraRefused(
                503, f"{lead}, and the server is loading {self._loading} "
                f"({gb(self._building, True)} GB) now; retry when it is "
                "loaded", retry, transient=True)
        free = f"{gb(room, False)} GB is free"
        transient = True
        if held_up is not None:
            tail = held_up
        elif measured:
            busy = self._lora_busy(own)
            transient = bool(busy)
            tail = (f"{free} in the working set now; retry when the models "
                    "in use finish" if busy else
                    f"{free} in the working set now; close other apps or "
                    "unload a model to make room")
        else:
            busy = self._lora_busy(own, pinned=False)
            transient = bool(busy)
            tail = (f"{free} beside the models in use ({', '.join(busy)}); "
                    "retry when one is idle" if busy else
                    f"{free}; unload a model or retry when the requests in "
                    "flight finish")
        return LoraRefused(503, f"{lead}, and {tail}", retry,
                           transient=transient)

    def _aside_503(self, subject: str, n: int, room: int, retry: int,
                   es, ids, kind: str, short: int):
        """Under _lock: the 503 of a turn of worker ``kind`` that the idle
        entries of the other worker would make ``short`` bytes of room
        for, naming them, or None when they would not. A turn never waits
        on the other worker, so it evicts none of them."""
        names, freed = [], 0
        for e in sorted((e for e in es if e.is_image and e.kind != kind
                         and id(e) not in ids and not e.pinned
                         and e.busy == 0), key=lambda e: e.seq):
            if freed >= short:
                break
            names.append(repr(self._lora_name(e)))
            freed += self._entry_bytes(e)
        if not names or freed < short:
            return None
        other = next(k for k in _WORKER_ORDER if k != kind)
        them = "it" if len(names) == 1 else "them"
        verb = "is" if len(names) == 1 else "are"
        return LoraRefused(
            503, f"{subject} need {gb(n, True)} GB of memory, "
            f"{gb(room, False)} GB is free, and {', '.join(names)} {verb} "
            f"idle on the {_WORKER_WORDS[other]} worker; unload {them} "
            "(POST /unload) to make room", retry)

    def _refusal(self, owner, need, head, live, held, subject, retry,
                 beside):
        """Under _lock, in the hold that found the turn short: raise the
        413 when no unload makes room, the 503 at once when no chat model
        in use holds the bytes, else return (the release count, the 503)
        for the turn to wait on. The entries the job holds are its own,
        never waited for."""
        budget = self._budget
        es = list(self._entries.values())
        self._hopeless_413(budget, es, owner, beside, need + held, subject)
        own = self._own(owner, beside)
        ids = {id(e) for e in own}
        kind = _owner_kind(owner)
        chat = [e for e in es if id(e) not in ids and not e.pinned
                and not e.is_image and e.busy > e.retained]
        resident = self._lora_resident(live)
        if budget and resident + need > budget:
            # The idle entries this turn can evict: chat and its worker's.
            idle = sum(self._entry_bytes(e) for e in es
                       if id(e) not in ids and not e.pinned and e.busy == 0
                       and _mine(e, kind))
            freed = idle + sum(self._entry_bytes(e) for e in chat)
            short = resident + need - budget
            if freed < short:
                aside = self._aside_503(subject, need,
                                        max(0, budget - resident), retry,
                                        es, ids, kind, short - freed)
                if aside is not None:
                    raise aside
            err = self._lora_503(subject, need, max(0, budget - resident),
                                 retry, own)
            if not chat or freed < short:
                raise err
            return self._releases, err
        # The measured room is short: the budget passed in this hold.
        room = self._measured_room(owner, head, live)
        err = self._lora_503(subject, need, room or 0, retry, own,
                             measured=True)
        if not chat:
            raise err
        return self._releases, err

    def _late_503(self, owner, need: int, subject: str, retry: int, head,
                  beside=(), worker: bool = False) -> "LoraRefused":
        """Under _lock: the 503 of a turn that never reached its room
        check by its deadline, from the pool now and the measure ``head``
        read before the hold: the cold load in flight, else the budget
        short, else the measured room short, else what held the turn up,
        its worker's job (``worker``) or the build lock."""
        own = self._own(owner, beside)
        live = self._uploads()
        free = self._budget - self._lora_resident(live)
        if self._budget and need > free:
            return self._lora_503(subject, need, max(0, free), retry, own)
        room = self._measured_room(owner, head, live)
        if room is not None and need > room:
            return self._lora_503(subject, need, max(0, room), retry, own,
                                  measured=True)
        if worker:
            kind = _owner_kind(owner)
            names = [self._lora_name(e) for e in self._entries.values()
                     if e.kind == kind and e.busy > e.retained]
            held_up = (f"the {_WORKER_WORDS[kind]} worker is busy"
                       + (f" with {', '.join(names)}" if names else "")
                       + "; retry when it is idle")
        else:
            held_up = ("the server is loading or unloading another model "
                       "now; retry when it is done")
        return self._lora_503(subject, need, 0, retry, own, held_up=held_up)

    def lora_late(self, owner, need: int, subject: str) -> None:
        """The 503 of a turn that its worker did not start within
        its deadline. The measure and the deadline are read before the
        pool's lock."""
        from .capacity import overcommit

        retry = self._retry()
        head = None if overcommit() else self._measured_lora_room()
        with self._lock:
            raise self._late_503(owner, int(need), subject, retry, head,
                                 worker=True)

    def _entry_bytes(self, e) -> int:
        led = e.lora
        return e.footprint + (0 if led is None else led.held + led.pending)

    def lora_settle(self, entry, subject: str) -> None:
        """A job's settle, on the worker in its finally: the ledger set to
        what the caches hold once it ran, its booking ended. A ledger past
        its booking logs one line."""
        figs = self._read_lora(entry)
        with self._lock:
            over = self._reconcile(entry, figs)
            self._publish()
        self._overrun(subject, over)

    def lora_ledger(self, entry) -> tuple:
        """(pending, {key: bytes}) of the ledger of ``entry``."""
        with self._lock:
            led = entry.lora
            if led is None:
                return 0, {}
            return led.pending, dict(led.keys)

    def lora_hopeless(self, owner, n: int, subject: str,
                      beside=()) -> None:
        """A route's check of ``n`` bytes its job's turn would book, from
        ``figures`` with no lock: the 413 when no unload makes room beside
        the job's own entries, ``owner`` and ``beside``."""
        from .capacity import overcommit

        f = self.figures
        if overcommit():
            return
        self._hopeless_413(f.budget, f.entries, owner, tuple(beside),
                           int(n), subject)

    @staticmethod
    def _lora_name(e) -> str:
        return e.loaded_as or os.path.basename(str(e.model_path))

    def _measured_lora_room(self):
        """The free working set measured now, against the serve ceiling,
        or None where it cannot be measured. Read with no pool lock."""
        self._hit("measure")
        if self._headroom_fn is not None:
            return self._headroom_fn()
        from gmlx.gen.prefill_decay import headroom_bytes

        from .capacity import working_budget_bytes, working_set_bytes

        head = headroom_bytes()
        if head is None:
            return None
        budget, ws = working_budget_bytes(), working_set_bytes()
        if budget is not None and ws is not None:
            head -= max(0.0, ws - budget)
        return head

    def _lora_busy(self, own=(), pinned: bool = True) -> list:
        """Under _lock: the names of the models in use beside the entries
        ``own`` a job holds, the pinned ones only with ``pinned``."""
        ids = {id(e) for e in own}
        return [self._lora_name(e) for e in self._entries.values()
                if e.busy > e.retained and id(e) not in ids
                and (pinned or not e.pinned)]

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

    def _torn(self, kind, n: int) -> None:
        """Under _lock: ``n`` tearing bytes, of a pipeline of worker
        ``kind`` or of a chat model (None), are gone. A turn that waits on
        the other worker's release wakes."""
        self._tearing = max(0, self._tearing - n)
        if kind is not None:
            left = max(0, self._tearing_kind[kind] - n)
            self._freed_kind[kind] += self._tearing_kind[kind] - left
            self._tearing_kind[kind] = left
        self._publish()
        if n:
            self._idle.notify_all()

    def _has_aside(self, owner) -> bool:
        with self._lock:
            return self._aside(owner) > 0

    def _aside(self, owner) -> int:
        """Under _lock: the tearing bytes of the pipelines the other worker
        than the turn's of ``owner`` releases. With the build lock held,
        each is a release queued there by a teardown done after it dropped
        the lock, which frees its bytes once it runs."""
        kind = _owner_kind(owner)
        return sum(n for k, n in self._tearing_kind.items() if k != kind)

    def _aside_freed(self, owner) -> int:
        """Under _lock: the running count of the bytes the releases on the
        other worker than the turn's of ``owner`` freed."""
        kind = _owner_kind(owner)
        return sum(n for k, n in self._freed_kind.items() if k != kind)

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

    def _image_fn(self, i: int) -> Callable[..., Any]:
        if self._image_fns is None:
            from . import image_service

            self._image_fns = (image_service.entry_footprint,
                               image_service.load_on_worker,
                               image_service.release_on_worker)
        return self._image_fns[i]

    def _kind_fn(self, kind: str, i: int) -> Callable[..., Any]:
        """Function ``i`` of a worker kind's (footprint, load, release)."""
        if kind == IMAGE_KIND:
            return self._image_fn(i)
        if self._video_fns is None:
            from . import video_service

            self._video_fns = (video_service.entry_footprint,
                               video_service.load_on_worker,
                               video_service.release_on_worker)
        return self._video_fns[i]

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

    def _build_image(self, cache_key, model_path, footprint, *, ttl=None,
                     build_spec=None, kind: str = IMAGE_KIND) -> _Entry:
        """An image or video entry: the kind's ``load`` on the image
        worker, gated on the weights and the activation reserve together.
        Never reaches the stock loader."""
        self._gate(footprint, model_path, image=True)
        pipe = self._kind_fn(kind, 1)(build_spec)
        return self._new_entry(cache_key, model_path, footprint, ttl,
                               kind=kind, pipeline=pipe)

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

    def _worker_for(self, build_kind=None, *, incoming: int | None = None,
                    force: frozenset = frozenset(), teardown=None):
        """Hold the workers before the build lock, image before video,
        that a build loads its pipeline on (``build_kind``) or may release
        one on: those whose entries the eviction for ``incoming`` bytes
        takes, or, with ``incoming`` None, a teardown's, the workers of the
        idle entries ``teardown(entry)`` picks (every idle one when None);
        and those of ``force``. Then the lock is never held
        while a worker runs a job, and a chat cold load waits behind no
        job of a worker it does not release on. Without any, nothing is
        held."""
        with self._lock:
            kinds = set(force)
            if build_kind in _WORKER_KINDS:
                kinds.add(build_kind)
            if incoming is None:
                kinds.update(e.kind for e in self._entries.values()
                             if e.is_image and e.busy == 0
                             and (teardown is None or teardown(e)))
            else:
                kinds.update(self._victim_kinds(incoming))
        if not kinds:
            return contextlib.nullcontext()
        return _reserved_workers(frozenset(kinds))

    def _release_images(self, pipelines: list) -> None:
        """Release image and video pipelines, ``(kind, pipeline, bytes)``,
        each on its kind's worker, where its arrays were made, and wait for
        each, so the next gate measures the memory back. Each one's bytes
        leave the tearing count once its release ends. Call without
        ``_lock``."""
        while pipelines:
            kind, pipe, n = pipelines.pop()
            try:
                if pipe is None:
                    continue
                box = [pipe]
                try:
                    self._kind_fn(kind, 2)(box)
                except Exception:
                    _log.warning("teardown: image pipeline release failed",
                                 exc_info=True)
                finally:
                    box.clear()
                    del pipe
                    import gc
                    gc.collect()
            finally:
                if n:
                    with self._lock:
                        self._torn(kind, n)

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


def _default_lora_figures(entry):
    from . import image_service

    return image_service.lora_figures(entry)


def _default_lora_shed(entry, nbytes: int, hold, own: bool) -> int:
    from . import image_service

    return image_service.shed_lora(entry, nbytes, hold, own)


def _default_store_shed(nbytes: int, hold=()) -> int:
    from .image_steps import STATE

    return STATE.adapters.shed(nbytes, hold)


def _default_store_live() -> int:
    from .image_steps import STATE

    return STATE.adapters.bytes


def _holds_image_worker() -> bool:
    from . import image_service

    return image_service.worker_reserved()


def _holds_video_worker() -> bool:
    from . import video_service

    return video_service.worker_reserved()


def _holds_worker(kind: str) -> bool:
    """Whether the calling thread holds the worker of ``kind`` or is it."""
    return _holds_image_worker() if kind == IMAGE_KIND else \
        _holds_video_worker()


@contextlib.contextmanager
def _reserved_workers(kinds: frozenset):
    """Hold the workers of ``kinds``, image before video, the one order
    every thread takes them in."""
    from . import image_service, video_service

    holds = {IMAGE_KIND: image_service.reserved_worker,
             VIDEO_KIND: video_service.reserved_worker}
    with contextlib.ExitStack() as stack:
        for k in _WORKER_ORDER:
            if k in kinds:
                stack.enter_context(holds[k]())
        yield


def _collect_failed_build() -> None:
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
    except Exception:  # noqa: S110 - cache release is advisory
        pass


def _pinned_from_env(preload_path):
    pinned = set()
    if preload_path:
        pinned.add(preload_path)
    extra = os.environ.get("MLX_VLM_PINNED_MODELS", "")
    pinned.update(p.strip() for p in extra.split(",") if p.strip())
    return pinned


def _http_from_load_deferred(model_id, exc):
    """The gate's retryable refusal (``capacity.LoadDeferred``) as the
    ``fastapi.HTTPException`` mlx-vlm's endpoints re-raise untouched: the
    typed 503 body the chat pre-warm serves (``model_load_deferred``) with
    the same ``Retry-After``; the app's HTTPException handler unwraps the
    envelope and forwards the header."""
    import json

    from fastapi import HTTPException

    from gmlx.serve.patches.request_flow import _load_deferred_response
    resp = _load_deferred_response(model_id, exc)
    body = json.loads(bytes(resp.body))
    err = dict(body["error"])
    err["retry_after_s"] = body.get("retry_after_s")
    return HTTPException(status_code=503, detail={"error": err},
                         headers={"Retry-After": resp.headers["retry-after"]})


def _http_from_resolver_error(exc):
    """Translate a serving resolver error into the ``fastapi.HTTPException`` that
    mlx-vlm's endpoints re-raise untouched (each wraps ``get_cached_model`` in
    ``try: ... except HTTPException: raise; except Exception: -> 500``). Raising a
    plain domain exception would be swallowed into a 500; this gives a clean 4xx
    with the available ids/profiles. The detail carries the same
    ``{"error": {"type", "message", ...}}`` body the app-level resolver handlers
    emit (``install_resolver_error_handlers``); the app's HTTPException handler
    unwraps it so both paths serve one shape. ``None`` if ``exc`` isn't a
    resolver error."""
    from fastapi import HTTPException

    from . import bridge_vlm as _serving
    if isinstance(exc, _serving.WrongModelKind):
        return HTTPException(status_code=400, detail={"error": {
            "type": "wrong_model_kind", "message": str(exc),
            "available_models": exc.available}})
    if isinstance(exc, _serving.ModelNotFound):
        return HTTPException(status_code=404, detail={"error": {
            "type": "model_not_found", "message": str(exc),
            "available_models": exc.available}})
    if isinstance(exc, _serving.ModelFileMissing):
        return HTTPException(status_code=404, detail={"error": {
            "type": "model_file_missing", "message": str(exc)}})
    if isinstance(exc, _serving.UnknownProfile):
        return HTTPException(status_code=400, detail={"error": {
            "type": "unknown_profile", "message": str(exc),
            "available_profiles": exc.available}})
    if isinstance(exc, _serving.NoModelSpecified):
        return HTTPException(status_code=400, detail={"error": {
            "type": "no_model_specified", "message": str(exc),
            "available_models": exc.available}})
    return None


def _installed_pool():
    import sys

    pkg = sys.modules.get("gmlx._vendor.mlx_vlm.server")
    return getattr(pkg, "_kq_residency_pool", None) if pkg else None


def acquire_image_model(model_field, *, pool=None) -> "_BusyHold":
    """Resolve an image route's ``model`` field and acquire its entry,
    loading it on a miss. Returns the busy hold the caller releases when
    its job ends; ``hold._entry.pipeline`` is the ``ServedImage``.
    Resolver errors and a deferred load raise the ``HTTPException`` the
    chat routes raise: a chat model id is a 400, as an image id is on a
    chat route."""
    from fastapi import HTTPException

    from . import bridge_vlm as _serving

    pool = pool or _installed_pool()
    if pool is None or _serving.server_config() is None:
        raise HTTPException(status_code=503, detail={"error": {
            "type": "server_error", "message": "no residency pool"}})
    try:
        path, spec = _serving.resolve_image_request(model_field)
    except (_serving.ModelNotFound, _serving.ModelFileMissing,
            _serving.NoModelSpecified) as e:
        raise (_http_from_resolver_error(e) or e) from e
    from gmlx.image.fingerprint import ImageModelError

    try:
        entry = pool.acquire(path, None, IMAGE_KIND, ttl=spec.ttl_s,
                             cache_key_extra=spec.load_signature(),
                             build_spec=spec, loaded_as=spec.id)
    except LoadDeferred as e:
        raise _http_from_load_deferred(str(model_field or spec.id), e) from e
    except FileNotFoundError as e:
        # An absolute path is not checked at resolve; a file that is gone
        # surfaces here, at load, with the 404 a chat model gets.
        miss = _serving.ModelFileMissing(spec.id, str(e))
        raise (_http_from_resolver_error(miss) or miss) from e
    except ImageModelError as e:
        # A part of the entry that is not of its family: the config names
        # the wrong file, which no retry fixes.
        raise HTTPException(status_code=400, detail={"error": {
            "type": "image_model_mismatch", "code": "image_model_mismatch",
            "message": str(e), "model": spec.id}}) from e
    return _BusyHold(pool, entry)


def acquire_video_model(model_field, *, pool=None) -> "_BusyHold":
    """Acquire the entry of a configured video model id, loading it on a
    miss, as :func:`acquire_image_model` does an image model;
    ``hold._entry.pipeline`` is the ``ServedVideo``. A file that is no
    model of the family, or an entry whose default clip does not fit the
    working budget, is a 400."""
    from fastapi import HTTPException

    from gmlx.video.fingerprint import VideoModelError
    from gmlx.video.memfit import VideoMemoryError

    from . import bridge_vlm as _serving

    pool = pool or _installed_pool()
    if pool is None or _serving.server_config() is None:
        raise HTTPException(status_code=503, detail={"error": {
            "type": "server_error", "message": "no residency pool"}})
    try:
        path, spec = _serving.resolve_video_request(model_field)
    except (_serving.ModelNotFound, _serving.ModelFileMissing) as e:
        raise (_http_from_resolver_error(e) or e) from e
    try:
        entry = pool.acquire(path, None, VIDEO_KIND, ttl=spec.ttl_s,
                             cache_key_extra=spec.load_signature(),
                             build_spec=spec, loaded_as=spec.id)
    except LoadDeferred as e:
        raise _http_from_load_deferred(str(model_field), e) from e
    except FileNotFoundError as e:
        miss = _serving.ModelFileMissing(spec.id, str(e))
        raise (_http_from_resolver_error(miss) or miss) from e
    except (VideoModelError, VideoMemoryError) as e:
        raise HTTPException(status_code=400, detail={"error": {
            "type": "video_model_mismatch", "code": "video_model_mismatch",
            "message": str(e), "model": spec.id}}) from e
    return _BusyHold(pool, entry)


def install_gguf_residency_pool(budget_bytes=None, max_models=None, pinned=None) -> None:
    """Replace mlx-vlm's single-model slot with a pinned + LRU residency pool.

    Idempotent. Residency is bounded primarily by a **weight-byte budget**:
    ``budget_bytes`` (or ``MLX_VLM_RESIDENT_BUDGET_GB``), defaulting to
    ``_DEFAULT_BUDGET_FRACTION`` of the GPU's recommended working set. An
    optional secondary **count cap** comes from ``max_models`` (or
    ``MLX_VLM_MAX_RESIDENT_MODELS``); unset means bytes alone govern. The
    preloaded model (``MLX_VLM_PRELOAD_MODEL``) and any in
    ``MLX_VLM_PINNED_MODELS`` are pinned. Must be called before the lifespan
    preload runs.
    """
    import importlib
    import sys

    package = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    if getattr(package, _INSTALL_FLAG, False):
        return

    if budget_bytes is None:
        env_gb = os.environ.get("MLX_VLM_RESIDENT_BUDGET_GB")
        budget_bytes = int(float(env_gb) * 1024**3) if env_gb else _default_budget_bytes()
    if max_models is None:
        env_max = os.environ.get("MLX_VLM_MAX_RESIDENT_MODELS")
        max_models = int(env_max) if env_max else None
    pinned_paths = set(pinned or ()) | _pinned_from_env(
        os.environ.get("MLX_VLM_PRELOAD_MODEL")
    )

    original_runtime = sys.modules["gmlx._vendor.mlx_vlm.server.runtime"].runtime
    proxy = _RuntimeProxy(original_runtime)
    pool = _ResidencyPool(
        proxy,
        app.get_cached_model,
        app.unload_model_sync,
        budget_bytes,
        pinned_paths,
        max_models=max_models,
    )
    # Startup banner, not an ops message: install runs before uvicorn's
    # dictConfig wires the gmlx logger, so an INFO record here would be
    # dropped silently. print keeps the resolved budget visible.
    print(
        f"[residency] weight-byte budget {budget_bytes / 1e9:.1f} GB"
        + (f", count cap {max_models}" if max_models is not None else "")
    )

    inherit = app._INHERIT_ADAPTER

    import gmlx.config as _config
    from . import bridge_vlm as _serving

    def pooled_get_cached_model(model_path, adapter_path=inherit, *, model_kind="auto"):
        if model_kind in _STOCK_IMAGE_KINDS:
            # mlx-vlm's image handlers would load its own safetensors
            # pipelines from a request; gmlx serves images itself.
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail={"error": {
                "type": "invalid_request_error",
                "message": f"model kind {model_kind!r} is not served; image "
                           "models load through gmlx's image routes"}})
        # The incoming ``model_path`` is a friendly id (maybe ``id@profile``):
        # resolve it to a concrete GGUF abspath + merged spec, bind the spec for
        # this request (read at the gen-args seam), fold the load_signature into
        # the cache_key, and carry the load/APC env into the build window. Every
        # `gmlx serve` mode registers a config, a single model and a discovery
        # scan included. With none registered, a request never names a load
        # path or a hub id: a client holding the key could otherwise make the
        # Mac load any file.
        if _serving.server_config() is None:
            miss = _serving.ModelNotFound(str(model_path), {})
            raise _http_from_resolver_error(miss) or miss
        try:
            # The body `profile` field rides a request-scoped ContextVar
            # (install_request_profile_capture) - this seam only receives
            # the model string. Inline `id@profile` still wins inside.
            load_path, spec = _serving.resolve_request_model(
                model_path,
                profile_field=_serving.get_request_profile())
        except (_serving.ModelNotFound, _serving.ModelFileMissing,
                _serving.UnknownProfile, _serving.NoModelSpecified) as e:
            raise (_http_from_resolver_error(e) or e) from e
        _serving.set_active_spec(spec)
        ttl = spec.ttl_s
        cache_key_extra = spec.load_signature()
        env = _config.env_for(spec)
        build_spec = spec          # crosses into the load worker thread (see _build)
        loaded_as = spec.id + (f"@{spec.profile_name}" if spec.profile_name else "")
        if adapter_path is inherit:
            adapter_path = pool.resident_adapter(load_path)
        try:
            entry = pool.acquire(load_path, adapter_path, model_kind,
                                 ttl=ttl, cache_key_extra=cache_key_extra, env=env,
                                 build_spec=build_spec, loaded_as=loaded_as)
        except LoadDeferred as e:
            # The chat pre-warm answers the common case typed; a load that
            # begins here (another request took the room between the
            # pre-warm and this acquire, or a route without a pre-warm)
            # must serve the same 503, not the stock handler's 500.
            raise _http_from_load_deferred(str(model_path), e) from e
        except FileNotFoundError as e:
            # Absolute-path entries skip the resolver's existence check by
            # contract ("you said exactly where it is"), so a deleted file
            # first surfaces here, at load. Give it the same typed 404 the
            # relative/hf shapes get instead of a raw 500 errno.
            if _serving.server_config() is not None:
                miss = _serving.ModelFileMissing(str(model_path), str(e))
                raise (_http_from_resolver_error(miss) or miss) from e
            raise
        _active_entry.set(entry)
        # The acquire's in-flight hold: the generate seam (_GenerationGuard)
        # transfers it to the token stream, so eviction skips this entry until
        # the generation finishes. Same-context re-acquire rebalances first.
        # The lifespan preload's hold lives as long as its context, keeping the
        # preloaded model eviction-proof (it is normally pinned anyway); the
        # drained-gated TTL reaper is unaffected.
        prev = _active_hold.get()
        if prev is not None:
            prev.release()
        _active_hold.set(_BusyHold(pool, entry))
        mc = entry.model_cache
        return mc["model"], mc.get("processor"), mc.get("config")

    def pooled_unload_model_sync():
        return pool.clear()

    # Swap the runtime singleton everywhere it is bound.
    for modname in _RUNTIME_MODULES:
        mod = sys.modules.get(modname)
        if mod is not None and getattr(mod, "runtime", None) is original_runtime:
            setattr(mod, "runtime", proxy)

    # Patch the model resolver on both the package (handler resolution order)
    # and app (the lifespan's direct call).
    for target in (package, app):
        setattr(target, "get_cached_model", pooled_get_cached_model)
        setattr(target, "unload_model_sync", pooled_unload_model_sync)

    setattr(package, "_kq_residency_pool", pool)
    setattr(package, _INSTALL_FLAG, True)

    # Graceful-shutdown flush: drain each resident model's APC disk writer so
    # shard writes queued at exit aren't lost. uvicorn's SIGTERM unwinds to a
    # normal interpreter exit, so atexit fires; SIGKILL can't be caught (nothing
    # can flush there). Eviction handles still-running models; this handles the
    # ones left resident when the server stops.
    atexit.register(pool.flush_all)

    # Idle-TTL reaper: a daemon that periodically unloads non-pinned models gone
    # unused past their resolved ttl_s (gated on the server being drained - see
    # _ResidencyPool.reap_idle). LRU-under-pressure stays the floor; this just
    # frees memory when the server is quiet. Disable with MLX_VLM_RESIDENT_TTL_DISABLE.
    if os.environ.get("MLX_VLM_RESIDENT_TTL_DISABLE", "").strip().lower() not in (
        "1", "true", "yes", "on"
    ):
        tick = env_float("MLX_VLM_RESIDENT_TTL_TICK", 30.0)

        def _ttl_reaper():
            while True:
                time.sleep(tick)
                try:
                    pool.reap_idle()
                except Exception as exc:  # never let the reaper kill its thread
                    _log.error("TTL reaper error: %s", exc)

        threading.Thread(target=_ttl_reaper, name="gmlx-residency-ttl",
                         daemon=True).start()
