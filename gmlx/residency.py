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
inherits mlx-vlm's exact load path (vision cache, APC manager, KV-quant config,
``ResponseGenerator`` with its speculative drafter) rather than duplicating it.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass


from . import loadlog
from .envflags import env_float

_log = logging.getLogger(__name__)

# The entry bound to the current request (set by the pooled get_cached_model).
_active_entry: ContextVar["_Entry | None"] = ContextVar("_kq_active_entry", default=None)
# The current request's one-shot busy hold (set beside _active_entry); the
# generate seam transfers it to the generation's token iterator.
_active_hold: ContextVar["_BusyHold | None"] = ContextVar("_kq_active_hold", default=None)
# Scratch target for an in-flight stock load/teardown, isolated to the building
# context so it never leaks into concurrent resident-model requests.
_build_scratch: ContextVar["_Scratch | None"] = ContextVar(
    "_kq_build_scratch", default=None
)

# Archs whose mlx-lm model hand-rolls attention in a way that mishandles the batch
# generator's ragged left-padded mask, so concurrent/ragged batched decode is
# unreliable. The server batches, so warn the operator once per model. gemma2 is the
# lone case: it hand-rolls the +/-50 attn-logit softcap then applies the mask manually
# instead of via scaled_dot_product_attention. Verified upstream, not a gmlx/kq
# bug - single-stream matches llama.cpp token-for-token, and the sdpa-based gemma3 /
# gemma4 batch cleanly (see tests/test_batch_parity.py CANDIDATE_ARCHES note).
_BATCH_UNSAFE_ARCHES = {"gemma2"}
_batch_unsafe_warned: set = set()


def _warn_if_batch_unsafe(model_path: str) -> None:
    """Warn once, to stderr, if ``model_path``'s arch is unreliable under the server's
    ragged batched decode. Best-effort: a header-read failure stays silent (the load
    itself will surface any real problem)."""
    if not model_path or model_path in _batch_unsafe_warned:
        return
    _batch_unsafe_warned.add(model_path)
    try:
        from gguf import GGUFReader

        f = GGUFReader(model_path, "r").fields.get("general.architecture")
        arch = bytes(f.parts[f.data[0]]).decode("utf-8") if f is not None else ""
    except Exception:
        return  # advisory warning only; an unreadable header stays silent
    if arch in _BATCH_UNSAFE_ARCHES:
        _log.warning(
            "%r batched decode is unreliable - mlx-lm hand-rolls %s softcap "
            "attention and its manual mask mishandles the batch generator's "
            "ragged left-padded mask, so concurrent requests can corrupt "
            "output. Single-stream output is correct; serve %s one request "
            "at a time until this is fixed upstream.",
            arch, arch, arch,
        )


_INSTALL_FLAG = "_kq_gguf_residency_installed"
# Resident models are bounded by a weight-byte budget (the primary mechanism),
# with an optional secondary count cap. The default budget is this fraction of
# the GPU's recommended working set, leaving headroom for the active model's KV
# cache + activations; override with MLX_VLM_RESIDENT_BUDGET_GB.
_DEFAULT_BUDGET_FRACTION = 0.8
# Fallback working set if the device can't report one (kept conservative).
_FALLBACK_WORKING_SET = 32 * 1024**3

_RUNTIME_MODULES = (
    "mlx_vlm.server.runtime",
    "mlx_vlm.server.app",
    "mlx_vlm.server.anthropic",
    "mlx_vlm.server.generation",
    "mlx_vlm.server.openai",
    "mlx_vlm.server",
)


def _gguf_footprint_bytes(model_path: str) -> int:
    """On-disk size of a GGUF (summed over split shards) - a precise a-priori
    estimate of a model's resident weight bytes, since the loader mmaps the wire
    bytes zero-copy. Returns 0 if the path can't be sized (e.g. a non-GGUF
    mlx-vlm source), so such a model simply doesn't count against the budget."""
    try:
        from .preflight import find_split_shards

        return sum(os.path.getsize(s) for s in find_split_shards(model_path))
    except Exception:
        return 0


def _default_budget_bytes() -> int:
    """``_DEFAULT_BUDGET_FRACTION`` of the GPU's recommended working set."""
    import mlx.core as mx

    try:
        working_set = int(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        working_set = _FALLBACK_WORKING_SET
    return int(_DEFAULT_BUDGET_FRACTION * working_set)


class _Scratch:
    """Mutable stand-in for the three pooled runtime fields during a stock
    load or teardown. Empty ``model_cache`` makes stock ``get_cached_model``
    take the cache-miss path (no spurious unload of a resident model)."""

    __slots__ = ("response_generator", "model_cache", "apc_manager")

    def __init__(self):
        self.response_generator = None
        self.model_cache = {}
        self.apc_manager = None


class ModelBusyError(RuntimeError):
    """An explicit unload targeted a model with in-flight requests."""

    def __init__(self, model_path, in_flight: int):
        self.model_path = model_path
        self.in_flight = in_flight
        super().__init__(
            f"{model_path} has {in_flight} in-flight request(s)")


@dataclass
class _Entry:
    cache_key: tuple
    model_path: str
    model_cache: dict
    response_generator: object
    apc_manager: object
    pinned: bool = False
    seq: int = 0
    footprint: int = 0          # resident weight bytes (on-disk GGUF size)
    ttl: float | None = None  # idle auto-unload seconds (None/0 => never)
    last_access: float = 0.0    # monotonic time of the last acquire/touch
    busy: int = 0               # in-flight refcount; busy entries are never LRU-evicted


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
        except Exception:
            pass  # GC-time cleanup must never raise


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
        except Exception:
            pass  # GC-time cleanup must never raise


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


class _ResidencyPool:
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
    ):
        self._proxy = proxy
        self._stock_get = stock_get
        self._stock_unload = stock_unload
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

    # public
    def acquire(self, model_path, adapter_path, model_kind, *,
                ttl=None, cache_key_extra=(), env=None, build_spec=None) -> _Entry:
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
                return entry
        # Cold path: one build at a time (bounds peak memory).
        incoming = self._footprint_fn(model_path)
        # Refuse an unloadable GGUF *before* evicting healthy residents for
        # room - a corrupt/unsupported file would otherwise cost every LRU
        # victim a full cold reload for a build that was never going to work.
        # Header-only, so the cost is milliseconds; the loader re-validates.
        if (isinstance(model_path, str) and model_path.endswith(".gguf")
                and os.path.isfile(model_path)):
            from .preflight import preflight
            preflight(model_path)
        with self._build_lock:
            with self._lock:
                entry = self._entries.get(cache_key)
                if entry is not None:
                    if ttl is not None:
                        entry.ttl = ttl
                    self._touch(entry)
                    entry.busy += 1
                    return entry
                self._evict_for_room(incoming)
            entry = self._build(cache_key, model_path, adapter_path, model_kind,
                                incoming, ttl=ttl, env=env, build_spec=build_spec)
            with self._lock:
                self._entries[cache_key] = entry
                self._touch(entry)
                entry.busy += 1
            return entry

    def release(self, entry: _Entry) -> None:
        """Drop one in-flight reference taken by :meth:`acquire`. At zero the
        entry becomes LRU-evictable again."""
        with self._lock:
            entry.busy = max(0, entry.busy - 1)

    def resident_adapter(self, model_path):
        """Adapter path of a resident entry for ``model_path`` (adapter
        inheritance), or None."""
        with self._lock:
            for entry in self._entries.values():
                if entry.model_path == model_path:
                    return entry.cache_key[1]
        return None

    def clear(self) -> bool:
        """Tear down every resident entry that has no in-flight request. A busy
        entry (a client is mid-stream on it) is skipped, not stopped - tearing
        it down would kill that client's running generation; the TTL reaper or
        a later unload collects it once released. Returns whether anything was
        torn down."""
        with self._build_lock, self._lock:
            cleared = False
            for key, entry in list(self._entries.items()):
                if entry.busy > 0:
                    loadlog.verbose_print(
                        f"[residency] unload skipping busy model "
                        f"{os.path.basename(str(entry.model_path))} "
                        f"({entry.busy} in-flight)")
                    continue
                self._teardown(entry)
                if self._proxy._last_entry is entry:
                    self._proxy._last_entry = None
                del self._entries[key]
                cleared = True
            return cleared

    def busy_paths(self) -> list:
        """Model paths that currently have in-flight requests."""
        with self._lock:
            return sorted({str(e.model_path) for e in self._entries.values()
                           if e.busy > 0})

    def evict(self, model_path) -> bool:
        """Tear down every resident entry backing ``model_path`` (a GGUF abspath),
        across all of its load profiles. Returns whether anything was evicted;
        raises :class:`ModelBusyError` if the model has in-flight requests (an
        unload must never stop another client's running generation).

        Also drops any keep mark - an explicit unload is a full release, so the
        path won't be silently re-kept (and TTL-exempted) on a later reload."""
        with self._build_lock, self._lock:
            self._keep_paths.discard(model_path)
            keys = [k for k, e in self._entries.items()
                    if e.model_path == model_path]
            in_flight = sum(self._entries[k].busy for k in keys)
            if in_flight:
                raise ModelBusyError(model_path, in_flight)
            for k in keys:
                entry = self._entries[k]
                self._teardown(entry)
                if self._proxy._last_entry is entry:
                    self._proxy._last_entry = None
                del self._entries[k]
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
            except Exception:
                pass  # best-effort sweep; count only the successful closes
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
                        "pinned": e.pinned,
                        "kept": e.model_path in self._keep_paths,
                        "seq": e.seq,
                        "busy": e.busy,
                        "footprint_bytes": e.footprint,
                        "ttl_s": e.ttl,
                        "idle_s": max(0.0, now - e.last_access),
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
        with self._build_lock, self._lock:
            for entry in list(self._entries.values()):
                if (entry.pinned or entry.model_path in self._keep_paths
                        or not entry.ttl or entry.ttl <= 0):
                    continue
                if now - entry.last_access < entry.ttl:
                    continue
                _log.info("idle TTL unload (%.0fs): %s",
                          entry.ttl, entry.model_path)
                self._teardown(entry)
                del self._entries[entry.cache_key]
                if self._proxy._last_entry is entry:
                    self._proxy._last_entry = None
                reaped.append(entry.model_path)
        return reaped

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
        self._proxy._last_entry = entry

    def _resident_bytes(self) -> int:
        return sum(e.footprint for e in self._entries.values())

    def _over_capacity(self, incoming: int) -> bool:
        """True if admitting ``incoming`` more weight bytes would exceed the
        byte budget or the optional count cap."""
        if self._budget and self._resident_bytes() + incoming > self._budget:
            return True
        if self._max is not None and len(self._entries) >= self._max:
            return True
        return False

    def _evict_for_room(self, incoming: int):
        # Evict LRU-unpinned models until the incoming one fits the budget (and
        # the optional count cap). Never evict a pinned or busy (in-flight
        # generation) model; if only those remain, exceed the budget rather than
        # refuse the request or kill a live generation.
        while self._entries and self._over_capacity(incoming):
            victim = min(
                (e for e in self._entries.values() if not e.pinned and e.busy == 0),
                key=lambda e: e.seq,
                default=None,
            )
            if victim is None:
                _log.warning(
                    "all %d resident models pinned or busy (%.1f GB); incoming "
                    "%.1f GB will exceed the %.1f GB budget",
                    len(self._entries), self._resident_bytes() / 1e9,
                    incoming / 1e9, self._budget / 1e9,
                )
                return
            _log.info(
                "evicting LRU model (%.1f GB) to fit %.1f GB: %s",
                victim.footprint / 1e9, incoming / 1e9, victim.model_path,
            )
            self._teardown(victim)
            del self._entries[victim.cache_key]

    def _build(self, cache_key, model_path, adapter_path, model_kind, footprint,
               *, ttl=None, env=None, build_spec=None) -> _Entry:
        from . import server_bridge_vlm as _serving

        _warn_if_batch_unsafe(model_path)
        scratch = _Scratch()
        token = _build_scratch.set(scratch)
        # Per-model load-param + APC/SSD-KV env window: set this model's vars
        # around the stock load, then restore. Builds are serialized by
        # ``_build_lock``, so this os.environ mutation never races a sibling load.
        saved_env = self._apply_env(env)
        # Publish the resolved spec for this build so the load bridge can read its
        # profile overrides (e.g. chat_template). ``_stock_get`` blocks here while
        # the engine's generation worker thread runs the bridge, where the
        # request-thread ``_active_spec`` ContextVar is invisible - this module
        # global crosses that boundary. Safe because the build lock serialises
        # builds (one spec live at a time); cleared in the finally.
        _serving.set_build_spec(build_spec)
        try:
            self._stock_get(model_path, adapter_path, model_kind=model_kind)
        finally:
            _serving.set_build_spec(None)
            self._restore_env(saved_env)
            _build_scratch.reset(token)
        return _Entry(
            cache_key=cache_key,
            model_path=model_path,
            model_cache=scratch.model_cache,
            response_generator=scratch.response_generator,
            apc_manager=scratch.apc_manager,
            pinned=model_path in self._pinned_paths,
            footprint=footprint,
            ttl=ttl,
            last_access=self._time_fn(),
        )

    @staticmethod
    def _apply_env(env):
        """Set ``env`` in ``os.environ``, returning the prior values for restore."""
        if not env:
            return None
        saved = {}
        for k, v in env.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = str(v)
        return saved

    @staticmethod
    def _restore_env(saved):
        if not saved:
            return
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old

    def _teardown(self, entry: _Entry):
        from . import server_bridge_vlm as _serving

        # Drop any stashed in-memory MTP drafter for this path, or its weights
        # stay referenced after the model is gone.
        _serving.drop_mtp_stash(entry.model_path)
        # Capture the expert prefetcher before the stock unload empties
        # model_cache; it holds one open fd per GGUF shard and nothing else
        # closes it (repeated load/unload cycles would creep toward EMFILE).
        _prefetcher = getattr(entry.model_cache.get("model"),
                              "_kq_prefetcher", None)
        # Same for the prefill feeder: shard fds + staging pools + ~GBs of
        # host ring slots that must not outlive the model.
        _feeder = getattr(entry.model_cache.get("model"), "_kq_feeder", None)
        # And the decode feeder: its mlocked wired arena (a large fraction of
        # RAM on hybrid over-RAM MoE models), read pool, and shard fds sit in
        # a feeder<->module reference cycle, so refcounting alone won't
        # reclaim them before the next model sizes its own arena.
        _decode_feeder = getattr(entry.model_cache.get("model"),
                                 "_kq_decode_feeder", None)
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
        for owner in (entry.apc_manager, _prefetcher, _feeder, _decode_feeder):
            close = getattr(owner, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception:
                pass
        # A larger-than-RAM model leaves a page-cache remnant that taxes
        # whoever faults next (gmlx.pagecache). Process exit sweeps it for
        # CLI runs; a long-lived server sweeps at eviction, before the next
        # model loads against the stale cache. No-op for in-RAM models.
        try:
            from .pagecache import release_streaming_for
            release_streaming_for(entry.model_path)
        except Exception:
            pass


def _pinned_from_env(preload_path):
    pinned = set()
    if preload_path:
        pinned.add(preload_path)
    extra = os.environ.get("MLX_VLM_PINNED_MODELS", "")
    pinned.update(p.strip() for p in extra.split(",") if p.strip())
    return pinned


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

    from . import server_bridge_vlm as _serving
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

    package = importlib.import_module("mlx_vlm.server")
    app = importlib.import_module("mlx_vlm.server.app")
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

    original_runtime = sys.modules["mlx_vlm.server.runtime"].runtime
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

    from . import config as _config
    from . import server_bridge_vlm as _serving

    def pooled_get_cached_model(model_path, adapter_path=inherit, *, model_kind="auto"):
        # In config mode the incoming ``model_path`` is a friendly id (maybe
        # ``id@profile``): resolve it to a concrete GGUF abspath + merged spec,
        # bind the spec for this request (read at the gen-args seam), fold the
        # load_signature into the cache_key, and carry the load/APC env into the
        # build window. Without a registered config (single-model launch) the path
        # is used verbatim - today's behaviour, unchanged.
        ttl = None
        cache_key_extra = ()
        env = None
        build_spec = None
        load_path = model_path
        if _serving.server_config() is not None:
            try:
                # The body `profile` field rides a request-scoped ContextVar
                # (install_request_profile_capture) - this seam only receives
                # the model string. Inline `id@profile` still wins inside.
                load_path, spec = _serving.resolve_request_model(
                    model_path,
                    profile_field=_serving.get_request_profile())
            except (_serving.ModelNotFound, _serving.ModelFileMissing,
                    _serving.UnknownProfile, _serving.NoModelSpecified) as e:
                raise _http_from_resolver_error(e) from e
            _serving.set_active_spec(spec)
            ttl = spec.ttl_s
            cache_key_extra = spec.load_signature()
            env = _config.env_for(spec)
            build_spec = spec          # crosses into the load worker thread (see _build)
        if adapter_path is inherit:
            adapter_path = pool.resident_adapter(load_path)
        try:
            entry = pool.acquire(load_path, adapter_path, model_kind,
                                 ttl=ttl, cache_key_extra=cache_key_extra, env=env,
                                 build_spec=build_spec)
        except FileNotFoundError as e:
            # Absolute-path entries skip the resolver's existence check by
            # contract ("you said exactly where it is"), so a deleted file
            # first surfaces here, at load. Give it the same typed 404 the
            # relative/hf shapes get instead of a raw 500 errno.
            if _serving.server_config() is not None:
                miss = _serving.ModelFileMissing(str(model_path), str(e))
                raise _http_from_resolver_error(miss) from e
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
