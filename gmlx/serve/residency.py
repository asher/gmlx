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
import logging
import os
import threading
import time
from contextvars import ContextVar

from .capacity import LoadDeferred
from gmlx.envflags import env_float

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
    text_adapter_bytes,
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
from .residency_workers import (  # noqa: F401 - re-exported
    _holds_worker,
    _Workers,
)
from .residency_teardown import (  # noqa: F401 - re-exported
    _Teardown,
)
from .residency_lora import (  # noqa: F401 - re-exported
    _Aside,
    _default_lora_shed,
    _default_store_live,
    _default_store_shed,
    _LoraTurn,
)
from .residency_pool import (  # noqa: F401 - re-exported
    _default_lora_figures,
    _PoolCore,
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


class _ResidencyPool(_PoolCore, _LoraTurn, _Workers, _Teardown, _ChatBuild):
    """Pinned + LRU pool of resident models, backed by stock load/teardown."""


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

    from gmlx.load.decision import DecisionModelOnly

    from . import bridge_vlm as _serving
    if isinstance(exc, DecisionModelOnly):
        return HTTPException(status_code=400, detail={"error": {
            "type": "decision_model_only", "message": str(exc)}})
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


def _refuse_decision_model(load_path, spec) -> None:
    """A 400 for a decision model outside the systemone route and the
    startup preload, read from its header before any load. Every route
    that generates loads through here, so a new one is refused unless it
    opens a decision scope."""
    from gmlx.load.decision import DecisionModelOnly, in_decision_scope
    from gmlx.load.discovery import header_meta

    if in_decision_scope() or not isinstance(load_path, str):
        return
    if not (header_meta(load_path) or {}).get("decision"):
        return
    exc = DecisionModelOnly(getattr(spec, "id", None) or "this model")
    raise _http_from_resolver_error(exc) from exc


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
        _refuse_decision_model(load_path, spec)
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
