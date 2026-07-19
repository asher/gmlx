"""Shared scaffolding for the server's aux sub-services (embeddings, rerank,
STT, TTS): the single-worker executor that owns a service's Metal stream, the
4xx request-error shape the routes map to JSON errors, the GGUF model-holder
cache, and the wizard preset lookup. Each service module keeps its own forward
pass and request/response shape; only the plumbing lives here."""

from __future__ import annotations

import concurrent.futures
import os
import sys
import threading


class SingleWorker:
    """Lazily-created single-thread executor owning all of one sub-service's
    model work (the load and every forward). MLX streams are per-thread:
    running a forward from Starlette's threadpool on a model loaded elsewhere
    raises "There is no Stream(gpu, N) in current thread" and aborts the
    process. ``max_workers=1`` also serializes the service's Metal jobs (one
    at a time, still interleaving with LLM batch decode). Created lazily so
    importing the service module stays side-effect free."""

    def __init__(self, thread_name_prefix: str):
        self._prefix = thread_name_prefix
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def get(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix=self._prefix)
        return self._executor

    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        return self.get().submit(fn, *args, **kwargs)


class SubserviceRequestError(Exception):
    """A client-side problem with a sub-service request (HTTP 4xx)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class GGUFModelHolder:
    """Process-wide single-model cache for a GGUF-backed sub-service: loads via
    the runtime's own :func:`gmlx.loader.load_model` (an mlx-lm ``Model``
    with K-quant leaves) and keeps it **separate** from the chat residency pool -
    a RAG re-index must not evict the chat model, nor the reverse. Reload only
    on a path change. Subclass per service: the class attributes hold the state,
    so each subclass caches independently."""

    model = None
    tokenizer = None
    model_path = None

    @classmethod
    def get(cls, model_path: str):
        if cls.model is None or model_path != cls.model_path:
            from .loader import load_model
            model, _config, tokenizer = load_model(model_path, verbose=False)
            cls.model, cls.tokenizer = model, tokenizer
            cls.model_path = model_path
        return cls.model, cls.tokenizer


def preset_for(alias, presets):
    """The preset dict whose ``alias`` matches (trimmed, lowercased), or None."""
    a = (alias or "").strip().lower()
    for p in presets:
        if p["alias"] == a:
            return p
    return None


def resolve_alias_or_path(value, *, aliases: dict, default_alias: str,
                          default_names: tuple) -> str:
    """Normalize a configured sub-service model value: ``True`` or a default
    name -> the default alias's repo, a known alias -> its repo, an existing
    local directory -> its abspath, anything else -> returned as-is (an HF
    repo id, or a path that fails loudly at load)."""
    if value is True:
        return aliases[default_alias]
    v = str(value).strip()
    low = v.lower()
    if low in default_names:
        return aliases[default_alias]
    if low in aliases:
        return aliases[low]
    expanded = os.path.expanduser(v)
    if os.path.isdir(expanded):
        return os.path.abspath(expanded)
    return v


def effective_model(requested: str, configured: str, *, accepted_names,
                    resolver, error_cls, kind: str, hint: str) -> str:
    """Map a request's ``model`` field onto the service's configured model.

    The conventional OpenAI names in ``accepted_names`` and anything
    ``resolver`` maps to the configured repo are accepted; anything else is
    rejected with a 400 - clients must not be able to make the server pull
    arbitrary HF repos."""
    from .config import ConfigError

    req = (requested or "").strip()
    if req.lower() in accepted_names or req == configured:
        return configured
    try:
        resolved = resolver(req)
    except ConfigError:
        # A name that doesn't resolve locally is still just "not this
        # server's model" to the client - a 400, never a 500.
        resolved = None
    if resolved == configured:
        return configured
    raise error_cls(
        400,
        f"model {req!r} is not the {kind} model this server is configured "
        f"with ({configured!r}); send model='{hint}' (or omit it)")


def prewarm(worker: SingleWorker, loader, label: str, *,
            missing_hint: str | None = None) -> concurrent.futures.Future:
    """Background-load a sub-service's configured model at server startup.

    Moves the cold HF download + load off the first request onto the service's
    dedicated worker thread - the kernels' stream affinity requires the thread
    that loads to be the thread that later serves, and the single worker makes
    a request that races the warm wait on this load instead of starting a
    second one. Best-effort: failures are logged and the model just loads
    lazily on first request. Returns the submitted Future (callers may ignore
    it; tests wait on it)."""
    def _run():
        try:
            loader()
        except FileNotFoundError as exc:
            # A retry can't fix a missing file - don't promise one.
            print(f"[server] {label} model is missing on disk ({exc})"
                  + (f"; {missing_hint}" if missing_hint else ""),
                  file=sys.stderr)
        except Exception as exc:          # noqa: BLE001 - best-effort warm
            print(f"[server] {label} prewarm failed ({exc}); "
                  f"will load on first request", file=sys.stderr)

    return worker.submit(_run)


def run_on_worker(worker: SingleWorker, job, *, error_cls, what: str):
    """Run ``job`` on the service's worker and unwrap the result: 4xx request
    errors and ``FileNotFoundError`` (the endpoint's 404) pass through;
    anything else surfaces as the route's 500."""
    try:
        return worker.submit(job).result()
    except error_cls:
        raise
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{what} failed: {exc}") from exc
