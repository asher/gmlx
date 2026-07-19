"""Config-driven HTTP routes: /v1/models override, HF-download gate,
runtime-snapshot enrichment, pool-aware /unload, the audio/embeddings/rerank
subservice routes, /v1/reload, keep/preload warming, and resolver error
bodies."""

from __future__ import annotations

import importlib
import os
import time

from fastapi import Request  # module-level so stringized annotations resolve

from .. import server_bridge_vlm as serving
from ._common import (
    _PATCH_FLAG,
    _get_pool,
    _remove_routes,
)


def _mtime(path) -> int:
    try:
        return int(os.path.getmtime(path))
    except Exception:
        return 0


def _models_payload() -> dict:
    """The ``/v1/models`` body: every configured/discovered id, plus alias entries
    (pickable presets, marked ``alias_of``), with resident / pinned / capability /
    ``default`` markers. Never the HF cache."""
    serving.reregister_missing_models()   # a restored file re-appears here
    models = serving.resolved_models()
    resident, pinned = set(), set()
    pool = _get_pool()
    if pool is not None:
        for e in pool.stats()["resident"]:
            resident.add(e["model_path"])
            if e["pinned"]:
                pinned.add(e["model_path"])
    default_id = serving.default_model_id()

    def _entry(mid, rm, *, alias_of=None, profile=None):
        return {
            "id": mid,
            "object": "model",
            "created": _mtime(rm.path),
            "owned_by": "gmlx",
            "resident": rm.path in resident,
            "pinned": bool(rm.pin) or rm.path in pinned,
            "speculative": bool(rm.speculative),
            "vlm": rm.mmproj is not None,
            "profile": profile if alias_of else rm.profile_name,
            "family": getattr(rm, "family", None),   # sampling family (profiles.py)
            "default": mid == default_id,            # aliases never == default_id
            **({"alias_of": alias_of} if alias_of else {}),
        }

    data = [_entry(mid, rm) for mid, rm in models.items()]
    # Alias entries point at a real model's bytes (so they inherit its residency /
    # capability flags) but carry the alias's own baked profile.
    for name, (target_id, profile) in serving.aliases().items():
        rm = models.get(target_id)
        if rm is not None:
            data.append(_entry(name, rm, alias_of=target_id,
                               profile=profile or rm.profile_name))
    return {"object": "list", "data": data}


def _service_display(value) -> str:
    """A service's ``alias_of`` for ``/v1/models`` - portable refs (aliases,
    repo ids, ``hf:``) pass through; filesystem paths shrink to their basename
    so the listing never leaks local directory layout."""
    v = str(value)
    if v.startswith("hf:"):
        return v
    if v.startswith(("/", "~")) or (os.sep in v and v.endswith(".gguf")):
        return os.path.basename(os.path.expanduser(v).rstrip(os.sep)) or v
    return v


def _service_entry(sid: str, marker: str, value) -> dict:
    """A service advertisement shaped like a chat entry (same keys, so naive
    consumers can index any of them) plus its capability marker."""
    return {
        "id": sid, "object": "model", "created": 0, "owned_by": "gmlx",
        "resident": False, "pinned": False, "speculative": False, "vlm": False,
        "profile": None, "family": None, "default": False,
        marker: True, "alias_of": _service_display(value),
    }


def _service_file_on_disk(value, model_dirs) -> bool:
    """False only when ``value`` names a GGUF whose file is verifiably absent
    right now - the /v1/models advertisement for a service must degrade with
    the disk the way chat entries do. Repo-id services have nothing to stat."""
    from .. import embeddings as _emb
    if not _emb._is_gguf_ref(value):
        return True
    try:
        from ..config import resolve_path
        p = resolve_path(value, list(model_dirs))
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def install_models_endpoint_override(stt_model: str | None = None,
                                     tts_model: str | None = None,
                                     embeddings_model: str | None = None,
                                     rerank_model: str | None = None,
                                     model_dirs=()
                                     ) -> None:
    """Replace ``/v1/models`` + ``/models`` so they list configured ids, not the
    HF cache. Plain-dict return (no response_model) so the markers survive.
    When STT/TTS/embeddings/rerank is configured, a ``whisper-1`` / ``tts-1`` /
    ``text-embedding-3-small`` / ``reranker`` entry advertises it (the names
    OpenAI/Cohere clients probe for), shaped like a chat entry plus a ``stt`` /
    ``tts`` / ``embeddings`` / ``rerank`` marker; ``alias_of`` names the
    configured model without leaking local paths. A GGUF-backed service whose
    file is missing on disk is de-listed (checked per request; it re-appears
    the moment the file is back, like a chat entry)."""
    app = importlib.import_module("mlx_vlm.server.app").app

    async def models_endpoint():
        payload = _models_payload()
        if stt_model:
            payload["data"].append(_service_entry("whisper-1", "stt", stt_model))
        if tts_model:
            payload["data"].append(_service_entry("tts-1", "tts", tts_model))
        if embeddings_model and _service_file_on_disk(embeddings_model,
                                                      model_dirs):
            payload["data"].append(_service_entry(
                "text-embedding-3-small", "embeddings", embeddings_model))
        if rerank_model and _service_file_on_disk(rerank_model, model_dirs):
            payload["data"].append(_service_entry(
                "reranker", "rerank", rerank_model))
        return payload

    _remove_routes(app, "/v1/models", "/models")
    app.add_api_route("/v1/models", models_endpoint, methods=["GET"],
                      include_in_schema=False)
    app.add_api_route("/models", models_endpoint, methods=["GET"],
                      include_in_schema=False)


def install_auto_docs_removal() -> None:
    """Drop FastAPI's auto-docs routes (``/openapi.json``, ``/docs``,
    ``/redoc``) - the OpenAI-compatible API itself is untouched. The generated
    schema is mlx-vlm's, not this server's: after the route surgery here it
    describes endpoints that don't exist and misses the ones that do. A 404 is
    honest; the real surface is documented in docs/server-config.md."""
    app = importlib.import_module("mlx_vlm.server.app").app
    _remove_routes(app, "/openapi.json", "/docs", "/docs/oauth2-redirect",
                   "/redoc")


# HF-download gate
class HFAccessDisabled(RuntimeError):
    """A request would resolve a non-local, non-GGUF id from HF, but HF access is
    off. Enable ``server.hf_cache`` to resolve named hf ids from the local cache."""


def _gate_model_path(path_or_hf_repo, hf_cache: bool, original, *args, **kwargs):
    p = str(path_or_hf_repo)
    if p.endswith(".gguf") or os.path.exists(os.path.expanduser(p)):
        return original(path_or_hf_repo, *args, **kwargs)   # local / GGUF: allow
    if not hf_cache:
        raise HFAccessDisabled(
            f"HF access is disabled and {p!r} is not a local GGUF. Configure it "
            f"in models:, or set server.hf_cache to resolve named hf ids from the "
            f"local cache (never the network).")
    return original(path_or_hf_repo, *args, **kwargs)        # offline-env enforced


def install_hf_download_gate(hf_cache: bool) -> None:
    """Wrap ``utils.get_model_path`` to refuse a network download. With ``hf_cache``
    on, force local-cache-only resolution (``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``)
    so a named hf id resolves from the cache but never the network. Idempotent."""
    utils = importlib.import_module("mlx_vlm.utils")
    if hf_cache:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if getattr(utils.get_model_path, _PATCH_FLAG, False):
        return
    original = utils.get_model_path

    def get_model_path(path_or_hf_repo, *args, **kwargs):
        return _gate_model_path(path_or_hf_repo, hf_cache, original, *args, **kwargs)

    get_model_path.__dict__[_PATCH_FLAG] = True
    utils.get_model_path = get_model_path


# Runtime-snapshot enrichment (lifts /health + /v1/metrics)
def _resident_models_view() -> list:
    """A per-entry view of the residency pool for the runtime snapshot - id(s),
    pinned, idle/ttl, footprint. Empty when no pool is installed."""
    pool = _get_pool()
    if pool is None:
        return []
    path_to_ids = getattr(serving, "_PATH_TO_IDS", {})
    out = []
    for e in pool.stats()["resident"]:
        out.append({
            "ids": path_to_ids.get(e["model_path"], []),
            "model_path": e["model_path"],
            "pinned": e["pinned"],
            "kept": e.get("kept", False),
            "busy": e.get("busy", 0),
            "footprint_bytes": e["footprint_bytes"],
            "idle_s": round(e.get("idle_s", 0.0), 1),
            "ttl_s": e.get("ttl_s"),
        })
    return out


def install_runtime_snapshot_enrichment() -> None:
    """Add ``resident_models[]`` to ``_server_runtime_snapshot`` so both ``/health``
    and ``/v1/metrics`` report the full pool (the handlers call it by name).
    Idempotent."""
    app = importlib.import_module("mlx_vlm.server.app")
    if getattr(app._server_runtime_snapshot, _PATCH_FLAG, False):
        return
    original = app._server_runtime_snapshot

    def snapshot():
        base = original()
        base["resident_models"] = _resident_models_view()
        return base

    snapshot.__dict__[_PATCH_FLAG] = True
    app._server_runtime_snapshot = snapshot


# Pool-aware /unload + /v1/reload
def install_pool_aware_unload() -> None:
    """Replace ``/unload`` so an optional ``{"model": "<id>"}`` evicts just that
    resident model; no body clears the whole pool (stock behaviour).

    ``Request`` must be imported at module level: with ``from __future__ import
    annotations`` the ``request: Request`` annotation is a string FastAPI resolves
    against ``unload_endpoint.__globals__``. A local import wouldn't be in globals,
    so FastAPI would miss the special ``Request`` type and treat ``request`` as a
    required query param - every POST then 422s before the body is ever read."""
    from fastapi.responses import JSONResponse

    app = importlib.import_module("mlx_vlm.server.app").app

    async def unload_endpoint(request: Request):
        pool = _get_pool()
        try:
            body = await request.json()
        except Exception:
            body = None
        model_id = (body or {}).get("model") if isinstance(body, dict) else None

        if model_id:
            if pool is None:
                # 503, not 200: without a pool the unload cannot be honored.
                return JSONResponse(status_code=503, content={
                    "status": "error", "message": "no residency pool"})
            try:
                path, _spec = serving.resolve_request_model(model_id)
            except (KeyError, serving.ModelFileMissing):
                # 404, matching /v1/keep: a typo'd unload must not read as
                # success (clients check the status code).
                return JSONResponse(status_code=404, content={
                    "status": "unknown_model", "model": model_id})
            from ..residency import ModelBusyError
            try:
                evicted = pool.evict(path)
            except ModelBusyError as e:
                return JSONResponse(status_code=409, content={
                    "status": "busy", "model": model_id,
                    "in_flight": e.in_flight})
            return {"status": "success" if evicted else "not_resident",
                    "unloaded": model_id if evicted else None}

        cleared = pool.clear() if pool is not None else False
        busy = pool.busy_paths() if pool is not None else []
        if busy:
            return {"status": "success" if cleared else "busy",
                    "skipped_busy": [os.path.basename(p) for p in busy]}
        return {"status": "success" if cleared else "no_model_loaded"}

    _remove_routes(app, "/unload")
    app.add_api_route("/unload", unload_endpoint, methods=["POST"],
                      include_in_schema=False)


def _log_aux_request(endpoint: str, model: str | None, started: float,
                     *, status: str = "ok", **fields) -> None:
    """Emit one ``[req]`` line for an auxiliary (non-generation) route.

    The audio (``/v1/audio/speech``, ``/v1/audio/transcriptions``) and
    ``/v1/embeddings`` routes are served directly here, bypassing the
    ``ServerMetricsStore`` funnel that :func:`install_request_timing_log` hooks -
    so they never produce the ``[req]`` line every chat/completions request does.
    They also sit in ``_TIMED_PATHS``, so :class:`_AccessNoiseFilter` suppresses
    their uvicorn access line on the assumption a timing line replaces it. Without
    this helper they are doubly invisible: no access line and no ``[req]`` line.
    Best-effort - a logging hiccup must never disturb the response."""
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.monotonic() - started
        extra = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        mark = "" if status == "ok" else f" {status}"
        tail = f"{extra} total={elapsed:.2f}s" if extra else f"total={elapsed:.2f}s"
        print(f"[req] {ts} {endpoint} {model or '?'}{mark} {tail}", flush=True)
    except Exception:
        pass


def _aux_error(status: int, message: str, *, err_type: str = "invalid_request_error"):
    """OpenAI-shaped error body, shared by the aux routes below."""
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status,
                        content={"error": {"message": message,
                                           "type": err_type}})


async def _aux_json_body(request: Request):
    """Parse a JSON-object request body. Returns ``(body, None)`` on success or
    ``(None, error_response)`` for the caller to return as-is."""
    try:
        body = await request.json()
    except Exception as exc:
        return None, _aux_error(400, f"could not parse JSON body: {exc}")
    if not isinstance(body, dict):
        return None, _aux_error(400, "request body must be a JSON object")
    return body, None


def _aux_server_error(endpoint: str, model: str, started: float, exc: Exception):
    """Log + 500 for an unexpected aux-route failure. The client body stays
    generic - raw exception text can carry local paths and internals; the
    detail goes to the server log."""
    _log_aux_request(endpoint, model, started,
                     status="failed", error=type(exc).__name__, detail=exc)
    return _aux_error(
        500, f"{endpoint} failed ({type(exc).__name__}); see the server log",
        err_type="server_error")


def _aux_model_file_missing(endpoint: str, config_key: str, model: str,
                            started: float, exc: Exception):
    """Log + 404 when a configured aux model's file vanished from disk. The
    resolver detail (searched directories) stays in the server log."""
    _log_aux_request(endpoint, model, started,
                     status="failed", error=type(exc).__name__, detail=exc)
    return _aux_error(
        404,
        f"the configured `{config_key}` model is missing on disk; "
        f"restore the file, update the config, or run `gmlx sync-models`",
        err_type="model_file_missing")


def _install_audio_task_route(stt_model: str | None, *, path: str, task: str,
                              accept_language: bool) -> None:
    """Shared installer for the two Whisper multipart routes. ``task`` is
    "transcribe" (``/v1/audio/transcriptions``) or "translate" (any-language audio
    -> English, ``/v1/audio/translations``); both share the mlx-whisper model
    (``stt`` extra), so they differ only in the Whisper task and whether the
    OpenAI endpoint takes a ``language`` field (transcription does, translation
    doesn't). ``stt_model`` None => no route (404).

    The multipart parse + upload read stay on the event loop; the Whisper Metal
    work runs in Starlette's threadpool under ``stt._TRANSCRIBE_LOCK``, so requests
    serialize against each other but interleave freely with LLM batch decode."""
    if not stt_model:
        return
    from fastapi.responses import JSONResponse, PlainTextResponse
    from starlette.concurrency import run_in_threadpool

    from .. import stt

    app = importlib.import_module("mlx_vlm.server.app").app

    async def audio_endpoint(request: Request):
        started = time.monotonic()
        try:
            form = await request.form()
        except Exception as exc:
            return _aux_error(400, f"could not parse multipart form data: {exc} "
                                   "(the stt extra includes python-multipart - "
                                   "pip install 'gmlx[stt]')")
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return _aux_error(400, "multipart field 'file' is required")
        audio_bytes = await upload.read()
        try:
            content, media_type = await run_in_threadpool(
                stt.run_transcription, audio_bytes,
                filename=getattr(upload, "filename", "") or "",
                configured_model=stt_model,
                model=str(form.get("model") or ""),
                language=(str(form.get("language") or "") if accept_language
                          else ""),
                prompt=str(form.get("prompt") or ""),
                temperature=str(form.get("temperature") or ""),
                response_format=str(form.get("response_format") or ""),
                task=task,
            )
        except stt.STTRequestError as exc:
            return _aux_error(exc.status_code, str(exc))
        except Exception as exc:
            return _aux_server_error(path, stt_model, started, exc)
        _log_aux_request(path, stt_model, started,
                         file=(getattr(upload, "filename", "") or None),
                         in_bytes=len(audio_bytes),
                         fmt=(str(form.get("response_format") or "") or None))
        if media_type == "application/json":
            return JSONResponse(content=content)
        return PlainTextResponse(content=content, media_type=media_type)

    _remove_routes(app, path)
    app.add_api_route(path, audio_endpoint, methods=["POST"],
                      include_in_schema=False)


def install_audio_transcription_route(stt_model: str | None) -> None:
    """Add OpenAI-compatible ``POST /v1/audio/transcriptions`` (same-language
    speech-to-text) backed by the optional mlx-whisper (``stt`` extra)."""
    _install_audio_task_route(stt_model, path="/v1/audio/transcriptions",
                              task="transcribe", accept_language=True)


def install_audio_translation_route(stt_model: str | None) -> None:
    """Add OpenAI-compatible ``POST /v1/audio/translations`` (any-language audio ->
    English) backed by the same mlx-whisper model via Whisper's ``translate`` task.
    The OpenAI translations endpoint takes no ``language`` field."""
    _install_audio_task_route(stt_model, path="/v1/audio/translations",
                              task="translate", accept_language=False)


def install_audio_speech_route(tts_model: str | None) -> None:
    """Add OpenAI-compatible ``POST /v1/audio/speech`` backed by the optional
    mlx-audio (``tts`` extra). ``tts_model`` is the resolved model (repo id or
    local dir) from ``ServerCfg.tts``; None => no route (404).

    The JSON parse stays on the event loop; ``run_synthesis`` runs in
    Starlette's threadpool but hands the model's Metal work to a dedicated
    single TTS worker thread (mlx-audio binds a model to its loading thread),
    so requests serialize against each other but interleave with LLM decode."""
    if not tts_model:
        # mlx-vlm >= 0.6.4 registers its own speech routes (per-request model
        # loading) at app build; drop them so the cfg.tts contract holds --
        # clients feature-detect TTS by probing this route for 404.
        app = importlib.import_module("mlx_vlm.server.app").app
        _remove_routes(app, "/v1/audio/speech", "/audio/speech")
        return
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool

    from .. import tts

    app = importlib.import_module("mlx_vlm.server.app").app

    async def speech_endpoint(request: Request):
        started = time.monotonic()
        body, err = await _aux_json_body(request)
        if err is not None:
            return err
        try:
            content, media_type = await run_in_threadpool(
                tts.run_synthesis, str(body.get("input") or ""),
                configured_model=tts_model,
                model=str(body.get("model") or ""),
                voice=str(body.get("voice") or ""),
                response_format=str(body.get("response_format") or ""),
                speed=str(body.get("speed") if body.get("speed") is not None
                          else ""),
            )
        except tts.TTSRequestError as exc:
            return _aux_error(exc.status_code, str(exc))
        except Exception as exc:
            return _aux_server_error("/v1/audio/speech", tts_model, started, exc)
        _log_aux_request("/v1/audio/speech", tts_model, started,
                         voice=(str(body.get("voice") or "") or None),
                         chars=len(str(body.get("input") or "")),
                         bytes=len(content))
        return Response(content=content, media_type=media_type)

    _remove_routes(app, "/v1/audio/speech", "/audio/speech")
    app.add_api_route("/v1/audio/speech", speech_endpoint,
                      methods=["POST"], include_in_schema=False)


def install_audio_voices_route(tts_model: str | None) -> None:
    """Add ``GET /v1/audio/voices`` listing the configured TTS model's voice
    names (Kokoro's presets / qwen3-tts speakers) so clients can offer a
    picker - the OpenAI API has no voice-listing endpoint to mirror. None =>
    no route (404), which clients treat as "no listing"."""
    if not tts_model:
        return
    from starlette.concurrency import run_in_threadpool

    from .. import tts

    app = importlib.import_module("mlx_vlm.server.app").app

    async def voices_endpoint():
        voices = await run_in_threadpool(tts.available_voices, tts_model)
        return {"model": tts_model, "voices": voices,
                "default": tts.DEFAULT_VOICE if voices else None}

    _remove_routes(app, "/v1/audio/voices")
    app.add_api_route("/v1/audio/voices", voices_endpoint,
                      methods=["GET"], include_in_schema=False)


def install_embeddings_route(embeddings_model: str | None) -> None:
    """Add OpenAI-compatible ``POST /v1/embeddings`` backed by the optional
    mlx-embeddings (``embeddings`` extra). ``embeddings_model`` is the resolved
    model (repo id or local dir) from ``ServerCfg.embeddings``; None => no route
    (404). This is what ``gmlx launch open-webui`` points RAG at.

    The JSON parse stays on the event loop; the embedding pass (the model's Metal
    work) dispatches from Starlette's threadpool to the service's single worker
    thread (``subservice.SingleWorker``), so requests serialize against each
    other but interleave freely with LLM batch decode."""
    if not embeddings_model:
        return
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    from .. import embeddings as embeddings_mod

    app = importlib.import_module("mlx_vlm.server.app").app

    async def embeddings_endpoint(request: Request):
        started = time.monotonic()
        body, err = await _aux_json_body(request)
        if err is not None:
            return err
        try:
            payload = await run_in_threadpool(
                embeddings_mod.run_embeddings, body.get("input"),
                configured_model=embeddings_model,
                model=str(body.get("model") or ""),
                encoding_format=str(body.get("encoding_format") or ""),
            )
        except embeddings_mod.EmbeddingsRequestError as exc:
            return _aux_error(exc.status_code, str(exc))
        except FileNotFoundError as exc:
            return _aux_model_file_missing("/v1/embeddings", "server.embeddings",
                                           embeddings_model, started, exc)
        except Exception as exc:
            return _aux_server_error("/v1/embeddings", embeddings_model,
                                     started, exc)
        data = payload.get("data") if isinstance(payload, dict) else None
        n_inputs = len(data) if isinstance(data, list) else None
        dims = (len(data[0].get("embedding", []))
                if n_inputs and isinstance(data[0], dict) else None)
        tokens = (payload.get("usage", {}) or {}).get("prompt_tokens") \
            if isinstance(payload, dict) else None
        _log_aux_request("/v1/embeddings", embeddings_model, started,
                         inputs=n_inputs, dims=dims, tokens=tokens)
        return JSONResponse(content=payload)

    _remove_routes(app, "/v1/embeddings")
    app.add_api_route("/v1/embeddings", embeddings_endpoint,
                      methods=["POST"], include_in_schema=False)


def install_rerank_route(rerank_model: str | None) -> None:
    """Add the Cohere/Jina-shaped ``POST /v1/rerank`` (+ ``/rerank``) backed by a
    Qwen3-Reranker GGUF. ``rerank_model`` is the resolved local GGUF from
    ``ServerCfg.rerank``; None => no route (404). This is what Open WebUI's external
    reranker points at.

    The JSON parse stays on the event loop; the scoring (one model forward per
    document) dispatches from Starlette's threadpool to the service's single
    worker thread (``subservice.SingleWorker``), so requests serialize against
    each other but interleave with LLM batch decode."""
    if not rerank_model:
        return
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    from .. import rerank as rerank_mod

    app = importlib.import_module("mlx_vlm.server.app").app

    async def rerank_endpoint(request: Request):
        started = time.monotonic()
        body, err = await _aux_json_body(request)
        if err is not None:
            return err
        try:
            payload = await run_in_threadpool(
                rerank_mod.run_rerank, body.get("query"), body.get("documents"),
                configured_model=rerank_model,
                model=str(body.get("model") or ""),
                top_n=body.get("top_n", body.get("top_k")),
                instruction=body.get("instruction"),
                return_documents=bool(body.get("return_documents", True)),
            )
        except rerank_mod.RerankRequestError as exc:
            return _aux_error(exc.status_code, str(exc))
        except FileNotFoundError as exc:
            return _aux_model_file_missing("/v1/rerank", "server.rerank",
                                           rerank_model, started, exc)
        except Exception as exc:
            return _aux_server_error("/v1/rerank", rerank_model, started, exc)
        results = payload.get("results") if isinstance(payload, dict) else None
        tokens = (payload.get("usage", {}) or {}).get("total_tokens") \
            if isinstance(payload, dict) else None
        _log_aux_request("/v1/rerank", rerank_model, started,
                         docs=(len(results) if isinstance(results, list) else None),
                         tokens=tokens)
        return JSONResponse(content=payload)

    _remove_routes(app, "/v1/rerank", "/rerank")
    app.add_api_route("/v1/rerank", rerank_endpoint,
                      methods=["POST"], include_in_schema=False)
    app.add_api_route("/rerank", rerank_endpoint,
                      methods=["POST"], include_in_schema=False)


def install_reload_route(reload_fn) -> None:
    """Add ``POST /v1/reload`` running ``reload_fn()`` (re-read config + re-register
    models, keeping warm entries). A no-op 501 when ``reload_fn`` is None."""
    app = importlib.import_module("mlx_vlm.server.app").app

    from fastapi.responses import JSONResponse

    async def reload_endpoint():
        if reload_fn is None:
            return JSONResponse(status_code=501, content={
                "status": "unsupported",
                "message": "reload requires a --config launch"})
        try:
            result = reload_fn() or {}
        except Exception as exc:
            # An admin route: the message (usually a config-validation error)
            # is the point, but a failed reload must not answer 200.
            return JSONResponse(status_code=500, content={
                "status": "error", "message": str(exc)})
        return {"status": "success", **result}

    _remove_routes(app, "/v1/reload")
    app.add_api_route("/v1/reload", reload_endpoint, methods=["POST"],
                      include_in_schema=False)


# Sentinel: warm with the model's inherited (resident) adapter - the common path.
_WARM_DEFAULT = object()

# Preload holds retained for the process lifetime so a background-warmed preload
# stays eviction-proof, exactly as mlx-vlm's lifespan preload hold does (its hold
# lives as long as the lifespan context - here, as long as this module).
_PRELOAD_HOLDS: list = []


def _load_resident(model_id, adapter=_WARM_DEFAULT):
    """Load ``model_id`` resident through the pooled resolver and return the busy
    hold it took (or None). Must run off the event loop (a worker/daemon thread):
    the pooled ``get_cached_model`` does the multi-second build synchronously, and
    the whole point is to keep that off the loop. The caller decides the hold's
    fate - release it (a transient warm; the entry stays LRU/TTL-managed) or retain
    it (a preload kept resident for the server's lifetime)."""
    from .. import residency
    app = importlib.import_module("mlx_vlm.server.app")
    if adapter is _WARM_DEFAULT:
        app.get_cached_model(model_id)            # pooled: resolve -> acquire
    else:
        app.get_cached_model(model_id, adapter)
    return residency._active_hold.get()


def _warm_and_release(model_id, adapter=_WARM_DEFAULT) -> None:
    """Load ``model_id`` resident, then drop the in-flight hold so it stays
    LRU/TTL-managed. The transient warm used by the keep route and the chat
    load-offload (a later request re-acquires the now-resident entry)."""
    hold = _load_resident(model_id, adapter)
    if hold is not None:
        hold.release()


def _spawn_keep_warm(model_id: str):
    """Background-load ``model_id`` resident (so it is hot before the first request),
    then drop the in-flight hold so it stays LRU-eligible - best-effort. A load
    failure is swallowed: the keep mark stands, and the first real request loads the
    model or surfaces the error normally. The residency build lock serializes this
    against other loads. Returns the started daemon thread (tests join on it)."""
    import threading

    def _run():
        try:
            _warm_and_release(model_id)
        except Exception:
            pass

    thread = threading.Thread(target=_run, name="gmlx-keep-warm", daemon=True)
    thread.start()
    return thread


def _preload_extra_over_budget(model_id: str) -> bool:
    """True when an extra preload model's shard bytes exceed the streaming
    threshold (0.9x the wired budget): its page cache can't be populated and
    the streaming decode arena would fight the LRU pool for RAM, so only the
    primary preload may be a streaming model. False when undeterminable (the
    warm itself then degrades best-effort)."""
    try:
        from ..preflight import find_split_shards

        path = serving.resolved_models()[model_id].path
        total = sum(os.path.getsize(p) for p in find_split_shards(path))
        import mlx.core as mx

        budget = int(0.9 * mx.device_info()["max_recommended_working_set_size"])
        return total > budget
    except Exception:
        return False


def spawn_preload_warm(model_id: str | None, extras=()):
    """Background-load the preload set without blocking startup. The primary
    ``model_id`` keeps its busy hold for the process lifetime so it stays
    eviction-proof - the same effect mlx-vlm's lifespan preload has, but off
    the startup path so the port binds and ``/health`` answers immediately
    while the load runs. ``extras`` (``server.defaults.preload``) warm
    afterwards, sequentially, through ``_warm_and_release`` so they stay
    LRU/TTL-evictable (retained holds on a multi-model set would wedge the
    pool); over-budget streaming extras are skipped with a notice. Best-effort
    per model: a load failure is swallowed and that model loads lazily on
    first request. Returns the started daemon thread (tests join on it)."""
    import threading

    def _run():
        if model_id:
            try:
                hold = _load_resident(model_id)
                if hold is not None:
                    _PRELOAD_HOLDS.append(hold)       # retain -> eviction-proof
            except Exception:
                pass
        for mid in extras:
            if _preload_extra_over_budget(mid):
                print(f"[server] preload: skipping {mid} - over the wired "
                      "budget (streams from disk; loads on first request)")
                continue
            try:
                _warm_and_release(mid)
            except Exception:
                pass

    thread = threading.Thread(target=_run, name="gmlx-preload-warm", daemon=True)
    thread.start()
    return thread


def install_keep_route() -> None:
    """Add ``POST /v1/keep {model, warm, keep}``: keep a model resident through the
    idle-TTL reaper, but not against LRU - it stays evictable under memory pressure.
    A softer tier than config ``pin: true`` (which exempts from both). ``warm``
    (default true) background-loads it so it is hot before the first request.
    ``keep: false`` releases the hold without evicting (the model returns to normal
    LRU/TTL life - a voice session ending should not dump a hot model); ``/unload``
    both releases and evicts.

    Module-level ``Request`` import is required (see ``install_pool_aware_unload``)."""
    app = importlib.import_module("mlx_vlm.server.app").app

    async def keep_endpoint(request: Request):
        from fastapi.responses import JSONResponse

        pool = _get_pool()
        if pool is None:
            # 503, matching /unload: without a pool the keep cannot be honored.
            return JSONResponse(status_code=503, content={
                "status": "error", "message": "no residency pool"})
        try:
            body = await request.json()
        except Exception:
            body = None
        model_id = (body or {}).get("model") if isinstance(body, dict) else None
        if not model_id:
            return JSONResponse(status_code=400, content={
                "status": "error", "message": "missing 'model'"})
        try:
            path, _spec = serving.resolve_request_model(model_id)
        except (serving.ModelNotFound, serving.ModelFileMissing):
            # 404, not 200: a typo'd keep must not read as success (the
            # `launch` client checks the status code and this exact body shape).
            return JSONResponse(status_code=404, content={
                "status": "unknown_model", "model": model_id})
        except (serving.UnknownProfile, serving.NoModelSpecified) as exc:
            # A bad profile / ambiguous default is a client error (400), matching
            # install_resolver_error_handlers and the documented API - not a 404.
            return JSONResponse(status_code=400, content={
                "status": "error", "model": model_id, "message": str(exc)})
        if not bool((body or {}).get("keep", True)):
            pool.set_keep(path, False)
            return {"status": "released", "model": model_id}
        pool.set_keep(path, True)
        warm = bool((body or {}).get("warm", True))
        if warm:
            _spawn_keep_warm(model_id)
        return {"status": "kept", "model": model_id, "warming": warm}

    _remove_routes(app, "/v1/keep")
    app.add_api_route("/v1/keep", keep_endpoint, methods=["POST"],
                      include_in_schema=False)


# Helpful error bodies for the resolver exceptions
def install_resolver_error_handlers() -> None:
    """Map the friendly-id resolver errors to clean HTTP bodies: unknown id -> 404
    (listing available ids), configured id whose file is gone -> 404 (naming the
    sync-models fix), unknown profile / ambiguous default -> 400. Bodies are
    dialect-shaped per route (``_error_content``): OpenAI-style envelopes
    everywhere, Anthropic's ``{"type": "error", ...}`` on ``/v1/messages``.

    Also registers an ``HTTPException`` handler so an exception carrying the
    same ``{"error": {...}}`` detail (the residency resolver path raises these
    from inside mlx-vlm's endpoint wrappers) serves that body directly instead
    of FastAPI's ``{"detail": ...}`` wrapper, and a plain-string detail is
    wrapped into the standard envelope."""
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from ._common import _error_content

    app = importlib.import_module("mlx_vlm.server.app").app

    def _resolver_response(request, status, err_type, exc, **extra):
        return JSONResponse(status_code=status, content=_error_content(
            request.url.path, status, err_type, str(exc), **extra))

    @app.exception_handler(serving.ModelNotFound)
    async def _model_not_found(request, exc):
        return _resolver_response(request, 404, "model_not_found", exc,
                                  available_models=exc.available)

    @app.exception_handler(serving.ModelFileMissing)
    async def _model_file_missing(request, exc):
        return _resolver_response(request, 404, "model_file_missing", exc)

    @app.exception_handler(serving.UnknownProfile)
    async def _unknown_profile(request, exc):
        return _resolver_response(request, 400, "unknown_profile", exc,
                                  available_profiles=exc.available)

    @app.exception_handler(serving.NoModelSpecified)
    async def _no_model(request, exc):
        return _resolver_response(request, 400, "no_model_specified", exc,
                                  available_models=exc.available)

    @app.exception_handler(HFAccessDisabled)
    async def _hf_disabled(request, exc):
        return _resolver_response(request, 403, "hf_access_disabled", exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request, exc):
        detail = exc.detail
        headers = getattr(exc, "headers", None)
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            err = dict(detail["error"])
            err_type = err.pop("type", "invalid_request_error")
            message = err.pop("message", "")
            content = _error_content(request.url.path, exc.status_code,
                                     err_type, message, **err)
            return JSONResponse(status_code=exc.status_code, content=content,
                                headers=headers)
        if isinstance(detail, str):
            err_type = "server_error" if exc.status_code >= 500 \
                else "invalid_request_error"
            content = _error_content(request.url.path, exc.status_code,
                                     err_type, detail)
            return JSONResponse(status_code=exc.status_code, content=content,
                                headers=headers)
        return JSONResponse(status_code=exc.status_code,
                            content={"detail": detail}, headers=headers)
