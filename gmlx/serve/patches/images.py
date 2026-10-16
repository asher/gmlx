"""The OpenAI image routes: generations, edits and variations, served by
the image models of the residency pool.

``install_image_routes`` replaces mlx-vlm's image handlers, which load a
model and read reference images by the paths a request names, whether or
not the config sets up an image model. Each handler parses its body on the
event loop, acquires the image model off it, and runs the job on the image
worker. A disconnect sets the job's stop event, which the sampler reads
between steps. ``response_format: url`` answers with a link to
``GET /v1/images/files/{token}.{ext}``, which the API key does not guard:
an ``img`` tag sends no header, so the 32 byte token is the capability.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import statistics
import threading
import time

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import gmlx.serve.bridge_vlm as serving

from ._common import _error_content, _remove_routes
from .api_contract import (
    IMAGES_EDITS_CONSUMED,
    IMAGES_GENERATIONS_CONSUMED,
    IMAGES_VARIATIONS_CONSUMED,
    warn_ignored_fields,
)

_log = logging.getLogger(__name__)

ROUTES = ("generations", "edits", "variations")
STOCK_PATHS = ("/v1/images/generations", "/images/generations",
               "/v1/images/edits", "/images/edits",
               "/v1/images/variations", "/images/variations")
FILE_PREFIXES = ("/v1/images/files/", "/images/files/")
_DISCONNECT_POLL_S = 0.25
_CONSUMED = {"generations": IMAGES_GENERATIONS_CONSUMED,
             "edits": IMAGES_EDITS_CONSUMED,
             "variations": IMAGES_VARIATIONS_CONSUMED}


class _State:
    """What the installed routes read: the image settings, the model
    folders, the URL file store and the count of image requests in
    flight."""

    def __init__(self):
        from gmlx.config import ImageServerCfg

        self.settings = ImageServerCfg()
        self.model_dirs: tuple = ()
        self.store = None
        self.lock = threading.Lock()
        self.in_flight = 0
        self.durations: list = []           # seconds of recent jobs
        self.counts = {"jobs": 0, "failed": 0, "cancelled": 0, "images": 0,
                       "steps": 0}
        self.recent: list = []              # (image_s, step_s) per image

    def configure(self, cfg) -> None:
        from gmlx.config import ImageServerCfg

        from ..image_service import (
            FileStore,
            configure_caches,
            default_store_root,
        )

        self.settings = getattr(cfg, "image", None) or ImageServerCfg()
        self.model_dirs = tuple(getattr(cfg, "model_dirs", ()) or ())
        configure_caches(self.settings)
        # A reload keeps the folder the server started with, so its links
        # stay good.
        root = (self.store.root if self.store is not None
                else default_store_root(getattr(cfg, "port", None)))
        self.store = FileStore(root, self.settings.url_ttl_s,
                               self.settings.url_max_bytes)

    def admit(self) -> bool:
        """Take a slot, or refuse when ``max_queue`` requests wait behind
        the one running."""
        with self.lock:
            if self.in_flight > self.settings.max_queue:
                return False
            self.in_flight += 1
            return True

    def leave(self, seconds: float | None = None) -> None:
        with self.lock:
            self.in_flight = max(0, self.in_flight - 1)
            if seconds is not None:
                self.durations = (self.durations + [seconds])[-16:]

    def record(self, status: str, result=None) -> None:
        """Count a finished job; an ``ok`` one adds its seconds per image
        and per step."""
        with self.lock:
            if status == "ok":
                self.counts["jobs"] += 1
                if result is not None and result.images:
                    n = len(result.images)
                    self.counts["images"] += n
                    self.counts["steps"] += n * result.run_steps
                    per = (result.image_s / n,
                           result.step_s / (n * result.run_steps)
                           if result.run_steps else None)
                    self.recent = (self.recent + [per])[-16:]
            elif status == "cancelled":
                self.counts["cancelled"] += 1
            elif status in ("failed", "timeout"):
                self.counts["failed"] += 1

    def metrics(self) -> dict:
        """The ``images`` section of ``/v1/metrics``."""
        with self.lock:
            img = [a for a, _ in self.recent]
            step = [b for _, b in self.recent if b is not None]
            return {
                "jobs_total": self.counts["jobs"],
                "jobs_failed_total": self.counts["failed"],
                "jobs_cancelled_total": self.counts["cancelled"],
                "images_total": self.counts["images"],
                "steps_total": self.counts["steps"],
                "in_flight": self.in_flight,
                "waiting": max(0, self.in_flight - 1),
                "max_queue": self.settings.max_queue,
                "image_s": round(statistics.median(img), 3) if img else None,
                "step_s": round(statistics.median(step), 4) if step else None,
            }

    def retry_after(self) -> int:
        with self.lock:
            typical = statistics.median(self.durations) if self.durations else 10.0
            waiting = max(1, self.in_flight)
        return int(min(600, max(2, round(typical * waiting))))


STATE = _State()


def _app():
    return importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app


def _error(path: str, status: int, message: str, *,
           err_type: str = "invalid_request_error", param=None, code=None,
           headers=None) -> JSONResponse:
    extra = {"param": param, "code": code}
    return JSONResponse(status_code=status, headers=headers,
                        content=_error_content(path, status, err_type,
                                               message, **extra))


def _from_request_error(path: str, e) -> JSONResponse:
    return _error(path, e.status, str(e), err_type=e.err_type, param=e.param,
                  code=e.code, headers=e.headers or None)


def _from_http_exception(path: str, e: HTTPException) -> JSONResponse:
    detail = e.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        err = dict(detail["error"])
        return JSONResponse(status_code=e.status_code,
                            headers=getattr(e, "headers", None),
                            content={"error": err})
    return _error(path, e.status_code, str(detail),
                  err_type="server_error" if e.status_code >= 500
                  else "invalid_request_error",
                  headers=getattr(e, "headers", None))


async def _watch_disconnect(request: Request, stop: threading.Event):
    """Set ``stop`` when the client goes away. The body is read by now, so
    the next message the request receives is its disconnect.
    ``Request.is_disconnected`` never sees it behind the server's http
    middlewares, whose receive yields under its cancelled scope."""
    try:
        while not stop.is_set():
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                stop.set()
                return
            await asyncio.sleep(_DISCONNECT_POLL_S)
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 - a receive the server no longer serves
        _log.debug("image request disconnect watch ended", exc_info=True)


def _json_image(value, param: str):
    """A JSON edit image: ``{"image_url": ...}`` or a bare data URL. A
    ``file_id`` is refused, and so is a path or a URL the media gate does
    not accept."""
    from ..image_service import _bad, decode_image
    from .media_gate import MediaRefused, _decode_edit_image

    if isinstance(value, dict):
        if value.get("file_id") is not None:
            raise _bad("file_id is not supported; send the image as a data "
                       "URL in image_url, or as a multipart file", param)
        value = value.get("image_url")
        if isinstance(value, dict):
            value = value.get("url")
    if not isinstance(value, str) or not value:
        raise _bad(f"{param} must be an image_url holding a data URL", param)
    # A JSON body carries its images inline: a data URL or base64. The
    # server reads no path and fetches no URL from it.
    if value.startswith("data:"):
        try:
            data, _suffix, _px = _decode_edit_image(value)
        except MediaRefused as e:
            raise _bad(str(e), param) from None
    else:
        import base64
        import binascii

        try:
            data = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise _bad(f"{param} must be a data URL or base64 image data; "
                       "send a file as multipart form data", param) from None
    return decode_image(data, param)


async def _read_body(route: str, request: Request):
    """``(fields, images, mask)`` of a JSON body or a multipart form."""
    from ..image_service import _bad, decode_image

    ctype = request.headers.get("content-type", "").lower()
    images, mask = [], None
    if ctype.startswith("multipart/form-data"):
        if route == "generations":
            raise _bad("this route takes a JSON body, not a multipart form")
        try:
            form = await request.form(max_files=64)
        except Exception as e:  # noqa: BLE001 - any parse failure is a 400
            raise _bad(f"could not parse multipart form data: {e}") from None
        fields = {}
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                data = await value.read()
                if key in ("image", "image[]"):
                    images.append(decode_image(data, "image"))
                elif key == "mask":
                    mask = decode_image(data, "mask")
                else:
                    raise _bad(f"file field {key!r} is not read by this route",
                               key)
                fields[key] = "<file>"
            else:
                fields[key] = value
        return fields, images, mask
    try:
        body = json.loads(await request.body() or b"{}")
    except ValueError as e:
        raise _bad(f"could not parse JSON body: {e}") from None
    if not isinstance(body, dict):
        raise _bad("request body must be a JSON object")
    if route != "generations":
        raw = body.get("images", body.get("image"))
        for item in raw if isinstance(raw, list) else ([] if raw is None else [raw]):
            images.append(_json_image(item, "images"))
        if body.get("mask") is not None:
            mask = _json_image(body["mask"], "mask")
    return body, images, mask


def _budget(hold):
    """The bytes a request may plan on: the working budget less the other
    resident models, or None with GMLX_OVERCOMMIT set."""
    from ..capacity import overcommit, working_budget_bytes

    if overcommit():
        return None
    total = working_budget_bytes()
    if total is None:
        return None
    pool = hold._pool
    others = pool.stats()["resident_bytes"] - hold._entry.footprint
    return max(0, total - others)


def _text_model(model):
    """The text model an engine serves: a text-only GGUF is served inside
    a VLM-shaped wrapper, and the encoder reads the model it wraps."""
    if getattr(model, "_is_text_model", False):
        return model.language_model._model
    return model


def _borrow_for(spec):
    """``(borrow, release)`` for an image model that shares the encoder of
    a chat model: ``borrow(fn)`` runs ``fn`` with the chat model as the
    pipeline's encoder, on that model's engine thread. Runs in the request
    context, since the pooled load binds the engine to it."""
    from gmlx.serve.engine_jobs import run_on_engine

    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    runtime = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
    app_mod.get_cached_model(spec.encoder_model)
    guard = runtime.response_generator
    rg = getattr(guard, "_rg", guard)
    hold = getattr(guard, "_hold", None)
    if rg is None:
        if hold is not None:
            hold.release()
        raise HTTPException(status_code=500, detail=(
            "the shared encoder's engine is unavailable; restart the server"))

    def borrow(served_pipe, fn, stop, timeout_s):
        from gmlx.image import api
        from gmlx.image.text_encoder import encoder_spec

        def job(engine_rg, should_stop):
            # A chat model's engine runs a job as steps between its decode
            # steps; the encode is one step.
            served_pipe.encoder = api._borrowed(
                (_text_model(engine_rg.model), engine_rg.tokenizer),
                encoder_spec(served_pipe.family))
            try:
                return fn()
            finally:
                served_pipe.encoder = None

        def steps(engine_rg, should_stop):
            return job(engine_rg, should_stop)
            yield  # noqa: B901 - a generator of one step

        from ..engine_jobs import JobCancelled, JobTimeout
        from ..image_service import JobStopped

        # A shared encoder that times out or is cancelled ends the job as
        # the sampler's own stop does: 504 and 499.
        try:
            return run_on_engine(rg, steps, request_id=f"img-{id(fn):x}",
                                 prompt_tokens=1, stop=stop,
                                 timeout_s=timeout_s, stepwise=True)
        except JobTimeout:
            raise JobStopped(timed_out=True) from None
        except JobCancelled:
            raise JobStopped(timed_out=False) from None

    return borrow, hold


def _response_body(req, result, urls) -> dict:
    data = []
    for i, blob in enumerate(result.images):
        if urls is not None:
            data.append({"url": urls[i]})
        else:
            import base64

            data.append({"b64_json": base64.b64encode(blob).decode("ascii")})
    return {"created": int(time.time()), "data": data,
            "background": ("transparent" if req.background == "transparent"
                           else "opaque"),
            "output_format": req.output_format,
            "quality": req.quality,
            "size": f"{result.size[0]}x{result.size[1]}",
            "usage": result.usage}


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def make_endpoint(route: str):
    from .. import residency
    from ..image_service import (
        WORKER,
        ImageRequestError,
        JobStopped,
        check_family,
        fit_check,
        parse_request,
        run_job,
        target_size,
    )
    from .routes import _log_aux_request

    path = f"/v1/images/{route}"
    event_prefix = "image_edit" if route == "edits" else "image_generation"

    async def endpoint(request: Request):
        started = time.monotonic()
        try:
            fields, images, mask = await _read_body(route, request)
            req = parse_request(route, fields)
            req.images, req.mask = images, mask
        except ImageRequestError as e:
            _log_aux_request(path, None, started, status="refused", error=str(e))
            return _from_request_error(path, e)
        ignored = (set(fields) - _CONSUMED[route]) | req.ignored
        warn_ignored_fields(path, ignored)
        if not serving.image_models():
            _log_aux_request(path, req.model or None, started, status="refused")
            return _error(path, 404, "no image model is configured on this "
                          "server; add one under image_models in the config",
                          err_type="model_not_found", code="model_not_found")
        if not STATE.admit():
            wait = STATE.retry_after()
            _log_aux_request(path, req.model or None, started, status="busy")
            return _error(path, 429, f"{STATE.settings.max_queue} image "
                          "requests are waiting; retry later",
                          err_type="server_overloaded",
                          headers={"Retry-After": str(wait)})
        stop = threading.Event()
        holds: list = []
        gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
        timeout_s = gen.get_token_queue_timeout()

        def prepare():
            hold = residency.acquire_image_model(req.model)
            holds.append(hold)
            served = hold._entry.pipeline
            check_family(req, served)
            size = target_size(req, served)
            budget = _budget(hold)
            fit_check(req, served, size, budget)
            borrow = None
            if served.spec.encoder_model is not None:
                fn, chat_hold = _borrow_for(served.spec)
                if chat_hold is not None:
                    holds.append(chat_hold)

                def borrow(run, _fn=fn, _pipe=served.pipe):
                    return _fn(_pipe, run, stop, timeout_s)
            return served, size, budget, borrow

        def release():
            for h in holds:
                try:
                    h.release()
                except Exception:  # noqa: BLE001 - a release must not mask the answer
                    _log.debug("image hold release failed", exc_info=True)
            holds.clear()

        watcher = asyncio.create_task(_watch_disconnect(request, stop))
        try:
            served, size, budget, borrow = await asyncio.to_thread(prepare)
        except ImageRequestError as e:
            stop.set()
            watcher.cancel()
            release()
            STATE.leave()
            _log_aux_request(path, req.model or None, started,
                             status="refused", error=str(e))
            return _from_request_error(path, e)
        except HTTPException as e:
            stop.set()
            watcher.cancel()
            release()
            STATE.leave()
            _log_aux_request(path, req.model or None, started,
                             status="refused", status_code=e.status_code)
            return _from_http_exception(path, e)
        except Exception as e:  # noqa: BLE001 - a 500 with the detail in the log
            stop.set()
            watcher.cancel()
            release()
            STATE.leave()
            _log.exception("%s: model load failed", path)
            _log_aux_request(path, req.model or None, started,
                             status="failed", error=type(e).__name__)
            return _error(path, 500, f"{path} failed ({type(e).__name__}); "
                          "see the server log", err_type="server_error")
        model_id = served.spec.id
        submitted = time.monotonic()
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()

        def emit(index: int, blob: bytes):
            loop.call_soon_threadsafe(events.put_nowait, ("partial", index, blob))

        def job():
            from gmlx.image.pacing import paced

            from ..gpu_share import pacer

            # Chat decode keeps its turns on the GPU while the job runs.
            with paced(pacer()):
                return run_job(served, req, size=size, budget=budget,
                               stop=stop, timeout_s=timeout_s,
                               submitted=submitted, borrow=borrow,
                               emit=emit if req.stream else None,
                               model_dirs=STATE.model_dirs,
                               embed_metadata=STATE.settings.embed_metadata)

        future = WORKER.submit(job)

        finished: list = []

        def finish(status: str, result=None, **extra):
            # Once per request, from the route or from the job's end.
            if finished:
                return
            finished.append(status)
            release()
            STATE.leave(time.monotonic() - submitted if status == "ok" else None)
            STATE.record(status, result)
            _log_aux_request(path, model_id, started, status=status,
                             n=req.n, size=f"{size[0]}x{size[1]}", **extra)

        def store_urls(result):
            if req.response_format != "url":
                return None
            base = str(request.base_url).rstrip("/")
            return [f"{base}/v1/images/files/"
                    f"{STATE.store.put(blob, req.output_format)}."
                    f"{req.output_format}" for blob in result.images]

        if not req.stream:
            try:
                result = await asyncio.wrap_future(future)
            except JobStopped as e:
                finish("timeout" if e.timed_out else "cancelled")
                if e.timed_out:
                    return _error(path, 504, "the image job timed out",
                                  err_type="timeout")
                return Response(status_code=499)
            except ImageRequestError as e:
                finish("refused", error=str(e))
                return _from_request_error(path, e)
            except Exception as e:  # noqa: BLE001 - mapped below
                return _job_failure(path, e, finish)
            finally:
                stop.set()
                watcher.cancel()
            urls = store_urls(result)
            finish("ok", result, steps=result.steps, fmt=req.output_format)
            return JSONResponse(content=_response_body(req, result, urls))

        common = {"background": req.background, "output_format": req.output_format,
                  "quality": req.quality, "size": f"{size[0]}x{size[1]}"}

        def settle(f):
            # The end of a streamed job frees its hold and its queue slot,
            # whether or not the client still reads the stream, and even
            # when the stream never started.
            try:
                loop.call_soon_threadsafe(watcher.cancel)
            except RuntimeError:
                pass
            stop.set()
            e = None if f.cancelled() else f.exception()
            if f.cancelled():
                finish("cancelled")
            elif e is None:
                result = f.result()
                finish("ok", result, steps=result.steps,
                       fmt=req.output_format, stream=1)
            elif isinstance(e, JobStopped):
                finish("timeout" if e.timed_out else "cancelled")
            elif isinstance(e, ImageRequestError):
                finish("refused", error=str(e))
            elif (refusal := _job_refusal(e)) is not None:
                finish("refused", error=refusal[2])
            else:
                _log.error("%s: image job failed", path, exc_info=e)
                finish("failed", error=type(e).__name__)

        future.add_done_callback(settle)

        async def stream():
            import base64

            done = asyncio.ensure_future(asyncio.wrap_future(future))
            getter = None
            try:
                while True:
                    getter = asyncio.ensure_future(events.get())
                    finished_, _ = await asyncio.wait(
                        {getter, done}, return_when=asyncio.FIRST_COMPLETED)
                    if getter in finished_:
                        _kind, index, blob = getter.result()
                        yield _sse(f"{event_prefix}.partial_image", {
                            "type": f"{event_prefix}.partial_image",
                            "b64_json": base64.b64encode(blob).decode("ascii"),
                            "created_at": int(time.time()),
                            "partial_image_index": index, **common})
                        continue
                    getter.cancel()
                    while not events.empty():
                        _kind, index, blob = events.get_nowait()
                        yield _sse(f"{event_prefix}.partial_image", {
                            "type": f"{event_prefix}.partial_image",
                            "b64_json": base64.b64encode(blob).decode("ascii"),
                            "created_at": int(time.time()),
                            "partial_image_index": index, **common})
                    break
                try:
                    result = done.result()
                except JobStopped as e:
                    if e.timed_out:
                        yield _sse("error", {"error": {
                            "type": "timeout", "message": "the image job timed out"}})
                    return
                except Exception as e:  # noqa: BLE001 - an error frame, then close
                    yield _sse("error", _stream_error(path, e))
                    return
                body = {"type": f"{event_prefix}.completed",
                        "created_at": int(time.time()), "usage": result.usage,
                        **common}
                urls = store_urls(result)
                if urls is not None:
                    body["url"] = urls[0]
                body["b64_json"] = base64.b64encode(result.images[0]).decode("ascii")
                yield _sse(f"{event_prefix}.completed", body)
            finally:
                # The client may have left: the job stops at its next step,
                # and settle frees the slot when it does.
                stop.set()
                watcher.cancel()
                if getter is not None:
                    getter.cancel()
                if done.done():
                    _retrieved(done)
                else:
                    done.add_done_callback(_retrieved)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return endpoint


def _retrieved(f) -> None:
    """Retrieve the outcome of a job future no one awaits any more."""
    if not f.cancelled():
        f.exception()


def _job_refusal(e: Exception) -> tuple | None:
    """``(message, param, logged error)`` of an error a job raised that
    refuses the request, which both paths answer as a 400, or None for a
    job that failed. A refusal is one of the types gmlx raises for a bad
    request: a plan that does not fit, a primitive's RequestRefusal, or an
    adapter the request names that does not load. Any other error, a
    ValueError among them, is the server's."""
    from gmlx.image.lora.reader import AdapterError
    from gmlx.image.memfit import ImageMemoryError
    from gmlx.image.refusal import RequestRefusal

    if isinstance(e, ImageMemoryError):
        return str(e), "size", "ImageMemoryError"
    if isinstance(e, (RequestRefusal, AdapterError)):
        return str(e), None, str(e)
    return None


def _job_failure(path: str, e: Exception, finish) -> JSONResponse:
    refusal = _job_refusal(e)
    if refusal is not None:
        message, param, logged = refusal
        finish("refused", error=logged)
        return _error(path, 400, message, param=param)
    finish("failed", error=type(e).__name__)
    _log.error("%s: image job failed", path, exc_info=e)
    return _error(path, 500, f"{path} failed ({type(e).__name__}); see the "
                  "server log", err_type="server_error")


def _stream_error(path: str, e: Exception) -> dict:
    """The error frame of a streamed job that raised ``e``: the error the
    non-streamed path answers with."""
    from ..image_service import ImageRequestError

    if isinstance(e, ImageRequestError):
        return _error_content(path, e.status, e.err_type, str(e),
                              param=e.param, code=e.code)
    refusal = _job_refusal(e)
    if refusal is not None:
        message, param, _ = refusal
        return _error_content(path, 400, "invalid_request_error", message,
                              param=param, code=None)
    return _error_content(path, 500, "server_error",
                          f"{path} failed ({type(e).__name__}); see the "
                          "server log", param=None, code=None)


async def file_endpoint(name: str):
    from ..image_service import mime_type

    token, _, ext = name.rpartition(".")
    store = STATE.store
    p = None if store is None else store.path(token, ext)
    if p is None:
        return _error("/v1/images/files", 404, "no such image, or it expired",
                      err_type="not_found")
    return FileResponse(p, media_type=mime_type(ext),
                        headers={"Cache-Control": "private, max-age=3600"})


def is_file_path(path: str, method: str) -> bool:
    """Whether a request is a read of one URL-response image, the one route
    the API key does not guard: a GET of a token and an extension, and
    nothing beside it."""
    import re

    if method != "GET":
        return False
    return re.fullmatch(r"/(?:v1/)?images/files/[A-Za-z0-9_-]{43}\.(?:png|jpeg|webp)",
                        path) is not None


def install_image_routes(cfg) -> None:
    """Replace mlx-vlm's image routes with gmlx's, at both prefixes, with
    or without an image model in ``cfg``, and serve the URL-response
    files. A reload calls it again with the new config."""
    STATE.configure(cfg)
    if STATE.store is not None:
        STATE.store.sweep()
    app = _app()
    _remove_routes(app, *STOCK_PATHS)
    _remove_routes(app, "/v1/images/files/{name}", "/images/files/{name}")
    for route in ROUTES:
        endpoint = make_endpoint(route)
        for p in (f"/v1/images/{route}", f"/images/{route}"):
            app.add_api_route(p, endpoint, methods=["POST"],
                              include_in_schema=False)
    for prefix in FILE_PREFIXES:
        app.add_api_route(prefix + "{name}", file_endpoint, methods=["GET"],
                          include_in_schema=False)
