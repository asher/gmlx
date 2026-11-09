"""The routes of the image step contract under ``/internal/image-steps/v1``.

A client whose sampler runs elsewhere, such as the ComfyUI node pack,
uploads its conditionings and adapter matrices once by hash and then asks
for one denoise step at a time. ``gmlx.serve.image_steps`` holds the state
and the work; this module parses the bodies, runs the work on the image
worker and answers.

The routes answer a loopback peer, and any peer the API key admits when
``server.image.steps_remote`` is set. Every request but ``capabilities``
names the contract version it speaks in ``X-Gmlx-Steps-Contract``, and a
major version this server does not speak is refused with the side to
upgrade. Tensors travel as safetensors containers
(``application/vnd.gmlx.tensors+safetensors``), and errors and the other
bodies as JSON.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import json
import logging
import time

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

import gmlx.serve.bridge_vlm as serving

from ..image_wire import MAX_BODY_BYTES
from ._common import _remove_routes

_log = logging.getLogger(__name__)


def _app():
    return importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app


class ServeHost:
    """Where the step routes find their models: the configured image and
    video models and the residency pool of the server."""

    log_requests = True

    def resolve(self, model):
        """``(path, spec)`` of an image or video model id, or an image
        model path."""
        if serving.is_video_model(model):
            return serving.resolve_video_request(model)
        return serving.resolve_image_request(model)

    def acquire(self, model):
        """A hold of the loaded model; ``hold._entry.pipeline`` is its
        ServedImage or ServedVideo, and ``hold.release()`` lets it go."""
        from .. import residency

        if serving.is_video_model(model):
            return residency.acquire_video_model(model)
        return residency.acquire_image_model(model)

    def revise(self, model, text, requested, has_refs, **clip):
        """The upsampler's caption of ``text``, or None to encode it as
        typed. ``clip`` holds a video clip's ``frames``, ``first`` and
        ``last``, when the encode names them, and ``stop``, the event that
        ends the rewrite."""
        from ..image_upsample import revise

        return revise(model, text, requested, has_refs, resolve=self.resolve,
                      **clip)

    def models(self) -> dict:
        out = serving.image_models()
        for vid in serving.video_model_ids():
            try:
                out[vid] = serving.resolve_video_request(vid)[1]
            except (serving.ModelNotFound, serving.ModelFileMissing):
                continue
        return out

    def default_model(self) -> str | None:
        """The id an image request with no model takes, or None."""
        return serving.default_image_model_id()

    def resident(self, path) -> bool:
        from ..residency import _installed_pool

        pool = _installed_pool()
        return pool is not None and pool.resident_entry(path) is not None

    def plan(self, spec):
        from ..image_service import entry_plan
        from ..image_steps import is_video

        if is_video(spec):
            from ..video_service import entry_plan as video_plan

            return video_plan(spec)
        return entry_plan(spec)

    def lora_pool(self):
        """The residency pool that books LoRA bytes, or None."""
        from ..residency import _installed_pool

        return _installed_pool()

    def budget(self, hold):
        """The bytes a step may plan on, as an image request may."""
        from .images import _budget

        return _budget(hold)

    def unload(self, path, spec):
        from ..residency import ModelBusyError, _installed_pool
        from .routes import _evict_ignoring_retained, _release_preload_holds

        pool = _installed_pool()
        if pool is None:
            return _err(503, "no residency pool")
        try:
            evicted = _evict_ignoring_retained(pool, path)
        except ModelBusyError as e:
            return _err(409, f"{spec.id!r} is busy", code="busy",
                        in_flight=e.in_flight)
        _release_preload_holds(pool, path, only_evicted=True)
        return _json(200, {"model": spec.id, "unloaded": bool(evicted)})


class _Settings:
    def __init__(self):
        from gmlx.config import ImageServerCfg

        self.settings = ImageServerCfg()
        self.model_dirs: tuple = ()
        self.host = ServeHost()


SETTINGS = _Settings()


def _json(status: int, body: dict, headers: dict | None = None):
    return JSONResponse(status_code=status, content=body, headers=headers)


def _err(status: int, message: str, **extra):
    return _json(status, {"error": message, **extra})


def _container(tensors: dict, fields: dict | None = None,
               headers: dict | None = None) -> Response:
    from ..image_wire import CONTENT_TYPE, write

    return Response(content=write(tensors, fields), media_type=CONTENT_TYPE,
                    headers=headers)


def _gate(request: Request, *, versioned: bool = True):
    """None for a request the routes serve, else the refusal: a peer that
    is not loopback without ``steps_remote``, and a contract version this
    server does not speak."""
    from ..image_wire import CONTRACT_HEADER, WireError, check_version
    from .hardening import _remote_peer

    if not request.client:
        return _err(403, "the peer address is unknown")
    peer = _remote_peer(request.client)
    if peer is not None and not SETTINGS.settings.steps_remote:
        return _err(403, f"the step routes answer loopback peers; {peer} is "
                    "not one. Set server.image.steps_remote to serve it")
    if versioned:
        try:
            check_version(request.headers.get(CONTRACT_HEADER))
        except WireError as e:
            return _err(e.status, str(e))
    return None


async def _fields(request: Request) -> dict:
    from ..image_steps import StepError

    raw = await request.body()
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        raise StepError(400, "the body is not JSON") from None
    if not isinstance(body, dict):
        raise StepError(400, "the body is a JSON object")
    return body


def _content_type(request: Request) -> None:
    from ..image_steps import StepError
    from ..image_wire import CONTENT_TYPE

    ctype = request.headers.get("content-type", "").split(";")[0].strip()
    if ctype != CONTENT_TYPE:
        raise StepError(415, f"the body is {CONTENT_TYPE}, not "
                        f"{ctype or 'untyped'}")


def _content_length(request: Request, max_bytes: int) -> None:
    """Refuse a body whose Content-Length is past ``max_bytes`` before a
    byte of it is read; the answer closes the connection."""
    from ..image_steps import StepError

    length = request.headers.get("content-length")
    if length is None:
        return
    try:
        n = int(length)
    except ValueError:
        raise StepError(400, "the Content-Length is not a number") from None
    if n > max_bytes:
        raise StepError(413, f"the body is {n} bytes, over the limit of "
                        f"{max_bytes}")


async def _wire(request: Request, max_bytes: int):
    """The tensors and fields of a safetensors body, read to ``max_bytes``:
    the media gate leaves it unread, so its Content-Length is held to the
    bound before the read, and a body with none is refused once it passes
    it."""
    from ..image_steps import StepError
    from ..image_wire import CONTENT_TYPE, WireError, read

    ctype = request.headers.get("content-type", "").split(";")[0].strip()
    if ctype != CONTENT_TYPE:
        raise StepError(415, f"the body is {CONTENT_TYPE}, not "
                        f"{ctype or 'untyped'}")
    _content_length(request, max_bytes)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > max_bytes:
            raise StepError(413, f"the body is over the limit of "
                            f"{max_bytes} bytes")
    try:
        return read(body, max_bytes=max_bytes)
    except WireError as e:
        raise StepError(e.status, str(e)) from None


def _spec(model):
    """``(path, spec)`` of an image model field, or the HTTP refusal."""
    from ..image_steps import StepError

    try:
        return SETTINGS.host.resolve(model)
    except Exception as e:  # noqa: BLE001 - every resolver error is a 404
        raise StepError(404, str(e) or type(e).__name__) from None


def _resident(path) -> bool:
    return SETTINGS.host.resident(path)


async def _acquire(model):
    return await asyncio.to_thread(SETTINGS.host.acquire, model)


def _http(e: HTTPException):
    detail = e.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return _err(e.status_code, str(detail["error"].get("message")))
    return _err(e.status_code, str(detail))


def _route(fn, *, versioned: bool = True, name: str = ""):
    """An endpoint around ``fn(request)`` with the gate, the refusals and
    one log line for a refused or failed request."""
    from ..image_steps import STATE, StepError, StepInterrupted
    from .routes import _log_aux_request as _aux

    path = f"/internal/image-steps/v1/{name}"

    def _log_aux_request(*args, **kw):
        if STATE.log_requests:
            _aux(*args, **kw)

    async def endpoint(request: Request):
        started = time.monotonic()
        refused = _gate(request, versioned=versioned)
        if refused is not None:
            _log_aux_request(path, None, started, status="refused")
            return refused
        try:
            return await fn(request)
        except StepError as e:
            if e.status != 409:
                _log_aux_request(path, None, started, status="refused",
                                 error=str(e))
            headers = None
            if e.status == 413:
                # The body past the limit is not read: end the connection.
                headers = {"Connection": "close"}
            if e.status == 503:
                retry = getattr(e, "retry_after", None)
                if retry is None:
                    from .request_flow import deferred_retry_after

                    retry = deferred_retry_after()
                headers = {"Retry-After": str(retry)}
            return _json(e.status, e.body(), headers)
        except StepInterrupted:
            return _err(409, "the step was interrupted", code="interrupted")
        except HTTPException as e:
            _log_aux_request(path, None, started, status="refused",
                             status_code=e.status_code)
            return _http(e)
        except ValueError as e:
            _log_aux_request(path, None, started, status="refused",
                             error=str(e))
            return _err(400, str(e))
        except Exception as e:  # noqa: BLE001 - a 500 with the detail in the log
            _log.exception("%s failed", path)
            _log_aux_request(path, None, started, status="failed",
                             error=type(e).__name__)
            return _err(500, f"{path} failed ({type(e).__name__}); see the "
                        "server log")

    return endpoint


async def _on_worker(fn, *args, on=None):
    """``fn`` on the worker of the entry that serves ``on``: the video
    worker for a video model, else the image worker."""
    from .. import image_service, video_service

    worker = video_service.WORKER if on is not None and _video(on) \
        else image_service.WORKER
    return await asyncio.wrap_future(worker.submit(fn, *args))


def _lora_pool():
    fn = getattr(SETTINGS.host, "lora_pool", None)
    return fn() if fn is not None else None


def _settled(hold, fn):
    """``fn`` as a worker job that sets the entry's LoRA ledger to its
    caches in its finally: an encode lets LoRA bytes go and books none."""
    pool = _lora_pool()
    if pool is None:
        return fn

    def job(*args):
        try:
            return fn(*args)
        finally:
            from ..image_service import lora_settle
            from ..image_steps import ENCODE

            lora_settle(pool, hold._entry, ENCODE)
    return job


def _refused(e):
    """The StepError of a LoraRefused, with its Retry-After."""
    from ..image_steps import StepError

    err = StepError(e.status, e.message)
    err.retry_after = e.retry_after
    err.transient = e.transient
    return err


def _served(hold):
    return hold._entry.pipeline


def _video(served) -> bool:
    from ..video_service import ServedVideo

    return isinstance(served, ServedVideo)


def _image_only(served, what: str) -> None:
    """Refuse a route of the image families on a video model."""
    from ..image_steps import StepError

    if _video(served):
        raise StepError(400, f"{served.spec.id!r} is a video model, and "
                        f"{what} takes an image model")


def _check_size(hold, width: int, height: int, *, what: str,
                cfg: float = 1.0, init: bool = False, phase: str):
    """Refuse a step or a VAE pass over the size limits of the image
    routes with 413, and one whose memory plan does not fit beside the
    models loaded with 400. ``phase`` is the phase of the plan the call
    runs, which alone counts the held conditioning states. Returns the
    budget it planned against."""
    from ..image_service import MAX_PIXELS, MAX_SIDE, size_fits
    from ..image_steps import StepError

    if width * height > MAX_PIXELS or max(width, height) > MAX_SIDE:
        raise StepError(413, f"{what} of {width}x{height} pixels is over the "
                        f"limit of {MAX_PIXELS} pixels (2048x2048) and a "
                        f"side of {MAX_SIDE}")
    budget_fn = getattr(SETTINGS.host, "budget", None)
    budget = budget_fn(hold) if budget_fn is not None else None
    if not size_fits(_served(hold), width, height, budget, cfg=cfg,
                     init=init, held_in=(phase,)):
        raise StepError(400, f"{what} of {width}x{height} pixels does not "
                        "fit in memory beside the models now loaded")
    return budget


def _check_refs(served, rows) -> None:
    """Refuse a row with more reference latents than the family takes."""
    from ..image_service import MAX_REFS
    from ..image_steps import StepError

    fam = served.pipe.family
    if fam.edit is None:
        return
    cap = min(fam.edit.max_refs, MAX_REFS)
    for i, r in enumerate(rows):
        n = len(r.get("ref_latents") or [])
        if n > cap:
            raise StepError(400, f"{fam.title} takes at most {cap} reference "
                            f"latents in a row, and row {i} has {n}")


def _px(served, part: str, h: int, w: int) -> tuple:
    """(width, height) in pixels of latents (h, w) in the model's latent
    space (``part`` "latent") or the VAE's (``part`` "vae")."""
    from ..image_steps import COMFY

    stride = int(COMFY[served.pipe.family.name][part]["stride"])
    return int(w) * stride, int(h) * stride


# Endpoints.

async def capabilities(request: Request):
    from ..image_steps import capabilities as caps

    return _json(200, caps())


async def models(request: Request):
    from ..image_steps import family_of

    out = []
    for mid, spec in SETTINGS.host.models().items():
        try:
            fam, variant = family_of(spec)
        except Exception:  # noqa: BLE001 - a file that went bad is listed bare
            out.append({"model": mid, "family": None})
            continue
        out.append({"model": mid, "family": fam.name, "variant": variant,
                    "resident": _resident(spec.path)})
    default = getattr(SETTINGS.host, "default_model", None)
    return _json(200, {"models": out,
                       "default_model": default() if default else None})


async def status(request: Request):
    from ..image_steps import status as fam_status

    path, spec = _spec(request.path_params["model"])
    resident = _resident(path)
    plan = await asyncio.to_thread(SETTINGS.host.plan, spec)
    return _json(200, fam_status(spec, resident, plan))


async def manifest(request: Request):
    from ..image_steps import manifest as fam_manifest

    _, spec = _spec(request.path_params["model"])
    return _json(200, await asyncio.to_thread(fam_manifest, spec))


async def load(request: Request):
    body = await _fields(request)
    _spec(body.get("model"))
    hold = await _acquire(body.get("model"))
    model = _served(hold).spec.id
    hold.release()
    return _json(200, {"model": model, "resident": True})


async def unload(request: Request):
    body = await _fields(request)
    path, spec = _spec(body.get("model"))
    # An unload waits on the build lock and the image worker: off the
    # event loop.
    return await asyncio.to_thread(SETTINGS.host.unload, path, spec)


def _hash_path(request: Request) -> str:
    from ..image_steps import StepError

    h = request.path_params["digest"]
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise StepError(400, "a hash is 64 lowercase hex digits")
    return h


async def put_tensor(request: Request):
    from ..image_steps import MAX_TENSOR_BYTES, STATE

    h = _hash_path(request)
    tensors, _ = await _wire(request, MAX_BODY_BYTES)
    STATE.put_tensor(STATE.tensors, h, tensors, MAX_TENSOR_BYTES)
    return _json(200, {"hash": h})


async def head_tensor(request: Request):
    from ..image_steps import STATE

    h = _hash_path(request)
    return Response(status_code=200 if h in STATE.tensors else 404)


def _admit(pool, n: int) -> None:
    """Admit ``n`` bytes of an upload into the store before they are read,
    off the event loop: at once when the pool's room holds them beside the
    store, else at a store turn on the image worker, which sheds and
    evicts for them. The put holds the store to its cap. The caller's
    ``pool.upload_done`` ends the count. Raises the StepError of a
    refusal."""
    from ..image_service import store_turn
    from ..image_steps import UPLOAD
    from ..residency import STORE, LoraRefused

    _hopeless(pool, STORE, n, UPLOAD)
    if pool.upload_admit(n, pool._measured_lora_room()):
        return
    try:
        store_turn(pool, n, UPLOAD)
    except LoraRefused as e:
        raise _refused(e) from None


async def _admitted_wire(request: Request, pool):
    """``(tensors, admitted)`` of an upload body read once the store
    admitted its bytes: its Content-Length, or BODY_STEP bytes at a time
    as a body with none is read. The media gate leaves the body unread, so
    the admission comes before its first byte. A refusal ends the read; a
    413 closes the connection. The bytes admitted count in the pool until
    the caller's ``pool.upload_done``."""
    from ..image_steps import BODY_STEP, StepError
    from ..image_wire import CONTENT_TYPE, WireError, read

    ctype = request.headers.get("content-type", "").split(";")[0].strip()
    if ctype != CONTENT_TYPE:
        raise StepError(415, f"the body is {CONTENT_TYPE}, not "
                        f"{ctype or 'untyped'}")
    length = request.headers.get("content-length")
    n = None
    if length is not None:
        try:
            n = int(length)
        except ValueError:
            raise StepError(400, "the Content-Length is not a number") \
                from None
        if n > MAX_BODY_BYTES:
            raise StepError(413, f"the body is {n} bytes, over the limit of "
                            f"{MAX_BODY_BYTES}")
    admitted = BODY_STEP if n is None else n
    await asyncio.to_thread(_admit, pool, admitted)
    try:
        if n is not None:
            body = await request.body()
        else:
            body = bytearray()
            async for chunk in request.stream():
                body += chunk
                if len(body) > MAX_BODY_BYTES:
                    raise StepError(413, f"the body is over the limit of "
                                    f"{MAX_BODY_BYTES} bytes")
                while len(body) > admitted:
                    await asyncio.to_thread(_admit, pool, BODY_STEP)
                    admitted += BODY_STEP
        try:
            tensors, _ = read(body, max_bytes=MAX_BODY_BYTES)
        except WireError as e:
            raise StepError(e.status, str(e)) from None
    except BaseException:
        pool.upload_done(admitted)
        raise
    return tensors, admitted


async def put_adapter(request: Request):
    """Keep an uploaded LoRA matrix. Its body is admitted into the store
    before it is read, and kept as a view, with no copy."""
    from ..image_steps import MAX_ADAPTER_BYTES, STATE

    h = _hash_path(request)
    pool = _lora_pool()
    if pool is None:
        tensors, _ = await _wire(request, MAX_BODY_BYTES)
        STATE.put_tensor(STATE.adapters, h, tensors, MAX_ADAPTER_BYTES,
                         copy=False)
        return _json(200, {"hash": h})
    tensors, admitted = await _admitted_wire(request, pool)
    try:
        STATE.put_tensor(STATE.adapters, h, tensors, MAX_ADAPTER_BYTES,
                         copy=False)
    finally:
        # After the put: the store counts the bytes before they leave the
        # admitted count.
        pool.upload_done(admitted)
    return _json(200, {"hash": h})


async def head_adapter(request: Request):
    from ..image_steps import STATE

    h = _hash_path(request)
    return Response(status_code=200 if h in STATE.adapters else 404)


async def register_adapter(request: Request):
    """Read an adapter file the server can see, under a model folder, into
    the adapter cache, and answer the hash of each of its tensors."""
    from ..image_steps import MAX_ADAPTER_BYTES, STATE, StepError, os_path_under
    from ..image_wire import WireError, read

    body = await _fields(request)
    path = os_path_under(str(body.get("path") or ""), SETTINGS.model_dirs)
    if path is None:
        raise StepError(404, "no such adapter file under the server's model "
                        "folders")

    pool = _lora_pool()

    def work():
        import os

        size = os.path.getsize(path)
        if size > 4 * MAX_ADAPTER_BYTES:
            raise StepError(413, "the adapter file is over the limit")
        if pool is not None:
            # The file is admitted whole before it is read.
            _admit(pool, size)
        try:
            with open(path, "rb") as f:
                data = f.read()
            try:
                tensors, _ = read(data)
            except WireError as e:
                raise StepError(400, f"the adapter file: {e}") from None
            out = {}
            for name, t in tensors.items():
                h = t.hash()
                STATE.put_tensor(STATE.adapters, h, {name: t},
                                 MAX_ADAPTER_BYTES, copy=False)
                out[name] = h
            return out
        finally:
            if pool is not None:
                pool.upload_done(size)

    return _json(200, {"tensors": await asyncio.to_thread(work)})


async def put_patch_table(request: Request):
    from ..image_steps import STATE, StepError, define_table, table_id

    want = _hash_path(request)
    raw = await request.body()
    got = table_id(raw)
    if got != want:
        raise StepError(400, "the table's id is not the one in the path",
                        got=got)
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise StepError(400, "a patch table body is a JSON object")
    model = request.query_params.get("model") or body.get("model")
    # The body's matrices stay in the store from its arrival to its turn's
    # end: no shed lets them go between.
    with STATE.adapters.holding(_sizes(body)):
        hold = await _acquire(model)
        try:
            served = _served(hold)
            pool = _lora_pool()
            # The pairs the define holds at its forward, the entry's own
            # left out, from the wire bytes of their matrices: hashed off
            # the loop.
            own = await asyncio.to_thread(_request_sizes, served, body)
            if pool is not None:
                from ..image_steps import PAIRS

                _hopeless(pool, hold._entry, sum(own.values()), PAIRS)

            def work():
                try:
                    if pool is not None:
                        _define_turn(pool, hold._entry, served, body)
                    core = define_table(served, body, STATE.adapters)
                    STATE.tables_of(served.spec.id, served.pipe).put(
                        want, core, raw, own)
                    return core
                finally:
                    if pool is not None:
                        from ..image_service import lora_settle
                        from ..image_steps import PAIRS

                        lora_settle(pool, hold._entry, PAIRS)

            await _on_worker(work, on=served)
        finally:
            hold.release()
    return _json(200, {"id": want})


def _hopeless(pool, entry, n: int, subject: str) -> None:
    """The route's 413 for ``n`` LoRA bytes no unload makes room for,
    from the pool's figures, with no lock."""
    from ..residency import LoraRefused

    try:
        pool.lora_hopeless(entry, n, subject)
    except LoraRefused as e:
        raise _refused(e) from None


def _sizes(body: dict) -> dict:
    from ..image_steps import STATE, matrix_sizes

    return matrix_sizes(body, STATE.adapters)


def _request_sizes(served, body: dict) -> dict:
    """{hash: bytes} of the matrices of ``body`` the store holds, those of
    the entry's own pairs left out: the request's own LoRA bytes."""
    from ..image_steps import STATE, request_matrices

    return request_matrices(served, body, STATE.adapters)


def _define_turn(pool, entry, served, body: dict,
                 subject: str | None = None) -> None:
    """The turn of a define on the worker, before ``.mx()``: the pairs of
    ``body`` not held and not the entry's own. Its held pairs and the
    uploaded matrices it names are its own keys, which its shed keeps."""
    from ..image_service import ensure_dit, lora_turn, own_bytes
    from ..image_steps import PAIRS, STATE, body_pairs, is_video, pairs_due
    from ..residency import LoraRefused
    from .. import video_service as vs

    # The define reads the DiT, which an encode lets go.
    if is_video(served.spec):
        vs.ensure_dit(served)
    else:
        ensure_dit(served)
    pairs = body_pairs(body, STATE.adapters)
    due = pairs_due(served, pairs)
    held = own_bytes(served.pipe, set(pairs) - set(due))
    try:
        lora_turn(pool, entry, sum(due.values()), subject=subject or PAIRS,
                  held=held, hold=tuple(pairs), store=tuple(_sizes(body)))
    except LoraRefused as e:
        raise _refused(e) from None


def _held(pipe, core: str) -> bool:
    """Whether the pipeline holds the core table ``core`` and every pair
    it names. Another table's define can shed the pairs of a table it
    did not name, and the client then sends that table again."""
    table = pipe.patch_tables.get(core)
    return table is not None and all(
        pipe.adapters.peek(key) is not None
        for _path, _band, patches in table["entries"]
        for key, _st, _alpha in patches)


def _table(served, tid):
    """The core table of the client table ``tid`` a step names. A core
    the pipeline let go, or whose pairs it shed, passes when the server
    holds its body and every matrix the body names: the step defines it
    again in its own turn (``_held_table``). Any other is missing, and the
    client sends its table again, with the 409's memory sentence of
    ``_table_hint``."""
    from ..image_steps import STATE, Missing

    if not tid:
        return None
    tables = STATE.tables_of(served.spec.id, served.pipe)
    core = tables.get(tid)
    if core is None or not (_held(served.pipe, core)
                            or _definable(tables.body(tid))):
        raise Missing([tid])
    return core


def _table_hint(served, tid) -> str | None:
    """The memory sentence of the 409 of a step's missing table ``tid``.
    It takes no pool lock."""
    from ..image_steps import STATE

    if not tid:
        return None
    return _memory_hint(STATE.tables_of(served.spec.id, served.pipe), tid)


def _memory_hint(tables, tid) -> str | None:
    """The sentence of a 409 whose table's matrices the server let go:
    their bytes."""
    from ..image_steps import STATE
    from ..residency import gb

    gone = sum(n for h, n in tables.sizes(tid).items()
               if h not in STATE.adapters)
    if _lora_pool() is None or not gone:
        return None
    return (f"{gb(gone, True)} GB of LoRA matrices are not held: the "
            "server let them go for memory")


def _definable(raw: bytes | None) -> bool:
    """Whether the server holds every matrix of the table body ``raw``."""
    from ..image_steps import STATE

    if raw is None:
        return False
    return all(h in STATE.adapters
               for e in json.loads(raw)["entries"]
               for p in e.get("patches") or []
               for h in (p["up"], p["down"]))


def _held_table(served, tid, core):
    """The step's core table ``core``, of the client table ``tid``, held
    for the step. An encode or another table's define since the step's
    check can let the core or its pairs go: it is defined again here from
    the body the client sent, in the step's own turn on the worker, so no
    other define lands between it and the step. A matrix of the body the
    server let go since is a 409 naming ``tid``. Runs on the worker."""
    from ..image_steps import STATE, Missing, define_table

    if core is None or _held(served.pipe, core):
        return core
    tables = STATE.tables_of(served.spec.id, served.pipe)
    raw = tables.body(tid)
    if raw is None:
        raise Missing([tid])
    try:
        core = define_table(served, json.loads(raw), STATE.adapters)
    except Missing:
        raise Missing([tid], hint=_memory_hint(tables, tid)) from None
    tables.put(tid, core, raw)
    return core


def _factor_turn(pool, entry, served, core, subject: str | None = None
                 ) -> None:
    """The turn of a forward on ``core``, after its pairs are held: the
    factors and kernel pairs it builds yet, or the growth of the default
    table's own bytes, booked as ``subject`` (a step's factors by
    default)."""
    from ..image_service import lora_turn, own_bytes, turn_figures
    from ..image_steps import FACTORS
    from ..residency import LoraRefused

    pipe = served.pipe
    need, held = turn_figures(entry, pipe, core)
    tid = core if core is not None else getattr(pipe, "default_table", None)
    table = pipe.patch_tables.get(tid) if tid is not None else None
    hold = [tid] if tid is not None else []
    if table:
        pairs = {k for _p, _b, patches in table["entries"]
                 for k, _s, _a in patches}
        hold += list(pairs)
        # The table's pairs are the job's own keys too, never idle to it.
        held += own_bytes(pipe, pairs)
    try:
        lora_turn(pool, entry, need, subject=subject or FACTORS, held=held,
                  hold=hold)
    except LoraRefused as e:
        raise _refused(e) from None


def _step_estimate(served, tid) -> int:
    """The LoRA bytes a step's forward holds of its client table ``tid``,
    the entry's own left out, from the sizes its define kept: a lower
    bound with no factors, for the route's 413. It reads no pipeline."""
    from ..image_steps import STATE

    if not tid:
        return 0
    return sum(STATE.tables_of(served.spec.id, served.pipe)
               .sizes(tid).values())


def _step_holds(served, tid) -> dict:
    """The uploaded matrices a step on ``tid`` defines its table from
    again, which it holds in the store to its turn's end."""
    from ..image_steps import STATE

    if not tid:
        return {}
    tables = STATE.tables_of(served.spec.id, served.pipe)
    return _sizes(json.loads(tables.body(tid) or b'{"entries": []}'))


def _skip(served, raw) -> tuple:
    """The blocks a step's rows leave out, zero-based, from its ``skip``
    field."""
    from ..image_steps import StepError

    if not isinstance(raw, list) or any(
            isinstance(i, bool) or not isinstance(i, int) for i in raw):
        raise StepError(400, "skip is a list of block indices")
    if not raw:
        return ()
    title = served.pipe.family.title
    n = served.facts.blocks
    if n is None:
        raise StepError(400, f"the {title} DiT skips no blocks")
    bad = sorted({i for i in raw if not 0 <= i < n})
    if bad:
        raise StepError(400, f"the {title} DiT has blocks 0 to {n - 1}, "
                        f"and skip names {bad}")
    return tuple(sorted(set(raw)))


async def step(request: Request):
    from ..image_steps import (CONVENTIONS, MAX_ROWS, STATE, Missing,
                               StepError, flow_targets, row_hashes, run_step)

    t0 = time.perf_counter()
    tensors, fields = await _wire(request, MAX_BODY_BYTES)
    run = STATE.get_run(fields.get("run"))
    if "video" in tensors or "audio" in tensors:
        return await _av_step(t0, tensors, fields, run)
    x_w, t_w = tensors.get("x"), tensors.get("timestep")
    if x_w is None or t_w is None:
        raise StepError(400, "a step body holds x and timestep")
    rows = fields.get("rows")
    if not isinstance(rows, list) or not rows:
        raise StepError(400, "a step names one row or more")
    if len(x_w.shape) != 4 or x_w.shape[0] != len(rows):
        raise StepError(400, f"x is (B, C, h, w) with one row per entry of "
                        f"rows, not {list(x_w.shape)} for {len(rows)}")
    if len(rows) > MAX_ROWS:
        raise StepError(413, f"a step takes at most {MAX_ROWS} rows")
    if tuple(t_w.shape) not in ((len(rows),), (1,), ()):
        raise StepError(400, "timestep holds one value or one per row")
    missing = [h for r in rows for h in row_hashes(r)
               if h not in STATE.tensors]
    model = fields.get("model") or (run.model if run else None)
    hold = await _acquire(model)
    keep = contextlib.ExitStack()
    try:
        served = _served(hold)
        _image_only(served, "a step on x")
        w_px, h_px = _px(served, "latent", *x_w.shape[2:])
        guided = 2.0 if len(rows) > 1 else 1.0
        budget = _check_size(hold, w_px, h_px, what="a step", cfg=guided,
                             phase="denoise")
        _check_refs(served, rows)
        fam = served.pipe.family.name
        conv = CONVENTIONS.get(fields.get("convention", "comfy"), {}).get(fam)
        if conv is None:
            raise StepError(400, f"convention {fields.get('convention')!r} "
                            f"has no adapter for {fam}")
        ref_cache = fields.get("ref_cache", False)
        if not isinstance(ref_cache, bool):
            raise StepError(400, "ref_cache is true or false")
        title = served.pipe.family.title
        parts = served.pipe.family.step is not None
        if ref_cache and run is None:
            raise StepError(400, "a step on a cache of reference K and V "
                            "names its run, whose end drops the cache")
        if ref_cache and not served.facts.ref_cache:
            raise StepError(400, f"the {title} DiT keeps no cache of "
                            "reference K and V")
        if ref_cache and parts and conv.time == "reference":
            raise StepError(400, f"the {title} DiT caches its references "
                            "by itself under the reference convention")
        skip = _skip(served, fields.get("skip", []))
        if skip and ref_cache:
            raise StepError(400, "a step on a cache of reference K and V "
                            "skips no blocks")
        table = None
        hint = None
        # The matrices the step may define its table again from stay in
        # the store from its check to its turn's end.
        keep.enter_context(STATE.adapters.holding(
            _step_holds(served, fields.get("patch_table"))))
        try:
            table = _table(served, fields.get("patch_table"))
        except Missing as m:
            missing += m.extra["missing"]
            hint = _table_hint(served, fields.get("patch_table"))
        if missing:
            raise Missing(missing, hint=hint)
        import numpy as np

        ts = np.broadcast_to(np.asarray(t_w.numpy(), np.float32).reshape(-1),
                             (len(rows),))
        # In float64, so a scaled timestep gives back its time in float32.
        sig = [conv.scale * float(t) + conv.shift for t in ts]
        targets = None
        if served.facts.flow_map:
            # The schedule is in the client's timestep convention.
            s_w = tensors.get("sample_sigmas")
            if s_w is None:
                raise StepError(400, f"the {served.pipe.family.title} DiT is "
                                "a flow map: a step body holds "
                                "sample_sigmas, the schedule the client "
                                "samples on")
            targets = [conv.scale * float(t) + conv.shift
                       for t in flow_targets(s_w.numpy(), ts)]
        gen = run.gen if run is not None else 0
        pool = _lora_pool()
        if pool is not None:
            from ..image_steps import FACTORS

            _hopeless(pool, hold._entry,
                      _step_estimate(served, fields.get("patch_table")),
                      FACTORS)

        def work():
            try:
                return _work()
            finally:
                if pool is not None:
                    from ..image_service import lora_settle
                    from ..image_steps import FACTORS

                    lora_settle(pool, hold._entry, FACTORS)

        def _work():
            # Every MLX op of a step runs here: MLX streams belong to the
            # thread that made them.
            import mlx.core as mx

            from gmlx.image import api, memfit

            from ..image_service import ensure_dit, size_fits
            from ..image_steps import FACTORS, _host, conditioning

            # A staged entry's encode let the DiT and the VAE go.
            ensure_dit(served)
            tid = fields.get("patch_table")
            if pool is not None and table is not None \
                    and not _held(served.pipe, table):
                raw = STATE.tables_of(served.spec.id, served.pipe).body(tid)
                if raw is not None:
                    _define_turn(pool, hold._entry, served, json.loads(raw),
                                 FACTORS)
            core = _held_table(served, tid, table)
            if pool is not None:
                _factor_turn(pool, hold._entry, served, core)
            fam = served.pipe.family.name
            conds = [conditioning(r, STATE.tensors, fam) for r in rows]
            # The plan of the rows as they are: their text, a prefix and
            # the reference latents each step runs. A step in parts runs
            # whole under another time rule, and a step on the run's cache
            # keeps the K and V of references it has none of yet, one copy
            # a row.
            building = {}
            if ref_cache:
                for i, c in enumerate(conds):
                    k = api._ref_key(c, i)
                    if c.ref_latents and k not in run.ref_cache:
                        building[k] = served.pipe.dit.ref_cache_bytes(
                            c.ref_latents)
            try:
                size_fits(served, w_px, h_px, budget, cfg=guided,
                          conds=conds, patch_table=core,
                          joined=parts and conv.time != "reference",
                          ref_cache=run.ref_cache if ref_cache else None,
                          building=sum(building.values()),
                          held_in=("denoise",), hint=True)
            except memfit.ImageMemoryError as e:
                raise StepError(400, f"a step of {w_px}x{h_px} pixels with "
                                "the references of its rows does not fit in "
                                "memory beside the models now loaded. "
                                f"{e.hint}") from None
            scope = conds
            if run is not None:
                # The run's conditionings keep their states between its
                # steps, as those of one request do in sample.
                for c in conds:
                    if c.key:
                        run.conds.add(c.key)
                        run.cond_objs.setdefault(
                            (c.key, c.ref_hashes, c.image_slots), c)
                scope = list(run.cond_objs.values()) + [
                    c for c in conds if not c.key]
            x = x_w.mx().astype(mx.float32)
            with api.state_scope(served.pipe, scope):
                v, fwd_s, cache = run_step(served, x, sig, conds, core, run,
                                           gen, targets, conv.time,
                                           ref_cache, skip)
            return _host(v), fwd_s, cache

        v, fwd_s, cache = await _on_worker(work, on=served)
        if conv.sign != 1.0:
            v = v * np.float32(conv.sign)
        if run is not None:
            run.steps += 1
            run.rows += len(rows)
            run.forward_s += fwd_s
            run.cache_hits += cache == "hit"
            run.last = STATE.clock()
        server_ms = (time.perf_counter() - t0) * 1000
        return _container({"out": v}, None, headers={
            "X-Gmlx-Forward-Ms": f"{fwd_s * 1000:.2f}",
            "X-Gmlx-Server-Ms": f"{server_ms:.2f}",
            "X-Gmlx-Prefix-Cache": cache})
    finally:
        keep.close()
        hold.release()


def _number(fields: dict, name: str, default=None) -> float:
    import math

    from ..image_steps import StepError

    v = fields.get(name, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or \
            not math.isfinite(v) or v <= 0:
        raise StepError(400, f"{name} is a positive number")
    return float(v)


def _clip_of(fam, vshape, ashape=None) -> tuple:
    """(frames, width, height) of the clip whose latents are video
    ``vshape`` and audio ``ashape``, or the refusal of latents that are no
    clip gmlx runs: 17 k + 5 frames on a canvas in multiples of 32."""
    from ..image_steps import COMFY, StepError

    stride = int(COMFY[fam.name]["latent"]["stride"])
    t, h, w = (int(d) for d in vshape[2:])
    frames = fam.clip_frames(t)
    width, height = w * stride, h * stride
    try:
        want = (None if frames is None
                else fam.latent_shapes(frames, width, height))
    except ValueError:
        want = None
    got = (tuple(vshape), tuple(ashape) if ashape is not None else None)
    if want is None or got != (want[0], want[1] if ashape is not None
                               else None):
        shown = f"video {list(vshape)}" + (
            f" and audio {list(ashape)}" if ashape is not None else "")
        raise StepError(400, f"latents {shown} are no {fam.title} clip gmlx "
                        "runs: a clip is 17 k + 5 frames on a canvas in "
                        "multiples of 32")
    return frames, width, height


def _check_clip(hold, served, *, frames: int, width: int, height: int,
                text: int, phase: str, what: str, table: str | None = None,
                audio: int | None = None, decodes: bool = False,
                anchors=(), shown: str | None = None,
                int8: str | None = None) -> None:
    """Refuse a step or a decode of a clip that does not fit beside the
    models loaded. The DiT stays loaded through each, and each is priced
    as the image routes price theirs: the DiT, the larger of its
    configured LoRAs and the pairs and patch tables it holds, the caches
    the server holds and the phase's work. A step under a patch table
    with patches, ``table``, runs a LoRA. ``audio`` is the audio decode's
    latents. With ``decodes``, a step is priced at the larger of the step
    and the clip's two decodes, so a clip that cannot decode stops before
    its first step. ``anchors`` are the clip's placed
    ``layout.AnchorRows``, and an anchors phase prices their encodes;
    ``shown`` names what is priced in the refusal. ``int8`` is the
    request's int8 setting, else the entry's."""
    from ..image_steps import StepError

    got = _clip_terms(hold, served, frames=frames, width=width,
                      height=height, text=text, phase=phase, table=table,
                      audio=audio, decodes=decodes, anchors=anchors,
                      int8=int8)
    if got is None:
        return
    held, work, budget = got
    if held + work > budget:
        clip = shown or (f"{audio} audio latents" if phase == "audio"
                         else f"{width}x{height} at {frames} frames")
        raise StepError(400, f"{what} of {clip} does not fit in memory "
                        "beside the models now loaded")


def _clip_terms(hold, served, *, frames: int, width: int, height: int,
                text: int, phase: str, table: str | None = None,
                audio: int | None = None, decodes: bool = False,
                anchors=(), int8: str | None = None
                ) -> tuple[int, int, int] | None:
    """``(held, work, budget)`` bytes of a phase as ``_check_clip`` prices
    it, or None when the host sets no budget."""
    from ..video_service import held_bytes, plan_clip

    budget_fn = getattr(SETTINGS.host, "budget", None)
    budget = budget_fn(hold) if budget_fn is not None else None
    if budget is None:
        return None
    pipe = served.pipe
    tid = pipe.default_table if table is None else table
    got = pipe.patch_tables.get(tid)
    # A client's core table the pipeline let go is defined again in the
    # step's own turn, so it is priced as a table with patches.
    live = bool(got["entries"]) if got is not None else table is not None
    plan = plan_clip(served.plan, frames=frames, width=width, height=height,
                     text=text, budget=budget,
                     decode=phase != "denoise" or decodes, audio=audio,
                     live_lora=live and phase == "denoise", anchors=anchors,
                     int8=int8)
    adapters, caches = held_bytes(served)
    held = (plan.terms["denoise"]["dit"]
            + max(plan.terms["denoise"]["lora"], adapters) + caches)
    work = (plan.terms["denoise"]["step"] if phase == "denoise"
            else plan.phases[phase])
    if decodes:
        work = max(work, plan.phases["audio"], plan.phases["decode"])
    return held, work, budget


def _check_encode(hold, served, text: str, sizes=()) -> None:
    """Refuse a prompt whose encode does not fit beside the caches the
    server holds, as the video command refuses it by the prompt's rows,
    with keyframes of ``sizes`` (width, height) through the vision tower.
    The DiT and its LoRA pairs go before the encoder loads."""
    from gmlx.video import memfit

    from ..image_steps import StepError
    from ..video_service import held_bytes, plan_clip, prompt_rows

    budget_fn = getattr(SETTINGS.host, "budget", None)
    budget = budget_fn(hold) if budget_fn is not None else None
    if budget is None:
        return
    rows = prompt_rows(served, text, sizes)
    blocks = (memfit.keyframe_blocks(served.pipe.video_family, sizes)
              if sizes else ())
    _adapters, caches = held_bytes(served)
    width, height = served.plan.size
    plan = plan_clip(served.plan, frames=served.plan.frames, width=width,
                     height=height, text=rows, budget=budget - caches,
                     decode=False, blocks=blocks)
    if "encode" in plan.over:
        kf = (f" with {len(sizes)} keyframe{'s' if len(sizes) > 1 else ''}"
              if sizes else "")
        raise StepError(400, f"the encode of {rows} text rows{kf} does not "
                        "fit in memory beside the models now loaded; "
                        f"{plan.hint()}")


async def _av_step(t0, tensors, fields, run):
    """A step of a model with a video and an audio stream (``AV_FAMILIES``):
    the latents as ComfyUI's sampler carries them, one row, with the
    anchors and vision spans it names (``video_service.step_anchors``),
    at the ``int8`` setting it names, else the entry's."""
    import numpy as np

    from gmlx.video import layout

    from ..image_steps import (AV_FAMILIES, STATE, Missing, StepError,
                               _cond_key, _f32, _host, row_hashes, step_pacer)
    from .. import video_service as vs

    v_w, a_w = tensors.get("video"), tensors.get("audio")
    t_w = tensors.get("timestep")
    if v_w is None or a_w is None or t_w is None:
        raise StepError(400, "an AV step body holds video, audio and timestep")
    if len(v_w.shape) != 5 or len(a_w.shape) != 4 or v_w.shape[0] != 1 \
            or a_w.shape[0] != 1:
        raise StepError(400, f"an AV step takes one clip, video (1, C, T, h, "
                        f"w) and audio (1, C, 2, T), not {list(v_w.shape)} "
                        f"and {list(a_w.shape)}")
    if tuple(t_w.shape) not in ((1,), ()):
        raise StepError(400, "an AV step's timestep holds one value")
    rows = fields.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise StepError(400, "an AV step names one row")
    hashes = row_hashes(rows[0])
    if len(hashes) != 1:
        raise StepError(400, "an AV step's row holds its cond alone")
    shifts = fields.get("shifts")
    if not isinstance(shifts, dict):
        raise StepError(400, "an AV step names its shifts, video and audio")
    shifts = {k: _number(shifts, k) for k in ("video", "audio")}
    scale = _number(fields, "audio_scale", 1.0)
    try:
        sent = vs.parse_int8(fields.get("int8"))
    except ValueError as e:
        raise StepError(400, str(e), param="int8") from None
    try:
        refs, spans = vs.step_anchors(fields)
    except ValueError as e:
        raise StepError(400, str(e)) from None
    named = [h for r in refs for h in r.hashes()]
    missing = [h for h in hashes + named if h not in STATE.tensors]
    model = fields.get("model") or (run.model if run else None)
    hold = await _acquire(model)
    keep = contextlib.ExitStack()
    try:
        served = _served(hold)
        if not _video(served) or served.plan.family not in AV_FAMILIES:
            raise StepError(400, f"{served.spec.id!r} is no video model; "
                            "its step takes x")
        frames, width, height = _clip_of(served.pipe.video_family,
                                         v_w.shape, a_w.shape)
        # Each vision span is a keyframe's image.
        why = vs.anchor_refusal(served, len(refs) or len(spans))
        if why:
            raise StepError(400, why)
        try:
            int8 = vs.request_int8(served.plan, served.pipe.video_family,
                                   sent)
        except ValueError as e:
            raise StepError(400, str(e), param="int8") from None
        table = None
        # The matrices the step may define its table again from stay in
        # the store from its check to its turn's end.
        keep.enter_context(STATE.adapters.holding(
            _step_holds(served, fields.get("patch_table"))))
        try:
            table = _table(served, fields.get("patch_table"))
        except Missing as m:
            missing += m.extra["missing"]
        if missing:
            raise Missing(missing)
        held = STATE.tensors.get(hashes[0])
        if held is None:
            raise Missing(hashes)
        n_text = int(held.shape[-2])
        got = {h: STATE.tensors.get(h) for h in named}
        lost = [h for h, t in got.items() if t is None]
        if lost:
            raise Missing(lost)
        try:
            placed = vs.anchor_rows(served, refs, got, frames, v_w.shape,
                                    a_w.shape)
            layout.text_segments(n_text, spans)
        except ValueError as e:
            raise StepError(400, str(e)) from None
        # A run's first step counts the clip's decodes too.
        first = run is not None and run.steps == 0
        _check_clip(hold, served, frames=frames, width=width, height=height,
                    text=n_text, phase="denoise", table=table, decodes=first,
                    what="a clip" if first else "a step", anchors=placed,
                    int8=int8)
        ts = float(np.asarray(t_w.numpy(), np.float32).reshape(-1)[0])
        gen = run.gen if run is not None else 0
        tid = fields.get("patch_table")
        pool = _lora_pool()
        if pool is not None:
            from ..image_steps import FACTORS

            _hopeless(pool, hold._entry, _step_estimate(served, tid),
                      FACTORS)

        def work():
            try:
                return _work()
            finally:
                if pool is not None:
                    from ..image_service import lora_settle
                    from ..image_steps import FACTORS

                    lora_settle(pool, hold._entry, FACTORS)

        def _work():
            # An encode let the DiT and every pair go: the own pairs load
            # again with the DiT, within the footprint, and the turns book
            # the table's pairs, its factors and the own table's excess.
            import mlx.core as mx

            from gmlx.video import api

            from ..image_steps import FACTORS, StepInterrupted

            vs.ensure_dit(served)
            if pool is not None and table is not None \
                    and not _held(served.pipe, table):
                raw = STATE.tables_of(served.spec.id, served.pipe).body(tid)
                if raw is not None:
                    _define_turn(pool, hold._entry, served, json.loads(raw),
                                 FACTORS)
            core = _held_table(served, tid, table)
            if pool is not None:
                _factor_turn(pool, hold._entry, served, core)
            feats = _f32(held)
            if feats.ndim == 2:
                feats = feats[None]
            dim = int(served.pipe.dit.cfg.text_dim)
            if feats.ndim != 3 or feats.shape[0] != 1 or \
                    feats.shape[1] == 0 or feats.shape[2] != dim:
                raise StepError(400, f"a row's cond is (1, L, {dim}) encoder "
                                f"features, not {tuple(feats.shape)}")
            cond = ((), ())
            if refs:
                try:
                    cond = vs.anchor_cond(served, refs, placed, got)
                except ValueError as e:
                    raise StepError(400, str(e)) from None
            t1 = time.perf_counter()
            vv, va, hit = vs.av_step(
                served, v_w.mx(), a_w.mx(), ts, shifts, scale, feats,
                _cond_key(hashes[0]), core, anchors=placed, cond=cond,
                spans=spans, int8=int8,
                pacer=api.Budget(step_pacer(run, gen)))
            mx.eval(vv, va)
            if run is not None and (run.gen != gen or run.ended):
                raise StepInterrupted()
            return _host(vv), _host(va), time.perf_counter() - t1, hit

        vv, va, fwd_s, hit = await _on_worker(work, on=served)
        if run is not None:
            run.steps += 1
            run.rows += 1
            run.forward_s += fwd_s
            run.cache_hits += bool(hit)
            run.last = STATE.clock()
        server_ms = (time.perf_counter() - t0) * 1000
        return _container({"video": vv, "audio": va}, None, headers={
            "X-Gmlx-Forward-Ms": f"{fwd_s * 1000:.2f}",
            "X-Gmlx-Server-Ms": f"{server_ms:.2f}",
            "X-Gmlx-Prefix-Cache": "hit" if hit else "miss"})
    finally:
        keep.close()
        hold.release()


async def encode(request: Request):
    from ..image_service import ImageRequestError, JobStopped, decode_image
    from ..image_steps import STATE, StepError
    from ..image_steps import encode as do_encode
    from ..image_wire import CONTENT_TYPE

    ctype = request.headers.get("content-type", "").split(";")[0].strip()
    images = []
    if ctype == CONTENT_TYPE:
        tensors, fields = await _wire(request, MAX_BODY_BYTES)
        for name in sorted(tensors, key=lambda n: (len(n), n)):
            t = tensors[name]
            if t.dtype != "U8" or len(t.shape) != 3 or t.shape[2] not in (3, 4):
                raise StepError(400, f"image {name!r} is u8 (H, W, 3 or 4)")
            images.append(t.numpy().copy())
    else:
        fields = await _fields(request)
        for i, b64 in enumerate(fields.get("images") or []):
            try:
                data = base64.b64decode(b64, validate=True)
            except ValueError:
                raise StepError(400, f"images[{i}] is not base64") from None
            images.append(decode_image(data, "images"))
    text = fields.get("text")
    if not isinstance(text, str):
        raise StepError(400, "encode needs text")
    want_refs = fields.get("refs", True)
    if not isinstance(want_refs, bool):
        raise StepError(400, "refs is true or false")
    requested = fields.get("prompt_upsampling")
    if requested is not None and not isinstance(requested, bool):
        raise StepError(400, "prompt_upsampling is true or false")
    clip = _clip_fields(fields, len(images))
    token = fields.get("interrupt_id")
    if token is not None and (not isinstance(token, str) or not token):
        raise StepError(400, "interrupt_id is a non-empty string")
    revise = getattr(SETTINGS.host, "revise", None)
    revised = None
    if revise is not None:
        # An interrupt naming the interrupt_id ends the rewrite, and the
        # encode with it.
        stop = None
        if token is not None:
            stop = clip["stop"] = STATE.begin_encode(token)
        try:
            # Without the clip's frames, a prompt with images runs as
            # typed: a video entry's keyframe task names the clip's length.
            revised = await asyncio.to_thread(
                revise, fields.get("model"), text, requested, bool(images),
                **clip)
        except JobStopped:
            raise StepError(409, "the encode was interrupted",
                            code="interrupted") from None
        except ImageRequestError as e:
            raise StepError(e.status, str(e)) from None
        except asyncio.CancelledError:
            # The client went away: end the rewrite on its thread too.
            if stop is not None:
                stop.set()
            raise
        finally:
            if token is not None:
                STATE.end_encode(token)
        if stop is not None and stop.is_set():
            # A stop after the caption's last token still ends the encode.
            raise StepError(409, "the encode was interrupted",
                            code="interrupted")
    hold = await _acquire(fields.get("model"))
    try:
        served = _served(hold)
        if _video(served):
            from ..video_service import encode as encode_prompt
            from ..video_service import keyframe_refusal

            # Keyframes are the clip's canvas, through the vision tower.
            frames = [i[..., :3] for i in images]
            why = keyframe_refusal(served, frames)
            if why:
                raise StepError(400, why)
            sizes = [(int(i.shape[1]), int(i.shape[0])) for i in frames]
            text = revised or text
            await asyncio.to_thread(_check_encode, hold, served, text, sizes)
            feats, spans = await _on_worker(_settled(hold, encode_prompt),
                                            served, text, frames, on=served)
            got = {"cond": feats, "refs": [], "image_slots": [],
                   "spans": spans}
            return _held_encode(got, revised)
        borrow = None
        if served.spec.encoder_model is not None:
            from .images import _borrow_for

            fn, chat_hold = await asyncio.to_thread(_borrow_for, served.spec)
            import threading

            stop = threading.Event()

            def borrow(run, _fn=fn, _pipe=served.pipe):
                return _fn(_pipe, run, stop, None)
        else:
            chat_hold = None
        try:
            typed = (text,) if revised else ()
            got = await _on_worker(_settled(hold, do_encode), served,
                                   revised or text,
                                   [i[..., :3] for i in images], borrow,
                                   want_refs, *typed, on=served)
        finally:
            if chat_hold is not None:
                chat_hold.release()
    finally:
        hold.release()
    return _held_encode(got, revised)


def _clip_fields(fields: dict, images: int) -> dict:
    """The rewrite's facts of a video clip from an encode's ``frames``
    and ``keyframes``, the role of each image in order: ``{frames,
    first, last}``, or {} when the encode names no frames."""
    from ..image_steps import StepError

    frames, roles = fields.get("frames"), fields.get("keyframes")
    if frames is None:
        if roles:
            raise StepError(400, "keyframes needs frames: a keyframe "
                            "prompt names the clip's length")
        return {}
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise StepError(400, "frames is the clip's frame count, a positive "
                        "integer")
    roles = [] if roles is None else roles
    if not isinstance(roles, list) or roles not in (
            [], ["first"], ["last"], ["first", "last"]):
        raise StepError(400, "keyframes lists the role of each image, "
                        "\"first\" then \"last\"")
    if len(roles) != images:
        raise StepError(400, f"keyframes names {len(roles)} role"
                        f"{'' if len(roles) == 1 else 's'} for {images} "
                        f"image{'' if images == 1 else 's'}")
    return {"frames": frames, "first": "first" in roles,
            "last": "last" in roles}


def _held_encode(got: dict, revised):
    """The encode's answer, with each tensor held by its hash."""
    from ..image_steps import STATE
    from ..image_wire import read, write

    out = {"cond": got["cond"]}
    if "typed" in got:
        out["cond_typed"] = got["typed"]
    for i, z in enumerate(got["refs"]):
        out[f"ref_{i}"] = z
    body = write(out)
    held, _ = read(body)
    hashes = {}
    for name, t in held.items():
        h = t.hash()
        STATE.tensors.put(h, t)
        hashes[name] = h
    fields_out = {"cond": hashes["cond"],
                  "ref_latents": [hashes[f"ref_{i}"]
                                  for i in range(len(got["refs"]))],
                  "image_slots": got["image_slots"],
                  "num_tokens": int(got["cond"].shape[1])}
    if revised:
        fields_out["revised_prompt"] = revised
    if "cond_typed" in hashes:
        # The typed prompt's cond, for references added after the encode.
        fields_out["cond_typed"] = hashes["cond_typed"]
    if got.get("spans"):
        fields_out["spans"] = [list(sp) for sp in got["spans"]]
    return _container(out, fields_out)


async def vae_decode(request: Request):
    from ..image_steps import MAX_ROWS, StepError
    from ..image_steps import vae_decode as do_decode

    tensors, fields = await _wire(request, MAX_BODY_BYTES)
    lat = tensors.get("latents")
    if lat is None or len(lat.shape) != 4:
        raise StepError(400, "vae/decode takes latents (B, C, h, w)")
    if lat.shape[0] > MAX_ROWS:
        raise StepError(413, f"a VAE pass takes at most {MAX_ROWS} images")
    hold = await _acquire(fields.get("model"))
    try:
        served = _served(hold)
        _image_only(served, "vae/decode")
        _check_size(hold, *_px(served, "vae", *lat.shape[2:]),
                    what="a VAE decode", phase="decode")
        img = await _on_worker(do_decode, served, lat,
                               bool(fields.get("alpha")), on=served)
    finally:
        hold.release()
    return _container({"image": img})


async def vae_encode(request: Request):
    from ..image_steps import MAX_ROWS, StepError
    from ..image_steps import vae_encode as do_encode

    tensors, fields = await _wire(request, MAX_BODY_BYTES)
    img = tensors.get("image")
    if img is None or len(img.shape) != 4:
        raise StepError(400, "vae/encode takes an image (B, H, W, C) in "
                        "[0, 1]")
    if img.shape[0] > MAX_ROWS:
        raise StepError(413, f"a VAE pass takes at most {MAX_ROWS} images")
    hold = await _acquire(fields.get("model"))
    try:
        served = _served(hold)
        _image_only(served, "vae/encode")
        _check_size(hold, int(img.shape[2]), int(img.shape[1]),
                    what="a VAE encode", init=True, phase="init")
        lat = await _on_worker(do_encode, served, img, on=served)
    finally:
        hold.release()
    return _container({"latents": lat})


async def _video_hold(request: Request, name: str, ndim: int, shape: str,
                      key: str = "latents", one: bool = True):
    """The tensor ``key`` of a VAE route of video models, of ``ndim``
    dimensions and the first of them 1 with ``one``, its hold and the
    served entry. ``shape`` names it in the refusal."""
    from ..image_steps import StepError

    tensors, fields = await _wire(request, MAX_BODY_BYTES)
    lat = tensors.get(key)
    if lat is None or len(lat.shape) != ndim or (one and lat.shape[0] != 1):
        raise StepError(400, f"{name} takes {shape}")
    hold = await _acquire(fields.get("model"))
    served = _served(hold)
    if not _video(served):
        hold.release()
        raise StepError(400, f"{served.spec.id!r} is an image model; {name} "
                        "takes a video model")
    return lat, hold, served


async def video_decode(request: Request):
    """Model-space video latents (1, 24, T, h, w) to frames (F, H, W, 3)
    f16 in [0, 1]."""
    from ..video_service import decode_video

    lat, hold, served = await _video_hold(
        request, "vae/video/decode", 5,
        "the latents of one clip (1, C, T, h, w)")
    try:
        frames, width, height = _clip_of(served.pipe.video_family, lat.shape)
        _check_clip(hold, served, frames=frames, width=width, height=height,
                    text=1, phase="decode", what="a video decode")
        img = await _on_worker(lambda: decode_video(served, lat.mx()), on=served)
    finally:
        hold.release()
    return _container({"frames": img})


async def audio_decode(request: Request):
    """Model-space audio latents (1, 32, 2, T) to the waveform
    (1, 2, samples) f32."""
    from ..image_steps import COMFY, StepError
    from ..video_service import decode_audio

    lat, hold, served = await _video_hold(
        request, "vae/audio/decode", 4, "the latents of one clip (1, C, 2, T)")
    try:
        row = COMFY[served.pipe.video_family.name]["audio"]
        if tuple(lat.shape[1:3]) != (row["channels"], row["stereo"]):
            raise StepError(400, f"audio latents are (1, {row['channels']}, "
                            f"{row['stereo']}, T), not {list(lat.shape)}")
        width, height = served.plan.size
        _check_clip(hold, served, frames=served.plan.frames, width=width,
                    height=height, text=1, phase="audio",
                    what="an audio decode", audio=int(lat.shape[3]))
        img = await _on_worker(lambda: decode_audio(served, lat.mx()), on=served)
    finally:
        hold.release()
    return _container({"waveform": img})


class _Stream:
    """A request body read as far as its safetensors header, the rest
    read or drained after: a refusal after the header reads the rest a
    chunk at a time and keeps none of it, so a client still sending gets
    the answer."""

    def __init__(self, request: Request):
        self._it = request.stream().__aiter__()
        self.got = bytearray()

    async def _more(self) -> bool:
        try:
            self.got += await self._it.__anext__()
        except StopAsyncIteration:
            return False
        return True

    async def header(self, max_bytes: int) -> dict:
        """The header's tensor specs and the request fields."""
        from ..image_steps import StepError
        from ..image_wire import MAX_HEADER_BYTES

        while len(self.got) < 8 and await self._more():
            pass
        n = int.from_bytes(bytes(self.got[:8]), "little") \
            if len(self.got) >= 8 else -1
        if not 0 <= n <= MAX_HEADER_BYTES:
            raise StepError(400, "the body has no safetensors header")
        while len(self.got) < 8 + n and await self._more():
            if len(self.got) > max_bytes:
                raise StepError(413, f"the body is over the limit of "
                                f"{max_bytes} bytes")
        try:
            head = json.loads(bytes(self.got[8:8 + n]))
            fields = json.loads((head.pop("__metadata__", None) or {}).get(
                "gmlx", "{}"))
        except (ValueError, AttributeError):
            raise StepError(400, "the safetensors header is not JSON") \
                from None
        if not isinstance(head, dict) or not isinstance(fields, dict):
            raise StepError(400, "the safetensors header is not a JSON "
                            "object")
        return head, fields

    async def rest(self, max_bytes: int) -> bytearray:
        from ..image_steps import StepError

        while await self._more():
            if len(self.got) > max_bytes:
                raise StepError(413, f"the body is over the limit of "
                                f"{max_bytes} bytes")
        return self.got

    async def drain(self) -> None:
        self.got = bytearray()
        while await self._more():
            self.got = bytearray()


# The video encode's window rules, as its refusals state them.
WINDOW_RULES = (
    "a guide clip goes in windows, each at frame `at` of `clip_frames`: "
    "an open window is 17 m frames from a multiple of 17, 1 <= m <= {p}, "
    "ending before the clip does; the last window is the clip's 1 frame or "
    "17 j + 5 frames from a multiple of 17 to the clip's end, j <= {p}")


def _window(n: int, at: int, clip: int, is_open: bool, pieces: int) -> None:
    """Refuse a window of ``n`` frames at frame ``at`` of a ``clip``-frame
    clip that breaks the window rules."""
    from ..image_steps import StepError

    ok = at >= 0 and at % 17 == 0
    if ok and is_open:
        ok = n % 17 == 0 and 1 <= n // 17 <= pieces and at + n < clip
    elif ok:
        ok = at + n == clip and (n == clip == 1 or (
            n >= 5 and (n - 5) % 17 == 0 and (n - 5) // 17 <= pieces))
    if not ok:
        raise StepError(400, f"a window of {n} frames at frame {at} of a "
                        f"{clip}-frame clip{' (open)' if is_open else ''} "
                        "breaks the rules: " + WINDOW_RULES.format(p=pieces))


def _window_fit(hold, served, *, n: int, width: int, height: int,
                level: int, is_open: bool):
    """``(need, free, largest)``: the bytes a window of ``n`` frames needs
    beside the models now loaded, the bytes free beside them, and the
    largest k whose windows of 17 k + 5 frames fit (0 for none, None when
    the host sets no budget). An open window of 17 m frames is priced as
    the clip of 17 m + 5, its encode a piece shy of that one's. The wire
    bytes count twice: the body and its MLX copy."""
    from gmlx.video.layout import AnchorRows

    def need(f, wire):
        t = 1 if f == 1 else (f - 5) // 17 * 5 + 2
        got = _clip_terms(hold, served, frames=max(served.plan.frames, f),
                          width=width, height=height, text=1,
                          phase="anchors", anchors=(AnchorRows(0, t, 0),))
        if got is None:
            return None
        held, work, budget = got
        return work + 2 * wire * height * width * 3 * level, budget - held

    first = need(n + 5 if is_open else n, n)
    if first is None:
        return None
    k = (n - 5) // 17 if n > 1 else 0
    while k > 0:
        w, free = need(17 * k + 5, 17 * k + 5)
        if w <= free:
            break
        k -= 1
    return first[0], first[1], k


async def video_encode(request: Request):
    """Frames (T, H, W, 3) of a keyframe or a guide clip, float in [0, 1]
    or uint8 levels, 1 or 17 k + 5 of them on a canvas of the clip, to
    model-space video latents (1, 24, t, h, w) f32, priced as the anchor
    encode of a clip. A long clip goes in windows of whole pieces, by the
    fields ``at``, ``clip_frames`` and ``open``: an open window keeps the
    latents the clip's end drops, 5 a piece. The media gate leaves the
    body unread, and the window is priced from its header before a byte
    of its frames is read. A refusal past the header reads the frames a
    chunk at a time and drops them, so the connection serves the next
    request; a body past MAX_BODY_BYTES answers 413 and closes it."""
    from ..image_steps import COMFY, StepError
    from ..image_wire import WireError, read
    from ..video_service import encode_video, on_canvas

    _content_type(request)
    _content_length(request, MAX_BODY_BYTES)
    body = _Stream(request)
    try:
        head, fields = await body.header(MAX_BODY_BYTES)
        spec = head.get("frames")
        shape = spec.get("shape") if isinstance(spec, dict) else None
        dtype = spec.get("dtype") if isinstance(spec, dict) else None
        if (not isinstance(shape, list) or len(shape) != 4
                or not all(isinstance(d, int) for d in shape)):
            raise StepError(400, "vae/video/encode takes frames (T, H, W, 3)"
                            " in [0, 1]")
        n, height, width, c = shape
        if c != 3 or dtype not in ("F32", "F16", "BF16", "U8"):
            raise StepError(400, "vae/video/encode takes frames (T, H, W, 3) "
                            f"in [0, 1] or uint8 levels, not {dtype} {shape}")
        hold = await _acquire(fields.get("model"))
        try:
            served = _served(hold)
            if not _video(served):
                raise StepError(400, f"{served.spec.id!r} is an image model; "
                                "vae/video/encode takes a video model")
            fam = served.pipe.video_family
            pieces = int(COMFY[fam.name]["vae"]["encode_pieces"])
            at, clip = int(fields.get("at", 0)), int(
                fields.get("clip_frames", n))
            is_open = bool(fields.get("open", False))
            if not on_canvas(fam, width, height) or not (
                    clip == 1 or (clip >= 5 and (clip - 5) % 17 == 0)):
                what = f"frames {shape}" if clip == n else (
                    f"a clip of {clip} frames of {width}x{height}")
                raise StepError(400, f"{what} are no {fam.title} keyframe or "
                                "guide clip: 1 or 17 k + 5 frames on a canvas "
                                "in multiples of 32")
            _window(n, at, clip, is_open, pieces)
            level = 1 if dtype == "U8" else 2 if dtype in ("F16", "BF16") \
                else 4
            fit = _window_fit(hold, served, n=n, width=width, height=height,
                              level=level, is_open=is_open)
            if fit is not None and fit[0] > fit[1]:
                need, free, k = fit
                cost = (f"{dtype.lower()} on the wire, needs "
                        f"{need / (1 << 30):.2f} GiB, and "
                        f"{max(free, 0) / (1 << 30):.2f} GiB is free")
                if clip == 1:
                    shown = f"a {width}x{height} frame"
                    told = f"the frame, {cost}"
                else:
                    shown = (f"a guide clip of {clip} frames of "
                             f"{width}x{height}")
                    told = (f"its window of {n} frames, {cost}; " + (
                        f"a window of {17 * k} frames, or {17 * k + 5} at "
                        "the clip's end, fits" if k else "no window fits"))
                raise StepError(
                    400, f"a video encode of {shown} does not fit in memory "
                    f"beside the models now loaded: {told}", window=k)
            raw = await body.rest(MAX_BODY_BYTES)
            try:
                tensors, _ = read(raw, max_bytes=MAX_BODY_BYTES)
            except WireError as e:
                raise StepError(e.status, str(e)) from None
            img = tensors["frames"].mx()
            del raw, tensors
            lat = await _on_worker(lambda: encode_video(served, img,
                                                        open=is_open), on=served)
        finally:
            hold.release()
    except StepError as e:
        if e.status != 413:
            await body.drain()
        raise
    return _container({"latents": lat})


async def audio_encode(request: Request):
    """A stereo waveform (1, 2, samples) f32 at the audio VAE's rate to
    model-space audio latents (1, 32, 2, T) f32, the samples padded to a
    whole latent, priced as the anchor encode of a clip."""
    from gmlx.video.layout import AnchorRows

    from ..image_steps import COMFY, StepError
    from ..video_service import encode_audio

    wav, hold, served = await _video_hold(
        request, "vae/audio/encode", 3, "the waveform of one sound (1, 2, n)",
        key="waveform")
    try:
        row = COMFY[served.pipe.video_family.name]["audio"]
        n = int(wav.shape[2])
        if int(wav.shape[1]) != row["stereo"] or n < 1:
            raise StepError(400, f"a waveform is (1, {row['stereo']}, n) at "
                            f"{row['sample_rate']} Hz, not {list(wav.shape)}")
        hop = row["sample_rate"] // row["latents_per_second"]
        t = -(-n // hop)
        width, height = served.plan.size
        _check_clip(hold, served, frames=served.plan.frames, width=width,
                    height=height, text=1, phase="anchors",
                    what="an audio encode", anchors=(AnchorRows(0, 0, t),),
                    shown=f"{n} samples")
        lat = await _on_worker(lambda: encode_audio(served, wav.mx()), on=served)
    finally:
        hold.release()
    return _container({"latents": lat})


async def run_begin(request: Request):
    from ..image_steps import RUN_IDLE_S, STATE

    body = await _fields(request)
    hold = await _acquire(body.get("model"))
    try:
        run = STATE.begin(_served(hold).spec.id, hold)
    except BaseException:
        hold.release()
        raise
    return _json(200, {"run": run.id, "idle_s": RUN_IDLE_S})


async def run_end(request: Request):
    from ..image_steps import STATE

    rid = request.path_params["run"]
    with STATE.lock:
        run = STATE.runs.get(rid)
    if run is not None:
        STATE.end(run)
    return _json(200, {"run": rid, "ended": run is not None})


async def interrupt(request: Request):
    from ..image_steps import STATE

    body = await _fields(request)
    return _json(200, {"interrupted": STATE.interrupt(body.get("run"))})


ROUTES = (
    ("capabilities", "GET", capabilities, False),
    ("models", "GET", models, True),
    ("models/load", "POST", load, True),
    ("models/unload", "POST", unload, True),
    ("models/{model}/status", "GET", status, True),
    ("models/{model}/manifest", "GET", manifest, True),
    ("encode", "POST", encode, True),
    ("tensors/{digest}", "PUT", put_tensor, True),
    ("tensors/{digest}", "HEAD", head_tensor, True),
    ("adapters/register", "POST", register_adapter, True),
    ("adapters/{digest}", "PUT", put_adapter, True),
    ("adapters/{digest}", "HEAD", head_adapter, True),
    ("patch-tables/{digest}", "PUT", put_patch_table, True),
    ("step", "POST", step, True),
    ("vae/decode", "POST", vae_decode, True),
    ("vae/encode", "POST", vae_encode, True),
    ("vae/video/decode", "POST", video_decode, True),
    ("vae/audio/decode", "POST", audio_decode, True),
    ("vae/video/encode", "POST", video_encode, True),
    ("vae/audio/encode", "POST", audio_encode, True),
    ("runs/begin", "POST", run_begin, True),
    ("runs/{run}/end", "POST", run_end, True),
    ("interrupt", "POST", interrupt, True),
)


def mount(app, settings, host=None, model_dirs=()) -> None:
    """Serve the step routes on ``app`` with the image settings
    ``settings``, finding models through ``host`` (the server's image
    models by default), or remove them when ``settings.steps_api`` is off.
    Calling it again replaces the routes."""
    from ..image_steps import STATE
    from ..image_wire import PREFIX

    SETTINGS.settings = settings
    SETTINGS.model_dirs = tuple(model_dirs or ())
    SETTINGS.host = host if host is not None else ServeHost()
    STATE.configure(settings)
    STATE.log_requests = getattr(SETTINGS.host, "log_requests", True)
    _remove_routes(app, *{f"{PREFIX}/{p}" for p, *_ in ROUTES})
    if not settings.steps_api:
        STATE.reset()
        return
    for p, method, fn, versioned in ROUTES:
        app.add_api_route(f"{PREFIX}/{p}", _route(fn, versioned=versioned,
                                                  name=p),
                          methods=[method], include_in_schema=False)


def install_image_step_routes(cfg) -> None:
    """Serve the step routes when ``server.image.steps_api`` is set, and
    remove them when it is not. A reload calls it again."""
    from gmlx.config import ImageServerCfg

    mount(_app(), getattr(cfg, "image", None) or ImageServerCfg(),
          model_dirs=getattr(cfg, "model_dirs", ()))
