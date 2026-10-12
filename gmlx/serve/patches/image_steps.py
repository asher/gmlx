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
    """Where the step routes find their image models: the configured
    image models and the residency pool of the server."""

    log_requests = True

    def resolve(self, model):
        """``(path, spec)`` of an image model id or path."""
        return serving.resolve_image_request(model)

    def acquire(self, model):
        """A hold of the loaded model; ``hold._entry.pipeline`` is its
        ServedImage, and ``hold.release()`` lets it go."""
        from .. import residency

        return residency.acquire_image_model(model)

    def models(self) -> dict:
        return serving.image_models()

    def default_model(self) -> str | None:
        """The id an image request with no model takes, or None."""
        return serving.default_image_model_id()

    def resident(self, path) -> bool:
        from ..residency import _installed_pool

        pool = _installed_pool()
        return pool is not None and pool.resident_entry(path) is not None

    def plan(self, spec):
        from ..image_service import entry_plan

        return entry_plan(spec)

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


async def _wire(request: Request, max_bytes: int):
    from ..image_steps import StepError
    from ..image_wire import CONTENT_TYPE, WireError, read

    ctype = request.headers.get("content-type", "").split(";")[0].strip()
    if ctype != CONTENT_TYPE:
        raise StepError(415, f"the body is {CONTENT_TYPE}, not "
                        f"{ctype or 'untyped'}")
    body = await request.body()
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
            return _json(e.status, e.body())
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


async def _on_worker(fn, *args):
    from ..image_service import WORKER

    return await asyncio.wrap_future(WORKER.submit(fn, *args))


def _served(hold):
    return hold._entry.pipeline


def _check_size(hold, width: int, height: int, *, what: str,
                cfg: float = 1.0, init: bool = False) -> None:
    """Refuse a step or a VAE pass over the size limits of the image
    routes with 413, and one whose memory plan does not fit beside the
    models loaded with 400."""
    from ..image_service import MAX_PIXELS, MAX_SIDE, size_fits
    from ..image_steps import StepError

    if width * height > MAX_PIXELS or max(width, height) > MAX_SIDE:
        raise StepError(413, f"{what} of {width}x{height} pixels is over the "
                        f"limit of {MAX_PIXELS} pixels (2048x2048) and a "
                        f"side of {MAX_SIDE}")
    budget_fn = getattr(SETTINGS.host, "budget", None)
    budget = budget_fn(hold) if budget_fn is not None else None
    if not size_fits(_served(hold), width, height, budget, cfg=cfg,
                     init=init):
        raise StepError(400, f"{what} of {width}x{height} pixels does not "
                        "fit in memory beside the models now loaded")


def _px(served, part: str, h: int, w: int) -> tuple:
    """(width, height) in pixels of latents (h, w) in the model's latent
    space (``part`` "latent") or the VAE's (``part`` "vae")."""
    from ..image_steps import COMFY, StepError

    facts = COMFY.get(served.pipe.family.name)
    if facts is None:
        raise StepError(400, f"{served.pipe.family.title} has no step "
                        "contract adapter")
    stride = int(facts[part]["stride"])
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
    return SETTINGS.host.unload(path, spec)


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


async def put_adapter(request: Request):
    from ..image_steps import MAX_ADAPTER_BYTES, STATE

    h = _hash_path(request)
    tensors, _ = await _wire(request, MAX_BODY_BYTES)
    STATE.put_tensor(STATE.adapters, h, tensors, MAX_ADAPTER_BYTES)
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

    def work():
        import os

        if os.path.getsize(path) > 4 * MAX_ADAPTER_BYTES:
            raise StepError(413, "the adapter file is over the limit")
        with open(path, "rb") as f:
            data = f.read()
        try:
            tensors, _ = read(data)
        except WireError as e:
            raise StepError(400, f"the adapter file: {e}") from None
        out = {}
        for name, t in tensors.items():
            h = t.hash()
            STATE.put_tensor(STATE.adapters, h, {name: t}, MAX_ADAPTER_BYTES)
            out[name] = h
        return out

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
    hold = await _acquire(model)
    try:
        served = _served(hold)

        def work():
            core = define_table(served, body, STATE.adapters)
            STATE.tables_of(served.spec.id, served.pipe)[want] = core
            return core

        await _on_worker(work)
    finally:
        hold.release()
    return _json(200, {"id": want})


def _table(served, tid):
    from ..image_steps import STATE, Missing

    if not tid:
        return None
    core = STATE.tables_of(served.spec.id, served.pipe).get(tid)
    if core is None or core not in served.pipe.patch_tables:
        raise Missing([tid])
    return core


async def step(request: Request):
    from ..image_steps import (CONVENTIONS, MAX_ROWS, STATE,
                               Missing, StepError, row_hashes, run_step)

    t0 = time.perf_counter()
    tensors, fields = await _wire(request, MAX_BODY_BYTES)
    run = STATE.get_run(fields.get("run"))
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
    try:
        served = _served(hold)
        _check_size(hold, *_px(served, "latent", *x_w.shape[2:]),
                    what="a step", cfg=2.0 if len(rows) > 1 else 1.0)
        fam = served.pipe.family.name
        conv = CONVENTIONS.get(fields.get("convention", "comfy"), {}).get(fam)
        if conv is None:
            raise StepError(400, f"convention {fields.get('convention')!r} "
                            f"has no adapter for {fam}")
        table = None
        try:
            table = _table(served, fields.get("patch_table"))
        except Missing as m:
            missing += m.extra["missing"]
        if missing:
            raise Missing(missing)
        import numpy as np

        ts = np.broadcast_to(np.asarray(t_w.numpy(), np.float32).reshape(-1),
                             (len(rows),))
        sig = [float(conv.scale * t + conv.shift) for t in ts]
        gen = run.gen if run is not None else 0

        def work():
            # Every MLX op of a step runs here: MLX streams belong to the
            # thread that made them.
            import mlx.core as mx

            from gmlx.image import api

            from ..image_steps import _host, conditioning

            fam = served.pipe.family.name
            conds = [conditioning(r, STATE.tensors, fam) for r in rows]
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
                v, fwd_s, cache = run_step(served, x, sig, conds, table, run,
                                           gen)
            return _host(v), fwd_s, cache

        v, fwd_s, cache = await _on_worker(work)
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
        hold.release()


async def encode(request: Request):
    from ..image_service import decode_image
    from ..image_steps import STATE, StepError
    from ..image_steps import encode as do_encode
    from ..image_wire import CONTENT_TYPE, read, write

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
    hold = await _acquire(fields.get("model"))
    try:
        served = _served(hold)
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
            got = await _on_worker(do_encode, served, text,
                                   [i[..., :3] for i in images], borrow,
                                   want_refs)
        finally:
            if chat_hold is not None:
                chat_hold.release()
    finally:
        hold.release()
    out = {"cond": got["cond"]}
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
        _check_size(hold, *_px(served, "vae", *lat.shape[2:]),
                    what="a VAE decode")
        img = await _on_worker(do_decode, served, lat,
                               bool(fields.get("alpha")))
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
        _check_size(hold, int(img.shape[2]), int(img.shape[1]),
                    what="a VAE encode", init=True)
        lat = await _on_worker(do_encode, served, img)
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
