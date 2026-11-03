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

    def revise(self, model, text, requested, has_refs):
        """The upsampler's caption of ``text``, or None to encode it as
        typed."""
        from ..image_upsample import revise

        return revise(model, text, requested, has_refs, resolve=self.resolve)

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


def _video(served) -> bool:
    from ..video_service import ServedVideo

    return isinstance(served, ServedVideo)


def _names_video(model) -> bool:
    """Whether a model field names a configured video model."""
    from ..image_steps import is_video

    try:
        return is_video(SETTINGS.host.resolve(model)[1])
    except Exception:  # noqa: BLE001 - the acquire refuses what fails here
        return False


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


def _skip(served, raw) -> tuple:
    """The blocks a step's rows leave out, zero-based, from its ``skip``
    field."""
    from ..image_steps import StepError

    if not isinstance(raw, list) or any(
            isinstance(i, bool) or not isinstance(i, int) for i in raw):
        raise StepError(400, "skip is a list of block indices")
    if not raw:
        return ()
    dit = served.pipe.dit
    title = served.pipe.family.title
    if not getattr(dit, "skips_blocks", False):
        raise StepError(400, f"the {title} DiT skips no blocks")
    n = len(dit.layers)
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
        if ref_cache and not hasattr(served.pipe.dit, "extract"):
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
        try:
            table = _table(served, fields.get("patch_table"))
        except Missing as m:
            missing += m.extra["missing"]
        if missing:
            raise Missing(missing)
        import numpy as np

        ts = np.broadcast_to(np.asarray(t_w.numpy(), np.float32).reshape(-1),
                             (len(rows),))
        # In float64, so a scaled timestep gives back its time in float32.
        sig = [conv.scale * float(t) + conv.shift for t in ts]
        targets = None
        if getattr(served.pipe.dit.cfg, "flow_map", False):
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

        def work():
            # Every MLX op of a step runs here: MLX streams belong to the
            # thread that made them.
            import mlx.core as mx

            from gmlx.image import api, memfit

            from ..image_service import size_fits
            from ..image_steps import _host, conditioning

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
                          conds=conds, patch_table=table,
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
                v, fwd_s, cache = run_step(served, x, sig, conds, table, run,
                                           gen, targets, conv.time,
                                           ref_cache, skip)
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
    frames = (t - 2) // 5 * 17 + 5 if t >= 2 else 0
    width, height = w * stride, h * stride
    try:
        want = fam.latent_shapes(frames, width, height)
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
                audio: int | None = None) -> None:
    """Refuse a step or a decode of a clip that does not fit beside the
    models loaded. The DiT stays loaded through each, and each is priced
    as the image routes price theirs: the DiT, the larger of its
    configured LoRAs and the pairs and patch tables it holds, the caches
    the server holds and the phase's work. A step under a patch table
    with patches, ``table``, runs a LoRA. ``audio`` is the audio decode's
    latents."""
    from ..image_steps import StepError
    from ..video_service import held_bytes, plan_clip

    budget_fn = getattr(SETTINGS.host, "budget", None)
    budget = budget_fn(hold) if budget_fn is not None else None
    if budget is None:
        return
    pipe = served.pipe
    tid = pipe.default_table if table is None else table
    live = bool(pipe.patch_tables.get(tid, {}).get("entries"))
    plan = plan_clip(served.plan, frames=frames, width=width, height=height,
                     text=text, budget=budget, decode=phase != "denoise",
                     audio=audio, live_lora=live and phase == "denoise")
    adapters, caches = held_bytes(served)
    held = (plan.terms["denoise"]["dit"]
            + max(plan.terms["denoise"]["lora"], adapters) + caches)
    work = (plan.terms["denoise"]["step"] if phase == "denoise"
            else plan.phases[phase])
    if held + work > budget:
        clip = (f"{audio} audio latents" if phase == "audio"
                else f"{width}x{height} at {frames} frames")
        raise StepError(400, f"{what} of {clip} does not fit in memory "
                        "beside the models now loaded")


async def _av_step(t0, tensors, fields, run):
    """A step of a model with a video and an audio stream (``AV_FAMILIES``):
    the latents as ComfyUI's sampler carries them, one row."""
    import numpy as np

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
    missing = [h for h in hashes if h not in STATE.tensors]
    model = fields.get("model") or (run.model if run else None)
    hold = await _acquire(model)
    try:
        served = _served(hold)
        if not _video(served) or served.plan.family not in AV_FAMILIES:
            raise StepError(400, f"{served.spec.id!r} is no video model; "
                            "its step takes x")
        frames, width, height = _clip_of(served.pipe.video_family,
                                         v_w.shape, a_w.shape)
        table = None
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
        _check_clip(hold, served, frames=frames, width=width, height=height,
                    text=n_text, phase="denoise", what="a step", table=table)
        ts = float(np.asarray(t_w.numpy(), np.float32).reshape(-1)[0])
        gen = run.gen if run is not None else 0

        def work():
            import mlx.core as mx

            from gmlx.image.pacing import paced

            from ..image_steps import StepInterrupted

            vs.ensure_dit(served)
            feats = _f32(held)
            if feats.ndim == 2:
                feats = feats[None]
            dim = int(served.pipe.dit.cfg.text_dim)
            if feats.ndim != 3 or feats.shape[0] != 1 or \
                    feats.shape[1] == 0 or feats.shape[2] != dim:
                raise StepError(400, f"a row's cond is (1, L, {dim}) encoder "
                                f"features, not {tuple(feats.shape)}")
            pace = step_pacer(run, gen)

            def pacer(arrays):
                pace(arrays)
                mx.eval(*arrays)

            t1 = time.perf_counter()
            with paced(pacer):
                vv, va, hit = vs.av_step(
                    served, v_w.mx(), a_w.mx(), ts, shifts, scale, feats,
                    _cond_key(hashes[0]), table)
                mx.eval(vv, va)
            if run is not None and (run.gen != gen or run.ended):
                raise StepInterrupted()
            return _host(vv), _host(va), time.perf_counter() - t1, hit

        vv, va, fwd_s, hit = await _on_worker(work)
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
        hold.release()


async def encode(request: Request):
    from ..image_service import ImageRequestError, decode_image
    from ..image_steps import StepError
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
    revise = getattr(SETTINGS.host, "revise", None)
    revised = None
    if _names_video(fields.get("model")):
        # A video family has no upsampler.
        revise = None
    if revise is not None:
        try:
            revised = await asyncio.to_thread(
                revise, fields.get("model"), text, requested, bool(images))
        except ImageRequestError as e:
            raise StepError(e.status, str(e)) from None
    hold = await _acquire(fields.get("model"))
    try:
        served = _served(hold)
        if _video(served):
            from ..video_service import encode as video_encode

            feats = await _on_worker(video_encode, served, text, images)
            got = {"cond": feats, "refs": [], "image_slots": []}
            return _held_encode(got, None)
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
            got = await _on_worker(do_encode, served, revised or text,
                                   [i[..., :3] for i in images], borrow,
                                   want_refs, *typed)
        finally:
            if chat_hold is not None:
                chat_hold.release()
    finally:
        hold.release()
    return _held_encode(got, revised)


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
        fields_out["cond_typed"] = hashes["cond_typed"]
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
                    what="a VAE decode", phase="decode")
        _image_only(served, "vae/decode")
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
        _image_only(served, "vae/encode")
        _check_size(hold, int(img.shape[2]), int(img.shape[1]),
                    what="a VAE encode", init=True, phase="init")
        lat = await _on_worker(do_encode, served, img)
    finally:
        hold.release()
    return _container({"latents": lat})


async def _video_decode_hold(request: Request, name: str, ndim: int,
                             shape: str):
    from ..image_steps import StepError

    tensors, fields = await _wire(request, MAX_BODY_BYTES)
    lat = tensors.get("latents")
    if lat is None or len(lat.shape) != ndim or lat.shape[0] != 1:
        raise StepError(400, f"{name} takes the latents of one clip {shape}")
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

    lat, hold, served = await _video_decode_hold(
        request, "vae/video/decode", 5, "(1, C, T, h, w)")
    try:
        frames, width, height = _clip_of(served.pipe.video_family, lat.shape)
        _check_clip(hold, served, frames=frames, width=width, height=height,
                    text=1, phase="decode", what="a video decode")
        img = await _on_worker(lambda: decode_video(served, lat.mx()))
    finally:
        hold.release()
    return _container({"frames": img})


async def audio_decode(request: Request):
    """Model-space audio latents (1, 32, 2, T) to the waveform
    (1, 2, samples) f32."""
    from ..image_steps import COMFY, StepError
    from ..video_service import decode_audio

    lat, hold, served = await _video_decode_hold(
        request, "vae/audio/decode", 4, "(1, C, 2, T)")
    try:
        row = COMFY[served.pipe.video_family.name]["audio"]
        if tuple(lat.shape[1:3]) != (row["channels"], row["stereo"]):
            raise StepError(400, f"audio latents are (1, {row['channels']}, "
                            f"{row['stereo']}, T), not {list(lat.shape)}")
        width, height = served.plan.size
        _check_clip(hold, served, frames=served.plan.frames, width=width,
                    height=height, text=1, phase="audio",
                    what="an audio decode", audio=int(lat.shape[3]))
        img = await _on_worker(lambda: decode_audio(served, lat.mx()))
    finally:
        hold.release()
    return _container({"waveform": img})


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
