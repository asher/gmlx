"""The OpenAI video routes: POST /v1/videos queues a job, GET
/v1/videos/{id} reads it, GET /v1/videos/{id}/content gives its MP4, and
DELETE /v1/videos/{id} cancels it or removes it.

The media gate leaves a POST body unread (``body_intake``). A JSON body
is priced against the bytes free beside the models loaded, at the copies
its read and parse hold, before it is read, and takes its keyframes as
data URLs; a multipart form streams its
files to the job store, within the room it reserves there before its
first byte, and its text fields are priced together against free
memory. The route then estimates the
clip from the entry's headers against the budget the pinned models
leave, with no pool lock, and books nothing: the job's own checks on
the video worker decide its memory. See :mod:`gmlx.serve.video_jobs`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import importlib
import json
import logging
import os
import threading
from types import SimpleNamespace

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

import gmlx.serve.bridge_vlm as serving

from ._common import _error_content, _remove_routes
from .body_intake import (
    BodyRefused,
    _mib,
    arriving_note,
    check_media,
    read_form_to_disk,
    read_in_ram,
)

_log = logging.getLogger(__name__)

PATH = "/v1/videos"
FIELDS = ("prompt", "model", "seconds", "size", "seed", "prompt_upsampling",
          "first_frame", "last_frame", "input_reference", "guides",
          "guide_no_audio", "references")
KEYFRAMES = ("first_frame", "last_frame")
# A JSON body is held about four times while it is read: the body, the
# gate's json.loads, the route's, and a data-URL keyframe's decode,
# measured at 4.0. The post-V6 parse change removes the copies. Two of
# them are strs (the gate's json.loads and the route's), which widen in a
# body that is not ASCII, and the values it parses to add
# body_intake.JSON_VALUE_BYTES each.
JSON_COPIES = 4.25
JSON_STR_COPIES = 2
_KINDS = {"video": "--guide", "audio": "--guide-audio"}
# The entry plans and encoder tokenizers the estimate reads, by entry.
_PLANS: dict = {}
_TOKENIZERS: dict = {}
_lock = threading.Lock()


class Refused(Exception):
    def __init__(self, status: int, message: str, param: str | None = None,
                 code: str | None = None, err_type: str | None = None,
                 headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.param = param
        self.code = code
        self.err_type = err_type or ("server_error" if status >= 500
                                     else "invalid_request_error")
        self.headers = headers or {}


def _store():
    from ..video_jobs import STORE

    return STORE


class _Post:
    """The store calls of one POST, each in its thread under this POST's
    lock. Once :meth:`abandon` ran, a call still to come does nothing,
    whatever order the threads take, so a begin, take or submit still in
    its thread when the request ended leaves nothing behind."""

    def __init__(self, store, jid: str):
        self.store = store
        self.jid = jid
        self._lock = threading.Lock()
        self._ended = False

    def _call(self, fn, *args, gone=None):
        with self._lock:
            if self._ended:
                return gone
            return fn(self.jid, *args)

    def begin(self) -> None:
        self._call(self.store.begin)

    def take(self, n: int | None) -> tuple[bool, int, int]:
        return self._call(self.store.take, n, gone=(False, 0, 0))

    def settle(self) -> None:
        self._call(self.store.settle)

    def submit(self, record: dict, params: dict):
        return self._call(self.store.submit, record, params)

    def abandon(self) -> None:
        with self._lock:
            self._ended = True
            self.store.abandon(self.jid)


def _answer(e: Refused) -> JSONResponse:
    return JSONResponse(status_code=e.status, headers=e.headers,
                        content=_error_content(PATH, e.status, e.err_type,
                                               str(e), param=e.param,
                                               code=e.code))


def _from_http(e: HTTPException) -> Refused:
    detail = e.detail
    err = detail.get("error", {}) if isinstance(detail, dict) else {}
    if not isinstance(err, dict):
        err = {}
    return Refused(e.status_code, str(err.get("message") or detail),
                   code=err.get("code"), err_type=err.get("type"),
                   headers=dict(e.headers or {}))


def _bool(fields: dict, name: str) -> bool | None:
    v = fields.get(name)
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    raise Refused(400, f"{name} is true or false", name)


def _int(fields: dict, name: str) -> int | None:
    v = fields.get(name)
    if v is None:
        return None
    if isinstance(v, bool):
        raise Refused(400, f"{name} is a whole number", name)
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise Refused(400, f"{name} is a whole number", name) from None
    if n < 0 or n >= 2 ** 32 or (isinstance(v, float) and v != n):
        raise Refused(400, f"{name} is a whole number from 0 to 2^32 - 1",
                      name)
    return n


def _data_url(value, name: str, folder: str, take) -> str:
    """Write the image of the data URL ``value`` to ``folder``, within the
    room ``take(n)`` reserves in the job store."""
    if not isinstance(value, str) or not value.startswith("data:image/") \
            or ";base64," not in value:
        raise Refused(400, f"{name} is an image as a data URL "
                      "(data:image/png;base64,...) in a JSON body, or a "
                      "file in a multipart form", name)
    try:
        data = base64.b64decode(value.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError):
        raise Refused(400, f"{name} is not base64", name) from None
    if take is not None:
        ok, left, arriving = take(len(data))
        if not ok:
            raise Refused(413, f"the keyframe {name} is "
                          f"{_mib(len(data), True)}, and the video job "
                          f"store has {_mib(left)} of room left"
                          f"{arriving_note(arriving)}; send a smaller "
                          "image, delete finished jobs, or raise "
                          "server.video.store_max_bytes", name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _pixels(path: str, name: str) -> bool:
    """Whether ``path`` is an image, refusing one past the media pixel
    limit; a video is no image."""
    from PIL import Image

    from .media_gate import IMAGE_FORMATS, MEDIA_MAX_PIXELS

    try:
        with Image.open(path, formats=IMAGE_FORMATS) as im:
            w, h = im.size
    except Image.DecompressionBombError:
        raise Refused(400, f"{name} is over the {MEDIA_MAX_PIXELS} pixels "
                      "the server decodes", name) from None
    except Exception:  # noqa: BLE001 - Pillow raises many types for bad data
        return False
    if w * h > MEDIA_MAX_PIXELS:
        raise Refused(400, f"{name} is {w}x{h}, over the {MEDIA_MAX_PIXELS} "
                      "pixels the server decodes", name)
    return True


def _model(fields: dict) -> str:
    model = fields.get("model")
    ids = serving.video_model_ids()
    if model is None and len(ids) == 1:
        return ids[0]
    if not isinstance(model, str) or not model:
        raise Refused(400, "model names a video model of the server's "
                      "config: " + ", ".join(repr(i) for i in ids), "model")
    return model


def _guides(fields: dict, files: dict) -> list:
    raw = fields.get("guides")
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raise Refused(400, "guides is a JSON list", "guides") from None
    if not isinstance(raw, list):
        raise Refused(400, "guides is a JSON list", "guides")
    out = []
    for i, g in enumerate(raw):
        if not isinstance(g, dict) or set(g) - {"file", "frame", "kind"}:
            raise Refused(400, f"guides[{i}] is an object of file, frame and "
                          "kind", "guides")
        name, kind = g.get("file"), g.get("kind", "video")
        if kind not in _KINDS:
            raise Refused(400, f"guides[{i}].kind is video or audio",
                          "guides")
        if name not in files or name in KEYFRAMES or name == "input_reference":
            raise Refused(400, f"guides[{i}].file names a file field of the "
                          "form", "guides")
        frame = g.get("frame", 0)
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise Refused(400, f"guides[{i}].frame is a frame of the clip, "
                          "from its end when below 0", "guides")
        out.append((_KINDS[kind], files[name], frame,
                    f"guides[{i}] ({name})"))
    return out


def _entry_plan(spec):
    from ..video_service import entry_plan

    key = (spec.id, spec.load_signature())
    with _lock:
        got = _PLANS.get(key)
    if got is None:
        got = entry_plan(spec)
        with _lock:
            _PLANS[key] = got
    return got


def _tokenizer(fam, spec):
    from gmlx.video.encoder import header_tokenizer

    key = os.path.expanduser(spec.encoder)
    with _lock:
        got = _TOKENIZERS.get(key)
    if got is None:
        got = header_tokenizer(fam, key)
        with _lock:
            _TOKENIZERS[key] = got
    return got


def estimate(spec, fam, plan, s: dict, prompt: str, sizes, rewrites: bool,
             path: str) -> None:
    """The POST's one estimate: the clip at the job's shape on the entry's
    plan, at the typed prompt's rows, or the keyframes' alone when the
    upsampler rewrites it, a lower bound, against the budget less the
    pinned models beside the entry, from the pool's figures with no
    lock. Over it, 413."""
    from gmlx.video import memfit

    from ..capacity import overcommit
    from ..residency import _installed_pool
    from ..video_service import plan_clip

    pool = _installed_pool()
    if pool is None or overcommit():
        return
    f = pool.figures
    if not f.budget:
        return
    pinned = sum(e.footprint for e in f.entries
                 if e.pinned and str(e.model_path) != str(path))
    rows = memfit.prompt_rows(fam, "" if rewrites else prompt,
                              _tokenizer(fam, spec), sizes)
    blocks = memfit.keyframe_blocks(fam, sizes) if sizes else ()
    got = plan_clip(plan, frames=s["frames"], width=s["width"],
                    height=s["height"], text=rows, blocks=blocks,
                    budget=f.budget - pinned)
    if got.over:
        raise Refused(413, f"a {s['width']}x{s['height']} clip of "
                      f"{s['frames']} frames on {spec.id!r} does not fit in "
                      f"memory: {got.describe()}; {got.hint()}", "size")


def prepare(fields: dict, files: dict, folder: str, ffmpeg: str,
            take=None) -> tuple:
    """``(record, params)`` of a job from the request's fields and files,
    or Refused. A data URL keyframe goes to ``folder`` within the room
    ``take(n)`` reserves in the job store. Fields and files the route
    does not read are named in a warning, as the images API does."""
    from gmlx.commands import video as cmd
    from gmlx.video.registry import get_family

    from ..image_service import ImageRequestError
    from ..image_upsample import upsampler_for
    from ..residency import _http_from_resolver_error

    refs = fields.get("references")
    if refs not in (None, "", [], "[]"):
        raise Refused(400, "references asks for MiniMax-H3's Ref2VA mode, "
                      "which the server does not run; send the job without "
                      "references", "references")
    prompt = fields.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise Refused(400, "prompt is the text of the clip", "prompt")
    model = _model(fields)
    try:
        path, spec = serving.resolve_video_request(model)
    except (serving.ModelNotFound, serving.ModelFileMissing) as e:
        http = _http_from_resolver_error(e)
        if http is None:
            raise
        raise _from_http(http) from None
    from gmlx.video.fingerprint import VideoModelError
    from gmlx.video.memfit import VideoMemoryError

    try:
        plan = _entry_plan(spec)
    except (VideoModelError, VideoMemoryError) as e:
        raise Refused(400, str(e), "model", code="video_model_mismatch") \
            from None
    fam = get_family(plan.family)
    try:
        seconds = (None if fields.get("seconds") is None
                   else cmd._seconds(str(fields["seconds"])))
    except argparse.ArgumentTypeError as e:
        raise Refused(400, str(e).replace("--seconds", "seconds"),
                      "seconds") from None
    try:
        size = (None if fields.get("size") is None
                else cmd._size(str(fields["size"])))
    except argparse.ArgumentTypeError as e:
        raise Refused(400, f"size: {e}", "size") from None
    seed = _int(fields, "seed")
    try:
        s = cmd._settings(SimpleNamespace(
            entry_defaults=spec.defaults or {}, frames=None, seconds=seconds,
            size=size, shift=None, steps=None, seed=seed), fam, plan.variant)
    except cmd._Usage as e:
        raise Refused(400, str(e)) from None
    # input_reference is the OpenAI field of the first frame.
    sent = {k: (files.get(k), fields.get(k)) for k in
            ("first_frame", "input_reference", "last_frame")}
    if any(sent["first_frame"]) and any(sent["input_reference"]):
        raise Refused(400, "first_frame and input_reference both name the "
                      "first frame; send one", "input_reference")
    keys = {}
    for name, alias in (("first_frame", "input_reference"),
                        ("last_frame", "last_frame")):
        file, value = sent[name] if any(sent[name]) else sent[alias]
        keys[name] = file if file is not None else (
            None if value is None else _data_url(value, name, folder, take))
    for name, v in keys.items():
        if v is not None and not _pixels(v, name):
            raise Refused(400, f"{name} is no image the server reads", name)
    if isinstance(fields.get("guides"), list) and fields["guides"]:
        raise Refused(400, "guides take files: send the job as a multipart "
                      "form", "guides")
    guides = _guides(fields, files)
    for g in guides:
        _pixels(g[1], g[3])
    row = fam.variant_row(plan.variant)
    n = sum(v is not None for v in keys.values()) + len(guides)
    if n and row is not None and row.vsa_sparsity:
        from gmlx.video.families.h3 import vsa

        raise Refused(400, str(vsa.anchor_refusal(n)))
    requested = _bool(fields, "prompt_upsampling")
    try:
        rewrites = upsampler_for(spec, fam, requested, False) is not None
    except ImageRequestError as e:
        raise Refused(e.status, str(e), e.param) from None
    sizes = [(s["width"], s["height"])] * sum(v is not None
                                              for v in keys.values())
    estimate(spec, fam, plan, s, prompt, sizes, rewrites, path)
    from .api_contract import warn_ignored_fields

    used = set(KEYFRAMES) | {"input_reference"} | {
        os.path.basename(g[1]) for g in guides}
    warn_ignored_fields(PATH, (set(fields) - set(FIELDS))
                        | (set(files) - used))
    from gmlx.video.families.h3 import family as hf

    record = {"model": spec.id, "prompt": prompt,
              "seconds": f"{s['frames'] / hf.FPS:.2f}",
              "size": f"{s['width']}x{s['height']}",
              "gmlx": {"frames": s["frames"], "seed": s["seed"],
                       "steps": s["steps"], "phase": "queued", "step": 0,
                       "revised_prompt": None}}
    params = {"model": spec.id, "prompt": prompt, "settings": s,
              "first_frame": keys["first_frame"],
              "last_frame": keys["last_frame"], "guides": guides,
              "guide_no_audio": bool(_bool(fields, "guide_no_audio")),
              "upsampling": requested, "rewrites": rewrites,
              "ffmpeg": ffmpeg}
    return record, params


async def create(request: Request):
    from .. import media_programs
    from ..video_jobs import QueueFull

    store = _store()
    jid = store.new_id()
    post = _Post(store, jid)
    folder = store.inputs(jid)
    # Refused before the body is read: the answer ends the connection, so
    # the server does not drain a body it will not use.
    close = {"Connection": "close"}
    try:
        ffmpeg = media_programs.find("ffmpeg")
        if ffmpeg is None:
            raise Refused(501, media_programs.problem("ffmpeg") or
                          "the server cannot run ffmpeg", code="ffmpeg_missing",
                          headers=close)
        full, retry = await asyncio.to_thread(
            lambda: (store.full(), store.retry_after()))
        if full:
            raise Refused(429, "the video job queue is full; retry later",
                          code="queue_full", headers={
                              "Retry-After": str(retry), **close})
        # From here the sweep leaves this job's inputs, and its bytes
        # count in the room, until submit or abandon.
        await asyncio.to_thread(post.begin)
        take = post.take

        async def take_async(n):
            return await asyncio.to_thread(take, n)

        ctype = request.headers.get("content-type", "").split(";")[0].strip()
        try:
            if ctype == "multipart/form-data":
                pairs, files = await read_form_to_disk(request, folder,
                                                       take_async)
                await asyncio.to_thread(post.settle)
                fields = dict(pairs)
            else:
                raw = await read_in_ram(request, copies=JSON_COPIES,
                                        str_copies=JSON_STR_COPIES)
                await check_media(request, raw)
                try:
                    fields = json.loads(raw or b"{}")
                except ValueError:
                    raise Refused(400, "the body is no JSON object") from None
                if not isinstance(fields, dict):
                    raise Refused(400, "the body is no JSON object")
                files = {}
        except BodyRefused as e:
            raise Refused(e.status, str(e), e.param,
                          headers=e.headers) from None
        record, params = await asyncio.to_thread(prepare, fields, files,
                                                 folder, ffmpeg, take)
        try:
            job = await asyncio.to_thread(post.submit, record, params)
        except QueueFull as e:
            raise Refused(429, "the video job queue is full; retry later",
                          code="queue_full",
                          headers={"Retry-After": str(e.retry_after)}) \
                from None
    except Refused as e:
        await asyncio.to_thread(post.abandon)
        return _answer(e)
    except BaseException:
        # The request is ending (a client that left, a cancel): the
        # abandon runs in the executor, and nothing waits for it.
        asyncio.get_running_loop().run_in_executor(None, post.abandon)
        raise
    rec = await asyncio.to_thread(store.get, job.id)
    return JSONResponse(status_code=200, content=rec or job.record)


def _missing(jid: str) -> JSONResponse:
    return _answer(Refused(404, f"no video job {jid!r}; a job is gone once "
                           "it expires or is deleted", "video_id",
                           code="not_found"))


async def retrieve(video_id: str):
    rec = await asyncio.to_thread(_store().get, video_id)
    return _missing(video_id) if rec is None else JSONResponse(rec)


async def content(video_id: str):
    store = _store()
    rec, path = await asyncio.to_thread(
        lambda: (store.get(video_id), store.content(video_id)))
    if rec is None:
        return _missing(video_id)
    if path is None:
        return _answer(Refused(409, f"video job {video_id!r} is "
                               f"{rec['status']}; its content is ready "
                               "once it is completed", "video_id",
                               code="not_ready"))
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{video_id}.mp4")


async def delete(video_id: str):
    got = await asyncio.to_thread(_store().delete, video_id)
    return _missing(video_id) if got is None else JSONResponse(got)


def _app():
    return importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app


ROUTES = ((PATH, create, "POST"), (PATH + "/{video_id}", retrieve, "GET"),
          (PATH + "/{video_id}/content", content, "GET"),
          (PATH + "/{video_id}", delete, "DELETE"))


def install_video_routes(cfg, app=None) -> None:
    """Serve the video routes when the config sets up a video model, and
    remove them when it does not. Each call takes the server.video
    settings and sweeps the store; a reload calls it again."""
    app = app or _app()
    _remove_routes(app, *{p for p, _fn, _m in ROUTES})
    with _lock:
        _PLANS.clear()
        _TOKENIZERS.clear()
    if not getattr(cfg, "video_models", None):
        return
    _store().configure(cfg)
    for p, fn, method in ROUTES:
        app.add_api_route(p, fn, methods=[method], include_in_schema=False)
