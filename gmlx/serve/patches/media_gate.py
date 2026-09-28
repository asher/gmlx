"""Request media gate: media in a request arrives only as inline data.

A request that names an image, audio or video by a file path makes the
server open that file on the Mac, and one that names it by URL makes the
server fetch it. The server's API key is also handed to clients that run
with less access than the server has, such as a client in a launch
container, so neither may happen by default. The gate reads every JSON
request body before its route runs and answers 400 for any media reference
that is not a ``data:`` URI of the right kind. It never stats, opens or
fetches the reference. ``server.media_urls`` lets http(s) URLs through;
a file path is never accepted.

The same check runs again on the lists the generation path receives, in
:mod:`gmlx.serve.mem_preflight`, so a request shape the body walk does not
know still cannot reach a loader with a path.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import json
import re

from ._common import _error_content

_FLAG = "_kq_media_gate"

# Whether http(s) URLs are accepted, set once by install_media_gate.
_allow_urls = False

_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_VIDEO_PART_TYPES = frozenset({"video", "video_url", "input_video"})
_IMAGE_ROUTES = ("/images/generations", "/images/edits")
_SPEECH_ROUTE = "/audio/speech"
# Fields of the image routes that write files or load a model by path.
_IMAGE_PATH_FIELDS = ("output_path", "output_dir", "prompt_expansion_model")
# A Kokoro voice name, or several joined by commas. A name with a dot or a
# slash would load a voice file by path.
_VOICE = re.compile(r"[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*")
_MEDIA_TYPE = re.compile(r"[a-z]+/[A-Za-z0-9.+-]+")
_URL_HINT = "Set server.media_urls to let the server fetch http(s) URLs."


class MediaRefused(ValueError):
    """A request names media in a form the server does not accept."""


def _is_url(value: str) -> bool:
    return value[:8].lower().startswith(("http://", "https://"))


def _data_uri(value: str, kind: str) -> bytes | None:
    """The decoded bytes of a ``data:<kind>/...;base64,`` URI, else None."""
    head, comma, payload = value.partition(",")
    if not comma or not head.lower().startswith(f"data:{kind}/"):
        return None
    params = head[5:].split(";")
    if not _MEDIA_TYPE.fullmatch(params[0]) or "base64" not in params[1:]:
        return None
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None


def _video_container(data: bytes) -> bool:
    """Whether ``data`` starts like an MP4 or QuickTime, Matroska or WebM,
    or AVI file. Text formats such as playlists make the video reader open
    other files and URLs, so they are refused."""
    return (data[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip")
            or data.startswith(b"\x1a\x45\xdf\xa3")
            or (data.startswith(b"RIFF") and data[8:12] == b"AVI "))


def check_image(value, field: str) -> None:
    if not isinstance(value, str):
        raise MediaRefused(f"{field} must be a string")
    if value.startswith("data:"):
        if _data_uri(value, "image") is None:
            raise MediaRefused(f"{field} must be a base64 data:image/... URI")
        return
    _refuse_reference(field, value, "data:image/...;base64,...")


def check_video(value, field: str) -> None:
    if not isinstance(value, str):
        raise MediaRefused(f"{field} must be a string")
    if value.startswith("data:"):
        data = _data_uri(value, "video")
        if data is None:
            raise MediaRefused(f"{field} must be a base64 data:video/... URI")
        if not _video_container(data):
            raise MediaRefused(f"{field} holds no MP4, QuickTime, Matroska, WebM or "
                               "AVI video")
        return
    _refuse_reference(field, value, "data:video/...;base64,...")


def check_audio_data(value, field: str) -> None:
    """``input_audio.data``: a ``data:audio/`` URI or bare base64. A value
    that looks like a path or a URL is read as one downstream, so a bare
    value must be valid base64 and must not start like a path."""
    if not isinstance(value, str):
        raise MediaRefused(f"{field} must be a string")
    text = value.strip()
    if text.startswith("data:"):
        if _data_uri(text, "audio") is None:
            raise MediaRefused(f"{field} must be a base64 data:audio/... URI")
        return
    if _is_url(text):
        _refuse_reference(field, text, "base64 audio")
        return
    refused = MediaRefused(f"{field} must be base64 audio or a data:audio/... URI. "
                           "This server does not open files named in a request.")
    # The reader takes a value that starts like a path as a path.
    if text.startswith(("/", "./", "../", "~", "file:")):
        raise refused
    try:
        base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise refused from None


def _refuse_reference(field: str, value: str, want: str) -> None:
    if _is_url(value):
        if _allow_urls:
            return
        raise MediaRefused(f"{field} names a URL, and this server takes media only "
                           f"as {want}. {_URL_HINT}")
    raise MediaRefused(f"{field} must be {want}. This server does not open files "
                       "named in a request.")


def _check_part(d: dict) -> None:
    kind = d.get("type")
    if kind in _IMAGE_PART_TYPES:
        if d.get("file_id") is not None:
            raise MediaRefused(f"{kind}.file_id is not supported; send the image as "
                               "a data: URI")
        ref = d.get("image_url")
        if isinstance(ref, dict):
            ref = ref.get("url")
        if ref is not None:
            check_image(ref, kind)
    elif kind == "image":
        # Anthropic image block, or an OpenAI-style {"type": "image", "image": ...}.
        if "image" in d:
            check_image(d["image"], "image")
        source = d.get("source")
        if isinstance(source, dict):
            stype = source.get("type")
            if stype == "base64":
                media = source.get("media_type") or "image/png"
                if not (isinstance(media, str) and media.startswith("image/")
                        and _MEDIA_TYPE.fullmatch(media)):
                    raise MediaRefused("image.source.media_type must be an image type")
            elif stype == "url":
                check_image(source.get("url"), "image.source.url")
            else:
                raise MediaRefused(f"image.source.type {stype!r} is not supported; "
                                   "use base64")
    elif kind in _VIDEO_PART_TYPES:
        for key in ("video", "video_url"):
            ref = d.get(key)
            if isinstance(ref, dict):
                ref = ref.get("url")
            if ref is not None:
                check_video(ref, key)
    elif kind == "input_audio":
        audio = d.get("input_audio")
        if isinstance(audio, dict) and audio.get("data") is not None:
            check_audio_data(audio["data"], "input_audio.data")


def check_body(body, path: str) -> None:
    """Raise :class:`MediaRefused` for a media reference the server does not
    accept anywhere in the JSON ``body`` of a request to ``path``."""
    if isinstance(body, dict):
        if path.endswith(_IMAGE_ROUTES):
            for key in _IMAGE_PATH_FIELDS:
                if body.get(key) is not None:
                    raise MediaRefused(f"{key} is not accepted by this server")
            if body.get("response_format") == "path":
                raise MediaRefused("response_format 'path' is not accepted by this "
                                   "server; use b64_json")
            if path.endswith("/images/edits") and "image" in body:
                refs = body["image"]
                for ref in refs if isinstance(refs, list) else [refs]:
                    check_image(ref, "image")
        if path.endswith(_SPEECH_ROUTE):
            voice = body.get("voice")
            if voice not in (None, "") and not (isinstance(voice, str)
                                               and _VOICE.fullmatch(voice.strip())):
                raise MediaRefused("voice must be a voice name, such as af_heart")
            if body.get("ref_audio") is not None:
                ref = body["ref_audio"]
                if not (isinstance(ref, str) and _data_uri(ref, "audio") is not None):
                    raise MediaRefused("ref_audio must be a base64 data:audio/... URI")
    # Iterative, so a deeply nested body cannot exhaust the stack.
    stack = [body]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("type"), str):
                _check_part(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def check_media_lists(images=None, audio=None, videos=None) -> None:
    """The same rules on the lists the generation path loads from. Decoded
    objects pass; a string must be an accepted reference."""
    for value in images or ():
        if isinstance(value, str):
            check_image(value, "image")
    for value in audio or ():
        if isinstance(value, str):
            _refuse_reference("audio", value, "base64 audio")
    for value in videos or ():
        if isinstance(value, str):
            check_video(value, "video")


def install_media_gate(allow_urls: bool = False, app=None) -> None:
    """Answer 400 for a request whose JSON body names media by a file path,
    or by a URL unless ``allow_urls``. Install it before the API-key
    middleware, so an unauthenticated request gets 401 and learns nothing
    from the gate. Idempotent."""
    global _allow_urls
    from fastapi.responses import JSONResponse

    _allow_urls = bool(allow_urls)
    if app is None:
        app = importlib.import_module("mlx_vlm.server.app").app
    if getattr(app.state, _FLAG, False):
        return

    async def _media_gate(request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            ct = request.headers.get("content-type", "").lower()
            if not ct.startswith("multipart/"):
                raw = await request.body()
                try:
                    body = json.loads(raw) if raw else None
                except (ValueError, RecursionError):
                    body = None               # the route reports a bad body
                try:
                    check_body(body, request.url.path)
                except MediaRefused as e:
                    return JSONResponse(status_code=400, content=_error_content(
                        request.url.path, 400, "invalid_request_error", str(e)))
        return await call_next(request)

    app.middleware_stack = None          # allow install after a stack build
    app.middleware("http")(_media_gate)
    setattr(app.state, _FLAG, True)
