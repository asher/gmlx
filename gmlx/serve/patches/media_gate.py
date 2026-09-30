"""Request media gate: media in a request arrives inline, or from the
server's own media folder.

A request that names an image, audio or video by a file path makes the
server open that file on the Mac, and one that names it by URL makes the
server fetch it. A client can run with less access than the server has,
such as a client in a launch container, which gets a placeholder key and
reaches the server through a scoped session socket. So neither may happen
by default. The gate reads every request body before its route runs and
answers 400 for any media reference that is not a ``data:`` URI of the
right kind, or an absolute path inside the media folder
(``$XDG_CACHE_HOME/gmlx/media``). A request on a launch session socket may
not name the media folder either. The gate never stats, opens or fetches the
reference. ``server.media_urls`` lets http(s) URLs through.

The gate is the fast, clearly worded refusal. The boundary is in
:mod:`gmlx.serve.media_sinks`, which checks every reference again in the
functions that open it, whatever route or body shape carried it.

A route reads its body with ``json.loads`` whatever the Content-Type says,
so the gate parses every body the same way. A form body reaches only the
routes that take an upload, and the gate checks its text fields with the
parser those routes use.

The same checks run again where the loaders are called: on the lists the
generation path receives, in :mod:`gmlx.serve.mem_preflight`, and in the
image generation and edit calls, which :func:`install_media_gate` wraps. A
request shape the body walk does not know therefore still cannot reach a
loader with a path or write a file.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import contextvars
import dataclasses
import importlib
import io
import json
import os
import re
import shutil
import tempfile

from gmlx.safe_path import parts_below

from ._common import SESSION_SCOPE_KEY, _error_content

_FLAG = "_kq_media_gate"

# The largest request body the server reads.
BODY_MAX_BYTES = 64 << 20
# The largest form the audio upload routes read, so that a long recording
# fits.
UPLOAD_MAX_BYTES = 1 << 30
# The same two limits on a launch session socket. A container client sends
# text and inline images, such as a 20 MB clipboard image, and the session's
# connection cap times these bounds the memory it can make the server hold.
SESSION_BODY_MAX_BYTES = 32 << 20
SESSION_UPLOAD_MAX_BYTES = 64 << 20
# The largest image, audio clip or video one inline reference or one fetch
# may hold, decoded.
MEDIA_MAX_BYTES = 32 << 20
# The most pixels an image may decode to.
MEDIA_MAX_PIXELS = 64 << 20

# Whether http(s) URLs are accepted, set once by install_media_gate.
_allow_urls = False
# The media folder, as shown in messages, and every spelling of it a path
# may start with. Empty until the server creates the folder.
_media_root: str | None = None
_media_root_forms: tuple[str, ...] = ()
# True while the gate checks a request that came through a launch session
# socket, which takes media only inline: no file and no URL.
_inline_only: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gmlx_media_inline_only", default=False)

_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_VIDEO_PART_TYPES = frozenset({"video", "video_url", "input_video"})
_IMAGE_ROUTES = ("/images/generations", "/images/edits")
_SPEECH_ROUTE = "/audio/speech"
# The routes that read a multipart form with a file part. Every other route
# reads JSON, so a form body there is refused.
_UPLOAD_ROUTES = ("/audio/transcriptions", "/audio/translations")
_UPLOAD_PATHS = frozenset(p for r in _UPLOAD_ROUTES for p in (r, "/v1" + r))
# The routes that carry a conversation, whose body grows with its history.
_CONVERSATION_ROUTES = ("/chat/completions", "/messages", "/messages/count_tokens",
                        "/responses", "/responses/input_tokens")
_CONVERSATION_PATHS = frozenset(p for r in _CONVERSATION_ROUTES for p in (r, "/v1" + r))
_MULTIPART = b"multipart/form-data"
_URLENCODED = b"application/x-www-form-urlencoded"
# Fields of the image routes that write files or load a model by path.
_IMAGE_PATH_FIELDS = ("output_path", "output_dir", "prompt_expansion_model")
# A Kokoro voice name, or several joined by commas. A name with a dot or a
# slash would load a voice file by path.
_VOICE = re.compile(r"[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*")
_MEDIA_TYPE = re.compile(r"[a-z]+/[A-Za-z0-9.+-]+")
_URL_HINT = ("Set server.media_urls to let the server fetch http(s) URLs, then run "
             "gmlx restart.")
_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*)://")
# The image loaders open files only, so an inline reference image of an
# edit is written to a private temporary file for the call.
_EDIT_IMAGE_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif"}


class MediaRefused(ValueError):
    """A request names media in a form the server does not accept."""


def set_media_root(root: str | None, forms: tuple[str, ...] = ()) -> None:
    """Accept absolute paths inside ``root``, spelled as any of ``forms``,
    or refuse every path with None."""
    global _media_root, _media_root_forms
    _media_root = root
    _media_root_forms = tuple(dict.fromkeys(forms or ((root,) if root else ())))


def media_root() -> str | None:
    return _media_root


def _is_url(value: str) -> bool:
    return value[:8].lower().startswith(("http://", "https://"))


def media_parts(value: str) -> tuple[str, list[str]] | None:
    """The media folder spelling ``value`` starts with and the components
    below it, for an absolute path or a ``file://`` URL. Nothing is
    resolved, and a ``.`` or ``..`` component fails the match."""
    path = value[7:] if value[:7].lower() == "file://" else value
    if not path.startswith("/"):
        return None
    for form in _media_root_forms:
        parts = parts_below(path, form)
        if parts and not any(p in (".", "..") for p in parts):
            return form, parts
    return None


def too_many_pixels(field: str, width: int | None = None,
                    height: int | None = None) -> MediaRefused:
    size = f"{width}x{height} pixels, " if width is not None else ""
    return MediaRefused(f"{field} is {size}over the limit of {MEDIA_MAX_PIXELS} "
                        "pixels. Send a smaller image.")


def _too_large(field: str) -> MediaRefused:
    return MediaRefused(f"{field} is larger than the {MEDIA_MAX_BYTES >> 20} MiB limit "
                        "for one image, audio clip or video. Send a smaller one.")


def _data_uri(value: str, kind: str, field: str = "media") -> bytes | None:
    """The decoded bytes of a ``data:<kind>/...;base64,`` URI, else None.
    A payload over :data:`MEDIA_MAX_BYTES` is refused before it is
    decoded."""
    head, comma, payload = value.partition(",")
    if not comma or not head.lower().startswith(f"data:{kind}/"):
        return None
    params = head[5:].split(";")
    if not _MEDIA_TYPE.fullmatch(params[0]) or "base64" not in params[1:]:
        return None
    if len(payload) // 4 * 3 > MEDIA_MAX_BYTES + 3:
        raise _too_large(field)
    try:
        data = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None
    if len(data) > MEDIA_MAX_BYTES:
        raise _too_large(field)
    return data


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
        if _data_uri(value, "image", field) is None:
            raise MediaRefused(f"{field} must be a base64 data:image/... URI")
        return
    _refuse_reference(field, value, "a base64 data:image/... URI")


def check_video(value, field: str) -> None:
    if not isinstance(value, str):
        raise MediaRefused(f"{field} must be a string")
    if value.startswith("data:"):
        data = _data_uri(value, "video", field)
        if data is None:
            raise MediaRefused(f"{field} must be a base64 data:video/... URI")
        if not _video_container(data):
            raise MediaRefused(f"{field} holds no MP4, QuickTime, Matroska, WebM or "
                               "AVI video")
        return
    _refuse_reference(field, value, "a base64 data:video/... URI")


def check_audio_data(value, field: str) -> None:
    """``input_audio.data``: a ``data:audio/`` URI or bare base64. A value
    that looks like a path or a URL is read as one downstream, so a bare
    value must be valid base64 and must not start like a path."""
    if not isinstance(value, str):
        raise MediaRefused(f"{field} must be a string")
    text = value.strip()
    want = "base64 audio or a data:audio/... URI"
    if text.startswith("data:"):
        if _data_uri(text, "audio", field) is None:
            raise MediaRefused(f"{field} must be a base64 data:audio/... URI")
        return
    # The reader takes a value that starts like a path or a URL as one.
    if _is_url(text) or text.startswith(("/", "./", "../", "~", "file:")):
        _refuse_reference(field, text, want)
        return
    if len(text) // 4 * 3 > MEDIA_MAX_BYTES + 3:
        raise _too_large(field)
    try:
        base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise MediaRefused(f"{field} must be {want}") from None


def reference_refusal(field: str, value: str, want: str) -> MediaRefused:
    """The refusal for a reference that is neither inline data, an allowed
    URL nor a file in the media folder."""
    scheme = _SCHEME.match(value)
    if scheme and scheme.group(1).lower() != "file":
        return MediaRefused(f"{field} names a {scheme.group(1).lower()}:// URL, and this "
                            f"server takes media only as {want}.")
    return path_refusal(field, want)


def path_refusal(field: str, want: str) -> MediaRefused:
    """The refusal for a file path the server does not open."""
    if _inline_only.get():
        return MediaRefused(f"{field} names a file, and a launch container session "
                            f"takes media only inline, as {want}.")
    if _media_root is None:
        return MediaRefused(f"{field} must be {want}. This server does not open "
                            "files named in a request.")
    return MediaRefused(
        f"{field} names a file outside {_media_root}, the only folder this server "
        f"opens media files from. Copy the file there, for example with "
        f"cp -c FILE {_media_root}/, or send it inline as {want}.")


def _refuse_reference(field: str, value: str, want: str) -> None:
    if _is_url(value):
        if _inline_only.get():
            raise MediaRefused(f"{field} names a URL, and a launch container session "
                               f"takes media only inline, as {want}.")
        if _allow_urls:
            return
        raise MediaRefused(f"{field} names a URL, and this server takes media only "
                           f"as {want}. {_URL_HINT}")
    if not _inline_only.get() and media_parts(value) is not None:
        return
    raise reference_refusal(field, value, want)


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
                if not (isinstance(ref, str)
                        and _data_uri(ref, "audio", "ref_audio") is not None):
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


def form_type(content_type: str | None) -> bytes:
    """The form type a route's ``request.form()`` sees in ``content_type``,
    parsed as Starlette parses it, or ``b""`` for none."""
    try:
        from python_multipart.multipart import parse_options_header
    except ImportError:
        # Without python-multipart a route cannot parse a form either, so a
        # plain split is enough to refuse one.
        head = (content_type or "").split(";", 1)[0].strip().lower()
        return head.encode("latin-1", "replace")
    return parse_options_header(content_type)[0]


class _Discard:
    """Stands in for an upload's file, so the gate keeps no file data."""

    async def write(self, data: bytes) -> None:
        return None

    async def seek(self, offset: int) -> None:
        return None


async def multipart_text_fields(headers, raw: bytes) -> list[tuple[str, str]]:
    """The text fields of a multipart body, parsed by the parser a route's
    ``request.form()`` uses, with the file parts discarded unread."""
    from starlette.formparsers import MultiPartParser

    class _TextParser(MultiPartParser):
        def on_headers_finished(self) -> None:
            super().on_headers_finished()
            if self._current_part.file is not None:
                self._current_part.file = _Discard()  # type: ignore[assignment]

    async def stream():
        yield raw

    parser = _TextParser(headers, stream())
    try:
        form = await parser.parse()
    finally:
        for f in parser._files_to_close_on_error:
            f.close()
    return [(k, v) for k, v in form.multi_items() if isinstance(v, str)]


async def check_request(method: str, path: str, headers, raw: bytes, *,
                        inline_only: bool = False) -> None:
    """Raise :class:`MediaRefused` when the body of a request to ``path``
    names media the server does not accept, is a form sent to a route that
    takes JSON, or is not JSON. ``inline_only`` refuses files in the media
    folder and URLs too."""
    if not raw:
        return
    token = _inline_only.set(inline_only)
    try:
        kind = form_type(headers.get("content-type"))
        if kind == _MULTIPART:
            if not path.endswith(_UPLOAD_ROUTES):
                raise MediaRefused("this route takes a JSON body, not a multipart form")
            try:
                fields = await multipart_text_fields(headers, raw)
            except Exception as e:
                raise MediaRefused(f"the multipart form cannot be read: {e}") from None
            for name, value in fields:
                check_body({name: value}, path)
            return
        # Every other route reads its body with json.loads, whatever the
        # Content-Type says, so a form-encoded body that holds JSON is JSON.
        try:
            body = json.loads(raw)
        except RecursionError:
            raise MediaRefused("the request body nests too deeply") from None
        except ValueError as e:
            if kind == _URLENCODED:
                raise MediaRefused("this route takes a JSON body, not a form") from None
            raise MediaRefused(f"the request body is not valid JSON ({e})") from None
        check_body(body, path)
    finally:
        _inline_only.reset(token)


def _decode_edit_image(value: str) -> tuple[bytes, str]:
    """The bytes and file suffix of a reference image, inline or in the
    media folder. Pillow reads it with the allowed format plugins only and
    verifies it."""
    from PIL import Image

    if value.startswith("data:"):
        data = _data_uri(value, "image", "image")
        if data is None:
            raise MediaRefused("image must be a base64 data:image/... URI")
    else:
        from gmlx.serve.media_sinks import reference_bytes
        data = reference_bytes(value, "image", "image", "a base64 data:image/... URI")
    formats = list(_EDIT_IMAGE_FORMATS)
    try:
        with Image.open(io.BytesIO(data), formats=formats) as image:
            fmt = image.format
            width, height = image.size
            image.verify()
    except Image.DecompressionBombError:
        raise too_many_pixels("image") from None
    except Exception as e:  # noqa: BLE001 - Pillow raises many types for bad data
        raise MediaRefused("image is not a readable PNG, JPEG, WebP or GIF image") from e
    if width * height > MEDIA_MAX_PIXELS:
        raise too_many_pixels("image", width, height)
    return data, _EDIT_IMAGE_FORMATS[fmt or ""]


@contextlib.contextmanager
def _inline_image_files(values):
    """Write each inline image to a file in a new private folder, yield the
    paths and remove the folder afterwards."""
    decoded = [_decode_edit_image(v) for v in values]
    folder = tempfile.mkdtemp(prefix="gmlx-edit-")          # mode 0700
    try:
        paths = []
        for i, (data, suffix) in enumerate(decoded):
            path = os.path.join(folder, f"image-{i}{suffix}")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            paths.append(path)
        yield paths
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _refuse_image_writes(openai) -> None:
    """Wrap the image generation and edit calls, so a request that names a
    file to write, a model to load by path or a reference image that is not
    inline data is refused there too, whatever reached the route."""
    from fastapi import HTTPException

    if getattr(openai, "_kq_media_guarded", False):
        return
    generate_image, edit_image = openai.generate_image, openai.edit_image

    def _refuse(e: MediaRefused):
        return HTTPException(status_code=400, detail=str(e))

    def _no_output(output_path) -> None:
        if output_path is not None:
            raise MediaRefused("output_path, output_dir and response_format 'path' "
                               "are not accepted by this server; use b64_json")

    def _check_size(request) -> None:
        # The request's size sets the memory a generation takes, so it gets
        # the same ceiling as a decoded image.
        width, height = getattr(request, "width", None), getattr(request, "height", None)
        if not isinstance(width, int) or not isinstance(height, int):
            return
        if width <= 0 or height <= 0:
            raise MediaRefused(f"size {width}x{height} is not a positive width and height")
        if width * height > MEDIA_MAX_PIXELS:
            raise MediaRefused(f"size is {width}x{height} pixels, over the limit of "
                               f"{MEDIA_MAX_PIXELS} pixels. Ask for a smaller image.")

    def _check_call(request, kwargs, *, edit: bool) -> None:
        # Both calls take these keyword-only. A string request is a prompt.
        _no_output(kwargs.get("output_path"))
        _check_size(request)
        extra = {**(getattr(request, "extra", None) or {}), **kwargs}
        if extra.get("prompt_expansion_model") is not None:
            raise MediaRefused("prompt_expansion_model is not accepted by this server")
        refs = [*(kwargs.get("image_paths") or ())]
        if edit:
            refs += [*(getattr(request, "image_paths", None) or ())]
        for ref in refs:
            check_image(ref, "image")

    def guarded_generate_image(model, request, **kwargs):
        try:
            _check_call(request, kwargs, edit=False)
            with contextlib.ExitStack() as stack:
                # An edit task opens each reference image by file name.
                if kwargs.get("image_paths") is not None:
                    kwargs["image_paths"] = stack.enter_context(
                        _inline_image_files(kwargs["image_paths"]))
                return generate_image(model, request, **kwargs)
        except MediaRefused as e:
            raise _refuse(e) from None

    def guarded_edit_image(model, request, **kwargs):
        try:
            _check_call(request, kwargs, edit=True)
            with contextlib.ExitStack() as stack:
                # The loaders open each reference image by file name.
                if kwargs.get("image_paths") is not None:
                    kwargs["image_paths"] = stack.enter_context(
                        _inline_image_files(kwargs["image_paths"]))
                if getattr(request, "image_paths", None):
                    request = dataclasses.replace(request, image_paths=tuple(
                        stack.enter_context(_inline_image_files(request.image_paths))))
                return edit_image(model, request, **kwargs)
        except MediaRefused as e:
            raise _refuse(e) from None

    openai.generate_image = guarded_generate_image
    openai.edit_image = guarded_edit_image
    openai._kq_media_guarded = True


def _drop_websocket_routes(app) -> None:
    """Remove every WebSocket route, in every router the app includes.
    HTTP middleware, the API key check among it, never sees a WebSocket,
    and mlx-vlm's ``/v1/realtime`` loads the model a message names. gmlx
    serves no WebSocket route."""
    from starlette.routing import WebSocketRoute

    from ._common import _invalidate_route_caches, _iter_route_lists

    for routes in list(_iter_route_lists(app)):
        routes[:] = [r for r in routes if not isinstance(r, WebSocketRoute)]
    _invalidate_route_caches(app)


class _RefuseWebSockets:
    """Close every WebSocket connection before any route sees it, so a
    WebSocket route that a later mlx-vlm adds fails closed. HTTP middleware
    never sees a WebSocket, so this is plain ASGI middleware."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            # Closing before the accept makes the server answer the
            # handshake with 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)


def _is_upload(request) -> bool:
    """True for a multipart form sent to an audio upload route, which may be
    as large as :data:`UPLOAD_MAX_BYTES`."""
    return (request.url.path in _UPLOAD_PATHS
            and form_type(request.headers.get("content-type")) == _MULTIPART)


def body_limit(upload: bool, session: bool) -> int:
    """The largest body the server reads for a request of this kind."""
    if session:
        return SESSION_UPLOAD_MAX_BYTES if upload else SESSION_BODY_MAX_BYTES
    return UPLOAD_MAX_BYTES if upload else BODY_MAX_BYTES


def _body_refusal(upload: bool, session: bool = False, path: str = "") -> str:
    limit = f"{body_limit(upload, session) >> 20} MiB limit"
    if upload:
        where = ("of a launch session" if session
                 else "of the transcription and translation routes")
        return (f"the audio upload is larger than the {limit} {where}. Send a "
                "compressed file, such as MP3 or M4A, or split the recording.")
    where = " of a launch session" if session else ""
    advice = ("Start a new conversation, compact this one, or send fewer or smaller "
              "images." if path in _CONVERSATION_PATHS else "Send fewer or shorter inputs.")
    return f"the request body is larger than the {limit}{where}. {advice}"


async def _read_body(request, limit: int) -> bytes | None:
    """The request body, or None when it is larger than ``limit``, which is
    refused before it is all read."""
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        return None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    raw = b"".join(chunks)
    # The routes below read this copy, as they would after request.body().
    request._body = raw
    return raw


def install_media_gate(allow_urls: bool = False, app=None) -> None:
    """Answer 400 for a request whose body names media by a file path, or by
    a URL unless ``allow_urls``, or that sends a form to a route that takes
    JSON. Install it before the API-key middleware, so an unauthenticated
    request gets 401 and learns nothing from the gate. Idempotent."""
    global _allow_urls
    from fastapi.responses import JSONResponse

    _allow_urls = bool(allow_urls)
    stock = app is None
    if app is None:
        app = importlib.import_module("mlx_vlm.server.app").app
    if stock:
        _refuse_image_writes(importlib.import_module("mlx_vlm.server.openai"))
        from gmlx.serve import media_sinks
        media_sinks.install()
    _drop_websocket_routes(app)
    if getattr(app.state, _FLAG, False):
        return
    app.middleware_stack = None
    app.add_middleware(_RefuseWebSockets)

    async def _media_gate(request, call_next):
        # Every method: a route reads a body whatever the method is.
        upload = _is_upload(request)
        session = request.scope.get(SESSION_SCOPE_KEY) is not None
        raw = await _read_body(request, body_limit(upload, session))
        if raw is None:
            return JSONResponse(status_code=413, content=_error_content(
                request.url.path, 413, "invalid_request_error",
                _body_refusal(upload, session, request.url.path)))
        try:
            await check_request(
                request.method, request.url.path, request.headers, raw,
                inline_only=session)
        except MediaRefused as e:
            return JSONResponse(status_code=400, content=_error_content(
                request.url.path, 400, "invalid_request_error", str(e)))
        return await call_next(request)

    app.middleware_stack = None          # allow install after a stack build
    app.middleware("http")(_media_gate)
    setattr(app.state, _FLAG, True)
