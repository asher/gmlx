"""The request bodies the media gate leaves unread, read by their routes.

The media gate (``media_gate``) reads every other body whole before its
route runs, to a fixed bound. The routes here take bodies whose size is
the user's payload: the images API's input images and a video job's
form files. Each prices its body before its first byte is read, from its
Content-Length, so a body the machine cannot hold is refused with the
figures, and none is read past them:

- what stays in RAM (a JSON body, a form's text fields, the images API's
  files, which decode into arrays) against the bytes free beside the
  models loaded, read from the pool's figures with no lock, at the copies
  of it the read and the parse hold;
- a video job's form files, which go to disk, against the room left in
  the job store.

A body with no Content-Length is read until it passes its figure, and
refused there; a video form with none takes its room a step at a time. The gate's own checks of the media a JSON or images API
body names run here, on the body read, as the gate runs them on every
other route. A video form's text fields have no gate rule: the route's
own checks of each field it reads are their check, and refuse a path or
a URL where an image goes.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os

from starlette.requests import Request

# The routes whose bodies the gate leaves to them: POST only. The images
# API is served with and without the /v1 prefix.
IMAGE_PATHS = frozenset(f"{p}/images/{r}" for p in ("/v1", "")
                        for r in ("generations", "edits", "variations"))
VIDEO_PATHS = frozenset({"/v1/videos"})

# A video form's text fields are held twice while the form is parsed:
# Starlette's bytearray of each part, with its growth slack, and the str
# it decodes to. The sum is priced at that factor, measured at 2.1; the
# post-V6 parse change removes the copy. Once a text chunk is not ASCII
# the str can take 4 bytes a character: measured at 6.1.
FORM_TEXT_COPIES = 2.25
FORM_TEXT_WIDE_COPIES = 6.25
# Seconds a video form may send nothing before its end: past them, the
# read ends and the room the form holds in the job store is freed. Each
# chunk starts the wait again, so a slow client that keeps sending reads
# to its end.
STALL_S = 60.0


class BodyRefused(Exception):
    """A body a route does not read: ``status``, the message, and the
    request field it names as ``param`` when there is one."""

    def __init__(self, status: int, message: str, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.param = param

    @property
    def headers(self) -> dict:
        """A 413 or a 408 leaves the body unread past its figure: the
        answer ends the connection."""
        return {"Connection": "close"} if self.status in (408, 413) else {}


def log_refusal(status: int, message: str) -> None:
    """Print a refusal of a body or its media to the log, as the media
    gate prints the refusals of the routes it reads."""
    from .hardening import _log_refusal

    _log_refusal(f"{status} {message}", message, status, kind="media")


def reads_own_body(request) -> bool:
    """Whether the route of ``request`` reads and prices its own body, so
    the gate leaves it unread."""
    return (request.method == "POST"
            and request.url.path in IMAGE_PATHS | VIDEO_PATHS)


def _mib(n: int, up: bool = False) -> str:
    """``n`` bytes in GiB, MiB or KiB, whichever it reaches, rounded up
    when ``up`` and down otherwise. A refusal rounds the bytes it needs up
    and the figure it has down, so a need over the figure never reads at
    or under it."""
    for unit, shift, places in (("GiB", 30, 2), ("MiB", 20, 1),
                                ("KiB", 10, 1)):
        if n >= 1 << shift or unit == "KiB":
            x = n * 10 ** places / (1 << shift)
            x = math.ceil(x) if up else math.floor(x)
            return f"{x / 10 ** places:.{places}f} {unit}"
    raise AssertionError("unreachable")


def free_bytes() -> int | None:
    """The bytes free beside what the pool holds and the uploads admitted,
    from its figures with no lock, or None with no budget to price by
    (no pool, no working budget, or GMLX_OVERCOMMIT set)."""
    from ..capacity import overcommit

    if overcommit():
        return None
    from ._common import _get_pool

    pool = _get_pool()
    if pool is None:
        return None
    f = pool.figures
    if not f.budget:
        return None
    return max(0, f.budget - f.resident - f.incoming)


def content_length(request) -> int | None:
    """The body's Content-Length, or None with none sent."""
    length = request.headers.get("content-length")
    if length is None:
        return None
    try:
        n = int(length)
    except ValueError:
        raise BodyRefused(400, "the Content-Length is not a number") from None
    if n < 0:
        raise BodyRefused(400, "the Content-Length is not a number")
    return n


def held(n: int, copies: float) -> int:
    """The bytes a read of ``n`` bytes holds at ``copies`` copies."""
    return math.ceil(n * copies)


# Bytes a parsed JSON value holds past its text, 63.7 at most by tracemalloc.
JSON_VALUE_BYTES = 80
# The bytes a character of a str takes at most, in a body that is not
# ASCII or holds a \u escape.
WIDE_CHAR_BYTES = 4


def json_held(raw: bytes, copies: float, str_copies: float) -> int:
    """The bytes a JSON body's read and parse hold: ``copies`` of its
    length, of which ``str_copies`` are the strs it parses to, each at 4
    bytes a character when the body is not ASCII or holds a ``\\u``
    escape, and JSON_VALUE_BYTES for each value or container it can parse
    to. A value or container starts the body or follows a ",", ":", "["
    or "{", so their count bounds it, read with no parse."""
    if not raw.isascii() or b"\\u" in raw:
        copies += str_copies * (WIDE_CHAR_BYTES - 1)
    count = (raw.count(b",") + raw.count(b":") + raw.count(b"[")
             + raw.count(b"{"))
    return held(len(raw), copies) + JSON_VALUE_BYTES * (count + 1)


def _text_over_free(n: int, free: int, copies: float) -> BodyRefused:
    """The 413 of a form whose text fields, ``n`` bytes so far, passed the
    memory free at ``copies``."""
    return BodyRefused(413, f"the text fields of this form are over "
                       f"{_mib(n)}, their read holds {copies:g} times that, "
                       f"and {_mib(free)} of memory is free beside the "
                       "models loaded; send shorter text fields, or unload "
                       "a model to make room")


def _over_free(n: int, free: int, sent: bool,
               copies: float = 1) -> BodyRefused:
    size = _mib(n, True) if sent else f"over {_mib(n)}"
    times = f", its read holds {copies:g} times that," if copies != 1 else ","
    return BodyRefused(413, f"the request body is {size}{times} and "
                       f"{_mib(free)} of memory is free beside the models "
                       "loaded; send a smaller body, or unload a model to "
                       "make room")


async def read_in_ram(request: Request, free: int | None = None,
                      copies: float = 1,
                      str_copies: float | None = None) -> bytes:
    """The body of ``request``, priced against ``free`` bytes (the pool's
    free figure when None) before a byte of it is read, at the ``copies``
    of it the route's read and parse hold. A body with no Content-Length
    is refused once what arrived, at that factor, passes the figure. A
    JSON body, whose parse makes ``str_copies`` of its strs, is priced
    again once read and before it is parsed, with the width of its strs
    and the values it can parse to (:func:`json_held`). The body
    read is kept on the request, so ``request.form()`` and
    ``request.json()`` parse it after."""
    if free is None:
        free = free_bytes()
    n = content_length(request)
    if n is not None and free is not None and held(n, copies) > free:
        raise _over_free(n, free, True, copies)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if free is not None and held(size, copies) > free:
            raise _over_free(size, free, False, copies)
        chunks.append(chunk)
    raw = b"".join(chunks)
    if str_copies is not None and free is not None:
        need = json_held(raw, copies, str_copies)
        if need > free:
            raise BodyRefused(413, f"the request body is "
                              f"{_mib(len(raw), True)}, its read and parse "
                              f"hold {_mib(need, True)} with the values it "
                              f"parses to, and {_mib(free)} of memory is "
                              "free beside the models loaded; send a "
                              "smaller body, or unload a model to make room")
    request._body = raw
    return raw


async def check_media(request: Request, raw: bytes) -> None:
    """The media gate's checks of the body ``raw``, as the gate runs them
    on every route it reads: a media reference the server does not take,
    a form sent to a route that takes JSON, a body that is not JSON. The
    body was priced whole, so an inline image takes no byte limit of its
    own; the item count and the pixel limits hold."""
    from ._common import SESSION_BODY_KEY, SESSION_SCOPE_KEY
    from .media_gate import MediaRefused, body_priced, check_request

    session = request.scope.get(SESSION_SCOPE_KEY) is not None
    parsed = request.scope.pop(SESSION_BODY_KEY, None)
    try:
        with body_priced():
            await check_request(request.method, request.url.path,
                                request.headers, raw, inline_only=session,
                                parsed=parsed)
    except MediaRefused as e:
        raise BodyRefused(400, str(e), e.param) from None


def arriving_note(arriving: int) -> str:
    """The part of a store refusal that names the bytes the other POSTs
    still arriving hold, when they hold any."""
    return (f", and {_mib(arriving)} of it is held by uploads arriving now"
            if arriving else "")


def _over_room(n: int | None, room: int, arriving: int = 0) -> BodyRefused:
    """The 413 of a form past the job store's room: its Content-Length
    ``n``, or None for a stream cut at the room. ``arriving`` is what the
    other POSTs still arriving hold of the store."""
    what = (f"over the {_mib(room)} of room the video job store has left"
            if n is None else f"{_mib(n, True)}, and the video job store "
            f"has {_mib(room)} of room left")
    return BodyRefused(413, f"the files of this job are {what}"
                       f"{arriving_note(arriving)}; send smaller files, "
                       "delete finished jobs, or raise "
                       "server.video.store_max_bytes")


async def _within(timeout: float, aw):
    """The wait for a form's next chunk, ``timeout`` seconds at most."""
    return await asyncio.wait_for(aw, timeout)


def _stalled(stall: float, got: int, n: int | None) -> BodyRefused:
    """The 408 of a form that sent nothing for ``stall`` seconds, ``got``
    bytes into its Content-Length ``n`` (None with none sent)."""
    left = (f" with {_mib(n - got, True)} of it still to come"
            if n is not None else "")
    return BodyRefused(408, f"the form sent nothing for {stall:g} s"
                       f"{left}, so the server ended the request and freed "
                       "the room it held in the video job store; send the "
                       "form again")


async def read_form_to_disk(request: Request, folder: str, take,
                            free: int | None = None) -> tuple[list, dict]:
    """``(fields, files)`` of a multipart form: its text fields (name,
    value) in RAM, priced together against ``free`` bytes (the pool's
    figure when None) as they arrive, and its files written to
    ``folder``, by field name, each as ``NAME.part`` while it arrives and
    renamed once whole. ``await take(n)`` reserves the body's ``n`` bytes
    of the room left in the job store before the read and answers ``(ok,
    bytes, arriving)``: the bytes reserved, or the room left and the bytes
    other POSTs hold when they do not fit. A body with no Content-Length
    reserves a step with ``take(None)`` before its first byte and another
    each time the bytes arrived pass what it holds, so it holds the room
    as its body comes, and the read stops where a step does not fit. A
    form that sends nothing for STALL_S seconds is refused with a 408. A
    refusal or a failed read removes every file it began."""
    from starlette.datastructures import UploadFile
    from starlette.formparsers import MultiPartException, MultiPartParser
    from starlette.requests import ClientDisconnect

    if free is None:
        free = free_bytes()
    n = content_length(request)
    ok, room, arriving = await take(n)
    if not ok:
        raise _over_room(n, room, arriving)
    os.makedirs(folder, exist_ok=True)
    began: list = []
    names: dict = {}
    text = [0]
    wide = [False]

    class _ToDisk(MultiPartParser):
        def on_headers_finished(self) -> None:
            super().on_headers_finished()
            part = self._current_part
            if part.file is None:
                return
            part.file.file.close()
            name = part.field_name
            if not name.replace("_", "").isalnum() or name in names:
                raise MultiPartException(f"file field {name!r} is not read "
                                         "by this route, or comes twice")
            path = os.path.join(folder, f"{name}.part")
            began.append(path)
            names[name] = path
            part.file = UploadFile(open(path, "wb+"),  # noqa: SIM115
                                   filename=part.file.filename,
                                   headers=part.file.headers)

        def on_part_data(self, data: bytes, start: int, end: int) -> None:
            part = self._current_part
            if part.file is not None:
                super().on_part_data(data, start, end)
                return
            # The text fields stay in RAM: their sum is priced, not each,
            # at the copies the parse holds, wide once a chunk is not ASCII.
            chunk = data[start:end]
            text[0] += len(chunk)
            wide[0] = wide[0] or not chunk.isascii()
            copies = FORM_TEXT_WIDE_COPIES if wide[0] else FORM_TEXT_COPIES
            if free is not None and held(text[0], copies) > free:
                raise _text_over_free(text[0], free, copies)
            part.data.extend(chunk)

    async def stream():
        held, got = room, 0
        chunks = request.stream().__aiter__()
        while True:
            stall = STALL_S
            try:
                chunk = await _within(stall, chunks.__anext__())
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise _stalled(stall, got, n) from None
            got += len(chunk)
            while got > held:
                ok, more, arriving = ((False, 0, 0) if n is not None
                                      else await take(None))
                if not ok:
                    raise _over_room(None, held, arriving)
                held += more
            yield chunk

    parser = _ToDisk(request.headers, stream(), max_files=16,
                     max_fields=64)
    try:
        try:
            form = await parser.parse()
        except (BodyRefused, ClientDisconnect):
            raise
        except MultiPartException as e:
            raise BodyRefused(400, f"the multipart form cannot be read: "
                              f"{e.message}") from None
        except Exception as e:  # noqa: BLE001 - as the gate's form parse
            raise BodyRefused(400, f"the multipart form cannot be read: "
                              f"{e}") from None
        files = {}
        fields = []
        for key, value in form.multi_items():
            if isinstance(value, str):
                fields.append((key, value))
                continue
            value.file.close()
            done = names[key][:-len(".part")]
            os.replace(names[key], done)
            files[key] = done
        return fields, files
    except BaseException:
        for path in began:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path[:-len(".part")])
        raise
