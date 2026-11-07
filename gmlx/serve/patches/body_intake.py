"""The request bodies the media gate leaves unread, read by their routes.

The media gate (``media_gate``) reads every other body whole before its
route runs, to a fixed bound. The routes here take bodies whose size is
the user's payload: the images API's input images and a video job's
form files. Each prices its body before its first byte is read, from its
Content-Length, so a body the machine cannot hold is refused with the
figures, and none is read past them:

- what stays in RAM (a JSON body, a form's text fields, the images API's
  files, which decode into arrays) against the bytes free beside the
  models loaded, read from the pool's figures with no lock, at what the
  read and the parse hold at their peak: a JSON body is read into one
  buffer and parsed once, each step dropping the copy before it, and a form is parsed as it arrives, each text field
  priced at its own width;
- a video job's form files, which go to disk, against the room left in
  the job store.

A body with no Content-Length is read until it passes its figure, and
refused there; a video form with none takes its room a step at a time.
The gate's own checks of the media a JSON or images API body names run
here, on the body read, as the gate runs them on every other route. A
video form's text fields have no gate rule: the route's own checks of
each field it reads are their check, and refuse a path or a URL where an
image goes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re

from starlette.requests import Request

# The routes whose bodies the gate leaves to them: POST only. The images
# API is served with and without the /v1 prefix.
IMAGE_PATHS = frozenset(f"{p}/images/{r}" for p in ("/v1", "")
                        for r in ("generations", "edits", "variations"))
VIDEO_PATHS = frozenset({"/v1/videos"})

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


# Bytes a parsed JSON value holds past its text, 63.7 at most by tracemalloc.
JSON_VALUE_BYTES = 80
# The bytes a character of a str takes at most.
WIDE_CHAR_BYTES = 4
# The bytes a read and a parse hold beside the figures priced: a chunk as
# it arrives, a piece of a body escaped to ASCII, the objects' headers.
_SLACK = 1 << 20
# A str of a JSON body at most this long is priced with the text around
# it; a longer one on its own.
_LONG = 256
# The strs of a body longer than _LONG, each matched after the run of
# other text and shorter strs before it, so no str is matched from its
# closing quote.
_LONG_STR = re.compile(rb'(?:[^"]++|"(?>(?:[^"\\]|\\.){0,%d})")*+'
                       rb'("(?:[^"\\]++|\\.)*+")' % _LONG, re.S)
_HIGH = re.compile(rb"[\x80-\xff]")
# The UTF-8 lead bytes of a character past U+FFFF, and of one past U+00FF.
_LEAD4 = re.compile(rb"[\xf0-\xff]")
_LEAD2 = re.compile(rb"[\xc4-\xef]")
# The bytes a str decoded from UTF-8 holds for each byte decoded, by its
# kind (:func:`_kind`): at the end of its decode, the str and the narrower
# copy the decoder widens it from, by tracemalloc; once decoded, the str.
_DECODE = {0: 1, 1: 2, 2: 3, 4: 6}
_DECODED = {0: 1, 1: 1, 2: 2, 4: 4}
_HIGH_BYTES = bytes(range(0x80, 0x100))
_ASTRAL = re.compile(rb"\\U([0-9a-f]{8})")
_NON_ASCII = re.compile(r"[^\x00-\x7f]+")
# The bytes of a body escaped to ASCII at a time.
_PIECE = 64 << 10


def _kind(buf) -> int:
    """0 when UTF-8 ``buf`` is ASCII, else the bytes a character takes in
    the str it decodes to, from the lead byte of its widest character."""
    if buf.isascii():
        return 0
    return 4 if _LEAD4.search(buf) else 2 if _LEAD2.search(buf) else 1


def _high_bytes(buf) -> int:
    """The bytes of ``buf`` that are not ASCII, counted a piece at a
    time."""
    if buf.isascii():
        return 0
    mv = memoryview(buf)
    return sum(len(p) - len(p.translate(None, _HIGH_BYTES))
               for p in (bytes(mv[i:i + _PIECE])
                         for i in range(0, len(buf), _PIECE)))


def json_price(buf) -> int:
    """The bytes the parse of the UTF-8 JSON body ``buf`` holds at its
    peak, read from its bytes with no parse (:func:`_json_plan`)."""
    return _json_plan(buf)[0]


def _json_plan(buf) -> tuple[int, int]:
    """``(price, escape)``: the bytes the parse of the UTF-8 JSON body
    ``buf`` holds at its peak, read from its bytes with no parse, and the
    bytes of it that are not ASCII when the parse escapes it to ASCII
    first, the way that holds less, else 0. A str takes the width of its widest character, so a body
    decoded as it is takes the width of its widest at every character; a
    body escaped to ASCII takes a byte at each, at 3 bytes for each byte
    that is not ASCII. The strs the parse makes take their own width
    either way. Each step holds:

    - the escape: the body and its escaped copy;
    - the decode: the body, escaped or not, and its str, with the
      narrower copy the decoder widens it from;
    - the parse: that str, every str it makes, JSON_VALUE_BYTES for each
      value, and the str it is making, which grows past its end;
    - an inline image or video decoded after the parse: every str, a
      data URL's payload cut from its str, and the bytes it decodes to.

    A str the parse makes takes a byte for each byte of it, or
    WIDE_CHAR_BYTES once it is not ASCII or holds a \\u escape. It grows
    1.25 times past its end, and once a \\u escape widens it, to 6.25
    times its characters with the narrow copy it widens from, by
    tracemalloc. A value or container starts the body or follows a ",",
    ":", "[" or "{", so their count bounds the values."""
    n = len(buf)
    high = _high_bytes(buf)
    wide = high > 0 or buf.find(b"\\u") >= 0
    strs = longs = 0
    grows = decode = int(1.75 * _LONG * WIDE_CHAR_BYTES)
    pos = 0
    spans = []
    while (m := _LONG_STR.match(buf, pos)) is not None:
        spans.append(m.span(1))
        pos = m.end()
    # A str with no closing quote is made up to the end of the body.
    quote = buf.find(b'"', pos)
    if quote >= 0:
        spans.append((quote, n))
    for s, e in spans:
        size = e - s
        longs += size
        u = buf.find(b"\\u", s, e) >= 0 or _HIGH.search(buf, s, e) is not None
        own = size * WIDE_CHAR_BYTES if u else size
        strs += own
        if u:
            grows = max(grows, math.ceil(size * 2.25))
        elif buf.find(b"\\", s, e) >= 0:
            grows = max(grows, math.ceil(size * 0.25))
        cut = own if buf.startswith(b'"data:', s) else 0
        decode = max(decode, cut + math.ceil(size * 0.75))
    strs += (n - longs) * (WIDE_CHAR_BYTES if wide else 1)
    values = JSON_VALUE_BYTES * (sum(buf.count(c) for c in b",:[{") + 1)
    after = strs + values + decode
    # Decoded as it is, the str takes the width of the widest character.
    kind = _kind(buf)
    plain = max(n + n * _DECODE[kind],
                n * _DECODED[kind] + strs + grows + values, after)
    if not high:
        return plain + _SLACK, 0
    escaped = n + 2 * high
    ascii_ = max(2 * n + 2 * high, 2 * escaped,
                 escaped + strs + grows + values, after)
    return min(plain, ascii_) + _SLACK, high if ascii_ < plain else 0


def _escaped(buf, high: int):
    """``buf``, a UTF-8 JSON body with ``high`` bytes that are not ASCII,
    with each character that is not ASCII written as its \\u escape, as
    ``json.dumps`` writes one, so the str it decodes to is ASCII and each
    str the parse makes takes the width of its own characters. Bytes that
    are not UTF-8 raise ValueError, as ``json.loads`` raises."""
    # With no \x or \U in the body, every one after Python's backslash
    # escape is one it wrote.
    fast = buf.find(b"\\x") < 0 and buf.find(b"\\U") < 0
    n = len(buf)
    out = bytearray(n + 2 * high)
    mv = memoryview(buf)
    i = o = 0
    while i < n:
        j = min(i + _PIECE, n)
        # End the piece on the first byte of a character.
        while i < j < n and buf[j] & 0xC0 == 0x80:
            j -= 1
        if j == i:
            j = min(i + _PIECE, n)
        piece = bytes(mv[i:j])
        if not piece.isascii():
            try:
                text = piece.decode("utf-8", "surrogatepass")
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"'utf-8' codec can't decode byte 0x{piece[e.start]:02x} "
                    f"in position {i + e.start}: {e.reason}") from None
            if fast:
                piece = text.encode("ascii", "backslashreplace").replace(
                    b"\\x", b"\\u00")
                if b"\\U" in piece:
                    piece = _ASTRAL.sub(_surrogates, piece)
            else:
                piece = _NON_ASCII.sub(lambda m: json.dumps(m.group())[1:-1],
                                       text).encode("ascii")
        out[o:o + len(piece)] = piece
        o += len(piece)
        i = j
    mv.release()
    del out[o:]
    return out


def _surrogates(m) -> bytes:
    """The \\u escapes of the surrogate pair of a \\U escape."""
    c = int(m.group(1), 16) - 0x10000
    return b"\\u%04x\\u%04x" % (0xD800 + (c >> 10), 0xDC00 + (c & 0x3FF))


def _over_free(n: int, free: int, sent: bool,
               need: int | None = None) -> BodyRefused:
    size = _mib(n, True) if sent else f"over {_mib(n)}"
    hold = (f", its read and parse hold at least {_mib(need, True)},"
            if need is not None else ",")
    return BodyRefused(413, f"the request body is {size}{hold} and "
                       f"{_mib(free)} of memory is free beside the models "
                       "loaded; send a smaller body, or unload a model to "
                       "make room")


def _text_over_free(n: int, free: int, need: int) -> BodyRefused:
    """The 413 of a form whose text fields, ``n`` bytes so far, hold
    ``need`` bytes as they are parsed, past the memory free."""
    return BodyRefused(413, f"the text fields of this form are over "
                       f"{_mib(n)}, their read and parse hold "
                       f"{_mib(need, True)}, and {_mib(free)} of memory is "
                       "free beside the models loaded; send shorter text "
                       "fields, or unload a model to make room")


class _TextPrice:
    """The bytes a form's text fields hold as they are parsed: each field
    read, as the str it decoded to, and the field arriving, as its bytes
    with the room the bytearray grows by and the str they decode to at
    its end. Each str takes the width of its own widest character
    (:data:`_DECODE`)."""

    # The room a bytearray holds past its bytes as it grows, one eighth.
    GROW = 1.125

    def __init__(self) -> None:
        self.read = self.done = self.size = self.kind = 0

    def add(self, chunk: bytes) -> int:
        """Count ``chunk`` of the field arriving; answer what all hold."""
        self.read += len(chunk)
        self.size += len(chunk)
        self.kind = max(self.kind, _kind(chunk))
        return self.held()

    def end(self) -> None:
        """The field arriving ends: it is now its str."""
        self.done += self.size * _DECODED[self.kind]
        self.size = self.kind = 0

    def held(self) -> int:
        return (self.done + math.ceil(self.size * self.GROW)
                + self.size * _DECODE[self.kind] + _SLACK)


def _min_price(n: int) -> int:
    """The least a JSON body of ``n`` bytes holds as it is parsed: its
    bytes and their str, beside each other at the decode."""
    return 2 * n


async def _read_bytes(request: Request, n: int | None,
                      free: int | None) -> bytearray:
    """The body of ``request`` in one bytearray, made at its Content-Length
    ``n`` before the read, so it holds no copy of the chunks. A body with
    no length is held as its chunks until its end, refused once
    :func:`_min_price` of what arrived passes ``free``, then moved into
    one, a chunk at a time."""
    if n is not None:
        buf = bytearray(n)
        got = 0
        async for chunk in request.stream():
            if got + len(chunk) > n:
                raise BodyRefused(400, "the request body is longer than "
                                  "its Content-Length")
            buf[got:got + len(chunk)] = chunk
            got += len(chunk)
        del buf[got:]
        return buf
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if free is not None and _min_price(size) > free:
            raise _over_free(size, free, False, _min_price(size))
        chunks.append(chunk)
    buf = bytearray(size)
    chunks.reverse()
    got = 0
    while chunks:
        chunk = chunks.pop()
        buf[got:got + len(chunk)] = chunk
        got += len(chunk)
    return buf


def _parse(box: list, path: str, ctype: str | None, free: int | None,
           inline_only: bool, parsed):
    """The JSON document of the body in ``box``, which this takes, so the
    body is freed as its parse goes: priced against ``free``, escaped to
    ASCII, decoded and parsed once, then held to the media gate's checks.
    ``parsed`` is the length and document of a body the session socket
    parsed already."""
    from .media_gate import (
        _MULTIPART,
        _URLENCODED,
        MediaRefused,
        body_priced,
        check_json,
        form_type,
    )

    kind = form_type(ctype)
    if kind == _MULTIPART:
        raise BodyRefused(400, "this route takes a JSON body, not a "
                          "multipart form")
    if parsed is not None and parsed[0] == len(box[0]):
        box.clear()
        doc = parsed[1]
    else:
        try:
            doc = _loads(box, free)
        except RecursionError:
            raise BodyRefused(400, "the request body nests too deeply") \
                from None
        except ValueError as e:
            if kind == _URLENCODED:
                raise BodyRefused(400, "this route takes a JSON body, not "
                                  "a form") from None
            raise BodyRefused(400, f"the request body is not valid JSON "
                              f"({e})") from None
    try:
        with body_priced():
            check_json(doc, path, inline_only=inline_only)
    except MediaRefused as e:
        raise BodyRefused(400, str(e), e.param) from None
    return doc


def _loads(box: list, free: int | None):
    """The document of the JSON body in ``box``, which this takes, priced
    against ``free`` (:func:`json_price`), escaped to ASCII, decoded and
    parsed once. Each step drops the copy before it."""
    buf = box.pop()
    encoding = json.detect_encoding(bytes(buf[:4]))
    if encoding == "utf-8-sig":
        del buf[:3]
    elif encoding != "utf-8":
        # A UTF-16 or UTF-32 body, as json.loads reads one: the body, its
        # str, at most twice its bytes, and that str in UTF-8, at most 1.5
        # times.
        need = math.ceil(4.5 * len(buf)) + _SLACK
        if free is not None and need > free:
            raise _parse_over_free(len(buf), need, free)
        buf = buf.decode(encoding).encode("utf-8", "surrogatepass")
    need, high = _json_plan(buf)
    if free is not None and need > free:
        raise _parse_over_free(len(buf), need, free)
    if high:
        buf = _escaped(buf, high)
    text = buf.decode("utf-8", "surrogatepass")
    del buf
    return json.loads(text)


def _parse_over_free(n: int, need: int, free: int) -> BodyRefused:
    return BodyRefused(413, f"the request body is {_mib(n, True)}, its read "
                       f"and parse hold {_mib(need, True)} with the values "
                       f"it parses to, and {_mib(free)} of memory is free "
                       "beside the models loaded; send a smaller body, or "
                       "unload a model to make room")


async def read_json(request: Request, free: int | None = None):
    """The JSON document of the body of ``request``, ``{}`` for an empty
    body, held to the media gate's checks as the gate holds every route it
    reads. The body is priced against ``free`` bytes (the pool's free
    figure when None): at :func:`_min_price` of its Content-Length before
    a byte of it is read, or of what arrived as a body with none arrives,
    then at :func:`json_price` once read. The body was priced whole, so an inline image takes no byte
    limit of its own; the item count and the pixel limits hold."""
    from ._common import SESSION_BODY_KEY, SESSION_SCOPE_KEY

    if free is None:
        free = free_bytes()
    n = content_length(request)
    if n is not None and free is not None and _min_price(n) > free:
        raise _over_free(n, free, True, _min_price(n))
    box = [await _read_bytes(request, n, free)]
    session = request.scope.get(SESSION_SCOPE_KEY) is not None
    parsed = request.scope.pop(SESSION_BODY_KEY, None)
    if not box[0]:
        return {}
    return _parse(box, request.url.path, request.headers.get("content-type"),
                  free, session, parsed)


async def read_form_in_ram(request: Request, free: int | None = None,
                           max_files: int = 64):
    """The multipart form of ``request``, parsed as it arrives with no
    copy of its body held, its text fields held to the media gate's checks
    as the gate holds every route it reads. Its text fields are priced
    against ``free`` bytes (the pool's figure when None) as they arrive,
    each at its own width (:class:`_TextPrice`), with the files spooled
    beside them; once parsed, with the largest file, which the route reads
    whole. A Content-Length past the figure is refused before the read."""
    from starlette.formparsers import MultiPartParser
    from starlette.requests import ClientDisconnect

    from ._common import SESSION_SCOPE_KEY
    from .media_gate import _FORM_ROUTES, MediaRefused, check_form_fields

    path = request.url.path
    if not path.endswith(_FORM_ROUTES):
        raise BodyRefused(400, "this route takes a JSON body, not a "
                          "multipart form")
    if free is None:
        free = free_bytes()
    n = content_length(request)
    if n is not None and free is not None and n > free:
        raise _over_free(n, free, True)
    text = _TextPrice()
    spool = MultiPartParser.spool_max_size
    files: list = []

    def spooled(moving: bool = False) -> int:
        # A file holds its bytes in RAM up to the spool's size, with the
        # room its buffer grows by, and none once it moves to disk; as it
        # moves, that buffer and a copy of its bytes.
        held = sum(f for f in files if f <= spool)
        return math.ceil(held * _TextPrice.GROW) + (
            math.ceil(spool * _TextPrice.GROW) + spool if moving else 0)

    def check(need: int, largest: int = 0) -> None:
        if free is None or need <= free:
            return
        what = (f"the largest file of this form is {_mib(largest, True)}, "
                f"its read holds {_mib(need, True)} beside the text fields "
                "and the files spooled" if largest else
                f"the text fields of this form are over {_mib(text.read)}, "
                f"their read and parse hold {_mib(need, True)} with the "
                "files spooled")
        raise BodyRefused(413, f"{what}, and {_mib(free)} of memory is free "
                          "beside the models loaded; send shorter text "
                          "fields or smaller files, or unload a model to "
                          "make room")

    class _Priced(MultiPartParser):
        def on_part_data(self, data: bytes, start: int, end: int) -> None:
            if self._current_part.file is None:
                check(text.add(data[start:end]) + spooled())
            else:
                before = files[-1]
                files[-1] += end - start
                check(text.held() + spooled(before <= spool < files[-1]))
            super().on_part_data(data, start, end)

        def on_part_end(self) -> None:
            if self._current_part.file is None:
                text.end()
            super().on_part_end()

        def on_headers_finished(self) -> None:
            super().on_headers_finished()
            if self._current_part.file is not None:
                files.append(0)

    parser = _Priced(request.headers, request.stream(), max_files=max_files,
                     max_part_size=n if n else 1 << 62)
    try:
        form = await parser.parse()
    except (BodyRefused, ClientDisconnect):
        for f in parser._files_to_close_on_error:
            f.close()
        raise
    except Exception as e:  # noqa: BLE001 - as the gate's form parse
        for f in parser._files_to_close_on_error:
            f.close()
        raise BodyRefused(400, f"the multipart form cannot be read: {e}") \
            from None
    try:
        largest = max(files, default=0)
        check(text.held() + spooled() + largest, largest)
        fields = [(k, v) for k, v in form.multi_items() if isinstance(v, str)]
        try:
            check_form_fields(
                fields, path,
                inline_only=request.scope.get(SESSION_SCOPE_KEY) is not None)
        except MediaRefused as e:
            raise BodyRefused(400, str(e), e.param) from None
    except BaseException:
        await form.close()
        raise
    return form


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
    value) in RAM, priced against ``free`` bytes (the pool's figure when
    None) as they arrive, each at its own width (:class:`_TextPrice`),
    and its files written to
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
    text = _TextPrice()

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
            # The text fields stay in RAM, each priced at its own width.
            chunk = data[start:end]
            need = text.add(chunk)
            if free is not None and need > free:
                raise _text_over_free(text.read, free, need)
            part.data.extend(chunk)

        def on_part_end(self) -> None:
            if self._current_part.file is None:
                text.end()
            super().on_part_end()

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
