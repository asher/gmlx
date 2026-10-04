"""The Mac side of the clipboard stand-ins of an interactive session.

The guest's clipboard stand-ins connect to the session's clipboard socket and
send one request line: ``TYPES`` for the image types on the Mac clipboard, or
``IMAGE image/png`` for the image itself. The answer is ``OK <length>`` and
that many bytes, or ``ERR <message>``.

The server answers only after you press the client's image paste key in the
session's terminal, which the terminal relay passes to :meth:`grant`. Each
press lets the client read one image: the ``TYPES`` requests that come
before it and the one ``IMAGE`` request, whose answer ends the grant. A
``TYPES`` answer that lists no image ends it too, since there is nothing to
paste, and so does :data:`GRANT_TTL` seconds without an ``IMAGE`` request.
A second press before the end starts the time again and adds no second
image. A request without a grant gets ``ERR`` with :data:`NOT_GRANTED`. A
launch that joins the session sends its presses to the datagram socket at
``grant_path``, which lies in the session folder on the Mac, out of the
guest's reach.

The relay loop reads each request line without blocking, then hands the
socket to one worker thread and never touches it again. The worker reads the
pasteboard, which can take a while for a large image or wait on a macOS
privacy prompt, answers, and closes the socket. Relay connections therefore
never wait on the clipboard.

A guest cannot hold the Mac's file descriptors with idle connections: each
request must arrive within :data:`READ_DEADLINE`, at most
:data:`CONNECTIONS_MAX` connections are open at a time, and a request that
finds :data:`QUEUE_MAX` others waiting for the worker gets ``ERR busy``.
"""

from __future__ import annotations

import os
import queue
import selectors
import socket
import threading
import time
from typing import Callable

from .relay import CappedListener, RelayLoop, listen_socket

REQUEST_MAX = 256
IMAGE_MAX = 20 * 1024 * 1024
# The largest image in another type that the Mac converts to PNG. It covers
# an uncompressed 5K screenshot. Larger data is refused before conversion.
CONVERT_MAX = 64 * 1024 * 1024
# A TYPES request is logged the first time and then once per this many.
TYPES_LOG_EVERY = 100
SEND_TIMEOUT = 30.0
READ_DEADLINE = 5.0
CONNECTIONS_MAX = 32
QUEUE_MAX = 8
BUSY = "busy: other clipboard requests are waiting, try again"
PNG = "image/png"
# How long a press of the paste key lets the client read an image.
GRANT_TTL = 10.0
GRANT = b"GRANT"
NOT_GRANTED = ("the Mac clipboard opens only right after you press the image paste key, such "
               "as Ctrl-V, in the terminal of this session")

# Pasteboard types that hold an image, PNG first so it needs no conversion.
IMAGE_TYPES = ("public.png", "public.tiff", "public.jpeg", "com.compuserve.gif",
               "com.microsoft.bmp", "public.heic")
# NSPasteboardAccessBehaviorAlwaysDeny
_ACCESS_DENY = 3

DENIED = ("macOS denies this app access to the clipboard. Allow it in System Settings, "
          "Privacy & Security, Paste from Other Apps.")


def general_pasteboard():
    from AppKit import NSPasteboard
    return NSPasteboard.generalPasteboard()


def image_types(pasteboard) -> list[str]:
    """The image types the stand-ins offer, from the pasteboard's type list
    alone. Reading the list reads no data."""
    types = pasteboard.types() or []
    return [PNG] if any(t in types for t in IMAGE_TYPES) else []


def access_denied(pasteboard) -> bool:
    if not pasteboard.respondsToSelector_("accessBehavior"):
        return False
    return pasteboard.accessBehavior() == _ACCESS_DENY


def to_png(data: bytes) -> bytes | None:
    from AppKit import NSBitmapImageFileTypePNG, NSBitmapImageRep
    from Foundation import NSData
    rep = NSBitmapImageRep.imageRepWithData_(NSData.dataWithBytes_length_(data, len(data)))
    if rep is None:
        return None
    png = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
    return bytes(png) if png is not None else None


class TooLarge(Exception):
    """No image type on the clipboard gives a PNG within the size limits, and
    at least one type is over a limit, before or after its conversion. The
    message names the first limit that a type is over."""


class Unreadable(Exception):
    """The clipboard holds an image that the Mac cannot convert to PNG in any
    of its types. The message names the types."""


def _png_too_large(size: int) -> str:
    return (f"the image is {size / (1024 * 1024):.0f} MiB as PNG, over the "
            f"{IMAGE_MAX // (1024 * 1024)} MiB limit")


def _length(data) -> int:
    length = getattr(data, "length", None)
    return int(length()) if callable(length) else len(data)


def read_image_png(pasteboard) -> bytes | None:
    """The clipboard image as PNG, converting other image types, or None when
    the clipboard holds no image. A type that is too large, that does not
    convert, or that converts to a PNG over :data:`IMAGE_MAX` gives way to the
    next one. When no type gives a PNG, :class:`TooLarge` comes if a type was
    too large, else :class:`Unreadable`. The first size check reads the size
    of the pasteboard data alone, before the data is copied or converted."""
    types = pasteboard.types() or []
    failed = []
    too_large = None
    for kind in IMAGE_TYPES:
        if kind not in types:
            continue
        data = pasteboard.dataForType_(kind)
        if data is None:
            continue
        size = _length(data)
        if kind == "public.png" and size > IMAGE_MAX:
            too_large = too_large or _png_too_large(size)
            continue
        if kind != "public.png" and size > CONVERT_MAX:
            too_large = too_large or (
                f"the image is {size / (1024 * 1024):.0f} MiB, over the "
                f"{CONVERT_MAX // (1024 * 1024)} MiB the Mac converts to PNG")
            continue
        raw = bytes(data)
        if kind == "public.png":
            return raw
        png = to_png(raw)
        if png is None:
            failed.append(kind)
        elif len(png) > IMAGE_MAX:
            too_large = too_large or _png_too_large(len(png))
        else:
            return png
    if too_large:
        raise TooLarge(too_large)
    if failed:
        raise Unreadable(", ".join(failed))
    return None


def _ok(body: bytes) -> bytes:
    return b"OK %d\n" % len(body) + body


def _err(message: str) -> bytes:
    return f"ERR {message}\n".encode()


def send_grant(path: str) -> None:
    """Pass a press of the paste key to the session whose grant socket is
    at ``path``."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        sock.sendto(GRANT, path)


class ClipboardServer(CappedListener):
    """Serves the stand-ins' requests on ``path``. ``pasteboard`` returns the
    pasteboard to read, and tests pass a stub. ``grant_path`` is the
    datagram socket for the presses of joined launches, and ``clock`` is
    the clock of the grants, which tests replace."""

    def __init__(self, loop: RelayLoop, path: str, *,
                 pasteboard: Callable[[], object] = general_pasteboard,
                 send_timeout: float = SEND_TIMEOUT,
                 read_deadline: float = READ_DEADLINE,
                 max_connections: int = CONNECTIONS_MAX,
                 queue_max: int = QUEUE_MAX, grant_path: str | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 grant_ttl: float = GRANT_TTL):
        super().__init__(loop, listen_socket(path), "clipboard", max_connections)
        self.path = path
        self.pasteboard = pasteboard
        self.send_timeout = send_timeout
        self.read_deadline = read_deadline
        self.clock, self.grant_ttl = clock, grant_ttl
        self.types_asked = 0              # worker thread only
        self._closing = False
        self._grant_lock = threading.Lock()
        self._granted_until: float | None = None
        self.grant_path = grant_path
        self._grant_sock = _grant_socket(grant_path) if grant_path else None
        self._queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self._worker = threading.Thread(target=self._work, name="gmlx-clipboard",
                                        daemon=True)
        self._worker.start()
        loop.call_soon(self._register)
        if self._grant_sock is not None:
            loop.call_soon(self._register_grants)

    # Grants: any thread

    def grant(self) -> None:
        """You pressed the paste key, so the client may read one image."""
        with self._grant_lock:
            self._granted_until = self.clock() + self.grant_ttl

    def _granted(self, *, use: bool) -> bool:
        """Whether a grant holds now. ``use`` ends it, and so does a
        grant that has run out."""
        with self._grant_lock:
            until = self._granted_until
            held = until is not None and self.clock() < until
            if use or not held:
                self._granted_until = None
            return held

    def _end_grant(self) -> None:
        with self._grant_lock:
            self._granted_until = None

    def _register_grants(self) -> None:
        sock = self._grant_sock
        assert sock is not None
        self.loop.own(sock)
        self.loop.watch(sock, selectors.EVENT_READ, self._on_grant)

    def _on_grant(self, mask: int) -> None:
        sock = self._grant_sock
        assert sock is not None
        while True:
            try:
                data = sock.recv(64)
            except (BlockingIOError, InterruptedError):
                return
            except OSError as e:
                self.loop.log(f"clipboard: cannot read the paste key socket ({e})")
                return
            if data == GRANT:
                self.grant()

    # Loop thread

    def _cap_line(self) -> str:
        return f"clipboard: {self.open} requests are open, so new ones wait until one ends"

    def _on_accept(self, mask: int) -> None:
        while (got := self._accept()) is not None:
            conn, _ = got
            self.open += 1
            conn.setblocking(False)
            self.loop.own(conn)
            buf = bytearray()
            pending = {"conn": conn}
            self.loop.watch(conn, selectors.EVENT_READ,
                            lambda m, p=pending, b=buf: self._on_request(p, b))
            self.loop.call_later(self.read_deadline, lambda p=pending: self._expire(p))

    def _expire(self, pending: dict) -> None:
        conn = pending.pop("conn", None)
        if conn is not None:
            self.loop.log(f"clipboard: no request in {self.read_deadline:.0f} s, "
                          "connection closed")
            self._drop(conn)

    def _drop(self, conn: socket.socket) -> None:
        self.loop.unwatch(conn)
        self.loop.disown(conn)
        conn.close()
        self.released()

    def _on_request(self, pending: dict, buf: bytearray) -> None:
        conn = pending.get("conn")
        if conn is None:
            return
        try:
            data = conn.recv(REQUEST_MAX + 1 - len(buf))
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            pending.pop("conn")
            self._drop(conn)
            return
        if not data:
            pending.pop("conn")
            self._drop(conn)                    # closed before a full request
            return
        buf += data
        end = buf.find(b"\n")
        if end < 0:
            if len(buf) > REQUEST_MAX:
                self.loop.log("clipboard: request too long, connection closed")
                pending.pop("conn")
                self._drop(conn)
            return
        line = bytes(buf[:end]).decode("ascii", "replace")
        pending.pop("conn")
        if self._closing:
            self._drop(conn)
            return
        try:
            # From here the worker owns the socket.
            self._queue.put_nowait((conn, line))
        except queue.Full:
            try:
                conn.send(_err(BUSY))           # a short line fits the empty buffer
            except OSError:
                pass
            self.loop.log("clipboard: too many requests are waiting, answered busy")
            self._drop(conn)
            return
        self.loop.unwatch(conn)
        self.loop.disown(conn)

    # Worker thread

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            conn, line = item
            try:
                if self._closing:
                    continue
                conn.setblocking(True)
                conn.settimeout(self.send_timeout)
                conn.sendall(self.answer(line))
            except OSError as e:
                self.loop.log(f"clipboard: cannot answer the container ({e})")
            finally:
                conn.close()
                self.loop.call_soon(self.released)

    def answer(self, line: str) -> bytes:
        """The reply to one request line."""
        try:
            if line == "TYPES":
                if not self._granted(use=False):
                    return self._refused("which image types the Mac clipboard holds")
                self.types_asked += 1
                if self.types_asked % TYPES_LOG_EVERY == 1 or TYPES_LOG_EVERY == 1:
                    self.loop.log(f"clipboard: the container asked which image types the Mac "
                                  f"clipboard holds ({self.types_asked} times this session)")
                types = image_types(self.pasteboard())
                if not types:
                    self._end_grant()             # nothing to paste
                return _ok("".join(f"{t}\n" for t in types).encode())
            if line.startswith("IMAGE "):
                if not self._granted(use=True):
                    return self._refused("the image on the Mac clipboard")
                return self._image(line[len("IMAGE "):])
        except Exception as e:  # noqa: BLE001 - any pasteboard failure becomes a reply
            self.loop.log(f"clipboard: cannot read the Mac clipboard ({type(e).__name__}: {e})")
            # The guest gets no detail of the Mac's Python or AppKit.
            return _err("cannot read the Mac clipboard")
        return _err("unknown request")

    def _refused(self, what: str) -> bytes:
        self.loop.log(f"clipboard: refused to tell the container {what}, because the paste "
                      "key was not pressed just before")
        return _err(NOT_GRANTED)

    def _image(self, kind: str) -> bytes:
        if kind != PNG:
            return _err(f"the Mac clipboard offers images as {PNG} only, not {kind}")
        pasteboard = self.pasteboard()
        if access_denied(pasteboard):
            return _err(DENIED)
        try:
            png = read_image_png(pasteboard)
        except TooLarge as e:
            self.loop.log(f"clipboard: refused an image ({e})")
            return _err(str(e))
        except Unreadable as e:
            self.loop.log(f"clipboard: cannot convert the image on the clipboard to PNG ({e})")
            return _err("the Mac cannot read the image on the clipboard, so copy it again "
                        "in another format, such as PNG")
        if png is None:
            return _err("there is no image on the Mac clipboard")
        if len(png) > IMAGE_MAX:
            return _err(_png_too_large(len(png)))
        self.loop.event(f"clipboard: sent an image of {len(png):,} bytes")
        return _ok(png)

    def close(self) -> None:
        """Stop accepting, close the requests that wait for the worker, and
        end the worker."""
        def done():
            self.pause.closed = True
            for sock, path in ((self.sock, self.path), (self._grant_sock, self.grant_path)):
                if sock is None or path is None:
                    continue
                self.loop.unwatch(sock)
                self.loop.disown(sock)
                sock.close()
                try:
                    os.unlink(path)
                except OSError:
                    pass
        self._closing = True
        self.loop.call_soon(done)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                item = None
            if item is not None:
                item[0].close()
                continue
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                continue                  # the loop queued one more; close it too


def _grant_socket(path: str) -> socket.socket:
    """The datagram socket at ``path`` that takes the presses of joined
    launches, readable only by you."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        sock.bind(path)
        os.chmod(path, 0o600)
        sock.setblocking(False)
    except BaseException:
        sock.close()
        raise
    return sock
