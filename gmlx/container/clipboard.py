"""The Mac side of ``clipboard: images``.

The guest's clipboard stand-ins connect to the session's clipboard socket and
send one request line: ``TYPES`` for the image types on the Mac clipboard, or
``IMAGE image/png`` for the image itself. The answer is ``OK <length>`` and
that many bytes, or ``ERR <message>``.

The relay loop reads each request line without blocking, then hands the
socket to one worker thread and never touches it again. The worker reads the
pasteboard, which can take a while for a large image or wait on a macOS
privacy prompt, answers, and closes the socket. Relay connections therefore
never wait on the clipboard.
"""

from __future__ import annotations

import os
import queue
import selectors
import socket
import threading
from typing import Callable

from .relay import AcceptPause, RelayLoop, listen_socket

REQUEST_MAX = 256
IMAGE_MAX = 20 * 1024 * 1024
SEND_TIMEOUT = 30.0
PNG = "image/png"

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


def read_image_png(pasteboard) -> bytes | None:
    """The clipboard image as PNG, converting other image types, or None when
    the clipboard holds no image."""
    types = pasteboard.types() or []
    for kind in IMAGE_TYPES:
        if kind not in types:
            continue
        data = pasteboard.dataForType_(kind)
        if data is None:
            continue
        raw = bytes(data)
        return raw if kind == "public.png" else to_png(raw)
    return None


def _ok(body: bytes) -> bytes:
    return b"OK %d\n" % len(body) + body


def _err(message: str) -> bytes:
    return f"ERR {message}\n".encode()


class ClipboardServer:
    """Serves the stand-ins' requests on ``path``. ``pasteboard`` returns the
    pasteboard to read, and tests pass a stub."""

    def __init__(self, loop: RelayLoop, path: str, *,
                 pasteboard: Callable[[], object] = general_pasteboard,
                 send_timeout: float = SEND_TIMEOUT):
        self.loop, self.path = loop, path
        self.pasteboard = pasteboard
        self.send_timeout = send_timeout
        self.sock = listen_socket(path)
        self.pause = AcceptPause(loop, self.sock, self._on_accept, "clipboard")
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="gmlx-clipboard",
                                        daemon=True)
        self._worker.start()
        loop.call_soon(self._register)

    # Loop thread

    def _register(self) -> None:
        self.loop.own(self.sock)
        self.loop.watch(self.sock, selectors.EVENT_READ, self._on_accept)

    def _on_accept(self, mask: int) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except (BlockingIOError, InterruptedError):
                return
            except OSError as e:
                self.pause.failed(e)
                return
            self.pause.ok()
            conn.setblocking(False)
            self.loop.own(conn)
            buf = bytearray()
            self.loop.watch(conn, selectors.EVENT_READ,
                            lambda m, c=conn, b=buf: self._on_request(c, b))

    def _drop(self, conn: socket.socket) -> None:
        self.loop.unwatch(conn)
        self.loop.disown(conn)
        conn.close()

    def _on_request(self, conn: socket.socket, buf: bytearray) -> None:
        try:
            data = conn.recv(REQUEST_MAX + 1 - len(buf))
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._drop(conn)
            return
        if not data:
            self._drop(conn)                    # closed before a full request
            return
        buf += data
        end = buf.find(b"\n")
        if end < 0:
            if len(buf) > REQUEST_MAX:
                self.loop.log("clipboard: request too long, connection closed")
                self._drop(conn)
            return
        line = bytes(buf[:end]).decode("ascii", "replace")
        # From here the worker owns the socket.
        self.loop.unwatch(conn)
        self.loop.disown(conn)
        self._queue.put((conn, line))

    # Worker thread

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            conn, line = item
            try:
                conn.setblocking(True)
                conn.settimeout(self.send_timeout)
                conn.sendall(self.answer(line))
            except OSError as e:
                self.loop.log(f"clipboard: cannot answer the guest ({e})")
            finally:
                conn.close()

    def answer(self, line: str) -> bytes:
        """The reply to one request line."""
        try:
            if line == "TYPES":
                return _ok("".join(f"{t}\n" for t in image_types(self.pasteboard())).encode())
            if line.startswith("IMAGE "):
                return self._image(line[len("IMAGE "):])
        except Exception as e:  # noqa: BLE001 - any pasteboard failure becomes a reply
            self.loop.log(f"clipboard: cannot read the Mac clipboard ({type(e).__name__}: {e})")
            return _err(f"cannot read the Mac clipboard ({e})")
        return _err("unknown request")

    def _image(self, kind: str) -> bytes:
        if kind != PNG:
            return _err(f"the Mac clipboard offers images as {PNG} only, not {kind}")
        pasteboard = self.pasteboard()
        if access_denied(pasteboard):
            return _err(DENIED)
        png = read_image_png(pasteboard)
        if png is None:
            return _err("there is no image on the Mac clipboard")
        if len(png) > IMAGE_MAX:
            mb = len(png) / (1024 * 1024)
            return _err(f"the image is {mb:.0f} MB as PNG, over the 20 MB limit")
        self.loop.log(f"clipboard: sent an image of {len(png):,} bytes")
        return _ok(png)

    def close(self) -> None:
        """Stop accepting and end the worker once its queue is empty."""
        def done():
            self.pause.closed = True
            self.loop.unwatch(self.sock)
            self.loop.disown(self.sock)
            self.sock.close()
            try:
                os.unlink(self.path)
            except OSError:
                pass
        self.loop.call_soon(done)
        self._queue.put(None)
