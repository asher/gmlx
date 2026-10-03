"""The input relay between your terminal and an interactive container session.

``container run -t`` and ``container exec -t`` read the terminal on their
standard input. Launch gives them a pseudo-terminal of their own there
instead, and copies what you type into it. Their output goes straight to
your terminal, so launch handles only your input. That lets launch see two
things that the client needs from the Mac:

- A paste can hold the Mac paths of files, which :class:`pastes.Pastes`
  places in the private home. A bracketed paste counts, and so does one
  read of input that holds nothing but file paths, as a terminal that
  sends a dragged file without the paste markers gives. Typed keys come one
  or two to a read, so typing never makes such a read.
- The key that pastes a clipboard image, such as Ctrl-V, or a paste with
  nothing in it lets the client read one image from the Mac clipboard, as
  :class:`clipboard.ClipboardServer` describes.

Every other byte passes through unchanged and in order.

The CLI leads a session of its own, with the pseudo-terminal as its
controlling terminal. Launch puts your terminal in raw mode, as the CLI did
before, and gives it back its settings when the relay closes. A resize of
your terminal is copied to the pseudo-terminal, and the system then sends
the CLI SIGWINCH. When your terminal closes, the relay closes the
pseudo-terminal, and the system sends the CLI the SIGHUP that a closed
terminal sends. The relay does the same when it fails, so a session never
stays without input.

Until the CLI sets raw mode on its pseudo-terminal, that terminal has the
settings that yours had, so a Ctrl-C while the container starts still
sends the CLI SIGINT. The echo of what you type then goes from the
pseudo-terminal to your terminal.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import re
import select
import signal
import termios
import threading
from typing import Callable

from .pastes import Pastes

# Bracketed paste: a terminal puts these around a paste once the program
# turns the mode on.
PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"
# The most bytes of a paste held until its end. A larger paste passes
# through as it is.
PASTE_MAX = 1 << 20
# The keys whose press lets the client read one clipboard image. Every
# client that pastes clipboard images reads them on Ctrl-V, and hermes also
# on Alt-V. A paste with nothing in it counts as a press too, since
# opencode, omp and hermes read a clipboard image when one comes.
PASTE_KEYS = frozenset({"ctrl+v", "alt+v"})
# An escape sequence cut off at the end of a read waits this many seconds
# for its rest. A lone Escape key waits as long.
ESC_WAIT = 0.05
# A cut-off sequence longer than this is not one the relay looks for.
SEQUENCE_MAX = 64
READ_MAX = 1 << 16
# How long :meth:`TerminalRelay.close` waits for the relay thread.
JOIN_WAIT = 5.0

Event = tuple[str, object]

# kitty keyboard protocol: CSI code[:alternates] ; modifiers[:event] ; text u
_KITTY = re.compile(rb"\x1b\[(\d+)(?::\d*)*(?:;(\d*)(?::(\d+))?(?:;[\d:]*)?)?u\Z")
# xterm modifyOtherKeys: CSI 27 ; modifiers ; code ~
_OTHER_KEYS = re.compile(rb"\x1b\[27;(\d+);(\d+)~\Z")
_MODIFIERS = ((4, "ctrl"), (2, "alt"), (1, "shift"), (8, "super"), (16, "hyper"), (32, "meta"))


# The terminal on stdin

def terminal_mode(fd: int = 0) -> list | None:
    """The settings of the terminal at ``fd``, or None without one."""
    try:
        return termios.tcgetattr(fd)
    except (termios.error, OSError):
        return None


def _in_background(fd: int) -> bool:
    """Whether launch runs in the background of the terminal at ``fd``,
    where a change to its settings would stop launch. A terminal that is
    not launch's controlling terminal has no foreground."""
    try:
        return os.tcgetpgrp(fd) != os.getpgrp()
    except OSError:
        return False


def restore_mode(mode: list, fd: int = 0) -> None:
    """Put back the terminal settings ``mode`` unless launch runs in the
    background, and drop the input that waits, as :func:`flush_input`
    does. A closed terminal takes none. The change does not wait for the
    output to drain, so a terminal that reads no more output, such as one
    that Ctrl-S paused, cannot hold launch."""
    with contextlib.suppress(termios.error, OSError):
        if not _in_background(fd):
            termios.tcsetattr(fd, termios.TCSANOW, mode)
            termios.tcflush(fd, termios.TCIFLUSH)


def flush_input(fd: int = 0) -> None:
    """Drop the input that waits on the terminal unless launch runs in the
    background. A client can send the terminal a query as it quits, and the
    answer arrives after the client stopped reading. The shell would then
    read that answer as typed input."""
    with contextlib.suppress(termios.error, OSError):
        if not _in_background(fd):
            termios.tcflush(fd, termios.TCIFLUSH)


def raw_mode(mode: list) -> list:
    """``mode`` changed as cfmakeraw(3) changes it."""
    raw = [*mode[:6], list(mode[6])]
    raw[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.ISTRIP
                | termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IXON)
    raw[1] &= ~termios.OPOST
    raw[2] = (raw[2] & ~(termios.CSIZE | termios.PARENB)) | termios.CS8
    raw[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG
                | termios.IEXTEN)
    raw[6][termios.VMIN] = 1
    raw[6][termios.VTIME] = 0
    return raw


def copy_size(source: int, target: int, last: bytes | None = None) -> bytes | None:
    """Give the terminal at ``target`` the window size of ``source`` when it
    differs from ``last``, and return that size, or None when it cannot be
    read."""
    try:
        size = fcntl.ioctl(source, termios.TIOCGWINSZ, b"\0" * 8)
    except OSError:
        return None
    if size != last:
        with contextlib.suppress(OSError):
            fcntl.ioctl(target, termios.TIOCSWINSZ, size)
    return size


def take_terminal() -> None:
    """Make standard input the controlling terminal of the session that
    the child leads. It runs in the child between fork and exec, so it
    makes one system call and touches no lock."""
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


# Your input

def key_name(seq: bytes) -> str | None:
    """The name of the V key with its modifiers that ``seq`` encodes, such
    as ``ctrl+v``, or None for any other sequence. It reads the control
    byte, ESC v, the kitty keyboard protocol and xterm's modifyOtherKeys.
    A key release does not count."""
    if seq == b"\x16":
        return "ctrl+v"
    if seq == b"\x1bv":
        return "alt+v"
    m = _KITTY.match(seq)
    if m:
        if int(m[1]) != 118 or m[3] == b"3":
            return None
        return _with_modifiers(int(m[2] or 1))
    m = _OTHER_KEYS.match(seq)
    if m and int(m[2]) in (118, 86):
        return _with_modifiers(int(m[1]))
    return None


def _with_modifiers(value: int) -> str | None:
    # Caps lock and num lock, 64 and 128, do not count.
    bits = (value - 1) & 63
    words = [word for bit, word in _MODIFIERS if bits & bit]
    return "+".join([*words, "v"]) if words else None


def escape_length(buf: bytes, i: int) -> tuple[int, bool]:
    """The length of the escape sequence at ``buf[i]``, and whether it is
    whole. A CSI sequence runs to its final byte. ESC and one other byte,
    as Alt and a key send, is whole at two bytes."""
    n = len(buf)
    if i + 1 >= n:
        return 1, False
    if buf[i + 1] == 0x1b:
        return 1, True
    if buf[i + 1] != 0x5b:                         # not "["
        return 2, True
    j = i + 2
    while j < n and 0x30 <= buf[j] <= 0x3f:        # parameter bytes
        j += 1
    while j < n and 0x20 <= buf[j] <= 0x2f:        # intermediate bytes
        j += 1
    if j >= n:
        return n - i, False
    if 0x40 <= buf[j] <= 0x7e:                     # the final byte
        return j - i + 1, True
    return j - i, True                             # not a sequence, passed as it is


class InputParser:
    """Splits your input into events, in order: ``("data", bytes)`` to pass
    on, ``("key", name)`` for a press of a key in ``keys``, which comes
    before the key's own bytes, and ``("paste", body)`` for a bracketed
    paste without its markers. A paste is held until its end
    marker. Past :data:`PASTE_MAX` bytes it passes through as it is, and a
    key in it is not a key press."""

    def __init__(self, keys: frozenset[str] = PASTE_KEYS):
        self.keys = keys
        self._held = b""                       # a cut-off escape sequence
        self._paste: bytearray | None = None   # the body of an open paste
        self._spill: bytes | None = None       # past the cap: the last bytes

    def clear(self) -> bool:
        """Whether no bytes are held and no paste is open, so the next read
        starts fresh."""
        return not self._held and self._paste is None and self._spill is None

    def waiting(self) -> bool:
        """Whether bytes wait for more input that may never come, which
        :meth:`idle` then passes on."""
        return bool(self._held or self._spill)

    def feed(self, data: bytes) -> list[Event]:
        out: list[Event] = []
        buf, self._held = self._held + data, b""
        while buf:
            if self._paste is not None:
                buf = self._in_paste(buf, out)
            elif self._spill is not None:
                buf = self._past_cap(buf, out)
            else:
                buf = self._outside(buf, out)
        return out

    def idle(self) -> list[Event]:
        """Pass on the bytes that wait: a cut-off escape sequence, such as a
        lone Escape key, and the last bytes of a paste past the cap."""
        out: list[Event] = []
        if self._held:
            out.append(("data", self._held))
            self._held = b""
        if self._spill:
            out.append(("data", self._spill))
            self._spill = b""
        return out

    def end(self) -> list[Event]:
        """Pass on every byte held, as it came, when the input ends."""
        out = self.idle()
        if self._paste is not None:
            out.append(("data", PASTE_START + bytes(self._paste)))
            self._paste = None
        return out

    def _key(self, seq: bytes, run: bytearray, out: list[Event]) -> None:
        name = key_name(seq)
        if name in self.keys:
            if run:
                out.append(("data", bytes(run)))
                run.clear()
            out.append(("key", name))

    def _outside(self, buf: bytes, out: list[Event]) -> bytes:
        """Read ``buf`` outside a paste, and return what follows a paste
        start marker, or nothing."""
        run = bytearray()
        rest = b""
        i = 0
        while i < len(buf):
            byte = buf[i]
            if byte == 0x16:
                self._key(b"\x16", run, out)
            if byte != 0x1b:
                run.append(byte)
                i += 1
                continue
            size, whole = escape_length(buf, i)
            if not whole:
                if len(buf) - i <= SEQUENCE_MAX:
                    self._held = buf[i:]
                    break
                size = len(buf) - i
            seq = buf[i:i + size]
            if seq == PASTE_START:
                self._paste = bytearray()
                rest = buf[i + size:]
                break
            self._key(seq, run, out)
            run += seq
            i += size
        if run:
            out.append(("data", bytes(run)))
        return rest

    def _in_paste(self, buf: bytes, out: list[Event]) -> bytes:
        assert self._paste is not None
        start = max(0, len(self._paste) - (len(PASTE_END) - 1))
        self._paste += buf
        end = self._paste.find(PASTE_END, start)
        if end >= 0:
            body, rest = bytes(self._paste[:end]), bytes(self._paste[end + len(PASTE_END):])
            self._paste = None
            if len(body) > PASTE_MAX:
                out.append(("data", PASTE_START + body + PASTE_END))
            else:
                out.append(("paste", body))
            return rest
        if len(self._paste) > PASTE_MAX:
            keep = len(PASTE_END) - 1
            out.append(("data", PASTE_START + bytes(self._paste[:-keep])))
            self._spill = bytes(self._paste[-keep:])
            self._paste = None
        return b""

    def _past_cap(self, buf: bytes, out: list[Event]) -> bytes:
        """Pass a paste past the cap through, up to its end marker."""
        chunk = (self._spill or b"") + buf
        end = chunk.find(PASTE_END)
        if end >= 0:
            cut = end + len(PASTE_END)
            out.append(("data", chunk[:cut]))
            self._spill = None
            return chunk[cut:]
        keep = len(PASTE_END) - 1
        if len(chunk) > keep:
            out.append(("data", chunk[:-keep]))
        self._spill = chunk[-keep:]
        return b""


# The relay

class TerminalRelay:
    """Copies your input from the terminal at ``term`` into a new
    pseudo-terminal, which the child gets as its standard input. Pass
    :attr:`slave` to the child with :func:`take_terminal` and a session of
    its own, call :meth:`start` once it runs, and :meth:`close` when it has
    exited. ``on_key`` runs for each press of a key in ``keys`` and for
    each paste with nothing in it, before the key or the paste passes on. ``pastes`` rewrites each paste, and ``log`` takes the
    relay's own lines. Raises OSError when ``term`` is not a terminal."""

    def __init__(self, *, keys: frozenset[str] = PASTE_KEYS,
                 on_key: Callable[[], None] | None = None, pastes: Pastes | None = None,
                 log: Callable[[str], None] = lambda line: None, term: int = 0,
                 out: int = 1, esc_wait: float = ESC_WAIT):
        saved = terminal_mode(term)
        if saved is None:
            raise OSError(errno.ENOTTY, "standard input is not a terminal")
        self.saved: list = saved
        self.term, self.out, self.esc_wait = term, out, esc_wait
        self.on_key, self.pastes, self.log = on_key, pastes, log
        self.parser = InputParser(keys)
        self.master, self.slave = os.openpty()
        self._fds = [self.master, self.slave]
        try:
            # Until the CLI sets raw mode, the pseudo-terminal acts as yours did.
            termios.tcsetattr(self.slave, termios.TCSANOW, saved)
            self._size = copy_size(term, self.master)
            self._wake_r, self._wake_w = os.pipe()
            self._fds += [self._wake_r, self._wake_w]
            os.set_blocking(self._wake_w, False)
        except BaseException:
            self._close_fds()
            raise
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._closed = False
        self._raw = False
        self._resize_handler: list = []

    def start(self) -> None:
        """Put your terminal in raw mode and start copying. The child holds
        its own copy of the slave, so launch closes its own."""
        self._close_fd(self.slave)
        self._raw = True
        termios.tcsetattr(self.term, termios.TCSANOW, raw_mode(self.saved))
        with contextlib.suppress(ValueError):      # only the main thread sets handlers
            self._resize_handler.append(signal.signal(signal.SIGWINCH, self._on_resize))
        self._thread = threading.Thread(target=self._run, name="gmlx-terminal", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop copying and give your terminal back its settings. Later
        calls do nothing."""
        if self._closed:
            return
        self._closed = True
        self._stopping = True
        self._wake()
        thread = self._thread
        if thread is not None:
            thread.join(JOIN_WAIT)
        if self._resize_handler:
            signal.signal(signal.SIGWINCH, self._resize_handler[0] or signal.SIG_DFL)
        if self._raw:
            restore_mode(self.saved, self.term)
        if thread is None or not thread.is_alive():
            self._close_fds()

    def _on_resize(self, signum, frame) -> None:
        self._wake()

    def _wake(self) -> None:
        with contextlib.suppress(OSError):
            os.write(self._wake_w, b"x")

    def _close_fd(self, fd: int) -> None:
        if fd in self._fds:
            self._fds.remove(fd)
            with contextlib.suppress(OSError):
                os.close(fd)

    def _close_fds(self) -> None:
        for fd in list(self._fds):
            self._close_fd(fd)

    # The relay thread

    def _run(self) -> None:
        why = "failed"
        try:
            why = self._copy()
        except Exception as e:  # noqa: BLE001 - the session must not stay without input
            self.log(f"terminal: the input relay stopped ({type(e).__name__}: {e})")
        finally:
            if why != "stopped":
                # Your terminal closed or the relay failed, so the client
                # ends as it does when its terminal closes.
                restore_mode(self.saved, self.term)
                self._close_fd(self.master)

    def _copy(self) -> str:
        """Copy until :meth:`close` asks the relay to stop, which returns
        ``stopped``, or until your terminal closes, which returns
        ``hangup``."""
        watched = [self.term, self.master, self._wake_r]
        while True:
            wait = self.esc_wait if self.parser.waiting() else None
            ready = select.select(watched, [], [], wait)[0]
            if not ready:
                self._send(self.parser.idle())
                continue
            if self._wake_r in ready:
                with contextlib.suppress(OSError):
                    os.read(self._wake_r, 512)
                if self._stopping:
                    return "stopped"
                self._size = copy_size(self.term, self.master, self._size)
            if self.master in ready:
                # The echo of what you type before the CLI sets raw mode.
                try:
                    echo = os.read(self.master, READ_MAX)
                except OSError:
                    echo = b""
                if echo:
                    with contextlib.suppress(OSError):
                        _write_all(self.out, echo)
                else:
                    watched.remove(self.master)    # the CLI has exited
            if self.term in ready:
                try:
                    data = os.read(self.term, READ_MAX)
                except BlockingIOError:
                    continue
                except OSError:
                    return "hangup"
                if not data:
                    return "hangup"
                # A resize reaches the relay as a signal, which can come after
                # the input typed once the window had its new size. Checking
                # the size before each input keeps the order.
                self._size = copy_size(self.term, self.master, self._size)
                self._send(self.parser.feed(data))

    def _send(self, events: list[Event]) -> None:
        for kind, value in events:
            if kind == "key":
                self._pressed()
            elif kind == "paste":
                assert isinstance(value, bytes)
                if not value:
                    self._pressed()
                self._write(PASTE_START + self._rewrite(value) + PASTE_END)
            else:
                assert isinstance(value, bytes)
                self._write(value)

    def _pressed(self) -> None:
        if self.on_key is None:
            return
        try:
            self.on_key()
        except Exception as e:  # noqa: BLE001 - a key press must still pass
            self.log(f"terminal: the paste key could not reach the clipboard "
                     f"({type(e).__name__}: {e})")

    def _rewrite(self, body: bytes) -> bytes:
        if self.pastes is None:
            return body
        try:
            return self.pastes.rewrite(body)
        except Exception as e:  # noqa: BLE001 - the paste passes as it came
            self.log(f"terminal: a paste passed as it came ({type(e).__name__}: {e})")
            return body

    def _write(self, data: bytes) -> None:
        # Once the CLI has exited, what you type goes nowhere.
        with contextlib.suppress(OSError):
            _write_all(self.master, data)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]
