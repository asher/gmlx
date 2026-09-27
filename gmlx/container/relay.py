"""The Mac side of the container socket relays.

One :class:`RelayLoop` runs a single ``selectors`` loop in one daemon thread
and serves every listener of a launch session: the API socket, a browser
app's web port, each forwarded port and the clipboard socket. A
:class:`Relay` accepts on one address and joins each connection to a
connection it opens to its target. Reads pause while the other side's buffer
is full, and an end of file in one direction becomes a half-close of the
other side. Nothing in the loop blocks.
"""

from __future__ import annotations

import collections
import errno
import heapq
import itertools
import os
import selectors
import socket
import threading
import time
from typing import Callable, Union

# An address is a Unix socket path or a (host, port) pair.
Address = Union[str, tuple]
Log = Callable[[str], None]

_READ, _WRITE = selectors.EVENT_READ, selectors.EVENT_WRITE
_CHUNK = 64 * 1024
# Per direction: stop reading once this much waits for the other side.
# Accept errors that last until something else frees a resource.
LASTING_ACCEPT_ERRORS = frozenset({errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ENOMEM})
ACCEPT_PAUSE = 0.1
BUFFER_CAP = 256 * 1024
# The most connections one listener holds open at a time. More wait in the
# listen queue until one closes, so a guest cannot use up the supervisor's
# file descriptors.
CONNECTIONS_MAX = 256
# A relayed connection that moves no byte in either direction in this many
# seconds closes. Each one holds a connection to the target too, such as the
# gmlx server, so idle guest connections cannot use up its descriptors. Once
# bytes flow there is no deadline, so a quiet stream stays open.
IDLE_DEADLINE = 30.0
PROBE_TIMEOUT = 1.0


def _describe(addr: Address) -> str:
    if isinstance(addr, str):
        return addr
    host, port = addr[0], addr[1]
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _family(addr: Address) -> int:
    if isinstance(addr, str):
        return socket.AF_UNIX
    return socket.AF_INET6 if ":" in addr[0] else socket.AF_INET


def answering(addr: tuple) -> bool:
    """Whether a program accepts connections at ``addr``."""
    try:
        sock = socket.socket(_family(addr), socket.SOCK_STREAM)
    except OSError:
        return False
    with sock:
        sock.settimeout(PROBE_TIMEOUT)
        return sock.connect_ex(addr) == 0


def _other_loopback(addr: tuple) -> tuple:
    return ("::1" if addr[0] == "127.0.0.1" else "127.0.0.1", addr[1])


def listen_socket(addr: Address, backlog: int = 128) -> socket.socket:
    """A bound, listening, non-blocking socket. A Unix path is replaced when a
    stale socket file sits there.

    A loopback TCP port binds without ``SO_REUSEADDR`` first, which fails
    while another program listens on that port on any address, such as
    0.0.0.0 or ::. Only then does launch ask whether a program answers. When
    none does, the port holds only closed connections in TIME_WAIT, and the
    bind repeats with ``SO_REUSEADDR``. Once bound, the other loopback
    address is checked too, since a program that listens only on ::1 would
    take the traffic of a browser that tries ::1 first."""
    if isinstance(addr, str):
        try:
            os.unlink(addr)
        except FileNotFoundError:
            pass
        return _bind(addr, backlog, reuse=False)
    loopback = addr[0] in ("127.0.0.1", "::1")
    try:
        sock = _bind(addr, backlog, reuse=not loopback)
    except OSError as e:
        if not loopback or e.errno != errno.EADDRINUSE:
            raise
        if answering(addr):
            raise OSError(errno.EADDRINUSE,
                          f"another program answers on {_describe(addr)}") from None
        sock = _bind(addr, backlog, reuse=True)
    if loopback:
        other = _other_loopback((addr[0], sock.getsockname()[1]))
        if answering(other):
            sock.close()
            raise OSError(errno.EADDRINUSE, f"another program answers on {_describe(other)}")
    return sock


def _bind(addr: Address, backlog: int, *, reuse: bool) -> socket.socket:
    sock = socket.socket(_family(addr), socket.SOCK_STREAM)
    try:
        if reuse:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(addr)
        sock.listen(backlog)
        sock.setblocking(False)
    except OSError:
        sock.close()
        raise
    return sock


class RelayLoop:
    """The one selectors loop of a session, in one daemon thread. Other
    threads hand it work with :meth:`call_soon`."""

    def __init__(self, log: Log | None = None):
        self.log: Log = log or (lambda message: None)
        self._sel = selectors.DefaultSelector()
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._sel.register(self._wake_r, _READ, None)
        self._calls: collections.deque = collections.deque()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._owned: set = set()          # every socket to close at stop
        self._timers: list = []           # (due, seq, fn), loop thread only
        self._seq = itertools.count()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gmlx-relay",
                                        daemon=True)
        self._thread.start()

    def call_soon(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` in the loop thread."""
        self._calls.append(fn)
        try:
            self._wake_w.send(b"x")
        except (BlockingIOError, OSError):
            pass                          # a wake byte is already pending

    def call_later(self, delay: float, fn: Callable[[], None]) -> None:
        """Run ``fn`` in the loop thread after ``delay`` seconds. The loop
        keeps the timers itself and starts no thread for them."""
        due = time.monotonic() + delay

        def add() -> None:
            heapq.heappush(self._timers, (due, next(self._seq), fn))
        if threading.current_thread() is self._thread:
            add()
        else:
            self.call_soon(add)

    def stop(self, timeout: float = 5.0) -> None:
        """End the loop and close every socket it serves."""
        self._stopping = True
        self.call_soon(lambda: None)
        if self._thread is not None:
            self._thread.join(timeout)
        for sock in list(self._owned):
            try:
                sock.close()
            except OSError:
                pass
        self._owned.clear()
        self._sel.close()
        self._wake_r.close()
        self._wake_w.close()

    # Loop-thread API: call these only from the loop thread.
    def own(self, sock: socket.socket) -> None:
        self._owned.add(sock)

    def disown(self, sock: socket.socket) -> None:
        self._owned.discard(sock)

    def watch(self, sock: socket.socket, events: int,
              callback: Callable[[int], None]) -> None:
        """Set the events ``callback(mask)`` runs for; 0 stops watching."""
        try:
            key = self._sel.get_key(sock)
        except (KeyError, ValueError):
            key = None
        if events == 0:
            if key is not None:
                self._sel.unregister(sock)
        elif key is None:
            self._sel.register(sock, events, callback)
        elif key.events != events or key.data is not callback:
            self._sel.modify(sock, events, callback)

    def unwatch(self, sock: socket.socket) -> None:
        self.watch(sock, 0, lambda mask: None)

    def is_watched(self, sock: socket.socket) -> bool:
        try:
            self._sel.get_key(sock)
            return True
        except (KeyError, ValueError):
            return False

    def _run_due_timers(self) -> float | None:
        """Run the timers that are due, and return the wait until the next."""
        while self._timers:
            due = self._timers[0][0]
            now = time.monotonic()
            if due > now:
                return due - now
            _, _, fn = heapq.heappop(self._timers)
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - one bad timer must not end the loop
                self.log(f"relay loop: {type(e).__name__}: {e}")
        return None

    def _run(self) -> None:
        while not self._stopping:
            wait = self._run_due_timers()
            try:
                events = self._sel.select(wait)
            except OSError as e:
                self.log(f"relay loop: select failed ({e})")
                return
            for key, mask in events:
                if key.fileobj is self._wake_r:
                    try:
                        while self._wake_r.recv(4096):
                            pass
                    except (BlockingIOError, OSError):
                        pass
                    continue
                try:
                    key.data(mask)
                except Exception as e:  # noqa: BLE001 - one bad connection must not end the loop
                    self.log(f"relay loop: {type(e).__name__}: {e}")
            while self._calls:
                fn = self._calls.popleft()
                try:
                    fn()
                except Exception as e:  # noqa: BLE001 - as above
                    self.log(f"relay loop: {type(e).__name__}: {e}")


class AcceptPause:
    """Handles accept errors for one listener. An error that lasts, such as
    running out of file descriptors, is logged once per run of failures and
    stops watching the listener for 100 ms, so the loop does not spin on it
    and the log does not grow with one line per attempt."""

    def __init__(self, loop: RelayLoop, sock: socket.socket,
                 callback: Callable[[int], None], name: str):
        self.loop, self.sock, self.callback, self.name = loop, sock, callback, name
        self.logged = False
        self.closed = False

    def ok(self) -> None:
        self.logged = False

    def failed(self, e: OSError) -> None:
        lasting = e.errno in LASTING_ACCEPT_ERRORS
        if not (lasting and self.logged):
            self.loop.log(f"{self.name}: accept failed ({e})")
        if lasting:
            self.logged = True
            self.loop.unwatch(self.sock)
            self.loop.call_later(ACCEPT_PAUSE, self._resume)

    def _resume(self) -> None:
        if not self.closed:
            self.loop.watch(self.sock, _READ, self.callback)


def _upstream_socket(family: int) -> socket.socket:
    """A socket for the target side of a pair. Tests replace it."""
    return socket.socket(family, socket.SOCK_STREAM)


class _Pair:
    """One accepted connection joined to its upstream connection."""

    def __init__(self, loop: RelayLoop, down: socket.socket, targets: list,
                 name: str, owner: "Relay | None" = None,
                 idle_deadline: float | None = IDLE_DEADLINE):
        self.loop, self.down, self.name, self.owner = loop, down, name, owner
        self.targets = list(targets)
        self.tried: list = []
        self.up: socket.socket | None = None
        self.connecting = False
        self.to_up = bytearray()
        self.to_down = bytearray()
        self.down_eof = self.up_eof = False
        self.up_shut = self.down_shut = False
        self.closed = False
        self.last_error = "no address to connect to"
        self.moved = False                # a byte went one way or the other
        down.setblocking(False)
        loop.own(down)
        if idle_deadline is not None:
            loop.call_later(idle_deadline, self._expire)
        self._connect_next()

    def _expire(self) -> None:
        if not self.closed and not self.moved:
            self.close()

    def _connect_next(self) -> None:
        while self.targets:
            addr = self.targets.pop(0)
            self.tried.append(addr)
            try:
                sock = _upstream_socket(_family(addr))
            except OSError as e:
                # Such as no free file descriptor. The pair closes below
                # when no address is left, which frees its slot.
                self.last_error = e.strerror or str(e)
                continue
            sock.setblocking(False)
            try:
                rc = sock.connect_ex(addr)
            except OSError as e:
                rc = e.errno or errno.ECONNREFUSED
            if rc in (0, errno.EINPROGRESS, errno.EAGAIN):
                self.up, self.connecting = sock, rc != 0
                self.loop.own(sock)
                if rc == 0:
                    self._reached()
                self._update()
                return
            sock.close()
            self.last_error = os.strerror(rc)
        where = " or ".join(_describe(a) for a in self.tried)
        message = f"{self.name}: cannot reach {where} ({self.last_error})"
        if self.owner is not None:
            self.owner.unreachable(message)
        else:
            self.loop.log(message)
        self.close()

    def _reached(self) -> None:
        if self.owner is not None:
            self.owner.reached()

    def _update(self) -> None:
        if self.closed:
            return
        down_ev = 0
        if not self.down_eof and len(self.to_up) < BUFFER_CAP:
            down_ev |= _READ
        if self.to_down:
            down_ev |= _WRITE
        self.loop.watch(self.down, down_ev, self._on_down)
        if self.up is None:
            return
        if self.connecting:
            self.loop.watch(self.up, _WRITE, self._on_up)
            return
        up_ev = 0
        if not self.up_eof and len(self.to_down) < BUFFER_CAP:
            up_ev |= _READ
        if self.to_up:
            up_ev |= _WRITE
        self.loop.watch(self.up, up_ev, self._on_up)

    def _half_close(self) -> None:
        """Pass each end of file on once its buffer has drained, and close
        the pair when both directions are done."""
        if self.up is not None and not self.connecting:
            if self.down_eof and not self.to_up and not self.up_shut:
                self.up_shut = True
                try:
                    self.up.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
            if self.up_eof and not self.to_down and not self.down_shut:
                self.down_shut = True
                try:
                    self.down.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
        if self.up_shut and self.down_shut:
            self.close()

    def _recv(self, sock: socket.socket) -> bytes | None:
        try:
            return sock.recv(_CHUNK)
        except (BlockingIOError, InterruptedError):
            return None

    def _on_down(self, mask: int) -> None:
        try:
            if mask & _READ:
                data = self._recv(self.down)
                if data == b"":
                    self.down_eof = True
                elif data:
                    self.to_up += data
                    self.moved = True
            if mask & _WRITE and self.to_down:
                sent = self.down.send(self.to_down)
                del self.to_down[:sent]
        except OSError:
            self.close()
            return
        self._half_close()
        self._update()

    def _on_up(self, mask: int) -> None:
        assert self.up is not None
        if self.connecting:
            err = self.up.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err:
                self.loop.unwatch(self.up)
                self.loop.disown(self.up)
                self.up.close()
                self.up, self.connecting = None, False
                self.last_error = os.strerror(err)
                self._connect_next()
                return
            self.connecting = False
            self._reached()
        try:
            if mask & _READ:
                data = self._recv(self.up)
                if data == b"":
                    self.up_eof = True
                elif data:
                    self.to_down += data
                    self.moved = True
            if mask & _WRITE and self.to_up:
                sent = self.up.send(self.to_up)
                del self.to_up[:sent]
        except OSError:
            self.close()
            return
        self._half_close()
        self._update()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for sock in (self.down, self.up):
            if sock is None:
                continue
            self.loop.unwatch(sock)
            self.loop.disown(sock)
            try:
                sock.close()
            except OSError:
                pass
        if self.owner is not None:
            self.owner.released()


class Relay:
    """Accepts on ``listen`` and joins each connection to ``connect``, a
    target address or a list of them tried in order until one accepts. The
    listener binds here, so a busy port fails before the container starts.
    At most ``max_connections`` connections are open at a time, and a run of
    failures to reach the target is logged once."""

    def __init__(self, loop: RelayLoop, listen: Address,
                 connect: Address | list, *, name: str | None = None,
                 max_connections: int = CONNECTIONS_MAX,
                 idle_deadline: float | None = IDLE_DEADLINE):
        self.loop = loop
        self.listen = listen
        self.targets = list(connect) if isinstance(connect, list) else [connect]
        self.name = name or _describe(listen)
        self.max_connections = max_connections
        self.idle_deadline = idle_deadline
        self.open = 0
        self.full = False
        self.cap_logged = False
        self.failing = False
        self.sock = listen_socket(listen)
        self.pause = AcceptPause(loop, self.sock, self._on_accept, self.name)
        loop.call_soon(self._register)

    def _register(self) -> None:
        self.loop.own(self.sock)
        self.loop.watch(self.sock, _READ, self._on_accept)

    def _on_accept(self, mask: int) -> None:
        while True:
            if self.open >= self.max_connections:
                if not self.cap_logged:
                    self.loop.log(f"{self.name}: {self.open} connections are open, so new "
                                  "ones wait until one closes")
                self.cap_logged = self.full = True
                self.loop.unwatch(self.sock)
                return
            try:
                conn, _ = self.sock.accept()
            except (BlockingIOError, InterruptedError):
                return
            except OSError as e:
                self.pause.failed(e)
                return
            self.pause.ok()
            self.open += 1
            try:
                _Pair(self.loop, conn, self.targets, self.name, owner=self,
                      idle_deadline=self.idle_deadline)
            except OSError as e:
                # A socket for the upstream side could not be made.
                self.loop.log(f"{self.name}: cannot relay a connection ({e})")
                self.loop.disown(conn)
                conn.close()
                self.released()

    # Called by each _Pair, in the loop thread.

    def released(self) -> None:
        self.open -= 1
        if self.open <= self.max_connections // 2:
            # The count fell well below the cap, so reaching it again is a
            # new run and is logged again.
            self.cap_logged = False
        if self.full and self.open < self.max_connections and not self.pause.closed:
            self.full = False
            self.loop.watch(self.sock, _READ, self._on_accept)

    def unreachable(self, message: str) -> None:
        if not self.failing:
            self.loop.log(message)
        self.failing = True

    def reached(self) -> None:
        self.failing = False

    def close(self) -> None:
        """Stop accepting; connections already joined keep running."""
        def done():
            self.pause.closed = True
            self.loop.unwatch(self.sock)
            self.loop.disown(self.sock)
            self.sock.close()
            if isinstance(self.listen, str):
                try:
                    os.unlink(self.listen)
                except OSError:
                    pass
        self.loop.call_soon(done)


def loopback_targets(port: int) -> list:
    """``127.0.0.1`` then ``::1``, for Mac services that listen on only one."""
    return [("127.0.0.1", port), ("::1", port)]


def resolve_targets(host: str, port: int) -> list:
    """Every address ``host`` resolves to, in resolver order. Launch calls
    this only for ``localhost`` and plain server hosts, before the loop runs."""
    seen = []
    for family, _, _, _, sockaddr in socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM):
        addr = (sockaddr[0], sockaddr[1])
        if family in (socket.AF_INET, socket.AF_INET6) and addr not in seen:
            seen.append(addr)
    return seen
