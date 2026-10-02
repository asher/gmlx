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
import ipaddress
import itertools
import os
import selectors
import socket
import stat
import threading
import time
from typing import Callable, Union

from .text import printable

# An address is a Unix socket path or a (host, port) pair.
Address = Union[str, tuple]
Log = Callable[[str], None]

_READ, _WRITE = selectors.EVENT_READ, selectors.EVENT_WRITE
_CHUNK = 64 * 1024
# Per direction: stop reading once this much waits for the other side.
# Accept errors that last until something else frees a resource.
LASTING_ACCEPT_ERRORS = frozenset({errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ENOMEM})
ACCEPT_PAUSE = 0.1
# Each listener accepts at most this many connections a second, after a
# first burst, so a guest that opens and closes connections in a loop cannot
# keep the relay thread busy. Connections over the rate stay in the listen
# queue until the listener accepts again.
ACCEPT_RATE = 200.0
ACCEPT_BURST = 256
BUFFER_CAP = 256 * 1024
# The most connections one listener holds open at a time, so a guest cannot
# use up the supervisor's file descriptors. More stay in the listen queue
# until one closes, and once that queue is full the system refuses them.
CONNECTIONS_MAX = 256
# A relayed connection that moves no byte in either direction in this many
# seconds closes. Each one holds a connection to the target too, such as the
# gmlx server, so idle guest connections cannot use up its descriptors. Once
# bytes flow there is no deadline, so a quiet stream stays open, until one
# side ends its half. Once the target ends its half, the answer is complete,
# and the connection closes when no byte moves for this long, since the
# client may never end its own.
IDLE_DEADLINE = 30.0
# Once the client ends its half, it waits for an answer, which the target
# can take minutes to compute and send, such as a long reply that does not
# stream. So the connection waits this long with no byte before it closes,
# and a target that never answers or ends cannot hold it for ever.
ANSWER_DEADLINE = 3600.0
PROBE_TIMEOUT = 1.0
# A connect failure that means nothing listens at a Unix socket path, such
# as after the server that made it restarted. A relay with a renew hook asks
# for a new path once for each connection.
RENEW_ERRORS = frozenset({errno.ENOENT, errno.ECONNREFUSED})
# How often a relay with ``check_every`` checks that its Unix socket target
# is still there. A server that restarts removes the session sockets of its
# earlier run, and the relay then asks for a new one at once.
TARGET_CHECK_GAP = 2.0
# The most bytes of a first request head that a relay with ``check_host``
# keeps before it refuses the connection.
HEAD_MAX = BUFFER_CAP


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


class Timer:
    """A timer that :meth:`RelayLoop.call_later` set. :meth:`cancel` drops
    its function, and with it what the function holds, such as a pair."""

    __slots__ = ("loop", "due", "seq", "fn")

    def __init__(self, loop: "RelayLoop", due: float, seq: int,
                 fn: Callable[[], None]):
        self.loop, self.due, self.seq = loop, due, seq
        self.fn: Callable[[], None] | None = fn

    def __lt__(self, other: "Timer") -> bool:
        return (self.due, self.seq) < (other.due, other.seq)

    def cancel(self) -> None:
        """Drop the timer. Call this only from the loop thread."""
        if self.fn is not None:
            self.fn = None
            self.loop._cancelled()


class RelayLoop:
    """The one selectors loop of a session, in one daemon thread. Other
    threads hand it work with :meth:`call_soon`."""

    def __init__(self, log: Log | None = None, event: Log | None = None):
        # ``log`` takes lines the guest causes, which the session log writes
        # once a minute for each kind. ``event`` takes a line for every
        # event, such as each image the clipboard sends.
        self.log: Log = log or (lambda message: None)
        self.event: Log = event or self.log
        self._sel = selectors.DefaultSelector()
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._sel.register(self._wake_r, _READ, None)
        self._calls: collections.deque = collections.deque()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._owned: set = set()          # every socket to close at stop
        self._timers: list[Timer] = []    # a heap, loop thread only
        self._dropped = 0                 # cancelled timers still in the heap
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

    def call_later(self, delay: float, fn: Callable[[], None]) -> Timer:
        """Run ``fn`` in the loop thread after ``delay`` seconds. The loop
        keeps the timers itself and starts no thread for them."""
        timer = Timer(self, time.monotonic() + delay, next(self._seq), fn)

        def add() -> None:
            if timer.fn is not None:
                heapq.heappush(self._timers, timer)
        if threading.current_thread() is self._thread:
            add()
        else:
            self.call_soon(add)
        return timer

    def _cancelled(self) -> None:
        # A cancelled timer stays in the heap until it is due, so the heap
        # is rebuilt once most of it is cancelled timers.
        self._dropped += 1
        if self._dropped > 64 and 2 * self._dropped > len(self._timers):
            self._timers = [t for t in self._timers if t.fn is not None]
            heapq.heapify(self._timers)
            self._dropped = 0

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
            due = self._timers[0].due
            now = time.monotonic()
            if due > now:
                return due - now
            timer = heapq.heappop(self._timers)
            fn, timer.fn = timer.fn, None
            if fn is None:
                self._dropped = max(0, self._dropped - 1)
                continue
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
    """Handles accept errors and the accept rate for one listener. An error
    that lasts, such as running out of file descriptors, is logged once per
    run of failures and stops watching the listener for 100 ms, so the loop
    does not spin on it and the log does not grow with one line per attempt.
    Past :data:`ACCEPT_RATE` accepts a second the listener also stops being
    watched until the next accept is due."""

    # Seconds between two notices that the accept rate was reached.
    RATE_NOTICE_GAP = 60.0

    def __init__(self, loop: RelayLoop, sock: socket.socket,
                 callback: Callable[[int], None], name: str, *,
                 rate: float = ACCEPT_RATE, burst: int = ACCEPT_BURST):
        self.loop, self.sock, self.callback, self.name = loop, sock, callback, name
        self.logged = False
        self.closed = False
        self.rate, self.burst = rate, burst
        self.now: Callable[[], float] = time.monotonic       # tests replace it
        self.tokens = float(burst)
        self.stamp = self.now()
        self.rate_noticed: float | None = None

    def take(self) -> bool:
        """Whether the listener may accept one more connection now. When it
        may not, it stops being watched until it may, and False returns."""
        now = self.now()
        self.tokens = min(float(self.burst), self.tokens + (now - self.stamp) * self.rate)
        self.stamp = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        if self.rate_noticed is None or now - self.rate_noticed >= self.RATE_NOTICE_GAP:
            self.rate_noticed = now
            self.loop.log(f"{self.name}: more than {self.rate:.0f} new connections a "
                          "second, so new ones wait")
        self.loop.unwatch(self.sock)
        self.loop.call_later((1.0 - self.tokens) / self.rate, self._resume)
        return False

    def refund(self) -> None:
        """Give back the accept :meth:`take` allowed when no connection was
        waiting after all. The next :meth:`take` caps the count at the burst."""
        self.tokens += 1.0

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
                 idle_deadline: float | None = IDLE_DEADLINE,
                 idle_until_head: bool = False,
                 answer_deadline: float = ANSWER_DEADLINE,
                 host: str | None = None):
        self.loop, self.down, self.name, self.owner = loop, down, name, owner
        # With ``host``, the pair connects to the target only once the first
        # request head names this host. Until then ``checking`` is set.
        self.host = host
        self.checking = host is not None
        self.refused = False
        self.targets = list(targets)
        self.tried: list = []
        self.last_errno: int | None = None
        self.renewed = False
        self.up: socket.socket | None = None
        self.connecting = False
        self.to_up = bytearray()
        self.to_down = bytearray()
        self.down_eof = self.up_eof = False
        self.up_shut = self.down_shut = False
        # The target side failed, so the client's bytes are read and dropped.
        self.discard = False
        self.closed = False
        self.last_error = "no address to connect to"
        # A byte went one way or the other, or with ``idle_until_head`` a
        # whole request head came from the client.
        self.moved = False
        self.idle_deadline = idle_deadline
        self.answer_deadline = answer_deadline
        self.last = time.monotonic()      # when a byte last moved
        self.until_head = idle_until_head
        self.head_tail = b""
        self.idle_timer: Timer | None = None
        self.quiet_timer: Timer | None = None
        down.setblocking(False)
        loop.own(down)
        if self.checking:
            self._update()
        else:
            self._start()
        # The deadline starts only once the connect ran, so a pair that
        # closed above is never released a second time.
        if idle_deadline is not None and not self.closed:
            self.idle_timer = loop.call_later(idle_deadline, self._expire)

    def _start(self) -> None:
        """Connect to the target."""
        try:
            self._connect_next()
        except OSError as e:
            # The pair closes here, which frees its slot once.
            self.loop.log(f"{self.name}: cannot relay a connection ({e})")
            self.close()

    def _check_head(self) -> None:
        """Connect to the target once the client's first request head is
        whole and names the expected host, and refuse the connection when it
        names another. No byte reaches the target before that. A browser
        opens a connection of its own for each host name, so the first head
        of a connection is enough."""
        end = self.to_up.find(b"\r\n\r\n")
        if end < 0:
            if self.down_eof:
                self.close()
            elif len(self.to_up) >= HEAD_MAX:
                self._refuse(431, "Request Header Fields Too Large",
                             "The request head is too large.", "a request head that is too "
                             "large")
            return
        hosts = _header_values(bytes(self.to_up[:end]), b"host")
        if len(hosts) == 1 and hosts[0].lower() == (self.host or "").lower():
            self.checking = False
            self.moved = True
            self._start()
            return
        if len(hosts) == 1:
            shown = f"Host: {printable(hosts[0][:100])}"
        else:
            shown = f"{len(hosts)} Host headers" if hosts else "no Host header"
        self._refuse(421, "Misdirected Request",
                     f"This app answers only at http://{self.host}/. Open that address.",
                     f"a request for another host name ({shown})")

    def _refuse(self, status: int, reason: str, body: str, what: str) -> None:
        """Answer with a short HTTP refusal and close once it is sent. The
        client's bytes are dropped, and no target connection is made."""
        self.checking = False
        self.refused = self.discard = True
        self.to_up.clear()
        self.up_eof = self.up_shut = True
        text = body.encode() + b"\n"
        self.to_down += (f"HTTP/1.1 {status} {reason}\r\n"
                         "Content-Type: text/plain; charset=utf-8\r\n"
                         f"Content-Length: {len(text)}\r\n"
                         "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
                         ).encode() + text
        self.loop.log(f"{self.name}: refused {what}. The app answers only at "
                      f"http://{self.host}/.")
        self._one_side_ended()

    def _expire(self) -> None:
        if not self.closed and not self.moved:
            self.close()

    def _quiet_limit(self) -> float | None:
        """How long the pair may stay with no byte moving, once a side has
        ended its half. While only the client has ended, an answer is
        pending, so the limit is the long answer deadline. Once the target
        has ended, the answer is complete and the idle deadline applies."""
        if self.idle_deadline is None:
            return None
        if self.up_eof:
            return self.idle_deadline
        if self.down_eof:
            return max(self.idle_deadline, self.answer_deadline)
        return None

    def _one_side_ended(self) -> None:
        """Start the quiet deadline for the side that just ended its half.
        A second end shortens the limit, so it replaces the timer."""
        limit = self._quiet_limit()
        if limit is not None:
            if self.quiet_timer is not None:
                self.quiet_timer.cancel()
            self.quiet_timer = self.loop.call_later(limit, self._expire_quiet)

    def _expire_quiet(self) -> None:
        """Close the pair when no byte moved for the current limit, else
        wait until that time has passed since the last byte."""
        limit = self._quiet_limit()
        if self.closed or limit is None:
            return
        quiet = time.monotonic() - self.last
        if quiet >= limit:
            self.close()
        else:
            self.quiet_timer = self.loop.call_later(limit - quiet, self._expire_quiet)

    def _note_down(self, data: bytes) -> None:
        """Mark the pair as moving: at the first byte from the client, or
        with ``until_head`` once the client sent a whole request head."""
        if self.moved:
            return
        if not self.until_head:
            self.moved = True
            return
        seen = self.head_tail + data
        if b"\r\n\r\n" in seen:
            self.moved = True
        self.head_tail = seen[-3:]

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
                self.last_errno = None
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
            self.last_errno = rc
        if (self.owner is not None and self.owner.renew is not None and not self.renewed
                and self.last_errno in RENEW_ERRORS):
            # The pair waits, with its client's bytes in the kernel buffer,
            # until the relay has a new path.
            self.renewed = True
            self.owner.renew_for(self, self.tried[-1])
            return
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
        elif self.refused and not self.to_down and not self.down_shut:
            self.down_shut = True
            try:
                self.down.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        # After the target side failed, the pair waits for the client to end
        # too, so that a client still sending can read the whole answer.
        if self.up_shut and self.down_shut and (self.down_eof or not self.discard):
            self.close()

    def _recv(self, sock: socket.socket) -> bytes | None:
        try:
            return sock.recv(_CHUNK)
        except (BlockingIOError, InterruptedError):
            return None

    def _on_down(self, mask: int) -> None:
        # One select call can return events for both sides, and the event
        # that ran first can have closed the pair.
        if self.closed:
            return
        try:
            if mask & _READ:
                data = self._recv(self.down)
                if data == b"":
                    self.down_eof = True
                    self._one_side_ended()
                elif data and not self.discard:
                    self.to_up += data
                    self.last = time.monotonic()
                    if not self.checking:
                        self._note_down(data)
            if mask & _WRITE and self.to_down:
                sent = self.down.send(self.to_down)
                del self.to_down[:sent]
        except OSError:
            self.close()
            return
        if self.checking:
            self._check_head()
            if self.closed:
                return
        self._half_close()
        self._update()

    def _on_up(self, mask: int) -> None:
        if self.closed:
            return
        assert self.up is not None
        if self.connecting:
            err = self.up.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err:
                self.loop.unwatch(self.up)
                self.loop.disown(self.up)
                self.up.close()
                self.up, self.connecting = None, False
                self.last_error = os.strerror(err)
                self.last_errno = err
                self._connect_next()
                return
            self.connecting = False
            self._reached()
        if mask & _READ:
            try:
                data = self._recv(self.up)
            except OSError:
                self._up_failed()
                data = b""
            if data == b"":
                self.up_eof = True
                self._one_side_ended()
            elif data:
                self.to_down += data
                self.last = time.monotonic()
                if not self.until_head:
                    self.moved = True
        if mask & _WRITE and self.to_up:
            try:
                del self.to_up[:self.up.send(self.to_up)]
            except (BlockingIOError, InterruptedError):
                pass
            except OSError:
                self._up_failed()
        self._half_close()
        self._update()

    def _up_failed(self) -> None:
        """A send to the target or a read from it failed, such as when a
        server answers 503 and closes before it reads the body. The client's
        bytes can no longer reach the target, so the pair drops them, and the
        client can finish its send. The answer the target sent still goes to
        the client, and then the client's half ends."""
        self.to_up.clear()
        self.up_shut = self.discard = True

    def close(self) -> None:
        """Close both sides. The pair drops its timers and buffers, so a
        closed pair holds no memory until a timer would have run."""
        if self.closed:
            return
        self.closed = True
        for timer in (self.idle_timer, self.quiet_timer):
            if timer is not None:
                timer.cancel()
        self.idle_timer = self.quiet_timer = None
        self.to_up.clear()
        self.to_down.clear()
        self.targets, self.tried = [], []
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
    failures to reach the target is logged once.

    A connection that moves no byte within ``idle_deadline`` seconds is
    closed. With ``idle_until_head`` it must instead send a whole HTTP
    request head, ending in an empty line, in that time. After the client
    ends its half, the connection closes once no byte moved for
    ``answer_deadline`` seconds, and after the target ends its half, for
    ``idle_deadline`` seconds. At most
    ``accept_rate`` connections a second are accepted, after a first
    ``accept_burst``.

    With ``check_host``, the relay reads the first request head of each
    connection before it connects to the target. When the head's Host header
    is not the listen address, such as ``[::1]:3100``, compared without
    case, the client gets a short HTTP refusal and the connection closes. No
    byte reaches the target then. So a page cannot load the app under
    another host name, such as localhost, that reaches the same port.

    ``renew``, when given, returns a new target address or None, and may
    block. When a connection finds nothing listening at the target, it runs
    in a thread of its own, and the connection tries the new address once.
    Connections that fail while it runs wait for the same answer. With
    ``check_every`` as well, the relay checks at that interval that its Unix
    socket target is still there, and runs ``renew`` when it is not, so it
    does not wait for a connection to fail."""

    def __init__(self, loop: RelayLoop, listen: Address,
                 connect: Address | list, *, name: str | None = None,
                 max_connections: int = CONNECTIONS_MAX,
                 idle_deadline: float | None = IDLE_DEADLINE,
                 idle_until_head: bool = False,
                 answer_deadline: float = ANSWER_DEADLINE,
                 accept_rate: float = ACCEPT_RATE, accept_burst: int = ACCEPT_BURST,
                 renew: Callable[[], Address | None] | None = None,
                 check_every: float | None = None, check_host: bool = False):
        self.loop = loop
        self.renew = renew
        self.renewing = False
        self.check_every = check_every if renew is not None else None
        self.waiting: list[_Pair] = []
        self.idle_until_head = idle_until_head
        self.answer_deadline = answer_deadline
        self.listen = listen
        self.targets = list(connect) if isinstance(connect, list) else [connect]
        self.name = name or _describe(listen)
        self.max_connections = max_connections
        self.idle_deadline = idle_deadline
        self.open = 0
        self.full = False
        self.cap_logged = False
        self.failing = False
        # A listener on a loopback address serves only this Mac. A peer with
        # another address came through a forwarding rule, such as the one a
        # localhost DNS domain of Apple container adds for its guests.
        self.loopback_only = not isinstance(listen, str) and listen[0] in ("127.0.0.1", "::1")
        self.sock = listen_socket(listen)
        # The Host header a browser sends for the listen address.
        self.host = (_describe(self.sock.getsockname()[:2])
                     if check_host and not isinstance(listen, str) else None)
        self.pause = AcceptPause(loop, self.sock, self._on_accept, self.name,
                                 rate=accept_rate, burst=accept_burst)
        loop.call_soon(self._register)

    def _register(self) -> None:
        self.loop.own(self.sock)
        self.loop.watch(self.sock, _READ, self._on_accept)
        if self.check_every is not None:
            self.loop.call_later(self.check_every, self._check_target)

    def _check_target(self) -> None:
        """Ask for a new target when the Unix socket the relay connects to
        is gone, such as after the server restarted."""
        if self.pause.closed or self.check_every is None:
            return
        target = self.targets[0] if len(self.targets) == 1 else None
        if isinstance(target, str) and not self.renewing and not _is_socket(target):
            self._renew()
        self.loop.call_later(self.check_every, self._check_target)

    def _on_accept(self, mask: int) -> None:
        while True:
            if self.open >= self.max_connections:
                if not self.cap_logged:
                    self.loop.log(f"{self.name}: {self.open} connections are open, so new "
                                  "ones wait until one closes")
                self.cap_logged = self.full = True
                self.loop.unwatch(self.sock)
                return
            if not self.pause.take():
                return
            try:
                conn, peer = self.sock.accept()
            except (BlockingIOError, InterruptedError):
                self.pause.refund()
                return
            except OSError as e:
                self.pause.failed(e)
                return
            self.pause.ok()
            if self.loopback_only and not _loopback_peer(peer):
                self.loop.log(f"{self.name}: refused a connection from outside this Mac's "
                              f"loopback addresses ({_describe(peer)}). A container can "
                              "connect this way through a localhost DNS domain, so remove "
                              "such a domain if no container needs it.")
                conn.close()
                continue
            self.open += 1
            try:
                _Pair(self.loop, conn, self.targets, self.name, owner=self,
                      idle_deadline=self.idle_deadline,
                      idle_until_head=self.idle_until_head,
                      answer_deadline=self.answer_deadline, host=self.host)
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

    def renew_for(self, pair: _Pair, failed: Address) -> None:
        """Get ``pair`` a new target after ``failed`` did not answer."""
        if self.targets != [failed]:
            # Another connection already got a new target.
            pair.targets = list(self.targets)
            pair._connect_next()
            return
        self.waiting.append(pair)
        self._renew()

    def _renew(self) -> None:
        """Run ``renew`` in a thread of its own, unless it runs already."""
        if self.renewing:
            return
        self.renewing = True
        renew = self.renew
        assert renew is not None

        def work() -> None:
            try:
                target = renew()
            except Exception as e:  # noqa: BLE001 - the waiting connections close instead
                self.loop.log(f"{self.name}: cannot get a new address ({type(e).__name__}: "
                              f"{e})")
                target = None
            self.loop.call_soon(lambda: self._renewed(target))

        threading.Thread(target=work, name=f"{self.name} renew", daemon=True).start()

    def _renewed(self, target: Address | None) -> None:
        self.renewing = False
        waiting, self.waiting = self.waiting, []
        if target is not None:
            self.targets = [target]
        for pair in waiting:
            if not pair.closed:
                pair.targets = [target] if target is not None else []
                pair._connect_next()

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


def _header_values(head: bytes, name: bytes) -> list[str]:
    """The values of each header line ``name`` in a request head, without
    the white space around them. The request line is not a header."""
    values = []
    for line in head.split(b"\r\n")[1:]:
        key, sep, value = line.partition(b":")
        if sep and key.lower() == name:
            values.append(value.strip(b" \t").decode("latin-1"))
    return values


def _loopback_peer(peer) -> bool:
    """Whether ``peer``, the address of an accepted connection, is a loopback
    address, in its IPv4-mapped form too."""
    try:
        ip = ipaddress.ip_address(str(peer[0]).split("%", 1)[0])
    except (ValueError, TypeError, IndexError):
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def _is_socket(path: str) -> bool:
    try:
        return stat.S_ISSOCK(os.lstat(path).st_mode)
    except OSError:
        return False


def loopback_targets(port: int) -> list:
    """The Mac address a forwarded port reaches: ``127.0.0.1`` only. Launch
    never falls back to ``::1``, because another program can listen on the
    same port there."""
    return [("127.0.0.1", port)]


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
