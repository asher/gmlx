"""The Mac-side relay of `gmlx launch --container`: one selectors loop that
joins accepted connections to their targets, and the clipboard server that
shares it. Real sockets on loopback and in a temp folder; no container, no
server. The clipboard tests read a stub pasteboard, except the conversion
test, which writes a private named NSPasteboard and never the user's own."""
from __future__ import annotations

import collections
import errno
import os
import select
import socket
import threading

import pytest

from gmlx.container import clipboard, relay


def _sse_server(events: int, step=None):
    """A TCP server that answers each connection with ``events`` SSE events.
    After each event it calls ``step``, when given, and it stops when that
    returns False. Returns (port, stop)."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(64)

    def handle(conn):
        with conn:
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            try:
                for i in range(events):
                    conn.sendall(f"data: {i}\n\n".encode())
                    if step is not None and not step():
                        return
            except OSError:
                pass

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1], srv.close


def _read_events(sock, count, each=None):
    """Read up to ``count`` SSE events, and call ``each`` after each one.
    Returns the number of events read."""
    sock.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
    buf, got = b"", 0
    while got < count:
        data = sock.recv(4096)
        if not data:
            break
        buf += data
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            if b"data:" in event:
                got += 1
                if each is not None:
                    each()
    return got


def _lockstep():
    """A ``step`` for :func:`_sse_server` and an ``each`` for
    :func:`_read_events`: the server sends the next event only after the
    client read the one before. A relay that kept an event until more bytes
    came would stop the stream, and the client's read would time out."""
    read = threading.Semaphore(0)
    return (lambda: read.acquire(timeout=10)), read.release


class _Seen(list):
    """A list that other threads append to, and that a test can wait on."""

    def __init__(self):
        super().__init__()
        self._cond = threading.Condition()

    def append(self, item):
        with self._cond:
            super().append(item)
            self._cond.notify_all()

    def wait_for(self, predicate=len, timeout=10.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: predicate(self), timeout)


class _Clock:
    """A clock for the timers and deadlines of a loop that moves only when
    the test moves it."""

    def __init__(self, loop, start=1000.0):
        self.loop, self.t = loop, start
        loop.now = lambda: self.t

    def advance(self, seconds):
        """Move the clock, and wait until every timer that came due ran: a
        timer set now runs after the timers due before it."""
        ran = threading.Event()

        def move():
            self.t += seconds
            self.loop.call_later(0, ran.set)
        self.loop.call_soon(move)
        assert ran.wait(10)


def _settled(loop):
    """Wait until the loop has handled the socket events that were ready
    before this call, such as a connection that waits to be accepted. A byte
    on a socket pair that the loop watches comes in the same select call as
    those events or in a later one, and a call that its callback queues runs
    after the other events of that select call."""
    a, b = socket.socketpair()
    done = threading.Event()

    def on_read(mask):
        loop.unwatch(a)
        loop.call_soon(done.set)
    try:
        _in_loop(loop, lambda: loop.watch(a, relay._READ, on_read))
        b.send(b"x")
        assert done.wait(10)
    finally:
        a.close()
        b.close()


def _when_released(loop, owner, name="released", now=True):
    """An event that is set each time the count of open connections of
    ``owner`` falls to 0, and at once when it is 0 now and ``now`` is true.
    A clipboard server binds its release in the worker thread, so a test
    sets this up before the requests whose release it waits for."""
    done = threading.Event()
    real = getattr(owner, name)

    def released():
        real()
        if owner.open == 0:
            done.set()

    def install():
        setattr(owner, name, released)
        if now and owner.open == 0:
            done.set()
    _in_loop(loop, install)
    return done


def _recv_exactly(sock, size):
    got = b""
    while len(got) < size:
        data = sock.recv(size - len(got))
        if not data:
            break
        got += data
    return got


@pytest.fixture
def tmp_path():
    # macOS caps a Unix socket path at 104 bytes, which pytest's tmp_path
    # exceeds, so these tests use a short folder under /tmp.
    import shutil
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp(prefix="gr-", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def loop():
    logged = _Seen()
    lp = relay.RelayLoop(log=logged.append)
    lp.logged = logged
    lp.start()
    yield lp
    lp.stop()


def _unix_client(path):
    c = socket.socket(socket.AF_UNIX)
    c.settimeout(10)
    c.connect(path)
    return c


def test_sse_events_arrive_unbunched(loop, tmp_path):
    # Each event arrives on its own, and none waits for a later one.
    step, each = _lockstep()
    port, stop = _sse_server(8, step)
    path = str(tmp_path / "api.sock")
    relay.Relay(loop, path, ("127.0.0.1", port))
    with _unix_client(path) as c:
        assert _read_events(c, 8, each) == 8
    stop()


def test_twenty_concurrent_streams(loop, tmp_path):
    port, stop = _sse_server(10)
    path = str(tmp_path / "api.sock")
    relay.Relay(loop, path, ("127.0.0.1", port))
    results = []

    def one():
        with _unix_client(path) as c:
            results.append(_read_events(c, 10))

    threads = [threading.Thread(target=one) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    stop()
    assert results == [10] * 20


def test_half_close_reaches_the_target(loop, tmp_path):
    # The client sends, half-closes, and still reads the answer the target
    # writes after it sees end of file.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def echo_after_eof():
        conn, _ = srv.accept()
        with conn:
            data = b""
            while chunk := conn.recv(4096):
                data += chunk
            conn.sendall(b"got " + data)

    threading.Thread(target=echo_after_eof, daemon=True).start()
    path = str(tmp_path / "h.sock")
    relay.Relay(loop, path, ("127.0.0.1", srv.getsockname()[1]))
    with _unix_client(path) as c:
        c.sendall(b"hello")
        c.shutdown(socket.SHUT_WR)
        reply = b""
        while chunk := c.recv(4096):
            reply += chunk
    srv.close()
    assert reply == b"got hello"


def test_backpressure_bounds_the_buffer(loop, tmp_path):
    # A target that floods a client that does not read: the relay stops
    # reading once its buffer is full, so the flood stalls instead of
    # growing memory, and every byte still arrives once the client reads.
    total = 64 << 20                        # far past any kernel socket buffer
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    stalled = threading.Event()

    def flood():
        conn, _ = srv.accept()
        conn.setblocking(False)
        n = 0
        with conn:
            while n < total:
                try:
                    n += conn.send(b"x" * min(65536, total - n))
                except BlockingIOError:
                    stalled.set()
                    select.select([], [conn], [], 10)

    threading.Thread(target=flood, daemon=True).start()
    path = str(tmp_path / "b.sock")
    relay.Relay(loop, path, ("127.0.0.1", srv.getsockname()[1]))
    with _unix_client(path) as c:
        assert stalled.wait(10)                      # the target had to wait
        got = 0
        while got < total:
            chunk = c.recv(1 << 20)
            if not chunk:
                break
            got += len(chunk)
    srv.close()
    assert got == total


def test_forward_never_falls_back_to_ipv6_loopback(loop, tmp_path):
    """A program on ::1 at the forwarded port can be another one than the
    Mac service the user named, so the forward reaches 127.0.0.1 only."""
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    srv = socket.socket(socket.AF_INET6)
    try:
        srv.bind(("::1", 0))
    except OSError:
        pytest.skip("no ::1 on this machine")
    srv.listen(1)
    port = srv.getsockname()[1]
    probe = socket.socket()
    if probe.connect_ex(("127.0.0.1", port)) == 0:
        pytest.skip("the port is also open on 127.0.0.1")
    probe.close()

    assert relay.loopback_targets(port) == [("127.0.0.1", port)]
    path = str(tmp_path / "fwd.sock")
    relay.Relay(loop, path, relay.loopback_targets(port), name=f"forward {port}")
    with _unix_client(path) as c:
        assert c.recv(10) == b""                       # closed, never relayed to ::1
    srv.close()
    assert _in_loop(loop, lambda: list(loop.logged)) == [
        f"forward {port}: cannot reach 127.0.0.1:{port} (Connection refused)"]


def test_unreachable_target_closes_the_client_and_logs_once(loop, tmp_path):
    # Unix socket paths where nothing listens. A TCP port closed for the test
    # could be taken by another program, and on macOS a port that is bound
    # but not listening does not refuse: the connection times out.
    first, second = str(tmp_path / "a.sock"), str(tmp_path / "b.sock")
    path = str(tmp_path / "down.sock")
    relay.Relay(loop, path, [first, second], name="forward 5432")
    with _unix_client(path) as c:
        assert c.recv(10) == b""                       # closed at once
    assert loop.logged.wait_for()
    assert len(loop.logged) == 1
    assert loop.logged[0].startswith(f"forward 5432: cannot reach {first} or {second} (")


def test_tcp_listener_to_unix_target(loop, tmp_path):
    # The browser-app direction: a Mac TCP port joined to a Unix socket.
    path = str(tmp_path / "web.sock")
    srv = relay.listen_socket(path)
    srv.setblocking(True)

    def answer():
        conn, _ = srv.accept()
        with conn:
            conn.sendall(b"web " + conn.recv(10))

    threading.Thread(target=answer, daemon=True).start()
    r = relay.Relay(loop, ("127.0.0.1", 0), path)
    port = r.sock.getsockname()[1]
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        c.sendall(b"hi")
        assert c.recv(20) == b"web hi"
    srv.close()


def test_busy_listen_port_fails_at_construction(loop):
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    with pytest.raises(OSError):
        relay.Relay(loop, ("127.0.0.1", held.getsockname()[1]), "/nowhere")
    held.close()


def test_close_stops_accepting_and_removes_the_socket(loop, tmp_path):
    path = tmp_path / "gone.sock"
    r = relay.Relay(loop, str(path), ("127.0.0.1", 9))
    r.close()
    _in_loop(loop, lambda: None)                   # the close ran in the loop
    assert not path.exists()


def _eof_target(path):
    """A Unix socket target that echoes, with an event that end of file from
    the relay sets."""
    srv = relay.listen_socket(path)
    srv.setblocking(True)
    ended = threading.Event()

    def handle(conn):
        with conn:
            while data := conn.recv(4096):
                conn.sendall(data)
        ended.set()

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    return srv, ended


def _joined(loop, tmp_path):
    """A relay with one joined pair. Returns the relay, the client, the
    target's listener and its end-of-file event."""
    srv, ended = _eof_target(str(tmp_path / "t.sock"))
    r = relay.Relay(loop, str(tmp_path / "api.sock"), str(tmp_path / "t.sock"))
    c = _unix_client(str(tmp_path / "api.sock"))
    c.sendall(b"ping")
    assert c.recv(4) == b"ping"                    # the pair is joined
    return r, c, srv, ended


def test_stop_closes_joined_pairs_and_both_ends_get_end_of_file(loop, tmp_path):
    r, c, srv, ended = _joined(loop, tmp_path)
    with c:
        loop.stop()
        assert c.recv(10) == b""
    assert ended.wait(10)
    assert not loop._thread.is_alive() and loop._owned == set()
    loop.stop()                                    # a second stop does nothing
    srv.close()


def test_a_stop_that_times_out_leaves_the_close_to_the_loop_thread(loop, tmp_path):
    """While a callback still runs, the loop thread can still use the
    sockets, so the stop returns and the thread closes them when it ends."""
    r, c, srv, ended = _joined(loop, tmp_path)
    entered, release = threading.Event(), threading.Event()
    loop.call_soon(lambda: (entered.set(), release.wait(10)))
    assert entered.wait(10)
    loop.stop(timeout=0)
    assert loop._thread.is_alive()
    assert loop._close_at_end and not loop._closed
    assert loop._owned                             # nothing closed under the callback
    release.set()
    loop._thread.join(10)
    assert not loop._thread.is_alive() and loop._closed
    with c:
        assert c.recv(10) == b""
    assert ended.wait(10)
    srv.close()


def test_close_keeps_joined_pairs_and_takes_no_new_connection(loop, tmp_path):
    r, c, srv, ended = _joined(loop, tmp_path)
    r.close()
    _in_loop(loop, lambda: None)                   # the close ran in the loop
    assert not os.path.exists(r.listen)
    with pytest.raises(OSError):
        _unix_client(r.listen)
    with c:
        c.sendall(b"more")
        assert c.recv(4) == b"more"                # the joined pair goes on
    assert ended.wait(10)                          # and its close reaches the target
    assert _when_released(loop, r).wait(10)
    srv.close()


class _FakeLoop:
    def __init__(self):
        self.logged, self.watching, self.later = [], True, []

    def log(self, line):
        self.logged.append(line)

    def unwatch(self, sock):
        self.watching = False

    def watch(self, sock, events, callback):
        self.watching = True

    def call_later(self, delay, fn):
        self.later.append((delay, fn))

    def now(self):
        return 1000.0


def test_a_lasting_accept_error_pauses_the_listener_and_logs_once():
    lp = _FakeLoop()
    pause = relay.AcceptPause(lp, object(), lambda mask: None, "gmlx api")
    for _ in range(50):
        pause.failed(OSError(errno.EMFILE, "Too many open files"))
    assert len(lp.logged) == 1 and "accept failed" in lp.logged[0]
    assert not lp.watching and lp.later[0][0] == relay.ACCEPT_PAUSE
    lp.later[0][1]()
    assert lp.watching                                # resumed after the pause
    pause.ok()
    pause.failed(OSError(errno.EMFILE, "Too many open files"))
    assert len(lp.logged) == 2                        # a new run of failures
    pause.closed = True
    lp.watching = False
    lp.later[-1][1]()
    assert not lp.watching                            # never resumes after close


def test_the_accept_rate_pauses_the_listener_and_logs_once_a_minute():
    lp = _FakeLoop()
    pause = relay.AcceptPause(lp, object(), lambda mask: None, "forward 5432",
                              rate=10.0, burst=3)
    clock = [100.0]
    pause.now = lambda: clock[0]
    pause.tokens, pause.stamp = 3.0, clock[0]
    assert [pause.take() for _ in range(3)] == [True, True, True]
    assert lp.watching and lp.logged == []
    assert not pause.take()                            # over the rate
    assert not lp.watching and lp.later[-1][0] == pytest.approx(0.1)
    assert lp.logged == ["forward 5432: more than 10 new connections a second, so new "
                         "ones wait"]
    lp.later[-1][1]()
    assert lp.watching                                 # resumed when one is due
    clock[0] += 0.11
    assert pause.take() and not pause.take()
    assert len(lp.logged) == 1                         # at most one line a minute
    clock[0] += relay.AcceptPause.RATE_NOTICE_GAP
    pause.tokens = 0.0
    pause.stamp = clock[0]
    assert not pause.take() and len(lp.logged) == 2
    pause.refund()
    pause.refund()
    assert pause.tokens == pytest.approx(2.0)
    clock[0] += 60.0
    assert pause.take() and pause.tokens == pytest.approx(2.0)     # capped at the burst


def test_connections_over_the_accept_rate_wait_in_the_listen_queue(loop, tmp_path):
    port, stop = _echo_server()
    path = str(tmp_path / "rate.sock")
    r = relay.Relay(loop, path, ("127.0.0.1", port), name="gmlx api",
                    accept_rate=4.0, accept_burst=2)
    first, second = _unix_client(path), _unix_client(path)
    assert _echoes(first, 5) and _echoes(second, 5)
    third = _unix_client(path)                        # connects into the queue
    third.sendall(b"ping")
    assert third.recv(4) == b"ping"                   # accepted once one is due
    assert "gmlx api: more than 4 new connections a second, so new ones wait" in \
        _in_loop(loop, lambda: list(loop.logged))
    assert _in_loop(loop, lambda: r.open) == 3
    for c in (first, second, third):
        c.close()
    stop()


def test_the_clipboard_listener_keeps_the_accept_rate(loop, tmp_path):
    server, path = _server(loop, tmp_path, StubPasteboard({"public.png": PNG_BYTES}))

    def spent():
        server.pause.tokens, server.pause.stamp = 0.0, server.pause.now()
        server.pause.rate = 5.0
    _in_loop(loop, spent)
    assert _ask(path, b"TYPES\n") == b"OK 10\nimage/png\n"   # answered once one is due
    assert "clipboard: more than 5 new connections a second, so new ones wait" in \
        _in_loop(loop, lambda: list(loop.logged))


def test_a_relay_with_no_target_closes_the_connection(loop, tmp_path):
    path = str(tmp_path / "none.sock")
    relay.Relay(loop, path, [], name="gmlx api")
    c = _unix_client(path)
    assert c.recv(1) == b""
    c.close()
    assert loop.logged.wait_for()
    assert "no address to connect to" in loop.logged[0]


def test_resolve_targets_lists_localhost_addresses():
    addrs = relay.resolve_targets("localhost", 8080)
    assert addrs and all(port == 8080 for _, port in addrs)
    assert {host for host, _ in addrs} <= {"127.0.0.1", "::1"}


def _echo_server():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(64)

    def handle(conn):
        with conn:
            while data := conn.recv(4096):
                conn.sendall(data)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1], srv.close


def _echoes(c, timeout):
    c.settimeout(timeout)
    c.sendall(b"ping")
    try:
        return c.recv(4) == b"ping"
    except TimeoutError:
        return False


def test_a_listener_holds_at_most_its_cap_of_connections(loop, tmp_path):
    port, stop = _echo_server()
    path = str(tmp_path / "cap.sock")
    r = relay.Relay(loop, path, ("127.0.0.1", port), name="gmlx api", max_connections=4)
    held = [_unix_client(path) for _ in range(4)]
    assert all(_echoes(c, 5) for c in held)

    def capped(lines):
        return sum("connections are open" in line for line in lines)
    extra = _unix_client(path)                     # waits in the listen queue
    extra.sendall(b"ping")
    assert loop.logged.wait_for(lambda lines: capped(lines) == 1)
    assert _in_loop(loop, lambda: (r.open, loop.is_watched(r.sock))) == (4, False)
    held.pop().close()                             # frees a slot
    assert extra.recv(4) == b"ping"
    held.append(extra)
    # Back at the cap at once is the same run, so it is logged once.
    _settled(loop)
    assert capped(loop.logged) == 1
    for c in held:
        c.close()
    assert _when_released(loop, r).wait(10)
    # The count fell to half the cap or less, so a new run is logged again.
    again = [_unix_client(path) for _ in range(5)]
    assert all(_echoes(c, 5) for c in again[:4])
    assert loop.logged.wait_for(lambda lines: capped(lines) == 2)
    for c in again:
        c.close()
    stop()


def test_unreachable_is_logged_once_per_run_of_failures(loop, tmp_path):
    target = str(tmp_path / "service.sock")        # nothing listens yet
    path = str(tmp_path / "down.sock")
    relay.Relay(loop, path, target, name="port 5432")
    for _ in range(5):
        with _unix_client(path) as c:
            assert c.recv(10) == b""
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(target)                               # the service comes up
    srv.listen(1)
    with _unix_client(path) as c:
        conn, _ = srv.accept()
        conn.close()
        assert c.recv(10) == b""
    srv.close()
    os.unlink(target)
    with _unix_client(path) as c:                  # and goes down again
        assert c.recv(10) == b""
    _in_loop(loop, lambda: None)
    assert sum("cannot reach" in line for line in loop.logged) == 2


@pytest.mark.parametrize("family, host", [(socket.AF_INET, "127.0.0.1"),
                                          (socket.AF_INET, "0.0.0.0"),
                                          (socket.AF_INET6, "::"),
                                          (socket.AF_INET6, "::1")])
def test_a_port_another_program_answers_on_is_busy(loop, family, host):
    held = socket.socket(family)
    held.bind((host, 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        with pytest.raises(OSError, match="another program answers on"):
            relay.Relay(loop, ("127.0.0.1", port), "/nowhere")
    finally:
        held.close()


def test_a_port_in_time_wait_still_binds(loop):
    lst = socket.socket()
    lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lst.bind(("127.0.0.1", 0))
    lst.listen(1)
    port = lst.getsockname()[1]
    c = socket.create_connection(("127.0.0.1", port))
    conn, _ = lst.accept()
    conn.close()                                   # this side closes first: TIME_WAIT
    c.close()
    lst.close()
    plain = socket.socket()
    with pytest.raises(OSError):
        plain.bind(("127.0.0.1", port))            # the case the fallback covers
    plain.close()
    sock = relay.listen_socket(("127.0.0.1", port))
    sock.close()


def test_the_bound_address_is_never_probed(monkeypatch):
    """The busy check asks whether a program answers only after a bind
    fails, so no other program can take the port between the check and the
    bind. The other loopback address is still asked."""
    asked = []
    monkeypatch.setattr(relay, "answering", lambda addr: asked.append(addr) or False)
    sock = relay.listen_socket(("127.0.0.1", 0))
    sock.close()
    assert [a[0] for a in asked] == ["::1"]


def _silent_server():
    """A server that accepts, never speaks first and echoes what it gets,
    with an event that a connection closed by the other side sets."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    closed = threading.Event()

    def handle(conn):
        with conn:
            while True:
                data = conn.recv(4096)
                if not data:
                    closed.set()
                    return
                conn.sendall(data)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    return srv, closed


def test_an_idle_relayed_connection_closes_at_the_deadline(loop, tmp_path):
    srv, closed = _silent_server()
    clock = _Clock(loop)
    path = str(tmp_path / "idle.sock")
    r = relay.Relay(loop, path, srv.getsockname(), name="gmlx api", idle_deadline=30)
    c = _unix_client(path)
    _settled(loop)                                 # accepted, so the deadline runs
    clock.advance(29.9)
    assert _in_loop(loop, lambda: r.open) == 1     # open until the deadline
    clock.advance(0.1)
    assert c.recv(10) == b""                       # the relay closed it
    c.close()
    assert closed.wait(10)                         # the server side closed too
    assert _in_loop(loop, lambda: r.open) == 0
    srv.close()


def test_a_connection_that_moved_bytes_has_no_deadline(loop, tmp_path):
    srv, _closed = _silent_server()
    path = str(tmp_path / "busy.sock")
    relay.Relay(loop, path, srv.getsockname(), name="gmlx api", idle_deadline=0.3)
    c = _unix_client(path)
    assert _echoes(c, 5)
    _past_deadline(loop, 0.3)                      # quiet, past the deadline
    assert _echoes(c, 5)
    c.close()
    srv.close()


def test_a_server_that_speaks_first_keeps_the_connection(loop, tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    path = str(tmp_path / "greet.sock")
    relay.Relay(loop, path, srv.getsockname(), name="port 3306", idle_deadline=0.3)
    c = _unix_client(path)
    conn, _ = srv.accept()
    conn.sendall(b"hello")                         # such as a MySQL greeting
    assert c.recv(5) == b"hello"
    _past_deadline(loop, 0.3)
    conn.sendall(b"again")
    assert c.recv(5) == b"again"
    c.close()
    conn.close()
    srv.close()


def _mute_server():
    """A server that accepts and then never reads, writes or closes."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    kept = []

    def serve():
        while True:
            try:
                kept.append(srv.accept()[0])
            except OSError:
                return
    threading.Thread(target=serve, daemon=True).start()
    return srv, kept


def _released_to_zero(r):
    """An event set when the relay's last open connection is released."""
    gone = threading.Event()
    real = r.released

    def released():
        real()
        if r.open == 0:
            gone.set()
    r.released = released
    return gone


@pytest.mark.parametrize("sent", [b"", b"x"])
def test_a_pair_the_client_ended_closes_once_quiet(loop, tmp_path, sent):
    """The client ends its half and the target never answers or closes. The
    pair must not linger, whether or not a byte moved before: it closes at
    the answer deadline."""
    srv, kept = _mute_server()
    path = str(tmp_path / "linger.sock")
    r = relay.Relay(loop, path, srv.getsockname(), name="forward 5432", idle_deadline=0.3,
                    answer_deadline=0.3)
    gone = _released_to_zero(r)
    c = _unix_client(path)
    if sent:
        c.sendall(sent)
    c.close()
    assert gone.wait(5)
    srv.close()
    for k in kept:
        k.close()


def test_a_slow_answer_after_the_client_ended_still_arrives(loop, tmp_path):
    """The client sends a request and ends its half, and the target thinks
    past the idle deadline before it answers, as a long reply that does not
    stream does. The answer must reach the client."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    answer = threading.Event()

    def serve():
        conn, _ = srv.accept()
        while conn.recv(1024):
            pass                                   # read until the client ended
        answer.wait(10)
        conn.sendall(b"the answer")
        conn.close()
    threading.Thread(target=serve, daemon=True).start()
    path = str(tmp_path / "slow.sock")
    relay.Relay(loop, path, srv.getsockname(), name="gmlx api", idle_deadline=0.3)
    c = _unix_client(path)
    c.sendall(b"GET / HTTP/1.1\r\n\r\n")
    c.shutdown(socket.SHUT_WR)
    _past_deadline(loop, 0.3)                      # the idle deadline has run
    _past_deadline(loop, 0.3)
    answer.set()
    got = b""
    while chunk := c.recv(64):
        got += chunk
    assert got == b"the answer"
    c.close()
    srv.close()


def test_a_pair_the_target_ended_closes_once_quiet(loop, tmp_path):
    """The target answers and ends its half, and the client never ends its
    own. The pair closes at the idle deadline, not the long answer one."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.sendall(b"bye")
        conn.shutdown(socket.SHUT_WR)
    threading.Thread(target=serve, daemon=True).start()
    path = str(tmp_path / "ended.sock")
    r = relay.Relay(loop, path, srv.getsockname(), name="forward 5432", idle_deadline=0.3,
                    answer_deadline=3600.0)
    gone = _released_to_zero(r)
    c = _unix_client(path)
    assert c.recv(3) == b"bye"
    assert gone.wait(5)
    c.close()
    srv.close()


def test_the_quiet_limit_follows_the_side_that_ended():
    pair = relay._Pair.__new__(relay._Pair)
    pair.idle_deadline, pair.answer_deadline = 30.0, 3600.0
    pair.down_eof = pair.up_eof = False
    assert pair._quiet_limit() is None
    pair.down_eof = True                           # an answer is pending
    assert pair._quiet_limit() == 3600.0
    pair.up_eof = True                             # the answer is complete
    assert pair._quiet_limit() == 30.0
    pair.down_eof = False
    assert pair._quiet_limit() == 30.0


def test_the_quiet_deadline_waits_for_the_last_byte():
    pair = relay._Pair.__new__(relay._Pair)
    lp = _FakeLoop()
    closed = []
    pair.loop, pair.closed, pair.idle_deadline = lp, False, 30.0
    pair.answer_deadline, pair.down_eof, pair.up_eof = 3600.0, False, True
    pair.close = lambda: closed.append(True)
    pair.last = lp.now() - 10.0                        # a byte moved 10 s ago
    pair._expire_quiet()
    assert not closed and lp.later[-1][0] == 20.0
    pair.last = lp.now() - 31.0
    pair._expire_quiet()
    assert closed == [True]


def test_no_socket_for_the_target_frees_the_slot(loop, tmp_path, monkeypatch):
    port, stop = _echo_server()
    path = str(tmp_path / "emfile.sock")
    r = relay.Relay(loop, path, ("127.0.0.1", port), name="gmlx api")

    def no_descriptor(family):
        raise OSError(errno.EMFILE, "Too many open files")
    monkeypatch.setattr(relay, "_upstream_socket", no_descriptor)
    with _unix_client(path) as c:
        assert c.recv(10) == b""                   # closed, not left open
    assert _in_loop(loop, lambda: r.open) == 0
    assert any("Too many open files" in line for line in loop.logged)
    monkeypatch.undo()
    with _unix_client(path) as c:
        assert _echoes(c, 5)
    stop()


class _RefusedLater:
    """A target socket whose connect starts and then fails, as a refused
    loopback connect does on macOS: EINPROGRESS, then ECONNREFUSED once the
    socket is writable. A socket pair end stands in, since it is writable at
    once, so the failure never depends on timing."""

    def __init__(self):
        self._sock, self._peer = socket.socketpair()
        self.checked = False

    def connect_ex(self, addr):
        return errno.EINPROGRESS

    def getsockopt(self, level, option, *rest):
        if option == socket.SO_ERROR:
            self.checked = True
            return errno.ECONNREFUSED
        return self._sock.getsockopt(level, option, *rest)

    def close(self):
        self._sock.close()
        self._peer.close()

    def __getattr__(self, name):
        return getattr(self._sock, name)


def _past_deadline(loop, delay):
    """Wait until a timer of ``delay`` seconds, set now, has run. The loop
    runs timers in order of their due time, so every deadline set earlier
    with the same delay has run too."""
    ran = threading.Event()
    loop.call_later(delay, ran.set)
    assert ran.wait(10)


def test_no_socket_after_a_failed_connect_frees_the_slot(loop, tmp_path, monkeypatch):
    """The leaking path: the first target fails after the connect started,
    and the socket for the next target cannot be made."""
    port, stop = _echo_server()
    path = str(tmp_path / "emfile2.sock")
    r = relay.Relay(loop, path, [("127.0.0.1", 9), ("127.0.0.1", port)], name="gmlx api")
    made = []

    def second_fails(family):
        if made:
            made.append(None)
            raise OSError(errno.EMFILE, "Too many open files")
        made.append(_RefusedLater())
        return made[0]
    monkeypatch.setattr(relay, "_upstream_socket", second_fails)
    with _unix_client(path) as c:
        assert c.recv(10) == b""                   # closed, not left open
    assert len(made) == 2 and made[0].checked     # the first failed after it started
    assert _in_loop(loop, lambda: r.open) == 0
    assert any("Too many open files" in line for line in loop.logged)
    stop()


def test_a_failed_connect_step_releases_the_slot_once(loop, tmp_path, monkeypatch):
    """The idle deadline starts after the connect, so a pair that closed
    while it connected is never released a second time."""
    port, stop = _echo_server()
    path = str(tmp_path / "once.sock")
    r = relay.Relay(loop, path, ("127.0.0.1", port), name="gmlx api", idle_deadline=0.2)

    def broken(self):
        raise OSError(errno.EBADF, "Bad file descriptor")
    monkeypatch.setattr(relay._Pair, "_connect_next", broken)
    with _unix_client(path) as c:
        assert c.recv(10) == b""
    _past_deadline(loop, 0.2)
    assert _in_loop(loop, lambda: r.open) == 0
    stop()


def test_the_api_relay_needs_a_whole_request_head_before_the_deadline(loop, tmp_path):
    srv, _closed = _silent_server()
    path = str(tmp_path / "head.sock")
    clock = _Clock(loop)
    r = relay.Relay(loop, path, srv.getsockname(), name="gmlx api", idle_deadline=30,
                    idle_until_head=True)
    # The test server echoes, so bytes flow both ways, but a request line
    # alone is not a whole head.
    trickle = _unix_client(path)
    line = b"GET /v1/models HTTP/1.1\r\n"
    trickle.sendall(line)
    assert _recv_exactly(trickle, len(line)) == line
    clock.advance(29.9)
    assert _in_loop(loop, lambda: r.open) == 1
    clock.advance(0.1)
    assert trickle.recv(64) == b""                  # closed at the deadline
    trickle.close()
    whole = _unix_client(path)
    head = b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n"
    whole.sendall(head[:-3])
    # The echo shows that the relay read the first part, so the empty line
    # spans two reads.
    assert _recv_exactly(whole, len(head) - 3) == head[:-3]
    whole.sendall(head[-3:])
    assert _recv_exactly(whole, 3) == head[-3:]
    clock.advance(31)
    assert _in_loop(loop, lambda: r.open) == 1      # still open past the deadline
    whole.close()
    srv.close()


_HEAD = b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n"


def _head_pair():
    """A pair that only watches for the request head, with no sockets."""
    pair = relay._Pair.__new__(relay._Pair)
    pair.moved, pair.until_head, pair.head_tail = False, True, b""
    return pair


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_a_request_head_split_anywhere_in_its_empty_line_counts(cut):
    pair = _head_pair()
    pair._note_down(_HEAD[:-cut])
    assert not pair.moved
    pair._note_down(_HEAD[-cut:])
    assert pair.moved


def test_a_request_head_sent_one_byte_at_a_time_counts():
    pair = _head_pair()
    for i in range(len(_HEAD)):
        assert not pair.moved
        pair._note_down(_HEAD[i:i + 1])
    assert pair.moved


def test_a_closed_pair_holds_no_memory(loop, tmp_path, monkeypatch):
    """Each pair arms timers of up to an hour. Once the pair closes, nothing
    may keep it alive until those timers would run, or a guest that opens
    and ends connections could grow the supervisor without limit."""
    import gc
    import weakref
    pairs = []

    class Kept(relay._Pair):
        def __init__(self, *a, **k):
            pairs.append(weakref.ref(self))
            super().__init__(*a, **k)
    monkeypatch.setattr(relay, "_Pair", Kept)
    port, stop = _echo_server()
    path = str(tmp_path / "many.sock")
    r = relay.Relay(loop, path, ("127.0.0.1", port), name="forward 5432")
    for _ in range(300):
        with _unix_client(path) as c:
            c.sendall(b"x")
            c.shutdown(socket.SHUT_WR)              # arms the hour-long answer deadline
            assert c.recv(1) == b"x" and c.recv(1) == b""
    assert _when_released(loop, r).wait(10)
    gc.collect()
    assert len(pairs) == 300 and not [p for p in pairs if p() is not None]
    assert _in_loop(loop, lambda: len(loop._timers)) < 200
    stop()


@pytest.mark.parametrize("side", ["up", "down"])
def test_a_stale_event_after_close_leaves_the_pair_alone(loop, tmp_path, side):
    """One select call can return events for both sides of a pair. When the
    event that runs first closes the pair, the other one must not arm a timer
    that keeps the closed pair."""
    target = relay.listen_socket(str(tmp_path / "t.sock"))
    down, client = socket.socketpair()
    try:
        pair = _in_loop(loop, lambda: relay._Pair(loop, down, [str(tmp_path / "t.sock")],
                                                  "forward 1"))
        assert _in_loop(loop, lambda: pair.connecting) is False

        def live():
            return sum(1 for t in loop._timers if t.fn is not None)

        def stale():
            pair.close()
            before = live()
            handler = pair._on_up if side == "up" else pair._on_down
            handler(relay._READ | relay._WRITE)
            return pair.quiet_timer, live() - before
        assert _in_loop(loop, stale) == (None, 0)
        assert loop.logged == []
    finally:
        client.close()
        target.close()


def test_a_cancelled_timer_never_runs(loop):
    ran, later = [], threading.Event()
    timer = _in_loop(loop, lambda: loop.call_later(0.05, lambda: ran.append(1)))
    _in_loop(loop, timer.cancel)
    loop.call_later(0.1, later.set)
    assert later.wait(5) and ran == []


def test_the_accept_pause_needs_no_timer_thread(loop, monkeypatch):
    def no_threads(*a, **k):
        raise RuntimeError("can't start new thread")
    monkeypatch.setattr(relay.threading, "Timer", no_threads)
    ran = threading.Event()
    loop.call_soon(lambda: loop.call_later(0.05, ran.set))
    assert ran.wait(5)
    ran.clear()
    loop.call_later(0.05, ran.set)                 # from another thread as well
    assert ran.wait(5)


# A target that fails while the client still sends

_BUSY = (b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 19\r\n"
         b"connection: close\r\n\r\nService Unavailable")


def _busy_server(addr):
    """A target that reads a request head, answers 503 and closes before it
    reads the body, as a server at its connection limit does. Returns the
    address it listens on and its listener."""
    srv = relay.listen_socket(addr)
    srv.setblocking(True)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                head = b""
                while b"\r\n\r\n" not in head and (data := conn.recv(4096)):
                    head += data
                conn.sendall(_BUSY)
    threading.Thread(target=serve, daemon=True).start()
    name = srv.getsockname()
    return (name if isinstance(name, str) else (name[0], name[1])), srv


@pytest.mark.parametrize("unix", [True, False], ids=["unix", "tcp"])
def test_an_answer_reaches_the_client_after_the_target_stops_reading(loop, tmp_path, unix):
    """The relay's send to the target fails, or its read gets a reset, while
    the target's answer waits in a buffer. The client must get that answer,
    not a closed connection with nothing in it."""
    target, srv = _busy_server(str(tmp_path / "t.sock") if unix else ("127.0.0.1", 0))
    path = str(tmp_path / "api.sock")
    r = relay.Relay(loop, path, [target], name="gmlx api", idle_until_head=True)
    body = 300 * 1024
    got = collections.Counter()
    for _ in range(100):
        with _unix_client(path) as c:
            try:
                c.sendall(b"POST /v1/messages HTTP/1.1\r\nhost: x\r\n"
                          b"content-length: %d\r\n\r\n" % body + b"x" * body)
            except OSError:
                pass
            answer = b""
            try:
                while data := c.recv(65536):
                    answer += data
            except OSError:
                pass
            got[answer == _BUSY] += 1
    assert got == {True: 100}
    assert _when_released(loop, r).wait(10)        # each pair closed once its client did
    srv.close()


# The clipboard server

class StubPasteboard:
    """Stands in for NSPasteboard: a type list, data per type, and an access
    behavior. ``reads`` records every data read."""

    def __init__(self, items=None, behavior=2, gate=None, error=None):
        self.items = dict(items or {})
        self.behavior, self.error = behavior, error
        # A data read waits until ``gate``, an event, is set, and ``entered``
        # is set once a read has started.
        self.gate, self.entered = gate, threading.Event()
        self.reads: list = []

    def types(self):
        return list(self.items)

    def dataForType_(self, kind):
        self.reads.append(kind)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.error:
            raise self.error
        return self.items.get(kind)

    def respondsToSelector_(self, name):
        return name == "accessBehavior"

    def accessBehavior(self):
        return self.behavior


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"x" * 100


def _server(loop, tmp_path, pb, **kw):
    path = str(tmp_path / "clip.sock")
    return clipboard.ClipboardServer(loop, path, pasteboard=lambda: pb, **kw), path


def _ask(path, line: bytes, timeout=10.0) -> bytes:
    with _unix_client(path) as c:
        c.settimeout(timeout)
        c.sendall(line)
        data = b""
        while chunk := c.recv(1 << 20):
            data += chunk
    return data


def _tiff(width=3, height=2) -> bytes:
    import io

    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(out, format="TIFF")
    return out.getvalue()


def test_text_only_clipboard_lists_nothing(loop, tmp_path):
    pb = StubPasteboard({"public.utf8-plain-text": b"secret"})
    _, path = _server(loop, tmp_path, pb)
    assert _ask(path, b"TYPES\n") == b"OK 0\n"
    assert _ask(path, b"IMAGE image/png\n") == b"ERR there is no image on the Mac clipboard\n"
    assert pb.reads == []                     # text is never read


def test_types_read_no_data_and_images_read_only_on_request(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES})
    _, path = _server(loop, tmp_path, pb)
    assert _ask(path, b"TYPES\n") == b"OK 10\nimage/png\n"
    assert pb.reads == []
    assert _ask(path, b"IMAGE image/png\n") == b"OK %d\n" % len(PNG_BYTES) + PNG_BYTES
    assert pb.reads == ["public.png"]
    assert loop.logged == [
        "clipboard: the container asked which image types the Mac clipboard holds "
        "(1 times this session)",
        "clipboard: sent an image of 108 bytes"]                      # one per image read


def test_types_requests_are_logged_once_per_hundred(loop, tmp_path):
    server, _ = _server(loop, tmp_path, StubPasteboard({"public.png": PNG_BYTES}))
    for _ in range(2 * clipboard.TYPES_LOG_EVERY + 1):
        assert server.answer("TYPES") == b"OK 10\nimage/png\n"
    assert [line.split("(")[1] for line in loop.logged] == [
        "1 times this session)", "101 times this session)", "201 times this session)"]


def test_tiff_comes_back_as_png(loop, tmp_path):
    import io

    from PIL import Image
    pb = StubPasteboard({"public.tiff": _tiff()})
    _, path = _server(loop, tmp_path, pb)
    reply = _ask(path, b"IMAGE image/png\n")
    head, body = reply.split(b"\n", 1)
    assert head == b"OK %d" % len(body) and body.startswith(b"\x89PNG")
    assert Image.open(io.BytesIO(body)).size == (3, 2)


def test_image_over_the_limit_is_refused(loop, tmp_path):
    pb = StubPasteboard({"public.png": b"\x89PNG" + b"x" * (clipboard.IMAGE_MAX + 1)})
    _, path = _server(loop, tmp_path, pb)
    reply = _ask(path, b"IMAGE image/png\n")
    assert reply.startswith(b"ERR ") and b"over the 20 MiB limit" in reply


class _HugeData:
    """Pasteboard data that reports its length and must never be copied."""

    def __init__(self, size):
        self.size = size

    def length(self):
        return self.size

    def __bytes__(self):
        raise AssertionError("the data was copied before the size check")


@pytest.mark.parametrize("kind, size, words", [
    ("public.png", clipboard.IMAGE_MAX + 1, b"over the 20 MiB limit"),
    ("public.tiff", clipboard.CONVERT_MAX + 1, b"over the 64 MiB the Mac converts"),
])
def test_a_large_image_is_refused_before_it_is_copied_or_converted(loop, tmp_path,
                                                                   monkeypatch, kind,
                                                                   size, words):
    def no_convert(data):
        raise AssertionError("converted before the size check")
    monkeypatch.setattr(clipboard, "to_png", no_convert)
    server, _ = _server(loop, tmp_path, StubPasteboard({kind: _HugeData(size)}))
    reply = server.answer("IMAGE image/png")
    assert reply.startswith(b"ERR ") and words in reply


def test_a_converted_image_over_the_limit_is_refused(loop, tmp_path, monkeypatch):
    monkeypatch.setattr(clipboard, "to_png", lambda data: b"x" * (clipboard.IMAGE_MAX + 1))
    server, _ = _server(loop, tmp_path, StubPasteboard({"public.tiff": b"small"}))
    reply = server.answer("IMAGE image/png")
    assert reply.startswith(b"ERR ") and b"over the 20 MiB limit" in reply


def test_an_image_type_that_does_not_convert_gives_way_to_the_next(loop, tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(clipboard, "to_png", lambda data: None if data == b"bad" else PNG_BYTES)
    pb = StubPasteboard({"public.tiff": b"bad", "public.jpeg": b"good"})
    server, _ = _server(loop, tmp_path, pb)
    assert server.answer("IMAGE image/png") == b"OK %d\n" % len(PNG_BYTES) + PNG_BYTES
    assert pb.reads == ["public.tiff", "public.jpeg"]


@pytest.mark.parametrize("kind, size", [("public.png", clipboard.IMAGE_MAX + 1),
                                        ("public.tiff", clipboard.CONVERT_MAX + 1)])
def test_an_image_type_over_a_limit_gives_way_to_the_next(loop, tmp_path, monkeypatch,
                                                         kind, size):
    monkeypatch.setattr(clipboard, "to_png", lambda data: PNG_BYTES)
    pb = StubPasteboard({kind: _HugeData(size), "public.jpeg": b"good"})
    server, _ = _server(loop, tmp_path, pb)
    assert server.answer("IMAGE image/png") == b"OK %d\n" % len(PNG_BYTES) + PNG_BYTES
    assert pb.reads == [kind, "public.jpeg"]


def test_an_image_type_that_converts_over_the_limit_gives_way_to_the_next(loop, tmp_path,
                                                                         monkeypatch):
    # A 16-bit TIFF can give a PNG over the limit where its JPEG gives one that fits.
    deep = b"x" * (clipboard.IMAGE_MAX + 1)
    monkeypatch.setattr(clipboard, "to_png",
                        lambda data: deep if data == b"deep" else PNG_BYTES)
    pb = StubPasteboard({"public.tiff": b"deep", "public.jpeg": b"good"})
    server, _ = _server(loop, tmp_path, pb)
    assert server.answer("IMAGE image/png") == b"OK %d\n" % len(PNG_BYTES) + PNG_BYTES
    assert pb.reads == ["public.tiff", "public.jpeg"]


def test_a_converted_image_over_the_limit_is_called_too_large_before_unreadable(
        loop, tmp_path, monkeypatch):
    deep = b"x" * (clipboard.IMAGE_MAX + 1)
    monkeypatch.setattr(clipboard, "to_png", lambda data: deep if data == b"deep" else None)
    pb = StubPasteboard({"public.tiff": b"deep", "public.jpeg": b"bad"})
    server, _ = _server(loop, tmp_path, pb)
    assert server.answer("IMAGE image/png") == (
        b"ERR the image is 20 MiB as PNG, over the 20 MiB limit\n")
    assert pb.reads == ["public.tiff", "public.jpeg"]
    assert any(line.startswith("clipboard: refused an image") for line in loop.logged)


def test_an_image_over_a_limit_is_called_too_large_before_unreadable(loop, tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(clipboard, "to_png", lambda data: None)
    pb = StubPasteboard({"public.tiff": _HugeData(clipboard.CONVERT_MAX + 1),
                         "public.jpeg": b"bad"})
    server, _ = _server(loop, tmp_path, pb)
    reply = server.answer("IMAGE image/png")
    assert reply.startswith(b"ERR ") and b"over the 64 MiB the Mac converts" in reply


def test_an_image_that_does_not_convert_is_not_called_missing(loop, tmp_path, monkeypatch):
    monkeypatch.setattr(clipboard, "to_png", lambda data: None)
    pb = StubPasteboard({"public.tiff": b"bad", "public.heic": b"bad"})
    server, _ = _server(loop, tmp_path, pb)
    reply = server.answer("IMAGE image/png")
    assert reply == (b"ERR the Mac cannot read the image on the clipboard, so copy it again "
                     b"in another format, such as PNG\n")
    assert any("public.tiff, public.heic" in line for line in loop.logged)


def test_denied_access_names_the_privacy_setting(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES}, behavior=3)
    _, path = _server(loop, tmp_path, pb)
    reply = _ask(path, b"IMAGE image/png\n")
    assert b"Privacy & Security, Paste from Other Apps" in reply
    assert pb.reads == []


def test_other_image_types_and_requests_are_refused(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES})
    _, path = _server(loop, tmp_path, pb)
    assert _ask(path, b"IMAGE image/jpeg\n").startswith(b"ERR ")
    assert _ask(path, b"TEXT\n") == b"ERR unknown request\n"
    assert pb.reads == []


def test_slow_read_delays_no_relay_connection(loop, tmp_path):
    step, each = _lockstep()
    port, stop = _sse_server(8, step)
    api = str(tmp_path / "api.sock")
    relay.Relay(loop, api, ("127.0.0.1", port))
    gate = threading.Event()
    pb = StubPasteboard({"public.png": PNG_BYTES}, gate=gate)
    _, path = _server(loop, tmp_path, pb)
    reply: list = []
    slow = threading.Thread(target=lambda: reply.append(_ask(path, b"IMAGE image/png\n")))
    slow.start()
    try:
        assert pb.entered.wait(10)            # the read is under way
        with _unix_client(api) as c:          # a new connection while it runs
            assert _read_events(c, 8, each) == 8
        stop()
        assert slow.is_alive() and not reply  # the read still runs
    finally:
        gate.set()
    slow.join(10)
    assert reply and reply[0].startswith(b"OK ")


def _captured_handoffs(server):
    """Record each socket the loop hands to the worker."""
    handed = _Seen()
    real = server._queue.put_nowait

    def put(item):
        real(item)
        if item is not None:
            handed.append(item[0])
    server._queue.put_nowait = put
    return handed


def _in_loop(loop, fn):
    done, out = threading.Event(), []
    loop.call_soon(lambda: (out.append(fn()), done.set()))
    assert done.wait(5)
    return out[0]


def test_handoff_unregisters_and_the_worker_closes(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES})
    server, path = _server(loop, tmp_path, pb)
    handed = _captured_handoffs(server)
    assert _ask(path, b"TYPES\n").startswith(b"OK ")      # EOF after the reply
    assert len(handed) == 1
    assert _in_loop(loop, lambda: loop.is_watched(handed[0])) is False
    assert handed[0].fileno() == -1                       # closed by the worker


def test_worker_closes_when_the_read_raises(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES}, error=RuntimeError("boom"))
    server, path = _server(loop, tmp_path, pb)
    handed = _captured_handoffs(server)
    reply = _ask(path, b"IMAGE image/png\n")
    # The guest gets a fixed message, and the log keeps the exception.
    assert reply == b"ERR cannot read the Mac clipboard\n"
    assert handed[0].fileno() == -1
    assert any("boom" in line for line in loop.logged)


def test_worker_survives_a_stand_in_that_went_away(loop, tmp_path):
    gate = threading.Event()
    pb = StubPasteboard({"public.png": PNG_BYTES}, gate=gate)
    server, path = _server(loop, tmp_path, pb)
    handed = _captured_handoffs(server)
    c = _unix_client(path)
    c.sendall(b"IMAGE image/png\n")
    assert pb.entered.wait(10)
    c.close()                                 # gone before the answer
    gate.set()
    assert _ask(path, b"IMAGE image/png\n").startswith(b"OK ")
    assert all(s.fileno() == -1 for s in handed) and len(handed) == 2


def test_stand_in_that_stops_reading_frees_the_worker(loop, tmp_path):
    big = b"\x89PNG" + b"x" * (8 << 20)     # far more than the socket buffers hold
    pb = StubPasteboard({"public.png": big})
    server, path = _server(loop, tmp_path, pb, send_timeout=0.5)
    stuck = _unix_client(path)
    stuck.sendall(b"IMAGE image/png\n")      # never read
    # The worker gives up on the stuck stand-in after the timeout and
    # answers the next request.
    assert _ask(path, b"TYPES\n", timeout=10) == b"OK 10\nimage/png\n"
    assert loop.logged.wait_for(
        lambda lines: any("cannot answer the container" in line for line in lines))
    stuck.close()


def test_long_or_unfinished_requests_never_reach_the_worker(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES})
    server, path = _server(loop, tmp_path, pb)
    handed = _captured_handoffs(server)
    assert _ask(path, b"x" * (clipboard.REQUEST_MAX + 10)) == b""   # closed by the loop
    c = _unix_client(path)
    c.sendall(b"IMAGE ima")
    c.shutdown(socket.SHUT_WR)                # ends before the newline
    assert c.recv(10) == b""
    c.close()
    assert handed == [] and pb.reads == []
    assert "clipboard: request too long, connection closed" in loop.logged


def test_an_idle_connection_is_closed_after_the_read_deadline(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES})
    clock = _Clock(loop)
    server, path = _server(loop, tmp_path, pb, read_deadline=30)
    idle = _unix_client(path)
    _settled(loop)                                  # accepted, so the deadline runs
    clock.advance(29.9)
    assert _in_loop(loop, lambda: server.open) == 1
    clock.advance(0.1)
    assert idle.recv(10) == b""                     # closed by the server
    idle.close()
    assert loop.logged.wait_for(lambda lines: any("no request in" in line for line in lines))
    # The worker frees the slot of the answered request after it closes the
    # socket, so the count reaches 0 a moment after the answer.
    released = _when_released(loop, server, "_released", now=False)
    assert _ask(path, b"TYPES\n") == b"OK 10\nimage/png\n"
    assert released.wait(10)


def test_a_full_queue_answers_busy(loop, tmp_path):
    gate = threading.Event()
    pb = StubPasteboard({"public.png": PNG_BYTES}, gate=gate)
    server, path = _server(loop, tmp_path, pb, queue_max=1)
    handed = _captured_handoffs(server)
    slow = [threading.Thread(target=_ask, args=(path, b"IMAGE image/png\n"))
            for _ in range(2)]
    slow[0].start()
    try:
        assert pb.entered.wait(10)                  # the worker holds the first
        slow[1].start()                             # the second fills the queue
        assert handed.wait_for(lambda h: len(h) == 2)
        assert _ask(path, b"TYPES\n") == clipboard._err(clipboard.BUSY)
    finally:
        gate.set()
    for t in slow:
        t.join(10)


def test_the_clipboard_holds_at_most_its_cap_of_connections(loop, tmp_path):
    server, path = _server(loop, tmp_path, StubPasteboard({"public.png": PNG_BYTES}),
                           max_connections=2)
    idle = [_unix_client(path), _unix_client(path)]
    waiting = _unix_client(path)
    waiting.sendall(b"TYPES\n")
    assert loop.logged.wait_for(lambda lines: any("requests are open" in line
                                                  for line in lines))
    # Not accepted yet.
    assert _in_loop(loop, lambda: (server.open, loop.is_watched(server.sock))) == (2, False)
    idle[0].close()
    assert waiting.recv(20) == b"OK 10\nimage/png\n"
    idle[1].close()
    waiting.close()


def test_close_closes_the_requests_that_wait_for_the_worker(loop, tmp_path):
    gate = threading.Event()
    pb = StubPasteboard({"public.png": PNG_BYTES}, gate=gate)
    server, path = _server(loop, tmp_path, pb)
    handed = _captured_handoffs(server)
    busy = threading.Thread(target=_ask, args=(path, b"IMAGE image/png\n"))
    busy.start()
    assert pb.entered.wait(10)
    waiting = _unix_client(path)
    waiting.sendall(b"TYPES\n")
    assert handed.wait_for(lambda h: len(h) == 2)
    server.close()
    assert handed[1].fileno() == -1                 # closed without an answer
    assert waiting.recv(20) == b""
    waiting.close()
    gate.set()
    busy.join(10)
    server._worker.join(5)
    assert not server._worker.is_alive()


def test_close_removes_the_socket(loop, tmp_path):
    server, path = _server(loop, tmp_path, StubPasteboard())
    server.close()
    _in_loop(loop, lambda: None)
    import os
    assert not os.path.exists(path)
    server._worker.join(5)
    assert not server._worker.is_alive()


def test_private_pasteboard_tiff_converts_to_png(loop, tmp_path):
    # A uniquely named pasteboard: the user's clipboard is never touched.
    from AppKit import NSPasteboard
    from Foundation import NSData
    pb = NSPasteboard.pasteboardWithUniqueName()
    try:
        tiff = _tiff(5, 4)
        pb.declareTypes_owner_(["public.tiff"], None)
        pb.setData_forType_(NSData.dataWithBytes_length_(tiff, len(tiff)), "public.tiff")
        assert clipboard.image_types(pb) == ["image/png"]
        png = clipboard.read_image_png(pb)
        assert png is not None and png.startswith(b"\x89PNG")
        import io

        from PIL import Image
        assert Image.open(io.BytesIO(png)).size == (5, 4)
        assert clipboard.access_denied(pb) is (pb.accessBehavior() == 3)
    finally:
        pb.releaseGlobally()


# The API relay after a server restart

def _unix_echo_server(path):
    srv = relay.listen_socket(path)
    srv.setblocking(True)

    def handle(conn):
        with conn:
            while data := conn.recv(4096):
                conn.sendall(data)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    return srv


def test_a_relay_asks_once_for_a_new_target_when_nothing_listens(loop, tmp_path):
    gone, fresh = str(tmp_path / "old.sock"), str(tmp_path / "new.sock")
    srv = _unix_echo_server(fresh)
    asked = []

    def renew():
        asked.append(1)
        return fresh
    path = str(tmp_path / "api.sock")
    r = relay.Relay(loop, path, [gone], name="gmlx api", renew=renew)
    for _ in range(2):
        with _unix_client(path) as c:
            assert _echoes(c, 10)
    assert asked == [1] and r.targets == [fresh]
    srv.close()


def test_a_renewal_that_fails_closes_the_connection(loop, tmp_path):
    gone, also_gone = str(tmp_path / "old.sock"), str(tmp_path / "new.sock")
    asked = []

    def renew():
        asked.append(1)
        return also_gone
    path = str(tmp_path / "api.sock")
    relay.Relay(loop, path, [gone], name="gmlx api", renew=renew)
    with _unix_client(path) as c:
        assert c.recv(10) == b""
    assert asked == [1]                   # one new target for each connection
    assert loop.logged.wait_for()
    assert loop.logged[0].startswith(f"gmlx api: cannot reach {gone} or {also_gone} (")


def test_a_renewal_with_no_answer_closes_the_connection(loop, tmp_path):
    def renew():
        raise OSError("server down")
    path = str(tmp_path / "api.sock")
    relay.Relay(loop, path, [str(tmp_path / "old.sock")], name="gmlx api", renew=renew)
    with _unix_client(path) as c:
        assert c.recv(10) == b""


def test_connections_that_fail_during_a_renewal_wait_for_it(loop, tmp_path):
    fresh = str(tmp_path / "new.sock")
    srv = _unix_echo_server(fresh)
    release, asked = threading.Event(), []

    def renew():
        asked.append(1)
        release.wait(10)
        return fresh
    path = str(tmp_path / "api.sock")
    relay.Relay(loop, path, [str(tmp_path / "old.sock")], name="gmlx api", renew=renew)
    clients = [_unix_client(path) for _ in range(3)]
    for c in clients:
        c.sendall(b"ping")
    release.set()
    for c in clients:
        with c:
            assert c.recv(4) == b"ping"
    assert asked == [1]
    srv.close()


def test_a_failed_renewal_holds_the_next_one_for_a_gap(loop, tmp_path, monkeypatch):
    """A client that connects again and again after a renewal that got no
    target does not make the relay ask the server at the same rate. The
    next connection waits for the renewal after the gap."""
    monkeypatch.setattr(relay, "RENEW_RETRY_GAP", 3600.0)
    fresh = str(tmp_path / "new.sock")
    answers, asked = [None, fresh], []

    def renew():
        asked.append(1)
        return answers[len(asked) - 1]
    path = str(tmp_path / "api.sock")
    r = relay.Relay(loop, path, [str(tmp_path / "old.sock")], name="gmlx api", renew=renew)
    clock = [100.0]
    r.now = lambda: clock[0]
    with _unix_client(path) as c:
        assert c.recv(10) == b""                 # the first renewal got no target
    assert _in_loop(loop, lambda: r.renew_failed) == 100.0
    held = threading.Event()
    real = r._renew

    def renew_called():
        real()
        held.set()
    r._renew = renew_called
    srv = _unix_echo_server(fresh)
    with _unix_client(path) as c:
        c.sendall(b"ping")
        assert held.wait(5)
        assert _in_loop(loop, lambda: (r.renew_timer is not None, len(r.waiting))) == (True, 1)
        assert asked == [1]

        def gap_over():
            clock[0] += 3600.0
            r.renew_timer.cancel()
            r._renew_later()
        _in_loop(loop, gap_over)
        assert c.recv(4) == b"ping"
    assert asked == [1, 1] and _in_loop(loop, lambda: r.renew_failed) is None
    srv.close()


def test_a_relay_without_renew_keeps_its_target(loop, tmp_path):
    gone = str(tmp_path / "old.sock")
    path = str(tmp_path / "api.sock")
    r = relay.Relay(loop, path, [gone], name="gmlx api")
    with _unix_client(path) as c:
        assert c.recv(10) == b""
    assert r.targets == [gone]


def test_a_connection_that_failed_on_an_old_target_takes_the_new_one(loop, tmp_path):
    asked = []
    r = relay.Relay(loop, str(tmp_path / "api.sock"), [str(tmp_path / "old.sock")],
                    name="gmlx api", renew=lambda: asked.append(1))
    r.targets = [str(tmp_path / "new.sock")]

    class Pair:
        targets = None
        connected = False

        def _connect_next(self):
            self.connected = True
    pair = Pair()
    r.renew_for(pair, str(tmp_path / "old.sock"))  # type: ignore[arg-type]
    assert pair.targets == [str(tmp_path / "new.sock")] and pair.connected
    assert not asked and not r.waiting


def test_a_relay_renews_once_its_socket_is_gone_with_no_connection(loop, tmp_path):
    """A restarted server removed the session socket. The relay asks for a
    new one without waiting for the client, so the server knows the
    session's web ports again."""
    old, fresh = str(tmp_path / "old.sock"), str(tmp_path / "new.sock")
    old_srv, new_srv = _unix_echo_server(old), _unix_echo_server(fresh)
    asked = threading.Event()

    def renew():
        asked.set()
        return fresh
    r = relay.Relay(loop, str(tmp_path / "api.sock"), [old], name="gmlx api", renew=renew,
                    check_every=0.05)
    _past_deadline(loop, 0.2)                      # a few checks ran
    assert not asked.is_set()
    old_srv.close()
    os.unlink(old)
    assert asked.wait(5)
    _past_deadline(loop, 0.1)
    assert _in_loop(loop, lambda: r.targets) == [fresh]
    new_srv.close()


def test_a_relay_checks_its_target_only_with_a_renew_hook(loop, tmp_path):
    r = relay.Relay(loop, str(tmp_path / "api.sock"), [str(tmp_path / "gone.sock")],
                    name="gmlx api", check_every=0.05)
    _past_deadline(loop, 0.2)
    assert r.check_every is None and r.targets == [str(tmp_path / "gone.sock")]


# Launch's own loopback listeners

class _PeerAs:
    """A listening socket whose connections seem to come from ``peer``."""

    def __init__(self, sock, peer):
        self._sock, self.peer = sock, peer

    def accept(self):
        conn, _ = self._sock.accept()
        return conn, self.peer

    def __getattr__(self, name):
        return getattr(self._sock, name)


@pytest.mark.parametrize("peer, served", [
    (("127.0.0.1", 50000), True),
    (("127.0.0.53", 50000), True),
    (("::1", 50000, 0, 0), True),
    (("::ffff:127.0.0.1", 50000, 0, 0), True),
    (("192.168.64.3", 50000), False),
    (("::ffff:192.168.64.3", 50000, 0, 0), False),
    (("fe80::1%bridge100", 50000, 0, 7), False),
])
def test_a_loopback_listener_serves_only_loopback_peers(loop, monkeypatch, peer, served):
    """A localhost DNS domain of Apple container forwards a guest's traffic to
    the Mac's loopback address with the guest's own source address. The web
    app's port must not serve other containers that way."""
    port, stop = _echo_server()
    real = relay.listen_socket
    monkeypatch.setattr(relay, "listen_socket",
                        lambda addr, backlog=128: _PeerAs(real(addr, backlog), peer))
    r = relay.Relay(loop, ("127.0.0.1", 0), ("127.0.0.1", port), name="web")
    with socket.create_connection(r.sock.getsockname()[:2], timeout=5) as c:
        if served:
            assert _echoes(c, 5)
        else:
            assert c.recv(4) == b""
    refused = [line for line in loop.logged if "refused a connection from outside" in line]
    assert bool(refused) is not served
    if not served:
        assert _in_loop(loop, lambda: r.open) == 0
    stop()


def test_a_full_target_buffer_is_not_a_failure():
    """A send that would block leaves the bytes to send later."""
    pair = relay._Pair.__new__(relay._Pair)

    class Full:
        def send(self, data):
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
    pair.up, pair.connecting, pair.to_up = Full(), False, bytearray(b"body")
    pair.up_shut = pair.discard = pair.closed = False
    pair._half_close = pair._update = lambda: None
    pair._on_up(relay._WRITE)
    assert pair.to_up == b"body" and not pair.up_shut and not pair.discard


# The Host check of a browser app's port

def _web_target(tmp_path):
    """A Unix socket target that echoes, with the count of the connections
    it accepted."""
    path = str(tmp_path / "web.sock")
    srv = relay.listen_socket(path)
    srv.setblocking(True)
    accepted = []

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            accepted.append(1)

            def handle(conn=conn):
                with conn:
                    while data := conn.recv(4096):
                        conn.sendall(data)
            threading.Thread(target=handle, daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    return path, accepted, srv.close


def _ask_web(port, head):
    with socket.create_connection(("::1", port), timeout=5) as c:
        c.sendall(head)
        got = b""
        while data := c.recv(4096):
            got += data
        return got


def test_the_web_relay_refuses_a_request_for_localhost(loop, tmp_path):
    """A page that loads the app at localhost:3100 would be the same site as
    the host-mode apps on localhost, so the relay refuses it, and no byte
    reaches the guest."""
    path, accepted, stop = _web_target(tmp_path)
    r = relay.Relay(loop, ("::1", 0), path, name="web", check_host=True)
    port = r.sock.getsockname()[1]
    assert r.host == f"[::1]:{port}"
    r.host = "[::1]:3100"                 # as for the session's own port
    got = _ask_web(port, b"GET / HTTP/1.1\r\nHost: localhost:3100\r\n\r\n")
    assert got.startswith(b"HTTP/1.1 421 Misdirected Request\r\n")
    assert got.endswith(b"\r\n\r\nThis app answers only at http://[::1]:3100/. "
                        b"Open that address.\n")
    assert b"Connection: close\r\n" in got
    assert accepted == []
    assert any("web: refused a request for another host name (Host: localhost:3100)" in line
               for line in loop.logged)
    head = b"GET / HTTP/1.1\r\nhost: [::1]:3100\r\n\r\n"
    with socket.create_connection(("::1", port), timeout=5) as c:
        c.sendall(head)
        echo = b""
        while len(echo) < len(head):
            echo += c.recv(4096)
    assert echo == head and accepted == [1]
    stop()


@pytest.mark.parametrize(("head", "shown"), [
    (b"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n", "Host: 127.0.0.1:{port}"),
    (b"GET / HTTP/1.1\r\nHost: evil.example:{port}\r\n\r\n", "Host: evil.example:{port}"),
    (b"GET / HTTP/1.1\r\nHost: [::1]:3\r\n\r\n", "Host: [::1]:3"),
    (b"GET / HTTP/1.1\r\nHost: [::1]:{port}\r\nHost: localhost:{port}\r\n\r\n",
     "2 Host headers"),
    (b"GET / HTTP/1.0\r\n\r\n", "no Host header"),
    (b"GET / HTTP/1.1\r\nX-Host: [::1]:{port}\r\n\r\n", "no Host header"),
])
def test_the_web_relay_refuses_every_other_host(loop, tmp_path, head, shown):
    path, accepted, stop = _web_target(tmp_path)
    r = relay.Relay(loop, ("::1", 0), path, name="web", check_host=True)
    port = r.sock.getsockname()[1]
    got = _ask_web(port, head.replace(b"{port}", str(port).encode()))
    assert got.startswith(b"HTTP/1.1 421 ")
    assert accepted == []
    line = f"web: refused a request for another host name ({shown.format(port=port)}). "
    assert any(logged.startswith(line) for logged in _in_loop(loop, lambda: list(loop.logged)))
    stop()


def test_the_web_relay_waits_for_the_whole_first_head(loop, tmp_path):
    """The head can arrive in parts, and the rest of the connection's bytes
    follow it to the guest."""
    path, accepted, stop = _web_target(tmp_path)
    r = relay.Relay(loop, ("::1", 0), path, name="web", check_host=True)
    port = r.sock.getsockname()[1]
    head = f"GET / HTTP/1.1\r\nHost: [::1]:{port}\r\n\r\nbody".encode()
    with socket.create_connection(("::1", port), timeout=5) as c:
        c.sendall(head[:20])
        _settled(loop)                              # the relay accepted the connection
        _settled(loop)                              # and read the first part
        assert accepted == []
        c.sendall(head[20:])
        echo = b""
        while len(echo) < len(head):
            echo += c.recv(4096)
    assert echo == head and accepted == [1]
    stop()


def test_the_web_relay_refuses_a_head_that_never_ends(loop, tmp_path, monkeypatch):
    monkeypatch.setattr(relay, "HEAD_MAX", 64)
    path, accepted, stop = _web_target(tmp_path)
    r = relay.Relay(loop, ("::1", 0), path, name="web", check_host=True)
    port = r.sock.getsockname()[1]
    got = _ask_web(port, b"GET / HTTP/1.1\r\nCookie: " + b"x" * 200)
    assert got.startswith(b"HTTP/1.1 431 ")
    assert accepted == []
    stop()


def test_a_client_that_ends_before_its_head_reaches_nothing(loop, tmp_path):
    path, accepted, stop = _web_target(tmp_path)
    r = relay.Relay(loop, ("::1", 0), path, name="web", check_host=True)
    port = r.sock.getsockname()[1]
    with socket.create_connection(("::1", port), timeout=5) as c:
        c.sendall(b"GET / HTTP/1.1\r\n")
        c.shutdown(socket.SHUT_WR)
        assert c.recv(10) == b""
    assert _when_released(loop, r).wait(10)
    assert accepted == []
    stop()


def test_a_relay_without_the_host_check_passes_any_host(loop, tmp_path):
    path, accepted, stop = _web_target(tmp_path)
    r = relay.Relay(loop, ("::1", 0), path, name="web")
    port = r.sock.getsockname()[1]
    head = b"GET / HTTP/1.1\r\nHost: localhost:3100\r\n\r\n"
    with socket.create_connection(("::1", port), timeout=5) as c:
        c.sendall(head)
        echo = b""
        while len(echo) < len(head):
            echo += c.recv(4096)
    assert echo == head and r.host is None
    stop()
