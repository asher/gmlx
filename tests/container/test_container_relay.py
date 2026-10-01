"""The Mac-side relay of `gmlx launch --container`: one selectors loop that
joins accepted connections to their targets, and the clipboard server that
shares it. Real sockets on loopback and in a temp folder; no container, no
server. The clipboard tests read a stub pasteboard, except the conversion
test, which writes a private named NSPasteboard and never the user's own."""
from __future__ import annotations

import collections
import errno
import os
import socket
import threading
import time

import pytest

from gmlx.container import clipboard, relay


def _sse_server(events: int, gap: float):
    """A TCP server that answers each connection with ``events`` SSE events
    ``gap`` seconds apart. Returns (port, stop)."""
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
                    time.sleep(gap)
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


def _read_events(sock, count):
    """Arrival times of ``count`` SSE events."""
    sock.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
    buf, times = b"", []
    while len(times) < count:
        data = sock.recv(4096)
        if not data:
            break
        buf += data
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            if b"data:" in event:
                times.append(time.monotonic())
    return times


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
    logged = []
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
    port, stop = _sse_server(8, 0.05)
    path = str(tmp_path / "api.sock")
    relay.Relay(loop, path, ("127.0.0.1", port))
    with _unix_client(path) as c:
        times = _read_events(c, 8)
    stop()
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert len(times) == 8
    assert min(gaps) > 0.02                  # each event arrives on its own
    assert max(gaps) < 0.5                   # and none waits for a later one


def test_twenty_concurrent_streams(loop, tmp_path):
    port, stop = _sse_server(10, 0.02)
    path = str(tmp_path / "api.sock")
    relay.Relay(loop, path, ("127.0.0.1", port))
    results = []

    def one():
        with _unix_client(path) as c:
            results.append(len(_read_events(c, 10)))

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
    sent = []

    def flood():
        conn, _ = srv.accept()
        conn.settimeout(1.0)
        n = 0
        with conn:
            while n < total:
                try:
                    n += conn.send(b"x" * min(65536, total - n))
                except socket.timeout:
                    sent.append(("stalled", n))
                    conn.settimeout(10)
        sent.append(("done", n))

    threading.Thread(target=flood, daemon=True).start()
    path = str(tmp_path / "b.sock")
    relay.Relay(loop, path, ("127.0.0.1", srv.getsockname()[1]))
    with _unix_client(path) as c:
        deadline = time.monotonic() + 10
        while not sent and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sent and sent[0][0] == "stalled"      # the target had to wait
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
    deadline = time.monotonic() + 5
    while not loop.logged and time.monotonic() < deadline:
        time.sleep(0.02)
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
    deadline = time.monotonic() + 5
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not path.exists()


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
    deadline = time.monotonic() + 5
    while not loop.logged and time.monotonic() < deadline:
        time.sleep(0.02)
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
    extra = _unix_client(path)                     # waits in the listen queue
    assert not _echoes(extra, 0.5)
    held.pop().close()                             # frees a slot
    extra.settimeout(5)
    assert extra.recv(4) == b"ping"
    held.append(extra)
    # Back at the cap at once is the same run, so it is logged once.
    _in_loop(loop, lambda: None)
    assert sum("connections are open" in line for line in loop.logged) == 1
    for c in held:
        c.close()
    deadline = time.monotonic() + 5
    while _in_loop(loop, lambda: r.open) != 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    # The count fell to half the cap or less, so a new run is logged again.
    again = [_unix_client(path) for _ in range(5)]
    assert all(_echoes(c, 5) for c in again[:4])
    _in_loop(loop, lambda: None)
    assert sum("connections are open" in line for line in loop.logged) == 2
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
    """A server that accepts and never sends, with the count of the
    connections it saw closed by the other side."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    closed = []

    def handle(conn):
        with conn:
            while True:
                data = conn.recv(4096)
                if not data:
                    closed.append(1)
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
    path = str(tmp_path / "idle.sock")
    r = relay.Relay(loop, path, srv.getsockname(), name="gmlx api", idle_deadline=0.3)
    c = _unix_client(path)
    start = time.monotonic()
    assert c.recv(10) == b""                       # the relay closed it
    assert 0.25 < time.monotonic() - start < 5
    c.close()
    deadline = time.monotonic() + 5
    while not closed or _in_loop(loop, lambda: r.open) != 0:
        assert time.monotonic() < deadline         # the server side closed too
        time.sleep(0.01)
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
    pair.last = time.monotonic() - 10.0                # a byte moved 10 s ago
    pair._expire_quiet()
    assert not closed and lp.later[-1][0] == pytest.approx(20.0, abs=1.0)
    pair.last = time.monotonic() - 31.0
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
    r = relay.Relay(loop, path, srv.getsockname(), name="gmlx api", idle_deadline=0.3,
                    idle_until_head=True)
    # The test server echoes, so bytes flow both ways, but a request line
    # alone is not a whole head.
    trickle = _unix_client(path)
    start = time.monotonic()
    trickle.sendall(b"GET /v1/models HTTP/1.1\r\n")
    while trickle.recv(64):
        pass                                        # the echo, then the close
    assert 0.25 < time.monotonic() - start < 5
    trickle.close()
    whole = _unix_client(path)
    head = b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n"
    whole.sendall(head[:-3])
    time.sleep(0.05)
    whole.sendall(head[-3:])                        # the empty line spans two reads
    got = b""
    while len(got) < len(head):
        got += whole.recv(64)
    _past_deadline(loop, 0.3)
    whole.settimeout(0.2)
    with pytest.raises(socket.timeout):             # still open past the deadline
        whole.recv(10)
    assert _in_loop(loop, lambda: r.open) == 1
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
    deadline = time.monotonic() + 5
    while _in_loop(loop, lambda: r.open):          # each pair closed once its client did
        assert time.monotonic() < deadline
        time.sleep(0.02)
    srv.close()


# The clipboard server

class StubPasteboard:
    """Stands in for NSPasteboard: a type list, data per type, and an access
    behavior. ``reads`` records every data read."""

    def __init__(self, items=None, behavior=2, delay=0.0, error=None):
        self.items = dict(items or {})
        self.behavior, self.delay, self.error = behavior, delay, error
        self.reads: list = []

    def types(self):
        return list(self.items)

    def dataForType_(self, kind):
        self.reads.append(kind)
        if self.delay:
            time.sleep(self.delay)
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
    port, stop = _sse_server(8, 0.05)
    api = str(tmp_path / "api.sock")
    relay.Relay(loop, api, ("127.0.0.1", port))
    pb = StubPasteboard({"public.png": PNG_BYTES}, delay=1.5)
    _, path = _server(loop, tmp_path, pb)
    reply: list = []
    slow = threading.Thread(target=lambda: reply.append(_ask(path, b"IMAGE image/png\n")))
    slow.start()
    time.sleep(0.2)                           # the read is under way
    start = time.monotonic()
    with _unix_client(api) as c:              # a new connection while it runs
        times = _read_events(c, 8)
    stop()
    assert len(times) == 8 and times[0] - start < 0.5
    assert min(b - a for a, b in zip(times, times[1:])) > 0.02
    assert slow.is_alive()                    # the read still ran meanwhile
    slow.join(10)
    assert reply and reply[0].startswith(b"OK ")


def _captured_handoffs(server):
    """Record each socket the loop hands to the worker."""
    handed = []
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
    pb = StubPasteboard({"public.png": PNG_BYTES}, delay=0.3)
    server, path = _server(loop, tmp_path, pb)
    handed = _captured_handoffs(server)
    c = _unix_client(path)
    c.sendall(b"IMAGE image/png\n")
    c.close()                                 # gone before the answer
    assert _ask(path, b"IMAGE image/png\n").startswith(b"OK ")
    assert all(s.fileno() == -1 for s in handed) and len(handed) == 2


def test_stand_in_that_stops_reading_frees_the_worker(loop, tmp_path):
    big = b"\x89PNG" + b"x" * (8 << 20)     # far more than the socket buffers hold
    pb = StubPasteboard({"public.png": big})
    server, path = _server(loop, tmp_path, pb, send_timeout=0.5)
    stuck = _unix_client(path)
    stuck.sendall(b"IMAGE image/png\n")      # never read
    start = time.monotonic()
    # The worker gives up on the stuck stand-in after the timeout and
    # answers the next request.
    assert _ask(path, b"TYPES\n", timeout=10) == b"OK 10\nimage/png\n"
    assert 0.4 < time.monotonic() - start < 5
    assert any("cannot answer the container" in line for line in loop.logged)
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
    server, path = _server(loop, tmp_path, pb, read_deadline=0.2)
    idle = _unix_client(path)
    start = time.monotonic()
    assert idle.recv(10) == b""                     # closed by the server
    assert time.monotonic() - start < 3
    idle.close()
    assert _ask(path, b"TYPES\n") == b"OK 10\nimage/png\n"
    assert any("no request in" in line for line in loop.logged)
    # The worker frees the slot of the answered request after it closes the
    # socket, so the count reaches 0 a moment after the answer.
    deadline = time.monotonic() + 5
    while _in_loop(loop, lambda: server.open) != 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_a_full_queue_answers_busy(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES}, delay=1.0)
    server, path = _server(loop, tmp_path, pb, queue_max=1)
    slow = [threading.Thread(target=_ask, args=(path, b"IMAGE image/png\n"))
            for _ in range(2)]
    slow[0].start()
    deadline = time.monotonic() + 5
    while not pb.reads and time.monotonic() < deadline:
        time.sleep(0.01)                            # the worker holds the first
    slow[1].start()                                 # the second fills the queue
    deadline = time.monotonic() + 5
    while server._queue.qsize() < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _ask(path, b"TYPES\n") == clipboard._err(clipboard.BUSY)
    for t in slow:
        t.join(10)


def test_the_clipboard_holds_at_most_its_cap_of_connections(loop, tmp_path):
    server, path = _server(loop, tmp_path, StubPasteboard({"public.png": PNG_BYTES}),
                           max_connections=2)
    idle = [_unix_client(path), _unix_client(path)]
    _in_loop(loop, lambda: None)
    waiting = _unix_client(path)
    waiting.settimeout(0.5)
    waiting.sendall(b"TYPES\n")
    with pytest.raises(TimeoutError):
        waiting.recv(20)                            # not accepted yet
    idle[0].close()
    waiting.settimeout(5)
    assert waiting.recv(20) == b"OK 10\nimage/png\n"
    idle[1].close()
    waiting.close()


def test_close_closes_the_requests_that_wait_for_the_worker(loop, tmp_path):
    pb = StubPasteboard({"public.png": PNG_BYTES}, delay=1.0)
    server, path = _server(loop, tmp_path, pb)
    handed = _captured_handoffs(server)
    busy = threading.Thread(target=_ask, args=(path, b"IMAGE image/png\n"))
    busy.start()
    deadline = time.monotonic() + 5
    while not pb.reads and time.monotonic() < deadline:
        time.sleep(0.01)
    waiting = _unix_client(path)
    waiting.sendall(b"TYPES\n")
    deadline = time.monotonic() + 5
    while len(handed) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    server.close()
    assert handed[1].fileno() == -1                 # closed without an answer
    assert waiting.recv(20) == b""
    waiting.close()
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
    deadline = time.monotonic() + 5
    while not loop.logged and time.monotonic() < deadline:
        time.sleep(0.02)
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
