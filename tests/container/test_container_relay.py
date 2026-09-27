"""The Mac-side relay of `gmlx launch --container`: one selectors loop that
joins accepted connections to their targets. Real sockets on loopback and in
a temp folder; no container, no server."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from gmlx.container import relay


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


def test_forward_tries_ipv6_loopback_when_ipv4_refuses(loop, tmp_path):
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

    def answer():
        conn, _ = srv.accept()
        with conn:
            conn.sendall(b"v6")

    threading.Thread(target=answer, daemon=True).start()
    path = str(tmp_path / "fwd.sock")
    relay.Relay(loop, path, relay.loopback_targets(port))
    with _unix_client(path) as c:
        assert c.recv(10) == b"v6"
    srv.close()


def test_unreachable_target_closes_the_client_and_logs_once(loop, tmp_path):
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()                                        # nothing listens there
    path = str(tmp_path / "down.sock")
    relay.Relay(loop, path, relay.loopback_targets(port), name="forward 5432")
    with _unix_client(path) as c:
        assert c.recv(10) == b""                       # closed at once
    deadline = time.monotonic() + 5
    while not loop.logged and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(loop.logged) == 1
    assert loop.logged[0].startswith(
        f"forward 5432: cannot reach 127.0.0.1:{port} or [::1]:{port} (")


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


def test_resolve_targets_lists_localhost_addresses():
    addrs = relay.resolve_targets("localhost", 8080)
    assert addrs and all(port == 8080 for _, port in addrs)
    assert {host for host, _ in addrs} <= {"127.0.0.1", "::1"}
