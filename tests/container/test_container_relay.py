"""The Mac-side relay of `gmlx launch --container`: one selectors loop that
joins accepted connections to their targets, and the clipboard server that
shares it. Real sockets on loopback and in a temp folder; no container, no
server. The clipboard tests read a stub pasteboard, except the conversion
test, which writes a private named NSPasteboard and never the user's own."""
from __future__ import annotations

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
    assert loop.logged == ["clipboard: sent an image of 108 bytes"]  # one per image read


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
    assert reply.startswith(b"ERR ") and b"over the 20 MB limit" in reply


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
    real = server._queue.put

    def put(item):
        if item is not None:
            handed.append(item[0])
        real(item)
    server._queue.put = put
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
    assert reply == b"ERR cannot read the Mac clipboard (boom)\n"
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
    assert any("cannot answer the guest" in line for line in loop.logged)
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
