"""An app in a uvicorn server on a loopback port, for tests of what a body
read does to the connection, which a TestClient cannot show: it buffers
the body whole before the app runs."""
from __future__ import annotations

import contextlib
import http.client
import socket
import threading

import uvicorn


@contextlib.contextmanager
def serving(app):
    """Yield the port of ``app`` served on 127.0.0.1 on its own thread.
    The socket listens before the thread starts, so no wait is needed."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", ws="none",
                                           log_config=None))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]},
                              daemon=True)
    thread.start()
    try:
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(30)
        sock.close()
    assert not thread.is_alive(), "the server did not stop"


def send(conn: http.client.HTTPConnection, path: str, head: bytes,
         rest: bytes = b"", *, length: int | None = None,
         ctype: str = "application/json",
         ready: threading.Event | None = None,
         headers: dict | None = None):
    """POST ``head`` then ``rest`` on ``conn``, under the Content-Length
    ``length`` (theirs by default), and return the response, read whole.
    With ``ready``, ``rest`` goes once the server sets it, so the server
    has acted on ``head`` alone. A server that refuses a body by its
    length answers and closes before reading it, so a send it cuts off
    is kept on the response as ``sent_error`` and the answer read."""
    conn.putrequest("POST", path)
    conn.putheader("Content-Type", ctype)
    for k, v in (headers or {}).items():
        conn.putheader(k, v)
    conn.putheader("Content-Length",
                   str(len(head) + len(rest) if length is None else length))
    sent = None
    try:
        conn.endheaders(head)
        if rest:
            assert ready is None or ready.wait(30), \
                "the server did not act on the head before the rest was sent"
            conn.send(rest)
    except (ConnectionError, OSError) as e:
        if isinstance(e, TimeoutError):
            raise
        sent = e
    r = conn.getresponse()
    r.body = r.read()
    r.sent_error = sent
    return r


def peer(conn: http.client.HTTPConnection) -> socket.socket:
    """A second handle on the socket of ``conn``, connected first, which
    stays open when http.client closes its own on an answer that says
    close, so the test can read whether the server closed."""
    if conn.sock is None:
        conn.connect()
    return conn.sock.dup()


def closed(sock: socket.socket, timeout: float = 10) -> bool:
    """Whether the server ended the connection of ``sock`` (from
    :func:`peer`): it reads EOF within ``timeout`` seconds."""
    sock.settimeout(timeout)
    try:
        return sock.recv(1) == b""
    except (ConnectionResetError, BrokenPipeError):
        return True
    except TimeoutError:
        return False
    finally:
        sock.close()
