"""The media URL fetcher reaches public addresses only, checks every
redirect, connects to the address it checked and caps the body."""

from __future__ import annotations

import http.server
import ipaddress
import socket
import ssl
import threading
import time

import pytest

from gmlx.serve import media_fetch as mf

_BODY = b"\x89PNG fake image bytes"
_CAP = 100 << 10


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hosts: list = []

    def log_message(self, format, *args):
        pass

    def _send(self, status, body=b"", headers=()):
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        if not any(k.lower() in ("content-length", "transfer-encoding")
                   for k, _ in headers):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        type(self).hosts.append(self.headers.get("Host"))
        port = _port(self.server)
        routes = {
            "/img": lambda: self._send(200, _BODY),
            "/query?a=1": lambda: self._send(200, b"query ok"),
            "/rel": lambda: self._send(302, headers=[("Location", "img")]),
            "/abs": lambda: self._send(
                301, headers=[("Location", f"http://127.0.0.1:{port}/img")]),
            "/to-private": lambda: self._send(
                302, headers=[("Location", "http://10.0.0.1/secret")]),
            "/to-metadata": lambda: self._send(
                307, headers=[("Location", "http://169.254.169.254/latest/")]),
            "/to-file": lambda: self._send(
                302, headers=[("Location", "file:///etc/passwd")]),
            "/to-ftp": lambda: self._send(
                308, headers=[("Location", "ftp://example.com/x")]),
            "/loop": lambda: self._send(302, headers=[("Location", "/loop")]),
            "/no-location": lambda: self._send(302),
            "/missing": lambda: self._send(404, b"no"),
            "/error": lambda: self._send(500, b"no"),
            "/big-length": self._big_length,
            "/big-chunked": self._big_chunked,
            "/small-chunked": self._small_chunked,
            "/drip-body": self._drip_body,
            "/drip-length": self._drip_length,
            "/drip-headers": self._drip_headers,
        }
        route = routes.get(self.path)
        if route is None:
            self._send(404)
        else:
            route()

    def _big_length(self):
        # Headers only: a fetch that tried to read the body would wait for
        # bytes that never come and fail with a timeout instead.
        self.send_response(200)
        self.send_header("Content-Length", str(_CAP + 1))
        self.end_headers()
        self.wfile.flush()
        _release(self.server).wait(10)

    def _drip(self, head: bytes, piece: bytes) -> None:
        # One small piece at a time, each well inside the read timeout, for
        # longer than the whole fetch may take.
        try:
            self.wfile.write(head)
            self.wfile.flush()
            for _ in range(200):
                if _release(self.server).wait(0.05):
                    return
                self.wfile.write(piece)
                self.wfile.flush()
        except OSError:
            pass

    def _drip_body(self):
        # No length: the body ends when the connection closes.
        self._drip(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n", b"x")

    def _drip_length(self):
        self._drip(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n", b"x")

    def _drip_headers(self):
        self._drip(b"HTTP/1.1 200 OK\r\n", b"X-Slow: 1\r\n")

    def _chunks(self, total):
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        sent, piece = 0, b"x" * 8192
        try:
            while sent < total:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                sent += len(piece)
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass

    def _big_chunked(self):
        self._chunks(8 << 20)

    def _small_chunked(self):
        self._chunks(16384)


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = threading.Event()

    def handle_error(self, request, client_address):
        # A fetch that refuses a body closes its connection mid-response.
        pass


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    _Handler.hosts = []
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.release.set()
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def loopback_allowed(monkeypatch):
    """Let the loopback test server through, and nothing else the rule
    refuses."""
    monkeypatch.setattr(mf, "_address_allowed",
                        lambda a: a.is_loopback or mf._is_public(a))


def _port(srv) -> int:
    return int(srv.server_address[1])


def _release(srv) -> threading.Event:
    assert isinstance(srv, _Server)
    return srv.release


def _url(srv, path):
    return f"http://127.0.0.1:{_port(srv)}{path}"


def _fetch(url, **kw):
    kw.setdefault("max_bytes", _CAP)
    kw.setdefault("timeout", 5.0)
    return mf.fetch(url, **kw)


# The address rule

_REFUSED_V4 = [
    "127.0.0.1", "127.1.2.3", "10.1.2.3", "172.16.0.1", "172.31.255.255",
    "192.168.1.1", "169.254.169.254", "100.64.0.1", "100.127.255.255",
    "224.0.0.1", "239.255.255.250", "0.0.0.0", "0.1.2.3", "255.255.255.255",
    "240.0.0.1", "192.0.0.9", "192.0.2.1", "198.51.100.1", "203.0.113.1",
    "198.18.0.1", "192.88.99.1"]
_REFUSED_V6 = [
    "::1", "::", "fe80::1", "fc00::1", "fd12:3456::1", "fec0::1", "ff02::1",
    "ff0e::1", "2001:db8::1", "3fff::1", "100::1", "::7f00:1", "::808:808",
    "2001:0:4136:e378:8000:63bf:3fff:fdd2", "64:ff9b:1::1"]


@pytest.mark.parametrize("addr", _REFUSED_V4 + _REFUSED_V6)
def test_the_explicit_ranges_hold_every_special_address(addr):
    # The rule does not depend on the stdlib's tables, which change
    # between Python versions.
    ip = ipaddress.ip_address(addr)
    ranges = mf._BLOCKED_V4 if ip.version == 4 else mf._BLOCKED_V6
    assert any(ip in n for n in ranges)


def _lenient(addr: str, **flags) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """``addr`` as an address whose stdlib flags all call it public, as a
    Python with other tables could, apart from ``flags``."""
    base = type(ipaddress.ip_address(addr))
    values = {"is_global": True, "is_private": False, "is_multicast": False,
              "is_reserved": False, "is_loopback": False, "is_link_local": False,
              "is_unspecified": False, "is_site_local": False, **flags}
    props = {k: property(lambda self, v=v: v) for k, v in values.items()
             if hasattr(base, k)}
    lenient = type("Lenient", (base,), props)(addr)
    assert isinstance(lenient, (ipaddress.IPv4Address, ipaddress.IPv6Address))
    return lenient


@pytest.mark.parametrize("addr", [
    a for a in _REFUSED_V4 + _REFUSED_V6])
def test_the_rule_holds_when_the_stdlib_calls_an_address_public(addr):
    assert not mf._is_public(_lenient(addr))


def test_a_teredo_address_is_refused_on_its_own(monkeypatch):
    monkeypatch.setattr(mf, "_BLOCKED_V6", ())
    assert not mf._is_public(_lenient("2001:0:4136:e378:8000:63bf:3fff:fdd2"))


def test_an_address_the_stdlib_calls_not_global_is_refused():
    assert mf._is_public(_lenient("8.8.8.8"))
    assert not mf._is_public(_lenient("8.8.8.8", is_global=False))


@pytest.mark.parametrize("addr", [
    "127.0.0.1", "127.1.2.3", "10.1.2.3", "172.16.0.1", "172.31.255.255",
    "192.168.1.1", "169.254.169.254", "100.64.0.1", "100.127.255.255",
    "224.0.0.1", "239.255.255.250", "0.0.0.0", "0.1.2.3", "255.255.255.255",
    "240.0.0.1", "192.0.0.9", "192.0.2.1", "198.51.100.1", "203.0.113.1",
    "198.18.0.1", "192.88.99.1",
    "::1", "::", "fe80::1", "fc00::1", "fd12:3456::1", "fec0::1", "ff02::1",
    "ff0e::1", "2001:db8::1", "3fff::1", "100::1", "::7f00:1", "::808:808",
    "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.169.254",
    "2002:a00:1::1", "2002:7f00:1::1", "2001:0:4136:e378:8000:63bf:3fff:fdd2",
    "64:ff9b::a00:1", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe", "64:ff9b:1::1",
])
def test_addresses_that_are_not_public_are_refused(addr):
    assert not mf._is_public(ipaddress.ip_address(addr))


@pytest.mark.parametrize("addr", [
    "8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111",
    "2001:4860:4860::8888", "::ffff:8.8.8.8", "2002:808:808::1",
    "64:ff9b::808:808",
])
def test_public_addresses_are_allowed(addr):
    assert mf._is_public(ipaddress.ip_address(addr))


def test_the_default_rule_is_the_public_rule():
    assert mf._address_allowed is mf._is_public


def _fake_resolver(monkeypatch, answers):
    """Answer ``getaddrinfo`` with ``answers`` (a list of IP strings, or a
    list of such lists for successive calls) and record each name asked."""
    calls = []

    def getaddrinfo(host, port, *args, **kwargs):
        calls.append(host)
        seq = answers[len(calls) - 1] if isinstance(answers[0], list) else answers
        out = []
        for ip in seq:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, port, 0, 0) if family == socket.AF_INET6 else (ip, port)
            out.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr))
        return out

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return calls


def _no_connect(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("connected to an unchecked host")
    monkeypatch.setattr(mf, "_connect", refuse)


def test_a_name_with_one_private_address_is_refused(monkeypatch):
    _fake_resolver(monkeypatch, ["93.184.216.34", "10.0.0.5"])
    _no_connect(monkeypatch)
    with pytest.raises(mf.FetchRefused, match="'media.example', which is on this Mac"):
        _fetch("http://media.example/cat.png")


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x", "http://[::1]/x", "http://169.254.169.254/latest/",
    "http://[::ffff:127.0.0.1]/x", "http://10.0.0.1:8080/x", "http://0.0.0.0/x",
])
def test_a_literal_address_that_is_not_public_is_refused(monkeypatch, url):
    _no_connect(monkeypatch)
    with pytest.raises(mf.FetchRefused, match="public addresses"):
        _fetch(url)


def test_the_connection_goes_to_the_checked_address(monkeypatch, server,
                                                     loopback_allowed):
    port = _port(server)
    calls = _fake_resolver(monkeypatch, [["127.0.0.1"], ["10.9.9.9"]])
    assert _fetch(f"http://media.example:{port}/img") == _BODY
    assert calls == ["media.example"]
    assert _Handler.hosts == [f"media.example:{port}"]


def test_no_proxy_setting_is_read(monkeypatch, server, loopback_allowed):
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                 "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://10.9.9.9:3128")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    assert _fetch(_url(server, "/img")) == _BODY


# Schemes and URL forms

@pytest.mark.parametrize("url, message", [
    ("file:///etc/passwd", "a file URL"),
    ("ftp://example.com/x", "a ftp URL"),
    ("gopher://example.com/x", "a gopher URL"),
    ("//example.com/x", "a relative URL"),
    ("/etc/passwd", "a relative URL"),
    ("http:///x", "names no host"),
    ("http://example.com:99999/x", "not a valid URL"),
])
def test_other_schemes_and_forms_are_refused(monkeypatch, url, message):
    _no_connect(monkeypatch)
    with pytest.raises(mf.FetchRefused, match=message):
        _fetch(url)


@pytest.mark.parametrize("url", [
    "http://user:pw@example.com/x", "http://user@example.com/x",
    "https://:pw@example.com/x",
])
def test_a_user_name_or_password_is_refused(monkeypatch, url):
    def resolve(*a, **k):
        raise AssertionError("resolved a URL with a user name")
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(mf.FetchRefused, match="user name or password"):
        _fetch(url)


# Redirects

@pytest.mark.parametrize("path", ["/rel", "/abs"])
def test_a_redirect_to_an_allowed_address_is_followed(server, loopback_allowed, path):
    assert _fetch(_url(server, path)) == _BODY


def test_the_query_is_sent(server, loopback_allowed):
    assert _fetch(_url(server, "/query?a=1#frag")) == b"query ok"


@pytest.mark.parametrize("path, host", [("/to-private", "10.0.0.1"),
                                        ("/to-metadata", "169.254.169.254")])
def test_a_redirect_to_an_address_that_is_not_public_is_refused(
        server, loopback_allowed, path, host):
    with pytest.raises(mf.FetchRefused, match=f"redirects to the host '{host}'"):
        _fetch(_url(server, path))


@pytest.mark.parametrize("path, scheme", [("/to-file", "file"), ("/to-ftp", "ftp")])
def test_a_redirect_to_another_scheme_is_refused(server, loopback_allowed, path,
                                                 scheme):
    with pytest.raises(mf.FetchRefused, match=f"redirects to a {scheme} URL"):
        _fetch(_url(server, path))


def test_a_redirect_chain_past_the_limit_is_refused(server, loopback_allowed):
    with pytest.raises(mf.FetchRefused, match="redirects more than 3 times"):
        _fetch(_url(server, "/loop"), max_redirects=3)
    assert len(_Handler.hosts) == 4


def test_a_redirect_with_no_location_is_refused(server, loopback_allowed):
    with pytest.raises(mf.FetchRefused, match="302 with no Location"):
        _fetch(_url(server, "/no-location"))


def test_every_hop_is_resolved_and_checked(monkeypatch, server):
    port = _port(server)
    checked = []

    def allowed(addr):
        checked.append(str(addr))
        return addr.is_loopback
    monkeypatch.setattr(mf, "_address_allowed", allowed)
    assert _fetch(f"http://127.0.0.1:{port}/rel") == _BODY
    assert checked == ["127.0.0.1", "127.0.0.1"]


# Status and size

@pytest.mark.parametrize("path, status", [("/missing", 404), ("/error", 500)])
def test_a_final_status_other_than_2xx_is_refused(server, loopback_allowed, path,
                                                  status):
    with pytest.raises(mf.FetchRefused, match=f"answered HTTP {status}"):
        _fetch(_url(server, path))


def test_a_content_length_over_the_cap_is_refused_before_the_body(
        server, loopback_allowed):
    with pytest.raises(mf.FetchRefused, match="larger than 102400 bytes") as e:
        _fetch(_url(server, "/big-length"), timeout=3.0)
    assert "did not answer" not in str(e.value)


def test_a_chunked_body_over_the_cap_is_cut_off(server, loopback_allowed):
    with pytest.raises(mf.FetchRefused, match="is larger than"):
        _fetch(_url(server, "/big-chunked"))


def test_a_chunked_body_under_the_cap_is_read(server, loopback_allowed):
    assert _fetch(_url(server, "/small-chunked")) == b"x" * 16384


def test_a_body_at_the_cap_is_read(server, loopback_allowed):
    assert _fetch(_url(server, "/img"), max_bytes=len(_BODY)) == _BODY
    with pytest.raises(mf.FetchRefused, match="is larger than"):
        _fetch(_url(server, "/img"), max_bytes=len(_BODY) - 1)


def test_a_silent_server_times_out(server, loopback_allowed):
    # The body never arrives, and the cap is high enough that it is read.
    with pytest.raises(mf.FetchRefused, match="did not answer within 0.5 seconds"):
        _fetch(_url(server, "/big-length"), max_bytes=_CAP + 1, timeout=0.5)


@pytest.mark.parametrize("path", ["/drip-body", "/drip-length", "/drip-headers"])
def test_a_server_that_sends_slowly_is_cut_off_at_the_deadline(server, loopback_allowed,
                                                               path):
    # Each piece comes well inside the read timeout, so only the deadline
    # ends the fetch, and the part of the body read by then is not returned.
    start = time.monotonic()
    with pytest.raises(mf.FetchRefused, match="did not arrive within 0.5 seconds"):
        _fetch(_url(server, path), timeout=5.0, total=0.5)
    assert time.monotonic() - start < 3.0


def test_the_deadline_covers_every_redirect(server, loopback_allowed, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(mf.time, "monotonic", lambda: clock[0])
    real = mf._checked_addresses

    def slow(host, port, *, redirected):
        clock[0] += 0.4          # each hop's name lookup takes 0.4 seconds
        return real(host, port, redirected=redirected)
    monkeypatch.setattr(mf, "_checked_addresses", slow)
    with pytest.raises(mf.FetchRefused, match="did not arrive within 1 seconds"):
        _fetch(_url(server, "/loop"), total=1.0, max_redirects=5)


def test_a_fetch_inside_the_deadline_is_read(server, loopback_allowed):
    assert _fetch(_url(server, "/rel"), total=5.0) == _BODY


def test_a_refused_connection_is_a_clean_refusal(loopback_allowed):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    with pytest.raises(mf.FetchRefused, match="cannot be fetched"):
        _fetch(f"http://127.0.0.1:{port}/x")


# TLS

def test_https_verifies_the_certificate_for_the_url_host(monkeypatch):
    _fake_resolver(monkeypatch, ["93.184.216.34"])
    seen = {}

    class _Sock:
        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(mf, "_connect", lambda infos, timeout, end: (
        seen.setdefault("connect", [i[4] for i in infos]), _Sock())[1])
    real = ssl.create_default_context

    def context(*a, **k):
        ctx = real(*a, **k)
        seen["check_hostname"] = ctx.check_hostname
        seen["verify_mode"] = ctx.verify_mode

        class _Ctx:
            # Python 3.11's HTTPSConnection also reads verify_mode and
            # check_hostname from the context.
            def __getattr__(self, name):
                return getattr(ctx, name)

            def wrap_socket(self, sock, server_hostname=None):
                seen["server_hostname"] = server_hostname
                raise ssl.SSLCertVerificationError("stop here")
        return _Ctx()

    monkeypatch.setattr(mf.ssl, "create_default_context", context)
    with pytest.raises(mf.FetchRefused, match="cannot be fetched"):
        _fetch("https://media.example/cat.png")
    assert seen["check_hostname"] is True
    assert seen["verify_mode"] == ssl.CERT_REQUIRED
    assert seen["server_hostname"] == "media.example"
    assert seen["connect"] == [("93.184.216.34", 443)]
    assert seen["closed"] is True
