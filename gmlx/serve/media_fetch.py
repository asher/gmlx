"""Fetch a media URL for a request, from public addresses only.

With ``server.media_urls`` on, the server fetches an image, audio or video
that a request names by an http(s) URL. The fetch runs on the Mac, so a
plain fetch would let any client that reaches the server use it to reach
the Mac's own loopback services, the local network or a cloud metadata
address, even from a container with no network.

:func:`fetch` resolves the host, refuses it when any of its addresses is not
public, and connects to an address it checked, so a second name lookup
cannot move the connection elsewhere. It checks every redirect the same
way, reads no proxy settings, stops reading past ``max_bytes``, and ends at
a fixed time however slowly the server sends.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import threading
import time
import urllib.parse

_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_READ_CHUNK = 64 << 10
_NAT64 = ipaddress.ip_network("64:ff9b::/96")

# Special-purpose ranges that are never a media host. ``is_global`` covers
# most of them, but its tables change between Python versions, and it
# counts multicast and IPv6 site-local addresses as global.
_BLOCKED_V4 = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
    "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24", "192.88.99.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4", "255.255.255.255/32"))
_BLOCKED_V6 = tuple(ipaddress.ip_network(n) for n in (
    "::/8", "64:ff9b:1::/48", "100::/64", "2001::/23", "2001:db8::/32",
    "3fff::/20", "fc00::/7", "fe80::/10", "fec0::/10", "ff00::/8"))


class FetchRefused(ValueError):
    """The server does not fetch this media URL. The message is for the
    client and names only the URL or host that the request or a redirect
    gave."""


def _embedded_ipv4(addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address that an IPv4-mapped, 6to4 or NAT64 address
    carries, else None."""
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    if addr.sixtofour is not None:
        return addr.sixtofour
    if addr in _NAT64:
        return ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    return None


def _is_public(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether ``addr`` is a public unicast address. An IPv6 address that
    carries an IPv4 address is judged by that IPv4 address, and a Teredo
    address is never public."""
    if isinstance(addr, ipaddress.IPv6Address):
        v4 = _embedded_ipv4(addr)
        if v4 is not None:
            return _is_public(v4)
        if addr.teredo is not None or any(addr in n for n in _BLOCKED_V6):
            return False
        if addr.is_site_local:
            return False
    elif any(addr in n for n in _BLOCKED_V4):
        return False
    return (addr.is_global and not addr.is_private and not addr.is_multicast
            and not addr.is_reserved and not addr.is_loopback
            and not addr.is_link_local and not addr.is_unspecified)


# The rule every resolved address must pass. Tests replace it to let a
# loopback test server through. Nothing reads it from the config or the
# environment.
_address_allowed = _is_public


def _checked_addresses(host: str, port: int, *, redirected: bool) -> list:
    """The ``getaddrinfo`` entries of ``host``, after every address passed
    :data:`_address_allowed`."""
    what = "redirects to" if redirected else "names"
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM,
                                   proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError) as e:
        raise FetchRefused(f"the media URL {what} the host {host!r}, which cannot be "
                           f"resolved ({e})") from None
    if not infos:
        raise FetchRefused(f"the media URL {what} the host {host!r}, which has no "
                           "address")
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            addr = None
        if addr is None or not _address_allowed(addr):
            raise FetchRefused(f"the media URL {what} the host {host!r}, which is on "
                               "this Mac or a private network, and the server fetches "
                               "media only from public addresses")
    return infos


def _connect(infos: list, timeout: float, end: float) -> socket.socket:
    """A socket connected to the first checked address that answers before
    ``end``. It connects by address, so no name is looked up again."""
    last: OSError | None = None
    for family, kind, proto, _canon, sockaddr in infos:
        left = end - time.monotonic()
        if left <= 0:
            raise TimeoutError("the fetch ran out of time")
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(min(timeout, left))
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            sock.close()
            last = e
    raise last or OSError("no address to connect to")


class _PinnedHTTP(http.client.HTTPConnection):
    """An HTTP connection to addresses already checked. ``host`` stays the
    URL's host, so the Host header names it."""

    def __init__(self, host: str, port: int, infos: list, timeout: float, end: float):
        super().__init__(host, port, timeout=timeout)
        self._infos, self._connect_timeout, self._end = infos, timeout, end
        self.held: socket.socket | None = None

    def connect(self) -> None:
        self.sock = self.held = _connect(self._infos, self._connect_timeout, self._end)
        _in_time(self._end)


class _PinnedHTTPS(http.client.HTTPSConnection):
    """An HTTPS connection to addresses already checked, verifying the
    certificate for the URL's host."""

    def __init__(self, host: str, port: int, infos: list, timeout: float, end: float):
        self._tls = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=self._tls)
        self._infos, self._connect_timeout, self._end = infos, timeout, end
        self.held: socket.socket | None = None

    def connect(self) -> None:
        # The plain socket is the connection's until the handshake ends, so
        # the deadline can shut it during the handshake too.
        sock = self.held = _connect(self._infos, self._connect_timeout, self._end)
        try:
            self.sock = self.held = self._tls.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise
        _in_time(self._end)


def _in_time(end: float) -> None:
    # The deadline timer may fire between a connect and the assignment of
    # the socket, when it has nothing to shut.
    if time.monotonic() >= end:
        raise TimeoutError("the fetch ran out of time")


def _cut(conn) -> None:
    """Shut the socket of ``conn``, which ends a read that waits on it. The
    connection keeps the socket in ``held``, since http.client passes it to
    the response of a body that ends when the connection closes."""
    sock = getattr(conn, "held", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _target(url: str, *, redirected: bool):
    """The scheme, host, port and request path of ``url``, refusing what
    the fetch does not support."""
    what = "redirects to" if redirected else "is"
    try:
        split = urllib.parse.urlsplit(url)
        port = split.port
    except ValueError:
        raise FetchRefused(f"the media URL {what} {url!r}, which is not a valid URL") \
            from None
    scheme = split.scheme.lower()
    if scheme not in ("http", "https"):
        raise FetchRefused(f"the media URL {what} a {scheme or 'relative'} URL, and the "
                           "server fetches only http and https")
    if split.username is not None or split.password is not None or "@" in split.netloc:
        raise FetchRefused("the media URL holds a user name or password, which the "
                           "server does not send")
    host = split.hostname
    if not host:
        raise FetchRefused(f"the media URL {what} {url!r}, which names no host")
    port = port or (443 if scheme == "https" else 80)
    path = split.path or "/"
    if split.query:
        path = f"{path}?{split.query}"
    return scheme, host, port, path


def _size(n: int) -> str:
    return f"{n >> 20} MiB" if n >= 1 << 20 and not n % (1 << 20) else f"{n} bytes"


def _read_body(resp: http.client.HTTPResponse, max_bytes: int, url: str) -> bytes:
    length = resp.getheader("Content-Length")
    if length is not None:
        try:
            size = int(length)
        except ValueError:
            size = None
        if size is not None and size > max_bytes:
            raise FetchRefused(f"the media at {url!r} is larger than "
                               f"{_size(max_bytes)}")
    chunks, total = [], 0
    while True:
        chunk = resp.read(min(_READ_CHUNK, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise FetchRefused(f"the media at {url!r} is larger than "
                               f"{_size(max_bytes)}")
    return b"".join(chunks)


def fetch(url: str, *, max_bytes: int, timeout: float = 10.0,
          max_redirects: int = 5, total: float = 60.0) -> bytes:
    """The body of an http(s) ``url``, fetched from public addresses only.

    Raises :class:`FetchRefused` for another scheme, a user name in the URL,
    a host with any address that is not public, a redirect that breaks one
    of these rules, more than ``max_redirects`` redirects, a final status
    other than 2xx, a body over ``max_bytes``, a connection that fails or
    stays silent for ``timeout`` seconds, or a fetch, redirects included,
    that takes more than ``total`` seconds."""
    end = time.monotonic() + total
    late = f"the media URL {url!r} did not arrive within {total:g} seconds"
    current, redirected = url, False
    for _hop in range(max_redirects + 1):
        scheme, host, port, path = _target(current, redirected=redirected)
        infos = _checked_addresses(host, port, redirected=redirected)
        left = end - time.monotonic()
        if left <= 0:
            raise FetchRefused(late)
        cls = _PinnedHTTPS if scheme == "https" else _PinnedHTTP
        conn = cls(host, port, infos, timeout, end)
        timer = threading.Timer(left, _cut, (conn,))
        timer.daemon = True
        timer.start()
        try:
            conn.request("GET", path, headers={
                "User-Agent": "gmlx", "Accept": "*/*",
                "Accept-Encoding": "identity"})
            resp = conn.getresponse()
            if resp.status in _REDIRECTS:
                location = resp.getheader("Location")
                if not location:
                    raise FetchRefused(f"the media URL {current!r} answered "
                                       f"{resp.status} with no Location")
                current = urllib.parse.urljoin(current, location.strip())
                redirected = True
                continue
            if not 200 <= resp.status < 300:
                raise FetchRefused(f"the media URL {current!r} answered HTTP "
                                   f"{resp.status}")
            body = _read_body(resp, max_bytes, current)
            # A shut socket reads as the end of a body with no length, so
            # a body that ends at the deadline may be cut short.
            if time.monotonic() >= end:
                raise FetchRefused(late)
            return body
        except FetchRefused:
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError) as e:
            if time.monotonic() >= end:
                raise FetchRefused(late) from None
            if isinstance(e, TimeoutError):
                raise FetchRefused(f"the media URL {current!r} did not answer within "
                                   f"{timeout:g} seconds") from None
            raise FetchRefused(f"the media URL {current!r} cannot be fetched "
                               f"({type(e).__name__}: {e})") from None
        finally:
            timer.cancel()
            conn.close()
    raise FetchRefused(f"the media URL {url!r} redirects more than {max_redirects} "
                       "times")
