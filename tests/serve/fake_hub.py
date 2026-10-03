"""A local stand-in for the Hugging Face Hub's HTTP side, for tests that run
the real huggingface_hub cache code against a scratch cache.

:class:`FakeHub` serves in-memory repos on 127.0.0.1 as the Hub serves
them: ``HEAD`` and ``GET`` of ``/<repo>/resolve/<revision>/<file>``, with
the ``X-Repo-Commit`` and ``ETag`` headers, a 404 with ``X-Error-Code:
EntryNotFound`` for a file the commit lacks, and the repo info route that
``snapshot_download`` reads. ``main`` resolves to the head commit. Every
request is recorded. :class:`BlackHole` accepts connections and never
answers, and :func:`refused_port` gives a port nothing listens on.
``fail_status`` makes the Hub answer every request with that status, and
``delay`` makes it wait that many seconds before each answer.

:func:`use_hub` points huggingface_hub at an endpoint and a scratch cache
for one test, by the environment and by the constants it read at import,
and hides the user's token, so a test neither reads it nor sends it.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


def git_sha1(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


class FakeHub:
    def __init__(self):
        self.repos: dict = {}          # repo -> {"head": sha, "commits": {sha: {name: bytes}}}
        self.requests: list = []       # (method, path)
        self.tokens: list = []         # the Authorization headers received
        self.fail_status = None        # answer every request with this status
        self.delay = 0.0               # seconds to wait before each answer
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.02},
                                        daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def commit(self, repo: str, sha: str, files: dict, head: bool = True):
        """Add commit ``sha`` of ``repo`` holding ``files`` (name -> bytes or
        str), and make it the head when ``head``."""
        r = self.repos.setdefault(repo, {"head": sha, "commits": {}})
        r["commits"][sha] = {k: v.encode() if isinstance(v, str) else v
                             for k, v in files.items()}
        if head:
            r["head"] = sha

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    def _resolve(self, repo: str, rev: str):
        r = self.repos.get(repo)
        if r is None:
            return None, None
        sha = r["head"] if rev == "main" else rev
        return sha, r["commits"].get(sha)

    def _handler(self):
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def handle(self):
                # A client that gave up on a delayed answer has closed.
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _send(self, status, headers, body=b""):
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _route(self):
                path = unquote(urlparse(self.path).path)
                hub.requests.append((self.command, path))
                if self.headers.get("Authorization"):
                    hub.tokens.append(self.headers["Authorization"])
                if hub.delay:
                    time.sleep(hub.delay)
                if hub.fail_status:
                    return self._send(hub.fail_status, {})
                parts = path.strip("/").split("/")
                if parts[:2] == ["api", "models"] and "revision" in parts:
                    i = parts.index("revision")
                    repo, rev = "/".join(parts[2:i]), parts[i + 1]
                    sha, files = hub._resolve(repo, rev)
                    if files is None:
                        return self._send(404, {"X-Error-Code": "RevisionNotFound"})
                    body = json.dumps({
                        "id": repo, "sha": sha,
                        "siblings": [{"rfilename": n} for n in sorted(files)],
                    }).encode()
                    return self._send(200, {"Content-Type": "application/json"},
                                      body)
                if "resolve" in parts:
                    i = parts.index("resolve")
                    repo, rev = "/".join(parts[:i]), parts[i + 1]
                    name = "/".join(parts[i + 2:])
                    sha, files = hub._resolve(repo, rev)
                    if files is None:
                        return self._send(404, {"X-Error-Code": "RevisionNotFound"})
                    if name not in files:
                        return self._send(404, {
                            "X-Error-Code": "EntryNotFound",
                            "X-Error-Message": "Entry not found",
                            "X-Repo-Commit": sha})
                    body = files[name]
                    return self._send(200, {
                        "X-Repo-Commit": sha,
                        "ETag": f'"{git_sha1(body)}"',
                        "Content-Type": "application/octet-stream"}, body)
                return self._send(404, {"X-Error-Code": "RepoNotFound"})

            do_GET = _route
            do_HEAD = _route

        return Handler


class BlackHole:
    """A listener that accepts connections and never answers. ``accepted``
    counts them."""

    def __init__(self):
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.accepted = 0
        self._held: list = []
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        host, port = self._sock.getsockname()[:2]
        return f"http://{host}:{port}"

    def _run(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.accepted += 1
            self._held.append(conn)

    def close(self):
        self._stop = True
        for c in self._held:
            c.close()
        self._sock.close()


def refused_port() -> str:
    """An endpoint on a port that nothing listens on."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def use_hub(monkeypatch, endpoint: str, cache: str, offline: bool = False):
    """Point huggingface_hub at ``endpoint`` and the scratch cache ``cache``
    for one test: HF_ENDPOINT and HF_HUB_CACHE, and the constants that
    huggingface_hub derived from them at import. The token paths point into
    ``cache``, where no token is."""
    from huggingface_hub import constants

    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(constants, "HF_TOKEN_PATH",
                        os.path.join(cache, "token"))
    monkeypatch.setattr(constants, "HF_STORED_TOKENS_PATH",
                        os.path.join(cache, "stored_tokens"))
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.setenv("HF_HUB_CACHE", cache)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(constants, "ENDPOINT", endpoint)
    monkeypatch.setattr(constants, "HUGGINGFACE_CO_URL_TEMPLATE",
                        endpoint + "/{repo_id}/resolve/{revision}/{filename}")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", cache)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    os.makedirs(cache, exist_ok=True)


def cache_state(cache: str, repo: str) -> dict:
    """refs/main and the files of each snapshot of ``repo`` in ``cache``."""
    store = os.path.join(cache, "models--" + repo.replace("/", "--"))
    snaps = os.path.join(store, "snapshots")
    out = {"main": None, "snapshots": {}}
    ref = os.path.join(store, "refs", "main")
    if os.path.isfile(ref):
        with open(ref) as f:
            out["main"] = f.read()
    if os.path.isdir(snaps):
        for sha in sorted(os.listdir(snaps)):
            root = os.path.join(snaps, sha)
            out["snapshots"][sha] = sorted(
                os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
                for d, _, fs in os.walk(root) for f in fs)
    return out
