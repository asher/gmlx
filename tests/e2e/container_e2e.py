"""Helpers for the end-to-end scripts that run ``gmlx launch`` in a real
Apple container: the check table, calls of the ``container`` CLI, waits on
real conditions, HTTP probes, and a launch that runs in the background.

Nothing here starts a container. The scripts that import it do that through
``gmlx launch``.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

ANSI = re.compile(r"\x1b\[[0-9;?<>=]*[A-Za-z~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Za-z0-9]"
                  r"|\x1b[=>78]")


def plain(text: str) -> str:
    """``text`` without terminal control sequences."""
    return ANSI.sub("", text)


class Check:
    """The table of checks. Each call prints PASS or FAIL with its detail."""

    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []
        self.group = ""

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        shown = f"{self.group}: {name}" if self.group else name
        self.rows.append((shown, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}: {shown}" + (f" ({detail})" if detail else ""),
              flush=True)
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.rows if not ok]


def container(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    """Run ``container ARGS``. A call that outlives ``timeout`` returns code -1."""
    try:
        return subprocess.run(["container", *args], capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, -1, "", f"timed out after {timeout:.0f}s")


def container_ready() -> str | None:
    """None when Apple container 1.5.0 or newer runs, else the reason."""
    if shutil.which("container") is None:
        return "Apple container is not installed"
    out = container("--version").stdout
    m = re.search(r"version (\d+)\.(\d+)\.(\d+)", out)
    if not m or tuple(int(x) for x in m.groups()) < (1, 5, 0):
        return f"Apple container 1.5.0 or newer is needed, found {out.strip() or 'none'}"
    status = container("system", "status")
    if status.returncode != 0 or not re.search(r"^status\s+running\s*$", status.stdout, re.M):
        return "the container service is not running (container system start)"
    return None


def rows(*args: str) -> list[dict]:
    """The JSON rows of ``container ARGS --format json``."""
    try:
        out = json.loads(container(*args, "--format", "json").stdout or "[]")
    except json.JSONDecodeError:
        return []
    return [r for r in out if isinstance(r, dict)]


def image_refs() -> dict[str, str]:
    """Each image reference in the store, with the digest it names."""
    refs = {}
    for r in rows("image", "list"):
        conf = r.get("configuration") or r
        name = conf.get("name")
        digest = (conf.get("descriptor") or {}).get("digest")
        if name and digest:
            refs[name] = digest
    return refs


def volume_names() -> set[str]:
    return {(r.get("configuration") or r).get("name", "") for r in rows("volume", "list")} - {""}


class Box:
    """One container as ``container ls --all`` lists it."""

    def __init__(self, row: dict):
        conf = row.get("configuration") or {}
        self.name = row.get("id") or conf.get("id", "")
        self.state = _state(row)
        self.labels = conf.get("labels") or {}

    @property
    def client(self) -> str:
        return self.labels.get("gmlx.launch.client", "")

    @property
    def project(self) -> str:
        return self.labels.get("gmlx.launch.project", "")

    @property
    def pid(self) -> str:
        return self.labels.get("gmlx.launch.pid", "")

    def __repr__(self) -> str:
        return f"{self.name} ({self.client}, {self.project}, {self.state})"


def _state(row: dict) -> str:
    status = row.get("status")
    if isinstance(status, dict):
        return str(status.get("state") or "")
    return str(status or row.get("state") or "")


def boxes(all_states: bool = True) -> list[Box]:
    return [Box(r) for r in rows("ls", *(["--all"] if all_states else []))]


def builder_state() -> str:
    """The state of the image builder, or an empty string when there is none."""
    for r in rows("ls", "--all"):
        if (r.get("id") or (r.get("configuration") or {}).get("id")) == "buildkit":
            return _state(r)
    return ""


def wait_until(test, timeout: float, step: float = 1.0) -> bool:
    """Wait until ``test()`` is true, checking every ``step`` seconds until
    ``timeout`` ends. The check is the condition itself, such as a port that
    answers or a container that is gone."""
    deadline = time.monotonic() + timeout
    while True:
        if test():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(step)


def fetch(method: str, url: str, *, body: bytes | None = None, headers: dict | None = None,
         timeout: float = 30) -> tuple[int | None, bytes, dict]:
    """The status, body and headers of one request. A connection error
    gives None as the status and the error as the body."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})
    except (OSError, ValueError) as e:
        return None, repr(e).encode(), {}


def get_json(url: str, timeout: float = 10) -> dict | None:
    status, body, _ = fetch("GET", url, timeout=timeout)
    if status != 200:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def get_with_host(port: int, host: str) -> tuple[int | None, str]:
    """GET / at [::1] and ``port`` with the Host header ``host``."""
    conn = http.client.HTTPConnection("::1", port, timeout=10)
    try:
        conn.request("GET", "/", headers={"Host": host})
        r = conn.getresponse()
        return r.status, r.read().decode(errors="replace")
    except OSError as e:
        return None, repr(e)
    finally:
        conn.close()


def refused(host: str, port: int) -> bool:
    """Whether nothing on the Mac accepts a connection at ``host`` and ``port``."""
    try:
        with socket.create_connection((host, port), timeout=3):
            return False
    except OSError:
        return True


def free_port(avoid: set[int] = frozenset()) -> int:
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        if port not in avoid:
            return port


def stop_group(proc: subprocess.Popen) -> str:
    """Stop a process and its group with SIGTERM, and with SIGKILL after 60
    seconds. Return the output read until then."""
    def kill(sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)

    interrupted = False
    out = None
    kill(signal.SIGTERM)
    try:
        out, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    except KeyboardInterrupt:
        interrupted = True
    if out is None:
        kill(signal.SIGKILL)
        try:
            out, _ = proc.communicate(timeout=30)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as e:
            interrupted = interrupted or isinstance(e, KeyboardInterrupt)
            partial = getattr(e, "output", None)
            out = partial.decode(errors="replace") if isinstance(partial, bytes) else ""
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
    if interrupted:
        raise KeyboardInterrupt
    return out or ""


class Background:
    """A process that runs while the script goes on, in a session of its
    own. A thread collects its output into the log and into :attr:`text`."""

    def __init__(self, argv: list[str], cwd: str, env: dict, log: str):
        self.proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)
        self._lines: list[str] = []
        self._log = log
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        with open(self._log, "a") as f:
            for line in self.proc.stdout:
                self._lines.append(line)
                f.write(line)
                f.flush()
        self.proc.wait()

    @property
    def text(self) -> str:
        return "".join(self._lines)

    def wait_for(self, pattern: str, timeout: float) -> re.Match | None:
        deadline = time.monotonic() + timeout
        while True:
            m = re.search(pattern, self.text)
            if m or time.monotonic() >= deadline:
                return m
            if self.proc.poll() is not None:
                self._thread.join(5)
                return re.search(pattern, self.text)
            time.sleep(0.5)

    def signal(self, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.proc.pid, sig)

    def wait(self, timeout: float) -> int | None:
        """The exit code, or None when the process outlives ``timeout``."""
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        self._thread.join(5)
        return self.proc.returncode

    def stop(self, timeout: float = 90.0) -> int | None:
        """The exit code after SIGTERM to the process alone, or None when it
        outlived ``timeout`` and its group got SIGKILL."""
        if self.proc.poll() is None:
            self.signal(signal.SIGTERM)
            if self.wait(timeout) is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=30)
                self._thread.join(5)
                return None
        self._thread.join(5)
        return self.proc.returncode


def expect_plain(p, needle: str, timeout: float, count: int = 1) -> bool:
    """Wait until the transcript of the pty session ``p``, without terminal
    control sequences, holds ``needle`` at least ``count`` times. False once
    the child has exited without it."""
    deadline = time.monotonic() + timeout
    while True:
        if plain(p.transcript).count(needle) >= count:
            return True
        if p.proc.poll() is not None:
            p._drain(0.3)
            return plain(p.transcript).count(needle) >= count
        if time.monotonic() >= deadline:
            return False
        p._drain(min(0.5, max(0.0, deadline - time.monotonic())))


def rss_bytes(pid: int) -> int | None:
    """The resident size of process ``pid``, or None when it is gone."""
    done = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    try:
        return int(done.stdout.strip()) * 1024
    except ValueError:
        return None


class PeakRss:
    """The highest resident size of a process while the block runs, read
    every ``step`` seconds by a thread."""

    def __init__(self, pid: int, step: float = 0.2):
        self.pid, self.step = pid, step
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while True:
            size = rss_bytes(self.pid)
            if size is not None:
                self.peak = max(self.peak, size)
            if self._stop.wait(self.step):
                return

    def __enter__(self) -> "PeakRss":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(5)
