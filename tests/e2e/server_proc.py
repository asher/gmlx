"""Launch / poll / tear down a real ``gmlx`` server subprocess.

Drives the actual entry point (``python -m gmlx.server``) on a free port so the
e2e exercises the whole wiring - argparse, monkeypatches, uvicorn, the residency
pool - exactly as a user runs it. Each scenario gets a fresh process in its own
session (clean group kill), with stdout/stderr captured to a log so a launch
failure is reportable.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class ServerProc:
    """A launched server. Use as a context manager; ``base_url`` is ready after
    ``wait_ready()``."""

    def __init__(self, serve_args: list, *, env_extra: Optional[dict] = None,
                 log_path: str, port: Optional[int] = None,
                 python: Optional[str] = None):
        self.port = port or free_port()
        self.serve_args = list(serve_args)
        self.env_extra = dict(env_extra or {})
        self.log_path = log_path
        self.python = python or sys.executable
        self.proc: Optional[subprocess.Popen] = None
        self._log_fh = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _argv(self) -> list:
        # positional/option serve args, then host/port forced to a loopback free port.
        # --foreground keeps the server attached to THIS process (the default start is
        # detached-background, which would exit the launcher and leak the child); the
        # menubar is GUI-only noise the harness never wants.
        return [self.python, "-m", "gmlx.server", *self.serve_args,
                "--foreground", "--no-menubar",
                "--host", "127.0.0.1", "--port", str(self.port)]

    def start(self) -> "ServerProc":
        env = dict(os.environ)
        # Guarantee no network: the HF gate + offline flags. The harness asserts no
        # download happens, so make the environment enforce it too.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.update(self.env_extra)
        self._log_fh = open(self.log_path, "w")
        self._log_fh.write(f"# argv: {' '.join(self._argv())}\n")
        for k in sorted(self.env_extra):
            self._log_fh.write(f"# env: {k}={self.env_extra[k]}\n")
        self._log_fh.flush()
        self.proc = subprocess.Popen(
            self._argv(), stdout=self._log_fh, stderr=subprocess.STDOUT,
            env=env, start_new_session=True)
        return self

    def wait_ready(self, timeout: float = 240.0, poll: float = 0.5) -> None:
        """Block until /health returns 200, the process exits, or timeout."""
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise ServerLaunchError(
                    f"server exited early (code {self.proc.returncode})\n"
                    + self.log_tail())
            try:
                with urllib.request.urlopen(url, timeout=poll * 2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(poll)
        raise ServerLaunchError(
            f"server not ready after {timeout:.0f}s\n" + self.log_tail())

    def log_tail(self, n: int = 60) -> str:
        try:
            if self._log_fh:
                self._log_fh.flush()
            with open(self.log_path) as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except OSError:
            return "(no log)"

    def stop(self, grace: float = 8.0) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self.proc.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> "ServerProc":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class ServerLaunchError(RuntimeError):
    pass
