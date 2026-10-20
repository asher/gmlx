"""Fixtures for the container-mode tests. ``test_container_entry.py`` needs
none of them, so the Linux CI job can run it with ``--noconftest``."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_FAKE = Path(__file__).with_name("fake_container.py")


class FakeContainer:
    """The state file of the fake ``container`` command."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        return json.loads(self.path.read_text())

    def save(self, state: dict) -> None:
        self.path.write_text(json.dumps(state))

    def update(self, **values) -> None:
        state = self.load()
        state.update(values)
        self.save(state)

    @property
    def log(self) -> list[list[str]]:
        return self.load().get("log", [])

    def calls(self, *prefix: str) -> list[list[str]]:
        return [a for a in self.log if a[:len(prefix)] == list(prefix)]


_RUN, _POPEN = subprocess.run, subprocess.Popen


def _in_process(binary: str):
    """``subprocess.run``, with each call of the fake ``container`` at
    ``binary`` run in this process, which saves a Python start per call. A
    call still starts a process when the test watches its process: through
    ``FAKE_CONTAINER_WATCH``, its own ``Popen``, or a timeout under a
    second."""
    def run(*args, **kw):
        argv = args[0] if args else kw.get("args")
        env = kw.get("env")
        environ = os.environ if env is None else env
        timeout = kw.get("timeout")
        if (not isinstance(argv, list) or argv[:1] != [binary]
                or environ.get("FAKE_CONTAINER_WATCH") or subprocess.Popen is not _POPEN
                or (timeout is not None and timeout < 1)):
            return _RUN(*args, **kw)
        import fake_container

        out, err = io.StringIO(), io.StringIO()
        # A signal waits for the end of the call, as the call of a process
        # of its own would end even if the signal stops launch.
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
        try:
            code = fake_container.call(argv[1:], environ, out, err,
                                       lambda: signal.raise_signal(signal.SIGINT))
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        if not kw.get("capture_output"):
            # As a child's would, the output goes to the descriptors.
            for fd, stream, text_out in ((1, sys.stdout, out), (2, sys.stderr, err)):
                stream.flush()
                os.write(fd, text_out.getvalue().encode())
            stdout = stderr = None
        elif kw.get("text") or kw.get("universal_newlines"):
            stdout, stderr = out.getvalue(), err.getvalue()
        else:
            stdout, stderr = out.getvalue().encode(), err.getvalue().encode()
        if kw.get("check") and code:
            raise subprocess.CalledProcessError(code, argv, stdout, stderr)
        return subprocess.CompletedProcess(argv, code, stdout, stderr)
    return run


def _remove_tree(path: str) -> None:
    """Remove ``path``, making a folder a test left unreadable readable first."""
    def fix(func, p, _exc):
        with contextlib.suppress(OSError):
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
            func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=fix)
    else:
        shutil.rmtree(path, onerror=fix)


@pytest.fixture
def tmp_path():
    """A folder of this test's own under /tmp. pytest's own lies in the
    per-user temporary folder, which launch never shares by default."""
    path = os.path.realpath(tempfile.mkdtemp(prefix="gmlx-tp-", dir="/tmp"))
    yield Path(path)
    _remove_tree(path)


@pytest.fixture(autouse=True)
def _own_launch_state(monkeypatch):
    """Launch data and cache folders of this test's own, outside
    ``tmp_path``, so no test reads the share history or runfiles of the
    real user."""
    path = os.path.realpath(tempfile.mkdtemp(prefix="gmlx-st-", dir="/tmp"))
    monkeypatch.setenv("XDG_DATA_HOME", os.path.join(path, "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", os.path.join(path, "cache"))
    # Apple container's own folder follows the test's HOME, so the kernel
    # this Mac has installed never reaches a test.
    monkeypatch.delenv("CONTAINER_APP_ROOT", raising=False)
    monkeypatch.setattr("gmlx.container.cli.account_home", Path.home)
    # The Mac has Rosetta, so no test edits Apple container's settings file
    # unless it takes Rosetta away itself, with HOME in its own folder.
    monkeypatch.setattr("gmlx.container.cli.ROSETTA_RUNTIME", Path(sys.executable))
    # A Mac of 64 GB, so the memory note of a 4G container comes only from a
    # test that sets a smaller Mac, whatever the machine that runs the tests.
    monkeypatch.setattr("gmlx.container.settings.mac_memory_bytes", lambda: 64 << 30)
    yield
    _remove_tree(path)


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    """No test opens an address in the Mac's browser. A test that checks
    what launch opens puts its own recorder in place of this one."""
    tried: list[str] = []

    def refuse(url):
        tried.append(url)
        pytest.fail(f"the test would open {url} in the Mac's browser")
    monkeypatch.setattr("gmlx.container.session.open_in_browser", refuse, raising=False)
    yield
    # The opener can run in a thread of its own, where pytest.fail ends only
    # that thread, so the test also fails here.
    if tried:
        pytest.fail(f"the test would open {', '.join(tried)} in the Mac's browser")


@pytest.fixture
def short_tmpdir(monkeypatch):
    """A short ``TMPDIR`` of this test's own. Session folders move there when
    the cache path is too long for a socket, and cleanup deletes launch
    folders there, so the real temporary folder must stay out of reach."""
    path = tempfile.mkdtemp(prefix="gmlx-t-", dir="/tmp")
    monkeypatch.setenv("TMPDIR", path)
    yield Path(path)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def fake_container(tmp_path, monkeypatch, short_tmpdir) -> Iterator[FakeContainer]:
    """A fake ``container`` first on ``PATH``, with the launch data and cache
    folders under ``tmp_path`` and a short ``TMPDIR`` of its own. Launch
    sees no other build on the Mac, because the fake containers of other
    test runs show in the Mac's process list. A test that needs another
    build sets ``images._other_builds`` itself."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    script = bin_dir / "container"
    script.write_text(f"#!/bin/sh\nexec {sys.executable} {_FAKE} \"$@\"\n")
    script.chmod(0o755)
    state = tmp_path / "container-state.json"
    state.write_text("{}")
    monkeypatch.setenv("FAKE_CONTAINER_STATE", str(state))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("gmlx.container.images._other_builds", lambda: False)
    monkeypatch.setattr(subprocess, "run", _in_process(str(script)))
    fake = FakeContainer(state)
    yield fake
    # A call that the real CLI would refuse fails the test, also when
    # launch handles the error.
    refused = fake.load().get("refused") if state.exists() else None
    if refused:
        pytest.fail("calls that Apple's container would refuse:\n" + "\n".join(refused))
