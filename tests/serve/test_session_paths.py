"""The rules for where a gmlx server keeps its launch session sockets, and
the check launch runs on a socket path a server names."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import pytest

from gmlx.serve import session_paths as sp


@pytest.fixture
def tmp(monkeypatch):
    # Short enough for a socket path, and the TMPDIR a server would use.
    root = tempfile.mkdtemp(prefix="sp-", dir="/tmp")
    monkeypatch.setenv("TMPDIR", root)
    socks = []

    def make(folder_name, name="0123456789ab.sock", folder_mode=0o700, mode=0o600,
             under=None):
        folder = Path(under or root) / folder_name
        folder.mkdir(mode=folder_mode, exist_ok=True)
        os.chmod(folder, folder_mode)
        path = folder / name
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(path))
        os.chmod(path, mode)
        socks.append(s)
        return str(path)
    yield make
    for s in socks:
        s.close()
    import shutil
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::", "::1", "localhost",
                                  "192.168.1.20"])
def test_every_folder_a_server_makes_passes_for_its_port(tmp, host):
    folder = sp.socket_folders(host, 8080)[1]
    assert sp.socket_refusal(tmp(folder.name), 8080) is None
    assert sp.socket_refusal(tmp(folder.name, "abcdef012345.sock"), 8081) is not None


def test_the_cache_folder_passes_too(tmp, monkeypatch):
    import shutil

    cache = Path(tempfile.mkdtemp(prefix="sc-", dir="/tmp"))
    try:
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        (cache / "gmlx").mkdir()
        folder = sp.socket_folders("127.0.0.1", 8080)[0]
        assert sp.socket_refusal(tmp(folder.name, under=cache / "gmlx"), 8080) is None
    finally:
        shutil.rmtree(cache, ignore_errors=True)


@pytest.mark.parametrize("kw, why", [
    ({"name": "agent.sock"}, "its name is not that of a session socket"),
    ({"folder_name": "run"}, "not in a session folder of a server on port 8080"),
    ({"folder_mode": 0o755}, "its folder is not a private folder of this user"),
    ({"mode": 0o777}, "it is not a private socket of this user"),
])
def test_a_path_that_is_not_a_private_session_socket_is_refused(tmp, kw, why):
    folder_name = kw.pop("folder_name", "gmlx-sessions-127-0-0-1-8080")
    assert why in sp.socket_refusal(tmp(folder_name, **kw), 8080)


def test_a_session_folder_outside_the_server_places_is_refused(tmp):
    elsewhere = Path(tempfile.mkdtemp(prefix="se-", dir="/tmp"))
    try:
        path = tmp("gmlx-sessions-127-0-0-1-8080", under=elsewhere)
        assert "not where a gmlx server keeps" in sp.socket_refusal(path, 8080)
    finally:
        import shutil
        shutil.rmtree(elsewhere, ignore_errors=True)


def test_a_file_or_a_linked_folder_is_refused(tmp):
    real = tmp("gmlx-sessions-127-0-0-1-8081")
    link = Path(os.environ["TMPDIR"]) / "gmlx-sessions-127-0-0-1-8080"
    link.symlink_to(Path(real).parent)
    assert "not a private folder" in sp.socket_refusal(str(link / Path(real).name), 8080)
    plain = Path(os.environ["TMPDIR"]) / "gmlx-sessions-8082"
    plain.mkdir(mode=0o700)
    (plain / "0123456789ab.sock").write_text("")
    os.chmod(plain / "0123456789ab.sock", 0o600)
    assert "not a private socket" in sp.socket_refusal(str(plain / "0123456789ab.sock"), 8082)
    assert sp.socket_refusal("relative/0123456789ab.sock", 8080) == "it is not an absolute path"
    assert "cannot be read" in sp.socket_refusal(
        os.environ["TMPDIR"] + "/gmlx-sessions-8083/0123456789ab.sock", 8083)
