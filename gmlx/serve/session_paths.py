"""Where a gmlx server puts its launch session sockets, and the check launch
runs on a socket path a server names.

The server imports uvicorn and FastAPI for the sockets themselves, in
:mod:`gmlx.serve.patches.session_sockets`. Launch needs only these rules,
so they live here.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path

# The most connections a session socket serves at a time. Past it the
# server answers 503 without reading a body, and the launch relay holds no
# more than this many, so one container client cannot make the server hold
# more than this many request bodies.
SESSION_CONNECTIONS_MAX = 16
# macOS holds 104 bytes for a socket path, the final NUL included.
SOCKET_PATH_MAX = 103
ID_BYTES = 6
SOCKET_NAME = re.compile(r"^[0-9a-f]{12}\.sock$")
SOCKET_NAME_LEN = 2 * ID_BYTES + len(".sock")


def folder_key(host: str, port) -> str:
    """The part of a session folder name that names the server's bind."""
    return re.sub(r"[^A-Za-z0-9]+", "-", f"{host}-{port}").strip("-")


def _cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "gmlx"


def _tmp_root() -> Path:
    return Path(os.environ.get("TMPDIR") or "/tmp")


def socket_folders(host: str, port) -> list[Path]:
    """The folders that can hold the session sockets of the server at
    ``host:port``: one in the gmlx cache folder, and a shorter one under
    ``$TMPDIR`` for when the first would make a socket path too long."""
    key = folder_key(host, port)
    return [_cache_root() / f"sessions-{key}", _tmp_root() / f"gmlx-sessions-{key}"]


def owned_folder(folder: str | os.PathLike) -> bool:
    """Whether ``folder`` is a real folder of this user, not a link."""
    try:
        st = os.lstat(folder)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid()


def _real(path: str | os.PathLike) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def socket_refusal(path: str, port: int) -> str | None:
    """Why ``path`` is not a session socket of a gmlx server on ``port``, or
    None when it is one.

    The server's bind address can be spelled several ways, so the folder is
    matched by its gmlx name and the port rather than by the exact bind.
    What makes the path safe to relay to is ownership: a socket of this user
    that others cannot open, in a folder of this user that others cannot
    write, in one of the places a gmlx server puts its session folders."""
    if not isinstance(path, str) or not os.path.isabs(path):
        return "it is not an absolute path"
    folder, name = os.path.split(path)
    if not SOCKET_NAME.match(name):
        return "its name is not that of a session socket"
    parent, folder_name = os.path.split(folder)
    port_part = re.escape(str(int(port)))
    if not re.fullmatch(rf"(?:gmlx-)?sessions-(?:[A-Za-z0-9-]+-)?{port_part}", folder_name):
        return f"it is not in a session folder of a server on port {port}"
    in_cache = not folder_name.startswith("gmlx-")
    roots = ({_real(_cache_root()), _real("~/.cache/gmlx")} if in_cache else
             {_real(_tmp_root()), _real(tempfile.gettempdir()), _real("/tmp")})
    if _real(parent) not in roots:
        return "its folder is not where a gmlx server keeps session sockets"
    try:
        fst = os.lstat(folder)
        sst = os.lstat(path)
    except OSError as e:
        return f"it cannot be read ({e.strerror or e})"
    uid = os.getuid()
    if not stat.S_ISDIR(fst.st_mode) or fst.st_uid != uid or fst.st_mode & 0o077:
        return "its folder is not a private folder of this user"
    if not stat.S_ISSOCK(sst.st_mode) or sst.st_uid != uid or sst.st_mode & 0o077:
        return "it is not a private socket of this user"
    return None
