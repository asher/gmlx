"""The Mac ports of the browser apps of container mode.

A browser keeps the service workers, the storage and the HTTP cache of a
page by its origin, and the origin holds the port. A guest page can leave
all three at its address, and they stay after the session ends. So each
project of a browser app gets a Mac port of its own from :data:`FIRST` to
:data:`LAST`. Host mode never uses these ports, so what a guest page leaves
never reaches the pages of another project or of a client that runs on the
Mac. Cookies are kept by host name only, so a port of its own does not keep
them apart.

A record in the launch data folder keeps the port of each client and
project, so a project keeps its address from one launch to the next. A lock
keeps two launches that start at the same time from taking one port. An
entry stays while its project has a private home, or while the launch that
took the port runs.
"""

from __future__ import annotations

import json
import os
import stat

from . import relay
from .settings import Busy, private_home_path
from .state import FileLock, data_dir, data_path, write_record

FIRST = 3100
LAST = 3199
_RECORD = "web-ports.json"
_LOCK = "web-ports.lock"
_RECORD_MAX = 1 << 20

Key = tuple[str, str]


def _alive(pid) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _in_range(port) -> bool:
    return isinstance(port, int) and not isinstance(port, bool) and FIRST <= port <= LAST


def _read() -> dict[Key, dict]:
    """The entries of the record by client and project. A missing or
    damaged record has none, and an entry that is not in the correct form
    is left out."""
    try:
        fd = os.open(data_path() / _RECORD, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return {}
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > _RECORD_MAX:
            return {}
        doc = json.loads(os.read(fd, _RECORD_MAX).decode())
    except (OSError, ValueError, RecursionError):
        return {}
    finally:
        os.close(fd)
    out: dict[Key, dict] = {}
    if not isinstance(doc, dict):
        return out
    for client, projects in doc.items():
        if not isinstance(projects, dict):
            continue
        for project, entry in projects.items():
            if isinstance(entry, dict) and _in_range(entry.get("port")):
                out[(client, project)] = {"port": entry["port"], "pid": entry.get("pid")}
    return out


def _write(entries: dict[Key, dict]) -> None:
    doc: dict[str, dict] = {}
    for (client, project), entry in sorted(entries.items()):
        doc.setdefault(client, {})[project] = entry
    write_record(data_dir() / _RECORD, json.dumps(doc, indent=1).encode())


def _kept(key: Key, entry: dict) -> bool:
    """Whether an entry still keeps its port: its project has a private
    home, or the launch that took the port runs. A launch takes the port
    before it makes the home."""
    return private_home_path(*key).is_dir() or _alive(entry.get("pid"))


def _free(port: int) -> bool:
    """Whether launch can listen on the port, as the session does."""
    try:
        sock = relay.listen_socket(("127.0.0.1", port))
    except OSError:
        return False
    sock.close()
    return True


def recorded(client: str, project: str) -> int | None:
    """The port the record keeps for a client's project, or None."""
    entry = _read().get((client, project))
    return entry["port"] if entry else None


def choose(client: str, project: str, *, avoid=frozenset(),
           record: bool = True) -> tuple[int, int | None]:
    """The Mac port of the web app of ``client`` in ``project``, and the
    port the record kept before when the app moves from it.

    The recorded port is used when launch can listen on it. Otherwise the
    first port of the range that no other entry keeps and that launch can
    listen on is used. A port in ``avoid``, such as the gmlx server's own
    port, is never used. With ``record`` the port is recorded for the
    project, and without it nothing is written, as for the dry run. When no
    port is free, :class:`settings.Busy` is raised."""
    key = (client, project)
    with FileLock(data_dir() / _LOCK):
        entries = _read()
        kept = {k: e for k, e in entries.items() if k == key or _kept(k, e)}
        others = {e["port"] for k, e in kept.items() if k != key}
        before = kept[key]["port"] if key in kept else None

        def usable(port: int) -> bool:
            return port not in others and port not in avoid and _free(port)

        if before is not None and usable(before):
            port, moved = before, None
        else:
            port = next((p for p in range(FIRST, LAST + 1) if usable(p)), None)
            if port is None:
                raise Busy(_full_message(client, kept, key))
            moved = before
        if record:
            kept[key] = {"port": port, "pid": os.getpid()}
            if kept != entries:
                _write(kept)
        return port, moved


def _full_message(client: str, kept: dict[Key, dict], key: Key) -> str:
    owners = sorted({c for c, p in kept if (c, p) != key})
    message = (f"no Mac port from {FIRST} to {LAST} is free for the {client} web app, "
               "because other projects keep them or other programs use them.")
    if owners:
        message += (f" To free the port of a project you no longer need, run gmlx launch "
                    f"{owners[0]} --remove-home in that project's folder. gmlx doctor lists "
                    "the projects that have a private home.")
    else:
        message += " Stop a program that uses one of these ports, then launch again."
    return message


def release(client: str, project: str) -> int | None:
    """Remove the entry of a client's project, and return the port it
    kept, or None when there was none."""
    if not (data_path() / _RECORD).exists():
        return None
    with FileLock(data_dir() / _LOCK):
        entries = _read()
        entry = entries.pop((client, project), None)
        if entry is not None:
            _write(entries)
    return entry["port"] if entry else None
