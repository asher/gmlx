"""The Mac ports of the browser apps of container mode.

A browser keeps the service workers, the storage and the HTTP cache of a
page by its origin, and the origin holds the port. A guest page can leave
all three at its address, and they stay after the session ends. So each
project of a browser app gets a Mac port of its own from :data:`FIRST` to
:data:`LAST`, and host mode never uses these ports. Cookies are kept by
host name only, so a port of its own does not keep them apart.

A record in the launch data folder keeps the port of each client and
project, so a project keeps its address from one launch to the next. A lock
keeps two launches that start at the same time from taking one port. An
entry stays while its project has a private home, or while the launch that
took the port runs.

The record also keeps each port that a session served, with the client and
project that last used it. That list stays when the project moves to
another port, when its entry goes and when its home is removed. A launch
takes a port that no session served before a port that a session of another
project served. It takes such a port only when no other port is free, and
then launch tells you to clear the site data of that address. Removing the
launch data folder removes the list too.
"""

from __future__ import annotations

import json
import os
import stat
from itertools import chain
from typing import NamedTuple

from . import relay
from .session import started_path
from .settings import Busy, private_home_path
from .state import FileLock, data_dir, data_path, write_record

FIRST = 3100
LAST = 3199
_RECORD = "web-ports.json"
_LOCK = "web-ports.lock"
_RECORD_MAX = 1 << 20

Key = tuple[str, str]


class Choice(NamedTuple):
    """The port a launch takes for the web app of one project."""

    port: int
    # The port the record kept for the project before, or None.
    before: int | None
    # Whether a session of another client or project served this port last.
    reused: bool

    @property
    def moved(self) -> bool:
        return self.before is not None and self.before != self.port


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


def _read() -> tuple[dict[Key, dict], dict[int, Key]]:
    """The entries of the record by client and project, and the ports that
    sessions served with the client and project that last used each. A
    missing or damaged record has none, and an item that is not in the
    correct form is left out."""
    try:
        fd = os.open(data_path() / _RECORD, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return {}, {}
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > _RECORD_MAX:
            return {}, {}
        doc = json.loads(os.read(fd, _RECORD_MAX).decode())
    except (OSError, ValueError, RecursionError):
        return {}, {}
    finally:
        os.close(fd)
    entries: dict[Key, dict] = {}
    served: dict[int, Key] = {}
    if not isinstance(doc, dict):
        return entries, served
    projects = doc.get("projects")
    for client, by_project in (projects.items() if isinstance(projects, dict) else ()):
        if not isinstance(by_project, dict):
            continue
        for project, entry in by_project.items():
            if isinstance(entry, dict) and _in_range(entry.get("port")):
                entries[(client, project)] = {"port": entry["port"], "pid": entry.get("pid")}
    listed = doc.get("served")
    for port, key in (listed.items() if isinstance(listed, dict) else ()):
        if (port.isascii() and port.isdigit() and _in_range(int(port))
                and isinstance(key, list) and len(key) == 2
                and all(isinstance(part, str) for part in key)):
            served[int(port)] = (key[0], key[1])
    return entries, served


def _with_started(entries: dict[Key, dict], served: dict[int, Key]) -> dict[int, Key]:
    """The served ports, with the port of each entry whose project has
    started a session when the list does not have that port. So the list
    gets back what the projects show when it is damaged. A launch makes the
    private home before the session starts, so the home alone does not show
    that the port served pages. The started mark is for the project, not
    for the port, so it never replaces the project that the list names."""
    out = dict(served)
    for key, entry in entries.items():
        if entry["port"] not in out and started_path(*key).exists():
            out[entry["port"]] = key
    return out


def _write(entries: dict[Key, dict], served: dict[int, Key]) -> None:
    projects: dict[str, dict] = {}
    for (client, project), entry in sorted(entries.items()):
        projects.setdefault(client, {})[project] = entry
    doc = {"projects": projects,
           "served": {str(port): list(key) for port, key in sorted(served.items())}}
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
    entry = _read()[0].get((client, project))
    return entry["port"] if entry else None


def choose(client: str, project: str, *, avoid=frozenset(), record: bool = True) -> Choice:
    """The Mac port of the web app of ``client`` in ``project``.

    The recorded port is used when launch can listen on it. Otherwise launch
    takes the first port that no other entry keeps and that it can listen
    on, in this order: a port that a session of this project served, a port
    that no session served, and a port that a session of another client or
    project served. A port in ``avoid``, such as the gmlx server's own port,
    is never used. With ``record`` the port is recorded for the project, and
    without it nothing is written, as for the dry run. When no port is free,
    :class:`settings.Busy` is raised."""
    key = (client, project)
    with FileLock(data_dir() / _LOCK):
        entries, listed = _read()
        served = _with_started(entries, listed)
        kept = {k: e for k, e in entries.items() if k == key or _kept(k, e)}
        others = {e["port"] for k, e in kept.items() if k != key}
        before = kept[key]["port"] if key in kept else None

        def usable(port: int) -> bool:
            return port not in others and port not in avoid and _free(port)

        ports = range(FIRST, LAST + 1)
        order = chain([] if before is None else [before],
                      (p for p in ports if served.get(p) == key),
                      (p for p in ports if p not in served),
                      (p for p in ports if p in served and served[p] != key))
        port = next((p for p in order if usable(p)), None)
        if port is None:
            raise Busy(_full_message(client, kept, key))
        if record:
            kept[key] = {"port": port, "pid": os.getpid()}
            if kept != entries or served != listed:
                _write(kept, served)
        return Choice(port, before, served.get(port, key) != key)


def mark_served(client: str, project: str, port: int) -> None:
    """Record that a session of ``client`` in ``project`` serves ``port``.
    A launch calls this when its session starts."""
    if not _in_range(port):
        return
    with FileLock(data_dir() / _LOCK):
        entries, listed = _read()
        served = _with_started(entries, listed)
        served[port] = (client, project)
        if served != listed:
            _write(entries, served)


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


def release(client: str, project: str, *, unless_running: bool = False) -> list[int]:
    """Remove the entry of a client's project, and return the ports that
    sessions of the project served last, lowest first. The port of the
    entry counts as served. The ports stay in the list of served ports, so
    another project takes them only after every other port. With
    ``unless_running`` the entry stays while the launch that took its port
    runs, for a caller that does not hold the project's session lock."""
    key = (client, project)
    if not (data_path() / _RECORD).exists():
        return []
    with FileLock(data_dir() / _LOCK):
        entries, listed = _read()
        served = _with_started(entries, listed)
        entry = entries.get(key)
        if entry is not None and not (unless_running and _alive(entry.get("pid"))):
            del entries[key]
            served[entry["port"]] = key
            _write(entries, served)
    return sorted(p for p, k in served.items() if k == key)
