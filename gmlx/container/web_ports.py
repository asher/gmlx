"""The Mac ports of the browser apps of container mode.

A browser keeps the service workers, the storage and the HTTP cache of a
page by its origin, and the origin holds the port. A guest page can leave
all three at its address, and they stay after the session ends. So each
project of a browser app gets a Mac port of its own from :data:`FIRST` to
:data:`LAST`, and host mode never uses these ports. Cookies are kept by
host name only, so a port of its own does not keep them apart. The apps
answer at [::1], which is not the same site as 127.0.0.1 or localhost, so
the cookies of host-mode apps do not reach them.

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
import re
import shlex
import stat
from itertools import chain
from typing import NamedTuple

from . import relay
from .session import (WEB_HOST, _launch_alive, launch_owner, started_path,
                      try_session_lock)
from .settings import (PROJECT_DEFAULT, Busy, _tilde, private_home_path, project_dir_path,
                       read_project_record)
from .state import FileLock, data_dir, data_path, write_record

FIRST = 3100
LAST = 3199
_RECORD = "web-ports.json"
_LOCK = "web-ports.lock"
_RECORD_MAX = 1 << 20
# The most projects the message for a full range names.
_NAMED_MAX = 3
# A client or project name that the message can put in a path.
_PLAIN_NAME = re.compile(r"[A-Za-z0-9._-]+")

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


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _alive(entry: dict) -> bool:
    """Whether the launch that took the port of ``entry`` still runs. The
    entry names it by process ID and start time, as a session record does,
    so a later process with the same ID, or a process of another user, is
    not that launch."""
    pid = entry.get("pid")
    return _int(pid) and pid > 0 and _launch_alive(entry)


def _in_range(port) -> bool:
    return _int(port) and FIRST <= port <= LAST


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
                entries[(client, project)] = {
                    "port": entry["port"], "pid": entry.get("pid"),
                    **({"pid_start": entry["pid_start"]} if _int(entry.get("pid_start"))
                       else {})}
    listed = doc.get("served")
    for port, key in (listed.items() if isinstance(listed, dict) else ()):
        if (port.isascii() and port.isdigit() and _in_range(int(port))
                and isinstance(key, list) and len(key) == 2
                and all(isinstance(part, str) for part in key)):
            served[int(port)] = (key[0], key[1])
    return entries, served


def _with_started(entries: dict[Key, dict], served: dict[int, Key],
                  started: frozenset[Key] = frozenset()) -> dict[int, Key]:
    """The served ports, with the port of each entry whose project has
    started a session when the list does not have that port. So the list
    gets back what the projects show when it is damaged. A launch makes the
    private home before the session starts, so the home alone does not show
    that the port served pages. The started mark is for the project, not
    for the port, so it never replaces the project that the list names.
    ``started`` holds the projects that started, for a caller that removed
    their marks."""
    out = dict(served)
    for key, entry in entries.items():
        if entry["port"] not in out and (key in started or started_path(*key).exists()):
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
    return private_home_path(*key).is_dir() or _alive(entry)


def _free(port: int) -> bool:
    """Whether launch can listen on the port, as the session does."""
    try:
        sock = relay.listen_socket((WEB_HOST, port))
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
    on. It tries first the ports that a session of this project served, then
    the ports that no session served. Last come the ports that a session of
    another client or project served. A port in ``avoid``, such as the gmlx
    server's own port, is never used. With ``record`` the port is recorded
    for the project, and without it nothing is written, as for the dry run.
    When no port is free, :class:`settings.Busy` is raised."""
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
            kept[key] = {"port": port, **launch_owner()}
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


def _remove_step(key: Key, entry: dict) -> tuple[float, str] | None:
    """When the project of ``key`` was last used, and the command that
    removes its private home, or None when its project record does not name
    the project's folder. The command keys the project whatever
    launch.container.mount_cwd says. --mount . keys the current folder.
    --no-mount-cwd keys the default project only in a folder that no share
    holds, and launch never shares /, because it holds the private homes.
    Launch finds a project by its folder, so for a folder that no longer
    exists the command is rm -rf of the project's folder in the launch data.
    That command does not wait for the session to end, so it is not given
    while the launch of ``entry`` runs or a session of the project runs. A
    folder that launch cannot look at, or one on a volume that is not
    mounted, can still exist, so it gets no command. The record lies beside
    the home, outside the guest's shares."""
    client, project = key
    doc = read_project_record(client, project)
    used, folder = doc.get("used"), doc.get("folder")
    when = float(used) if isinstance(used, (int, float)) and not isinstance(used, bool) else 0
    if project == PROJECT_DEFAULT:
        where = "" if client == "open-webui" else " --no-mount-cwd in /"
        return when, f"gmlx launch {client} --remove-home{where}"
    if not isinstance(folder, str) or not folder:
        return None
    try:
        is_dir = stat.S_ISDIR(os.stat(folder).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        is_dir = False
    except OSError:
        return None                       # it can exist where launch cannot look
    if is_dir:
        return when, f"gmlx launch {client} --remove-home --mount . in {_tilde(folder)}"
    if _unmounted(folder) or _alive(entry):
        return None
    step = _rm_step(client, project)
    if step is None or _session_runs(client, project):
        return None
    return when, step


def _unmounted(folder: str) -> bool:
    """Whether ``folder`` is on a volume that is not mounted now, so it can
    come back when the volume is mounted again."""
    parts = folder.split("/")
    return len(parts) > 2 and parts[1] == "Volumes" and not os.path.isdir(
        os.path.join("/Volumes", parts[2]))


def _session_runs(client: str, project: str) -> bool:
    """Whether a session of a client's project runs, that is, holds the
    project's session lock. Each launch holds that lock, but only a launch
    of a web app records itself in the entry of the project's port, so a
    launch of the dsh headless profile is not in the entry. A lock that
    launch cannot open counts as held."""
    try:
        lock = try_session_lock(client, project)
    except OSError:
        return True
    if lock is None:
        return True
    lock.release()
    return False


def _rm_step(client: str, project: str) -> str | None:
    """The rm -rf command for the folder of a client's project in the launch
    data, or None when a name from the record is not a plain folder name or
    the folder is a link."""
    if not all(_PLAIN_NAME.fullmatch(name) and name.strip(".") for name in (client, project)):
        return None
    target = project_dir_path(client, project)
    if target.is_symlink():
        return None
    shown = _tilde(str(target))
    if shown.startswith("~/"):
        return f"rm -rf ~/{shlex.quote(shown[2:])}"
    return f"rm -rf {shlex.quote(shown)}"


def _full_message(client: str, kept: dict[Key, dict], key: Key) -> str:
    """The refusal for a range with no free port. It names the commands that
    remove the private homes of the projects used longest ago, since each
    home keeps its project's port."""
    message = (f"no Mac port from {FIRST} to {LAST} is free for the {client} web app, "
               "because other projects keep them or other programs use them.")
    homes = [k for k in kept if k != key and private_home_path(*k).is_dir()]
    steps = sorted(step for step in (_remove_step(k, kept[k]) for k in homes)
                   if step is not None)
    if steps:
        named = [command for _, command in steps[:_NAMED_MAX]]
        listed = named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"
        which = "project" if len(named) == 1 else "projects"
        message += (" To free the port of a project you no longer need, remove its private "
                    f"home. For the {which} used longest ago, run {listed}.")
        removals = sum(command.startswith("rm ") for command in named)
        if removals:
            each = "The rm -rf step" if removals == 1 else "Each rm -rf step"
            message += (f" {each} removes the home of a project whose folder no longer "
                        "exists, because launch finds a project by its folder.")
    elif homes:
        # _remove_step names a step for each default project, so these
        # homes are of folder projects.
        run = " or ".join(f"gmlx launch {c} --remove-home --mount ."
                          for c in sorted({c for c, _ in homes}))
        message += (f" To free the port of a project you no longer need, run {run} in its "
                    "folder. gmlx doctor lists the projects that have a private home.")
    else:
        message += " Stop a program that uses one of these ports, then launch again."
    return message


def release(client: str, project: str, *, started: bool = False,
            unless_running: bool = False) -> list[int]:
    """Remove the entry of a client's project, and return the ports that
    sessions of the project served last, lowest first. The port of the
    entry counts as served only when a session of the project started: the
    project has the start mark, or ``started`` says so for a caller that
    removed the mark. A launch that stopped before its session started
    served no pages. The ports stay in the list of served ports, so another
    project takes them only after every other port. With ``unless_running``
    the entry stays while the launch that took its port runs, for a caller
    that does not hold the project's session lock."""
    key = (client, project)
    if not (data_path() / _RECORD).exists():
        return []
    with FileLock(data_dir() / _LOCK):
        entries, listed = _read()
        served = _with_started(entries, listed, frozenset([key]) if started else frozenset())
        entry = entries.get(key)
        if entry is not None and not (unless_running and _alive(entry)):
            del entries[key]
            _write(entries, served)
    return sorted(p for p, k in served.items() if k == key)
