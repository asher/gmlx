"""How the gmlx server finds the programs that it starts.

The server starts ffmpeg and ffprobe for audio, and the assistant starts
its MCP tool servers. The server finds such a program by name on its PATH.
That PATH can hold a folder that a container client writes, such as the
.venv/bin of a shared project, so the server never runs a program from
such a folder:

- :func:`search` skips each PATH entry whose path, as written or real,
  lies in a folder that a container session shares or shared read-write
  (launch's share history), or in the folder where launch keeps the
  private homes of the clients. It also skips each empty or relative
  entry, because such an entry names the folder that the server runs in.
  After the PATH, it searches the Homebrew and system folders that the
  PATH does not hold, such as /opt/homebrew/bin for a server that a login
  item starts.
- :func:`look_up` refuses the program that it finds when the real path of
  the file lies in such a folder, such as a link in ~/bin that leads into a
  shared project, or when a link on the way to the file does. A client can
  change such a link between the check and the start. It refuses a program
  of a Homebrew installation (/opt/homebrew or /usr/local) when a folder in
  that installation was shared read-write, such as /opt/homebrew/lib, since
  the program loads its libraries and settings from there. It refuses a
  program given by its path in the same way.

The checks compare canonical paths with :func:`gmlx.safe_path.path_inside`,
as launch does. A refusal that the share history causes names the command
that removes the folder from the history. They read the share history at each lookup, so a share of
a session that starts after the server counts before its client can write.
A tool server keeps the PATH that it gets at its start, so the assistant
checks that PATH and the program again with :func:`skipped_now` and
:func:`refusal` before each tool call.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from gmlx.safe_path import canonical, path_inside

_HOMES = "where launch keeps the private homes of the clients"
_SHARED = "a folder that a container session shared read-write"


class ProgramMissing(RuntimeError):
    """The server finds no program of that name."""


class ProgramRefused(RuntimeError):
    """The program lies in a folder that a container client can write."""


@dataclass(frozen=True)
class Search:
    """The folders that the server searches for a program by name."""
    folders: tuple[str, ...]            # in the order of the search
    skipped: tuple[tuple[str, str], ...]  # each entry left out, and why: "lies in ..."
    added: tuple[str, ...]              # the fixed folders the PATH does not hold


@dataclass(frozen=True)
class Lookup:
    """Where the server finds ``command``, and whether it runs it."""
    command: str
    path: str | None                    # the program that the search found
    refusal: str | None                 # why the server will not run ``path``
    search: Search
    history_step: str | None = None     # the step for a refusal that the share history causes


def _home() -> str:
    return canonical(os.path.expanduser("~"))


def tilde(path: str, home: str | None = None) -> str:
    """``path`` with the home folder written as ``~``."""
    home = home or _home()
    return "~" + path[len(home):] if path_inside(path, home) else path


def client_folders() -> list[tuple[str, str]]:
    """The folders that a container client can write, each with what it is:
    the private homes, and the folders that a container session shares or
    shared read-write. Launch records a share before the session starts,
    and the folder of the private homes of each launch, which another
    ``XDG_DATA_HOME`` moves. Raises
    :class:`gmlx.container.settings.HistoryDamaged` when the share history
    cannot be read."""
    from gmlx.container.settings import homes_history, shared_history
    from gmlx.container.state import data_path

    homes = dict.fromkeys([canonical(data_path()), *homes_history()])
    return [*((h, _HOMES) for h in homes), *((f, _SHARED) for f in shared_history())]


def _homes_only() -> list[tuple[str, str]]:
    """The folder of the private homes of this server's environment, for a
    search while the share history cannot be read."""
    from gmlx.container.state import data_path

    return [(canonical(data_path()), _HOMES)]


def forget_step(folder: str) -> str:
    """The sentence that tells how to remove ``folder`` from the share
    history."""
    from gmlx.container.settings import forget_step as step

    return step(folder, _home())


def _holders(paths: Sequence[str],
             folders: Sequence[tuple[str, str]]) -> list[tuple[str, str] | None]:
    """For each path, the first folder in ``folders`` that holds it, with
    what that folder is, or None. The outer loop runs over the folders, so
    the case check of each folder runs once."""
    out: list[tuple[str, str] | None] = [None] * len(paths)
    for folder, what in folders:
        for i, p in enumerate(paths):
            if out[i] is None and path_inside(p, folder):
                out[i] = (folder, what)
    return out


def _system_folders() -> list[str]:
    from gmlx.container.settings import SYSTEM_PATH

    return [f for f in SYSTEM_PATH.split(os.pathsep) if f]


def search(path: str | None = None,
           folders: Sequence[tuple[str, str]] | None = None) -> Search:
    """The folders that the server searches for a program by name: the
    entries of ``path`` (the PATH of this process when None), then the
    Homebrew and system folders that it does not hold, without each folder
    that a client can write and each empty or relative entry."""
    if folders is None:
        from gmlx.container.settings import HistoryDamaged
        try:
            folders = client_folders()
        except HistoryDamaged:
            # Every program is refused then (see why), so the search only
            # finds the program that the refusal names.
            folders = _homes_only()
    home = _home()
    entries = (os.environ.get("PATH", os.defpath) if path is None else path).split(os.pathsep)
    given = list(dict.fromkeys(entries))
    fixed = [f for f in _system_folders() if f not in given]
    absolute = [e for e in [*given, *fixed] if os.path.isabs(e)]
    reals = [canonical(e) for e in absolute]
    held = _holders([*absolute, *reals], folders)
    kept: list[str] = []
    seen: set[str] = set()
    skipped: list[tuple[str, str]] = []
    added: list[str] = []
    for entry in given:
        if not os.path.isabs(entry):
            skipped.append((entry, f"is {'relative' if entry else 'empty'}, so it names the "
                                   "folder that the server runs in"))
    for i, entry in enumerate(absolute):
        real = reals[i]
        hit, via = held[i], held[len(absolute) + i]
        # The checks come before the merge of entries with the same real
        # path. An entry in a share can be a link to an earlier entry now,
        # and a client can change it back later, so it is always skipped.
        if hit is not None:
            skipped.append((entry, f"lies in {tilde(hit[0], home)}, {hit[1]}"))
        elif via is not None:
            skipped.append((entry, f"leads to {tilde(real, home)}, in "
                                   f"{tilde(via[0], home)}, {via[1]}"))
        elif real not in seen:
            # Only a kept entry hides a later entry with the same real path.
            # A skipped link in a share can lead to a safe folder that a
            # later entry names.
            seen.add(real)
            kept.append(entry)
            if entry in fixed:
                added.append(entry)
    return Search(tuple(kept), tuple(skipped), tuple(added))


def skipped_now(entries: Sequence[str]) -> list[tuple[str, str]]:
    """Each folder of ``entries``, the folders of an earlier :func:`search`,
    that a search skips now, with why. A session that starts after that
    search can share such a folder."""
    found = search(os.pathsep.join(entries))
    return [(entry, why) for entry, why in found.skipped if entry in entries]


def _installations() -> list[str]:
    """The Homebrew installations that hold a folder of the fixed search,
    such as /opt/homebrew for /opt/homebrew/bin, as launch finds them."""
    from gmlx.container.settings import SEALED_PATH

    return list(dict.fromkeys(canonical(os.path.dirname(f)) for f in _system_folders()
                              if f not in SEALED_PATH and os.path.dirname(f) != "/"))


def why(path: str, folders: Sequence[tuple[str, str]] | None = None
        ) -> tuple[str, str | None] | None:
    """Why the server will not run the program at the absolute ``path``,
    as a phrase that follows the path, with the step for a refusal that
    the share history causes, or None. The path as written, the real path
    of the file or a path that resolving it visits lies in a folder that a
    client can write, or the program comes from a Homebrew installation
    that holds such a folder. While the share history cannot be read,
    every program is refused."""
    from gmlx.container.settings import (INSTALLATION_UNREAD, HistoryDamaged,
                                         _resolution_paths)

    if folders is None:
        try:
            folders = client_folders()
        except HistoryDamaged as e:
            return (f"may lie in a folder that a container session shared read-write, "
                    f"and {e.reason}", e.step)
    home = _home()
    real = canonical(path)
    hit, via = _holders([path, real], folders)
    if hit is not None:
        return f"lies in {tilde(hit[0], home)}, {hit[1]}", _history(hit)
    if via is not None:
        return (f"leads to {tilde(real, home)}, in {tilde(via[0], home)}, {via[1]}",
                _history(via))
    trail = [p for p in dict.fromkeys(_resolution_paths(path)) if p not in (path, real)]
    for folder, what in folders:
        hits = [p for p in trail if path_inside(p, folder)]
        if hits:
            link = next((p for p in hits if os.path.islink(p)), hits[0])
            return (f"leads to {tilde(real, home)} through {tilde(link, home)}, in "
                    f"{tilde(folder, home)}, {what}, and a client can change where that "
                    "link leads", _history((folder, what)))
    reached = [path, *trail, real]
    for inst in _installations():
        if not any(path_inside(p, inst) for p in reached):
            continue
        # A program reads no file in these folders of its installation,
        # unless it leads through one.
        unread = [u for u in (os.path.join(inst, sub) for sub in INSTALLATION_UNREAD)
                  if not any(path_inside(p, u) for p in reached)]
        for folder, what in folders:
            if path_inside(folder, inst) and not any(path_inside(folder, u) for u in unread):
                return (f"comes from the installation {tilde(inst, home)}, which holds "
                        f"{tilde(folder, home)}, {what}, and it loads its libraries and "
                        "settings from that installation", _history((folder, what)))
    return None


def _history(hit: tuple[str, str]) -> str | None:
    """The forget step for the folder of ``hit`` when the share history
    holds it, else None."""
    return forget_step(hit[0]) if hit[1] == _SHARED else None


def refusal(path: str, folders: Sequence[tuple[str, str]] | None = None) -> str | None:
    """Why the server will not run the program at the absolute ``path``,
    as a phrase that follows the path, or None (see :func:`why`)."""
    found = why(path, folders)
    return found[0] if found is not None else None


def look_up(command: str, path: str | None = None) -> Lookup:
    """Find the program that the server runs for ``command``. A command
    with a slash is a path, which the server takes from its working folder
    when it is relative. Any other command is a name, which the server
    looks for in :func:`search` of ``path``."""
    from gmlx.container.settings import HistoryDamaged
    try:
        folders: list[tuple[str, str]] | None = client_folders()
    except HistoryDamaged:
        # why() refuses every program then, and names the file.
        folders = None
    found = search(path, folders if folders is not None else _homes_only())
    if "/" in command:
        program: str | None = os.path.abspath(command)
    else:
        program = shutil.which(command, path=os.pathsep.join(found.folders))
    hit = why(program, folders) if program is not None else None
    return Lookup(command, program, hit[0] if hit else None, found, hit[1] if hit else None)


def problem(lookup: Lookup, step: str, who: str = "The gmlx server") -> str | None:
    """Why ``who`` cannot run the program of ``lookup``, with ``step``, the
    next step for the user, or None when it can run it. An empty ``step``
    gives the reason only."""
    home = _home()
    if lookup.path is not None and lookup.refusal is not None:
        return " ".join(filter(None, [
            f"{who} will not run {tilde(lookup.path, home)}, because it {lookup.refusal}.",
            "A container client could have changed what runs.", step,
            lookup.history_step if step else None]))
    if lookup.path is not None:
        return None
    where = "on its PATH"
    if lookup.search.added:
        where += " or in " + _phrase([tilde(f, home) for f in lookup.search.added])
    return " ".join(filter(None, [f"{who} finds no {lookup.command} {where}.",
                                  *skips_that_hold(lookup.search, lookup.command), step]))


def skipped_holders(found: Search, name: str) -> list[tuple[str, str]]:
    """Each PATH entry that the search skips and that holds a program
    ``name``, such as one that the user expects to run, as it is shown,
    with why the search skips it. A relative entry is read from the working
    folder, as a search of it would be. A path is not a name, so it gives
    none."""
    if "/" in name:
        return []
    home = _home()
    return [(tilde(entry, home) if entry else "the empty PATH entry", why)
            for entry, why in found.skipped if os.path.isfile(os.path.join(entry, name))]


def skips_that_hold(found: Search, name: str) -> list[str]:
    """A sentence for each PATH entry that the search skips and that holds
    a program ``name`` (see :func:`skipped_holders`)."""
    return [f"It does not look in {shown}, because that PATH entry {why}."
            for shown, why in skipped_holders(found, name)]


def _phrase(items: Sequence[str]) -> str:
    """Items as one phrase, such as "a, b or c"."""
    return f"{', '.join(items[:-1])} or {items[-1]}" if len(items) > 1 else items[0]


def checked(lookup: Lookup, step: str, who: str = "The gmlx server") -> Lookup:
    """``lookup`` when ``who`` can run its program. Raises
    :class:`ProgramMissing` or :class:`ProgramRefused`, with ``step`` in
    the message, when it cannot. Every program that the server starts goes
    through :func:`look_up` and this check right before the start."""
    text = problem(lookup, step, who)
    if text is not None:
        raise (ProgramRefused if lookup.path is not None else ProgramMissing)(text)
    return lookup


_logged: dict[tuple[str, str], str] = {}
_log_lock = threading.Lock()


def log(lookup: Lookup, step: str) -> None:
    """Write to the server log which program the server uses for the
    command of ``lookup``, and each PATH entry that the search skips, with
    the reason. A line is written once, and again when it changes."""
    home = _home()
    text = problem(lookup, step)
    shown = (f"none. {text}" if text is not None else tilde(lookup.path or "", home))
    lines = [(("program", lookup.command), f"[server] {lookup.command}: {shown}")]
    for entry, why in lookup.search.skipped:
        name = f"the PATH entry {tilde(entry, home)}" if entry else "the empty PATH entry"
        lines.append((("skip", entry),
                      f"[server] no program runs from {name}, because it {why}."))
    with _log_lock:
        for key, line in lines:
            if _logged.get(key) != line:
                _logged[key] = line
                print(line, file=sys.stderr, flush=True)
