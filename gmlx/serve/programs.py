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
  shared project. It refuses a program given by its path in the same way.

The checks compare canonical paths with :func:`gmlx.safe_path.path_inside`,
as launch does. They read the share history at each lookup, so a share of
a session that starts after the server counts before its client can write.
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


def _home() -> str:
    return canonical(os.path.expanduser("~"))


def tilde(path: str, home: str | None = None) -> str:
    """``path`` with the home folder written as ``~``."""
    home = home or _home()
    return "~" + path[len(home):] if path_inside(path, home) else path


def client_folders() -> list[tuple[str, str]]:
    """The folders that a container client can write, each with what it is:
    the private homes, and the folders that a container session shares or
    shared read-write. Launch records a share before the session starts."""
    from gmlx.container.settings import shared_history
    from gmlx.container.state import data_path

    return [(canonical(data_path()), _HOMES), *((f, _SHARED) for f in shared_history())]


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
    folders = client_folders() if folders is None else folders
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
        if real in seen:
            continue
        seen.add(real)
        hit, via = held[i], held[len(absolute) + i]
        if hit is not None:
            skipped.append((entry, f"lies in {tilde(hit[0], home)}, {hit[1]}"))
        elif via is not None:
            skipped.append((entry, f"leads to {tilde(real, home)}, in "
                                   f"{tilde(via[0], home)}, {via[1]}"))
        else:
            kept.append(entry)
            if entry in fixed:
                added.append(entry)
    return Search(tuple(kept), tuple(skipped), tuple(added))


def refusal(path: str, folders: Sequence[tuple[str, str]] | None = None) -> str | None:
    """Why the server will not run the program at the absolute ``path``,
    as a phrase that follows the path, or None. The path as written or
    the real path of the file lies in a folder that a client can write."""
    folders = client_folders() if folders is None else folders
    home = _home()
    real = canonical(path)
    hit, via = _holders([path, real], folders)
    if hit is not None:
        return f"lies in {tilde(hit[0], home)}, {hit[1]}"
    if via is not None:
        return f"leads to {tilde(real, home)}, in {tilde(via[0], home)}, {via[1]}"
    return None


def look_up(command: str, path: str | None = None) -> Lookup:
    """Find the program that the server runs for ``command``. A command
    with a slash is a path, which the server takes from its working folder
    when it is relative. Any other command is a name, which the server
    looks for in :func:`search` of ``path``."""
    folders = client_folders()
    found = search(path, folders)
    if "/" in command:
        program: str | None = os.path.abspath(command)
    else:
        program = shutil.which(command, path=os.pathsep.join(found.folders))
    why = refusal(program, folders) if program is not None else None
    return Lookup(command, program, why, found)


def problem(lookup: Lookup, step: str) -> str | None:
    """Why the server cannot run the program of ``lookup``, with ``step``,
    the next step for the user, or None when it can run it."""
    home = _home()
    if lookup.path is not None and lookup.refusal is not None:
        return (f"The gmlx server will not run {tilde(lookup.path, home)}, because it "
                f"{lookup.refusal}. A container client could have written that file. {step}")
    if lookup.path is not None:
        return None
    where = "on its PATH"
    if lookup.search.added:
        where += " or in " + _phrase([tilde(f, home) for f in lookup.search.added])
    text = f"The gmlx server finds no {lookup.command} {where}."
    for entry, why in lookup.search.skipped:
        if os.path.isabs(entry) and os.path.isfile(os.path.join(entry, lookup.command)):
            text += f" It does not look in {tilde(entry, home)}, because that PATH entry {why}."
    return f"{text} {step}"


def _phrase(items: Sequence[str]) -> str:
    """Items as one phrase, such as "a, b or c"."""
    return f"{', '.join(items[:-1])} or {items[-1]}" if len(items) > 1 else items[0]


def resolve(command: str, step: str, path: str | None = None) -> Lookup:
    """:func:`look_up` for a program that the server starts now, through
    :func:`checked`. Every program that the server starts goes through
    this check."""
    return checked(look_up(command, path), step)


def checked(lookup: Lookup, step: str) -> Lookup:
    """``lookup`` when the server can run its program. Raises
    :class:`ProgramMissing` or :class:`ProgramRefused`, with ``step`` in
    the message, when it cannot."""
    text = problem(lookup, step)
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
