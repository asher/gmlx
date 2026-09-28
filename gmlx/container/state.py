"""Where container mode keeps its files, and the ``flock`` locks on them.

Persistent data (private homes, image records, volume and runtime locks)
lives under ``$XDG_DATA_HOME/gmlx/launch``. Per-session sockets and the relay
log live under ``$XDG_CACHE_HOME/gmlx/launch``. Every lock is an advisory
``flock`` that the kernel releases when its holder dies, so a killed launch
never leaves a stale lock behind. Lock files are opened close-on-exec, so no
lock passes to a ``container`` child process.
"""

from __future__ import annotations

import fcntl
import functools
import os
import unicodedata
from pathlib import Path
from typing import Callable


# pathconf name of _PC_CASE_SENSITIVE on macOS, which Python does not list.
_PC_CASE_SENSITIVE = 11


@functools.lru_cache(maxsize=256)
def _case_insensitive(folder: str) -> bool:
    """Whether the volume that holds ``folder`` compares names without
    case, as APFS does by default."""
    p = folder
    while p != "/" and not os.path.exists(p):
        p = os.path.dirname(p)
    try:
        return os.pathconf(p, _PC_CASE_SENSITIVE) == 0
    except (OSError, ValueError):
        return False


def _fold(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def path_inside(path: str, folder: str) -> bool:
    """True when ``path`` is ``folder`` or lies inside it, by whole path
    components. On a volume that ignores case, as APFS does, ``~/SRC`` and
    ``~/src`` are one folder, so the names are compared the same way."""
    if folder == "/":
        return path.startswith("/")
    if _case_insensitive(folder):
        path, folder = _fold(path), _fold(folder)
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def data_path() -> Path:
    """``$XDG_DATA_HOME/gmlx/launch``, without creating it."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "gmlx" / "launch"


def data_dir() -> Path:
    """``$XDG_DATA_HOME/gmlx/launch``, created on first use."""
    d = data_path()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """``$XDG_CACHE_HOME/gmlx/launch``, created on first use."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = Path(base) / "gmlx" / "launch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def images_dir() -> Path:
    d = data_dir() / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


class LockHeld(RuntimeError):
    """A non-blocking lock attempt found the lock held by another process."""


class FileLock:
    """One ``flock`` on one file, held until :meth:`release` or process exit.

    ``shared`` takes a shared lock, else an exclusive one. With
    ``blocking=False`` a held lock raises :class:`LockHeld`. With
    ``blocking=True``, ``on_wait`` runs once before the call starts waiting,
    so the caller can say why it stalls.
    """

    def __init__(self, path: Path, *, shared: bool = False,
                 blocking: bool = True,
                 on_wait: Callable[[], None] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
            except BlockingIOError:
                if not blocking:
                    raise LockHeld(str(self.path)) from None
                if on_wait is not None:
                    on_wait()
                fcntl.flock(fd, mode)
        except BaseException:
            os.close(fd)
            raise
        self.fd: int | None = fd
        self.inode = os.fstat(fd).st_ino

    def still_current(self) -> bool:
        """True while the locked file is still the file at :attr:`path`. A
        cleanup that deleted the path after the open makes this False."""
        try:
            return os.stat(self.path).st_ino == self.inode
        except FileNotFoundError:
            return False

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)             # closing the last descriptor unlocks
            self.fd = None

    def __enter__(self) -> "FileLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
