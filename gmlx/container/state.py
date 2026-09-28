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
# fcntl command that returns the path of an open file on macOS.
_F_GETPATH = getattr(fcntl, "F_GETPATH", 50)
# Container-mode data is private to you, whatever the umask.
ROOT_MODE = 0o700


def fd_path(fd: int) -> str:
    """The path macOS itself gives an open file or folder."""
    raw = fcntl.fcntl(fd, _F_GETPATH, b"\0" * 1024)
    return raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")


def canonical(path: str | os.PathLike) -> str:
    """``path`` with its links resolved and in the form macOS gives it.

    ``os.path.realpath`` keeps a firmlink alias such as
    ``/System/Volumes/Data/Users/you`` and the case the name was typed in,
    so a check by path would miss that it is ``/Users/you``. The folder is
    opened and macOS names it. A path that is not a folder is named through
    its folder."""
    real = os.path.realpath(os.path.expanduser(str(path)))
    try:
        fd = os.open(real, os.O_RDONLY | os.O_DIRECTORY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        parent, name = os.path.split(real)
        if not name or parent == real:
            return real
        return os.path.join(canonical(parent), name)
    try:
        return fd_path(fd) or real
    except OSError:
        return real
    finally:
        os.close(fd)


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


def _private_root(d: Path) -> Path:
    """Create ``d`` and any missing folder above it with mode 0700, and
    keep ``d`` itself at 0700, so no other local user can reach anything in
    it whatever the umask."""
    missing = []
    p = d
    while not p.exists() and p != p.parent:
        missing.append(p)
        p = p.parent
    for folder in reversed(missing):
        try:
            os.mkdir(folder, ROOT_MODE)
        except FileExistsError:
            pass
    if not d.is_dir() or d.is_symlink():
        raise NotADirectoryError(f"{d} is not a folder.")
    if os.stat(d).st_mode & 0o777 != ROOT_MODE:
        os.chmod(d, ROOT_MODE)
    return d


def data_dir() -> Path:
    """``$XDG_DATA_HOME/gmlx/launch``, created on first use, mode 0700."""
    return _private_root(data_path())


def cache_dir() -> Path:
    """``$XDG_CACHE_HOME/gmlx/launch``, created on first use, mode 0700."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return _private_root(Path(base) / "gmlx" / "launch")


def write_record(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Replace ``path`` with ``data`` through a new file that no link can
    redirect, so a crash never leaves it half written."""
    import secrets

    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                 mode)
    try:
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
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
