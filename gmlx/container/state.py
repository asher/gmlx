"""Where container mode keeps its files, and the ``flock`` locks on them.

Persistent data (private homes, image records, volume and runtime locks)
lives under ``$XDG_DATA_HOME/gmlx/launch``. The share history lives under
``~/.local/share/gmlx/launch`` whatever ``XDG_DATA_HOME`` says, so that a
gmlx server with another environment reads the same file. Per-session
sockets and the relay log live under ``$XDG_CACHE_HOME/gmlx/launch``. Every lock is an advisory
``flock`` that the kernel releases when its holder dies, so a killed launch
never leaves a stale lock behind. Lock files are opened close-on-exec, so no
lock passes to a ``container`` child process.
"""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path
from typing import Callable

# Launch modules import these from here.
from gmlx.safe_path import canonical, fd_path, path_inside  # noqa: F401


# Container-mode data is private to you, whatever the umask.
ROOT_MODE = 0o700


def data_path() -> Path:
    """``$XDG_DATA_HOME/gmlx/launch``, without creating it."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "gmlx" / "launch"


def history_path() -> Path:
    """``~/.local/share/gmlx/launch``, the folder of the share history,
    without creating it. ``XDG_DATA_HOME`` does not move it: a server that
    a login item or another shell starts can have another value than the
    launch, and it must see each folder that launch shared read-write."""
    return Path(os.path.expanduser("~/.local/share")) / "gmlx" / "launch"


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
        raise NotADirectoryError(f"{d} is not a folder, and gmlx keeps the files of "
                                 "container mode there. Move or remove it.")
    if os.stat(d).st_mode & 0o777 != ROOT_MODE:
        os.chmod(d, ROOT_MODE)
    # The gmlx folder above it can come from other gmlx commands, made with
    # the umask, such as 0777 under umask 000. Keep it at 0700 too.
    parent = d.parent
    try:
        st = os.lstat(parent)
        if (parent.name == "gmlx" and stat.S_ISDIR(st.st_mode)
                and st.st_uid == os.getuid() and st.st_mode & 0o777 != ROOT_MODE):
            os.chmod(parent, ROOT_MODE)
    except OSError:
        pass
    return d


def data_dir() -> Path:
    """``$XDG_DATA_HOME/gmlx/launch``, created on first use, mode 0700."""
    return _private_root(data_path())


def history_dir() -> Path:
    """:func:`history_path`, created on first use, mode 0700."""
    return _private_root(history_path())


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
    d.mkdir(mode=ROOT_MODE, exist_ok=True)
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

    @classmethod
    def adopt(cls, path: Path, fd: int) -> "FileLock":
        """The lock that ``fd`` holds on ``path``, such as one that another
        process took and passed on. :meth:`release` closes ``fd``."""
        lock = cls.__new__(cls)
        lock.path, lock.fd, lock.inode = Path(path), fd, os.fstat(fd).st_ino
        return lock

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
