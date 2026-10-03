"""Path checks that follow no symbolic link, for files a less trusted party
can change.

Launch uses these for a container's private home, and the server uses them
for the folder it reads request media from. :func:`open_dir_below` walks a
path one folder at a time with ``O_NOFOLLOW`` and folder descriptors, so no
symbolic link is followed and no folder can be swapped for a link between a
check and the use. :func:`path_inside` and :func:`canonical` compare paths in
the form macOS gives them. :func:`read_regular` reads a small regular file
with a limit on its size, and never waits on a named pipe.
"""

from __future__ import annotations

import errno
import fcntl
import functools
import json
import os
import stat
import unicodedata

# pathconf name of _PC_CASE_SENSITIVE on macOS, which Python does not list.
_PC_CASE_SENSITIVE = 11
# fcntl command that returns the path of an open file on macOS.
_F_GETPATH = getattr(fcntl, "F_GETPATH", 50)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class LeavesRoot(ValueError):
    """A path component is ``.`` or ``..``, or holds a slash."""

    def __init__(self, shown: str):
        super().__init__(f"{shown} leaves the folder it must stay in.")
        self.shown = shown


class NotFollowed(OSError):
    """A path component is a symbolic link, is not a folder, or cannot be
    opened. ``error`` is the error the open raised."""

    def __init__(self, shown: str, error: OSError):
        super().__init__(error.errno, f"{shown}: {error.strerror or error}")
        self.shown = shown
        self.error = error


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


def folded(path: str) -> str:
    """``path`` in the form that :func:`path_inside` compares first: in NFC,
    with its case folded. A path that is inside a folder by its exact names
    is also inside it by these forms, so a caller that checks many pairs
    can compare these forms first and skip a pair that does not match."""
    return _fold(path)


def same_name(name: str, other: str, folder: str) -> bool:
    """Whether ``name`` and ``other``, two names in ``folder``, name one
    entry of it. On a volume that ignores case, as APFS does, names that
    differ only in case name one entry. The volume that holds the entries
    of ``folder`` decides. When an entry is a link to another volume, the
    volume of its target does not decide."""
    return name == other or (_fold(name) == _fold(other) and _case_insensitive(folder))


def _within(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def path_inside(path: str, folder: str) -> bool:
    """True when ``path`` is ``folder`` or lies inside it, by whole path
    components. On a volume that ignores case, as APFS does, ``~/SRC`` and
    ``~/src`` are one folder, so the names are compared the same way.

    The folded names are compared first. A path that is inside the folder
    by its exact names is also inside by the folded names, so most pairs
    are answered without the question to the volume. A check against a
    long list of folders then never pushes the answers for the volumes out
    of their cache."""
    if folder == "/":
        return path.startswith("/")
    if not _within(_fold(path), _fold(folder)):
        return False
    return _case_insensitive(folder) or _within(path, folder)


def parts_below(path: str, folder: str) -> list[str] | None:
    """The components of ``path`` below ``folder``, or None when ``path``
    does not lie in it. Nothing is resolved, so ``..`` stays a component and
    :func:`open_dir_below` refuses it."""
    if not path_inside(path, folder):
        return None
    depth = len([c for c in folder.split("/") if c])
    return [c for c in path.split("/") if c][depth:]


def open_dir_below(root: str | os.PathLike, parts: list[str], *,
                   create: bool = False, mode: int = 0o755) -> int:
    """A descriptor of the folder ``parts`` below ``root``, opened one
    component at a time without following a link. Missing folders are
    created with ``mode`` when ``create`` is set, else FileNotFoundError.
    Raises :class:`LeavesRoot` for a ``.`` or ``..`` component and
    :class:`NotFollowed` for a link, a file or a folder that cannot be
    opened."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    shown = str(root)
    try:
        for part in parts:
            if part in ("", ".", "..") or "/" in part:
                raise LeavesRoot(f"{shown}/{part}")
            shown = f"{shown}/{part}"
            try:
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode, dir_fd=fd)
                except FileExistsError:
                    pass
                try:
                    nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
                except OSError as e:
                    raise NotFollowed(shown, e) from None
            except OSError as e:
                raise NotFollowed(shown, e) from None
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_file_below(root: str | os.PathLike, parts: list[str]) -> int:
    """A read-only descriptor of the file ``parts`` below ``root``, with no
    link followed on the way and none at the file itself. The open never
    blocks, so a named pipe cannot hold the caller. The caller checks what
    kind of file it got."""
    if not parts:
        raise LeavesRoot(str(root))
    dir_fd = open_dir_below(root, parts[:-1])
    shown = os.path.join(str(root), *parts)
    if parts[-1] in (".", "..") or "/" in parts[-1]:
        os.close(dir_fd)
        raise LeavesRoot(shown)
    try:
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                       dir_fd=dir_fd)
    except FileNotFoundError:
        raise
    except OSError as e:
        raise NotFollowed(shown, e) from None
    finally:
        os.close(dir_fd)


class NotRegular(OSError):
    """The path names a folder, a named pipe, a device or a socket, not a
    regular file."""

    def __init__(self):
        super().__init__(errno.EINVAL, "not a regular file")

    def __str__(self) -> str:
        return "not a regular file"


class TooLarge(OSError):
    """The file holds more bytes than the caller reads. ``limit`` is that
    number of bytes."""

    def __init__(self, limit: int):
        super().__init__(errno.EFBIG, f"larger than {_size(limit)}")
        self.limit = limit

    def __str__(self) -> str:
        return f"larger than {_size(self.limit)}"


def _size(n: int) -> str:
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if n >= 1 << shift and n % (1 << shift) == 0:
            return f"{n >> shift} {unit}"
    return f"{n} bytes"


def open_regular(path: str | os.PathLike, *, follow: bool = False,
                 dir_fd: int | None = None) -> tuple[int, os.stat_result]:
    """A read-only descriptor of the regular file at ``path``, and its
    status. The open never blocks, so a named pipe cannot hold the caller,
    and follows no link at the file itself unless ``follow`` is set.
    ``dir_fd`` is the folder that a relative ``path`` is in. Raises
    :class:`NotRegular` for anything but a regular file, and the error of
    the open otherwise."""
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | (0 if follow else os.O_NOFOLLOW)
    fd = os.open(path, flags, dir_fd=dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise NotRegular()
    except BaseException:
        os.close(fd)
        raise
    return fd, st


def read_fd(fd: int, limit: int, st: os.stat_result | None = None) -> bytes:
    """All bytes of the open regular file ``fd``, at most ``limit`` of them.
    Raises :class:`TooLarge` when the file holds more, by its size in ``st``
    or by what the read gets, so a file that grows during the read is
    refused too."""
    if (st or os.fstat(fd)).st_size > limit:
        raise TooLarge(limit)
    chunks, left = [], limit + 1
    while left > 0:
        chunk = os.read(fd, min(left, 1 << 20))
        if not chunk:
            break
        chunks.append(chunk)
        left -= len(chunk)
    if left <= 0:
        raise TooLarge(limit)
    return b"".join(chunks)


def read_regular(path: str | os.PathLike, limit: int, *, follow: bool = False,
                 dir_fd: int | None = None) -> bytes:
    """The bytes of the regular file at ``path``, at most ``limit`` of them,
    read as :func:`open_regular` opens it and :func:`read_fd` reads it."""
    fd, st = open_regular(path, follow=follow, dir_fd=dir_fd)
    try:
        return read_fd(fd, limit, st)
    finally:
        os.close(fd)


def read_json_object(path: str | os.PathLike, limit: int) -> dict | None:
    """The JSON object in the regular file at ``path``, read with
    :func:`read_regular`. None when the file is missing, is a link, is not
    a regular file, holds more than ``limit`` bytes, cannot be read, or
    does not hold a JSON object."""
    try:
        doc = json.loads(read_regular(path, limit).decode())
    except (OSError, ValueError, RecursionError):
        return None
    return doc if isinstance(doc, dict) else None
