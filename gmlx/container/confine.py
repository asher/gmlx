"""File access inside a private home that the guest can change.

In container mode the launch handlers run on the Mac with your rights, while
the guest can put anything in its private home, including symbolic links and
named pipes. A link planted at a config path, or at a folder above it, would
make a plain ``open`` or ``mkdir`` read or write a Mac file outside the private
home. Every access here walks the path one folder at a time with ``O_NOFOLLOW``
and folder descriptors, so no symbolic link is followed and no folder can be
swapped for a link between a check and the write.

:func:`confined` turns the checks on for one private home. Outside it, the
functions act on the Mac's own files: a write through a symbolic link whose
target stays inside your home folder goes to the target, and any other link
is refused.
"""

from __future__ import annotations

import contextlib
import errno
import os
import secrets
import stat
from pathlib import Path

from gmlx.safe_path import LeavesRoot, NotFollowed, open_dir_below

# The largest config file a handler reads.
READ_MAX = 16 << 20

_root: Path | None = None
_aliases: tuple[str, ...] = ()


class ConfinedError(RuntimeError):
    """A path leaves the private home or passes through a link or a file
    that is not what it should be. The message names the path."""


@contextlib.contextmanager
def confined(home: Path):
    """Confine every function in this module to ``home`` until the block
    ends."""
    global _root, _aliases
    saved = _root, _aliases
    real = Path(os.path.realpath(home))
    _root, _aliases = real, tuple(dict.fromkeys((str(real), os.path.abspath(home))))
    try:
        yield
    finally:
        _root, _aliases = saved


def active() -> bool:
    return _root is not None


def _parts(path: Path) -> list[str]:
    """The components of ``path`` below the private home."""
    p = os.path.abspath(os.path.expanduser(str(path)))
    for alias in _aliases:
        if p == alias:
            return []
        if p.startswith(alias.rstrip("/") + "/"):
            return [c for c in p[len(alias.rstrip("/")) + 1:].split("/") if c]
    raise ConfinedError(f"{p} is outside the private home {_root}.")


def _refuse_link(shown: str, e: OSError):
    if e.errno in (errno.ELOOP, errno.EMLINK, errno.ENOTDIR):
        raise ConfinedError(f"{shown} in the private home is a symbolic link or not a "
                            "folder, so launch will not follow it.") from None
    # Anything else the guest can put there, such as a socket, is refused
    # with its path rather than a traceback.
    raise ConfinedError(f"{shown} in the private home cannot be used "
                        f"({e.strerror or e}).") from None


def _open_dir(parts: list[str], *, create: bool) -> int:
    """A descriptor of the folder ``parts`` below the private home. Missing
    folders are created when ``create`` is set, else FileNotFoundError."""
    assert _root is not None
    try:
        return open_dir_below(_root, parts, create=create)
    except LeavesRoot as e:
        raise ConfinedError(f"{e.shown} leaves the private home.") from None
    except NotFollowed as e:
        _refuse_link(e.shown, e.error)
        raise


def _lstat_in(dir_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


# Mac-side paths, outside confinement

def _home_real() -> str:
    from .state import canonical

    return canonical(os.path.expanduser("~"))


def _host_target(path: Path) -> Path:
    """The file a write to ``path`` should replace. A symbolic link whose
    target stays inside your home folder is written through, so a dotfiles
    link keeps working. Any other link is refused."""
    if not os.path.islink(path):
        return path
    from .state import canonical, path_inside

    real = canonical(path)
    home = _home_real()
    if real == home or not path_inside(real, home):
        raise ConfinedError(f"{path} is a symbolic link to {real}, outside your home "
                            "folder, so launch will not replace it.")
    return Path(real)


def _in_private_home(p: str) -> bool:
    """Whether the absolute path ``p`` is or lies in a private home."""
    from .state import canonical, data_path, path_inside

    data = canonical(data_path())
    if p == data or not path_inside(p, data):
        return False
    # <client>/projects/<id>/home
    rest = [part.casefold() for part in p.split("/")[len(data.rstrip("/").split("/")):]]
    return len(rest) >= 4 and rest[1] == "projects" and rest[3] == "home"


def _refuse_unconfined(path) -> None:
    """Fail closed: a private home is read or written only inside
    :func:`confined`, so a new call site can never follow the guest's links
    by mistake."""
    from .state import canonical

    for p in {os.path.abspath(os.path.expanduser(str(path))),
              canonical(os.path.expanduser(str(path)))}:
        if _in_private_home(p):
            raise ConfinedError(f"{p} is in a private home, which launch reads "
                                "only with the links in it checked.")


def host_path(path) -> Path:
    """The file that ``path`` leads to on the Mac, resolved once, so a read
    or a write outside a private home acts on that one file. A client can
    change the links in a private home and in a folder that a session
    shared read-write. So a path is refused when resolving it passes
    through a private home, or through such a folder and then leads out of
    it."""
    from . import settings
    from .state import canonical, path_inside

    written = os.path.abspath(os.path.expanduser(str(path)))
    _refuse_unconfined(written)
    real = canonical(written)
    visited = settings._resolution_paths(written)
    for p in visited:
        if _in_private_home(p):
            raise ConfinedError(f"{path} leads through {p} in a private home, so launch "
                                "will not follow it.")
    for folder in settings.shared_history():
        if any(path_inside(p, folder) for p in visited) and not path_inside(real, folder):
            raise ConfinedError(f"{path} leads through {folder}, which a container session "
                                f"shared read-write, to {real} outside that folder. A client "
                                "may have left a link there, so launch will not follow it.")
    return Path(real)


def read_host_file(path) -> tuple[bytes, os.stat_result, Path] | None:
    """The bytes and the status of the Mac file that ``path`` leads to, and
    its path, as :func:`host_path` resolves it, or None when it does not
    exist. Only a regular file is read, never one that would block, such as
    a named pipe."""
    target = host_path(path)
    try:
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise ConfinedError(f"{path} changed into a symbolic link while launch read it, "
                                "so launch did not read it.") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ConfinedError(f"{path} is not a regular file, so launch did not read it.")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks), st, target


# The functions the handlers use

def exists(path: Path) -> bool:
    """Whether ``path`` exists. In the private home a symbolic link at the
    path, or at a folder above it, is refused."""
    if _root is None:
        return os.path.exists(host_path(path))
    parts = _parts(path)
    if not parts:
        return True
    try:
        fd = _open_dir(parts[:-1], create=False)
    except FileNotFoundError:
        return False
    try:
        st = _lstat_in(fd, parts[-1])
    finally:
        os.close(fd)
    if st is not None and stat.S_ISLNK(st.st_mode):
        raise ConfinedError(f"{path} in the private home is a symbolic link, so launch "
                            "will not follow it.")
    return st is not None


def read_text(path: Path) -> str | None:
    """The file's text, or None when it does not exist. In the private home
    only a regular file is read, never through a link, and never one that
    would block, such as a named pipe."""
    if _root is None:
        got = read_host_file(path)
        if got is None:
            return None
        # With the newlines of a file opened as text.
        return got[0].decode().replace("\r\n", "\n").replace("\r", "\n")
    parts = _parts(path)
    if not parts:
        raise ConfinedError(f"{path} is the private home, not a file.")
    try:
        dir_fd = _open_dir(parts[:-1], create=False)
    except FileNotFoundError:
        return None
    try:
        try:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=dir_fd)
        except FileNotFoundError:
            return None
        except OSError as e:
            _refuse_link(str(path), e)
    finally:
        os.close(dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ConfinedError(f"{path} in the private home is not a regular file.")
        if st.st_size > READ_MAX:
            raise ConfinedError(f"{path} in the private home is larger than "
                                f"{READ_MAX >> 20} MiB.")
        chunks, left = [], READ_MAX + 1
        while left > 0:
            chunk = os.read(fd, min(left, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
    finally:
        os.close(fd)
    try:
        return b"".join(chunks).decode()
    except UnicodeDecodeError:
        raise ConfinedError(f"{path} in the private home is not UTF-8 text.") from None


def mkdirs(path: Path) -> None:
    """Create ``path`` and the folders above it."""
    if _root is None:
        _refuse_unconfined(path)
        Path(path).mkdir(parents=True, exist_ok=True)
        return
    os.close(_open_dir(_parts(path), create=True))


def _new_mode(existing: os.stat_result | None, mode: int | None = None) -> int:
    if existing is not None:
        return stat.S_IMODE(existing.st_mode)
    if mode is not None:
        return mode
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def write_bytes(path: Path, data: bytes, mode: int | None = None) -> None:
    """Replace ``path`` with ``data`` through a new file in the same folder,
    so a crash never leaves it half written. The file keeps its mode, so a
    0600 file that holds keys stays 0600. ``mode`` sets the mode of a new
    file."""
    write_stream(path, lambda fd: write_all(fd, data), mode)


def write_stream(path: Path, fill, mode: int | None = None) -> None:
    """:func:`write_bytes` for data that ``fill(fd)`` writes to the new
    file in pieces, so a large file never has to fit in memory. When
    ``fill`` raises, the new file is removed and ``path`` is unchanged."""
    if _root is None:
        _host_target(Path(path))
        target = host_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = os.stat(target)
        except FileNotFoundError:
            existing = None
        tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     _new_mode(existing, mode))
        try:
            try:
                os.fchmod(fd, _new_mode(existing, mode))
                fill(fd)
            finally:
                os.close(fd)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return
    parts = _parts(path)
    if not parts:
        raise ConfinedError(f"{path} is the private home, not a file.")
    dir_fd = _open_dir(parts[:-1], create=True)
    name = parts[-1]
    try:
        existing = _lstat_in(dir_fd, name)
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ConfinedError(f"{path} in the private home is a symbolic link or not "
                                "a regular file, so launch will not replace it.")
        tmp = f".{name}.{secrets.token_hex(4)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     _new_mode(existing, mode), dir_fd=dir_fd)
        try:
            try:
                os.fchmod(fd, _new_mode(existing, mode))
                fill(fd)
            finally:
                os.close(fd)
            os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp, dir_fd=dir_fd)
            raise
    finally:
        os.close(dir_fd)


def write_text(path: Path, text: str, mode: int | None = None) -> None:
    write_bytes(path, text.encode(), mode)


def symlink(target: str, path: Path) -> None:
    """Create a symbolic link at ``path``, as a seeded folder may hold. The
    link is never followed here."""
    if _root is None:
        _refuse_unconfined(path)
        os.symlink(target, path)
        return
    parts = _parts(path)
    dir_fd = _open_dir(parts[:-1], create=True)
    try:
        os.symlink(target, parts[-1], dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def rename(path: Path, new_name: str) -> None:
    """Rename ``path`` to ``new_name`` in the same folder. An entry that
    already has the new name is never replaced."""
    if _root is None:
        raise ConfinedError(f"rename of {path} is only for a private home.")
    parts = _parts(path)
    if not parts or "/" in new_name or new_name in (".", ".."):
        raise ConfinedError(f"cannot rename {path} to {new_name}.")
    dir_fd = _open_dir(parts[:-1], create=False)
    try:
        if _lstat_in(dir_fd, new_name) is not None:
            raise ConfinedError(f"{new_name} already exists beside {path}.")
        os.rename(parts[-1], new_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def remove_tree(path: Path) -> None:
    """Delete ``path`` and everything below it, never through a link. A
    missing path is not an error."""
    if _root is None:
        raise ConfinedError(f"remove_tree of {path} is only for a private home.")
    parts = _parts(path)
    if not parts:
        raise ConfinedError(f"{path} is the private home itself.")
    try:
        dir_fd = _open_dir(parts[:-1], create=False)
    except FileNotFoundError:
        return
    try:
        _remove_in(dir_fd, parts[-1])
    finally:
        os.close(dir_fd)


def _remove_in(dir_fd: int, name: str) -> None:
    """Delete ``name`` in ``dir_fd`` and everything below it, never through
    a link. It walks with a list rather than recursion, so a tree nested
    deeper than Python's recursion limit is removed too."""
    st = _lstat_in(dir_fd, name)
    if st is None:
        return
    if not stat.S_ISDIR(st.st_mode):
        os.unlink(name, dir_fd=dir_fd)
        return
    # Each entry is the parent folder's descriptor, the name, the folder's
    # own descriptor and the names in it still to delete. Only the folders
    # on the current path are open at a time.
    def opened(parent: int, child: str) -> tuple[int, str, int, list[str]]:
        fd = os.open(child, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            return parent, child, fd, os.listdir(fd)
        except BaseException:
            os.close(fd)
            raise

    stack = [opened(dir_fd, name)]
    try:
        while stack:
            parent, here, fd, left = stack[-1]
            if not left:
                stack.pop()
                os.close(fd)
                os.rmdir(here, dir_fd=parent)
                continue
            child = left.pop()
            cst = _lstat_in(fd, child)
            if cst is None:
                continue
            if stat.S_ISDIR(cst.st_mode):
                stack.append(opened(fd, child))
            else:
                os.unlink(child, dir_fd=fd)
    finally:
        for _, _, fd, _ in reversed(stack):
            os.close(fd)


def tree_stamp(path: Path, limit: int) -> list[int] | None:
    """:func:`stamp_in` for ``path`` in the private home, with no link on
    the way to it followed."""
    if _root is None:
        raise ConfinedError(f"tree_stamp of {path} is only for a private home.")
    parts = _parts(path)
    if not parts:
        raise ConfinedError(f"{path} is the private home itself.")
    try:
        dir_fd = _open_dir(parts[:-1], create=False)
    except FileNotFoundError:
        return None
    try:
        return stamp_in(dir_fd, parts[-1], limit)
    finally:
        os.close(dir_fd)


def stamp_in(dir_fd: int, name: str, limit: int) -> list[int] | None:
    """The number of entries, the bytes of the files and the newest
    modification time in nanoseconds of ``name`` in ``dir_fd`` and
    everything below it, which a change to any file or folder in the tree
    changes. A link counts as itself and is never followed. None when
    ``name`` is missing or cannot be read, or when the tree holds more than
    ``limit`` entries."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        st = _lstat_in(dir_fd, name)
        if st is None:
            return None
        entries, size, newest = 1, 0, st.st_mtime_ns
        if not stat.S_ISDIR(st.st_mode):
            return [entries, st.st_size, newest]
        stack = [os.open(name, flags, dir_fd=dir_fd)]
        try:
            while stack:
                fd = stack[-1]
                names = os.listdir(fd)
                stack.pop()
                try:
                    for child in names:
                        cst = _lstat_in(fd, child)
                        if cst is None:
                            continue
                        entries += 1
                        if entries > limit:
                            return None
                        newest = max(newest, cst.st_mtime_ns)
                        if stat.S_ISDIR(cst.st_mode):
                            stack.append(os.open(child, flags, dir_fd=fd))
                        else:
                            size += cst.st_size
                finally:
                    os.close(fd)
        finally:
            for fd in stack:
                os.close(fd)
    except OSError:
        return None
    return [entries, size, newest]


def listdir(path: Path) -> list[str]:
    """The names in a folder, or [] when it does not exist. In the private
    home no link on the way is followed."""
    if _root is None:
        _refuse_unconfined(path)
        try:
            return os.listdir(path)
        except (FileNotFoundError, NotADirectoryError):
            return []
    try:
        fd = _open_dir(_parts(path), create=False)
    except FileNotFoundError:
        return []
    try:
        return os.listdir(fd)
    finally:
        os.close(fd)


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]
