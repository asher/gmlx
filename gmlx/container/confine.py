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
    fd = os.open(_root, os.O_RDONLY | os.O_DIRECTORY)
    shown = str(_root)
    try:
        for part in parts:
            if part in (".", ".."):
                raise ConfinedError(f"{shown}/{part} leaves the private home.")
            shown = f"{shown}/{part}"
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            try:
                nxt = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
                try:
                    nxt = os.open(part, flags, dir_fd=fd)
                except OSError as e:
                    _refuse_link(shown, e)
            except OSError as e:
                _refuse_link(shown, e)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _lstat_in(dir_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


# Mac-side paths, outside confinement

def _home_real() -> str:
    return os.path.realpath(os.path.expanduser("~"))


def _host_target(path: Path) -> Path:
    """The file a write to ``path`` should replace. A symbolic link whose
    target stays inside your home folder is written through, so a dotfiles
    link keeps working. Any other link is refused."""
    if not os.path.islink(path):
        return path
    real = os.path.realpath(path)
    home = _home_real()
    if real == home or not real.startswith(home.rstrip("/") + "/"):
        raise ConfinedError(f"{path} is a symbolic link to {real}, outside your home "
                            "folder, so launch will not replace it.")
    return Path(real)


def _refuse_unconfined(path) -> None:
    """Fail closed: a private home is read or written only inside
    :func:`confined`, so a new call site can never follow the guest's links
    by mistake."""
    from .state import data_path

    data = os.path.realpath(data_path())
    for p in {os.path.abspath(os.path.expanduser(str(path))),
              os.path.realpath(os.path.expanduser(str(path)))}:
        if p.startswith(data.rstrip("/") + "/"):
            rest = p[len(data.rstrip("/")) + 1:].split("/")
            if len(rest) >= 2 and rest[1] == "home":
                raise ConfinedError(f"{p} is in a private home, which launch reads "
                                    "only with the links in it checked.")


# The functions the handlers use

def exists(path: Path) -> bool:
    """Whether ``path`` exists. In the private home a symbolic link at the
    path, or at a folder above it, is refused."""
    if _root is None:
        _refuse_unconfined(path)
        return os.path.exists(path)
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
        _refuse_unconfined(path)
        try:
            return Path(path).read_text()
        except FileNotFoundError:
            return None
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
    if _root is None:
        _refuse_unconfined(path)
        target = _host_target(Path(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = os.stat(target)
        except FileNotFoundError:
            existing = None
        tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     _new_mode(existing, mode))
        try:
            os.fchmod(fd, _new_mode(existing, mode))
            _write_all(fd, data)
        finally:
            os.close(fd)
        try:
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
            os.fchmod(fd, _new_mode(existing, mode))
            _write_all(fd, data)
        finally:
            os.close(fd)
        try:
            os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp, dir_fd=dir_fd)
            raise
    finally:
        os.close(dir_fd)


def write_text(path: Path, text: str) -> None:
    write_bytes(path, text.encode())


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
    st = _lstat_in(dir_fd, name)
    if st is None:
        return
    if not stat.S_ISDIR(st.st_mode):
        os.unlink(name, dir_fd=dir_fd)
        return
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
    try:
        for child in os.listdir(fd):
            _remove_in(fd, child)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=dir_fd)


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


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]
