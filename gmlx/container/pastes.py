"""Files that you paste or drag into an interactive container session.

A Mac terminal pastes a file that you drag onto it, or a file that you copy
in Finder and paste, as the text of its path. The client in the container
sees a Mac path only when a share holds it at the same path. The terminal
relay hands each bracketed paste to :meth:`Pastes.rewrite`. When the paste
holds nothing but file paths, each file is copied into the private home,
and the path of the copy takes the place of the Mac path. The private home
has the same path on the Mac and in the container, so every client finds
the copy as it would find the file on the Mac.

A paste counts only when it is a list of paths, as a drag or a file paste
gives: words separated by spaces, tabs or line ends, each one a full path,
a path that starts with ``~/`` or a ``file://`` URL. A word can have
backslash escapes and single or double quotes, as iTerm2, Terminal.app and
other terminals write them. A path inside other text stays text, so a
pasted log that names a file copies nothing.

Only a regular file is placed, opened with no link followed. Launch first
clones the file, which takes no space and no time when the file and the
private home are on one volume. When the clone fails for any reason, launch
copies the file, up to the size limit that the session gives. When that
fails too, the path stays as it is. A file in a folder that launch refuses
to share, because it holds credentials, gmlx's own data or settings that
the Mac runs, stays as it is, as does a path in a folder that the container
sees at the same path. The session log records each file placed and the
reason for each path that stays.

A file goes to ``.gmlx/pastes/<key>/<name>`` in the private home, where
``<key>`` is 16 hex digits of a SHA-256 over the file's device, inode, size
and modification time, and ``<name>`` is the file's own name, with any
control character in it replaced. A second paste of an unchanged file uses
the same folder. The folder keeps the newest :data:`KEEP` entries.

No error in a paste reaches the terminal relay. A paste that fails in any
way stays as it came, and the session log records why.

The new path keeps the form that the terminal gave the old one: quotes of
the same kind, the same backslash escapes, or a ``file://`` URL.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import unicodedata
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gmlx.safe_path import LeavesRoot, NotFollowed, TooLarge, open_file_below, path_inside

from . import confine

# The folder of the pasted files, below the private home.
PASTE_FOLDER = (".gmlx", "pastes")
# The number of entries that the folder keeps.
KEEP = 50
# The default limit of a copy, as launch.container.paste_copy_max sets it.
COPY_MAX = 1 << 30
# The most files that one paste places. The paths after these stay as they are.
PATHS_MAX = 20
COPY_CHUNK = 1 << 20

_SPACE = " \t\r\n"
# Characters that always get a backslash in an unquoted path, since a client
# would read the path as two words or as quoted otherwise.
_ALWAYS_ESCAPED = " \t\n\\'\""
# The characters that a backslash keeps in double quotes, as in a shell.
_DOUBLE_ESCAPED = '\\"$`'

Log = Callable[[str], None]


class Refused(Exception):
    """A pasted path that is not placed. The message says why."""


@dataclass
class Word:
    """One word of a paste: where it starts and ends in the text, the path
    it names, and how the terminal wrote it. ``form`` is ``plain``,
    ``single``, ``double`` or ``url``, and ``escaped`` holds the characters
    that had a backslash before them."""
    start: int
    end: int
    path: str
    form: str
    escaped: str = ""


def _read_word(text: str, i: int) -> tuple[str, int, str, str] | None:
    """The value of the shell-like word that starts at ``text[i]``, the
    index after it, its form and its escaped characters. None when a quote
    does not close."""
    out: list[str] = []
    escaped: list[str] = []
    form = {"'": "single", '"': "double"}.get(text[i], "plain")
    n = len(text)
    while i < n and text[i] not in _SPACE:
        c = text[i]
        if c == "\\" and i + 1 < n:
            out.append(text[i + 1])
            escaped.append(text[i + 1])
            i += 2
        elif c == "'":
            close = text.find("'", i + 1)
            if close < 0:
                return None
            out.append(text[i + 1:close])
            i = close + 1
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n and text[i + 1] in _DOUBLE_ESCAPED:
                    i += 1
                out.append(text[i])
                i += 1
            if i >= n:
                return None
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out), i, form, "".join(dict.fromkeys(escaped))


def path_words(text: str, home: str) -> list[Word] | None:
    """The words of a paste that holds nothing but file paths, or None for
    any other paste. ``home`` is the Mac home folder, for a path that starts
    with ``~/``."""
    words: list[Word] = []
    i, n = 0, len(text)
    while True:
        while i < n and text[i] in _SPACE:
            i += 1
        if i >= n:
            break
        got = _read_word(text, i)
        if got is None:
            return None
        value, end, form, escaped = got
        if value.startswith("file://"):
            parts = urllib.parse.urlsplit(value)
            if parts.netloc not in ("", "localhost") or parts.query or parts.fragment:
                return None
            path, form = urllib.parse.unquote(parts.path), "url"
        elif value.startswith("~/") and text[i] == "~":
            path = home.rstrip("/") + value[1:]
        else:
            path = value
        if not path.startswith("/") or "\0" in path:
            return None
        words.append(Word(i, end, path, form, escaped))
        i = end
    return words or None


def written(path: str, word: Word) -> str:
    """``path`` written in the form of ``word``."""
    if word.form == "url":
        return "file://" + urllib.parse.quote(path, safe="/")
    if word.form == "single":
        return "'" + path.replace("'", "'\\''") + "'"
    if word.form == "double":
        return '"' + "".join("\\" + c if c in _DOUBLE_ESCAPED else c for c in path) + '"'
    special = set(_ALWAYS_ESCAPED) | set(word.escaped)
    return "".join("\\" + c if c in special else c for c in path)


def guest_name(name: str) -> str:
    """``name`` with each control or format character and each slash
    replaced by ``_``, and ``file`` in place of an empty name or a dot
    name."""
    clean = "".join("_" if c == "/" or unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp")
                    else c for c in name)
    return clean if clean not in ("", ".", "..") else "file"


def size_text(n: int) -> str:
    """``n`` bytes as a short size, such as ``1 GiB`` or ``512 MiB``."""
    for unit, shift in (("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if n >= 1 << shift and n % (1 << shift) == 0:
            return f"{n >> shift} {unit}"
    return f"{n:,} bytes"


def _why(e: OSError) -> str:
    """The reason of an error that opening or reading a Mac file gave."""
    if e.errno in (errno.EPERM, errno.EACCES):
        return ("macOS did not let the terminal app read it. Allow the access in System "
                "Settings, Privacy and Security, then paste again")
    return e.strerror or str(e)


def open_file(path: str) -> tuple[int, os.stat_result, str]:
    """An open descriptor and the status of the regular file at the Mac
    path ``path``, and the real path it was opened at. Raises
    :class:`Refused` for a path whose last part is a symbolic link, that
    :func:`confine.host_path` refuses, such as one in a private home, that
    lies in a folder that launch refuses to share, or that is not a regular
    file. The file is opened from ``/`` with no link followed, so a folder
    swapped for a link after the check is refused."""
    from . import settings

    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            raise Refused("it is a symbolic link")
        real = str(confine.host_path(path))
    except FileNotFoundError:
        raise Refused("no such file") from None
    except confine.ConfinedError as e:
        raise Refused(str(e)) from None
    except OSError as e:
        raise Refused(_why(e)) from None
    home = settings._host_home()
    why = settings._sensitive_refusal(real, home)
    if why is not None:
        raise Refused(f"{settings._tilde(real, home)} {why}, which launch does not share")
    try:
        fd = open_file_below("/", [p for p in real.split("/") if p])
    except FileNotFoundError:
        raise Refused("no such file") from None
    except (LeavesRoot, NotFollowed):
        raise Refused(f"a folder on the way to {real} changed into a link") from None
    except OSError as e:
        raise Refused(_why(e)) from None
    try:
        st = os.fstat(fd)
        if stat.S_ISDIR(st.st_mode):
            raise Refused("it is a folder, and only files are placed")
        if not stat.S_ISREG(st.st_mode):
            raise Refused("it is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd, st, real


def copy_fd(src: int, dst: int, limit: int) -> int:
    """Copy the file ``src`` to ``dst`` from its start, and return the
    number of bytes. Raises :class:`TooLarge` past ``limit`` bytes, so a
    file that grows during the copy is refused too."""
    os.lseek(src, 0, os.SEEK_SET)
    done = 0
    while True:
        chunk = os.read(src, COPY_CHUNK)
        if not chunk:
            return done
        done += len(chunk)
        if done > limit:
            raise TooLarge(limit)
        confine.write_all(dst, chunk)


def entry_key(st: os.stat_result) -> str:
    """The folder name of a file in the pastes folder, from what identifies
    the file and its version on the Mac."""
    ident = f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"
    return hashlib.sha256(ident.encode()).hexdigest()[:16]


class Pastes:
    """Places the files of pastes in the private home ``home``.
    ``same_path`` lists the Mac folders that the container sees at the same
    path, such as the shares, and a path in one of them stays as it is.
    ``log`` takes a line for each file placed and each path that stays.
    ``copy_max`` is the largest file that is copied when a clone fails.
    ``mac_home`` is the Mac home folder, for a path that starts with
    ``~/``."""

    def __init__(self, home: str | os.PathLike, same_path: list[str], log: Log, *,
                 copy_max: int = COPY_MAX, mac_home: str | None = None):
        self.home = Path(home)
        self.same_path = list(same_path)
        self.log = log
        self.copy_max = copy_max
        self.mac_home = mac_home or os.path.expanduser("~")

    def rewrite(self, body: bytes) -> bytes:
        """The paste ``body`` with the path of each placed file in place of
        its Mac path. Any paste that is not a list of paths, or that is not
        UTF-8, comes back as it is. No error leaves this method: a paste
        that fails comes back as it is, and the log says why."""
        try:
            return self._rewrite(body)
        except Exception as e:  # noqa: BLE001 - a paste must never stop the input
            self._say(f"paste: the paste passed as it came, because launch failed to read "
                      f"it ({type(e).__name__}: {e})")
            return body

    def _say(self, line: str) -> None:
        try:
            self.log(line)
        except Exception:  # noqa: BLE001, S110 - a log that fails must not stop a paste
            pass

    def _rewrite(self, body: bytes) -> bytes:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return body
        words = path_words(text, self.mac_home)
        if words is None:
            return body
        if len(words) > PATHS_MAX:
            self._say(f"paste: the paste names more than {PATHS_MAX} files, so the ones "
                      "after that stay as they are")
        out, last = [], 0
        for word in words[:PATHS_MAX]:
            new = self.place(word.path)
            if new is None:
                continue
            out += [text[last:word.start], written(new, word)]
            last = word.end
        if not out:
            return body
        return ("".join(out) + text[last:]).encode("utf-8")

    def place(self, path: str) -> str | None:
        """The container path of the file at the Mac path ``path`` once it
        is placed in the private home, or None when the path stays as it
        is. Raises no error."""
        try:
            return self._place(path)
        except Exception as e:  # noqa: BLE001 - a paste must never stop the input
            self._say(f"paste: {path} stays as it is, because launch failed to place it "
                      f"({type(e).__name__}: {e})")
            return None

    def _place(self, path: str) -> str | None:
        if ".." not in path.split("/"):
            normal = os.path.normpath(path)
            if any(path_inside(normal, folder) for folder in self.same_path):
                self._say(f"paste: {path} is in a folder that the container sees at the "
                          "same path, so it stays as it is")
                return None
        try:
            fd, st, real = open_file(path)
        except Refused as e:
            self._say(f"paste: {path} stays as it is, because {e}")
            return None
        try:
            entry = self.home.joinpath(*PASTE_FOLDER, entry_key(st))
            target = entry / guest_name(os.path.basename(real))
            try:
                with confine.confined(self.home):
                    confine.clone(fd, target)
                how = "cloned"
            except Exception as e:  # noqa: BLE001 - any failure falls back to a copy
                failed = e.strerror if isinstance(e, OSError) and e.strerror else str(e)
                why = self._copy(fd, st, target)
                if why is not None:
                    self._say(f"paste: {path} stays as it is, because launch could not clone "
                              f"it ({failed}) and {why}")
                    return None
                how = "copied"
        finally:
            os.close(fd)
        try:
            with confine.confined(self.home):
                self._prune(entry)
        except Exception as e:  # noqa: BLE001 - the file is in place all the same
            self._say(f"paste: launch could not remove old entries of {entry.parent} ({e})")
        self._say(f"paste: {how} {path} to {target} ({st.st_size:,} bytes)")
        return str(target)

    def _copy(self, fd: int, st: os.stat_result, target: Path) -> str | None:
        """Copy the open file ``fd`` to ``target``. None when it is done,
        else the reason it failed."""
        if st.st_size > self.copy_max:
            return (f"it is larger than launch.container.paste_copy_max, "
                    f"{size_text(self.copy_max)}")
        try:
            with confine.confined(self.home):
                confine.write_stream(target, lambda dst: copy_fd(fd, dst, self.copy_max))
        except TooLarge:
            return (f"it grew past launch.container.paste_copy_max, "
                    f"{size_text(self.copy_max)}, during the copy")
        except Exception as e:  # noqa: BLE001 - any failure leaves the path as it is
            if isinstance(e, OSError) and not isinstance(e, confine.ConfinedError):
                return f"the copy failed ({_why(e)})"
            return f"the copy failed ({e})"
        return None

    def _prune(self, keep: Path) -> None:
        """Remove the oldest entries past :data:`KEEP`, never ``keep``, the
        one just placed. An entry is as old as its folder, which changes
        when a file is placed in it. Runs inside :func:`confine.confined`."""
        folder = keep.parent
        entries = []
        for name in confine.listdir(folder):
            if len(name) != 16 or name.strip("0123456789abcdef") or name == keep.name:
                continue
            st = confine.lstat(folder / name)
            # An entry that cannot be read counts as the oldest.
            entries.append((st.st_mtime_ns if st else 0, name))
        entries.sort(reverse=True)
        for _, name in entries[KEEP - 1:]:
            confine.remove_tree(folder / name)
