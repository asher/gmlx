"""Docker ignore-file matching, for the rebuild hash of a ``build:`` image.

The hash covers every context file the build can see, so the matcher must
agree with BuildKit. This module is a port of moby's ``patternmatcher``
(``New``, ``Pattern.compile``, ``Pattern.match`` and
``MatchesOrParentMatches``) and of ``ignorefile.ReadAll``. A pattern the
port does not handle, an escape or an unusual character class, makes
:func:`load` report it, and the caller then hashes every file. A file the
hash skips wrongly would never trigger a rebuild, and one it includes
wrongly only costs a rebuild.

The ignore file can lie in a folder the client writes, so it is read with a
size limit, and a pattern is matched in time linear in the path, as Go's
regexp matches it, rather than with Python's backtracking ``re``.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

_EXACT, _PREFIX, _SUFFIX, _REGEXP = range(4)
_ESCAPE = set(".+()|{}$")
# The largest ignore file read, and the most patterns in it.
IGNORE_MAX = 1 << 20
PATTERNS_MAX = 200
# The most matching work one rebuild check spends on the ignore patterns,
# counted as the pattern states each path character moves, about 10 to 50
# million a second. A check past it stops using the patterns, so a crafted
# ignore file cannot hold up a launch.
WORK_MAX = 50_000_000
# The most unbounded repeats in a pattern with a character class, which is
# matched with ``re``. Two keep a match of a long path quick.
_CLASS_REPEATS_MAX = 2
# The steps of a pattern without a character class, matched without ``re``.
_LIT, _ONE, _STAR, _DSTAR, _OPTDIR = range(5)
# A character class the port matches the way Go's regexp does.
_SIMPLE_CLASS = re.compile(r"\[\^?[^\]\\\[]+\]")


class UnsupportedPattern(ValueError):
    """A pattern outside what this port matches the way BuildKit does."""


class TooMuchWork(UnsupportedPattern):
    """Matching the patterns passed :data:`WORK_MAX` in one check."""


def _clean(path: str) -> str:
    """Go's ``filepath.Clean`` for slash paths."""
    if path == "":
        return "."
    rooted = path.startswith("/")
    out: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if out and out[-1] != "..":
                out.pop()
            elif not rooted:
                out.append("..")
            continue
        out.append(part)
    cleaned = "/".join(out)
    if rooted:
        return "/" + cleaned
    return cleaned or "."


def _check_syntax(pattern: str) -> None:
    """Refuse what this port cannot match exactly, and what Go's
    ``filepath.Match`` rejects."""
    if "\\" in pattern:
        raise UnsupportedPattern(f"{pattern!r} uses a backslash escape")
    rest = _SIMPLE_CLASS.sub("", pattern)
    if "[" in rest or "]" in rest:
        raise UnsupportedPattern(f"{pattern!r} uses a character class this check does not read")


@dataclass
class _Pattern:
    cleaned: str
    exclusion: bool
    kind: int = _EXACT
    regex: re.Pattern | None = None
    steps: list[tuple[int, str]] | None = None

    def compile(self) -> None:
        text, i, n = self.cleaned, 0, len(self.cleaned)
        reg, kind, index = "^", _EXACT, 0
        steps: list[tuple[int, str]] = []
        repeats = 0
        while i < n:
            ch = text[i]
            i += 1
            if ch == "*":
                if i < n and text[i] == "*":
                    i += 1
                    if i < n and text[i] == "/":
                        i += 1
                    if i >= n:
                        if kind == _EXACT:
                            kind = _PREFIX
                        else:
                            reg += ".*"
                            steps.append((_DSTAR, ""))
                            repeats += 1
                            kind = _REGEXP
                    else:
                        reg += "(.*/)?"
                        steps.append((_OPTDIR, ""))
                        repeats += 1
                        kind = _REGEXP
                    if index == 0:
                        kind = _SUFFIX
                else:
                    reg += "[^/]*"
                    steps.append((_STAR, ""))
                    repeats += 1
                    kind = _REGEXP
            elif ch == "?":
                reg += "[^/]"
                steps.append((_ONE, ""))
                kind = _REGEXP
            elif ch in _ESCAPE:
                reg += "\\" + ch
                steps.append((_LIT, ch))
            elif ch in "[]":
                reg += ch
                kind = _REGEXP
            else:
                reg += ch
                steps.append((_LIT, ch))
            index += 1
        self.kind = kind
        if kind == _REGEXP:
            # \Z, since Go's $ matches only at the end of the text. No
            # re.S, since Go's . does not match a newline either.
            try:
                self.regex = re.compile(reg + r"\Z")
            except re.error as e:
                # BuildKit refuses such a pattern too, as Go's regexp does.
                raise UnsupportedPattern(f"{self.cleaned!r} is not a valid pattern ({e})") from None
            if "[" in text:
                # A character class is matched with re, which backtracks.
                if repeats > _CLASS_REPEATS_MAX:
                    raise UnsupportedPattern(
                        f"{self.cleaned!r} has a character class and more than "
                        f"{_CLASS_REPEATS_MAX} stars")
            else:
                self.steps = steps

    def match(self, path: str, meter: list[int] | None = None) -> bool:
        """Whether the pattern matches ``path``. ``meter[0]`` grows by the
        work the match took."""
        if meter is not None and self.steps is None:
            # A regular expression here has at most two unbounded repeats.
            meter[0] += len(path) * (len(self.cleaned) if self.regex is not None else 1) + 1
        if self.kind == _EXACT:
            return path == self.cleaned
        if self.kind == _PREFIX:
            return path.startswith(self.cleaned[:-2])
        if self.kind == _SUFFIX:
            suffix = self.cleaned[2:]
            if path.endswith(suffix):
                return True
            return suffix.startswith("/") and path == suffix[1:]
        if self.steps is not None:
            return _run_steps(self.steps, path, meter)
        assert self.regex is not None
        return self.regex.match(path) is not None


def _run_steps(steps: list[tuple[int, str]], path: str,
               meter: list[int] | None = None) -> bool:
    """Whether ``steps`` match the whole of ``path``, as the regular
    expression that :meth:`_Pattern.compile` builds from them matches it.
    The set of reachable steps moves one character at a time, so the time is
    the path length times the number of steps. A state is a step index, or
    ``-1 - i`` inside the ``.*`` of a ``(.*/)?`` at step ``i``."""
    end = len(steps)

    def close(states: set[int]) -> set[int]:
        todo = list(states)
        while todo:
            s = todo.pop()
            # A star, a double star and an optional folder can match nothing.
            if 0 <= s < end and steps[s][0] in (_STAR, _DSTAR, _OPTDIR) and s + 1 not in states:
                states.add(s + 1)
                todo.append(s + 1)
        return states

    states = close({0})
    for c in path:
        if meter is not None:
            meter[0] += len(states)
        nxt: set[int] = set()
        for s in states:
            if s < 0:                          # inside (.*/)?
                inner = -1 - s
                if c != "\n":
                    nxt.add(s)
                if c == "/":
                    nxt.add(inner + 1)
                continue
            if s == end:
                continue
            kind, ch = steps[s]
            if kind == _LIT:
                if c == ch:
                    nxt.add(s + 1)
            elif kind == _ONE:
                if c != "/":
                    nxt.add(s + 1)
            elif kind == _STAR:
                if c != "/":
                    nxt.add(s)
            elif kind == _DSTAR:
                if c != "\n":
                    nxt.add(s)
            elif kind == _OPTDIR:
                if c != "\n":
                    nxt.add(-1 - s)
                if c == "/":
                    nxt.add(s + 1)
        if not nxt:
            return False
        states = close(nxt)
    return end in states


class Matcher:
    """The patterns of one ignore file, in order."""

    def __init__(self, patterns: list[str], work_max: int | None = None):
        if len(patterns) > PATTERNS_MAX:
            raise UnsupportedPattern(f"it holds more than {PATTERNS_MAX} patterns")
        self._meter = [0]
        self.work_max = WORK_MAX if work_max is None else work_max
        self.patterns: list[_Pattern] = []
        for raw in patterns:
            p = raw.strip()
            if not p:
                continue
            p = _clean(p)
            exclusion = p.startswith("!")
            if exclusion:
                if len(p) == 1:
                    raise UnsupportedPattern('illegal exclusion pattern: "!"')
                p = p[1:]
            _check_syntax(p)
            pat = _Pattern(p, exclusion)
            pat.compile()
            self.patterns.append(pat)
        self.has_exclusions = any(p.exclusion for p in self.patterns)

    def excluded(self, path: str) -> bool:
        """``MatchesOrParentMatches``: True when the last pattern that matches
        the path, or one of its parent folders, is not a ``!`` exception."""
        matched = False
        parent = _clean(path[:path.rfind("/") + 1])      # Go's filepath.Dir
        dirs = parent.split("/")
        for pat in self.patterns:
            if pat.exclusion != matched:
                continue
            hit = self._match(pat, path)
            if not hit and parent != ".":
                for i in range(len(dirs)):
                    if self._match(pat, "/".join(dirs[:i + 1])):
                        hit = True
                        break
            if hit:
                matched = not pat.exclusion
        return matched


    @property
    def work(self) -> int:
        return self._meter[0]

    def _match(self, pat: _Pattern, path: str) -> bool:
        hit = pat.match(path, self._meter)
        if self._meter[0] > self.work_max:
            raise TooMuchWork("matching its patterns takes too long")
        return hit

    def excludes_all_below(self, folder: str) -> bool:
        """Whether ``folder`` and every path below it are excluded, so a walk
        can skip the folder. The folder must be excluded, and no ``!``
        exception may match a path below it. An exception without ``**`` or
        a character class matches only paths with as many components as it
        has, so it cannot match below a folder at that depth or deeper. The
        folder's own result then carries to every path below it."""
        if not self.excluded(folder):
            return False
        depth = folder.count("/") + 1
        for pat in self.patterns:
            if not pat.exclusion:
                continue
            if "**" in pat.cleaned or "[" in pat.cleaned:
                return False
            if pat.cleaned.count("/") + 1 > depth:
                return False
        return True


def _lines(text: str) -> list[str]:
    """Go's ``bufio.ScanLines``: split on newlines only, and drop one
    carriage return at the end of each line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def read_patterns(text: str) -> list[str]:
    """``ignorefile.ReadAll``: comments, blank lines and leading slashes."""
    out = []
    for number, line in enumerate(_lines(text)):
        if number == 0:
            line = line.removeprefix("\ufeff")
        if line.startswith("#"):
            continue
        line = line.strip()
        if not line:
            continue
        invert = line.startswith("!")
        if invert:
            line = line[1:].strip()
        if line:
            line = _clean(line)
            if len(line) > 1 and line.startswith("/"):
                line = line[1:]
        out.append(("!" if invert else "") + line)
    return out


def ignore_file(containerfile: Path, context: Path) -> Path | None:
    """The ignore file a build of ``containerfile`` applies: the one named
    after the Containerfile beside it, else the context's ``.dockerignore``.
    A symbolic link counts as present, and :func:`read_ignore` then refuses
    it."""
    named = containerfile.with_name(containerfile.name + ".dockerignore")
    if os.path.lexists(named):
        return named
    root = context / ".dockerignore"
    return root if os.path.lexists(root) else None


def read_ignore(path: Path) -> str:
    """The text of an ignore file: a regular file of at most
    :data:`IGNORE_MAX` bytes, never read through a link, and never one that
    would block, such as a named pipe."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as e:
        if os.path.islink(path):
            raise UnsupportedPattern("it is a symbolic link") from None
        raise UnsupportedPattern(f"it cannot be read ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise UnsupportedPattern("it is not a regular file")
        if st.st_size > IGNORE_MAX:
            raise UnsupportedPattern(f"it is larger than {IGNORE_MAX >> 20} MiB")
        data = b""
        while len(data) <= IGNORE_MAX:
            chunk = os.read(fd, IGNORE_MAX + 1 - len(data))
            if not chunk:
                break
            data += chunk
        if len(data) > IGNORE_MAX:
            raise UnsupportedPattern(f"it is larger than {IGNORE_MAX >> 20} MiB")
    finally:
        os.close(fd)
    return data.decode("utf-8", errors="replace")


def load(containerfile: Path, context: Path) -> tuple[Matcher | None, str | None]:
    """The matcher for a build, and a notice when its ignore file cannot be
    read safely or holds a pattern this port does not read. Both are None
    without an ignore file."""
    path = ignore_file(containerfile, context)
    if path is None:
        return None, None
    try:
        return Matcher(read_patterns(read_ignore(path))), None
    except UnsupportedPattern as e:
        return None, (f"[launch] {path}: {e}, so every context file counts toward the "
                      "rebuild check.")
