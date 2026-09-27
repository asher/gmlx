"""Docker ignore-file matching, for the rebuild hash of a ``build:`` image.

The hash covers every context file the build can see, so the matcher must
agree with BuildKit. This module is a port of moby's ``patternmatcher``
(``New``, ``Pattern.compile``, ``Pattern.match`` and
``MatchesOrParentMatches``) and of ``ignorefile.ReadAll``. A pattern the
port does not handle, an escape or an unusual character class, makes
:func:`load` report it, and the caller then hashes every file. A file the
hash skips wrongly would never trigger a rebuild, and one it includes
wrongly only costs a rebuild.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_EXACT, _PREFIX, _SUFFIX, _REGEXP = range(4)
_ESCAPE = set(".+()|{}$")
# A character class the port matches the way Go's regexp does.
_SIMPLE_CLASS = re.compile(r"\[\^?[^\]\\\[]+\]")


class UnsupportedPattern(ValueError):
    """A pattern outside what this port matches the way BuildKit does."""


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

    def compile(self) -> None:
        text, i, n = self.cleaned, 0, len(self.cleaned)
        reg, kind, index = "^", _EXACT, 0
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
                            kind = _REGEXP
                    else:
                        reg += "(.*/)?"
                        kind = _REGEXP
                    if index == 0:
                        kind = _SUFFIX
                else:
                    reg += "[^/]*"
                    kind = _REGEXP
            elif ch == "?":
                reg += "[^/]"
                kind = _REGEXP
            elif ch in _ESCAPE:
                reg += "\\" + ch
            elif ch in "[]":
                reg += ch
                kind = _REGEXP
            else:
                reg += ch
            index += 1
        self.kind = kind
        if kind == _REGEXP:
            # \Z, since Go's $ matches only at the end of the text.
            try:
                self.regex = re.compile(reg + r"\Z", re.S)
            except re.error as e:
                # BuildKit refuses such a pattern too, as Go's regexp does.
                raise UnsupportedPattern(f"{self.cleaned!r} is not a valid pattern ({e})") from None

    def match(self, path: str) -> bool:
        if self.kind == _EXACT:
            return path == self.cleaned
        if self.kind == _PREFIX:
            return path.startswith(self.cleaned[:-2])
        if self.kind == _SUFFIX:
            suffix = self.cleaned[2:]
            if path.endswith(suffix):
                return True
            return suffix.startswith("/") and path == suffix[1:]
        assert self.regex is not None
        return self.regex.match(path) is not None


class Matcher:
    """The patterns of one ignore file, in order."""

    def __init__(self, patterns: list[str]):
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
            hit = pat.match(path)
            if not hit and parent != ".":
                for i in range(len(dirs)):
                    if pat.match("/".join(dirs[:i + 1])):
                        hit = True
                        break
            if hit:
                matched = not pat.exclusion
        return matched


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
    after the Containerfile beside it, else the context's ``.dockerignore``."""
    named = containerfile.with_name(containerfile.name + ".dockerignore")
    if named.is_file():
        return named
    root = context / ".dockerignore"
    return root if root.is_file() else None


def load(containerfile: Path, context: Path) -> tuple[Matcher | None, str | None]:
    """The matcher for a build, and a notice when its ignore file holds a
    pattern this port does not read. Both are None without an ignore file."""
    path = ignore_file(containerfile, context)
    if path is None:
        return None, None
    try:
        return Matcher(read_patterns(path.read_text(encoding="utf-8", errors="replace"))), None
    except UnsupportedPattern as e:
        return None, (f"[launch] {path}: {e}, so every context file counts toward the "
                      "rebuild check.")
