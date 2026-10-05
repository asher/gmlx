"""Lines that launch prints once, or once a day, for what they are about.

Some launch lines carry no news after the first time, such as the note that
macOS guards a shared folder. Such a line is a :class:`Once` with a key,
and :func:`due` passes it only when its key was never printed, or was
printed longer ago than the line's interval. The keys and the times they
were printed are kept in ``notices.json`` in the launch data folder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .state import FileLock, data_dir, write_record

# The most keys the record keeps. The oldest go first.
NOTICES_MAX = 500
DAY = 24 * 3600.0


class Once(str):
    """A line that prints once for ``key``, or once in ``every`` seconds."""

    key: str
    every: float | None

    def __new__(cls, text: str, key: str, every: float | None = None):
        line = super().__new__(cls, text)
        line.key, line.every = key, every
        return line


def _path() -> Path:
    return data_dir() / "notices.json"


def _read() -> dict[str, float]:
    try:
        seen = json.loads(_path().read_text())
    except (OSError, ValueError, RecursionError):
        return {}
    if not isinstance(seen, dict):
        return {}
    return {k: v for k, v in seen.items() if isinstance(v, (int, float))}


def due(lines: list[str], *, record: bool = True, now: float | None = None) -> list[str]:
    """The lines to print now: every plain line, and each :class:`Once`
    whose key was never printed, or was printed at least ``every`` seconds
    ago. With ``record``, the time each such line prints is recorded. A
    record that cannot be written costs only the repeat of the line."""
    if not any(isinstance(line, Once) for line in lines):
        return list(lines)
    now = time.time() if now is None else now
    out: list[str] = []
    try:
        with FileLock(data_dir() / "notices.lock"):
            seen = _read()
            changed = False
            for line in lines:
                if isinstance(line, Once):
                    last = seen.get(line.key)
                    if last is not None and (line.every is None or now - last < line.every):
                        continue
                    seen[line.key] = now
                    changed = True
                out.append(line)
            if record and changed:
                _write(seen)
    except OSError:
        return list(lines)
    return out


def record(lines: list[str], *, now: float | None = None) -> None:
    """Record that each :class:`Once` in ``lines`` printed at ``now``, such
    as the lines that :func:`due` passed without ``record``. A record that
    cannot be written costs only the repeat of the line."""
    keys = [line.key for line in lines if isinstance(line, Once)]
    if not keys:
        return
    now = time.time() if now is None else now
    try:
        with FileLock(data_dir() / "notices.lock"):
            _write({**_read(), **dict.fromkeys(keys, now)})
    except OSError:
        pass


def _write(seen: dict[str, float]) -> None:
    kept = dict(sorted(seen.items(), key=lambda kv: kv[1])[-NOTICES_MAX:])
    write_record(_path(), json.dumps(kept, indent=1, sort_keys=True).encode())
