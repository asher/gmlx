"""The ``--adapter PATH[:STRENGTH]`` flag that every command shares.

The text after the last colon is a strength only when it parses as a
float, so a path may hold colons. The strength defaults to 1.
"""
from __future__ import annotations

import argparse
import math


def parse_adapter(text: str) -> tuple[str, float]:
    """PATH or PATH:STRENGTH, as ``(path, strength)``."""
    head, sep, tail = text.rpartition(":")
    path, strength = text, 1.0
    if sep and head:
        try:
            path, strength = head, float(tail)
        except ValueError:
            path, strength = text, 1.0
        else:
            # float() reads 1_0 as 10, which nobody means as a strength.
            if "_" in tail:
                raise argparse.ArgumentTypeError(
                    f"a strength is written without underscores, got "
                    f"{tail!r} in {text!r}")
    if not path or not math.isfinite(strength):
        raise argparse.ArgumentTypeError(
            f"a LoRA file with an optional finite strength, as PATH or "
            f"PATH:STRENGTH, is required, got {text!r}")
    return path, strength


def format_adapter(path: str, strength: float) -> str:
    """The flag value that parses back to ``(path, strength)``."""
    return f"{path}:{float(strength)!r}"


def add_adapter_flag(p, help: str) -> None:
    """Add the repeatable flag. ``args.adapter`` is a list of
    ``(path, strength)`` or None."""
    p.add_argument("--adapter", action="append", type=parse_adapter,
                   default=None, metavar="PATH[:STRENGTH]", help=help)


def adapter_paths(items) -> list[str]:
    """The paths of ``args.adapter``."""
    return [p for p, _s in items or ()]


def duplicate_adapter(items) -> str | None:
    """A path given twice in ``items``, compared after normpath, or None."""
    import os

    seen: set = set()
    for p, _s in items or ():
        key = os.path.normpath(os.path.abspath(os.path.expanduser(p)))
        if key in seen:
            return p
        seen.add(key)
    return None
