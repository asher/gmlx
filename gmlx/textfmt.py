"""Tiny shared text-formatting helpers."""

from __future__ import annotations


def plural_s(n: int) -> str:
    """Pluralizing suffix: ``1 check`` / ``2 checks``, never ``check(s)``."""
    return "" if n == 1 else "s"
