"""Tolerant numeric parsing for GMLX_* environment flags.

A malformed value (``GMLX_WALK_PROFILE=yes``) must degrade to the
default, not crash an import or a serve boot with a bare ValueError.
"""
from __future__ import annotations

import os


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    val = os.environ.get(name, default).strip().lower()
    return val if val in choices else default


_BOOL_TRUE = frozenset(("1", "true", "yes", "on"))
_BOOL_FALSE = frozenset(("0", "false", "no", "off", ""))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _BOOL_TRUE:
        return True
    if val in _BOOL_FALSE:
        return False
    return default
