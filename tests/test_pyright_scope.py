#!/usr/bin/env python3
"""The pyright gate ([tool.pyright].include) and its backlog
([tool.gmlx.pyright].backlog) together name every file that imports
mlx_vlm or mlx_lm statically, minus gmlx/models (upstream mirrors,
annotation exempt). A new static import anywhere lands in one list or the
other; a file leaves the tree, it leaves both. Seams reached through
importlib.import_module are opaque to pyright and are not the gate's
business.
"""
from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = re.compile(
    r"^\s*(from (mlx_vlm|mlx_lm)(\.[\w.]+)? import\b"
    r"|import (mlx_vlm|mlx_lm)(\.[\w.]+)?( as \w+)?\s*$)",
    re.MULTILINE,
)


def _static_importers() -> set[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "gmlx/*.py", "scripts/*.py"],
        cwd=_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    out = set()
    for rel in tracked:
        if rel.startswith(("gmlx/_vendor/", "gmlx/models/")):
            continue
        if _STATIC.search((_ROOT / rel).read_text()):
            out.add(rel)
    return out


@pytest.fixture(scope="module")
def lists() -> tuple[set[str], set[str], set[str]]:
    with open(_ROOT / "pyproject.toml", "rb") as f:
        tool = tomllib.load(f)["tool"]
    gated = set(tool["pyright"]["include"])
    backlog = set(tool["gmlx"]["pyright"]["backlog"])
    return gated, backlog, _static_importers()


def test_gate_and_backlog_are_disjoint(lists):
    gated, backlog, _ = lists
    assert not gated & backlog, sorted(gated & backlog)


def test_ceiling_every_listed_file_is_a_static_importer(lists):
    gated, backlog, importers = lists
    stale = (gated | backlog) - importers
    assert not stale, f"no longer a static upstream importer: {sorted(stale)}"


def test_floor_every_static_importer_is_listed(lists):
    gated, backlog, importers = lists
    missing = importers - gated - backlog
    assert not missing, (
        f"static upstream importer in neither [tool.pyright].include nor "
        f"[tool.gmlx.pyright].backlog: {sorted(missing)}")
