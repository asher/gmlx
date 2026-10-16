#!/usr/bin/env python3
"""Each gmlx file that carries the copyright line of a ported source is
named in the license file of that source. CPU-only."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Copyright holder -> the license file that lists the files ported from it.
_SOURCES = {"Black Forest Labs, Inc.": "licenses/flux-3-image-klein-LICENSE"}


def test_each_ported_file_is_named_in_the_license_file_of_its_source():
    missing = []
    for path in sorted((_ROOT / "gmlx").rglob("*.py")):
        with open(path, encoding="utf-8", errors="replace") as f:
            head = [next(f, "") for _ in range(20)]
        rel = path.relative_to(_ROOT).as_posix()
        for holder, lic in _SOURCES.items():
            if any(line.startswith("# SPDX-FileCopyrightText:")
                   and holder in line for line in head) \
                    and f"- {rel}:" not in (_ROOT / lic).read_text():
                missing.append(f"{rel} is not named in {lic}")
    assert not missing, missing


def test_the_check_finds_the_ported_files():
    """The FLUX.3 klein files carry the line the check reads."""
    with open(_ROOT / "gmlx/image/families/flux3_klein/edit.py") as f:
        assert any(line.startswith("# SPDX-FileCopyrightText:")
                   and "Black Forest Labs, Inc." in line for line in f)
