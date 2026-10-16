#!/usr/bin/env python3
"""Each gmlx file that carries the copyright line of a ported source is
named in the license file of that source. CPU-only."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Copyright holder -> the license files that list the files ported from
# its sources. A ported file's header names the ones it follows.
_SOURCES = {"Black Forest Labs, Inc.": ("licenses/flux-3-image-klein-LICENSE",
                                        "licenses/flux2-LICENSE")}


def _head(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [next(f, "") for _ in range(20)]


def test_each_ported_file_is_named_in_the_license_file_of_its_source():
    missing = []
    for path in sorted((_ROOT / "gmlx").rglob("*.py")):
        head = _head(path)
        rel = path.relative_to(_ROOT).as_posix()
        for holder, lics in _SOURCES.items():
            if not any(line.startswith("# SPDX-FileCopyrightText:")
                       and holder in line for line in head):
                continue
            named = [lic for lic in lics if any(lic in line for line in head)]
            if not named:
                missing.append(f"{rel} names none of {', '.join(lics)}")
            missing += [f"{rel} is not named in {lic}" for lic in named
                        if f"- {rel}:" not in (_ROOT / lic).read_text()]
    assert not missing, missing


def test_each_file_a_license_file_names_exists():
    """The ported files a license file lists are files of gmlx."""
    gone = []
    for lics in _SOURCES.values():
        for lic in lics:
            for line in (_ROOT / lic).read_text().splitlines():
                if line.startswith("- gmlx/") and ":" in line:
                    rel = line[2:line.index(":")]
                    if not (_ROOT / rel).is_file():
                        gone.append(f"{lic} names {rel}")
    assert not gone, gone


def test_the_check_finds_the_ported_files():
    """The FLUX.3 klein and FLUX.2 klein files carry the line the check
    reads, and name their license files."""
    for rel, lic in (("gmlx/image/families/flux3_klein/edit.py",
                      "licenses/flux-3-image-klein-LICENSE"),
                     ("gmlx/image/families/flux2_klein/family.py",
                      "licenses/flux2-LICENSE")):
        head = _head(_ROOT / rel)
        assert any(line.startswith("# SPDX-FileCopyrightText:")
                   and "Black Forest Labs, Inc." in line for line in head)
        assert any(lic in line for line in head)
