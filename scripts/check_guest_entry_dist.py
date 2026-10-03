#!/usr/bin/env python3
"""Fail when the built sdist or wheel lacks the guest entry binary.

`gmlx launch --container` mounts gmlx/container/guest/gmlx-entry into the
container, so a release without it cannot run container mode. This checks
the newest sdist and wheel in a folder: the binary must be present, must be
a static aarch64 ELF file, and must keep its execute bit.

  python scripts/check_guest_entry_dist.py            # checks dist/
  python scripts/check_guest_entry_dist.py DIR

Stdlib only.
"""

from __future__ import annotations

import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_guest_entry import check_static_aarch64_elf  # noqa: E402

MEMBER = "gmlx/container/guest/gmlx-entry"


def _newest(folder: Path, pattern: str) -> Path:
    found = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"no {pattern} in {folder}")
    return found[-1]


def _check_bytes(name: str, data: bytes, mode: int) -> None:
    if not mode & 0o111:
        raise SystemExit(f"{name}: {MEMBER} has no execute bit (mode {mode:o})")
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            check_static_aarch64_elf(Path(tmp.name))
        except ValueError as e:
            raise SystemExit(f"{name}: {MEMBER}: {e}")


def check_sdist(path: Path) -> None:
    with tarfile.open(path) as tar:
        member = next((m for m in tar.getmembers()
                       if m.name.split("/", 1)[-1] == MEMBER), None)
        if member is None:
            raise SystemExit(f"{path.name}: {MEMBER} is missing")
        data = tar.extractfile(member).read()
        _check_bytes(path.name, data, member.mode)


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as whl:
        try:
            info = whl.getinfo(MEMBER)
        except KeyError:
            raise SystemExit(f"{path.name}: {MEMBER} is missing")
        _check_bytes(path.name, whl.read(info), info.external_attr >> 16)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    folder = Path(args[0] if args else "dist")
    sdist, wheel = _newest(folder, "*.tar.gz"), _newest(folder, "*.whl")
    check_sdist(sdist)
    check_wheel(wheel)
    print(f"{sdist.name} and {wheel.name} ship {MEMBER}, static aarch64, executable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
