#!/usr/bin/env python3
"""Say which kinds of file a change touches, for CI and the GPU gate.

A change is docs-only when every path it touches is under docs/ or
overrides/, is mkdocs.yml, or is a Markdown file outside gmlx/ and tests/.
Everything else is code, including tests and every file under gmlx/. A
change to the ComfyUI node pack, its tests or the verb that installs it
also needs the pack job.

  python scripts/changed_paths.py --base BASE --head HEAD

prints `code=true|false` and `pack=true|false`, one per line, ready for
$GITHUB_OUTPUT. The diff runs from the merge-base of BASE and HEAD, so a
branch behind main counts only its own changes. A base that is empty,
all zeros or not a commit here reports both as changed.

The file also holds the two test selections the CI legs, the GPU gate and
scripts/affected_tests.py share: the numerics folders and the docs tests.

Stdlib only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

DOCS_DIRS = ("docs/", "overrides/")
DOCS_FILES = ("mkdocs.yml",)
CODE_DIRS = ("gmlx/", "tests/")
PACK_PATHS = ("gmlx/comfyui_pack/", "tests/comfyui_pack/",
              "gmlx/commands/comfyui.py", ".github/workflows/test.yml")
# Test folders whose results depend on the device. The 3.11 leg skips them;
# the GPU gate runs them.
NUMERICS = ("models", "cache", "image", "gen", "spec", "distill", "tune", "load", "stream")
DOCS_TESTS = ("tests/test_docs_*.py", "tests/test_ascii_hygiene.py", "tests/test_mkdocs_hooks.py")


def is_docs(path: str) -> bool:
    if path.startswith(DOCS_DIRS) or path in DOCS_FILES:
        return True
    return PurePosixPath(path).suffix == ".md" and not path.startswith(CODE_DIRS)


def is_pack(path: str) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in PACK_PATHS)


def classify(paths: list[str] | None) -> dict[str, bool]:
    """``None`` means the change is unknown, which counts as everything."""
    if paths is None:
        return {"code": True, "pack": True}
    return {"code": not all(is_docs(p) for p in paths),
            "pack": any(is_pack(p) for p in paths)}


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def changed(base: str, head: str, cwd: str | None = None) -> list[str] | None:
    """The paths HEAD changes since its merge-base with BASE, or None when
    BASE is no commit to diff from."""
    if not base or set(base) == {"0"}:
        return None
    if _git("cat-file", "-e", f"{base}^{{commit}}", cwd=cwd).returncode != 0:
        return None
    diff = _git("diff", "--name-only", "--no-renames", f"{base}...{head}", cwd=cwd)
    if diff.returncode != 0:
        return None
    return [line for line in diff.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="", help="the commit the change starts from")
    ap.add_argument("--head", default="HEAD", help="the commit to classify")
    a = ap.parse_args(argv)
    for key, value in classify(changed(a.base, a.head)).items():
        print(f"{key}={'true' if value else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
