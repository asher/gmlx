"""tests/arg_order.py: the command line's test paths grouped by directory,
so a conftest's fixtures reach every test below it in any order."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from arg_order import grouped

_ROOT = Path(__file__).resolve().parent.parent


def test_each_directory_s_paths_are_consecutive_in_first_order():
    args = ["tests/serve/a.py", "tests/b.py", "tests/serve/c.py::t[1]",
            "tests/load/d.py", "tests/e.py", "tests/serve/sub/f.py"]
    assert grouped(args) == [
        "tests/serve/a.py", "tests/serve/c.py::t[1]", "tests/serve/sub/f.py",
        "tests/b.py", "tests/load/d.py", "tests/e.py"]
    assert grouped(["tests"]) == ["tests"]
    assert grouped([]) == []


def test_a_directory_named_twice_keeps_its_conftest_fixtures():
    """The order that hid tests/serve/conftest.py from a test of
    tests/serve on pytest 9.1: a tests/serve path, a tests path, then the
    test, whose fixture that conftest defines."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "no:xdist", "tests/serve/test_admit_gate.py",
         "tests/test_envflags.py",
         "tests/serve/test_server_patches.py::"
         "test_the_seam_restore_lets_the_media_salt_install_again"],
        cwd=_ROOT, capture_output=True, text=True, timeout=600,
        env=dict(os.environ))
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
