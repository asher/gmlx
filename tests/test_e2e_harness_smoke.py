"""The tests/e2e harnesses are not pytest-collected (they need GPUs, large
GGUFs, or a PTY), so a refactor can silently break their imports or argument
parsing. This smoke runs each one's --help in a subprocess: imports execute,
the argparse tree builds, exit code is 0. No server, no model."""

import pathlib
import subprocess
import sys

import pytest

E2E_DIR = pathlib.Path(__file__).parent / "e2e"
HARNESSES = sorted(p.name for p in E2E_DIR.glob("run_*.py"))


def test_harnesses_discovered():
    assert "run_apc_depth_e2e.py" in HARNESSES
    assert "run_apc_disk_e2e.py" in HARNESSES


@pytest.mark.parametrize("name", HARNESSES)
def test_harness_help_exits_clean(name):
    proc = subprocess.run(
        [sys.executable, str(E2E_DIR / name), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"{name} --help exited {proc.returncode}\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    )
    assert "usage:" in proc.stdout.lower()
