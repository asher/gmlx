# vendor_upstream: skip
"""The session blocker fails a run that recorded a blocked import, even
when the importer caught the ImportError and every test passed. Runs a
fresh pytest process with the blocker as its only plugin."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent

_GUARDED = """
def test_a_guarded_stock_import():
    try:
        import {name}  # noqa: F401
    except ImportError:
        pass
"""

_CLEAN = """
def test_nothing_stock():
    import json  # noqa: F401
"""


@pytest.mark.parametrize("name", ["mlx_lm", "mlx_vlm", "mlx_embeddings",
                                  None])
def test_a_caught_blocked_import_fails_the_run(tmp_path, name):
    body = _CLEAN if name is None else _GUARDED.format(name=name)
    code = 0 if name is None else 1
    (tmp_path / "test_inner.py").write_text(body)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(_TESTS)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "stock_blocker", "test_inner.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == code, r.stdout[-2000:]
    assert "1 passed" in r.stdout
    if name is not None:
        assert f"{name}  <- " in r.stdout
