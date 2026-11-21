"""Tests that compare float32 output with a float32 reference.

MLX multiplies float32 matrices at TF32 precision on M5-class GPUs unless
``MLX_ENABLE_TF32=0`` is set before the first product of the process, and
it reads the switch once. A test marked ``@f32_exact`` runs in place where
the process multiplies float32 exactly. Elsewhere it reruns itself in a
child process with TF32 off, on the same device.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _exact() -> bool:
    from gmlx.image import float32

    return float32.f32_products_exact()


def f32_exact(test):
    @functools.wraps(test)
    def run(*args, **kwargs):
        if _exact():
            return test(*args, **kwargs)
        if os.environ.get("MLX_ENABLE_TF32") == "0":
            pytest.fail("float32 products are inexact with MLX_ENABLE_TF32=0")
        node = os.environ["PYTEST_CURRENT_TEST"].rsplit(" (", 1)[0]
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
             node],
            cwd=_ROOT, env=dict(os.environ, MLX_ENABLE_TF32="0"),
            capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
        if " passed" not in r.stdout:
            skips = [line.split(": ", 1)[-1] for line in r.stdout.splitlines()
                     if line.startswith("SKIPPED")]
            if skips:
                pytest.skip(skips[0])
        assert " passed" in r.stdout, r.stdout[-2000:]

    return run
