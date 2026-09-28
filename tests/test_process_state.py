"""Each test hands the process back as it found it.

The first test runs CLI entry points that change the process for their own
run, as ``gmlx distill`` does when it turns TF32 off, and the MLX setters
that library entry points call. The second, which runs next, must find
everything as the first found it. The last check runs at the end of the
session (tests/conftest.py moves it there): the environment still holds
the session's MLX_ENABLE_TF32, and float32 matmuls in this process round as
in a fresh process started from the session environment.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import warnings

import mlx.core as mx
import mlx_kquant as kq
import pytest

import process_state

_BEFORE: dict = {}
# Set for each test by pytest and by the conftest's own fixtures.
_PER_TEST = ("PYTEST_CURRENT_TEST", "XDG_DATA_HOME")


def _observe() -> dict:
    return {
        "env": {k: v for k, v in os.environ.items() if k not in _PER_TEST},
        "formatwarning": warnings.formatwarning,
        "device": mx.default_device(),
        "stream": mx.default_stream(mx.default_device()),
        "rng": mx.random.state[0].tolist(),
        "set_wired_limit": mx.set_wired_limit,
        "cb_caps": process_state.cb_caps(),
    }


def _run(argv) -> int:
    from gmlx.commands import cli

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return cli.umbrella_main(argv)


def test_a_cli_run_changes_the_process():
    _BEFORE.clear()
    _BEFORE.update(_observe())
    # gmlx distill turns TF32 off in its own process; --cpu moves it to the
    # CPU device. Without --teacher it refuses after both.
    assert _run(["distill", "cache", "--cpu"]) == 2
    assert os.environ["MLX_ENABLE_TF32"] == "0"
    assert mx.default_device() == mx.cpu
    assert warnings.formatwarning is not _BEFORE["formatwarning"]
    # A streamed run lifts the command buffer caps before it parses.
    _run(["run", "--stream-cpu", "--no-such-flag"])
    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "400"
    # The setters library entry points call.
    mx.random.seed(5)
    _BEFORE["cache_limit"] = mx.set_cache_limit(123 << 20)
    _BEFORE["memory_limit"] = mx.set_memory_limit(1 << 40)
    mx.set_memory_limit(_BEFORE["memory_limit"] - (1 << 20))
    mx.set_default_stream(mx.new_stream(mx.default_device()))
    mx.set_wired_limit = lambda *_a, **_k: 0
    if _BEFORE["cb_caps"] is not None:
        kq.set_cb_caps(7, 7)


def test_b_the_next_test_finds_the_process_as_it_was():
    if not _BEFORE:
        pytest.skip("runs after test_a_cli_run_changes_the_process")
    now = _observe()
    for key in ("env", "device", "stream", "rng", "cb_caps"):
        assert now[key] == _BEFORE[key], key
    assert now["formatwarning"] is _BEFORE["formatwarning"]
    assert now["set_wired_limit"] is _BEFORE["set_wired_limit"]
    assert mx.set_cache_limit(_BEFORE["cache_limit"]) == _BEFORE["cache_limit"]
    assert mx.set_memory_limit(_BEFORE["memory_limit"]) == _BEFORE["memory_limit"]


_GEMM = r"""
import os

import mlx.core as mx


def gemm_error():
    a = mx.random.normal((256, 256), key=mx.random.key(0))
    b = mx.random.normal((256, 256), key=mx.random.key(1))
    ref = mx.matmul(a, b, stream=mx.cpu)
    return mx.abs(mx.matmul(a, b) - ref).max().item()


if __name__ == "__main__":
    if os.environ.get("KQUANT_FORCE_CPU"):
        mx.set_default_device(mx.cpu)
    print(gemm_error())
"""


def test_the_run_ends_on_production_numerics():
    start = process_state.SESSION
    assert os.environ.get("MLX_ENABLE_TF32") == start["env"].get(
        "MLX_ENABLE_TF32")
    assert str(mx.default_device()) == start["device"]
    ns: dict = {}
    exec(_GEMM, ns)
    here = ns["gemm_error"]()
    r = subprocess.run([sys.executable, "-c", _GEMM], env=start["env"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    fresh = float(r.stdout.strip().splitlines()[-1])
    # TF32 rounding leaves about 5e-2 on these products, float32 about 5e-5.
    assert (here > 1e-3) == (fresh > 1e-3), (here, fresh)
