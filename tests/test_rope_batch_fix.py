"""rope_batch_fix: mx.fast.rope int-offset batch corruption at T == 1.

Upstream (stock mlx 0.31.2, Metal): rope with a plain int offset on a
(B, *, 1, D) input with B > 1 corrupts every batch row past the first
(out-of-bounds reads; CPU path is correct). The fix expands the int
offset to a per-row int32 array, which routes onto the healthy kernel.

test_upstream_bug_still_present is a tripwire: when an mlx upgrade fixes
the kernel it starts failing, which is the signal to drop the workaround.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from gmlx import rope_batch_fix as rbf

D = 64
FREQS = mx.array([10000.0 ** (2 * i / D) for i in range(D // 2)],
                 dtype=mx.float32)

ON_CPU = bool(os.environ.get("KQUANT_FORCE_CPU"))


def _stock():
    return getattr(mx.fast.rope, "_gmlx_orig_rope", mx.fast.rope)


def _rope(fn, x, offset, stream=None):
    kw = {"stream": stream} if stream is not None else {}
    return fn(x, D, traditional=False, base=None, scale=1.0,
              offset=offset, freqs=FREQS, **kw)


@pytest.fixture()
def installed():
    saved = mx.fast.rope
    saved_installed = rbf._installed
    saved_orig = rbf._orig_rope
    rbf._installed = False
    assert rbf.install_rope_batch_fix()
    yield
    mx.fast.rope = saved
    rbf._installed = saved_installed
    rbf._orig_rope = saved_orig


@pytest.fixture()
def spy(installed, monkeypatch):
    calls = []
    real = rbf._orig_rope

    def recorder(a, dims, **kw):
        calls.append(kw["offset"])
        return real(a, dims, **kw)

    monkeypatch.setattr(rbf, "_orig_rope", recorder)
    return calls


def _rand(shape):
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.float32)
    mx.eval(x)
    return x


@pytest.mark.skipif(ON_CPU, reason="upstream bug is GPU-only")
def test_upstream_bug_still_present():
    # When this fails after an mlx bump, the kernel got fixed upstream:
    # drop rope_batch_fix and this file together.
    x = _rand((2, 1, D))
    gpu = _rope(_stock(), x, 48, stream=mx.gpu)
    cpu = _rope(_stock(), x, 48, stream=mx.cpu)
    mx.eval(gpu, cpu)
    assert not mx.allclose(gpu[1], cpu[1], atol=1e-3).item()


def test_fix_matches_cpu_reference(installed):
    for shape in ((2, 1, D), (3, 1, D), (2, 8, 1, D)):
        x = _rand(shape)
        fixed = _rope(mx.fast.rope, x, 48)
        cpu = _rope(rbf._orig_rope, x, 48, stream=mx.cpu)
        mx.eval(fixed, cpu)
        assert mx.allclose(fixed, cpu, atol=1e-4).item(), shape


def test_converts_only_broken_case(spy):
    _rope(mx.fast.rope, _rand((2, 1, D)), 48)
    _rope(mx.fast.rope, _rand((4, 8, 1, D)), 7)
    assert isinstance(spy[0], mx.array) and spy[0].shape == (2,)
    assert isinstance(spy[1], mx.array) and spy[1].shape == (4,)
    assert spy[1].dtype == mx.int32
    assert spy[1].tolist() == [7, 7, 7, 7]


def test_passthrough_cases(spy):
    _rope(mx.fast.rope, _rand((1, 1, D)), 48)          # B == 1
    _rope(mx.fast.rope, _rand((2, 4, D)), 48)          # T > 1
    _rope(mx.fast.rope, _rand((1, 8, 1, D)), 48)       # 4-D, B == 1
    arr = mx.array([1, 2], dtype=mx.int32)
    _rope(mx.fast.rope, _rand((2, 1, D)), arr)         # already array
    assert spy[0] == 48 and type(spy[0]) is int
    assert spy[1] == 48 and type(spy[1]) is int
    assert spy[2] == 48 and type(spy[2]) is int
    assert spy[3] is arr


def test_env_kill_switch(spy, monkeypatch):
    monkeypatch.setenv("GMLX_ROPE_BATCH_FIX", "0")
    _rope(mx.fast.rope, _rand((2, 1, D)), 48)
    assert spy[0] == 48 and type(spy[0]) is int


def test_install_idempotent_and_unwraps(installed):
    first = mx.fast.rope
    assert rbf.install_rope_batch_fix()
    assert mx.fast.rope is first
    # a re-install after module state reset must not wrap the wrapper
    rbf._installed = False
    assert rbf.install_rope_batch_fix()
    assert mx.fast.rope._gmlx_orig_rope is rbf._orig_rope
    assert not hasattr(rbf._orig_rope, "_gmlx_orig_rope")
