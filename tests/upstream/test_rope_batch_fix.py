"""rope_batch_fix: mx.fast.rope scalar-offset batch corruption.

Upstream history, two defects:
1. Single dispatch path (row-contiguous, T == 1, scalar offset): stock
   mlx 0.31.2 spanned the grid over one batch, leaving rows past the
   first unwritten (stale or donated buffer contents). Fixed in mlx
   0.32.1 (grid spans B*N); a guard test pins the fixed behavior.
2. General path (every other layout): still broken in 0.32.1. The host
   passes offset.strides()[0] as the per-batch offset stride, which is
   1 for a shape-(1,) array, so the kernel reads offset[batch] past the
   end of a size-1 offset buffer for every batch row after the first.
   Plain ints and 0-d arrays take stride 0 there and are safe.

The fix expands any offset carrying fewer entries than B at T == 1 to a
per-row int32 array. It does not cover the T > 1 size-1 case (its
trigger requires shape[-2] == 1); no gmlx path passes that shape.

The tripwires slice the offset out of a poison-valued array, so the OOB
read is deterministic: it lands inside the same live buffer and each
corrupt batch row rotates by exactly its poison value, with no
dependence on allocator history. They fail when an mlx upgrade fixes
the general-path stride handling, which is the signal to re-evaluate
the workaround.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

import gmlx.upstream.rope_batch_fix as rbf

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


_POISON = [48, 9000, 17000, 23000]


def _poison_offset():
    pool = mx.array(_POISON, dtype=mx.int32)
    off = pool[:1]
    mx.eval(off)
    return off, pool


def _general_arm(layout):
    # Shapes whose T == 1 non-contiguity, T > 1, or head-seq transposed
    # strides dispatch the general kernel with batch count 4.
    if layout == "noncontig_t1":
        return mx.swapaxes(_rand((8, 4, 1, D)), 0, 1)
    if layout == "t8":
        return _rand((4, 8, 8, D))
    return mx.swapaxes(_rand((4, 3, 8, D)), 1, 2)


@pytest.mark.skipif(ON_CPU, reason="upstream bug is GPU-only")
def test_upstream_single_path_fixed():
    # mlx 0.32.1 spans the single-path grid over all B*N rows. If this
    # fails, the 0.31.2 unwritten-batch-rows bug is back.
    for off in (48, mx.array(48), _poison_offset()[0]):
        x = _rand((2, 1, D))
        gpu = _rope(_stock(), x, off, stream=mx.gpu)
        cpu = _rope(_stock(), x, off, stream=mx.cpu)
        mx.eval(gpu, cpu)
        assert mx.allclose(gpu, cpu, atol=1e-3).item()


@pytest.mark.skipif(ON_CPU, reason="upstream bug is GPU-only")
@pytest.mark.parametrize("layout", ["noncontig_t1", "t8", "hst"])
def test_upstream_bug_still_present(layout):
    # When a layout goes clean after an mlx bump, the general-path
    # offset-stride handling got fixed upstream: re-evaluate the
    # workaround (and the T > 1 size-1 gap it does not cover).
    off, pool = _poison_offset()
    x = _general_arm(layout)
    gpu = _rope(_stock(), x, off, stream=mx.gpu)
    cpu = _rope(_stock(), x, off, stream=mx.cpu)
    mx.eval(gpu, cpu)
    assert mx.allclose(gpu[0], cpu[0], atol=1e-3).item()
    assert not mx.allclose(gpu[1], cpu[1], atol=1e-3).item()
    # The corrupt rows rotate by exactly their poison values: the same
    # call with the full poison pool as a per-batch offset agrees.
    ref = _rope(_stock(), x, pool, stream=mx.cpu)
    mx.eval(ref)
    for b in range(1, 4):
        assert mx.allclose(gpu[b], ref[b], atol=1e-2).item(), (layout, b)


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


def test_converts_scalar_array_offsets(spy):
    # 0-d (the mx.array(cache.offset) wrap in mlx-lm gemma4_text) and
    # size-1 arrays hit the same broken kernel path as plain ints.
    _rope(mx.fast.rope, _rand((3, 1, D)), mx.array(9))
    _rope(mx.fast.rope, _rand((4, 8, 1, D)), mx.array([11]))
    for got, want in zip(spy, ([9, 9, 9], [11, 11, 11, 11])):
        assert isinstance(got, mx.array)
        assert got.dtype == mx.int32
        assert got.tolist() == want


def test_fix_matches_cpu_reference_scalar_arrays(installed):
    for off in (mx.array(48), mx.array([48])):
        x = _rand((4, 8, 1, D))
        fixed = _rope(mx.fast.rope, x, off)
        cpu = _rope(rbf._orig_rope, x, off, stream=mx.cpu)
        mx.eval(fixed, cpu)
        assert mx.allclose(fixed, cpu, atol=1e-4).item(), off


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
