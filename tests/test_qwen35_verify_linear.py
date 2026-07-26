"""Owned verify-linear family: upstream drift tripwires + wrapper parity.

The verbatim copies must stay byte-identical to the pinned mlx-vlm
release (a release upgrade that changes a body fails here, loudly).
The gmlx wrappers must reproduce the patched-global chain the stock
fallback used to compose: the bf16 GEMV-ext lever first, the verbatim
upstream dispatcher for everything else.
"""

import ast
import inspect
import os
import textwrap

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.qwen3_5 import language as _L

from gmlx import gdn_patches as gp
from gmlx import qwen35_verify_linear as vl

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="the verify GEMV kernels are Metal-only")

_COPIES = (
    "_use_target_verify_dense",
    "_target_verify_weight",
    "_target_verify_qlinear_header",
    "_target_verify_qmv_kernel",
    "_target_verify_qargmax_kernel",
    "_can_target_verify_quantized",
    "_target_verify_quantized_linear",
    "_decode_quantized_linears_fused",
    "_target_verify_quantized_argmax",
    "_target_verify_timewise",
    "_target_verify_singletons",
    "_target_verify_linear",
    "_target_verify_linears",
    "_target_verify_embedding_as_linear",
)


def _norm(fn):
    return [
        line.rstrip()
        for line in textwrap.dedent(inspect.getsource(fn)).splitlines()
        if line.strip()
    ]


def _assign_source(mod, name):
    src = inspect.getsource(mod)
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found at top level of {mod.__name__}")


def test_copies_match_upstream_source():
    for name in _COPIES:
        assert _norm(getattr(vl, name)) == _norm(getattr(_L, name)), (
            f"owned copy of {name} drifted from the upstream original"
        )


def test_gemv_kernel_assignment_matches_upstream_source():
    """The module-level GEMV kernel build (Metal source included) is a
    byte-identical copy of the upstream assignment."""
    assert _assign_source(vl, "_TARGET_VERIFY_GEMV") == _assign_source(
        _L, "_TARGET_VERIFY_GEMV"
    )


# ---------------------------------------------------------------------------
# wrapper parity vs the patched-global oracle
# ---------------------------------------------------------------------------


@pytest.fixture
def _bf16_oracle():
    """Install the bf16 verify-linear patch on the upstream global and
    restore it afterwards (the patch itself has no uninstall)."""
    saved_fn = _L._target_verify_linear
    saved_flag = gp._BF16_VERIFY_LINEAR_PATCHED
    gp._BF16_VERIFY_LINEAR_PATCHED = False
    gp._patch_bf16_verify_linear()
    yield
    _L._target_verify_linear = saved_fn
    gp._BF16_VERIFY_LINEAR_PATCHED = saved_flag


def _linears():
    mx.random.seed(5)
    plain = nn.Linear(64, 128, bias=False)
    plain.weight = plain.weight.astype(mx.bfloat16)
    biased = nn.Linear(64, 128, bias=True)
    biased.weight = biased.weight.astype(mx.bfloat16)
    biased.bias = biased.bias.astype(mx.bfloat16)
    quant = nn.QuantizedLinear(64, 128, bias=False, group_size=32, bits=4)
    return plain, biased, quant


@_NEEDS_GPU
def test_verify_linear_matches_patched_oracle(_bf16_oracle):
    plain, biased, quant = _linears()
    mx.random.seed(7)
    xv = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)  # verify-shaped
    xd = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)  # decode-shaped
    for linear in (plain, biased, quant):
        for x in (xv, xd):
            for tv in (True, False):
                want = _L._target_verify_linear(linear, x, tv)
                got = vl.verify_linear(linear, x, tv)
                mx.eval(want, got)
                assert mx.array_equal(want, got).item(), (
                    type(linear).__name__, x.shape, tv,
                )


@_NEEDS_GPU
def test_verify_linears_matches_patched_oracle(_bf16_oracle):
    plain, biased, quant = _linears()
    mx.random.seed(9)
    xv = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)
    xd = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
    q4 = tuple(
        nn.QuantizedLinear(64, 128, bias=True, group_size=32, bits=4)
        for _ in range(4)
    )
    for linears, x, tv in (
        ((plain, biased), xv, True),
        ((plain, quant), xv, True),
        ((plain, biased), xv, False),
        (q4, xd, False),  # fused decode concat route
        ((plain, quant), xd, True),  # non-verify-shaped falls through
    ):
        want = _L._target_verify_linears(linears, x, tv)
        got = vl.verify_linears(linears, x, tv)
        mx.eval(*want, *got)
        assert len(want) == len(got)
        for a, b in zip(want, got):
            assert mx.array_equal(a, b).item(), (x.shape, tv)


@_NEEDS_GPU
def test_bf16_lever_engages(monkeypatch):
    """The wrapper claims verify-shaped non-quantized linears on the
    GEMV-ext kernel exactly where the old patch did."""
    plain, _biased, quant = _linears()
    calls = {"n": 0}
    inner = vl._f16_head_gemv

    def counting(x, w):
        calls["n"] += 1
        return inner(x, w)

    monkeypatch.setattr(vl, "_f16_head_gemv", counting)
    mx.random.seed(3)
    xv = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)
    mx.eval(vl.verify_linear(plain, xv, True))
    assert calls["n"] == 1
    mx.eval(vl.verify_linear(quant, xv, True))  # quantized: never claimed
    assert calls["n"] == 1
    mx.eval(vl.verify_linear(plain, xv, False))  # not verify: never claimed
    assert calls["n"] == 1
