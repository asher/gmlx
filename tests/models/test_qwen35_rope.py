"""Owned MRoPE apply chain: upstream drift tripwires + mirror parity.

The verbatim copies must stay byte-identical to the pinned mlx-vlm
release. The two free-function mirrors must be exact against the stock
``MRoPERotaryEmbedding`` forwards on every route the qwen3.5 family
reaches: fused Metal apply (2D and 3D position ids), the cos/sin
fallback, partial rotary dims (pass-through tail), and both dtypes.
"""

import ast
import inspect
import os
import textwrap

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.rope_utils")

from mlx_vlm.models import rope_utils as _R
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

import gmlx.models.qwen35.rope as rp

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="the fused MRoPE apply is Metal-only")

_COPIES = (
    "_interleaved_position_selector",
    "_chunked_position_selector",
    "_selected_mrope_freqs",
    "mrope_position_selector",
    "_selects_frequency_by_position",
    "_is_sectioned_style",
    "_has_mrope_apply_selector",
    "_uses_even_odd_pairing",
    "_needs_even_odd_layout",
    "_pairing_for_style",
    "_mrope_apply_kernel",
    "_mrope_apply_cos_sin",
    "_mrope_apply",
    "_fast_mrope_apply",
    "_compiled_mrope_apply",
    "get_mrope_section",
    "compute_inv_freq",
    "_apply_selected_mrope_frequency_layout",
    "apply_mrope_frequency_layout",
    "compute_mrope_frequencies",
    "rotate_half",
    "rotate_half_even_odd",
    "_apply_rotary_embedding",
    "_apply_interleaved_rotary_pos_emb_axis1",
    "apply_multimodal_rotary_pos_emb",
)

_CONSTANTS = ("_HAS_METAL", "_HALF_SPLIT", "_EVEN_ODD", "_HALF_COS", "_FULL_COS")


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
        assert _norm(getattr(rp, name)) == _norm(getattr(_R, name)), (
            f"owned copy of {name} drifted from the upstream original"
        )


def test_constants_match_upstream_source():
    for name in _CONSTANTS:
        assert _assign_source(rp, name) == _assign_source(_R, name), (
            f"owned constant {name} drifted from the upstream original"
        )


# ---------------------------------------------------------------------------
# free-function mirror parity vs the stock rotary forwards
# ---------------------------------------------------------------------------


def _emb(dim=16):
    return Qwen3_5RotaryEmbedding(
        dim, max_position_embeddings=2048, base=100000, mrope_section=[2, 1, 1]
    )


def _qk(dtype, head=32):
    mx.random.seed(3)
    q = mx.random.normal((2, 4, 5, head)).astype(dtype)
    k = mx.random.normal((2, 2, 5, head)).astype(dtype)
    return q, k


def _positions(ndim):
    if ndim == 2:
        return mx.tile(mx.arange(5)[None, :], (2, 1))
    return mx.tile(mx.arange(5)[None, None, :], (3, 2, 1))


@_NEEDS_GPU
@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("dtype", (mx.bfloat16, mx.float32))
def test_fused_apply_parity(ndim, dtype):
    emb = _emb()
    q, k = _qk(dtype)
    pos = _positions(ndim)
    qs, ks = emb.apply_rotary(q, k, pos)
    qo, ko = rp.rope_apply_rotary(emb, q, k, pos)
    mx.eval(qs, ks, qo, ko)
    assert mx.array_equal(qs, qo).item() and mx.array_equal(ks, ko).item()


@pytest.mark.parametrize("ndim", (2, 3))
def test_cos_sin_and_unfused_apply_parity(ndim):
    emb = _emb()
    q, k = _qk(mx.bfloat16)
    pos = _positions(ndim)
    cs, ss = emb(k, pos)
    co, so = rp.rope_cos_sin(emb, k, pos)
    mx.eval(cs, ss, co, so)
    assert mx.array_equal(cs, co).item() and mx.array_equal(ss, so).item()

    # Force the non-fused route on both arms; the mirrors must agree
    # through the owned interleaved apply.
    emb.fused_apply = False
    try:
        qs, ks = emb.apply_rotary(q, k, pos)
        qo, ko = rp.rope_apply_rotary(emb, q, k, pos)
        mx.eval(qs, ks, qo, ko)
        assert mx.array_equal(qs, qo).item() and mx.array_equal(ks, ko).item()
    finally:
        emb.fused_apply = True


@_NEEDS_GPU
def test_partial_rotary_pass_through_parity():
    # rotary dim 8 under head dim 32: the tail must pass through
    # untouched on the fused kernel route.
    emb = _emb(dim=8)
    q, k = _qk(mx.bfloat16)
    pos = _positions(3)
    qs, ks = emb.apply_rotary(q, k, pos)
    qo, ko = rp.rope_apply_rotary(emb, q, k, pos)
    mx.eval(qs, ks, qo, ko)
    assert mx.array_equal(qs, qo).item() and mx.array_equal(ks, ko).item()
    assert mx.array_equal(qs[..., 8:], q[..., 8:]).item()


@_NEEDS_GPU
def test_owned_memo_is_separate():
    """The owned mirror memoizes its compiled apply under an owned attr,
    so a stock-compiled entry can never masquerade as the owned path."""
    emb = _emb()
    q, k = _qk(mx.bfloat16)
    rp.rope_apply_rotary(emb, q, k, _positions(3))
    assert len(emb._gmlx_compiled_apply) == 1
    assert len(emb._compiled_apply) == 0, (
        "owned mirror wrote into the stock compiled-apply memo"
    )


@_NEEDS_GPU
def test_fused_route_is_active_on_the_gpu():
    assert rp.fused_rope_active(_emb())


def test_cpu_default_device_takes_the_cos_sin_route(monkeypatch):
    """The fused apply is a Metal kernel: stock MLX raises when it is
    dispatched while the default device is the CPU. The mirror routes
    around it there and lands on the cos/sin apply."""
    emb = _emb()
    if not emb.fused_apply:
        pytest.skip("no Metal device")

    def _no_kernel(*args, **kwargs):
        raise AssertionError("fused apply dispatched on the CPU device")

    monkeypatch.setattr(rp, "_compiled_mrope_apply", _no_kernel)
    q, k = _qk(mx.bfloat16)
    pos = _positions(3)
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert not rp.fused_rope_active(emb)
        qo, ko = rp.rope_apply_rotary(emb, q, k, pos)
        emb.fused_apply = False
        qs, ks = emb.apply_rotary(q, k, pos)
        mx.eval(qs, ks, qo, ko)
    finally:
        emb.fused_apply = True
        mx.set_default_device(prev)
    assert mx.array_equal(qs, qo).item() and mx.array_equal(ks, ko).item()
