#!/usr/bin/env python3
"""Numerics for the unified ragged decode plan (gmlx.spec.ragged_decode).

Stock ``_qwen3_5_ragged_decode_attention`` requires every row of a batch to
land in the same sdpa-vector plan bucket and returns None otherwise, dropping
mixed-depth batches onto a per-row python loop. The seam claims the
``k_size``-derived plan is valid for every row because both metal kernels
partition the padded ``k_size`` and mask per-row via ``pads``.

That claim is what these tests gate. The reference is per-row fused SDPA over
each row's unpadded keys -- the same answer computed the slow, obvious way.

Same-bucket batches must stay BIT-EXACT: the seam is supposed to be a no-op
there (it keeps the stock per-row plan when the buckets agree), so any drift
means it changed a path it was never meant to touch. Straddling batches are
newly claimed rather than declined, so they only have to land within bf16
rounding of the reference.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

# The numerics tests dispatch mx.fast.metal_kernel, which raises on the CPU
# device. conftest pins the device to CPU under KQUANT_FORCE_CPU (CI). Gate on
# the resolved device rather than the env var so a machine with no Metal at all
# is covered too. Applied per-test, not module-wide: the decline test asserts a
# guard rail that returns before any dispatch, so it must keep running on CPU.
_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="ragged decode numerics dispatch metal kernels; needs the GPU device",
)

D_SIZE = 256
KV_HEADS = 2
Q_HEADS = 16
DTYPE = mx.bfloat16
# bf16 carries a 7-bit mantissa (eps ~= 0.0078); one ulp of headroom.
BF16_ULP = 0.008


def _lang():
    return pytest.importorskip("mlx_vlm.models.qwen3_5.language")


def _reference(q, k, v, pads, scale):
    """Per-row fused SDPA over that row's unpadded keys."""
    outs = []
    for i, p in enumerate(pads):
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[i : i + 1], k[i : i + 1, :, p:], v[i : i + 1, :, p:],
                scale=scale, mask=None,
            )
        )
    return mx.concatenate(outs, axis=0)


def _run(k_size, pads):
    lang = _lang()
    B = len(pads)
    mx.random.seed(0)
    q = mx.random.normal((B, Q_HEADS, 1, D_SIZE)).astype(DTYPE)
    k = mx.random.normal((B, KV_HEADS, k_size, D_SIZE)).astype(DTYPE)
    v = mx.random.normal((B, KV_HEADS, k_size, D_SIZE)).astype(DTYPE)
    mx.eval(q, k, v)
    scale = D_SIZE**-0.5
    got = lang._qwen3_5_ragged_decode_attention(q, k, v, pads, scale)
    if got is None:
        return None, None
    ref = _reference(q, k, v, pads, scale)
    mx.eval(got, ref)
    return got, ref


def _rel_err(got, ref):
    err = mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item()
    mag = mx.max(mx.abs(ref.astype(mx.float32))).item()
    return err / max(mag, 1e-6)


@pytest.fixture
def unified():
    if not mx.metal.is_available():
        pytest.skip("ragged decode kernels are Metal-only")
    import gmlx.spec.ragged_decode as ragged_decode

    ragged_decode.install_unified_ragged_plan()
    return ragged_decode


@pytest.mark.parametrize(
    "k_size,pads",
    [
        (4096, [0, 0, 0]),            # uniform, unpadded
        (14772, [248, 217, 0]),       # ragged pads, one bucket (live d14k c3 shape)
        (4096, [0, 1000, 2000]),      # wide pad spread, still one bucket
    ],
)
@_NEEDS_GPU
def test_same_bucket_is_bit_exact(unified, k_size, pads):
    """Buckets agree -> seam keeps the stock plan -> no drift at all."""
    got, ref = _run(k_size, pads)
    assert got is not None, "kernel declined a same-bucket batch"
    assert mx.array_equal(got, ref), "same-bucket ragged decode drifted from per-row SDPA"


@pytest.mark.parametrize(
    "k_size,pads",
    [
        (8300, [0, 200, 300]),        # straddles the 8k plan boundary
        (32900, [0, 400, 900]),       # straddles the 32k plan boundary
    ],
)
@_NEEDS_GPU
def test_straddling_buckets_claimed_and_accurate(unified, k_size, pads):
    """Newly claimed (stock declines these); must match within bf16 rounding."""
    got, ref = _run(k_size, pads)
    assert got is not None, "unified plan declined a straddling batch"
    assert _rel_err(got, ref) < BF16_ULP


@_NEEDS_GPU
def test_pads_are_honored(unified):
    """A padded row must ignore its pad columns entirely: poisoning the pad
    region cannot change the output. This is the masking property a position
    bug would break."""
    lang = _lang()
    k_size, pads = 4096, [512, 0]
    B = len(pads)
    mx.random.seed(1)
    q = mx.random.normal((B, Q_HEADS, 1, D_SIZE)).astype(DTYPE)
    k = mx.random.normal((B, KV_HEADS, k_size, D_SIZE)).astype(DTYPE)
    v = mx.random.normal((B, KV_HEADS, k_size, D_SIZE)).astype(DTYPE)
    mx.eval(q, k, v)
    scale = D_SIZE**-0.5
    clean = lang._qwen3_5_ragged_decode_attention(q, k, v, pads, scale)
    assert clean is not None

    poison = mx.concatenate(
        [mx.full((1, KV_HEADS, pads[0], D_SIZE), 40.0, dtype=DTYPE),
         k[0:1, :, pads[0]:]], axis=2)
    k2 = mx.concatenate([poison, k[1:]], axis=0)
    dirty = lang._qwen3_5_ragged_decode_attention(q, k2, v, pads, scale)
    mx.eval(clean, dirty)
    assert mx.array_equal(clean, dirty), "pad columns leaked into the output"


def test_declines_unsupported_head_dim(unified):
    """Guard rails still hold: an unsupported d_size defers instead of
    silently computing something else."""
    lang = _lang()
    mx.random.seed(2)
    d = 80  # not in (64, 96, 128, 256)
    q = mx.random.normal((2, Q_HEADS, 1, d)).astype(DTYPE)
    k = mx.random.normal((2, KV_HEADS, 512, d)).astype(DTYPE)
    v = mx.random.normal((2, KV_HEADS, 512, d)).astype(DTYPE)
    mx.eval(q, k, v)
    assert lang._qwen3_5_ragged_decode_attention(q, k, v, [0, 0], d**-0.5) is None


def test_pre_m3_rebinds_even_with_plan_env_off(monkeypatch):
    """Pre-M3 the rebind is a threadgroup guard, not an optimization:
    GMLX_RAGGED_UNIFIED_PLAN=0 must not leave the stock 1024-thread
    kernels bound."""
    lang = _lang()
    import gmlx.spec.ragged_decode as ragged_decode
    from gmlx.models.qwen35 import attn

    monkeypatch.setattr(attn, "_wide_threadgroups_ok", lambda: False)
    monkeypatch.setenv("GMLX_RAGGED_UNIFIED_PLAN", "0")
    orig = lang._qwen3_5_ragged_decode_attention
    try:
        lang._qwen3_5_ragged_decode_attention = object()
        ragged_decode.install_unified_ragged_plan()
        assert (lang._qwen3_5_ragged_decode_attention
                is attn.ragged_decode_attention)
    finally:
        lang._qwen3_5_ragged_decode_attention = orig


def test_guard_rebinds_pre_m3_and_noops_on_wide(monkeypatch):
    lang = _lang()
    import gmlx.spec.ragged_decode as ragged_decode
    from gmlx.models.qwen35 import attn

    monkeypatch.delenv("GMLX_RAGGED_UNIFIED_PLAN", raising=False)
    orig = lang._qwen3_5_ragged_decode_attention
    sentinel = object()
    try:
        lang._qwen3_5_ragged_decode_attention = sentinel
        monkeypatch.setattr(attn, "_wide_threadgroups_ok", lambda: True)
        ragged_decode.install_pre_m3_ragged_guard()
        assert lang._qwen3_5_ragged_decode_attention is sentinel

        monkeypatch.setattr(attn, "_wide_threadgroups_ok", lambda: False)
        ragged_decode.install_pre_m3_ragged_guard()
        assert (lang._qwen3_5_ragged_decode_attention
                is attn.ragged_decode_attention)
    finally:
        lang._qwen3_5_ragged_decode_attention = orig
