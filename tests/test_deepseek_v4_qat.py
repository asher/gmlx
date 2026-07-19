#!/usr/bin/env python3
"""DeepSeek V4 Flash QAT round-trip emulation vs ds4's C semantics.

The V4 Flash checkpoint was quantization-aware-trained with fp8/fp4 round
trips on the KV latent, the compressor pool rows, and the indexer
activations; ds4 (the parity reference engine - llama.cpp has no deepseek4
support) applies them at inference (ds4.c:2440-2620) and matches the
official API logits. gmlx.deepseek_v4_model reproduces them in pure MLX
- a deliberate deviation from upstream mlx-lm PR #1192, which omits them.
These tests pin the emulation to hand-computed vectors of the C algorithms:

  fp8-E4M3 (KV nope dims + pooled rows, blocks of 64): per-block scale
    2^ceil(log2(amax/448)), amax floor 1e-4, clip +-448, RTNE mantissa.
  fp4-E2M1 (indexer q/k after Hadamard-128, blocks of 32): value set
    {0, .5, 1, 1.5, 2, 3, 4, 6} x sign, scale 2^ceil(log2(amax/6)),
    ties broken to the even value INDEX (ds4's tie-break).

No GGUF needed. The hand-vector rounding tests run anywhere; the
roundtrip-helper tests dispatch GPU-only kq ops and skip without Metal.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from gmlx.deepseek_v4_model import (  # noqa: E402
    _e2m1_round,
    _e4m3_round,
    _fp4_e2m1_roundtrip,
    _fp8_e4m3_roundtrip,
    _indexer_qat_roundtrip,
    _kv_qat_roundtrip,
)

# The roundtrip helpers dispatch mlx_kquant's dsa_kv_qat / dsa_indexer_qat,
# which are GPU-only. The pure rounding tests below run anywhere.
_NEEDS_DSA_OPS = pytest.mark.skipif(
    not mx.metal.is_available() or bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="mlx_kquant dsa qat ops are GPU-only")


def _f(x):
    return mx.array(x, dtype=mx.float32)


def test_e4m3_round_rtne_mantissa():
    # 3-bit mantissa: step 2^(e-3). At e=0 (values in [1,2)) the step is
    # 0.125; 1.0625 = 8.5 steps -> RTNE to 8 (1.0), 1.1875 = 9.5 -> 10 (1.25).
    got = _e4m3_round(_f([1.0, 1.0625, 1.1875, 1.25, -1.0625]))
    assert got.tolist() == [1.0, 1.0, 1.25, 1.25, -1.0]
    # Representable values pass through exactly across exponents.
    exact = [0.5, 0.5625, 2.0, 3.5, 448.0, -448.0, 0.015625]
    assert _e4m3_round(_f(exact)).tolist() == exact


def test_e2m1_round_value_buckets_ties_to_even_index():
    # ds4's tie-break is to the even value INDEX in {0,.5,1,1.5,2,3,4,6}:
    # 0.25 -> 0 (idx 0), 0.75 -> 1 (idx 2), 1.75 -> 2 (idx 4), 2.5 -> 2,
    # 3.5 -> 4 (idx 6), 5.0 -> 4.
    vals = [0.0, 0.25, 0.26, 0.74, 0.75, 1.25, 1.26, 1.74, 1.75,
            2.5, 2.51, 3.49, 3.5, 5.0, 5.01, 6.0, 7.0]
    want = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0,
            2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0]
    assert _e2m1_round(_f(vals)).tolist() == want
    # Sign symmetry.
    assert _e2m1_round(_f([-0.75, -1.75, -3.5])).tolist() == [-1.0, -2.0, -4.0]


def test_fp8_roundtrip_block_scale_pow2_ceil():
    # One block of 64 with amax 100: scale = 2^ceil(log2(100/448)) = 2^-2.
    # 100/0.25 = 400 (representable: 400 = 25*2^4, 6 mantissa steps at e=8
    # are 32 apart... 400/32 = 12.5 -> RTNE 12 -> 384*0.25... hand-check
    # instead on values chosen to be exact at scale 0.25: 448*0.25 = 112.
    x = mx.zeros((64,), dtype=mx.float32)
    x[0] = 100.0
    x[1] = 112.0    # 448 * 0.25 -> exact at the block scale
    x[2] = 0.03125  # 0.125 * 0.25 -> exact small value
    y = _fp8_e4m3_roundtrip(x[None], block=64)[0]
    assert y[1].item() == 112.0
    assert y[2].item() == 0.03125
    # 100/0.25 = 400: e=floor(log2 400)=8, step 2^5=32, 400/32=12.5 -> RTNE 12
    # -> 384 * 0.25 = 96.
    assert y[0].item() == 96.0
    # Idempotence: re-quantizing quantized data is a no-op (same scale - the
    # amax 112 survives the first pass).
    z = _fp8_e4m3_roundtrip(y[None], block=64)[0]
    assert mx.array_equal(z, y)


def test_fp8_per_block_scales_independent():
    # Two blocks with very different amax must scale independently: a value
    # that is exact under its own block's scale round-trips even when the
    # other block's amax differs by orders of magnitude.
    x = mx.zeros((128,), dtype=mx.float32)
    x[0] = 448.0      # block 0: scale 1
    x[64] = 3.5       # block 1: amax 3.5 -> scale 2^ceil(log2(3.5/448))=2^-7
    x[65] = 0.4375    # 56 * 2^-7 -> exact in block 1
    y = _fp8_e4m3_roundtrip(x[None], block=64)[0]
    assert y[0].item() == 448.0
    assert y[64].item() == 3.5
    assert y[65].item() == 0.4375


def test_fp4_roundtrip_scale_and_values():
    # Block of 32, amax 6 -> scale 1: inputs snap to the E2M1 value set.
    x = mx.zeros((32,), dtype=mx.float32)
    x[0], x[1], x[2], x[3] = 6.0, 2.51, -0.75, 0.2
    y = _fp4_e2m1_roundtrip(x[None], block=32)[0]
    assert y[0].item() == 6.0
    assert y[1].item() == 3.0
    assert y[2].item() == -1.0
    assert y[3].item() == 0.0   # 0.2 <= 0.25 -> 0
    # amax 12 -> scale 2: the set doubles.
    x2 = mx.zeros((32,), dtype=mx.float32)
    x2[0], x2[1] = 12.0, 5.1
    y2 = _fp4_e2m1_roundtrip(x2[None], block=32)[0]
    assert y2[0].item() == 12.0
    assert y2[1].item() == 6.0  # 5.1/2 = 2.55 -> 3 -> *2
    # Idempotence.
    assert mx.array_equal(_fp4_e2m1_roundtrip(y[None], block=32)[0], y)


@_NEEDS_DSA_OPS
def test_kv_qat_quantizes_nope_keeps_rope_tail_f16():
    # The KV latent round-trip applies fp8 to the nope dims only; the rope
    # tail is untouched by fp8 but the WHOLE row then rounds through f16.
    # 1 + 2^-9 is f16-exact but not fp8-representable (rounds to 1.0).
    n_rot, width = 8, 72                     # 64 nope (one fp8 block) + 8 rot
    kv = mx.full((1, 1, 1, width), 1.001953125, dtype=mx.float32)
    out = _kv_qat_roundtrip(kv, n_rot)
    nope, rot = out[..., :-n_rot], out[..., -n_rot:]
    assert mx.all(nope == 1.0).item()               # fp8'd
    assert mx.all(rot == 1.001953125).item()        # f16 round only
    assert out.dtype == mx.float32                  # dtype preserved


@_NEEDS_DSA_OPS
def test_indexer_qat_output_stays_in_hadamard_domain():
    # ds4 (dsv4_indexer_qat_row_inplace_cpu) Hadamard-transforms the row
    # (scale 1/sqrt(128)) and fp4-quantizes IN the transformed domain - no
    # inverse transform. That's sound because both indexer q and pooled k
    # rows get the same orthonormal transform, so dot-product scores are
    # preserved. Pin it with an exact vector: e0 * 6*sqrt(128) transforms to
    # a constant row of 6.0 (H's first column is all ones), which is on the
    # fp4 grid at block scale 1 -> the output must be exactly all 6.0. An
    # implementation that transformed back would return ~e0 * 6*sqrt(128).
    x = mx.zeros((1, 128), dtype=mx.float32)
    x[0, 0] = 6.0 * 128.0 ** 0.5
    y = _indexer_qat_roundtrip(x)
    assert y.shape == x.shape and y.dtype == mx.float32
    assert mx.all(y == 6.0).item()


@_NEEDS_DSA_OPS
def test_indexer_qat_preserves_dot_products_approximately():
    # The Hadamard/sqrt(n) transform is orthonormal, so q.k survives it
    # exactly; fp4 then adds bounded quantization noise. Norms must come
    # through within coarse-fp4 tolerance (and not be the identity map).
    mx.random.seed(7)
    x = mx.random.normal((4, 128)).astype(mx.float32)
    y = _indexer_qat_roundtrip(x)
    nx = mx.linalg.norm(x, axis=-1)
    ny = mx.linalg.norm(y, axis=-1)
    assert mx.all(mx.abs(ny - nx) / nx < 0.25).item()
    assert not mx.allclose(y, x, atol=1e-3, rtol=0).item()


def test_fp8_amax_floor_zero_block_stays_zero():
    # All-zero blocks hit the amax floor (1e-4) - must not NaN/Inf.
    y = _fp8_e4m3_roundtrip(mx.zeros((1, 64)), block=64)
    assert mx.all(y == 0.0).item() and not mx.any(mx.isnan(y)).item()
    y4 = _fp4_e2m1_roundtrip(mx.zeros((1, 32)), block=32)
    assert mx.all(y4 == 0.0).item() and not mx.any(mx.isnan(y4)).item()


# ---------------------------------------------------------------------------
# Bit-identity pins vs the ORIGINAL exponent-bit implementations.
#
# The live functions were reworked to be mx.compile-traceable (table-lookup
# exp2 + shapeless-compiled per-block cores). These frozen copies of the
# original view()-based forms are the ground truth: every output must match
# bit-for-bit over adversarial inputs (ties, clamp bounds, amax floors,
# subnormal-scale region, mixed dtypes, non-multiple block widths).


def _ref_exp2i(e):
    return ((e.astype(mx.int32) + 127) << 23).view(mx.float32)


def _ref_e4m3_round(v):
    s = mx.sign(v)
    a = mx.abs(v)
    e = mx.floor(mx.log2(mx.maximum(a, 2.0**-9)))
    e = mx.clip(e, -6.0, 8.0)
    q = _ref_exp2i(e - 3.0)
    return s * mx.round(a / q) * q


def _ref_e2m1_round(v):
    s = mx.sign(v)
    a = mx.abs(v)
    q = mx.where(
        a <= 0.25, 0.0,
        mx.where(a < 0.75, 0.5,
        mx.where(a <= 1.25, 1.0,
        mx.where(a < 1.75, 1.5,
        mx.where(a <= 2.5, 2.0,
        mx.where(a < 3.5, 3.0,
        mx.where(a <= 5.0, 4.0, 6.0)))))))
    return s * q


def _ref_fp8_e4m3_roundtrip(x, block=64):
    orig_dtype = x.dtype
    if x.shape[-1] % block:
        block = x.shape[-1]
    v = mx.unflatten(x.astype(mx.float32), -1, (-1, block))
    amax = mx.maximum(mx.max(mx.abs(v), axis=-1, keepdims=True), 1e-4)
    scale = _ref_exp2i(mx.ceil(mx.log2(amax / 448.0)))
    v = _ref_e4m3_round(mx.clip(v / scale, -448.0, 448.0)) * scale
    return mx.flatten(v, -2).astype(orig_dtype)


def _ref_fp4_e2m1_roundtrip(x, block=32):
    orig_dtype = x.dtype
    if x.shape[-1] % block:
        block = x.shape[-1]
    v = mx.unflatten(x.astype(mx.float32), -1, (-1, block))
    amax = mx.maximum(
        mx.max(mx.abs(v), axis=-1, keepdims=True), 7.052966104933725e-38
    )
    scale = _ref_exp2i(mx.ceil(mx.log2(amax / 6.0)))
    v = _ref_e2m1_round(mx.clip(v / scale, -6.0, 6.0)) * scale
    return mx.flatten(v, -2).astype(orig_dtype)


def _ref_kv_qat_roundtrip(kv, n_rot):
    orig_dtype = kv.dtype
    nope, rot = mx.split(kv, [kv.shape[-1] - n_rot], axis=-1)
    kv = mx.concatenate([_ref_fp8_e4m3_roundtrip(nope), rot], axis=-1)
    return kv.astype(mx.float16).astype(orig_dtype)


def _ref_indexer_qat_roundtrip(x):
    orig_dtype = x.dtype
    v = mx.hadamard_transform(x.astype(mx.float32))
    return _ref_fp4_e2m1_roundtrip(v).astype(orig_dtype)


def _adversarial_flat(seed: int) -> mx.array:
    """Every value class the round-trips branch on, plus bulk random."""
    vals = []
    # E4M3 half-step ties and exact representables in every normal octave.
    for e in range(-6, 9):
        step = 2.0 ** (e - 3)
        vals += [(k + 0.5) * step for k in range(8, 16)]
        vals += [k * step for k in range(8, 16)]
    # Subnormal region (below 2^-6): step 2^-9, incl. half-step ties.
    vals += [k * 2.0**-9 for k in range(17)]
    vals += [(k + 0.5) * 2.0**-9 for k in range(16)]
    # E2M1 tie points at several block scales.
    for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        vals += [t * 2.0**p for p in (-3, 0, 4)]
    # Clamp bounds, amax floors (fp8 1e-4; fp4 FLT_MIN*6), zeros, overshoot.
    vals += [0.0, -0.0, 1e-5, 1e-4, 2e-4, 1e-38, 7.052966104933725e-38,
             448.0, 449.0, 6.0, 7.0, 1000.0, 1e30, 1e-30]
    vals += [-v for v in vals]
    base = mx.array(vals, dtype=mx.float32)
    mx.random.seed(seed)
    rnd = mx.random.normal((4096,)).astype(mx.float32) * 3.0
    wide = mx.exp(mx.random.normal((4096,)) * 8.0).astype(mx.float32)
    return mx.concatenate([base, rnd, wide])


def test_exp2i_bit_identity_full_range():
    from gmlx.deepseek_v4_model import _exp2i

    e = mx.arange(-126, 128).astype(mx.float32)
    assert mx.array_equal(_exp2i(e), _ref_exp2i(e)).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("width", [64, 128, 24])
def test_fp8_fp4_bit_identity_adversarial(dtype, width):
    flat = _adversarial_flat(seed=3)
    n = (flat.size // width) * width
    x = flat[:n].reshape(-1, width).astype(dtype)
    # inf (from downcasting 1e30 etc. to f16) is outside the round-trips'
    # contract -- the ds4 C reference is UB on inf too, and old/new impls
    # produce different garbage. Pin to the dtype's finite max instead; the
    # f32 run keeps the raw 1e30/1e-30 extremes.
    if dtype != mx.float32:
        finite = mx.isfinite(x.astype(mx.float32))
        x = mx.where(finite, x, mx.array(mx.finfo(dtype).max, dtype))
    assert mx.array_equal(
        _fp8_e4m3_roundtrip(x), _ref_fp8_e4m3_roundtrip(x)
    ).item()
    assert mx.array_equal(
        _fp4_e2m1_roundtrip(x), _ref_fp4_e2m1_roundtrip(x)
    ).item()


@_NEEDS_DSA_OPS
def test_kv_indexer_bit_identity():
    mx.random.seed(11)
    kv = (mx.random.normal((2, 1, 3, 72)) * 5.0).astype(mx.float32)
    assert mx.array_equal(
        _kv_qat_roundtrip(kv, 8), _ref_kv_qat_roundtrip(kv, 8)
    ).item()
    kvh = kv.astype(mx.bfloat16)
    assert mx.array_equal(
        _kv_qat_roundtrip(kvh, 8), _ref_kv_qat_roundtrip(kvh, 8)
    ).item()
    x = (mx.random.normal((4, 128)) * 4.0).astype(mx.float32)
    assert mx.array_equal(
        _indexer_qat_roundtrip(x), _ref_indexer_qat_roundtrip(x)
    ).item()
