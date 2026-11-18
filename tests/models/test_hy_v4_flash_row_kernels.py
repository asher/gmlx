"""Hy4-flash row kernels: the fused key row and query rows of a decode
step against the ops they replace (RMSNorm, rope, the join).

The kernels repeat the arithmetic of mx.fast.rms_norm and mx.fast.rope,
so the gate is equal bits.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.models.hy_v4_flash import row_kernels

LATENT, ROPE = row_kernels.LATENT, row_kernels.ROPE
HEADS, NOPE = 16, 128


def _gpu() -> bool:
    return (mx.default_device().type == mx.DeviceType.gpu
            and row_kernels._kernels() is not None)


def _bits(a: mx.array) -> np.ndarray:
    return np.array(a.view(mx.uint16))


def _norm(dtype) -> nn.RMSNorm:
    norm = nn.RMSNorm(LATENT, eps=1e-5)
    norm.weight = mx.random.normal((LATENT,)).astype(dtype)
    return norm


def _ops_kv(compressed, norm, rope, offset):
    B, L, _ = compressed.shape
    lat, pe = mx.split(compressed, [LATENT], axis=-1)
    pe = pe.reshape(B, L, 1, ROPE).transpose(0, 2, 1, 3)
    lat = mx.expand_dims(norm(lat), axis=1)
    return mx.concatenate([lat, rope(pe, offset)], axis=-1)


def _ops_q(emb, q_flat, rope, offset):
    B, H, L, _ = emb.shape
    q = q_flat.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
    _, pe = mx.split(q, [NOPE], axis=-1)
    return mx.concatenate([emb, rope(pe, offset)], axis=-1)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("theta", [1e7, 1e4])
@pytest.mark.parametrize("batch,length", [(1, 1), (1, 4), (2, 1), (3, 8)])
def test_the_rows_carry_the_ops_bits(dtype, theta, batch, length):
    if not _gpu():
        pytest.skip("Metal row kernels unavailable")
    mx.random.seed(11)
    norm = _norm(dtype)
    rope = nn.RoPE(ROPE, traditional=True, base=theta)
    for offset in (0, 1, 511, 16385, 131071, 1048577):
        # Rows of mixed size, one near zero where eps carries the norm.
        scale = mx.array([[[3.0]], [[1e-3]], [[40.0]]])[:batch]
        compressed = (mx.random.normal((batch, length, LATENT + ROPE))
                      * scale).astype(dtype)
        assert row_kernels.eligible(compressed, norm, rope, LATENT, ROPE)
        got = row_kernels.kv_row(compressed, norm, rope, offset)
        want = _ops_kv(compressed, norm, rope, offset)
        assert got.shape == want.shape and got.dtype == dtype
        np.testing.assert_array_equal(_bits(got), _bits(want))

        emb = mx.random.normal((batch, HEADS, length, LATENT)).astype(dtype)
        q_flat = mx.random.normal(
            (batch, length, HEADS * (NOPE + ROPE))).astype(dtype)
        got = row_kernels.q_row(emb, q_flat, rope, offset)
        want = _ops_q(emb, q_flat, rope, offset)
        assert got.shape == want.shape and got.dtype == dtype
        np.testing.assert_array_equal(_bits(got), _bits(want))


def test_the_kernels_leave_other_shapes_to_the_ops(monkeypatch):
    if not _gpu():
        pytest.skip("Metal row kernels unavailable")
    norm = _norm(mx.bfloat16)
    rope = nn.RoPE(ROPE, traditional=True, base=1e4)
    x = mx.zeros((1, 1, LATENT + ROPE), dtype=mx.bfloat16)
    assert row_kernels.eligible(x, norm, rope, LATENT, ROPE)
    # A prefill chunk, fp32 rows, another rank, a scaled or split-half rope.
    wide = mx.zeros((1, row_kernels.MAX_ROWS + 1, LATENT + ROPE),
                    dtype=mx.bfloat16)
    assert not row_kernels.eligible(wide, norm, rope, LATENT, ROPE)
    assert not row_kernels.eligible(
        x.astype(mx.float32), norm, rope, LATENT, ROPE)
    assert not row_kernels.eligible(x, norm, rope, 256, ROPE)
    assert not row_kernels.eligible(
        x, norm, nn.RoPE(ROPE, traditional=True, base=1e4, scale=0.5),
        LATENT, ROPE)
    assert not row_kernels.eligible(
        x, norm, nn.RoPE(ROPE, traditional=False, base=1e4), LATENT, ROPE)
    monkeypatch.setenv("GMLX_HY4_ROW_KERNEL", "0")
    assert not row_kernels.eligible(x, norm, rope, LATENT, ROPE)
