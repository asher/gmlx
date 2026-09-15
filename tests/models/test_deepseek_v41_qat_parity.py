"""DeepSeek-V4.1-Flash QAT round-trips against the reference kernels.

The oracle is ``inference/kernel.py``: ``act_quant_kernel`` and
``fp4_quant_kernel``, ported here in numpy at the three geometries the
reference calls them with (``inference/model.py``):

  Attention._window_kv  act_quant(kv, 32, "ue8m0", e8m0, inplace)
  Attention._compress_kv fp4_act_quant(latent, 16, inplace, e4m3 scale)
  Indexer.forward       fp4_act_quant(q/k, 32, inplace)

The scale format is the trap: ``scale_fmt = "ue8m0"`` is not None, so the
FP8 scale rounds up to a power of two, and the E4M3 latent scale does not.
An amax floor differs per geometry as well.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.models.deepseek_v41.model import (
    _indexer_qat,
    _kv_qat,
    _latent_qat,
)

_E4M3_MAX = 448.0
_FP4_MAX = 6.0


def _round_ieee(a: np.ndarray, least_exp: int, most_exp: int,
                mantissa_bits: int, cap: float) -> np.ndarray:
    """Round magnitudes onto a float grid, ties to even mantissa.

    Below ``least_exp`` the format is subnormal and keeps that exponent's
    step rather than getting a finer grid, which is where a naive port
    goes wrong.
    """
    lo = 2.0 ** least_exp
    exp = np.clip(np.floor(np.log2(np.maximum(a, lo))), least_exp, most_exp)
    step = np.exp2(exp - mantissa_bits)
    return np.clip(np.round(a / step) * step, 0.0, cap)


def _e4m3(x: np.ndarray) -> np.ndarray:
    """E4M3FN: least normal exponent -6, 3 mantissa bits, max 448."""
    return np.sign(x) * _round_ieee(np.abs(x), -6, 8, 3, _E4M3_MAX)


def _e2m1(x: np.ndarray) -> np.ndarray:
    """E2M1: least normal exponent 0, 1 mantissa bit, grid
    {0, .5, 1, 1.5, 2, 3, 4, 6}."""
    return np.sign(x) * _round_ieee(np.abs(x), 0, 2, 1, _FP4_MAX)


def _blocks(x: np.ndarray, block: int) -> np.ndarray:
    return x.reshape(*x.shape[:-1], -1, block)


def _ref_act_quant(x: np.ndarray, block: int) -> np.ndarray:
    """``act_quant_kernel`` with ``round_scale=True`` and ``inplace=True``."""
    v = _blocks(x.astype(np.float64), block)
    amax = np.maximum(np.abs(v).max(axis=-1, keepdims=True), 1e-4)
    scale = np.exp2(np.ceil(np.log2(amax / _E4M3_MAX)))
    q = _e4m3(np.clip(v / scale, -_E4M3_MAX, _E4M3_MAX))
    return (q * scale).reshape(x.shape)


def _ref_fp4_quant(x: np.ndarray, block: int, e4m3_scale: bool) -> np.ndarray:
    """``fp4_quant_kernel`` with ``inplace=True``, either scale format."""
    v = _blocks(x.astype(np.float64), block)
    amax = np.abs(v).max(axis=-1, keepdims=True)
    if e4m3_scale:
        amax = np.maximum(amax, _FP4_MAX * 2.0**-9)
        scale = _e4m3(amax / _FP4_MAX)
    else:
        amax = np.maximum(amax, _FP4_MAX * 2.0**-126)
        scale = np.exp2(np.ceil(np.log2(amax / _FP4_MAX)))
    q = _e2m1(np.clip(v / scale, -_FP4_MAX, _FP4_MAX))
    return (q * scale).reshape(x.shape)


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 7, 512)).astype(np.float32) * 3.0
    x[0, 0, :32] = 0.0                     # an all-zero group hits the floor
    x[0, 1, :32] *= 1e-6                   # and a tiny one
    # Large, but with amax/6 inside e4m3: past 448 the reference scale cast
    # is NaN, so no geometry has a defined answer there.
    x[0, 2, :32] *= np.float32(2600.0) / np.abs(x[0, 2, :32]).max()
    # Exact half-way values, so a tie-break difference cannot hide.
    x[1, 0, :32] = np.float32(2.0**-6) * np.arange(1, 33, dtype=np.float32)
    x[1, 1, :32] = np.float32(0.5) * (2 * np.arange(32, dtype=np.float32) + 1)
    return x


def test_window_kv_is_fp8_block_32_with_a_power_of_two_scale(sample):
    got = np.array(_kv_qat(mx.array(sample)), dtype=np.float64)
    assert np.array_equal(got, _ref_act_quant(sample, 32))


def test_indexer_is_fp4_block_32_with_a_power_of_two_scale(sample):
    got = np.array(_indexer_qat(mx.array(sample)), dtype=np.float64)
    assert np.array_equal(got, _ref_fp4_quant(sample, 32, e4m3_scale=False))


def test_compressed_latent_is_fp4_groups_of_16_with_an_e4m3_scale(sample):
    got = np.array(_latent_qat(mx.array(sample)), dtype=np.float64)
    assert np.array_equal(got, _ref_fp4_quant(sample, 16, e4m3_scale=True))


def test_the_three_geometries_are_not_interchangeable(sample):
    """A guard on the test itself: if two of the reference geometries
    agreed, a wrong wiring would pass unnoticed."""
    kv = _ref_act_quant(sample, 32)
    idx = _ref_fp4_quant(sample, 32, e4m3_scale=False)
    lat = _ref_fp4_quant(sample, 16, e4m3_scale=True)
    assert not np.array_equal(kv, idx)
    assert not np.array_equal(idx, lat)


def test_qat_switches_off_together(monkeypatch, sample):
    monkeypatch.setenv("GMLX_DS41_QAT", "0")
    x = mx.array(sample)
    for fn in (_kv_qat, _indexer_qat, _latent_qat):
        assert np.array_equal(np.array(fn(x)), sample)
