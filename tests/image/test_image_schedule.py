"""Sigma schedules and the Euler sampler: the one-rounding multiply-add
and the torch linspace replica against exact rational arithmetic, the
Z-Image schedule, Euler against float32 numpy, and seeded noise. CPU."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import mlx.core as mx

from gmlx.image import samplers, schedule

_F32 = np.float32


def _round_f32(x: Fraction) -> np.float32:
    """Round an exact value to the nearest float32, ties to even."""
    near = _F32(float(x))
    best = None
    for v in (np.nextafter(near, _F32(-np.inf)), near,
              np.nextafter(near, _F32(np.inf))):
        d = abs(Fraction(float(v)) - x)
        key = (d, int(np.array(v).view(np.uint32)) & 1)
        if best is None or key < best[0]:
            best = (key, v)
    return _F32(best[1])


def _exact_fma(a, b, c) -> np.float32:
    return _round_f32(Fraction(float(a)) * Fraction(float(b))
                      + Fraction(float(c)))


def test_fma_rounds_once_on_random_and_tie_inputs():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(4000).astype(_F32)
    b = rng.standard_normal(4000).astype(_F32)
    c = (rng.standard_normal(4000) * 10.0 ** rng.integers(-8, 8, 4000)
         ).astype(_F32)
    # Products that land near a float32 halfway point after the addition.
    a[:8] = _F32(1 + 2 ** -12)
    b[:8] = _F32(1 + 2 ** -12)
    c[:8] = _F32(-1)
    got = schedule.fma_f32(a, b, c)
    want = np.array([_exact_fma(x, y, z) for x, y, z in zip(a, b, c)], _F32)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def _linspace_ref(start, end, n) -> np.ndarray:
    s, e = _F32(start), _F32(end)
    if n <= 1:
        return np.full(n, s, _F32)
    step = _F32(_F32(e - s) / _F32(n - 1))
    out = []
    for i in range(n):
        if i < n // 2:
            out.append(_exact_fma(step, _F32(i), s))
        else:
            out.append(_exact_fma(-step, _F32(n - 1 - i), e))
    return np.array(out, _F32)


@pytest.mark.parametrize("start,end", [(1.0, None), (0.0, 1.0), (1.0, 0.02),
                                       (999.0, 0.0)])
def test_linspace_matches_the_two_ended_one_rounding_fill(start, end):
    for n in list(range(0, 70)) + [100, 128, 333, 1000]:
        stop = 1.0 / n if end is None and n else (end or 0.0)
        got = schedule.linspace_f32(start, stop, n)
        want = _linspace_ref(start, stop, n)
        assert got.dtype == _F32 and got.shape == (n,)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), n
        if n > 1:
            assert got[0] == _F32(start) and got[-1] == _F32(stop)


def test_zimage_sigmas_shape_endpoints_and_shift():
    for n in (1, 4, 8, 9, 20, 28, 40, 50):
        s = schedule.zimage_sigmas(n)
        assert s.dtype == _F32 and s.shape == (n + 1,)
        assert s[0] == _F32(1.0) and s[-1] == _F32(0.0)
        assert np.all(np.diff(s) < 0)
        base = schedule.linspace_f32(1.0, 1.0 / n, n)
        want = (_F32(3.0) * base) / (_F32(1.0) + _F32(2.0) * base)
        assert np.array_equal(s[:-1].view(np.uint32), want.view(np.uint32))
    with pytest.raises(ValueError):
        schedule.zimage_sigmas(0)


def test_external_sigmas_are_checked():
    assert schedule.check_sigmas([1.0, 0.5, 0.0]).dtype == _F32
    for bad in ([1.0], [[1.0, 0.0]], [0.5, 1.0], [1.0, float("nan")]):
        with pytest.raises(ValueError):
            schedule.check_sigmas(bad)


def test_euler_equals_float32_numpy_bit_for_bit():
    sig = schedule.zimage_sigmas(8)
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((1, 4, 8, 8)).astype(_F32)
    x = mx.array(x_np)
    sampler = samplers.make_sampler("euler", sig)
    assert sampler.steps == 8
    for i in range(8):
        v_np = (np.sin(x_np) * sig[i] - _F32(0.25)).astype(_F32)
        x = sampler.step(x, mx.array(v_np), i)
        dt = sig[i + 1] - sig[i]
        x_np = (x_np + (dt * v_np).astype(_F32)).astype(_F32)
        got = np.array(x)
        assert got.dtype == _F32
        assert np.array_equal(got.view(np.uint32), x_np.view(np.uint32)), i
    with pytest.raises(IndexError):
        sampler.step(x, x, 8)


def test_unknown_sampler_names_the_known_ones():
    with pytest.raises(ValueError, match="known: euler"):
        samplers.make_sampler("dpmpp_2m", [1.0, 0.0])


def test_seeded_noise_is_float32_c_order_and_stable():
    a = samplers.noise_from_seed(1, (1, 2, 2, 2))
    b = samplers.noise_from_seed(1, (1, 2, 2, 2))
    assert a.dtype == _F32 and a.shape == (1, 2, 2, 2)
    assert np.array_equal(a, b)
    assert a.ravel().view(np.uint32).tolist() == [
        1071469380, 3216430992, 1065585956, 1072752548,
        1033494712, 1054186812, 3219716789, 1071846364]
    flat = samplers.noise_from_seed(1, (8,))
    assert np.array_equal(flat, a.ravel())
    assert not np.array_equal(samplers.noise_from_seed(2, (8,)), flat)
    with pytest.raises(ValueError):
        samplers.noise_from_seed(-1, (2,))


def test_krea2_sigmas_are_the_shifted_base_with_a_final_zero():
    for steps, mu in ((8, 1.15), (52, 0.93), (1, 1.15)):
        got = schedule.krea2_sigmas(steps, mu)
        base = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
        e = np.float32(np.exp(mu))
        want = (e / (e + (np.float32(1) / base - np.float32(1))))
        assert got.dtype == np.float32 and got.shape == (steps + 1,)
        assert np.array_equal(got[:-1], want.astype(np.float32))
        assert got[0] == 1.0 and got[-1] == 0.0
        assert np.all(np.diff(got) < 0)
    with pytest.raises(ValueError, match="at least 1"):
        schedule.krea2_sigmas(0, 1.15)


def test_the_krea2_schedule_shifts_turbo_by_its_constant():
    from gmlx.image.registry import get_family

    fam = get_family("krea2")
    turbo = fam.schedule(8, tokens=4096, variant="turbo")
    assert np.array_equal(turbo, schedule.krea2_sigmas(8, 1.15))
    assert np.array_equal(turbo, fam.schedule(8, tokens=64, variant="turbo"))
    raw = fam.schedule(52, tokens=4096, variant="raw")
    mu = schedule.calculate_shift(4096, 256, 6400, 0.5, 1.15)
    assert np.array_equal(raw, schedule.krea2_sigmas(52, mu))
    assert not np.array_equal(raw, fam.schedule(52, tokens=1024,
                                                variant="raw"))
    with pytest.raises(ValueError, match="pass tokens"):
        fam.schedule(8, tokens=None, variant=None)
