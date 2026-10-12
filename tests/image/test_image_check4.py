"""Check 4 of the image acceptance gate for Z-Image: the sigma schedule
equals diffusers' bit for bit, the linspace replica equals torch's float32
linspace, the Euler sampler follows diffusers' trajectory and the exact
one within 1e-6, and the UniPC sampler follows diffusers' UniPC within
1e-6 on a field that changes with the latents and the sigma. The arrays are small and committed, so this runs on the
CPU tier. scripts/image_oracle/check4_z_image.py writes them."""
from __future__ import annotations

import numpy as np
import pytest
from oracle_fixture import committed_fixture

import mlx.core as mx

from gmlx.image import samplers, schedule

NAME = "check4_z_image"


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def _bits(a) -> list:
    return np.asarray(a, np.float32).view(np.uint32).tolist()


def test_sigmas_are_bitwise_equal(check4):
    arrays, m = check4
    assert m["tolerances"]["sigmas"] == "bitwise"
    assert m["oracle"]["packages"]["torch"]["version"]
    for n in m["inputs"]["steps"]:
        want = arrays[f"sigmas_{n}"]
        got = schedule.zimage_sigmas(n, shift=m["inputs"]["shift"])
        assert got.dtype == np.float32 and got.shape == want.shape
        assert _bits(got) == _bits(want), n


def test_linspace_replica_matches_the_oracle_torch(check4):
    arrays, m = check4
    got = np.concatenate([schedule.linspace_f32(1.0, 1.0 / n, n)
                          for n in range(1, 101)])
    assert _bits(got) == _bits(arrays["linspace_probe"])


def test_euler_follows_the_oracle_and_the_exact_trajectory(check4):
    arrays, m = check4
    steps = m["inputs"]["trajectory_steps"]
    tol = m["tolerances"]["trajectory_max_abs"]
    ref = arrays[f"trajectory_{steps}"]
    noise, x0 = arrays["noise"], arrays["x0"]
    v = mx.array(noise - x0)
    sig = schedule.zimage_sigmas(steps)
    sampler = samplers.make_sampler("euler", sig)
    x = mx.array(noise)
    worst = worst_exact = 0.0
    for i in range(steps):
        x = sampler.step(x, v, i)
        got = np.array(x)
        exact = noise + (sig[i + 1] - 1) * (noise - x0)
        worst = max(worst, float(np.abs(got - ref[i + 1]).max()))
        worst_exact = max(worst_exact, float(np.abs(got - exact).max()))
    assert worst <= tol
    assert worst_exact <= m["tolerances"]["exact_max_abs"]
    assert float(np.abs(np.array(x) - x0).max()) <= tol


@pytest.mark.parametrize("n", [1, 2, 8, 50])
def test_unipc_follows_the_oracle(check4, n):
    arrays, m = check4
    assert n in m["inputs"]["unipc_steps"]
    tol = m["tolerances"]["unipc_max_abs"]
    ref = arrays[f"unipc_trajectory_{n}"]
    sampler = samplers.make_sampler("uni_pc_bh2", schedule.zimage_sigmas(n))
    assert _bits(sampler.sigmas) == _bits(arrays[f"unipc_sigmas_{n}"])
    x, x0 = mx.array(arrays["noise"]), mx.array(arrays["x0"])
    worst = 0.0
    for i in range(n):
        s = mx.array(sampler.sigmas[i], mx.float32)
        x = sampler.step(x, (1 + s) * x / 2 - x0, i)
        worst = max(worst, float(np.abs(np.array(x) - ref[i + 1]).max()))
    assert worst <= tol


def test_unipc_refuses_a_step_outside_the_schedule():
    sampler = samplers.make_sampler("uni_pc_bh2", schedule.zimage_sigmas(2))
    with pytest.raises(IndexError):
        sampler.step(mx.zeros((1,)), mx.zeros((1,)), 2)
