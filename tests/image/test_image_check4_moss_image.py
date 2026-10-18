"""Check 4 of the image acceptance gate for MOSS-Image: the family's
schedule against the release's MOSSImageScheduler at 50, 28 and 8 steps,
at 250 and 1000, where the snap to the training grid drops steps, and at
822, where a sigma ties between two grid points. Each sigma equals the
release's bit for bit, at the same step count. The release's timesteps
are its sigmas times 1000 in float32, as the DiT forms them. The arrays
are small and committed, so this runs on the CPU tier.
scripts/image_oracle/check4_moss_image.py writes them."""
from __future__ import annotations

import numpy as np
import pytest
from oracle_fixture import committed_fixture

from gmlx.image import schedule
from gmlx.image.registry import get_family

NAME = "check4_moss_image"


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def _ulps(got, want) -> int:
    g = np.asarray(got, np.float32).view(np.int32).astype(np.int64)
    w = np.asarray(want, np.float32).view(np.int32).astype(np.int64)
    return int(np.max(np.abs(g - w)))


def _gap(sigmas_of, arrays, n) -> tuple[int, int]:
    """Steps of gmlx and of the release, and the largest ulp gap when the
    counts agree."""
    got, want = sigmas_of(n), arrays[f"sigmas_{n}"]
    if got.shape != want.shape:
        return got.size - 1, want.size - 1
    return got.size - 1, _ulps(got, want)


def test_sigmas_equal_the_release_ones_bit_for_bit(check4):
    arrays, m = check4
    fam = get_family("moss_image")
    assert m["tolerances"]["count"] == "exact"
    assert m["tolerances"]["sigmas"] == "bitwise"
    assert 822 in m["inputs"]["steps"]
    for n in m["inputs"]["steps"]:
        got = fam.schedule(n, tokens=None, variant="sft")
        want = arrays[f"sigmas_{n}"]
        assert got.dtype == np.float32 and got[-1] == 0
        assert got.size - 1 == m["results"]["steps_run"][str(n)], n
        assert got.shape == want.shape, n
        assert _ulps(got, want) == 0, n
        # The release's timestep is its sigma times 1000 in float32, as the
        # DiT forms it from a sigma.
        ts = want[:-1] * np.float32(1000)
        assert _ulps(ts, arrays[f"timesteps_{n}"]) == 0, n
    assert m["results"]["steps_run"]["1000"] < 1000


def test_the_schedule_reads_the_scheduler_config(check4):
    _, m = check4
    c = get_family("moss_image").constants.values
    s = m["inputs"]["scheduler"]
    assert s["shift"] == c["scheduler_shift"]
    assert s["num_train_timesteps"] == c["num_train_timesteps"]
    assert s["training_shift"] == 1.0


def _unsnapped(n):
    v = schedule.linspace_f32(1.0, 5.0 / 1004, n)
    return np.append(schedule.shift_static(v, 5.0), np.float32(0))


def _grid_k_over_1000(n):
    """The snap to k / 1000 rounded once, which breaks the tie at 822
    steps the other way."""
    v = schedule.linspace_f32(1.0, 5.0 / 1004, n)
    s = schedule.shift_static(v, 5.0)
    grid = (np.arange(1, 1001, dtype=np.float64) / 1000).astype(np.float32)
    hi = np.minimum(np.searchsorted(grid, s, side="left"), 999)
    lo = np.maximum(hi - 1, 0)
    pick = np.where(np.abs(s - grid[lo]) <= np.abs(grid[hi] - s), lo, hi)
    snapped = grid[pick]
    keep = np.concatenate([[True], snapped[1:] != snapped[:-1]])
    return np.append(snapped[keep], np.float32(0)).astype(np.float32)


def _no_dedup(n):
    s = schedule.moss_sigmas(n)
    return np.concatenate([s[:-1], np.repeat(s[-2], n + 1 - s.size),
                           s[-1:]]) if s.size < n + 1 else s


WRONG = {"sigmas not snapped": _unsnapped, "repeats kept": _no_dedup,
         "the grid k / 1000": _grid_k_over_1000}


@pytest.mark.parametrize("name", list(WRONG))
def test_a_wrong_schedule_misses(check4, name):
    arrays, m = check4
    missed = []
    for n in m["inputs"]["steps"]:
        steps, gap = _gap(WRONG[name], arrays, n)
        if steps != m["results"]["steps_run"][str(n)] or gap > 0:
            missed.append(n)
    assert missed, name


def test_the_tie_at_822_steps_goes_to_the_release_point(check4):
    """At 822 steps one sigma lies halfway between two points of the
    release's grid, and the release takes the lower one. On k / 1000
    rounded once the same sigma goes to the upper point, a whole grid
    step off, at the same count."""
    arrays, _ = check4
    want = arrays["sigmas_822"]
    got = get_family("moss_image").schedule(822, tokens=None, variant="sft")
    assert _ulps(got, want) == 0
    old = _grid_k_over_1000(822)
    assert old.shape == want.shape
    off = np.flatnonzero(old != want)
    assert np.abs(old - want).max() == pytest.approx(1e-3, abs=1e-6), off
