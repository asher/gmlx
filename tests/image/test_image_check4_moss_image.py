"""Check 4 of the image acceptance gate for MOSS-Image: the family's
schedule against the release's MOSSImageScheduler at 50, 28 and 8 steps,
and at 250 and 1000, where the snap to the training grid drops steps.
Each sigma is within one float32 ulp of the release's, whose grid is a
float32 linspace, and the step count is exact. The release's timesteps
are its sigmas times 1000 in float32, as the DiT forms them. The arrays are small and
committed, so this runs on the CPU tier.
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


def test_sigmas_are_within_one_ulp_with_the_same_count(check4):
    arrays, m = check4
    fam = get_family("moss_image")
    assert m["tolerances"]["count"] == "exact"
    for n in m["inputs"]["steps"]:
        got = fam.schedule(n, tokens=None, variant="sft")
        want = arrays[f"sigmas_{n}"]
        assert got.dtype == np.float32 and got[-1] == 0
        assert got.size - 1 == m["results"]["steps_run"][str(n)], n
        assert got.shape == want.shape, n
        assert _ulps(got, want) <= 1, n
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


def _no_dedup(n):
    s = schedule.moss_sigmas(n)
    return np.concatenate([s[:-1], np.repeat(s[-2], n + 1 - s.size),
                           s[-1:]]) if s.size < n + 1 else s


WRONG = {"sigmas not snapped": _unsnapped, "repeats kept": _no_dedup}


@pytest.mark.parametrize("name", list(WRONG))
def test_a_wrong_schedule_misses(check4, name):
    arrays, m = check4
    missed = []
    for n in m["inputs"]["steps"]:
        steps, gap = _gap(WRONG[name], arrays, n)
        if steps != m["results"]["steps_run"][str(n)] or gap > 1:
            missed.append(n)
    assert missed, name
