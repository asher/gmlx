"""Check 4 of the video acceptance gate for MiniMax-H3: the video, audio
and 768p Turbo schedules against diffusers' MiniMaxH3Scheduler at 1 to
16, 20, 24, 30, 40, 50 and 100 forwards. Each sigma, and each time
1 - sigma that a forward runs at, equals the scheduler's bit for bit,
with the same count. The arrays are small and committed, so this runs on
the CPU tier. scripts/video_oracle/check4_h3.py writes them."""
from __future__ import annotations

import numpy as np
import pytest
from video_oracle_fixture import committed_fixture

from gmlx.video import api, schedule
from gmlx.video.families.h3 import family as hf

NAME = "check4_h3"


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def _bits(a) -> np.ndarray:
    return np.asarray(a, np.float32).view(np.int32)


def test_the_fixture_covers_the_family_shifts(check4):
    _, m = check4
    assert set(m["inputs"]["shifts"]) == {hf.SHIFTS["video"],
                                          hf.SHIFTS["audio"],
                                          hf.TURBO_768_SHIFT}
    assert m["tolerances"]["sigmas"] == m["tolerances"]["timesteps"] == \
        "bitwise"


def test_sigmas_and_times_equal_the_scheduler_bit_for_bit(check4):
    arrays, m = check4
    for shift in m["inputs"]["shifts"]:
        for n in m["inputs"]["forwards"]:
            k = f"{shift:g}_{n}"
            got = schedule.sigmas(n, shift)
            want = arrays[f"sigmas_{k}"]
            assert got.dtype == np.float32 and got[-1] == 0, k
            assert got.size - 1 == m["results"]["forwards_run"][k] == n, k
            assert got.shape == want.shape, k
            assert np.array_equal(_bits(got), _bits(want)), k
            times = [api.stream_time(s) for s in got[:-1]]
            assert np.array_equal(_bits(times),
                                  _bits(arrays[f"timesteps_{k}"])), k
