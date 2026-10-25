"""Kernel memory readers: the free-page stall watch."""

import types

import pytest

import gmlx.serve.kernel_vm as kv

GB = 1 << 30


@pytest.fixture
def box(monkeypatch):
    box = {"waiters": 3, "marks": (0, 0, 0, 0), "now": 1000.0}
    monkeypatch.setattr(kv, "free_page_waiters", lambda: box["waiters"])
    monkeypatch.setattr(kv, "reclaim_marks", lambda: box["marks"])
    monkeypatch.setattr(kv, "time", types.SimpleNamespace(
        monotonic=lambda: box["now"]))
    return box


def _read(watch, box, seconds, step=0.25, marks=None):
    out = 0.0
    end = box["now"] + seconds
    while box["now"] < end - 1e-9:
        box["now"] += step
        if marks is not None:
            box["marks"] = marks(box["now"] - 1000.0)
        out = watch()
    return out


def test_a_stall_is_waiters_while_the_kernel_frees_nothing(box):
    # The clock runs while threads wait and nothing is freed. Compression
    # that saves memory, or fewer threads waiting, holds it; no waiters
    # end the streak.
    watch = kv.StarveWatch()
    assert watch() == 0.0
    assert _read(watch, box, 1.0) == pytest.approx(1.0)
    fed = lambda t: (int(t * GB), int(t * GB) // 2, 0, 0)  # noqa: E731
    assert _read(watch, box, 1.0, marks=fed) == 0.0
    box["waiters"] = 2
    assert watch() == 0.0
    box["waiters"] = 0
    assert watch() == 0.0 and watch.last is None


def test_compression_that_saves_nothing_is_a_stall(box):
    # Quantized weights going through the compressor, as at the October
    # freeze: the compressor takes 1 GB a second and stores all of it.
    watch = kv.StarveWatch()
    watch()
    same = lambda t: (int(t * GB), int(t * GB), 0, 0)  # noqa: E731
    assert _read(watch, box, 2.0, marks=same) == pytest.approx(2.0)


def test_readings_far_apart_count_half_a_second_each(box):
    # Two readings 3 s apart with threads waiting at both are not a 3 s
    # wait.
    watch = kv.StarveWatch()
    watch()
    box["now"] += 3.0
    assert watch() == pytest.approx(kv._STARVE_STEP_S)


def test_reclaim_holds_the_clock_for_five_seconds_of_waiting(box):
    watch = kv.StarveWatch()
    watch()
    fed = lambda t: (int(t * GB), int(t * GB) // 2, 0, 0)  # noqa: E731
    assert _read(watch, box, kv._STARVE_CAP_S - 0.25, marks=fed) == 0.0
    assert _read(watch, box, 2.0, marks=fed) >= 2.0


@pytest.mark.parametrize("marks,freed", [
    ((0, 0, 1, 0), True),                     # a swap out
    ((0, 0, 0, (256 << 20) // kv._PAGE), True),   # file pages paged out
    ((1 << 30, 1 << 30, 0, 0), False),        # nothing compressed away
    ((1 << 20, 0, 0, 0), False),              # under the rate
])
def test_reclaim_is_memory_freed(marks, freed):
    assert kv.reclaimed((0, 0, 0, 0), marks, 1.0) is freed
