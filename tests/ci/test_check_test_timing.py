"""scripts/check-test-timing.py on small sources: a fixed sleep and an
elapsed upper bound are found, and polls and lower bounds are not."""
from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_check_test_timing",
                                                  _ROOT / "scripts" / "check-test-timing.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = _load()


def _found(tmp_path, source: str) -> list[tuple[int, str, str]]:
    path = tmp_path / "test_x.py"
    path.write_text(textwrap.dedent(source))
    return ct.findings(path)


def test_a_planted_sleep_and_an_upper_bound_are_found(tmp_path):
    assert _found(tmp_path, """\
        import asyncio, time

        def test_sleeps():
            time.sleep(0.1)

        async def test_awaits():
            await asyncio.sleep(1)

        def test_bounds():
            t0 = time.monotonic()
            run()
            assert time.monotonic() - t0 < 5
        """) == [(4, "test_sleeps", "sleep"), (7, "test_awaits", "sleep"),
                 (12, "test_bounds", "upper bound")]


def test_polls_lower_bounds_and_zero_sleeps_are_not_found(tmp_path):
    assert _found(tmp_path, """\
        import time

        def test_polls():
            deadline = time.monotonic() + 30
            while not ready():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            for _ in range(100):
                if ready():
                    break
                time.sleep(0.01)

        def test_lower_bound():
            t0 = time.monotonic()
            run()
            assert time.monotonic() - t0 >= 1
            time.sleep(0)
        """) == []


def test_a_fixed_wait_in_a_loop_that_cannot_end_early_is_found(tmp_path):
    assert _found(tmp_path, """\
        import time

        def test_each():
            for case in cases():
                time.sleep(0.2)
                check(case)
        """) == [(5, "test_each", "sleep")]


def test_a_clock_stored_on_an_object_does_not_make_the_object_a_clock(tmp_path):
    assert _found(tmp_path, """\
        import time

        def test_band(gen, gov):
            gen.stamp = (time.perf_counter(), 1)
            band = gov.band(gen)
            assert band <= gov.YELLOW
        """) == []
