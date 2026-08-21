"""Pinned-invariant probe suite for the mlx stream/unwind contract.

The serve path's OOM containment (catch the allocator throw, drain
the worker stream, keep serving) rests on mlx runtime behaviors that
live in C++ internals: what a mid-tape throw leaves behind, whether
async survivors' events ever signal, drain ordering across streams,
GIL release around buffer-read waits, and encoder thread-locality.
None of that is visible to source fingerprinting of Python seams, so
these tests pin the behaviors directly: each runs a small probe
script in a subprocess on the real GPU and asserts the outcome the
installed mlx wheel is expected to produce.

Expectations that changed with the eval unwind fix shipped in mlx
0.32.1 (ml-explore/mlx#3675) are keyed on UNWIND_FIXED; the suite is
the referee for that upgrade. Arms that abort the whole process on
pre-fix wheels (a misordered drain commits a fence wait whose update
is uncommitted, so the GPU times out) are skipped here on those
wheels and exercised once per flag day by scripts/flag_day_runbook.py.

The oversized allocation that trips every probe is rejected by the
allocator up front, so the suite never commits real memory pressure;
each probe finishes in seconds. Requires a real Apple GPU: skipped
under KQUANT_FORCE_CPU (CI) and on paravirtual Metal devices.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROBES = Path(__file__).parent / "invariant_probes"


def _mlx_version():
    import mlx.core as mx

    parts = []
    for p in mx.__version__.split(".")[:3]:
        m = re.match(r"\d+", p)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts)


def _real_apple_gpu() -> bool:
    if os.environ.get("KQUANT_FORCE_CPU"):
        return False
    try:
        import mlx.core as mx

        name = str(mx.device_info().get("device_name", ""))
    except Exception:
        return False
    return "Apple" in name and "Paravirtual" not in name


pytestmark = pytest.mark.skipif(
    not _real_apple_gpu(),
    reason="pinned-invariant probes need a real Apple GPU",
)

UNWIND_FIXED = _real_apple_gpu() and _mlx_version() >= (0, 32, 1)


def run_probe(script: str, *args: str, timeout: float = 60.0):
    return subprocess.run(
        [sys.executable, str(PROBES / script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_clean(r, token: str = "outcome=CLEAN"):
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    assert token in r.stdout, f"{r.stdout}\n{r.stderr}"


def test_sync_survivor_reads_clean_after_drain():
    assert_clean(run_probe("sync_survivor.py"))


def test_async_survivor_item_read():
    if UNWIND_FIXED:
        # The unwind handler signals survivor events: the read returns
        # promptly and correct.
        assert_clean(run_probe("async_survivor.py", "item"))
    else:
        # The survivor's event never signals; the read hangs forever.
        # The hang is the pinned behavior.
        with pytest.raises(subprocess.TimeoutExpired):
            run_probe("async_survivor.py", "item", timeout=20)


def test_async_survivor_samestream_consume_clean():
    assert_clean(run_probe("async_survivor.py", "samestream"))


def test_async_survivor_fresh_graph_clean():
    assert_clean(run_probe("async_survivor.py", "fresh"))


@pytest.mark.skipif(
    not UNWIND_FIXED,
    reason="aborts the process pre-0.32.1 (unsatisfiable event wait); "
    "flag-day runbook only",
)
def test_async_survivor_xstream_consume_clean():
    assert_clean(run_probe("async_survivor.py", "xstream"))


def test_gil_released_during_hung_read():
    r = run_probe("gil_release.py", "hang", timeout=90)
    if UNWIND_FIXED:
        assert_clean(r)
    else:
        # The read never returns but the interpreter lives: the ticker
        # thread keeps running and the in-process watchdog exits 42.
        assert r.returncode == 42, (
            f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
        )
        assert "outcome=ALIVE" in r.stdout, r.stdout


def test_gil_ticker_control_interleaves():
    r = run_probe("gil_release.py", "busy", timeout=90)
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    assert "outcome=BUSY-DONE" in r.stdout, r.stdout
    post_ticks = r.stdout.count("tick post=")
    assert post_ticks >= 3, f"ticker starved: {post_ticks} post ticks\n{r.stdout}"


def test_multistream_drain_peer_first_clean():
    assert_clean(run_probe("multistream_drain.py", "orderfix"))


@pytest.mark.skipif(
    not UNWIND_FIXED,
    reason="aborts the process pre-0.32.1 (fence wait committed before "
    "its update); flag-day runbook only",
)
def test_multistream_drain_thrower_first_clean():
    assert_clean(run_probe("multistream_drain.py", "thrower-first"))


@pytest.mark.skipif(
    not UNWIND_FIXED,
    reason="aborts at process teardown pre-0.32.1 (exit commits the "
    "default stream's fence wait, its update never committed); "
    "flag-day runbook only",
)
def test_reverse_topology_trip_survivable():
    # A red here on a post-fix wheel means the unwind handler's
    # stream-set drain order committed a fence wait ahead of its update
    # on this topology: that is the signal to take an order-free drain
    # request upstream.
    r = run_probe("reverse_topology.py", "trip-only")
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    assert "outcome=CAUGHT" in r.stdout, f"{r.stdout}\n{r.stderr}"


def test_reverse_topology_producer_first_clean():
    assert_clean(run_probe("reverse_topology.py", "producer-first"))


@pytest.mark.skipif(
    not UNWIND_FIXED,
    reason="aborts the process pre-0.32.1 (wait committed before its "
    "update); flag-day runbook only",
)
def test_reverse_topology_consumer_first_clean():
    assert_clean(run_probe("reverse_topology.py", "consumer-first"))


def test_thread_stream_ownership_matrix():
    r = run_probe("thread_rules.py")
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    want = {
        "a": "ok",
        "b": "ok",
        "c": "THROW",
        "d": "THROW",
        "e": "THROW",
    }
    got = dict(
        re.findall(r"^case=(\w) outcome=(\S+)", r.stdout, re.MULTILINE)
    )
    for case, expect in want.items():
        assert got.get(case, "").startswith(expect), (
            f"case {case}: want {expect}, got {got.get(case)!r}\n{r.stdout}"
        )
