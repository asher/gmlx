"""populate: frontier registry idempotence, worker top-up, wait release,
and load-time gates. No model, no GPU - reader threads are faked (or tiny
real sleepers), the budget gate is driven via device_info."""
from __future__ import annotations

import threading
import time

import pytest

from gmlx import populate


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    monkeypatch.setattr(populate, "_streams", {})


class FakeThread:
    spawned: list = []

    def __init__(self, *a, **k):
        FakeThread.spawned.append(self)
        self.alive = True

    def start(self):
        pass

    def is_alive(self):
        return self.alive


@pytest.fixture()
def fake_threads(monkeypatch):
    FakeThread.spawned = []
    monkeypatch.setattr(populate.threading, "Thread", FakeThread)
    monkeypatch.setattr(populate, "resident_fraction", lambda p: 0.0)
    return FakeThread.spawned


def test_start_populate_is_idempotent_per_file(tmp_path, fake_threads):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 128)

    populate.start_populate([str(p)], log=lambda *a: None)
    assert len(fake_threads) == populate._POPULATE_WORKERS

    # second full-width kickoff (e.g. the phase-7 residency warm) spawns
    # nothing, even though the file is not yet resident
    populate.start_populate([str(p)], log=lambda *a: None)
    assert len(fake_threads) == populate._POPULATE_WORKERS

    # a different path to the same file is still deduped (realpath keys)
    alias = tmp_path / "alias.gguf"
    alias.symlink_to(p)
    populate.start_populate([str(alias)], log=lambda *a: None)
    assert len(fake_threads) == populate._POPULATE_WORKERS


def test_resident_file_skipped(tmp_path, monkeypatch):
    FakeThread.spawned = []
    monkeypatch.setattr(populate.threading, "Thread", FakeThread)
    monkeypatch.setattr(populate, "resident_fraction", lambda p: 1.0)
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 128)
    populate.start_populate([str(p)], log=lambda *a: None)
    assert FakeThread.spawned == []
    # evicted since: the next load's kick re-probes and streams it
    monkeypatch.setattr(populate, "resident_fraction", lambda p: 0.0)
    populate.start_populate([str(p)], log=lambda *a: None)
    assert len(FakeThread.spawned) == populate._POPULATE_WORKERS


def test_finished_stream_restarts_after_eviction(tmp_path, fake_threads,
                                                 monkeypatch):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 128)
    populate.start_populate([str(p)], log=lambda *a: None)
    assert len(fake_threads) == populate._POPULATE_WORKERS
    for t in list(fake_threads):
        t.alive = False  # stream finished

    # finished and still resident: nothing to re-read
    monkeypatch.setattr(populate, "resident_fraction", lambda p: 1.0)
    populate.start_populate([str(p)], log=lambda *a: None)
    assert len(fake_threads) == populate._POPULATE_WORKERS

    # finished and evicted (server: TTL unload, neighbor loads evict, then
    # a re-request in the same process): the stream restarts
    monkeypatch.setattr(populate, "resident_fraction", lambda p: 0.0)
    populate.start_populate([str(p)], log=lambda *a: None)
    assert len(fake_threads) == 2 * populate._POPULATE_WORKERS


def test_wait_for_releases_on_small_tail(tmp_path, fake_threads):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 1000)
    populate.start_populate([str(p)], log=lambda *a: None)
    st = next(iter(populate._streams.values()))
    st.done = 800  # remaining 20% < default 25% tail
    t0 = time.perf_counter()
    populate.wait_for([str(p)], log=lambda *a: None)
    assert time.perf_counter() - t0 < 0.5


def test_wait_for_blocks_until_stream_done(tmp_path, monkeypatch):
    monkeypatch.setattr(populate, "resident_fraction", lambda p: 0.0)
    monkeypatch.setenv("GMLX_POPULATE_WAIT_TAIL", "0")
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 1000)

    release = threading.Event()
    monkeypatch.setattr(
        populate, "_reader", lambda st: release.wait(5))
    populate.start_populate([str(p)], log=lambda *a: None)

    lines = []
    threading.Timer(0.2, release.set).start()
    populate.wait_for([str(p)], log=lines.append)
    assert lines and "waited" in lines[0]

    monkeypatch.setenv("GMLX_POPULATE_WAIT", "0")
    t0 = time.perf_counter()
    populate.wait_for([str(p)], log=lines.append)
    assert time.perf_counter() - t0 < 0.05


def test_residency_warm_env_disables(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(populate, "start_populate",
                        lambda *a, **k: called.append(a))
    monkeypatch.setenv("GMLX_RESIDENCY_WARM", "0")
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x")
    populate.maybe_populate_for_load([str(p)])
    assert called == []


def test_populate_early_env_disables(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(populate, "start_populate",
                        lambda *a, **k: called.append(a))
    monkeypatch.setenv("GMLX_POPULATE_EARLY", "0")
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x")
    populate.maybe_populate_for_load([str(p)])
    assert called == []


def test_over_budget_streaming_model_never_populates(tmp_path, monkeypatch):
    import mlx.core as mx

    called = []
    monkeypatch.setattr(populate, "start_populate",
                        lambda paths, log=print, **kw: called.append(paths))
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 4096)

    monkeypatch.setattr(
        mx, "device_info",
        lambda: {"max_recommended_working_set_size": 0})
    populate.maybe_populate_for_load([str(p)])
    assert called == []

    monkeypatch.setattr(
        mx, "device_info",
        lambda: {"max_recommended_working_set_size": 1 << 40})
    populate.maybe_populate_for_load([str(p)])
    assert called == [[str(p)]]
