"""populate: frontier registry idempotence, worker top-up, wait release,
and load-time gates. No model, no GPU - reader threads are faked (or tiny
real sleepers), the budget gate is driven via device_info."""
from __future__ import annotations

import threading

import pytest

import gmlx.load.populate as populate


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    monkeypatch.setattr(populate, "_streams", {})


class _Clock:
    """populate's clock: each poll of ``wait_for`` moves it, and calls
    ``on_poll``."""

    def __init__(self, on_poll=None):
        self.now = 0.0
        self.polls = 0
        self._on_poll = on_poll

    def perf_counter(self):
        return self.now

    def sleep(self, s):
        self.polls += 1
        self.now += s
        if self._on_poll is not None:
            self._on_poll()


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


def test_wait_for_releases_on_small_tail(tmp_path, fake_threads, monkeypatch):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 1000)
    populate.start_populate([str(p)], log=lambda *a: None)
    st = next(iter(populate._streams.values()))
    st.done = 800  # remaining 20% < default 25% tail
    clock = _Clock()
    monkeypatch.setattr(populate, "time", clock)
    lines = []
    populate.wait_for([str(p)], log=lines.append)
    assert clock.polls == 0 and lines == []   # released without a wait


def test_wait_for_blocks_until_stream_done(tmp_path, monkeypatch):
    monkeypatch.setattr(populate, "resident_fraction", lambda p: 0.0)
    monkeypatch.setenv("GMLX_POPULATE_WAIT_TAIL", "0")
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 1000)

    release = threading.Event()
    monkeypatch.setattr(
        populate, "_reader", lambda st: release.wait(30))

    def end_stream():
        # The readers end only once wait_for waits on them.
        release.set()
        for st in populate._streams.values():
            for t in st.threads:
                t.join(30)

    clock = _Clock(on_poll=end_stream)
    monkeypatch.setattr(populate, "time", clock)
    populate.start_populate([str(p)], log=lambda *a: None)

    lines = []
    populate.wait_for([str(p)], log=lines.append)
    assert clock.polls >= 1
    assert lines and "waited" in lines[0]

    # Off, it does not wait on a stream whose readers still run.
    release.clear()
    populate.start_populate([str(p)], log=lambda *a: None)
    st = next(iter(populate._streams.values()))
    assert any(t.is_alive() for t in st.threads)
    monkeypatch.setenv("GMLX_POPULATE_WAIT", "0")
    polls = clock.polls
    populate.wait_for([str(p)], log=lines.append)
    assert clock.polls == polls and len(lines) == 1
    end_stream()


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
