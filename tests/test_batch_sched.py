"""Decode-priority prefill pacing: wrapper logic over a fake generator."""

import time

import pytest

from mlx_vlm.generate import ar

import gmlx.batch_sched as batch_sched


class FakeGen:
    completion_batch_size = 32

    def __init__(self):
        self._generation_batch = [1, 2]
        self._prompt_batch = object()
        self._unprocessed_sequences = []
        self._prompt_time_counter = 0.0
        self.chunks = 0
        self.decodes = 0
        self.arrivals = 0


def _fake_next(self, **kw):
    if len(self._generation_batch) > 0:
        self.decodes += 1
        time.sleep(0.001)
    # a handler thread may enqueue mid-call; appends bind to whatever list
    # the wrapper left on the instance
    self.arrivals += 1
    self._unprocessed_sequences.append(("uid", self.arrivals))
    if self._prompt_batch is not None:
        self.chunks += 1
        time.sleep(0.004)
        self._prompt_time_counter += 0.004
    return [], []


@pytest.fixture
def paced(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "1.0")
    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    batch_sched.install_decode_priority_sched()
    yield ar.BatchGenerator._next


def test_pacing_ratio_and_arrival_merge(paced):
    g = FakeGen()
    pending = g._unprocessed_sequences
    n = 120
    for _ in range(n):
        paced(g)
    # 4ms chunk / 1ms step at ratio 1.0 -> roughly 4 decode ticks per chunk
    per_chunk = g.decodes / g.chunks
    assert 2.5 <= per_chunk <= 7.0
    # every mid-call arrival survives the stash/restore merge
    assert g._unprocessed_sequences is pending
    assert len(pending) == g.arrivals == n


def test_idle_decode_prefill_unpaced(paced):
    g = FakeGen()
    g._generation_batch = []
    for _ in range(10):
        paced(g)
    assert g.chunks == 10


def test_pure_decode_passthrough(paced):
    g = FakeGen()
    g._prompt_batch = None
    for _ in range(10):
        g._unprocessed_sequences = []
        paced(g)
    assert g.decodes == 10


def test_ratio_zero_passthrough(paced, monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "0")
    g = FakeGen()
    for _ in range(10):
        paced(g)
    # stock 1 decode : 1 chunk per tick
    assert g.chunks == 10 and g.decodes == 10


def test_ratio_flips_mid_run(paced, monkeypatch):
    g = FakeGen()
    for _ in range(40):
        paced(g)
    paced_chunks = g.chunks
    assert g.decodes / g.chunks > 2  # paced at 1.0
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "0")
    for _ in range(10):
        paced(g)
    assert g.chunks == paced_chunks + 10  # per-tick read: now stock


def test_bad_ratio_warns_and_defaults(paced, monkeypatch, caplog):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "banana")
    g = FakeGen()
    with caplog.at_level("WARNING"):
        for _ in range(30):
            paced(g)
    assert g.decodes / g.chunks > 2  # behaved as 1.0
    assert sum("banana" in r.message for r in caplog.records) == 1
