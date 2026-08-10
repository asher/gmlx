"""Decode-priority prefill pacing: wrapper logic over a fake generator."""

from types import SimpleNamespace

import pytest

from mlx_vlm.generate import ar

import gmlx.batch_sched as batch_sched


class _Clock:
    """Deterministic stand-in for ``time.perf_counter``.

    The pacer charges a decode tick by wall clock, so sleeping real
    milliseconds makes the decode/chunk ratio a function of the runner's sleep
    granularity: on a loaded CI box a 1 ms sleep lands nearer 2 ms and the
    measured ratio halves. Advancing by exact amounts tests the pacing
    arithmetic instead of the scheduler underneath it."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


_CLOCK = _Clock()
DECODE_S = 0.001
CHUNK_S = 0.004


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
        _CLOCK.advance(DECODE_S)
    # a handler thread may enqueue mid-call; appends bind to whatever list
    # the wrapper left on the instance
    self.arrivals += 1
    self._unprocessed_sequences.append(("uid", self.arrivals))
    if self._prompt_batch is not None:
        self.chunks += 1
        _CLOCK.advance(CHUNK_S)
        self._prompt_time_counter += CHUNK_S
    return [], []


@pytest.fixture
def paced(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "1.0")
    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    # batch_sched reads only time.perf_counter; swap the whole name so the
    # patch is scoped to that module rather than the global time module.
    _CLOCK.t = 0.0
    monkeypatch.setattr(batch_sched, "time",
                        SimpleNamespace(perf_counter=_CLOCK))
    batch_sched.install_decode_priority_sched()
    yield ar.BatchGenerator._next


def test_pacing_ratio_and_arrival_merge(paced):
    g = FakeGen()
    pending = g._unprocessed_sequences
    n = 120
    for _ in range(n):
        paced(g)
    # Ratio 1.0 owes 4 ms of decode per 4 ms chunk, so a cycle is 4 paced
    # decode ticks plus the chunk tick, whose stock body also runs a decode
    # step: 5 decodes per chunk. n=120 is 24 whole cycles, and the clock is
    # exact, so this is an equality rather than a tolerance band.
    per_chunk = g.decodes / g.chunks
    assert per_chunk == 5.0
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


def test_auto_mode_wired_stock_without_incumbent(paced, monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "auto")
    g = FakeGen()
    for _ in range(10):
        paced(g)
    # burst shape (no admitted-row stamps precede the waiters): auto
    # resolves 0 every tick => stock 1 decode : 1 chunk
    assert g.chunks == 10 and g.decodes == 10


def test_auto_kill_switch_paces_static(paced, monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "auto")
    monkeypatch.setenv("GMLX_DECODE_PREFILL_AUTO", "0")
    g = FakeGen()
    for _ in range(40):
        paced(g)
    # auto disabled resolves to the static paced ratio (1.0 at rho 0.5)
    assert g.decodes / g.chunks == 5.0


def test_default_unset_is_auto(paced, monkeypatch):
    monkeypatch.delenv("GMLX_DECODE_PREFILL_RATIO", raising=False)
    g = FakeGen()
    for _ in range(10):
        paced(g)
    # auto with no incumbent resolves 0: stock 1 decode : 1 chunk
    assert g.chunks == 10 and g.decodes == 10


def test_bad_ratio_warns_and_defaults(paced, monkeypatch, caplog):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "banana")
    g = FakeGen()
    with caplog.at_level("WARNING"):
        for _ in range(30):
            paced(g)
    # falls back to the default (auto); burst shape resolves stock
    assert g.chunks == 30 and g.decodes == 30
    assert sum("banana" in r.message for r in caplog.records) == 1
