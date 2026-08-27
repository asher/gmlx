"""Prefill ticks: the concurrency-keyed term in the chunk-size resolver.

While the scheduler reports live decode rows, decayed_for_batch halves the
chunk until its predicted wall time (last observed chunk cost scaled by
the tier) fits the GMLX_PREFILL_TICK_MS budget. Inert at zero rows, with
no observation, or when disabled, so single-stream TTFT and the batch-job
regime are untouched.
"""

from types import SimpleNamespace

import pytest

from mlx_vlm.generate import ar

import gmlx.batch_sched as batch_sched
from gmlx import prefill_decay as pd

HEADS = 32


@pytest.fixture(autouse=True)
def _reset_tick_state():
    pd._LIVE_DECODE_ROWS = 0
    pd._CHUNK_COST = None
    pd._LAST_STEP = 0
    yield
    pd._LIVE_DECODE_ROWS = 0
    pd._CHUNK_COST = None
    pd._LAST_STEP = 0


def _cap(monkeypatch, gb):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", str(gb))


def _batch(base, depth, heads=HEADS):
    return SimpleNamespace(
        prefill_step_size=base,
        prompt_cache=[SimpleNamespace(offset=depth)],
        model=SimpleNamespace(
            config=SimpleNamespace(num_attention_heads=heads)),
    )


def _arm(rows=2, cost=(2048, 2.0)):
    pd.note_decode_pressure(rows)
    pd._LAST_STEP = cost[0]
    pd.note_chunk_cost(cost[1])


def test_inert_without_pressure_or_observation(monkeypatch):
    _cap(monkeypatch, 100.0)  # depth term never bites
    # no rows, no observation
    assert pd.decayed_for_batch(_batch(2048, 0)) == 2048
    # observation but no rows
    pd._LAST_STEP = 2048
    pd.note_chunk_cost(2.0)
    assert pd.decayed_for_batch(_batch(2048, 0)) == 2048
    # rows but no observation
    pd._CHUNK_COST = None
    pd.note_decode_pressure(2)
    assert pd.decayed_for_batch(_batch(2048, 0)) == 2048


def test_tier_math_against_budget(monkeypatch):
    _cap(monkeypatch, 100.0)
    _arm(rows=2, cost=(2048, 2.0))  # 0.9766 ms per token
    # 500 ms budget: 512 * 0.9766 = 500.0 fits exactly (boundary is <=)
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "500")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 512
    # just under the boundary: one more halving
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "499")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 256
    # 1100 ms budget: 1024 * 0.9766 = 1000.0 fits
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "1100")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 1024
    # generous budget: full chunk
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "5000")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 2048


def test_floor_and_disable(monkeypatch):
    _cap(monkeypatch, 100.0)
    _arm(rows=4, cost=(2048, 20.0))  # ~10 ms per token, over-RAM class
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "500")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 256  # default floor
    monkeypatch.setenv("GMLX_PREFILL_MIN_STEP", "128")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 128
    monkeypatch.delenv("GMLX_PREFILL_MIN_STEP")
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "0")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 2048  # term off


def test_min_of_depth_and_tick_terms(monkeypatch):
    # depth term alone -> 256 at this depth/cap; a generous tick budget must
    # not RAISE the step above the depth term
    _cap(monkeypatch, 1.0)
    _arm(rows=2, cost=(2048, 0.1))
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "5000")
    assert pd.decayed_for_batch(_batch(2048, 100_000)) == 256
    # tick term tighter than depth term at shallow depth
    _arm(rows=2, cost=(2048, 2.0))
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "499")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 256


def test_kill_switch_covers_tick_term(monkeypatch):
    _cap(monkeypatch, 100.0)
    _arm(rows=2, cost=(2048, 2.0))
    monkeypatch.setenv("GMLX_PREFILL_TICK_MS", "500")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd.decayed_for_batch(_batch(2048, 0)) == 2048


def test_chunk_cost_pairs_with_last_returned_step(monkeypatch):
    _cap(monkeypatch, 100.0)
    step = pd.decayed_for_batch(_batch(2048, 0))
    assert step == 2048 and pd._LAST_STEP == 2048
    pd.note_chunk_cost(1.5)
    assert pd._CHUNK_COST == (2048, 1.5)
    # before any resolution, an observation is dropped, not mispaired
    pd._LAST_STEP = 0
    pd._CHUNK_COST = None
    pd.note_chunk_cost(1.5)
    assert pd._CHUNK_COST is None
    # zero/negative wall never records
    pd._LAST_STEP = 2048
    pd.note_chunk_cost(0.0)
    assert pd._CHUNK_COST is None


# ---- scheduler feeding (fake generator, same rig as test_batch_sched) ----

_CLOCK_T = [0.0]
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


def _fake_next(self, **kw):
    if len(self._generation_batch) > 0:
        self.decodes += 1
        _CLOCK_T[0] += DECODE_S
    if self._prompt_batch is not None:
        self.chunks += 1
        _CLOCK_T[0] += CHUNK_S
        self._prompt_time_counter += CHUNK_S
    return [], []


@pytest.fixture
def paced(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "1.0")
    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    _CLOCK_T[0] = 0.0
    monkeypatch.setattr(
        batch_sched, "time",
        SimpleNamespace(perf_counter=lambda: _CLOCK_T[0]))
    batch_sched.install_decode_priority_sched()
    yield ar.BatchGenerator._next


def test_scheduler_notes_pressure_and_chunk_cost(paced):
    g = FakeGen()
    pd._LAST_STEP = 2048  # as if decayed_for_batch sized the chunk
    for _ in range(10):
        paced(g)
    assert pd._LIVE_DECODE_ROWS == 2
    assert g.chunks >= 1
    assert pd._CHUNK_COST == (2048, pytest.approx(CHUNK_S))
    # decode drains: pressure clears on the next tick
    g._generation_batch = []
    paced(g)
    assert pd._LIVE_DECODE_ROWS == 0


def test_scheduler_notes_cost_even_unpaced(paced, monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "0")
    g = FakeGen()
    pd._LAST_STEP = 1024
    paced(g)
    assert pd._CHUNK_COST == (1024, pytest.approx(CHUNK_S))


def test_paced_tick_does_not_clobber_observation(paced):
    g = FakeGen()
    pd._LAST_STEP = 2048
    for _ in range(2):
        paced(g)  # first tick admits a chunk; second is paced (no prefill)
    cost = pd._CHUNK_COST
    assert cost is not None
    g2 = FakeGen()
    g2._prompt_batch = None  # pure decode tick: no chunk, no update
    paced(g2)
    assert pd._CHUNK_COST == cost
