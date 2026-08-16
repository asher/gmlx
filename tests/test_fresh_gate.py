"""Freshness admission gate: hold rules, coverage peek, stash merge."""

import threading
from types import SimpleNamespace

import pytest

from mlx_vlm.generate import ar

import gmlx.fresh_gate as fg


class FakeBlock:
    def __init__(self, token_ids):
        self.token_ids = tuple(token_ids)


class FakeManager:
    block_size = 16

    def __init__(self):
        self.lock = threading.RLock()
        self.hash_table = {}
        self._exact_cache = {}
        self._kq_anchor_cache = {}

    def add_block_chain(self, ids, extra_hash=0):
        from mlx_vlm import apc as _apc
        parent = _apc.SEED_PARENT_HASH
        bs = self.block_size
        for i in range(len(ids) // bs):
            chunk = tuple(ids[i * bs:(i + 1) * bs])
            h = _apc._hash_tokens(parent, chunk, extra_hash)
            self.hash_table[h] = FakeBlock(chunk)
            parent = h

    def add_exact(self, ids, extra_hash=0):
        self._exact_cache[len(self._exact_cache)] = SimpleNamespace(
            token_ids=tuple(ids), extra_hash=extra_hash)

    def add_anchor(self, ids, extra_hash=0):
        self._kq_anchor_cache[(tuple(ids), extra_hash)] = (None, 0)


def _seq(uid, ids):
    return (uid, list(ids), 200, {}, None, None)


class FakeGen:
    completion_batch_size = 32
    prefill_batch_size = 8

    def __init__(self, sequences, manager=None):
        self.apc_manager = manager if manager is not None else FakeManager()
        self._generation_batch = []
        self._prompt_batch = None
        self._unprocessed_sequences = list(sequences)
        self.admitted = []

    def _apc_extra_hash(self, prompt_kwargs):
        return 0


SHARED = list(range(1000, 1512))  # 512 shared tokens
A = _seq("a", SHARED + [1, 2, 3])
B = _seq("b", SHARED + [7, 8, 9])
C = _seq("c", SHARED + [4, 5, 6])
LONER = _seq("z", list(range(5000, 5512)))


def test_lcp():
    assert fg._lcp([1, 2, 3], [1, 2, 4]) == 2
    assert fg._lcp([], [1]) == 0
    long = list(range(20000))
    div = long[:9000] + [-1] + long[9001:]
    assert fg._lcp(long, div) == 9000
    assert fg._lcp(long, long) == 20000


def test_holds_uncovered_sibling():
    g = FakeGen([A, B])
    assert fg._keep_count(g) == 1


def test_head_is_never_held():
    g = FakeGen([A])
    assert fg._keep_count(g) is None


def test_truncates_at_first_held_follower():
    g = FakeGen([A, B, C])
    assert fg._keep_count(g) == 1


def test_unrelated_follower_admits():
    g = FakeGen([A, LONER])
    assert fg._keep_count(g) is None


def test_unrelated_head_then_siblings_cut_after_leader():
    g = FakeGen([LONER, A, B])
    assert fg._keep_count(g) == 2


def test_block_coverage_admits():
    g = FakeGen([A, B])
    g.apc_manager.add_block_chain(SHARED)
    assert fg._keep_count(g) is None


def test_exact_coverage_admits():
    g = FakeGen([A, B])
    g.apc_manager.add_exact(SHARED)
    assert fg._keep_count(g) is None


def test_anchor_coverage_admits():
    g = FakeGen([A, B])
    g.apc_manager.add_anchor(SHARED)
    assert fg._keep_count(g) is None


def test_coverage_under_other_extra_hash_does_not_count():
    g = FakeGen([A, B])
    g.apc_manager.add_exact(SHARED, extra_hash=99)
    assert fg._keep_count(g) == 1


def test_short_shared_prefix_admits():
    short = list(range(100, 164))
    g = FakeGen([_seq("a", short + [1]), _seq("b", short + [2])])
    assert fg._keep_count(g) is None


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_APC_FRESH_WAIT_MS", "0")
    g = FakeGen([A, B])
    assert fg._keep_count(g) is None


def test_hold_ceiling_admits_cold(monkeypatch, caplog):
    clock = [0.0]
    monkeypatch.setattr(
        fg, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setenv("GMLX_APC_FRESH_WAIT_MS", "100")
    g = FakeGen([A, B])
    assert fg._keep_count(g) == 1
    clock[0] = 0.2
    with caplog.at_level("WARNING"):
        assert fg._keep_count(g) is None
    assert any("ceiling" in r.message for r in caplog.records)


def test_no_hold_while_prompt_batch_live():
    g = FakeGen([A, B])
    g._prompt_batch = object()
    assert fg._keep_count(g) is None


def test_no_hold_without_manager():
    g = FakeGen([A, B], manager=False)
    g.apc_manager = None
    assert fg._keep_count(g) is None


def test_held_state_pruned_when_request_leaves():
    g = FakeGen([A, B])
    assert fg._keep_count(g) == 1
    g._unprocessed_sequences = [A]
    fg._keep_count(g)
    assert "b" not in g._kq_fresh_held


def _fake_next(self, **kw):
    n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
    for s in self._unprocessed_sequences[:n]:
        self.admitted.append(s[0])
    self._unprocessed_sequences = self._unprocessed_sequences[n:]
    return [], []


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    fg.install_fresh_admission_gate()
    yield ar.BatchGenerator._next


def test_install_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_APC_FRESH_WAIT_MS", "0")
    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    fg.install_fresh_admission_gate()
    assert ar.BatchGenerator._next is _fake_next


def test_wrapper_serializes_siblings(gated):
    g = FakeGen([A, B])
    gated(g)
    assert g.admitted == ["a"]
    assert [s[0] for s in g._unprocessed_sequences] == ["b"]
    g.apc_manager.add_block_chain(SHARED)
    gated(g)
    assert g.admitted == ["a", "b"]


def test_wrapper_restores_tail_ahead_of_arrivals(gated, monkeypatch):
    g = FakeGen([A, B, C])

    def _next_with_arrival(self, **kw):
        out = _fake_next(self, **kw)
        self._unprocessed_sequences.append(_seq("late", [1, 2]))
        return out

    monkeypatch.setattr(ar.BatchGenerator, "_next", _next_with_arrival)
    fg.install_fresh_admission_gate()
    ar.BatchGenerator._next(g)
    assert g.admitted == ["a"]
    assert [s[0] for s in g._unprocessed_sequences] == ["b", "c", "late"]


def test_wrapper_decision_failure_degrades(gated, monkeypatch):
    def _boom(gen):
        raise RuntimeError("peek broke")

    monkeypatch.setattr(fg, "_keep_count", _boom)
    g = FakeGen([A, B])
    gated(g)
    assert g.admitted == ["a", "b"]


def test_covered_len_prefers_longest_tier():
    m = FakeManager()
    m.add_block_chain(SHARED[:256])
    m.add_exact(SHARED[:300])
    m.add_anchor(SHARED[:400])
    ids = SHARED + [1, 2]
    assert fg._covered_len(m, ids, 0) == 400


def test_fresh_stats_counts_holds():
    before = fg.fresh_stats()["holds"]
    g = FakeGen([A, B])
    fg._keep_count(g)
    st = fg.fresh_stats()
    assert st["holds"] == before + 1
    assert "shares" in st["last_hold_reason"]
