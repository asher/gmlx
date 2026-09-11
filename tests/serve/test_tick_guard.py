"""Engine-tick OOM containment: ledger, retry, retire-requeue, red exit."""

import types

import pytest

from mlx_vlm.generate import ar

import gmlx.eval_guard as eg
import gmlx.serve.tick_guard as tg


class FakeGB:
    def __init__(self, uids, num_tokens):
        self.uids = list(uids)
        self._all_uids = list(uids)
        self._num_tokens = list(num_tokens)

    def __len__(self):
        return len(self.uids)


class FakeGen:
    def __init__(self, pending, gb):
        self._unprocessed_sequences = list(pending)
        self._generation_batch = gb
        self.removed = []

    def remove(self, uid):
        self.removed.append(uid)
        if uid in self._generation_batch.uids:
            self._generation_batch.uids.remove(uid)
            return True
        return False


def resp(uid, token, finish=None):
    return types.SimpleNamespace(uid=uid, token=token,
                                 finish_reason=finish)


MEM_ERR = RuntimeError(
    "[metal::malloc] Attempting to allocate 103918075904 bytes")

OOM_ERR = RuntimeError(
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).")

TIMEOUT_ERR = RuntimeError(
    "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
    "(00000002:kIOGPUCommandBufferCallbackErrorTimeout).")

PAGE_FAULT_ERR = RuntimeError(
    "[METAL] Command buffer execution failed: Caused GPU Page Fault Error "
    "(00000003:kIOGPUCommandBufferCallbackErrorPageFault).")


@pytest.fixture
def wrapped(monkeypatch):
    """Install the guard over a scripted _next; yields (gen, script)."""
    monkeypatch.delenv("GMLX_TICK_GUARD", raising=False)
    script = []

    def scripted_next(self, **kw):
        step = script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(ar.BatchGenerator, "_next", scripted_next)
    drains = []
    monkeypatch.setattr(eg, "drain_for", lambda site: drains.append(site))
    assert tg.install_tick_guard() is True
    pending = [(7, [1, 2, 3], 16, {}, None, None),
               (8, [4, 5, 6, 9, 9, 9], 16, {}, None, None)]
    gen = FakeGen(pending, FakeGB([], []))
    return gen, script, drains


def tick(gen, **kw):
    return ar.BatchGenerator._next(gen, **kw)


def test_ledger_harvests_prompts_and_committed_tokens(wrapped):
    gen, script, _ = wrapped
    script.append(([], [resp(7, 11), resp(8, 21)]))
    script.append(([], [resp(7, 12), resp(8, 22, finish="stop")]))
    tick(gen)
    tick(gen)
    st = gen._kq_tick_guard
    assert st.ledger[7].prompt_ids == [1, 2, 3]
    assert st.ledger[7].committed == [11, 12]
    assert 8 not in st.ledger  # finished rows leave the ledger
    assert st.fail_streak == 0


def test_first_failure_drains_reclaims_and_retries(wrapped, monkeypatch):
    gen, script, drains = wrapped
    cleared = []
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: cleared.append(1))
    script.append(MEM_ERR)
    script.append(([], [resp(7, 11)]))
    assert tick(gen) == ([], [])  # contained: empty tick
    assert drains == ["engine-tick"] and cleared == [1]
    assert gen.removed == []      # no shed on first failure
    tick(gen)
    assert gen._kq_tick_guard.fail_streak == 0  # success resets


def test_repeat_failure_retires_largest_and_requeues_same_uid(
        wrapped, monkeypatch):
    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    # both rows decoding; row 8 is larger (prompt 6 + 5 emitted)
    gen._generation_batch = FakeGB([7, 8], [3, 5])
    script.append(([], [resp(7, 11), resp(8, 21)]))
    tick(gen)
    script.append(MEM_ERR)
    script.append(MEM_ERR)
    tick(gen)
    assert gen.removed == []
    tick(gen)
    assert gen.removed == [8]
    uid, replay, remaining, kw, lp, tc = gen._unprocessed_sequences[0]
    assert uid == 8
    assert replay == [4, 5, 6, 9, 9, 9, 21]  # prompt + delivered
    assert remaining == 15                    # 16 - 1 delivered
    assert 8 in gen._kq_tick_guard.rebuilt


def test_third_strike_fails_row_permanently(wrapped, monkeypatch):
    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append((uid, info))])
    gen._generation_batch = FakeGB([8], [2])
    gen._kq_tick_guard = st = tg._TickState()
    st.rebuilt.add(8)
    script.append(([], [resp(8, 21)]))
    tick(gen)
    script.extend([MEM_ERR, MEM_ERR])
    tick(gen)
    tick(gen)
    assert gen.removed == [8]
    assert failed and failed[0][0] == 8
    assert failed[0][1]["delivered"] == 1
    assert 8 not in st.ledger


def test_non_memory_errors_propagate(wrapped):
    gen, script, drains = wrapped
    script.append(RuntimeError("unrelated explosion"))
    with pytest.raises(RuntimeError, match="unrelated"):
        tick(gen)
    assert drains == []


def test_inject_throw_is_contained(wrapped, monkeypatch):
    gen, script, drains = wrapped
    monkeypatch.setenv("GMLX_OOM_INJECT", "throw@2")
    tg._INJECT_FIRED.clear()
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    script.append(([], [resp(7, 11)]))
    tick(gen)                              # tick 1: normal
    assert tick(gen) == ([], [])           # tick 2: synthetic, contained
    assert drains == ["engine-tick"]
    script.append(([], [resp(7, 12)]))
    tick(gen)                              # fires once; tick 3 normal
    assert gen._kq_tick_guard.fail_streak == 0


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_TICK_GUARD", "0")
    orig = ar.BatchGenerator._next
    assert tg.install_tick_guard() is False
    assert ar.BatchGenerator._next is orig


def test_classifier_separates_memory_from_gpu_faults():
    """Only the allocator's own words are a memory error. A timeout or a
    page fault says the submitted work was wrong, not that the box is
    full, so the ladder that sheds a row must not see them."""
    for e in (MEM_ERR, OOM_ERR):
        assert tg.is_memory_error(e) and not tg.is_gpu_fault(e), e
    for e in (TIMEOUT_ERR, PAGE_FAULT_ERR):
        assert tg.is_gpu_fault(e) and not tg.is_memory_error(e), e
    other = RuntimeError("shapes do not match")
    assert not tg.is_memory_error(other) and not tg.is_gpu_fault(other)


def test_gpu_fault_is_contained_once_then_propagates(wrapped, monkeypatch):
    """One retry covers a driver hiccup. A second fault propagates with
    the Metal error rather than shedding a row, because reclaiming
    buffers cannot fix a fault in the work itself."""
    gen, script, drains = wrapped
    cleared = []
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: cleared.append(1))
    script.append(TIMEOUT_ERR)
    assert tick(gen) == ([], [])
    assert drains == ["engine-tick"] and cleared == [1]
    assert gen.removed == []

    script.append(TIMEOUT_ERR)
    with pytest.raises(RuntimeError, match="Timeout"):
        tick(gen)
    assert gen.removed == []          # never sheds a row on a fault
    assert drains == ["engine-tick"]  # no second reclaim


def test_gpu_fault_streak_resets_on_a_good_tick(wrapped, monkeypatch):
    gen, script, _ = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    script.append(PAGE_FAULT_ERR)
    tick(gen)
    script.append(([], [resp(7, 11)]))
    tick(gen)
    assert gen._kq_tick_guard.fault_streak == 0
    script.append(PAGE_FAULT_ERR)
    assert tick(gen) == ([], [])      # contained again after recovery


def test_metal_oom_still_takes_the_memory_ladder(wrapped, monkeypatch):
    """The OOM that arrives as a command buffer failure keeps the old
    behaviour: contained, then a row retired and requeued."""
    gen, script, drains = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    gen._generation_batch = FakeGB([7, 8], [4, 40])
    script.append(OOM_ERR)
    tick(gen)
    assert gen.removed == []
    script.append(OOM_ERR)
    tick(gen)
    assert gen.removed == [8]         # largest row retired, as before
