"""Auto pacing resolver: one case per conjunct, hysteresis, dwell, grace,
queue band, gate coupling. Pure logic over a fake generator with an
injected clock (resolve takes ``now``; nothing sleeps)."""

import gmlx.auto_ratio as ar


class FakeBatch:
    def __init__(self, uids):
        self.uids = list(uids)

    def __len__(self):
        return len(self.uids)


class FakeGen:
    def __init__(self, rows=(101,), pending=()):
        self._generation_batch = FakeBatch(rows)
        self._prompt_batch = None
        self._unprocessed_sequences = [
            (u, [0] * 10, 64, {}, None, None) for u in pending]
        self._kq_last_chunk_time = 0.0
        self._prompt_time_counter = 0.0


def _feed_step(g, s=0.05, width=1):
    ar.observe(g, s, 0.0, width)


def _incumbent_gen(now=0.0):
    """Row 101 admitted at t=now, step cost fed, expensive chunk seen."""
    g = FakeGen(rows=(101,), pending=())
    ar.resolve(g, now)          # stamps admitted_at[101]=now
    _feed_step(g)
    g._kq_last_chunk_time = 1.0  # C = (1.0 - 0.05) / 0.05 = 19
    return g


def _add_waiter(g, uid=7):
    g._unprocessed_sequences.append((uid, [0] * 10, 64, {}, None, None))


def test_no_waiters_resolves_zero():
    g = _incumbent_gen()
    assert ar.resolve(g, 1.0) == 0.0


def test_incumbent_plus_costly_chunk_paces():
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    assert ar.resolve(g, 1.0) == ar.paced_ratio()


def test_burst_has_no_incumbent():
    # every row admitted the same tick the waiters were first seen
    g = FakeGen(rows=(1, 2), pending=(3, 4))
    _feed_step(g, width=2)
    g._kq_last_chunk_time = 1.0
    assert ar.resolve(g, 5.0) == 0.0  # tie rule: same-tick stamp loses


def test_grace_blocks_fresh_incumbent():
    g = FakeGen(rows=(101,))
    ar.resolve(g, 0.0)
    _feed_step(g)
    g._kq_last_chunk_time = 1.0
    _add_waiter(g)
    # waiter arrives 0.3s after admission, inside the 500ms grace
    assert ar.resolve(g, 0.3) == 0.0
    # the comparison is fixed at arrival: the same waiter never grants
    # incumbency however long both wait
    assert ar.resolve(g, 3.0) == 0.0
    # a waiter that arrives later (after the first was admitted) does
    # see the long-lived row as an incumbent
    g._unprocessed_sequences.clear()
    ar.resolve(g, 3.5)
    _add_waiter(g, 8)
    assert ar.resolve(g, 4.0) == ar.paced_ratio()


def test_cheap_chunk_stays_stock():
    g = _incumbent_gen(0.0)
    g._kq_last_chunk_time = 0.08  # C = (0.08 - 0.05) / 0.05 = 0.6 <= 1
    _add_waiter(g)
    assert ar.resolve(g, 1.0) == 0.0


def test_c_hysteresis_and_dwell():
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    assert ar.resolve(g, 1.0) == ar.paced_ratio()  # C=19, state on
    # C falls inside the band (< C_on but > C_on/hyst): stays on
    g._kq_last_chunk_time = 0.09  # C = 0.8
    assert ar.resolve(g, 2.5) == ar.paced_ratio()
    # C below C_on/hyst but dwell not elapsed since the flip at t=1.0
    g._kq_last_chunk_time = 0.06  # C = 0.2 < 1/1.5
    assert ar.resolve(g, 1.5) == ar.paced_ratio()
    # dwell elapsed: flips off
    assert ar.resolve(g, 2.6) == 0.0


def test_queue_band_hysteresis(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_QUEUE_MAX", "2")
    g = _incumbent_gen(0.0)
    for uid in (7, 8, 9):
        _add_waiter(g, uid)
    assert ar.resolve(g, 1.0) == 0.0        # 3 > 2: stand down
    g._unprocessed_sequences.pop()
    assert ar.resolve(g, 1.1) == 0.0        # 2 == 2: band holds
    g._unprocessed_sequences.pop()
    assert ar.resolve(g, 1.2) == ar.paced_ratio()  # 1 < 2: resume


def test_queue_band_initializes_permissive():
    # first contact at pending == queue_max must pace (the certified
    # second-client case)
    g = _incumbent_gen(0.0)
    _add_waiter(g)  # pending == queue_max == 1
    assert ar.resolve(g, 1.0) == ar.paced_ratio()


def test_deadline_forces_stock():
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    assert ar.resolve(g, 1.0) == ar.paced_ratio()  # stamps first_seen at 1.0
    assert ar.resolve(g, 5.0) == ar.paced_ratio()
    assert ar.resolve(g, 11.5) == 0.0  # age 10.5 > deadline 10


def test_gate_deferred_time_excluded_from_deadline():
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    ar.resolve(g, 1.0)
    g._kq_admit_deferred_s = {7: 5.0}
    # age 10.5 minus 5 gate-deferred = 5.5, inside the deadline
    assert ar.resolve(g, 11.5) == ar.paced_ratio()


def test_gate_decline_excludes_pending_count():
    g = _incumbent_gen(0.0)
    for uid in (7, 8):
        _add_waiter(g, uid)
    assert ar.resolve(g, 1.0) == 0.0  # 2 > queue_max 1: stand down
    # gate declined on the previous tick: waiters leave the pending count
    # and the band resumes below queue_max
    g._kq_admit_last_decline = 1.05
    assert ar.resolve(g, 1.1) == ar.paced_ratio()


def test_width_change_freezes_c_state():
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    assert ar.resolve(g, 1.0) == ar.paced_ratio()
    # admission widened the batch; width-2 bucket is only seeded, so the
    # C state holds even though the chunk now looks cheap
    g._generation_batch.uids.append(102)
    g._kq_last_chunk_time = 0.01
    assert ar.resolve(g, 2.5) == ar.paced_ratio()
    # real width-2 samples land: state re-evaluates and flips off
    _feed_step(g, width=2)
    assert ar.resolve(g, 4.0) == 0.0


def test_kill_switch_resolves_static_paced(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_AUTO", "0")
    g = FakeGen()
    assert ar.resolve(g, 0.0) == ar.paced_ratio()


def test_observe_skips_post_chunk_bracket():
    g = FakeGen()
    ar.observe(g, 0.4, 0.4, 1)    # chunk tick: poisons the next bracket
    ar.observe(g, 9.9, 0.0, 1)    # poisoned bracket: dropped
    ar.observe(g, 0.05, 0.0, 1)   # real step
    st = g._kq_auto
    assert st.s_by_width[1] == 0.05
    assert st.s_samples[1] == 1


def test_floor_env_derives_both_constants(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_PREFILL_FLOOR", "0.75")
    assert ar.paced_ratio() == 3.0
    assert abs(ar.c_threshold() - 1.0 / 3.0) < 1e-9
