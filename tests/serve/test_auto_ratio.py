"""Auto pacing resolver: one case per conjunct, hysteresis, dwell, grace,
deadline accrual, gate coupling. Pure logic over a fake generator with an
injected clock (resolve takes ``now``; nothing sleeps)."""

import gmlx.serve.auto_ratio as ar


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


def test_burst_stays_paced():
    # The regime C burst regression: a burst of queued waiters is not
    # queue pressure (they promote into the prompt batch within a tick
    # or two), and standing down on count was a race against promotion.
    # A burst paces; only deadline-aged queued waiters abandon the floor.
    g = _incumbent_gen(0.0)
    for uid in (7, 8, 9, 10):
        _add_waiter(g, uid)
    assert ar.resolve(g, 1.0) == ar.paced_ratio()
    # burst promotes into the prompt batch: still paced
    g._unprocessed_sequences.clear()
    g._prompt_batch = FakeBatch([7, 8, 9, 10])
    assert ar.resolve(g, 2.0) == ar.paced_ratio()


def test_deadline_forces_stock():
    # A waiter queued behind a live paced train accrues pacing-attributable
    # age (bounded 1 s per tick) and forces ratio 0 past the deadline.
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    assert ar.resolve(g, 1.0) == ar.paced_ratio()
    g._prompt_batch = FakeBatch([9])   # a train is live; waiter 7 queued
    t = 1.0
    while t < 11.0:                    # accrues 1.0 per 1 s tick
        t += 1.0
        assert ar.resolve(g, t) == ar.paced_ratio()
    assert ar.resolve(g, 12.0) == 0.0  # paced wait 11 > deadline 10
    # stood down is monotone: no further accrual, value holds past deadline
    assert ar.resolve(g, 13.0) == 0.0


def test_capacity_wait_does_not_age_toward_deadline():
    # No live train means the waiter is blocked by capacity (full decode
    # batch), not pacing; ratio 0 could not admit it sooner, so it never
    # ages and the floor holds indefinitely.
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    t = 0.0
    while t < 30.0:
        t += 1.0
        assert ar.resolve(g, t) == ar.paced_ratio()


def test_pacing_survives_promotion_to_prompt_batch():
    # The regime B regression: the waiter's prompt is admitted into the
    # prompt batch one tick after pacing starts. The chunk train is still
    # competing prefill work; dropping pacing at promotion unpaces the
    # whole prefill and puts the incumbent at the unpaced floor.
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    assert ar.resolve(g, 1.0) == ar.paced_ratio()
    g._unprocessed_sequences.clear()
    g._prompt_batch = FakeBatch([7])
    assert ar.resolve(g, 1.5) == ar.paced_ratio()
    # prompt batch completes, rows reach decode: back to no waiters
    g._prompt_batch = None
    g._generation_batch = FakeBatch([101, 7])
    assert ar.resolve(g, 2.0) == 0.0


def test_prefilling_waiter_does_not_age_toward_deadline():
    # A deep prompt's paced chunk train can far outlive deadline_s; its
    # TTFT is bounded by the paced ratio itself ((1+r)x), so only queued
    # waiters age toward the deadline.
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    ar.resolve(g, 1.0)
    g._unprocessed_sequences.clear()
    g._prompt_batch = FakeBatch([7])
    assert ar.resolve(g, 30.0) == ar.paced_ratio()


def test_queued_waiter_behind_prompt_batch_still_ages():
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    ar.resolve(g, 1.0)
    g._unprocessed_sequences.clear()
    g._prompt_batch = FakeBatch([7])   # waiter 7 promoted; its train paced
    _add_waiter(g, 8)                  # 8 queued behind it
    t = 1.0
    while t < 11.0:
        t += 1.0
        assert ar.resolve(g, t) == ar.paced_ratio()
    assert ar.resolve(g, 13.0) == 0.0  # 8's paced wait > deadline 10


def test_gate_hidden_ticks_freeze_deadline_and_keep_stamps():
    # The gate hides pending on declined ticks. Hidden ticks accrue no
    # paced wait (the gate, not pacing, is why the waiter waits), and
    # stamps survive so incumbency and accrued age resume, not reset.
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    g._prompt_batch = FakeBatch([9])
    for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        assert ar.resolve(g, t) == ar.paced_ratio()
    st = g._kq_auto
    assert st.paced_wait[7] == 5.0
    seen = st.first_seen[7]
    # gate declines: pending stashed empty, deferred dict marks uid 7
    stash = g._unprocessed_sequences
    g._unprocessed_sequences = []
    g._kq_admit_deferred_s = {7: 0.0}
    for t in (7.0, 8.0, 20.0):
        ar.resolve(g, t)
    assert st.paced_wait[7] == 5.0     # frozen, not accrued, not wiped
    assert st.first_seen[7] == seen    # arrival stamp survives hiding
    # gate admits: pending restored; accrual resumes where it left off
    g._unprocessed_sequences = stash
    g._kq_admit_deferred_s = {}
    for t in (21.0, 22.0, 23.0, 24.0, 25.0):
        assert ar.resolve(g, t) == ar.paced_ratio()
    assert ar.resolve(g, 27.0) == 0.0  # 5 + 6 accrued > deadline 10


def test_suppressed_transition_logs_when_rate_window_opens(caplog):
    # A transition inside the rate-limit window is deferred, not lost:
    # the persisted state logs on a later tick once the window opens.
    import logging
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    with caplog.at_level(logging.INFO, logger="gmlx.serve.auto_ratio"):
        ar.resolve(g, 1.0)                  # logs: pacing on
        g._unprocessed_sequences.clear()
        ar.resolve(g, 1.5)                  # off, suppressed (< 1 s)
        assert "pacing off" not in caplog.text
        ar.resolve(g, 2.1)                  # still off, window open: logs
        assert "pacing off" in caplog.text


def test_log_rate_env_zero_logs_every_transition(caplog, monkeypatch):
    import logging
    monkeypatch.setenv("GMLX_DECODE_PREFILL_LOG_S", "0")
    g = _incumbent_gen(0.0)
    _add_waiter(g)
    with caplog.at_level(logging.INFO, logger="gmlx.serve.auto_ratio"):
        ar.resolve(g, 1.0)
        g._unprocessed_sequences.clear()
        ar.resolve(g, 1.2)
    assert "pacing on" in caplog.text and "pacing off" in caplog.text


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
