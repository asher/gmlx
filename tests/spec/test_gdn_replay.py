"""Records form of the fused GDN verify on a tiny random qwen3_5 target.

At any batch size the verify scan stores each position's (delta, key, g)
record instead of the state after every position, and a rejected round
leaves each GDN cache as a pending replay, with each row's own record count,
that the next fused kernel runs in its prologue.
The replay repeats the scan's own float ops, so states, logits and tokens
match the per-position states form bit for bit.
"""

import os

import mlx.core as mx
import pytest

import gmlx.spec.speculative as sp
import gmlx.upstream.gdn_patches as gp
from gmlx.models.qwen35.gdn import OwnedQwen3_5GatedDeltaNet, prepare_gdn

from test_dflash2_drafter import BLOCK, N_GEN, _drafter, _engine_walk, _target

pytestmark = [
    pytest.mark.skipif(
        bool(os.environ.get("KQUANT_FORCE_CPU")),
        reason="the fused GDN kernels are Metal-only"),
    pytest.mark.skipif(gp._gdn_verify_rec_kernel is None,
                       reason="no Metal device"),
]

PROMPT = [[1, 2, 3, 4, 5]]
BLOCK_IDS = [[11, 22, 33, 44]]


def _armed(records):
    lm = _target()
    prepare_gdn(lm)
    for m in _gdns(lm):
        m._gdn_records = records
    return lm


def _gdns(lm):
    return [m for m in lm.modules() if isinstance(m, OwnedQwen3_5GatedDeltaNet)]


def _ssm(cache):
    return [c for c in cache if not c.is_trimmable()]


def _prefilled(lm):
    cache = lm.make_cache()
    mx.eval(lm(mx.array(PROMPT), cache=cache).logits)
    return cache


def _verify(lm, cache):
    hid, _, sink = lm.speculative_verify_hidden(mx.array(BLOCK_IDS), cache)
    mx.eval(hid, sink)
    return hid, sink


def test_prepare_gdn_arms_records_on_the_owned_tree(monkeypatch):
    monkeypatch.delenv("GMLX_GDN_REPLAY", raising=False)
    lm = _target()
    prepare_gdn(lm)
    assert all(m._gdn_records for m in _gdns(lm))
    monkeypatch.setenv("GMLX_GDN_REPLAY", "0")
    prepare_gdn(lm)
    assert not any(m._gdn_records for m in _gdns(lm))


def test_replayed_states_equal_the_stored_states():
    rec_lm, st_lm = _armed(True), _armed(False)
    rec_hid, rec_sink = _verify(rec_lm, _prefilled(rec_lm))
    st_hid, st_sink = _verify(st_lm, _prefilled(st_lm))
    assert mx.array_equal(rec_hid, st_hid)
    S = len(BLOCK_IDS[0])
    for rec_entry, st_entry in zip(rec_sink, st_sink, strict=True):
        rec, states = rec_entry[11], st_entry[11]
        assert isinstance(rec, gp.GdnRecords)
        for t in range(S - 1):
            got = gp.gdn_replay_state(rec.base, rec.records, t + 1)
            assert mx.array_equal(got, states[:, t])


@pytest.mark.parametrize("accepted", [0, 2, 3])
def test_rollback_then_decode_and_verify_match_the_states_form(accepted):
    """A rejection leaves a pending replay: the decode step and the next
    verify take it into their prologue, and a plain reader of the cache
    gets the same state through the lazy replay."""
    S = len(BLOCK_IDS[0])
    arms = []
    for records in (True, False):
        lm = _armed(records)
        cache = _prefilled(lm)
        _, sink = _verify(lm, cache)
        lm.rollback_speculative_cache(cache, sink, accepted, S)
        states = [c[1] for c in _ssm(cache)]
        step = lm(mx.array([[7]]), cache=cache).logits
        hid, _ = _verify(lm, cache)
        mx.eval(states, step)
        arms.append((states, step, hid, [c[1] for c in _ssm(cache)]))
    (r_states, r_step, r_hid, r_after), (s_states, s_step, s_hid, s_after) = arms
    for a, b in zip(r_states, s_states, strict=True):
        assert mx.array_equal(a, b)
    assert mx.array_equal(r_step, s_step)
    assert mx.array_equal(r_hid, s_hid)
    for a, b in zip(r_after, s_after, strict=True):
        assert mx.array_equal(a, b)


def test_a_full_accept_leaves_the_replay_of_every_position():
    lm = _armed(True)
    cache = _prefilled(lm)
    _, sink = _verify(lm, cache)
    S = len(BLOCK_IDS[0])
    for c, entry in zip(_ssm(cache), sink, strict=True):
        state, base, records, steps = c._gdn_replay
        assert state is c[1] and steps.tolist() == [S]
        assert base is entry[11].base


def test_a_replaced_state_voids_the_pending_replay():
    lm = _armed(True)
    cache = _prefilled(lm)
    _verify(lm, cache)
    c = _ssm(cache)[0]
    c[1] = mx.zeros_like(c[1])
    assert gp.take_gdn_replay(c) is None
    assert c._gdn_replay is None


def test_a_snapshot_restore_keeps_the_pending_replay():
    lm = _armed(True)
    cache = _prefilled(lm)
    _verify(lm, cache)
    snap = sp._cache_snapshot(cache)
    assert snap is not None
    _verify(lm, cache)
    sp._cache_restore(snap)
    for c in _ssm(cache):
        assert c._gdn_replay[0] is c[1]


def test_engine_rounds_match_the_states_form():
    out = []
    for records in (True, False):
        lm = _armed(records)
        out.append(_engine_walk(lm, _drafter(lm), mx.array(PROMPT), N_GEN))
    assert out[0] == out[1]
    assert len(out[0]) >= N_GEN and BLOCK > 1


# -- batches -------------------------------------------------------------
#
# The serve batch: each row prefilled alone, every layer merged into the
# batch classes (left padding on the attention caches). Rows accept
# different counts, so each row replays its own number of records.

ROWS = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19]]
JOIN = [21, 22, 23, 24, 25, 26]


def _row_cache(lm, ids):
    cache = lm.make_cache()
    mx.eval(lm(mx.array([ids]), cache=cache).logits)
    return cache


def _merged(lm, rows):
    per_row = [_row_cache(lm, ids) for ids in rows]
    cache = [type(per_row[0][i]).merge([r[i] for r in per_row])
             for i in range(len(per_row[0]))]
    mx.eval([c.state for c in cache])
    return cache


def _blocks(B, round_, S=4):
    return mx.array([[(37 * b + 11 * round_ + 5 * t) % 60 + 3 for t in range(S)]
                     for b in range(B)])


def _verify_rows(lm, cache, ids):
    hid, _, sink = lm.speculative_verify_hidden(ids, cache)
    mx.eval(hid, sink)
    return hid, sink


def test_a_batch_verify_takes_the_records_form():
    lm = _armed(True)
    cache = _merged(lm, ROWS[:3])
    _, sink = _verify_rows(lm, cache, _blocks(3, 0))
    assert sink and all(isinstance(e[11], gp.GdnRecords) for e in sink)
    for c, e in zip(_ssm(cache), sink, strict=True):
        state, base, records, steps = c._gdn_replay
        assert state is c[1] and base is e[11].base
        assert steps.tolist() == [4, 4, 4]


@pytest.mark.parametrize("steps", [[1, 1, 1], [3, 1, 2], [4, 2, 1]])
def test_batch_replay_equals_the_stored_states_per_row(steps):
    rec_lm, st_lm = _armed(True), _armed(False)
    rec_hid, rec_sink = _verify_rows(rec_lm, _merged(rec_lm, ROWS[:3]), _blocks(3, 0))
    st_hid, st_sink = _verify_rows(st_lm, _merged(st_lm, ROWS[:3]), _blocks(3, 0))
    assert mx.array_equal(rec_hid, st_hid)
    for rec_entry, st_entry in zip(rec_sink, st_sink, strict=True):
        rec, states = rec_entry[11], st_entry[11]
        got = gp.gdn_replay_state(rec.base, rec.records, steps)
        for b, t in enumerate(steps):
            want = states[b, t - 1] if t < 4 else None
            if want is not None:
                assert mx.array_equal(got[b], want)


def _two_arms(run):
    out = []
    for records in (True, False):
        lm = _armed(records)
        out.append(run(lm))
    return out


def _equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert mx.array_equal(x, y)


@pytest.mark.parametrize("accepted", [[0, 2, 3], [3, 0, 1], [3, 3, 0], [1, 1, 1]])
def test_ragged_rollback_then_decode_and_verify_match_the_states_form(accepted):
    """Rows accept different counts (a fully accepted row beside rejected
    ones, as a mixed round's padded plain rows sit beside speculating
    rows): the states, the next decode step and the next verify match the
    states form bit for bit."""
    def run(lm):
        cache = _merged(lm, ROWS[:3])
        _, sink = _verify_rows(lm, cache, _blocks(3, 0))
        lm.rollback_speculative_cache(cache, sink, accepted, 4)
        states = [c[1] for c in _ssm(cache)]
        convs = [c[0] for c in _ssm(cache)]
        step = lm(mx.array([[7], [8], [9]]), cache=cache).logits
        hid, _ = _verify_rows(lm, cache, _blocks(3, 1))
        after = [c[1] for c in _ssm(cache)]
        mx.eval(states, convs, step, after)
        return states + convs + [step, hid] + after
    rec, st = _two_arms(run)
    _equal(rec, st)


def test_rows_leave_and_join_with_a_pending_replay():
    """A row filtered out and a row extended in between rounds: the pending
    replay is taken by identity only, so the cache reads the replayed
    state and the batch keeps the states form's numbers."""
    def run(lm):
        cache = _merged(lm, ROWS[:3])
        _, sink = _verify_rows(lm, cache, _blocks(3, 0))
        lm.rollback_speculative_cache(cache, sink, [2, 0, 1], 4)
        keep = mx.array([0, 2])
        for c in cache:
            c.filter(keep)
        hid1, sink = _verify_rows(lm, cache, _blocks(2, 1))
        lm.rollback_speculative_cache(cache, sink, [1, 3], 4)
        new = _merged(lm, [JOIN])
        for c, n in zip(cache, new):
            c.extend(n)
        hid2, sink = _verify_rows(lm, cache, _blocks(3, 2))
        lm.rollback_speculative_cache(cache, sink, [0, 2, 3], 4)
        step = lm(mx.array([[7], [8], [9]]), cache=cache).logits
        mx.eval(step)
        return [hid1, hid2, step] + [c[1] for c in _ssm(cache)]
    rec, st = _two_arms(run)
    _equal(rec, st)


@pytest.mark.parametrize("B", [2, 3, 4])
def test_greedy_batch_rounds_match_the_states_form(B):
    """Greedy rounds on a batch: each row drafts its own greedy chain (plain
    decode alone) with a row-dependent wrong token, so rows accept
    different counts. Tokens, accepts and the final states match the
    states form."""
    def oracle(lm, ids, n):
        cache = lm.make_cache()
        out = []
        x = mx.array([ids])
        for _ in range(n):
            t = int(mx.argmax(lm(x, cache=cache).logits[0, -1]).item())
            out.append(t)
            x = mx.array([[t]])
        return out

    def run(lm):
        chains = [oracle(lm, ids, 40) for ids in ROWS[:B]]
        cache = _merged(lm, ROWS[:B])
        # The first token comes off the prefill and is not yet in the cache.
        last = [c[0] for c in chains]
        toks = [[c[0]] for c in chains]
        accs = []
        for r in range(8):
            drafts = []
            for b in range(B):
                pos = len(toks[b])
                d = chains[b][pos:pos + 3]
                bad = (b + r) % 4
                if bad < 3:
                    d[bad] = (d[bad] + 1) % 60 + 3
                drafts.append([last[b]] + d)
            hid, sink = _verify_rows(lm, cache, mx.array(drafts))
            pred = mx.argmax(lm.speculative_logits_from_hidden(hid), axis=-1).tolist()
            acc = []
            for b in range(B):
                a = 0
                while a < 3 and drafts[b][a + 1] == pred[b][a]:
                    a += 1
                acc.append(a)
                toks[b] += drafts[b][1:a + 1] + [pred[b][a]]
                last[b] = pred[b][a]
            accs.append(acc)
            if any(a < 3 for a in acc):
                lm.rollback_speculative_cache(cache, sink, acc, 4)
        states = [c[1] for c in _ssm(cache)]
        mx.eval(states)
        return toks, accs, states
    (rt, ra, rs), (st, sa, ss) = _two_arms(run)
    assert ra == sa and rt == st
    assert len({a for acc in ra for a in acc}) > 1
    _equal(rs, ss)
