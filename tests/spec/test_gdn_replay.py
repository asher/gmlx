"""Records form of the fused GDN verify on a tiny random qwen3_5 target.

At B=1 the verify scan stores each position's (delta, key, g) record instead
of the state after every position, and a rejected round leaves each GDN
cache as a pending replay that the next fused kernel runs in its prologue.
The replay repeats the scan's own float ops, so states, logits and tokens
match the per-position states form bit for bit.
"""

import os

import mlx.core as mx
import pytest

import gmlx.upstream.gdn_patches as gp
from gmlx.models.qwen35.gdn import OwnedQwen3_5GatedDeltaNet, prepare_gdn
from gmlx.upstream.gdn_patches import _patch_gated_delta_tiled_v

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


@pytest.fixture(scope="module", autouse=True)
def _tiled_v():
    # Every qwen3.5 GGUF load tiles mlx-lm's K->V map; the prefill takes it.
    _patch_gated_delta_tiled_v()


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
        assert state is c[1] and steps == S
        assert base is entry[11].base


def test_a_replaced_state_voids_the_pending_replay():
    lm = _armed(True)
    cache = _prefilled(lm)
    _verify(lm, cache)
    c = _ssm(cache)[0]
    c[1] = mx.zeros_like(c[1])
    assert gp.take_gdn_replay(c) is None
    assert c._gdn_replay is None


def test_engine_rounds_match_the_states_form():
    out = []
    for records in (True, False):
        lm = _armed(records)
        out.append(_engine_walk(lm, _drafter(lm), mx.array(PROMPT), N_GEN))
    assert out[0] == out[1]
    assert len(out[0]) >= N_GEN and BLOCK > 1
