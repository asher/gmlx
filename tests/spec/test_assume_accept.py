"""Assumed accept: a round submits its commit and the next draft behind the
verify, before the walk is read, as if every draft were accepted. The tokens
are the ones the plain order emits, and a round that is not fully accepted,
or a consumer that stops inside a round, leaves the target cache, the drafter
ring and the RNG key as the plain order leaves them."""

import mlx.core as mx
import pytest

import gmlx.spec.speculative as sp

from test_dflash2_drafter import (N_GEN, _annotated, _drafter,
                                  _engine_reference, _target)
from test_early_draft import PROMPT, _plain_rounds, _rng_keys, _run as _early_run, _same  # noqa: F401


@pytest.fixture(autouse=True)
def _plain_round_levers(monkeypatch):
    """The round levers start in their off form: each test turns on the
    ones it compares against the plain order."""
    import gmlx.spec.dflash_drafter as dd

    monkeypatch.setattr(sp, "_ACCEPT_COUNTED", False)
    monkeypatch.setattr(sp, "_VERIFY_AHEAD", 0)
    monkeypatch.setattr(dd, "_COUNTED_SHORT", False)


MODES = [False, True, "sharp"]


def _kw(monkeypatch, sampled):
    """Greedy, the serve sampler, or a near-greedy sampler and draft: a
    random target accepts a whole sampled block only when both are sharp."""
    if sampled == "sharp":
        monkeypatch.setattr(sp, "_STOCH_DRAFT", (1e-4,) + tuple(sp._STOCH_DRAFT[1:]))
        return {"sampler": _annotated(temp=1e-4)}
    return {"sampler": _annotated()} if sampled else {}


def _run(monkeypatch, on, **kw):
    monkeypatch.setattr(sp, "_ASSUME_ACCEPT", on)
    for k in sp._assume_stats:
        sp._assume_stats[k] = 0
    out = _early_run(monkeypatch, False, **kw)
    out["stats"] = dict(sp._assume_stats)
    return out


def test_greedy_rounds_emit_the_plain_chain_on_hits_and_misses(monkeypatch):
    lm = _target()
    ref = _engine_reference(lm, mx.array(PROMPT), N_GEN)
    plain = _run(monkeypatch, False, floor=0.5)
    on = _run(monkeypatch, True, floor=0.5)
    assert on["toks"][:N_GEN] == ref == plain["toks"][:N_GEN]
    _same(plain, on)
    assert not any(plain["stats"].values())
    assert on["stats"]["hit"] + on["stats"]["miss"] > 0


@pytest.mark.parametrize("sampled", MODES)
@pytest.mark.parametrize("n", [24, 64])
def test_long_runs_hit_and_miss_and_keep_the_plain_state(monkeypatch, sampled, n):
    kw = _kw(monkeypatch, sampled)
    plain = _run(monkeypatch, False, n=n, floor=0.5, **kw)
    kw = _kw(monkeypatch, sampled)
    on = _run(monkeypatch, True, n=n, floor=0.5, **kw)
    _same(plain, on)
    if n == 64 and sampled is not True:
        assert on["stats"]["hit"] > 0 and on["stats"]["miss"] > 0


@pytest.mark.parametrize("sampled", MODES)
def test_a_hit_submits_the_next_draft_before_its_round_goes_out(monkeypatch, sampled):
    kw = _kw(monkeypatch, sampled)
    plain = _run(monkeypatch, False, n=64, floor=0.5, **kw)
    kw = _kw(monkeypatch, sampled)
    on = _run(monkeypatch, True, n=64, floor=0.5, **kw)
    # One more draft call per miss and per redraft; a hit's draft started
    # with fewer tokens out than the plain order's.
    extra = on["stats"]["miss"] + on["stats"]["skip"]
    assert len(on["drafts"]) == len(plain["drafts"]) + extra
    assert on["drafts"] != plain["drafts"]


@pytest.mark.parametrize("sampled", MODES)
@pytest.mark.parametrize("stop_after", range(1, 13))
def test_a_stop_inside_a_round_leaves_the_plain_order_state(
        monkeypatch, sampled, stop_after):
    kw = _kw(monkeypatch, sampled)
    plain = _run(monkeypatch, False, stop_after=stop_after, n=64, floor=0.5, **kw)
    kw = _kw(monkeypatch, sampled)
    on = _run(monkeypatch, True, stop_after=stop_after, n=64, floor=0.5, **kw)
    _same(plain, on)


@pytest.mark.parametrize("sampled", MODES)
def test_a_bonus_in_the_skipped_range_redrafts_the_block(monkeypatch, sampled):
    from gmlx.spec.dflash_drafter import DFlash2Drafter

    orig = DFlash2Drafter.note_bonus
    calls = []

    def note_bonus(self, token):
        orig(self, token)
        calls.append(token)
        return len(calls) % 2 == 0

    kw = _kw(monkeypatch, sampled)
    plain = _run(monkeypatch, False, n=64, floor=0.5, **kw)
    monkeypatch.setattr(DFlash2Drafter, "note_bonus", note_bonus)
    kw = _kw(monkeypatch, sampled)
    on = _run(monkeypatch, True, n=64, floor=0.5, **kw)
    _same(plain, on)
    if sampled is not True:
        assert on["stats"]["skip"] > 0


def test_note_bonus_reports_a_token_in_the_skipped_range():
    lm = _target()
    from gmlx.spec import dflash_drafter as dd

    drafter = _drafter(lm)
    drafter.reset(lm)
    vocab = int(drafter.config.vocab_size)
    dd._open_ring(drafter._cache, vocab, [0])
    skip = drafter._head_skip()
    if skip is None:
        pytest.skip("vocabulary too small for a skipped range")
    assert drafter.note_bonus(0)
    assert not drafter.note_bonus(skip[0])


def _close(a, b, exact=True):
    """Same tokens, counters, target cache and RNG key; the ring equal to
    rounding (its rows come from another batch width or key order). Not
    ``exact``: the target cache equal to rounding too."""
    import numpy as np

    assert a["toks"] == b["toks"]
    assert a["counts"] == b["counts"]
    for key in ("target",) + _rng_keys(a, b):
        assert len(a[key]) == len(b[key])
        for x, y in zip(a[key], b[key]):
            assert x.shape == y.shape
            if key == "target" and not exact:
                assert np.allclose(x, y, rtol=1e-5, atol=1e-5)
            else:
                assert np.array_equal(x, y)
    assert len(a["ring"]) == len(b["ring"])
    for x, y in zip(a["ring"], b["ring"]):
        assert x.shape == y.shape and np.allclose(x, y, rtol=2e-2, atol=1e-2)
    assert (a["occ"] is None) == (b["occ"] is None)
    if a["occ"] is not None:
        assert np.array_equal(a["occ"], b["occ"])
    assert not a["pending"] and not b["pending"]


# --- counted: the accepted count read on the GPU, no assumption ---------------

def _run_counted(monkeypatch, on, window=8, **kw):
    """A short window, so the ring is full a few rounds in: the counted
    form reads a full window behind the live rows."""
    import test_early_draft

    monkeypatch.setattr(sp, "_ACCEPT_COUNTED", on)
    monkeypatch.setattr(test_early_draft, "_drafter",
                        lambda lm: _drafter(lm, window=window))
    return _run(monkeypatch, False, **kw)


@pytest.mark.parametrize("sampled", MODES)
@pytest.mark.parametrize("stop_after", [None, 2, 13, 31])
def test_counted_rounds_on_a_ring_shorter_than_the_window(
        monkeypatch, sampled, stop_after):
    """A window the run never fills: without the lever no round is
    counted, with it rounds are, and tokens and state stay the plain
    order's, through a stop too."""
    import gmlx.spec.dflash_drafter as dd

    kw = dict(n=96, floor=0.5, window=4096)
    if stop_after is not None:
        kw["stop_after"] = stop_after
    plain = _run_counted(monkeypatch, False, **kw, **_kw(monkeypatch, sampled))
    off = _run_counted(monkeypatch, True, **kw, **_kw(monkeypatch, sampled))
    assert off["stats"]["counted"] == 0
    monkeypatch.setattr(dd, "_COUNTED_SHORT", True)
    on = _run_counted(monkeypatch, True, **kw, **_kw(monkeypatch, sampled))
    _close(plain, on)
    if stop_after is None:
        assert on["stats"]["counted"] > 0


@pytest.mark.parametrize("sampled", MODES)
def test_counted_rounds_keep_the_plain_tokens_and_state(monkeypatch, sampled):
    plain = _run_counted(monkeypatch, False, n=96, floor=0.5,
                         **_kw(monkeypatch, sampled))
    on = _run_counted(monkeypatch, True, n=96, floor=0.5,
                      **_kw(monkeypatch, sampled))
    _close(plain, on)
    # Every round past the ring's fill is counted, whatever it accepted.
    assert on["stats"]["counted"] > 0
    assert on["stats"]["hit"] == 0


@pytest.mark.parametrize("sampled", MODES)
@pytest.mark.parametrize("stop_after", [1, 2, 3, 5, 8, 13, 21, 30, 31, 32, 33, 40])
def test_a_stop_inside_a_counted_round_leaves_the_plain_order_state(
        monkeypatch, sampled, stop_after):
    plain = _run_counted(monkeypatch, False, stop_after=stop_after, n=96,
                         floor=0.5, **_kw(monkeypatch, sampled))
    on = _run_counted(monkeypatch, True, stop_after=stop_after, n=96,
                      floor=0.5, **_kw(monkeypatch, sampled))
    _close(plain, on)


@pytest.mark.parametrize("gather", [True, False])
def test_a_counted_block_is_the_block_drafted_after_the_trim(monkeypatch, gather):
    """The counted draft against the plain draft of the same ring, for
    every count of dead rows. Read by row index it has the plain draft's
    keys in the plain order: the same bits."""
    import numpy as np

    import gmlx.spec.dflash_drafter as dd
    from test_dflash2_drafter import BLOCK, CAPTURE

    monkeypatch.setattr(dd, "_COUNTED_GATHER", gather)

    lm = _target()
    drafter = _drafter(lm, window=8)
    lm.set_dflash_capture(CAPTURE)
    drafter.reset(lm)
    hidden = int(drafter._hidden) * (1 + drafter._n_targets)
    window = int(drafter.config.sliding_window)
    mx.random.seed(3)
    drafter.append_context(drafter._captures(
        mx.random.normal((1, window + 3, hidden))))
    rows = mx.random.normal((1, BLOCK, hidden))
    block = mx.array([[5] + [int(drafter.config.mask_token_id)] * (BLOCK - 1)])
    most = BLOCK - 1
    mark = drafter.accept_mark(BLOCK)
    assert mark is not None and drafter.counted_ok(BLOCK, most, BLOCK)
    for dead in range(BLOCK):
        drafter.append_context(drafter._captures(rows))
        assert drafter.counted_ok(BLOCK, most)
        got = drafter._draft_hidden(block, (mx.array(dead), most))
        mx.eval(got)
        kept = drafter.accept_trim(mark, dead)
        want = drafter._draft_hidden(block)
        g = np.array(got.astype(mx.float32))
        w = np.array(want.astype(mx.float32))
        if gather:
            assert np.array_equal(g, w)
        else:
            # Same sums in another key order: equal to rounding.
            assert np.allclose(g, w, rtol=2e-3, atol=2e-4)
        drafter.accept_undo(kept)


def test_a_counted_block_on_a_short_ring_is_the_plain_block_to_rounding(
        monkeypatch):
    """The masked form (GMLX_DFLASH_COUNTED_SHORT) against the plain draft
    of the same ring, for every count of dead rows."""
    import numpy as np

    import gmlx.spec.dflash_drafter as dd
    from test_dflash2_drafter import BLOCK, CAPTURE

    monkeypatch.setattr(dd, "_COUNTED_SHORT", True)
    lm = _target()
    drafter = _drafter(lm, window=4096)
    lm.set_dflash_capture(CAPTURE)
    drafter.reset(lm)
    hidden = int(drafter._hidden) * (1 + drafter._n_targets)
    mx.random.seed(3)
    drafter.append_context(drafter._captures(
        mx.random.normal((1, 11, hidden))))
    rows = mx.random.normal((1, BLOCK, hidden))
    block = mx.array([[5] + [int(drafter.config.mask_token_id)] * (BLOCK - 1)])
    most = BLOCK - 1
    mark = drafter.accept_mark(BLOCK)
    assert mark is not None and drafter.counted_ok(BLOCK, most, BLOCK)
    for dead in range(BLOCK):
        drafter.append_context(drafter._captures(rows))
        got = drafter._draft_hidden(block, (mx.array(dead), most))
        mx.eval(got)
        kept = drafter.accept_trim(mark, dead)
        want = drafter._draft_hidden(block)
        assert np.allclose(np.array(got.astype(mx.float32)),
                           np.array(want.astype(mx.float32)),
                           rtol=2e-3, atol=2e-4)
        drafter.accept_undo(kept)


# --- verify ahead of the row cut ------------------------------------------------

def _run_ahead(monkeypatch, level, counted=False, **kw):
    monkeypatch.setattr(sp, "_VERIFY_AHEAD", level)
    return _run_counted(monkeypatch, counted, **kw)


def _cut_exact():
    """On the CPU stream a verify ahead of a cut leaves the target cache
    equal to rounding, not to the bit (measured 7e-7 of its largest
    value). The GPU keeps the bits."""
    return mx.default_device().type == mx.DeviceType.gpu


@pytest.mark.parametrize("sampled", MODES)
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("counted", [False, True])
def test_a_verify_ahead_of_a_cut_that_keeps_every_row_is_the_plain_round(
        monkeypatch, sampled, level, counted):
    # A floor every path clears: the cut keeps the block, so the rounds are
    # the plain order's and the draws line up.
    plain = _run_ahead(monkeypatch, 0, n=64, floor=1e-30,
                       **_kw(monkeypatch, sampled))
    on = _run_ahead(monkeypatch, level, counted, n=64, floor=1e-30,
                    **_kw(monkeypatch, sampled))
    (_close if counted else _same)(plain, on)
    assert on["stats"]["ahead"] > 0 and on["stats"]["ahead_cut"] == 0


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("counted", [False, True])
@pytest.mark.parametrize("stop_after", [None, 1, 4, 9, 17, 30])
def test_greedy_rounds_ahead_of_the_cut_emit_the_plain_chain(
        monkeypatch, level, counted, stop_after):
    plain = _run_ahead(monkeypatch, 0, n=64, floor=0.5, stop_after=stop_after)
    on = _run_ahead(monkeypatch, level, counted, n=64, floor=0.5,
                    stop_after=stop_after)
    _close(plain, on, exact=_cut_exact())
    if stop_after is None:
        assert on["stats"]["ahead"] > 0


@pytest.mark.parametrize("sampled", [True, "sharp"])
def test_sampled_rounds_ahead_of_the_cut_run_whole_blocks(monkeypatch, sampled):
    on = _run_ahead(monkeypatch, 2, True, n=64, floor=0.5,
                    **_kw(monkeypatch, sampled))
    assert len(on["toks"]) >= 64 and on["stats"]["ahead"] > 0
    assert not on["pending"]


# --- the next verify built during this one --------------------------------------

def _run_prebuilt(monkeypatch, on, level, counted, **kw):
    monkeypatch.setattr(sp, "_VERIFY_PREBUILT", on)
    monkeypatch.setattr(sp, "_VERIFY_AHEAD", level)
    if counted:
        return _run_counted(monkeypatch, True, **kw)
    return _run(monkeypatch, True, **kw)


@pytest.mark.parametrize("sampled", MODES)
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("counted", [False, True])
@pytest.mark.parametrize("stop_after", [None, 3, 9, 17, 30, 41])
@pytest.mark.parametrize("variants", [1, 3])
def test_a_verify_built_ahead_leaves_the_rounds_and_state_as_they_were(
        monkeypatch, sampled, level, counted, stop_after, variants):
    """Same rounds with and without the build ahead: tokens, target cache,
    ring and RNG key, over hits, misses and a stop inside a round, with the
    full accept built or the three counts seen most. A random target never
    takes a whole block, so the builds that are taken up are those of the
    other counts."""
    monkeypatch.setattr(sp, "_prebuilt_seen", {})
    off = _run_prebuilt(monkeypatch, 0, level, counted, n=64, floor=0.5,
                        stop_after=stop_after, **_kw(monkeypatch, sampled))
    on = _run_prebuilt(monkeypatch, variants, level, counted, n=64, floor=0.5,
                       stop_after=stop_after, **_kw(monkeypatch, sampled))
    _same(off, on)
    assert off["stats"]["prebuilt"] == 0
    if stop_after is None and level == 2 and counted and variants > 1:
        assert on["stats"]["prebuilt"] > 0
        assert on["stats"]["prebuilt_miss"] > 0


@pytest.mark.parametrize("sampled", MODES)
@pytest.mark.parametrize("late", [1, 2])
def test_builds_ahead_that_hold_their_rows_leave_the_rounds_as_they_were(
        monkeypatch, sampled, late):
    """GMLX_SPEC_PREBUILT_LATE_ROWS: the model is told to hold the table
    rows around the builds for counts other than the full accept (1) or
    around every build (2), and never left holding; the rounds and the
    state are those of the lever off."""
    monkeypatch.setattr(sp, "_prebuilt_seen", {})
    off = _run_prebuilt(monkeypatch, 3, 2, True, n=64, floor=0.5,
                        **_kw(monkeypatch, sampled))
    calls = []
    monkeypatch.setattr(type(_target()), "hold_verify_rows",
                        lambda self, on: calls.append(on), raising=False)
    monkeypatch.setattr(sp, "_PREBUILT_LATE_ROWS", late)
    monkeypatch.setattr(sp, "_prebuilt_seen", {})
    on = _run_prebuilt(monkeypatch, 3, 2, True, n=64, floor=0.5,
                       **_kw(monkeypatch, sampled))
    _same(off, on)
    assert calls and calls[0::2] == [True] * (len(calls) // 2)
    assert calls[1::2] == [False] * (len(calls) // 2) and len(calls) % 2 == 0
    held = {1: [], 2: []}
    for n in (1, 2):
        monkeypatch.setattr(type(_target()), "hold_verify_rows",
                            lambda self, on, n=n: held[n].append(on), raising=False)
        monkeypatch.setattr(sp, "_PREBUILT_LATE_ROWS", n)
        monkeypatch.setattr(sp, "_prebuilt_seen", {})
        _run_prebuilt(monkeypatch, 3, 2, True, n=64, floor=0.5,
                      **_kw(monkeypatch, sampled))
    assert len(held[2]) > len(held[1])


# --- the verify ahead of the cut at the width the cut is expected to keep ---------

@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("counted", [False, True])
@pytest.mark.parametrize("prebuilt", [0, 3])
@pytest.mark.parametrize("stop_after", [None, 4, 17, 30])
def test_greedy_rounds_at_a_predicted_width_emit_the_plain_chain(
        monkeypatch, rows, counted, prebuilt, stop_after):
    plain = _run_ahead(monkeypatch, 0, n=64, floor=0.5, stop_after=stop_after)
    monkeypatch.setattr(sp, "_AHEAD_PREDICT", rows)
    monkeypatch.setattr(sp, "_prebuilt_seen", {})
    on = _run_prebuilt(monkeypatch, prebuilt, 2, counted, n=64, floor=0.5,
                       stop_after=stop_after)
    _close(plain, on, exact=_cut_exact())
    if stop_after is None:
        assert on["stats"]["ahead"] > 0


@pytest.mark.parametrize("sampled", MODES)
@pytest.mark.parametrize("counted", [False, True])
@pytest.mark.parametrize("stop_after", [None, 9, 30])
def test_a_build_ahead_at_a_predicted_width_leaves_the_rounds_as_they_were(
        monkeypatch, sampled, counted, stop_after):
    """The verify built ahead takes the width the round would have run."""
    monkeypatch.setattr(sp, "_AHEAD_PREDICT", 1)
    monkeypatch.setattr(sp, "_prebuilt_seen", {})
    off = _run_prebuilt(monkeypatch, 0, 2, counted, n=64, floor=0.5,
                        stop_after=stop_after, **_kw(monkeypatch, sampled))
    on = _run_prebuilt(monkeypatch, 3, 2, counted, n=64, floor=0.5,
                       stop_after=stop_after, **_kw(monkeypatch, sampled))
    _same(off, on)
