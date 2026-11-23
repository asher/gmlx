"""Deferred row cut: the engine builds the full-width verify while the draft
runs, and a cut that keeps fewer rows restores the cache and builds again."""

import itertools

import mlx.core as mx
import pytest

from gmlx._vendor.mlx_vlm.models.cache import KVCache

import gmlx.spec.dflash_drafter as dd
import gmlx.spec.speculative as sp
from gmlx.spec.dflash_drafter import DraftStash

from test_dflash2_drafter import (
    BLOCK,
    N_GEN,
    SEED,
    _annotated,
    _drafter,
    _engine_reference,
    _engine_walk,
    _target,
)

PROMPT = [[1, 2, 3, 4, 5]]
WHOLE = BLOCK - 1
# Whole twice (the build ahead is kept), a short cut (it is undone), a round
# that follows a short cut (no build ahead), then whole again.
SCRIPT = (WHOLE, WHOLE, 1, 2, WHOLE)


def _scripted_cut(monkeypatch, script=SCRIPT):
    """Replace the floor test with a fixed cycle of keep counts, the same
    for the cut the drafter takes itself and the one it leaves pending. The
    walk kernel cuts inside its dispatch, so the script runs the op chain."""
    keeps = itertools.cycle(script)

    def rows_over_floor(lp, floor):
        return mx.minimum(mx.array(next(keeps)), lp.shape[0])

    monkeypatch.setattr(dd, "rows_over_floor", rows_over_floor)
    monkeypatch.setattr(dd, "_WALK_KERNEL", False)


def _floored(lm):
    drafter = _drafter(lm)
    object.__setattr__(drafter, "row_floor", 0.5)
    return drafter


def _counted(monkeypatch, name):
    calls = []
    orig = getattr(sp, name)

    def counted(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(sp, name, counted)
    return calls


@pytest.fixture(autouse=True)
def _plain_rounds(monkeypatch):
    monkeypatch.setattr(sp, "_ROUND_GATE", 0.0)
    monkeypatch.setattr(sp, "_ROUND_GATE_AUTO", False)
    monkeypatch.setattr(sp, "_ROUND_PROFILE", False)
    monkeypatch.setattr(sp, "_DEFER_CUT", True)


def test_deferred_cut_emits_the_greedy_chain(monkeypatch):
    _scripted_cut(monkeypatch)
    builds = _counted(monkeypatch, "_mtp_verify_target")
    restores = _counted(monkeypatch, "_cache_restore")
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    drafter = _floored(lm)
    got = _engine_walk(lm, drafter, prompt, N_GEN)
    assert got[:N_GEN] == ref
    # Every short cut that follows a whole round undoes one build.
    assert restores and len(builds) > len(restores)
    assert not drafter.row_cut_pending


def test_the_lever_off_leaves_the_cut_in_the_drafter(monkeypatch):
    monkeypatch.setattr(sp, "_DEFER_CUT", False)
    _scripted_cut(monkeypatch)
    restores = _counted(monkeypatch, "_cache_restore")
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    got = _engine_walk(lm, _floored(lm), prompt, N_GEN)
    assert got[:N_GEN] == ref
    assert not restores


def test_a_stack_the_snapshot_cannot_copy_cuts_before_the_build(monkeypatch):
    _scripted_cut(monkeypatch)
    monkeypatch.setattr(sp, "_cache_snapshot", lambda cache: None)
    builds = _counted(monkeypatch, "_mtp_verify_target")
    restores = _counted(monkeypatch, "_cache_restore")
    lm = _target()
    prompt = mx.array(PROMPT)
    ref = _engine_reference(lm, prompt, N_GEN)
    drafter = _floored(lm)
    drafts = []
    orig = drafter.draft_block

    def draft_block(*a, **kw):
        drafts.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(drafter, "draft_block", draft_block)
    got = _engine_walk(lm, drafter, prompt, N_GEN)
    assert got[:N_GEN] == ref
    assert not restores and len(builds) == len(drafts)


def test_sampled_rounds_draw_the_same_tokens_with_the_cut_deferred(monkeypatch):
    monkeypatch.setattr(sp, "_STOCH_ACCEPT", True)
    for name in ("_PQ_LOG", "_TOP2_LOG"):
        monkeypatch.setattr(sp, name, False)
    lm = _target()
    prompt = mx.array(PROMPT)
    runs = []
    for defer in (False, True):
        monkeypatch.setattr(sp, "_DEFER_CUT", defer)
        _scripted_cut(monkeypatch)
        restores = _counted(monkeypatch, "_cache_restore")
        mx.random.seed(SEED)
        runs.append(_engine_walk(lm, _floored(lm), prompt, N_GEN,
                                 sampler=_annotated()))
        assert bool(restores) == defer
    assert runs[0] == runs[1]


@pytest.mark.parametrize("sampled", [False, True])
def test_apply_row_cut_trims_the_round_to_the_kept_drafts(monkeypatch, sampled):
    _scripted_cut(monkeypatch, (2,))
    lm = _target()
    drafter = _floored(lm)
    stash = DraftStash(q=[]) if sampled else None
    mx.random.seed(SEED)
    toks = drafter.draft_block(
        3, None, None, BLOCK, _annotated() if sampled else None,
        greedy=not sampled, stash=stash, defer_cut=True)
    assert toks.shape == (1, WHOLE) and drafter.row_cut_pending
    assert drafter.pending_row_cut() == 2
    drafter.apply_row_cut(2)
    assert not drafter.row_cut_pending
    if sampled:
        assert len(stash.q) == 2
    else:
        first, edges, sel, lp = drafter._round_path
        assert edges.shape[0] == 1 and sel.shape[0] == 2
    # The cut the drafter takes itself returns the same drafts.
    mx.random.seed(SEED)
    _scripted_cut(monkeypatch, (2,))
    eager = drafter.draft_block(
        3, None, None, BLOCK, _annotated() if sampled else None,
        greedy=not sampled, stash=DraftStash(q=[]) if sampled else None)
    assert eager.tolist() == toks[:, :2].tolist()
    assert not drafter.row_cut_pending


def test_snapshot_copies_a_cache_that_carries_bytes():
    c = KVCache()
    c._gmlx_cascade = b"\x01\x02"
    assert sp._cache_snapshot([c]) is not None
