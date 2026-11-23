"""A row cut keeps the sampled distribution exact.

The cut keeps a draft when the path chance through the draft's own pick
clears the floor, so the first dropped draft was dropped because of what it
drew. The walk tests that draft against the target row behind the kept
drafts (DraftStash.cut) in place of a plain draw; without that the token at
the cut is not a sample of the target row."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import gmlx.spec.dflash_drafter as dd
import gmlx.spec.speculative as sp
from gmlx.spec.drafter_protocol import DraftStash
from gmlx.spec.speculative import annotate_sampling_params

P = np.array([0.7, 0.3, 0.0], dtype=np.float32)
Q = np.array([0.8, 0.2, 0.0], dtype=np.float32)
ONE = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _dense_sampler():
    def s(logprobs):
        raise AssertionError("the walk must not call the sampler")
    annotate_sampling_params(s, temp=1.0, top_p=1.0, top_k=0, min_p=0.0)
    return s


def _filter_sampler():
    from gmlx.serve.patches.sampling import _FastPositionedSampler

    return _FastPositionedSampler(temperature=1.0, top_p=1.0, top_k=2)


def _target(rows):
    logits = mx.array(np.log(np.maximum(np.stack(rows), 1e-30)))
    lm = SimpleNamespace(speculative_logits_from_hidden=lambda hidden: logits)
    verify = SimpleNamespace(target_tokens=None, hidden=mx.zeros((1, 1, 1)),
                             logits=None)
    return lm, verify


def _second_token(sampler, rng, use_cut):
    """One round of two drafts. The first is certain. The second is drawn
    from Q and kept by the cut only when it drew token 0 (the pick that
    clears the floor). Returns the token emitted at the second position."""
    x = int(rng.choice(3, p=Q))
    stash = DraftStash(q=[mx.array(ONE), mx.array(Q)])
    toks = mx.array([[0, x]], dtype=mx.int32)
    mx.eval(toks)
    n = 2 if x == 0 else 1
    dd._note_cut_draft(stash, toks, n)
    del stash.q[n:]
    cut = sp._take_cut({"stash": stash}) if use_cut else None
    assert (cut is None) == (n == 2 or not use_cut)
    lm, verify = _target([ONE, P, P][:n + 1])
    _, new = sp._stochastic_walk(lm, verify, toks[:, :n], sampler, 8,
                                 list(stash.q), cut=cut)
    return new[1]


@pytest.mark.parametrize("make", [_dense_sampler, _filter_sampler])
def test_the_token_at_a_row_cut_is_a_sample_of_the_target_row(make):
    mx.random.seed(11)
    rng = np.random.default_rng(11)
    sampler, n = make(), 3000
    got = np.bincount([_second_token(sampler, rng, True) for _ in range(n)],
                      minlength=3) / n
    # Standard error of the first frequency is 0.0084.
    assert np.abs(got - P).max() < 0.035, got


@pytest.mark.parametrize("make", [_dense_sampler, _filter_sampler])
def test_a_plain_draw_at_the_cut_is_biased(make):
    """What the walk emitted before: (0.84, 0.16) for this p and q."""
    mx.random.seed(12)
    rng = np.random.default_rng(12)
    sampler, n = make(), 3000
    got = np.bincount([_second_token(sampler, rng, False) for _ in range(n)],
                      minlength=3) / n
    assert abs(got[0] - 0.84) < 0.035 and abs(got[0] - P[0]) > 0.08, got


def test_the_cut_notes_the_first_dropped_draft():
    rows = [mx.array(ONE), mx.array(Q), mx.array(P)]
    toks = mx.array([[5, 6, 7]], dtype=mx.int32)
    mx.eval(toks)
    stash = DraftStash(q=list(rows))
    dd._note_cut_draft(stash, toks, 1)
    assert stash.cut[0] == 6 and stash.cut[1] is rows[1]
    dd._note_cut_draft(stash, toks, 3)
    assert stash.cut is None
    # Only a block drawn from q rows is tested through them.
    other = DraftStash(q=list(rows), pq=[])
    dd._note_cut_draft(other, toks, 1)
    assert other.cut is None
    dd._note_cut_draft(None, toks, 1)


def test_apply_row_cut_keeps_the_dropped_draft_for_the_walk():
    rows = [mx.array(ONE), mx.array(Q), mx.array(P)]
    toks = mx.array([[5, 6, 7]], dtype=mx.int32)
    keep = mx.array([2])
    mx.eval(toks, keep)
    stash = DraftStash(q=list(rows))
    fake = SimpleNamespace(_pending_cut=(keep, stash, toks), _round_path=None)
    dd.DFlash2Drafter.apply_row_cut(fake, 2)
    assert len(stash.q) == 2 and stash.cut[0] == 7 and stash.cut[1] is rows[2]
    assert sp._take_cut({"stash": stash}) is not None and stash.cut is None
    assert sp._take_cut({}) is None
