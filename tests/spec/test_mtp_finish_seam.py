"""The single-stream round loop's finish seam: the target KV and the drafter
head retire at one length, whichever token closed the final round.

The head holds row p as (token p+1, hidden p). A round that closes on the
bonus token leaves every committed position pairable, so the target keeps
its accepts and the head ingests them plus the bonus. A round the budget
or an EOS closes on an accepted draft leaves the last delivered token
without a replayable successor, so the target drops that token's KV and
the head ingests the delivered drafts without a bonus: both retire one
token short of the delivered text, and the retirement sidecar pairs with
the entry instead of being skipped as one row short.

Fakes come from the preempt suite: an echo target (next = input + 1) and a
drafter that drafts the echo continuation, so every round accepts its two
drafts and yields three tokens.
"""

from types import SimpleNamespace

import mlx.core as mx

import gmlx.spec.speculative as spec

from test_mtp_preempt_resume import _EchoDrafter, _VerifyEchoLM
from test_mtp_width_cap import _FakeCache


class _RollbackEchoLM(_VerifyEchoLM):
    def __init__(self):
        super().__init__()
        self.rollbacks = []

    def rollback_speculative_cache(self, cache, gdn_states, accepted,
                                   block_size):
        self.rollbacks.append((int(accepted), int(block_size)))


class _AcceptEchoDrafter(_EchoDrafter):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.accepts = []

    def accept_verified_tokens(self, hidden, draft_tokens, accepted,
                               new_tokens, sampler, dtype, **kw):
        self.accepts.append((int(accepted), [int(t) for t in new_tokens]))


def _make_gen(max_tokens):
    d = _AcceptEchoDrafter(cap=0)
    lm = _RollbackEchoLM()
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), d, lm, [_FakeCache(width=1)],
        hidden=mx.zeros((1, 4, 8)), b=3, shared_kv=shared,
        seed_tokens=None, emitted=1, max_tokens=max_tokens, sampler=None,
        draft_block_size=None)
    return gen, d, lm


def test_budget_on_the_bonus_keeps_every_accept():
    # Rounds yield [4, 5, 6], [7, 8, 9]: a budget of 7 ends on the bonus.
    gen, d, lm = _make_gen(max_tokens=7)
    out = [int(t) for t in gen]
    assert out == [4, 5, 6, 7, 8, 9]
    assert lm.rollbacks == []            # both drafts accepted, nothing to drop
    assert d.accepts[-1] == (2, [7, 8, 9])


def test_budget_inside_the_drafts_drops_the_last_delivered_kv():
    # A budget of 6 delivers [7, 8] of the second round: two accepted
    # drafts, no bonus. The target keeps one accept (its KV covers 7), the
    # head ingests both drafts (its last row is (8, hidden at 7)).
    gen, d, lm = _make_gen(max_tokens=6)
    out = [int(t) for t in gen]
    assert out == [4, 5, 6, 7, 8]
    assert lm.rollbacks[-1] == (1, 3)
    assert d.accepts[-1] == (2, [])


def test_budget_on_the_first_draft_drops_it():
    gen, d, lm = _make_gen(max_tokens=5)
    out = [int(t) for t in gen]
    assert out == [4, 5, 6, 7]
    assert lm.rollbacks[-1] == (0, 3)
    assert d.accepts[-1] == (1, [])


def test_consumer_stop_on_a_draft_matches_the_budget_seam():
    # EOS or a stop string closes the generator mid-round with one draft
    # delivered: the same seam as the budget case.
    gen, d, lm = _make_gen(max_tokens=20)
    out = [int(next(gen)) for _ in range(4)]
    gen.close()
    assert out == [4, 5, 6, 7]
    assert lm.rollbacks[-1] == (0, 3)
    assert d.accepts[-1] == (1, [])


def test_consumer_stop_on_the_bonus_keeps_the_round():
    gen, d, lm = _make_gen(max_tokens=20)
    out = [int(next(gen)) for _ in range(6)]
    gen.close()
    assert out == [4, 5, 6, 7, 8, 9]
    assert lm.rollbacks == []
    assert d.accepts[-1] == (2, [7, 8, 9])
