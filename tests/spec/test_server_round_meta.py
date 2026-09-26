"""Round metadata on the batch-size-1 serve rounds.

The upstream speculative batch takes every token of a round in one engine
tick when the round generator yields round_pos and round_len with each
token. owned_server_rounds reports them from the owned round loop, so a
single-request serve round costs one tick, as the batched rounds do.
Fakes come from the preempt suite: an echo target and a drafter that
drafts its continuation, so every round accepts fully.
"""

from types import SimpleNamespace

import mlx.core as mx

import gmlx.spec.speculative as spec

from test_mtp_preempt_resume import _EchoDrafter, _VerifyEchoLM
from test_mtp_width_cap import _FakeCache


def test_owned_rounds_report_round_position():
    pos = [None, None]
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), _EchoDrafter(cap=0), _VerifyEchoLM(),
        [_FakeCache(width=1)], hidden=mx.zeros((1, 4, 8)), b=5,
        shared_kv=shared, seed_tokens=None, emitted=1, max_tokens=9,
        sampler=None, draft_block_size=None, round_pos=pos)
    got = [(int(tok), tuple(pos)) for tok in gen]
    # Two full rounds of three, then the budget cuts the third to two.
    assert got == [
        (6, (0, 3)), (7, (1, 3)), (8, (2, 3)),
        (9, (0, 3)), (10, (1, 3)), (11, (2, 3)),
        (12, (0, 2)), (13, (1, 2)),
    ]


def _server_rounds(monkeypatch, rounds, **kw):
    def fake_rounds(model, drafter, lm, prompt_cache, *, round_pos, **_):
        for rnd in rounds:
            for i, t in enumerate(rnd):
                round_pos[:] = [i, len(rnd)]
                yield t

    monkeypatch.setattr(spec, "_owned_decode_rounds", fake_rounds)
    return spec.owned_server_rounds(
        SimpleNamespace(), _EchoDrafter(cap=0), [_FakeCache(width=1)],
        mx.zeros((1, 4, 8)), first_bonus=5, sampler=None,
        shared_kv_states=None, prompt_tokens=None, greedy_sampling=True, **kw)


def test_server_rounds_yield_round_metadata(monkeypatch):
    gen = _server_rounds(monkeypatch, [[6, 7, 8], [9, 10]], max_tokens=20)
    assert [(t[0], m) for t, m in gen] == [
        (6, {"round_pos": 0, "round_len": 3}),
        (7, {"round_pos": 1, "round_len": 3}),
        (8, {"round_pos": 2, "round_len": 3}),
        (9, {"round_pos": 0, "round_len": 2}),
        (10, {"round_pos": 1, "round_len": 2}),
    ]


def test_server_rounds_end_mid_round_on_eos(monkeypatch):
    # The engine keeps pulling while round_pos + 1 < round_len; after the
    # terminal token the generator ends, which stops that pull.
    gen = _server_rounds(monkeypatch, [[6, 7, 8]], max_tokens=20,
                         eos_token_ids={7})
    assert next(gen) == ([6], {"round_pos": 0, "round_len": 3})
    assert next(gen) == ([7], {"round_pos": 1, "round_len": 3})
    assert list(gen) == []
