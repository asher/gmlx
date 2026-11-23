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
import pytest

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


# --- the last token before the close and the prompt cache store ----------------

def _retiring_rounds(monkeypatch, defer, rounds, step=True, covered=False,
                     **kw):
    """Server rounds over fake rounds, with the retirement recorded, inside
    a serve step unless ``step`` is False. ``covered``: the drafter stores
    its own KV beside the prompt cache."""
    seen = {"closed": False, "retired": []}

    def fake_rounds(model, drafter, lm, prompt_cache, *, round_pos, **_):
        try:
            for rnd in rounds:
                for i, t in enumerate(rnd):
                    round_pos[:] = [i, len(rnd)]
                    yield t
        finally:
            seen["closed"] = True

    def fake_retire(model, prompt_cache, generated, retire, drafter=None,
                    sidecar_ctx=None):
        seen["retired"].append((list(generated), retire, seen["closed"]))

    monkeypatch.setattr(spec, "_owned_decode_rounds", fake_rounds)
    monkeypatch.setattr(spec, "_retire_b1", fake_retire)
    monkeypatch.setattr(spec, "_pop_retire_ctx", lambda cache: {"full_ids": [1]})
    monkeypatch.setattr(spec, "_RETIRE_AFTER_FINAL", defer)
    monkeypatch.setattr(spec._step_end, "jobs", [] if step else None,
                        raising=False)
    drafter = _EchoDrafter(cap=0)
    drafter._kq_head_covered = covered
    gen = spec.owned_server_rounds(
        SimpleNamespace(), drafter, [_FakeCache(width=1)],
        mx.zeros((1, 4, 8)), first_bonus=5, sampler=None,
        shared_kv_states=None, prompt_tokens=None, greedy_sampling=True, **kw)
    return gen, seen


def test_the_last_token_goes_out_before_a_deferred_retirement(monkeypatch):
    """Off: the round loop is closed and the KV retired before the last
    token is yielded. On: the token is yielded first and ends the tick
    (round_len is its own place), and the serve step's run closes the loop
    and retires the same tokens under the same context."""
    gen, off = _retiring_rounds(monkeypatch, False, [[6, 7, 8]],
                                max_tokens=20, eos_token_ids={7})
    assert next(gen) == ([6], {"round_pos": 0, "round_len": 3})
    assert next(gen) == ([7], {"round_pos": 1, "round_len": 3})
    # The first token (5, out before the rounds) is part of the sequence.
    assert off["retired"] == [([5, 6, 7], {"full_ids": [1]}, True)]
    assert list(gen) == []

    gen, on = _retiring_rounds(monkeypatch, True, [[6, 7, 8]],
                               max_tokens=20, eos_token_ids={7})
    assert next(gen) == ([6], {"round_pos": 0, "round_len": 3})
    assert next(gen) == ([7], {"round_pos": 1, "round_len": 2})
    assert on["retired"] == [] and not on["closed"]
    assert len(spec.step_end_queue()) == 1
    spec.run_step_end()
    assert on["retired"] == off["retired"]
    assert spec.step_end_queue() is None
    # Nothing more comes out, and nothing is retired twice.
    assert list(gen) == []
    spec.run_step_end()
    assert len(on["retired"]) == 1


def test_a_deferred_retirement_at_the_budget_and_a_stop_before_the_end(monkeypatch):
    """The budget's last token defers like an EOS; a client that leaves
    before the last token retires through the close, as without the lever."""
    gen, on = _retiring_rounds(monkeypatch, True, [[6, 7, 8]], max_tokens=3)
    assert [t[0] for t, _ in gen] == [6, 7]
    assert on["retired"] == []
    spec.run_step_end()
    assert on["retired"] == [([5, 6, 7], {"full_ids": [1]}, True)]

    gen, on = _retiring_rounds(monkeypatch, True, [[6, 7, 8]], max_tokens=20)
    assert next(gen)[0] == [6]
    gen.close()
    assert spec.step_end_queue() == []
    assert on["retired"] == [([5, 6], {"full_ids": [1]}, True)]


@pytest.mark.parametrize("step, covered", [(False, False), (True, True)])
def test_the_retirement_runs_before_the_last_token_where_it_cannot_wait(
        monkeypatch, step, covered):
    # Outside a serve step nothing would run a deferred retirement. A
    # drafter that stores its own KV is reset by the step's release before
    # the step end.
    gen, on = _retiring_rounds(monkeypatch, True, [[6, 7, 8]], step=step,
                               covered=covered, max_tokens=20,
                               eos_token_ids={7})
    assert next(gen) == ([6], {"round_pos": 0, "round_len": 3})
    assert on["retired"] == []
    assert next(gen) == ([7], {"round_pos": 1, "round_len": 3})
    assert on["retired"] == [([5, 6, 7], {"full_ids": [1]}, True)]
    assert not spec.step_end_queue()


def test_the_serve_step_runs_a_deferred_retirement_even_when_it_fails(monkeypatch):
    from gmlx._vendor.mlx_vlm.server.generation import ResponseGenerator

    ran = []

    def body(self, batch_gen, active, gen_kwargs=None):
        spec.step_end_queue().append(lambda: ran.append(1))
        raise RuntimeError("step failed")

    monkeypatch.setattr(ResponseGenerator, "_step_body", body)
    stub = SimpleNamespace()
    try:
        ResponseGenerator._step(stub, None, {})
    except RuntimeError:
        pass
    assert ran == [1] and spec.step_end_queue() is None
