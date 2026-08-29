"""^T finish-thinking on the owned MTP round loop.

A requested close is committed as fully-accepted forced rounds: the known
close ids ride a verify forward instead of a draft block, the stream emits
them contiguously at the next round boundary, and normal speculative rounds
resume after the close. Fakes come from the preempt/width-cap suites: an
echo target (next = input + 1) and a drafter that drafts the echo
continuation, so unforced rounds accept fully and are deterministic.
"""

from types import SimpleNamespace

import mlx.core as mx

import gmlx.spec.speculative as spec
from gmlx.gen.thinking_budget import MTPFinishThinking

from test_mtp_preempt_resume import VOCAB, _EchoDrafter, _VerifyEchoLM
from test_mtp_width_cap import _FakeCache

END, S1, S2 = 17, 11, 12  # close marker + wrap-phrase stand-ins (< VOCAB)


def _hook(skip_ids, *, start_in_thinking=True, **kw):
    return MTPFinishThinking(
        end_seq=(END,), skip_ids=list(skip_ids), reclose_ids=[13, END],
        start_seq=(16,), start_in_thinking=start_in_thinking, **kw)


def _make_gen(hook, *, max_tokens, b=5):
    d = _EchoDrafter(cap=0)
    lm = _VerifyEchoLM()
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), d, lm, [_FakeCache(width=1)],
        hidden=mx.zeros((1, 4, 8)), b=b, shared_kv=shared,
        seed_tokens=None, emitted=1, max_tokens=max_tokens, sampler=None,
        draft_block_size=None, thinking_hook=hook)
    return gen, d


def _drive(hook, *, press_after, max_tokens, b=5):
    gen, d = _make_gen(hook, max_tokens=max_tokens, b=b)
    out = []
    for tok in gen:
        out.append(int(tok))
        if len(out) == press_after:
            hook.request_close()
    return out, d


def test_forced_close_injected_at_round_boundary():
    hook = _hook([S1, S2, END])
    out, d = _drive(hook, press_after=2, max_tokens=13)
    # The in-flight round [6, 7, 8] completes, then the forced close, then
    # echo decode resumes from the close marker.
    assert out[:6] == [6, 7, 8, S1, S2, END]
    assert out[6:9] == [(END + 1) % VOCAB, (END + 2) % VOCAB,
                       (END + 3) % VOCAB]
    assert not hook.in_thinking and hook._spent
    # The forced round is not recorded as a speculative round.
    assert len(d.accept_lens) == 3 and all(a == 2 for a in d.accept_lens)


def test_long_forced_close_chunks_at_block_width():
    ids = [20, 21, 22, 23, 24, 25, 26, 27, END]  # 9 ids > block_size 4
    hook = _hook(ids)
    out, d = _drive(hook, press_after=1, max_tokens=14)
    # Chunked 4/3/2 (never a width-1 forced round), contiguous on the wire.
    assert out == [6, 7, 8] + ids + [(END + 1) % VOCAB]
    assert not hook.in_thinking


def test_close_outside_thinking_leaves_stream_untouched():
    hook = _hook([S1, S2, END], start_in_thinking=False)
    out, _d = _drive(hook, press_after=2, max_tokens=10)
    assert out == [6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert not hook._spent


def test_budget_trips_at_round_boundary_without_keypress():
    hook = _hook([S1, S2, END], budget=4, forced_ids=[14, 15, END])
    out, _d = _drive(hook, press_after=10**9, max_tokens=13)
    # Rounds emit 3 thinking tokens each; the count passes 4 during round
    # two, so the budget close (its own phrase) lands at the next boundary
    # and echo decode resumes after it.
    assert out == [6, 7, 8, 9, 10, 11, 14, 15, END,
                   (END + 1) % VOCAB, (END + 2) % VOCAB, (END + 3) % VOCAB]
    assert hook._spent and not hook.in_thinking


def test_generator_close_mid_forced_round():
    hook = _hook([S1, S2, END])
    gen, _d = _make_gen(hook, max_tokens=30)
    out = [int(next(gen)) for _ in range(3)]  # round one delivered
    hook.request_close()
    assert int(next(gen)) == S1  # the forced round starts
    gen.close()  # consumer stops mid-close; the finish seam must not raise
    assert out == [6, 7, 8]
