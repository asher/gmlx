"""Request logits processors (penalties, XTC, thinking-gated processors) on
the owned MTP rounds. Greedy speculative output must equal plain greedy
decode running the same processors with the same contexts."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.sample_utils import apply_xtc, make_logits_processors
from mlx_vlm.structured import ThinkingAwareLogitsProcessor

import gmlx.spec.speculative as spec
from gmlx.spec.row_procs import SpecRowProcessors, supported

from test_mtp_width_cap import _FakeCache
from test_spec_grammar import (
    BAD,
    DIGITS,
    END_THINK,
    EOS,
    TOK,
    _DigitDrafter,
    _PreferBadLM,
    _proc,
)

PROMPT = [DIGITS[3], DIGITS[4]]


def _penalties():
    return make_logits_processors(presence_penalty=7.0,
                                  presence_context_size=4)


def _xtc(_tokens, logits):
    return apply_xtc(logits, 1.0, 0.005, [EOS])


def _gated(fn, active=False):
    proc = ThinkingAwareLogitsProcessor(fn, TOK, enable_thinking=True)
    proc._active = active
    return proc


def _plain(procs_factory, b, n, prompt=PROMPT, lm_cls=None):
    """Plain greedy decode with the stock decode loop's processor calls."""
    lm, procs = (lm_cls or _PreferBadLM)(), procs_factory()
    ctx, out, cur = list(prompt) + [b], [], b
    for _ in range(n):
        logits = lm(mx.array([[cur]])).logits[:, -1, :]
        for p in procs:
            if hasattr(p, "process_last_token"):
                logits = p.process_last_token(cur, logits)
            else:
                logits = p(mx.array(ctx), logits)
        cur = int(mx.argmax(logits, axis=-1).item())
        out.append(cur)
        ctx.append(cur)
    return out


def _rows(procs_factory, prompt=PROMPT):
    return SpecRowProcessors.from_processors(procs_factory(), prompt)


def _scalar(drafter, rows, *, b=DIGITS[0], max_tokens=13):
    rows.start(b)
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), drafter, _PreferBadLM(), [_FakeCache(width=1)],
        hidden=mx.zeros((1, 4, 8)), b=b, shared_kv=shared,
        seed_tokens=None, emitted=1, max_tokens=max_tokens, sampler=None,
        draft_block_size=None, row_procs=rows)
    return [int(t) for t in gen]


def _batch(drafter, rows, *, B=2, max_tokens=12):
    gen = spec._owned_decode_rounds_batch(
        SimpleNamespace(), drafter, _PreferBadLM(), [_FakeCache(width=B)],
        hidden=None, b=[DIGITS[0]] * B, shared_kv=None, seed_tokens=None,
        emitted=[1] * B, max_tokens=max_tokens, sampler=None,
        draft_block_size=None, eos_token_ids={EOS}, row_procs=rows)
    out = [[] for _ in range(B)]
    for toks, _meta in gen:
        for r, t in enumerate(toks):
            if t is not None:
                out[r].append(int(t))
    return out


@pytest.mark.parametrize("factory", [_penalties, lambda: [_xtc]],
                         ids=["penalties", "xtc"])
@pytest.mark.parametrize("bad", [False, True])
def test_scalar_rounds_equal_plain_decode(factory, bad):
    d = _DigitDrafter(bad_rows=(0,) if bad else (), cap=0)
    out = _scalar(d, _rows(factory))
    assert out == _plain(factory, DIGITS[0], 12)
    assert out != [BAD] * 12


def test_penalties_use_the_drafts_ahead_of_each_position():
    # Presence penalty 7 knocks BAD (10) under the next digit (5) for four
    # tokens after each BAD, so the stream alternates and drafts are refused
    # or accepted depending on the context each position sees.
    out = _scalar(_DigitDrafter(cap=0), _rows(_penalties))
    assert BAD in out and any(t in DIGITS for t in out)


@pytest.mark.parametrize("cap", [0, 1], ids=["spec", "gated"])
def test_batch_rounds_process_only_their_row(cap):
    out = _batch(_DigitDrafter(cap=cap), [_rows(_penalties), None])
    assert out[0] == _plain(_penalties, DIGITS[0], 11)
    assert out[1] == [BAD] * 11


class _ThinkLM(_PreferBadLM):
    """Like _PreferBadLM, but DIGITS[2] is followed by </think>."""

    def __call__(self, x, cache=None, **kw):
        out = super().__call__(x, cache=cache, **kw)
        hit = (x == DIGITS[2])[..., None]
        bump = mx.zeros((out.logits.shape[-1],)).at[END_THINK].add(20.0)
        out.logits = out.logits + hit * bump
        return out


def _recorder(seen):
    def fn(tokens, logits):
        seen.append(tokens.tolist())
        return logits
    return fn


def test_gated_processor_sees_tokens_from_the_end_marker():
    rows = SpecRowProcessors.from_processors(
        [_gated(_recorder(seen := []))], PROMPT)
    rows.start(DIGITS[0])
    plan = rows.plan([DIGITS[1], END_THINK, DIGITS[5]])
    assert plan.starts == [[None, None, 4, 4]]
    rows.apply(mx.zeros((4, 8)), plan)
    assert seen == [[END_THINK], [END_THINK, DIGITS[5]]]
    rows.commit([DIGITS[1], END_THINK])
    rows.apply(mx.zeros((2, 8)), rows.plan([DIGITS[7]]))
    assert seen[2:] == [[END_THINK], [END_THINK, DIGITS[7]]]


def test_active_gated_processor_skips_the_prompt():
    rows = SpecRowProcessors.from_processors(
        [_gated(_recorder(seen := []), active=True)], PROMPT)
    rows.start(DIGITS[0])
    rows.apply(mx.zeros((2, 8)), rows.plan([DIGITS[1]]))
    assert seen == [[DIGITS[0]], [DIGITS[0], DIGITS[1]]]


def test_ungated_processor_sees_the_prompt():
    rows = SpecRowProcessors.from_processors([_recorder(seen := [])], PROMPT)
    rows.start(DIGITS[0])
    rows.apply(mx.zeros((2, 8)), rows.plan([DIGITS[1]]))
    assert seen == [PROMPT + [DIGITS[0]], PROMPT + [DIGITS[0], DIGITS[1]]]


def test_gated_xtc_starts_after_thinking_like_plain_decode():
    # DIGITS[2] -> </think>, then XTC turns on and removes BAD.
    def factory():
        return [_gated(_xtc)]
    rows = _rows(factory)
    rows.start(DIGITS[2])
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), _DigitDrafter(cap=0), _ThinkLM(),
        [_FakeCache(width=1)], hidden=mx.zeros((1, 4, 8)), b=DIGITS[2],
        shared_kv=shared, seed_tokens=None, emitted=1, max_tokens=13,
        sampler=None, draft_block_size=None, row_procs=rows)
    out = [int(t) for t in gen]
    assert out == _plain(factory, DIGITS[2], 12, lm_cls=_ThinkLM)
    assert out[:3] == [END_THINK, DIGITS[0], DIGITS[1]]


def test_supported_refuses_stateful_processors():
    class _Stateful:
        def process_last_token(self, token, logits):
            return logits

        def __call__(self, tokens, logits):
            return logits

    assert supported([_xtc, _proc(), _gated(_xtc)] + _penalties())
    assert not supported([_Stateful()])
    assert not supported(["not callable"])
