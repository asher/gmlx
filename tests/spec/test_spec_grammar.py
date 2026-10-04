"""Structured output (response_format) on the owned MTP rounds.

The grammar masks every verify position, so a constrained speculative
stream equals constrained greedy decode no matter what the drafter
proposes. Fakes: a byte-level tokenizer (no download), a regex grammar of
eight digits, and a target that prefers a token the grammar refuses, with
the next digit as its runner-up. Unconstrained, the target emits BAD
forever; constrained, it counts digits and then must end.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

llg = pytest.importorskip("llguidance")
import llguidance.hf  # noqa: E402
from mlx_vlm.structured import (  # noqa: E402
    LLGuidanceLogitsProcessor,
    ThinkingAwareLogitsProcessor,
)
from tokenizers import pre_tokenizers  # noqa: E402

import gmlx.spec.speculative as spec  # noqa: E402
from gmlx.load.tokenizer import load_tokenizer_from_gguf  # noqa: E402
from gmlx.spec.grammar import SpecGrammar, apply_masks  # noqa: E402

from test_mtp_preempt_resume import _ArmableDrafter  # noqa: E402
from test_mtp_width_cap import _FakeCache  # noqa: E402

_SPECIALS = ["<s>", "</s>", "<pad>", "<think>", "</think>"]
_ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())
_MERGED = ["He", "wo"]
_MERGES = ["H e", "w o"]


def _tokenizer():
    toks = _SPECIALS + _ALPHABET + _MERGED
    meta = {
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": toks,
        "tokenizer.ggml.merges": _MERGES,
        "tokenizer.ggml.token_type": [3] * len(_SPECIALS)
        + [1] * (len(_ALPHABET) + len(_MERGED)),
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 2,
    }
    return load_tokenizer_from_gguf(meta, "qwen2")


TOK = _tokenizer()
LLG_TOK = llguidance.hf.from_tokenizer(TOK)
VOCAB = len(TOK)
EOS = 1
END_THINK = TOK.convert_tokens_to_ids("</think>")
DIGITS = [TOK.convert_tokens_to_ids(str(d)) for d in range(10)]
BAD = TOK.convert_tokens_to_ids("x")
N_DIGITS = 8


def _proc(regex=f"[0-9]{{{N_DIGITS}}}"):
    return LLGuidanceLogitsProcessor(llg.grammar_from("regex", regex), LLG_TOK)


def _grammar(**kw):
    return SpecGrammar.from_processors([_proc(**kw)])


def _next_digit(tok: int) -> int:
    if tok in DIGITS:
        return DIGITS[(DIGITS.index(tok) + 1) % 10]
    return DIGITS[0]


def _reference(b: int, n: int) -> list[int]:
    """Constrained greedy decode after the first token b (itself the first
    digit): the next digits, then EOS once the grammar is complete."""
    out, cur = [], b
    for _ in range(n):
        cur = EOS if len(out) == N_DIGITS - 1 else _next_digit(cur)
        out.append(cur)
        if cur == EOS:
            break
    return out


def _allowed(mask_row, tok):
    return bool((int(mask_row[tok >> 5]) >> (tok & 31)) & 1)


# -- SpecGrammar ------------------------------------------------------------


def test_masks_match_a_fresh_matcher_per_position():
    g = _grammar()
    drafts = DIGITS[1:4]
    masks = g.masks(drafts)
    assert masks.shape == (4, (LLG_TOK.vocab_size + 31) // 32)
    for j in range(4):
        ref = llg.LLMatcher(LLG_TOK, _proc().grammar)
        assert ref.consume_tokens(drafts[:j])
        want = np.zeros((1, masks.shape[1]), dtype=np.int32)
        llguidance.numpy.fill_next_token_bitmask(ref, want, 0)
        assert (masks[j] == want[0]).all()
    # The walk rolled back: the same masks come out again.
    assert (g.masks(drafts) == masks).all()


def test_refused_draft_ends_the_walk():
    g = _grammar()
    masks = g.masks([DIGITS[1], BAD, DIGITS[2]])
    assert not _allowed(masks[1], BAD)
    assert (masks[2:] == -1).all()
    g.commit([DIGITS[1], DIGITS[2]])  # matcher state was left intact


def test_thinking_rows_are_unconstrained_until_the_end_marker():
    proc = ThinkingAwareLogitsProcessor(_proc(), TOK, enable_thinking=True)
    g = SpecGrammar.from_processors([proc])
    assert not g.active and g.end_id == END_THINK
    assert g.masks([BAD, BAD]) is None
    masks = g.masks([BAD, END_THINK, DIGITS[3]])
    assert (masks[:2] == -1).all()
    assert not _allowed(masks[2], BAD) and _allowed(masks[2], DIGITS[3])
    assert _allowed(masks[3], DIGITS[4])
    g.commit([BAD, END_THINK, DIGITS[3]])
    assert g.active


def test_row_after_a_completing_draft_allows_only_eos():
    g = _grammar()
    g.commit(DIGITS[:7])
    masks = g.masks([DIGITS[7], BAD])
    assert _allowed(masks[1], EOS) and not _allowed(masks[1], BAD)


def test_commit_stops_at_the_end_of_the_grammar():
    g = _grammar()
    g.commit(DIGITS[:8])
    masks = g.masks([])
    assert _allowed(masks[0], EOS) and not _allowed(masks[0], DIGITS[0])
    g.commit([EOS, BAD])  # tokens past the end are the row's tail


def test_commit_of_a_refused_token_raises():
    g = _grammar()
    with pytest.raises(ValueError, match="matcher error"):
        g.commit([BAD])


def test_start_commits_the_first_token_once():
    g = _grammar()
    g.start(DIGITS[0])
    g.start(DIGITS[0])  # a rebuilt round loop restarts from the same token
    assert _allowed(g.masks([])[0], DIGITS[1])
    g.commit(DIGITS[1:8])
    assert _allowed(g.masks([])[0], EOS)


def test_no_grammar_in_plain_processors():
    assert SpecGrammar.from_processors([lambda t, x: x]) is None
    assert SpecGrammar.from_processors(None) is None


def test_apply_masks_refuses_tokens_and_padded_vocab():
    g = _grammar()
    masks = g.masks([])
    padded = VOCAB + 40
    logits = mx.zeros((1, padded))
    out = apply_masks(logits, masks)
    finite = np.isfinite(np.array(out[0])).nonzero()[0].tolist()
    assert sorted(finite) == sorted(DIGITS)


# -- round loops ------------------------------------------------------------


class _PreferBadLM:
    """BAD scores highest, the next digit second, EOS third. Serves the
    plain-forward verify branch and the plain decode step."""

    def __init__(self):
        self._rope_deltas = None
        self.plain_calls = 0

    def __call__(self, x, cache=None, return_hidden=False,
                 return_shared_kv=False, **kw):
        B, S = x.shape
        flat = x.reshape(-1).tolist()
        rows = np.zeros((B * S, VOCAB), dtype=np.float32)
        for i, t in enumerate(flat):
            rows[i, BAD] = 10.0
            rows[i, _next_digit(int(t))] = 5.0
            rows[i, EOS] = 1.0
        logits = mx.array(rows).reshape(B, S, VOCAB)
        if return_hidden:
            return SimpleNamespace(
                logits=logits,
                hidden_states=[mx.zeros((B, S, 8))],
                shared_kv_states={"full": (mx.zeros((B, 2, S, 4)),
                                           mx.zeros((B, 2, S, 4)))},
                gdn_states=None)
        self.plain_calls += 1
        return SimpleNamespace(logits=logits)


class _DigitDrafter(_ArmableDrafter):
    """Drafts per row: the next two digits (accepted), or BAD first."""

    def __init__(self, bad_rows=(), **kw):
        super().__init__(**kw)
        self.bad_rows = set(bad_rows)

    def draft_block(self, b, hidden, kv, n, sampler, dtype, **kw):
        rows = mx.array(b).reshape(-1).tolist()
        out = []
        for r, t in enumerate(rows):
            if r in self.bad_rows:
                out.append([BAD, _next_digit(_next_digit(t))])
            else:
                d1 = _next_digit(t)
                out.append([d1, _next_digit(d1)])
        self.draft_calls.append(len(rows))
        return mx.array(out, dtype=dtype)


def _scalar(drafter, *, b=DIGITS[0], max_tokens=20, grammar=None):
    g = grammar if grammar is not None else _grammar()
    g.start(b)
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), drafter, _PreferBadLM(), [_FakeCache(width=1)],
        hidden=mx.zeros((1, 4, 8)), b=b, shared_kv=shared,
        seed_tokens=None, emitted=1, max_tokens=max_tokens, sampler=None,
        draft_block_size=None, grammar=g)
    out = []
    for tok in gen:
        out.append(int(tok))
        if tok == EOS:
            gen.close()
            break
    return out


@pytest.mark.parametrize("bad", [False, True])
def test_scalar_rounds_equal_constrained_greedy(bad):
    d = _DigitDrafter(bad_rows=(0,) if bad else (), cap=0)
    out = _scalar(d)
    assert out == _reference(DIGITS[0], 20)
    assert BAD not in out
    accepted = sum(d.accept_lens)
    assert (accepted == 0) if bad else (accepted > 0)


def test_scalar_rounds_without_grammar_follow_the_target():
    d = _DigitDrafter(cap=0)
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), d, _PreferBadLM(), [_FakeCache(width=1)],
        hidden=mx.zeros((1, 4, 8)), b=DIGITS[0], shared_kv=shared,
        seed_tokens=None, emitted=1, max_tokens=6, sampler=None,
        draft_block_size=None)
    assert [int(t) for t in gen] == [BAD] * 5


def test_scalar_rounds_sampled_stay_in_grammar():
    def sampler(logprobs):
        return mx.random.categorical(logprobs * 0.2, axis=-1)
    g = _grammar(regex="[0-9]{12}")
    g.start(DIGITS[0])
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), _DigitDrafter(cap=0), _PreferBadLM(),
        [_FakeCache(width=1)], hidden=mx.zeros((1, 4, 8)), b=DIGITS[0],
        shared_kv=shared, seed_tokens=None, emitted=1, max_tokens=13,
        sampler=sampler, draft_block_size=None, grammar=g)
    out = [int(t) for t in gen]
    assert all(t in DIGITS for t in out[:11]) and out[11] == EOS


def _batch(drafter, grammars, *, B=2, max_tokens=12, lm=None):
    model = SimpleNamespace()
    lm = lm if lm is not None else _PreferBadLM()
    gen = spec._owned_decode_rounds_batch(
        model, drafter, lm, [_FakeCache(width=B)],
        hidden=None, b=[DIGITS[0]] * B, shared_kv=None, seed_tokens=None,
        emitted=[1] * B, max_tokens=max_tokens, sampler=None,
        draft_block_size=None, eos_token_ids={EOS}, grammars=grammars)
    rows = [[] for _ in range(B)]
    for toks, _meta in gen:
        for r, t in enumerate(toks):
            if t is not None:
                rows[r].append(int(t))
    return rows


def test_batch_rounds_constrain_only_their_row():
    rows = _batch(_DigitDrafter(bad_rows=(1,), cap=0), [_grammar(), None])
    assert rows[0] == _reference(DIGITS[0], 11)
    assert rows[1] == [BAD] * 11


def test_gated_batch_rounds_mask_every_step():
    # Width cap 1 with two rows: the batch decodes plain (gated).
    lm = _PreferBadLM()
    d = _DigitDrafter(cap=1)
    rows = _batch(d, [_grammar(), _grammar()], lm=lm)
    assert rows == [_reference(DIGITS[0], 11)] * 2
    assert not d.draft_calls
    # One step per emitted token: no lookahead step ran past the end.
    assert lm.plain_calls == len(rows[0])


def test_injected_row_brings_its_grammar():
    model = SimpleNamespace()
    d = _DigitDrafter(cap=0)
    gen = spec._owned_decode_rounds_batch(
        model, d, _PreferBadLM(), [_FakeCache(width=1)],
        hidden=None, b=[DIGITS[0]], shared_kv=None, seed_tokens=None,
        emitted=[1], max_tokens=10, sampler=None, draft_block_size=None,
        eos_token_ids={EOS}, grammars=None)
    rows = [[], []]
    toks, _ = next(gen)
    rows[0].append(int(toks[0]))
    g = _grammar()
    model._generator_injections = [{
        "uids": [7],
        "prompt_cache": [_FakeCache(width=1)],
        "hidden": mx.zeros((1, 1, 8)),
        "shared_kv_states": None,
        "prompt_tokens": mx.array([[DIGITS[0]]]),
        "first_tokens": mx.array([DIGITS[5]]),
        "first_tokens_list": [DIGITS[5]],
        "max_tokens": [10],
        "grammars": [g],
    }]
    for toks, _ in gen:
        for r, t in enumerate(toks):
            if t is not None:
                rows[r].append(int(t))
    assert BAD in rows[0]
    assert rows[1] == _reference(DIGITS[5], 9)


# -- engine transport --------------------------------------------------------


def test_rounds_get_grammars_at_start_after_preempt_and_on_injection(
        monkeypatch):
    from mlx_vlm.generate import ar

    from gmlx.serve.patches.spec_grammar import _install_start
    from gmlx.spec.admission import install_continuous_batch_admission
    from test_mtp_preempt_resume import _make_batch

    install_continuous_batch_admission()
    _install_start(ar.SpeculativeGenerationBatch)
    seen = []

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        seen.append(("start", spec._pop_spec_grammars(prompt_cache)))
        width = int(kw["first_bonus"].shape[0])
        while True:
            inj = getattr(model, "_generator_injections", None)
            if inj:
                seen.append(("inject", [e.get("grammars") for e in inj]))
                width += sum(len(e["uids"]) for e in inj)
                inj.clear()
            yield [100] * width, None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    model = SimpleNamespace()
    host = _make_batch(ar, uids=(0,), model=model)
    host._kq_grammars = ["g-host"]
    host.next()                     # first token
    host.next()                     # scalar rounds start
    waiter = _make_batch(ar, uids=(7,), model=model)
    waiter._kq_grammars = ["g-wait"]
    host.extend(waiter)
    host.next()                     # preempt, rebuild, inject
    assert seen == [("start", ["g-host"]), ("start", ["g-host"]),
                    ("inject", [["g-wait"]])]
