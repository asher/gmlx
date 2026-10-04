"""run and chat on MTP: logit bias, penalties and XTC through stream_speculative.

Greedy speculative output must equal mlx-lm's own generate_step with the same
processors, whose processor context is the last prompt token plus the
generated tokens."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_logits_processors, make_sampler

import gmlx.gen.generation as generation
from gmlx.gen.generation import make_spec_sampler
from gmlx.spec.speculative import stream_speculative

from test_mtp_width_cap import _FakeCache
from test_spec_grammar import BAD, DIGITS, EOS, VOCAB, _DigitDrafter, _PreferBadLM

PROMPT = [BAD, DIGITS[4], DIGITS[5]]   # full-prompt context would penalize BAD


class _SpecLM(_PreferBadLM):
    def rollback_speculative_cache(self, *a, **k):
        pass


def _spec(procs, *, bad_rows=(), n=12, sampler=None):
    drafter = _DigitDrafter(bad_rows=bad_rows, cap=0)
    drafter.config = SimpleNamespace(block_size=3)
    gen = stream_speculative(
        SimpleNamespace(language_model=_SpecLM()), drafter,
        mx.array([PROMPT], dtype=mx.int32), prompt_cache=[_FakeCache(width=1)],
        max_tokens=n, sampler=sampler, logits_processors=procs)
    return [int(t) for t in gen]


def _plain(procs, n=12):
    lm = _PreferBadLM()

    def model(x, cache=None, **kw):
        return lm(x).logits

    gen = generate_step(mx.array(PROMPT, dtype=mx.int32), model,
                        max_tokens=n, logits_processors=procs, prompt_cache=[])
    return [int(t) for t, _ in gen]


@pytest.mark.parametrize("bad", [False, True])
@pytest.mark.parametrize("kw", [
    {"presence_penalty": 7.0, "presence_context_size": 4},
    {"repetition_penalty": 3.0, "repetition_context_size": 3},
    {"logit_bias": {BAD: -20.0}},
], ids=["presence", "repetition", "bias"])
def test_greedy_equals_mlx_lm_generate_step(kw, bad):
    out = _spec(make_logits_processors(**kw), bad_rows=(0,) if bad else ())
    assert out == _plain(make_logits_processors(**kw))
    assert out != [BAD] * 12


def test_no_processors_follows_the_target():
    assert _spec([]) == [BAD] * 12


def test_first_token_is_sampled_from_normalized_logprobs():
    seen = []

    def sampler(logprobs):
        seen.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    _spec([], n=1, sampler=sampler)
    lse = mx.logsumexp(seen[0], axis=-1)
    assert abs(float(lse.item())) < 1e-4


def test_spec_sampler_is_greedy_none_and_annotated_without_xtc():
    assert make_spec_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0,
                             xtc_probability=0.5) is None
    s = make_spec_sampler(temp=0.7, top_p=0.8, top_k=20, min_p=0.0)
    assert s.gmlx_sampling_params["top_k"] == 20


def _two_way(n_rows):
    # Top token A (index 10) clearly above B (11); XTC with threshold 0.1
    # removes A, leaving B. temp 0.01 makes the categorical draw an argmax.
    row = mx.full((VOCAB,), -30.0).at[10].add(32.0).at[11].add(31.0)
    lp = mx.broadcast_to(row, (n_rows, VOCAB))
    return lp - mx.logsumexp(lp, axis=-1, keepdims=True)


def test_spec_sampler_draws_xtc_per_row():
    kw = dict(temp=0.01, top_p=1.0, top_k=0, min_p=0.0,
              xtc_probability=0.5, xtc_threshold=0.1)
    mx.random.seed(3)
    out = make_spec_sampler(**kw)(_two_way(64)).tolist()
    assert set(out) == {10, 11}
    # One stock sampler call on the same rows takes one draw for all of them.
    mx.random.seed(3)
    stock = make_sampler(**kw)(_two_way(64)).tolist()
    assert len(set(stock)) == 1


def test_spec_sampler_matches_make_sampler_on_one_row():
    kw = dict(temp=0.8, top_p=0.9, top_k=5, min_p=0.0,
              xtc_probability=0.5, xtc_threshold=0.05)
    lp = mx.random.normal((1, VOCAB), key=mx.random.key(1))
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    for seed in range(8):
        mx.random.seed(seed)
        a = make_spec_sampler(**kw)(lp).tolist()
        mx.random.seed(seed)
        b = make_sampler(**kw)(lp).tolist()
        assert a == b


class _Tok:
    eos_token_ids = [EOS]
    gguf_suppress_tokens = [DIGITS[9]]

    def __len__(self):
        return VOCAB

    def encode(self, text):
        return [DIGITS[0]]


def test_spec_sampling_matches_the_plain_chain():
    _, procs = generation.spec_sampling(
        _Tok(), temp=0.0, top_p=1.0, top_k=0, min_p=0.0,
        repetition_penalty=1.0, presence_penalty=0.0)
    # The GGUF's suppressed token is biased away, as in plain decode; a
    # repetition penalty of 1 adds no processor.
    assert len(procs) == 1
    logits = procs[0](mx.array([1]), mx.zeros((1, VOCAB)))
    assert float(logits[0, DIGITS[9]].item()) < -1e3


@pytest.mark.parametrize("kw, wants", [
    ({}, False),
    ({"repetition_penalty": 1.0}, False),
    ({"repetition_penalty": 1.1}, True),
    ({"presence_penalty": 0.5}, True),
    ({"logit_bias": {1: -1.0}}, True),
    ({"xtc_probability": 0.5}, True),
])
def test_processors_take_the_owned_engine(monkeypatch, kw, wants):
    monkeypatch.setenv("GMLX_OWNED_ROUND", "0")
    monkeypatch.setattr("gmlx.spec.speculative.use_owned_engine",
                        lambda drafter, temp: False)
    owned = []
    monkeypatch.setattr(generation, "generate_speculative_owned",
                        lambda *a, **k: owned.append(k) or {"routed": "owned"})
    args = {"xtc_probability": 0.0, "logit_bias": None,
            "repetition_penalty": 0.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, **kw}
    assert generation._wants_processors(**args) is wants
    if wants:
        out = generation._generate_speculative(object(), object(), _Tok(),
                                               [1], **kw)
        assert out == {"routed": "owned"}
        for k, v in kw.items():
            assert owned[0][k] == v
