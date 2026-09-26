"""The letter reader on a tiny mlx-lm Qwen3.5 model.

Batched tails over broadcast prefix views must give the scores that a full
forward over the whole prompt gives, one row at a time, and a warm prefix
must equal a cold one. The recurrent layers read a broadcast state here, so
a view that reached them with the wrong batch would show up as a large
score difference, not a crash."""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx.models import vlm_text_only
from gmlx.systemone import ar_reader, letters
from gmlx.systemone.ar_reader import LetterReader, plan_buckets, run_to_end

qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")

LETTER_IDS = list(range(60, 112))
TOL = 0.05


def _model(tie=False, quantized=False):
    mx.random.seed(7)
    args = qwen3_5.ModelArgs.from_dict({
        "model_type": "qwen3_5", "hidden_size": 64, "intermediate_size": 128,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 32, "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "vocab_size": 128, "tie_word_embeddings": tie, "full_attention_interval": 4,
        "rope_parameters": {"type": "default", "mrope_section": [2, 1, 1],
                            "rope_theta": 100000, "partial_rotary_factor": 0.25},
    })
    model = qwen3_5.Model(args)
    if quantized:
        nn.quantize(model, group_size=32, bits=8)
    model.eval()
    mx.eval(model.parameters())
    return model


class FakeTokens:
    """A character tokenizer inside a fixed chat frame."""

    def ids(self, text):
        return [1] + [ord(c) % 50 + 2 for c in text] + [55, 56]

    def letter_ids(self):
        return list(LETTER_IDS)


def _reference(model, ids):
    """Letter scores at the last position of one full forward."""
    logits = model(mx.array([ids], dtype=mx.int32), cache=model.make_cache())
    logits = getattr(logits, "logits", logits)
    return logits[0, -1, mx.array(LETTER_IDS)].astype(mx.float32)


def _ids(n, seed):
    import random

    rnd = random.Random(seed)
    return [rnd.randrange(2, 58) for _ in range(n)]


@_NEEDS_GPU
@pytest.mark.parametrize("tie,quantized", [(False, False), (True, False), (False, True)],
                         ids=["linear", "tied", "quantized"])
def test_the_float32_rows_path_is_chosen_and_matches_the_logits(tie, quantized):
    model = _model(tie=tie, quantized=quantized)
    reader = LetterReader(model, LETTER_IDS)
    assert reader.check() == "rows"
    prefix = _ids(20, 1)
    cache = run_to_end(reader.prefill(prefix))
    (got,) = run_to_end(reader.tails(cache, prefix, [_ids(9, 2)]))
    want = _reference(model, prefix + _ids(9, 2))
    assert mx.max(mx.abs(mx.array(got) - want)).item() < TOL


@_NEEDS_GPU
def test_a_logit_transform_falls_back_to_full_logits():
    model = _model()

    class Scaled(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.language_model = inner.language_model

        def __call__(self, x, cache=None):
            return self.language_model(x, cache=cache) * 0.5

        def make_cache(self):
            return self.language_model.make_cache()

    scaled = Scaled(model)
    reader = LetterReader(scaled, LETTER_IDS)
    assert reader.check() == "logits"
    prefix = _ids(20, 1)
    cache = run_to_end(reader.prefill(prefix))
    (got,) = run_to_end(reader.tails(cache, prefix, [_ids(9, 2)]))
    want = _reference(scaled, prefix + _ids(9, 2))
    assert mx.max(mx.abs(mx.array(got) - want)).item() < TOL


@_NEEDS_GPU
@pytest.mark.parametrize("wrapped", [False, True], ids=["mlx_lm", "server_wrapper"])
def test_batched_tails_match_full_forwards(wrapped):
    model = _model()
    served = vlm_text_only.Model(model) if wrapped else model
    reader = LetterReader(served, LETTER_IDS)
    assert reader.check() == "rows"
    prefix = _ids(150, 3)       # more than one prefix chunk
    tails = [_ids(n, 10 + n) for n in (3, 7, 12, 12, 20, 31, 4)]
    cache = run_to_end(reader.prefill(prefix))
    got = run_to_end(reader.tails(cache, prefix, tails))
    for t, g in zip(tails, got):
        want = _reference(model, prefix + t)
        assert mx.max(mx.abs(mx.array(g) - want)).item() < TOL
        assert int(mx.argmax(mx.array(g))) == int(mx.argmax(want))


@_NEEDS_GPU
def test_a_tail_longer_than_a_forward_runs_on_its_own_copy():
    model = _model()
    reader = LetterReader(model, LETTER_IDS)
    reader.check()
    prefix = _ids(30, 4)
    long_tail = _ids(ar_reader.FORWARD_TOKENS + 40, 5)
    cache = run_to_end(reader.prefill(prefix))
    before = [mx.array(s) for c in cache for s in c.state if s is not None]
    got = run_to_end(reader.tails(cache, prefix, [long_tail, _ids(5, 6)]))
    want = _reference(model, prefix + long_tail)
    assert mx.max(mx.abs(mx.array(got[0]) - want)).item() < TOL
    after = [s for c in cache for s in c.state if s is not None]
    assert all(mx.array_equal(a, b).item() for a, b in zip(before, after))


@_NEEDS_GPU
def test_a_warm_prefix_equals_a_cold_one():
    model = _model()
    reader = LetterReader(model, LETTER_IDS)
    reader.check()
    prefix = _ids(200, 8)
    a = run_to_end(reader.prefill(prefix))
    b = run_to_end(reader.prefill(prefix))
    for ca, cb in zip(a, b):
        for sa, sb in zip(ca.state, cb.state):
            assert mx.array_equal(sa, sb).item()


@_NEEDS_GPU
def test_rows_do_not_change_with_their_bucket():
    model = _model()
    reader = LetterReader(model, LETTER_IDS)
    reader.check()
    prefix = _ids(40, 9)
    tails = [_ids(12, 20 + i) for i in range(6)]
    cache = run_to_end(reader.prefill(prefix))
    together = run_to_end(reader.tails(cache, prefix, tails))
    apart = run_to_end(reader.tails(cache, prefix, tails[:3]))
    for g, w in zip(together[:3], apart):
        assert mx.max(mx.abs(mx.array(g) - mx.array(w))).item() < 1e-2


@_NEEDS_GPU
def test_a_decision_reuses_a_given_prefix():
    model = _model()
    reader = LetterReader(model, LETTER_IDS)
    tokens = FakeTokens()
    body = {"state": {"ticket": "charged twice"}, "questions": {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": None, "shipping": None}},
        "angry": {"type": "noul", "instructions": "Angry?"},
    }}
    schema = letters.parse(body)
    state = letters.state_text(body)
    seen = {}
    cold = run_to_end(ar_reader.decide_letters(
        reader, tokens, schema, state, on_prefix=lambda ids, c: seen.update(p=(ids, c))))
    warm = run_to_end(ar_reader.decide_letters(
        reader, tokens, schema, state, prefix=seen["p"]))
    assert cold["answers"] == warm["answers"]
    cd, wd = cold["diagnostics"], warm["diagnostics"]
    assert (cd["prefix"]["reused"], wd["prefix"]["reused"]) == (False, True)
    assert cd["prefix"]["tokens"] == len(tokens.ids(letters.prefix_text(state))) - 2
    assert wd["computed_tokens"] == cd["computed_tokens"] - cd["prefix"]["tokens"]
    assert cd["prompt_tokens"] == sum(len(tokens.ids(p)) for p in (
        letters.prompt_text(state, q["instructions"], q["options"])
        for q in schema["questions"]))


def test_buckets_group_similar_lengths_within_the_budget():
    lengths = [5, 50, 6, 48, 7, 100]
    fits = lambda rows, width: rows * width <= 110  # noqa: E731
    groups = plan_buckets(lengths, fits, overhead=10)
    assert sorted(i for g in groups for i in g) == list(range(6))
    for g in groups:
        width = max(lengths[i] for i in g)
        assert len(g) == 1 or fits(len(g), width)
    assert [0, 2, 4] in groups and [3, 1] in groups and [5] in groups


def test_view_kinds_decline_other_caches():
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

    assert ar_reader._view_kinds([KVCache(), ArraysCache(2)]) == ["kv", "arr"]
    assert ar_reader._view_kinds([KVCache(), RotatingKVCache(max_size=8)]) is None
