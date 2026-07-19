#!/usr/bin/env python3
"""Stochastic p/q acceptance walk (GMLX_MTP_STOCH_ACCEPT).

Pins the Leviathan properties that make the scheme lossless: the emitted
marginal equals the target's effective sampling distribution regardless of
the proposal q, per-position acceptance equals sum(min(p, q)), and the
target-distribution reconstruction matches the real serve sampler.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import mlx.core as mx  # noqa: E402

import gmlx.speculative as sp  # noqa: E402
from gmlx.speculative import (  # noqa: E402
    _pq_probs,
    _stoch_supported_sampler,
    _stoch_target_probs,
    _stochastic_walk,
    annotate_sampling_params,
    set_stoch_accept,
    use_owned_engine,
)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _annotated_sampler(temp=1.0, top_p=1.0, top_k=0, min_p=0.0):
    def s(logprobs):  # never called by the stochastic walk
        raise AssertionError("walk must not invoke the sampler directly")
    annotate_sampling_params(s, temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    return s


def _fake_target(logits):
    lm = SimpleNamespace(
        speculative_logits_from_hidden=lambda hidden: mx.array(logits))
    verify = SimpleNamespace(target_tokens=None, hidden=mx.zeros((1, 1, 1)))
    return lm, verify


def test_set_stoch_accept_drives_owned_routing(monkeypatch):
    monkeypatch.setattr(sp, "_STOCH_ACCEPT", False)
    plain = SimpleNamespace(requires_owned_engine=False)
    owned = SimpleNamespace(requires_owned_engine=True)
    assert not use_owned_engine(plain, 1.0)   # default off: stock round
    assert use_owned_engine(owned, 0.0)       # drafter contract still routes
    set_stoch_accept(True)
    assert use_owned_engine(plain, 1.0)       # opt-in routes sampled runs
    assert not use_owned_engine(plain, 0.0)   # greedy stays on the stock round
    set_stoch_accept(False)
    assert not use_owned_engine(plain, 1.0)


def test_supported_sampler_detection():
    assert not _stoch_supported_sampler(None)
    assert _stoch_supported_sampler(_annotated_sampler())
    assert _stoch_supported_sampler(_annotated_sampler(min_p=0.05))
    assert not _stoch_supported_sampler(lambda lp: lp)  # opaque
    duck = SimpleNamespace(_filtered=lambda lp: None, temperature=1.0)
    assert _stoch_supported_sampler(duck)


def test_full_accept_emits_bonus():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 2, size=(3, 16)).astype(np.float32)
    lm, verify = _fake_target(logits)
    p = _softmax(logits)
    # q == p: min(1, p/q) accepts every position
    q_rows = [mx.array(p[0]), mx.array(p[1])]
    drafts = mx.array([[int(p[0].argmax()), int(p[1].argmax())]])
    acc, new = _stochastic_walk(
        lm, verify, drafts, _annotated_sampler(), 100, q_rows)
    assert acc == 2
    assert len(new) == 3  # both drafts + a bonus sampled from p[2]


def test_zero_p_draft_rejects_at_zero_and_budget_clamp():
    logits = np.full((3, 16), 0.0, dtype=np.float32)
    logits[0, 5] = 30.0  # p0 ~ one-hot on 5
    lm, verify = _fake_target(logits)
    q = np.zeros(16, dtype=np.float32)
    q[7] = 1.0  # proposal mass entirely on a token with p ~ 0
    drafts = mx.array([[7, 7]])
    acc, new = _stochastic_walk(
        lm, verify, drafts, _annotated_sampler(), 1, [mx.array(q), mx.array(q)])
    assert acc == 0
    assert len(new) == 1  # residual sample only, clamped to budget
    assert new[0] != 7    # residual max(p - q, 0) excludes the rejected token


def test_acceptance_rate_and_output_marginal_match_theory():
    # Empirical: accept rate at pos 0 == sum(min(p, q)); the emitted first
    # token's marginal == p regardless of the (different) proposal q.
    mx.random.seed(11)
    rng = np.random.default_rng(11)
    v = 12
    logits = rng.normal(0, 1.5, size=(2, v)).astype(np.float32)
    q_logits = rng.normal(0, 1.5, size=v).astype(np.float32)  # mismatched q
    p = _softmax(logits)
    q = _softmax(q_logits[None])[0]
    lm, verify = _fake_target(logits)
    sampler = _annotated_sampler()

    trials = 3000
    drafts_np = rng.choice(v, size=trials, p=q)
    accepts = 0
    first_counts = np.zeros(v)
    for t in range(trials):
        acc, new = _stochastic_walk(
            lm, verify, mx.array([[int(drafts_np[t])]]), sampler, 100,
            [mx.array(q)])
        accepts += acc
        first_counts[new[0]] += 1

    accept_rate = accepts / trials
    expected = np.minimum(p[0], q).sum()
    assert abs(accept_rate - expected) < 0.03

    tv = 0.5 * np.abs(first_counts / trials - p[0]).sum()
    assert tv < 0.05, f"emitted marginal deviates from target p (TV={tv:.3f})"


def test_target_probs_matches_serve_fast_sampler():
    from gmlx.server_patches.sampling import _FastPositionedSampler

    mx.random.seed(5)
    rng = np.random.default_rng(5)
    logits = rng.normal(0, 2.5, size=(1, 64)).astype(np.float32)
    sampler = _FastPositionedSampler(
        temperature=0.7, top_p=0.8, top_k=20, min_p=0.02)
    assert _stoch_supported_sampler(sampler)

    p = np.array(_stoch_target_probs(sampler, mx.array(logits)))[0]
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-5)

    lp = mx.array(logits) - mx.logsumexp(mx.array(logits), axis=-1, keepdims=True)
    draws = 4000
    counts = np.zeros(64)
    toks = np.array(mx.concatenate(
        [sampler(lp) for _ in range(draws)]))
    for t in toks:
        counts[t] += 1
    tv = 0.5 * np.abs(counts / draws - p).sum()
    assert tv < 0.05, f"reconstructed p deviates from serve sampler (TV={tv:.3f})"


def test_annotated_probs_match_pq_probs():
    rng = np.random.default_rng(9)
    logits = mx.array(rng.normal(0, 2, size=(3, 33)).astype(np.float32))
    s = _annotated_sampler(temp=0.7, top_p=0.8, top_k=10)
    got = np.array(_stoch_target_probs(s, logits))
    ref = np.array(_pq_probs(logits, 0.7, 10, 0.8))
    np.testing.assert_allclose(got, ref, atol=1e-6)

    s = _annotated_sampler(temp=1.0, top_p=0.95, min_p=0.05)
    got = np.array(_stoch_target_probs(s, logits))
    ref = np.array(_pq_probs(logits, 1.0, 0, 0.95, min_p=0.05))
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_stoch_draft_sampler_stashes_and_samples_from_sharp_q(monkeypatch):
    monkeypatch.setattr(sp, "_STOCH_DRAFT", (0.6, 4, 1.0))
    mx.random.seed(7)
    rng = np.random.default_rng(7)
    stash: list = []
    sampler = sp._stoch_draft_sampler(stash)
    logits = mx.array(rng.normal(0, 2, size=(1, 1, 32)).astype(np.float32))
    tok = sampler(logits)
    assert tok.shape == (1, 1)
    assert len(stash) == 1 and stash[0].shape == (32,)
    q = np.array(stash[0])
    np.testing.assert_allclose(q.sum(), 1.0, atol=1e-5)
    assert (q > 0).sum() == 4  # top-k truncated proposal
    assert q[int(tok.reshape(-1)[0].item())] > 0  # sampled inside the support
