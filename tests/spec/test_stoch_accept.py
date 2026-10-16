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
import pytest

import mlx.core as mx  # noqa: E402

import gmlx.spec.speculative as sp  # noqa: E402
from gmlx.spec.speculative import (  # noqa: E402
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
    verify = SimpleNamespace(target_tokens=None, hidden=mx.zeros((1, 1, 1)),
                             logits=None)
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


def _mlx_at_least(want: tuple) -> bool:
    parts = []
    for p in mx.__version__.split(".")[:3]:
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) >= want


@pytest.mark.skipif(
    mx.default_device().type == mx.DeviceType.cpu and _mlx_at_least((0, 32, 0)),
    reason="mlx 0.32.x CPU scheduler use-after-free: the 3000-trial loop "
    "segfaults or silently corrupts a gather (upstream bug, GPU unaffected)",
)
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
    from gmlx.serve.patches.sampling import _FastPositionedSampler

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


def _fast(**kw):
    from gmlx.serve.patches.sampling import _FastPositionedSampler

    return _FastPositionedSampler(**kw)


def test_candidate_walk_keeps_the_law_for_a_filtering_sampler():
    """The serve sampler's walk runs over its k survivors. Every position is
    an independent trial: acceptance is sum(min(p, q)), a rejection never
    re-emits the rejected draft unless the residual keeps it, and the
    emitted token's law is p."""
    mx.random.seed(2)
    rng = np.random.default_rng(2)
    v, n = 32, 20000
    row = rng.normal(0, 1.5, size=v).astype(np.float32)
    sampler = _fast(temperature=0.8, top_p=0.9, top_k=8)
    p = np.array(_stoch_target_probs(sampler, mx.array(row[None])))[0]
    q = _softmax(rng.normal(0, 1.5, size=v)[None])[0].astype(np.float32)
    drafts = rng.choice(v, size=n, p=q / q.sum())
    logits = mx.array(np.tile(row, (n + 1, 1)))
    q_rows = [mx.array(q)] * n
    acc, res, bonus = sp._stoch_walk_graph(
        sampler, logits, mx.array(drafts, dtype=mx.int32), q_rows)
    acc, res = np.array(acc), np.array(res)
    assert abs(acc.mean() - np.minimum(p, q).sum()) < 0.015
    emitted = np.where(acc, drafts, res)
    freq = np.bincount(emitted, minlength=v) / n
    assert 0.5 * np.abs(freq - p).sum() < 0.02
    assert p[int(np.array(bonus)[0])] > 0


def test_candidate_walk_point_mass_keeps_the_law():
    mx.random.seed(4)
    rng = np.random.default_rng(4)
    v, n = 32, 20000
    row = rng.normal(0, 1.5, size=v).astype(np.float32)
    sampler = _fast(temperature=1.0, top_k=6)
    p = np.array(_stoch_target_probs(sampler, mx.array(row[None])))[0]
    d = int(np.argsort(-p)[1])
    acc, res, _ = sp._stoch_walk_graph(
        sampler, mx.array(np.tile(row, (n + 1, 1))),
        mx.full((n,), d, dtype=mx.int32), None)
    acc, res = np.array(acc), np.array(res)
    assert abs(acc.mean() - p[d]) < 0.015 and not (res == d).any()
    freq = np.bincount(np.where(acc, d, res), minlength=v) / n
    assert 0.5 * np.abs(freq - p).sum() < 0.02


def test_walk_does_no_full_vocab_sort_for_a_filtering_sampler(monkeypatch):
    """A verify block's rows take the hierarchical top-k, and the walk
    after it works on k ids: no sort or partition spans the vocab."""
    widths = []
    for name in ("argpartition", "argsort", "sort"):
        orig = getattr(mx, name)

        def wrap(x, *a, _orig=orig, **k):
            widths.append(int(x.shape[-1]))
            return _orig(x, *a, **k)

        monkeypatch.setattr(mx, name, wrap)
    v = 16384
    logits = mx.random.normal((4, v), key=mx.random.key(1))
    q = mx.softmax(mx.random.normal((3, v), key=mx.random.key(2)), axis=-1)
    graph = sp._stoch_walk_graph(_fast(temperature=1.0, top_p=0.95, top_k=20),
                                 logits, mx.array([1, 2, 3], dtype=mx.int32),
                                 [q[i] for i in range(3)])
    mx.eval(*graph)
    assert widths and max(widths) < v



def test_the_walk_on_a_masked_row_of_a_padded_vocab_emits_vocab_ids():
    """A width that is not a multiple of 128 is padded inside the top-k.
    A structured-output row with fewer than k finite logits must still
    walk, draw residuals and draw the bonus over vocab ids only."""
    v = 9000
    row = np.full(v, -np.inf, dtype=np.float32)
    live = [8970, 8985, 8999]
    row[live] = [2.0, 1.0, 0.5]
    logits = mx.array(np.tile(row, (3, 1)))
    mx.random.seed(0)
    out = []
    for _ in range(20):
        _, res, bonus = sp._stoch_walk_graph(
            _fast(temperature=1.0, top_k=64), logits,
            mx.array([8970, 8985], dtype=mx.int32), None)
        out += np.array(res).reshape(-1).tolist() + np.array(bonus).reshape(-1).tolist()
    assert set(out) <= set(live)


def test_sharp_draw_on_a_masked_row_of_a_padded_vocab():
    v = 9000
    row = np.full(v, -np.inf, dtype=np.float32)
    live = [8970, 8985, 8999]
    row[live] = [2.0, 1.0, 0.5]
    idx, q = sp._sharp_draw(mx.array(np.tile(row, (4, 1))), 1.0, 64, 1.0)
    assert set(np.array(idx).tolist()) <= set(live)
    q = np.array(q)
    assert q.shape == (4, v)
    np.testing.assert_allclose(q[:, live].sum(-1), 1.0, atol=1e-6)

@pytest.mark.parametrize("setting", [(0.6, 20, 0.95, 0.0), (1.0, 20, 0.3, 0.0),
                                     (0.8, 10, 1.0, 0.05), (1.0, 64, 0.9, 0.0)])
def test_sharp_draw_on_a_wide_row_is_the_pq_probs_proposal(setting):
    """Built on the top-k ids, the proposal equals _pq_probs on every id,
    and each draw lands where q is positive."""
    v = 16384
    rows = mx.random.normal((3, v), key=mx.random.key(4)) * 3
    idx, q = sp._sharp_draw(rows, *setting)
    ref = np.array(_pq_probs(rows, *setting[:3], min_p=setting[3]))
    q = np.array(q)
    np.testing.assert_allclose(q, ref, atol=1e-6)
    for i, t in enumerate(np.array(idx).tolist()):
        assert q[i, t] > 0


def test_sharp_draw_frequencies_follow_q():
    v, n = 8192, 4000
    rows = mx.broadcast_to(
        mx.random.normal((1, v), key=mx.random.key(6)) * 4, (n, v))
    mx.random.seed(3)
    idx, q = sp._sharp_draw(rows, 0.6, 20, 0.95)
    q0 = np.array(q[0])
    counts = np.bincount(np.array(idx), minlength=v) / n
    assert 0.5 * np.abs(counts - q0).sum() < 0.05


def test_draft_sampler_does_no_full_vocab_sort(monkeypatch):
    """The sharpened draft proposal at the default top_k takes the
    hierarchical top-k: no sort or partition spans the vocab."""
    widths = []
    for name in ("argpartition", "argsort", "sort"):
        orig = getattr(mx, name)

        def wrap(x, *a, _orig=orig, **k):
            widths.append(int(x.shape[-1]))
            return _orig(x, *a, **k)

        monkeypatch.setattr(mx, name, wrap)
    v = 16384
    stash: list = []
    tok = sp._stoch_draft_sampler(stash)(
        mx.random.normal((1, 1, v), key=mx.random.key(1)))
    mx.eval(tok, stash[0])
    assert widths and max(widths) < v
    assert stash[0].shape == (v,) and float(stash[0][int(tok.item())]) > 0
