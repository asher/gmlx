#!/usr/bin/env python3
"""p/q counterfactual acceptance logger (GMLX_MTP_PQ_LOG).

Pure-logic tests: the sharpened-proposal probability transform against a
numpy reference, the per-position stats graph (including the tau->0 limit
where a fully sharpened proposal reproduces exact-match acceptance), and the
walk integration (accumulation, misalignment, greedy-skip) via a fake target.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import mlx.core as mx  # noqa: E402

import gmlx.speculative as sp  # noqa: E402
from gmlx.speculative import (  # noqa: E402
    _coupled_walk,
    _pq_expected_tokens,
    _pq_graph,
    _pq_parse,
    _pq_probs,
)


@pytest.fixture(autouse=True)
def clean_pq_stats():
    st = sp._pq_stats
    saved = {k: (list(v) if isinstance(v, list) else v) for k, v in st.items()}
    st["count"], st["match"], st["ceil"], st["cf"] = [], [], [], []
    st["rounds"], st["skipped"], st["misaligned"] = 0, 0, 0
    yield
    st.update(saved)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_pq_parse_formats():
    assert _pq_parse("1.0") == [(1.0, 0, 1.0, 0.0)]
    assert _pq_parse("0.6:20:0.95,0.8:40") == [
        (0.6, 20, 0.95, 0.0), (0.8, 40, 1.0, 0.0)]
    assert _pq_parse("0.7:10:0.9", single=True) == (0.7, 10, 0.9, 0.0)
    assert _pq_parse("1.0:0:0.95:0.05", single=True) == (1.0, 0, 0.95, 0.05)
    with pytest.raises(ValueError):
        _pq_parse("0:20:0.9")
    with pytest.raises(ValueError):
        _pq_parse("1.0:0:0.95:1.5")
    with pytest.raises(ValueError):
        _pq_parse("")


def test_pq_parse_env_degrades_on_malformed(monkeypatch):
    monkeypatch.setenv("GMLX_MTP_PQ_TARGET", "not-a-spec")
    assert sp._pq_parse_env("GMLX_MTP_PQ_TARGET", "1.0:0:1.0",
                            single=True) == (1.0, 0, 1.0, 0.0)
    monkeypatch.setenv("GMLX_MTP_PQ_TARGET", "0.7:10:0.9")
    assert sp._pq_parse_env("GMLX_MTP_PQ_TARGET", "1.0:0:1.0",
                            single=True) == (0.7, 10, 0.9, 0.0)


def test_pq_probs_matches_plain_softmax():
    rng = np.random.default_rng(7)
    logits = rng.normal(0, 3, size=(5, 97)).astype(np.float32)
    lm = mx.array(logits)

    plain = np.array(_pq_probs(lm, 1.0, 0, 1.0))
    np.testing.assert_allclose(plain, _softmax(logits), atol=1e-5)

    tempered = np.array(_pq_probs(lm, 0.6, 0, 1.0))
    np.testing.assert_allclose(tempered, _softmax(logits / 0.6), atol=1e-5)


@pytest.mark.parametrize("temp,top_k,top_p,min_p", [
    (1.0, 20, 0.95, 0.0),   # qwen family base/coding profile
    (0.7, 20, 0.8, 0.0),    # qwen @instruct profile
    (0.6, 8, 1.0, 0.0),
    (1.3, 0, 0.9, 0.0),
    (1.0, 0, 0.95, 0.05),   # bench_tg_depth spec-arm defaults
    (0.7, 20, 0.8, 0.1),
])
def test_pq_probs_matches_mlx_lm_sampler(temp, top_k, top_p, min_p):
    # Oracle: the venv's mlx_lm transforms composed exactly as make_sampler
    # composes them (top-p, then min-p, then top-k, on untempered logprobs;
    # temperature at the categorical).
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    rng = np.random.default_rng(13)
    logits = rng.normal(0, 3, size=(6, 97)).astype(np.float32)
    lm = mx.array(logits)

    lp = lm - mx.logsumexp(lm, axis=-1, keepdims=True)
    if 0.0 < top_p < 1.0:
        lp = apply_top_p(lp, top_p)
    if min_p > 0.0:
        lp = apply_min_p(lp, min_p)
    if top_k > 0:
        lp = apply_top_k(lp, top_k)
    oracle = np.array(mx.softmax(lp * (1 / temp), axis=-1))

    got = np.array(_pq_probs(lm, temp, top_k, top_p, min_p=min_p))
    np.testing.assert_allclose(got, oracle, atol=1e-5)
    np.testing.assert_allclose(got.sum(-1), 1.0, atol=1e-5)


def test_pq_graph_ceiling_identity_and_tau0_limit(monkeypatch):
    monkeypatch.setattr(sp, "_PQ_TARGET", (1.0, 0, 1.0))
    monkeypatch.setattr(sp, "_PQ_SWEEP", [(1.0, 0, 1.0)])
    rng = np.random.default_rng(11)
    logits = rng.normal(0, 3, size=(4, 64)).astype(np.float32)
    lm = mx.array(logits)
    ref = _softmax(logits)

    # q == p: rejection-sampling acceptance is exactly 1
    g = np.array(_pq_graph(lm, [lm[i] for i in range(4)]))
    np.testing.assert_allclose(g[0], ref.max(-1), atol=1e-5)
    np.testing.assert_allclose(g[1], 1.0, atol=1e-5)

    # tau->0 (fully sharpened q): acceptance degenerates to exact-match
    # p(argmax q), the scheme the engine runs today
    g0 = np.array(_pq_graph(lm, [lm[i] * 1e4 for i in range(4)]))
    p_mode = ref[np.arange(4), logits.argmax(-1)]
    np.testing.assert_allclose(g0[1], p_mode, atol=1e-4)


def test_pq_expected_tokens():
    assert _pq_expected_tokens([]) == 1.0
    assert _pq_expected_tokens([0.5, 0.5, 0.5]) == pytest.approx(1.875)


def _fake_target(logits):
    lm = SimpleNamespace(
        speculative_logits_from_hidden=lambda hidden: mx.array(logits))
    verify = SimpleNamespace(target_tokens=None, hidden=mx.zeros((1, 1, 1)))
    return lm, verify


def test_walk_accumulates_pq_stats():
    # 4 verify positions; drafts match target argmax at 0,1 and miss at 2
    logits = np.full((4, 16), -10.0, dtype=np.float32)
    for pos, tok in enumerate([3, 5, 7, 9]):
        logits[pos, tok] = 10.0
    lm, verify = _fake_target(logits)
    drafts = mx.array([[3, 5, 8]])
    q_rows = [mx.array(logits[j]) for j in range(3)]

    acc, new = _coupled_walk(lm, verify, drafts, None, 100, pq=q_rows)

    assert acc == 2
    assert new == [3, 5, 7]
    st = sp._pq_stats
    assert st["rounds"] == 1
    assert st["count"] == [1, 1, 1]
    assert st["match"] == [1.0, 1.0, 0.0]
    assert len(st["cf"][0]) == len(sp._PQ_SWEEP)
    # near-one-hot rows: ceiling ~1 everywhere
    assert all(c > 0.99 for c in st["ceil"])


def test_walk_counts_misaligned_and_skipped():
    logits = np.zeros((3, 8), dtype=np.float32)
    lm, verify = _fake_target(logits)
    drafts = mx.array([[0, 0]])

    _coupled_walk(lm, verify, drafts, None, 100, pq=[mx.array(logits[0])])
    assert sp._pq_stats["misaligned"] == 1
    assert sp._pq_stats["rounds"] == 0

    greedy_verify = SimpleNamespace(
        target_tokens=mx.array([[0, 0, 0]]), hidden=None)
    _coupled_walk(lm, greedy_verify, drafts, None, 100,
                  pq=[mx.array(logits[0]), mx.array(logits[1])])
    assert sp._pq_stats["skipped"] == 1
    assert sp._pq_stats["rounds"] == 0


def test_pq_report_prints_and_resets(monkeypatch, capsys):
    monkeypatch.setattr(sp, "_PQ_LOG", True)
    logits = np.full((2, 16), -10.0, dtype=np.float32)
    logits[0, 3] = 10.0
    logits[1, 5] = 10.0
    lm, verify = _fake_target(logits)
    _coupled_walk(lm, verify, mx.array([[3]]), None, 100,
                  pq=[mx.array(logits[0])])

    sp._pq_report()
    err = capsys.readouterr().err
    assert "[pq-log]" in err and "E[tok/round]" in err
    assert sp._pq_stats["rounds"] == 0 and sp._pq_stats["count"] == []
