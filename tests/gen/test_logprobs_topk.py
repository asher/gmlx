"""The serve engine's bounded top-k for logprob alternatives, against a
full argsort, and the JSON clamp."""

import math

import mlx.core as mx
import numpy as np
import pytest

from gmlx.gen.logprobs import RoundLogprobs, json_logprob, raw_logprobs, top_logprobs
from gmlx.spec.topk import HIER_MIN_V


def _reference(lp, k):
    a = np.array(lp)
    ids = np.argsort(-a, axis=-1, kind="stable")[:, :k]
    return ids, np.take_along_axis(a, ids, axis=-1)


def _rows(v, n=3, masked=False):
    rng = np.random.default_rng(v)
    a = rng.standard_normal((n, v)).astype(np.float32) * 4
    if masked:
        a[0, rng.choice(v, size=v - 5, replace=False)] = -np.inf
    return raw_logprobs(mx.array(a))


@pytest.mark.parametrize("v", [64, HIER_MIN_V, 248_320])
@pytest.mark.parametrize("k", [1, 20, 100])
def test_top_logprobs_matches_a_full_sort(v, k):
    lp = _rows(v)
    ids, vals = top_logprobs(lp, k)
    ref_ids, ref_vals = _reference(lp, min(k, v))
    np.testing.assert_array_equal(np.array(vals), ref_vals)
    assert ids.dtype == mx.int32
    # Ids may differ only where values tie.
    got = np.array(ids)
    a = np.array(lp)
    np.testing.assert_array_equal(np.take_along_axis(a, got, axis=-1), ref_vals)


def test_k_equal_to_the_vocabulary_sorts_every_column():
    lp = _rows(64)
    ids, vals = top_logprobs(lp, 64)
    assert sorted(np.array(ids)[0].tolist()) == list(range(64))
    assert (np.diff(np.array(vals), axis=-1) <= 0).all()


@pytest.mark.parametrize("v", [64, 248_320])
def test_rows_with_masked_columns_still_return_k_values(v):
    lp = _rows(v, masked=True)
    ids, vals = top_logprobs(lp, 20)
    row = np.array(vals)[0]
    assert np.isfinite(row[:5]).all() and np.isneginf(row[5:]).all()
    _, ref_vals = _reference(lp, 20)
    np.testing.assert_array_equal(np.array(vals), ref_vals)


def test_raw_logprobs_are_f32_of_bf16_logits():
    logits = mx.array(np.linspace(-3, 3, 64, dtype=np.float32)[None]).astype(mx.bfloat16)
    lp = raw_logprobs(logits)
    assert lp.dtype == mx.float32
    ref = np.array(logits.astype(mx.float32))
    ref = ref - np.log(np.exp(ref).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(np.array(lp), ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("x,want", [
    (-1.5, -1.5), (0.0, 0.0), (-1e6, -9999.0), (-math.inf, -9999.0),
    (math.nan, -9999.0),
])
def test_json_logprob_clamps_like_vllm(x, want):
    assert json_logprob(x) == want


def test_round_logprobs_gather_ids_outside_the_top_k():
    rng = np.random.default_rng(1)
    logits = mx.array(rng.standard_normal((2, 3, 64)).astype(np.float32))
    lp = np.array(raw_logprobs(logits))
    drafts = mx.array([[5, 6], [7, 8]], dtype=mx.int32)
    rl = RoundLogprobs(logits, 4, drafts=drafts)
    # Row 0: two drafts then an id outside the top 4. Row 1: the argmax.
    low = int(np.argsort(lp[0, 2])[0])
    top = int(np.argmax(lp[1, 0]))
    got = rl.entries([[5, 6, low], [top]])
    assert [e[0] for e in got[0]] == pytest.approx(
        [lp[0, 0, 5], lp[0, 1, 6], lp[0, 2, low]], abs=1e-6)
    assert got[1][0][0] == pytest.approx(lp[1, 0, top], abs=1e-6)
    assert got[1][0][1][0] == top
    assert len(got[0][2][1]) == 4
