"""The two-level indexer at decode width: the candidate source publishes a
position list and the layers after it score and select only those rows
through the kq decode scorer's ``cand`` arm, with the picks of the masked
full-width path."""
from __future__ import annotations

import importlib.util
import pathlib

import mlx.core as mx
import pytest

from gmlx.models.deepseek_v41 import model as v41

_SIBLING = pathlib.Path(__file__).with_name("test_deepseek_v41_model.py")


def _model_module():
    spec = importlib.util.spec_from_file_location("_v41_model_tests_cand", _SIBLING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_positions_list_the_mask():
    mx.random.seed(2)
    floor = mx.finfo(mx.float32).min
    scores = mx.random.normal((2, 3, 37))
    reach = mx.array([[[30], [37], [12]], [[37], [5], [20]]])
    scores = mx.where(mx.arange(37) < reach, scores, floor)
    mask = v41._select_candidate_blocks(scores, 3, 4, floor)
    pos = v41._candidate_positions(scores, 3, 4, floor)
    assert mask.shape == (2, 3, 37) and pos.shape == (2, 3, 12)
    assert pos.dtype == mx.int32
    assert mx.array_equal(v41._candidate_mask(pos, 37), mask)
    for b in range(2):
        for q in range(3):
            listed = sorted(int(p) for p in pos[b, q].tolist() if p >= 0)
            kept = [i for i, m in enumerate(mask[b, q].tolist()) if m]
            assert listed == kept, (b, q)
    assert v41._candidate_positions(scores, 10, 4, floor) is None


def test_switch_keeps_the_mask_path(monkeypatch):
    monkeypatch.setitem(v41._INDEXER_CAND, "on", None)
    monkeypatch.setenv("GMLX_DS41_INDEXER_CAND", "0")
    assert v41._indexer_cand_ok() is False


def test_small_topk_keeps_the_inline_scorer():
    q = mx.zeros((1, 64, 1, 128), mx.float16)
    k = mx.zeros((1, 40, 128), mx.float16)
    w = mx.zeros((1, 1, 64), mx.float16)
    assert v41._indexer_kernel_scorer(q, k, w, 1.0, 4, 0, 1) is None


@pytest.mark.skipif(mx.default_device() != mx.gpu,
                    reason="the kq decode scorer is a Metal kernel")
def test_decode_candidate_list_matches_the_masked_path(monkeypatch):
    """800 prompt tokens put layer 4 (ratio 1) past index_topk and past 96
    candidate blocks of 8; the decode steps then run layer 5 on the 768
    listed rows, and its logits match the masked full-width path."""
    monkeypatch.setitem(v41._INDEXER_CAND, "on", None)
    if not v41._indexer_cand_ok():
        pytest.skip("mlx-kquant's decode scorer has no cand arm here")
    import mlx_kquant as kq

    tm = _model_module()
    args = tm._args(
        index_n_heads=4, index_head_dim=128, index_topk=512,
        candidate_source_layer=4, candidate_topk_blocks=96,
        candidate_block_size=8,
    )
    mx.random.seed(0)
    model = tm._randomized(tm.Model(args))
    prompt = mx.array([[(5 + 2 * i) % 60 for i in range(800)]])
    real = kq.dsa_indexer_score_decode
    calls = []

    def spy(*a, **k):
        calls.append(k.get("cand") is not None)
        return real(*a, **k)

    monkeypatch.setattr(kq, "dsa_indexer_score_decode", spy)
    outs = {}
    for on in (False, True):
        monkeypatch.setitem(v41._INDEXER_CAND, "on", on)
        cache = model.make_cache()
        out = model(prompt, cache=cache)
        mx.eval(out)
        n = len(calls)
        steps = [model(mx.array([[t]]), cache=cache) for t in (33, 35, 37)]
        mx.eval(*steps)
        assert len(calls) > n
        assert any(calls[n:]) == on
        outs[on] = steps
    for a, b in zip(outs[False], outs[True]):
        assert mx.allclose(a, b, atol=1e-4, rtol=1e-4)
