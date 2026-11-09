"""Engine job steps on a tiny owned Qwen3.5 language model read at the
right positions while a live image row's rope state sits on the model.

Each step runs under ``engine_jobs._step_scope`` with a stale image row's
state planted before it: 3-D positions of the same batch size and
non-zero deltas. A chunked prefill is not bit-equal to one forward, so
the checks use ``TOL``. On this model the largest gap is about 5.4e-3
where float32 products run at TF32 precision, as on an M5 GPU, and about
3e-6 in exact float32. A position fault moves the logits by 0.1 or
more."""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm.models.qwen3_5.language")

from gmlx.cache.kv_cache import make_prompt_cache  # noqa: E402
from gmlx.serve.engine_jobs import _step_scope  # noqa: E402
from gmlx.systemone import ar_reader  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")

TOL = 2e-2
LETTERS = list(range(60, 112))


def _lm():
    import gmlx.models.qwen35.owned as qwen35_owned
    from tests.models.test_qwen35_owned import _cfg, _top

    mx.random.seed(11)
    lm = qwen35_owned.OwnedQwen3_5LanguageModel(_cfg(), _top())
    mx.eval(lm.parameters())
    lm.eval()
    return lm


def _ids(n, seed=5):
    mx.random.seed(seed)
    return mx.random.randint(0, 120, (n,)).tolist()


def _plant_image_row(lm):
    lm._position_ids = mx.broadcast_to(
        (mx.arange(400) // 3)[None, None, :].astype(mx.int32), (3, 1, 400))
    lm._rope_deltas = mx.array([[-37]], dtype=mx.int32)


def _one_shot(lm, ids):
    lm._position_ids = lm._rope_deltas = None
    out = lm(mx.array([ids]), cache=make_prompt_cache(lm))
    return out.logits[0].astype(mx.float32)


def _step(outer, fn):
    _plant_image_row(outer.language_model)
    with _step_scope(outer):
        out = fn()
        if out is not None:
            mx.eval(out)
    return out


def test_three_forwards_under_a_live_image_row_match_one_forward():
    lm = _lm()
    outer = SimpleNamespace(language_model=lm)
    ids = _ids(96)
    ref = _one_shot(lm, ids)
    cache = make_prompt_cache(lm)
    got = [_step(outer, lambda c=chunk: lm(mx.array([c]), cache=cache).logits[0])
           for chunk in (ids[:32], ids[32:64], ids[64:])]
    got = mx.concatenate(got).astype(mx.float32)
    assert mx.max(mx.abs(got - ref)).item() <= TOL


def test_a_first_step_on_a_kept_cache_reads_at_its_offset():
    lm = _lm()
    outer = SimpleNamespace(language_model=lm)
    ids = _ids(96)
    ref = _one_shot(lm, ids)
    cache = make_prompt_cache(lm)
    _step(outer, lambda: lm(mx.array([ids[:64]]), cache=cache).logits)
    tail = _step(outer, lambda: lm(mx.array([ids[64:]]), cache=cache).logits[0])
    assert mx.max(mx.abs(tail.astype(mx.float32) - ref[64:])).item() <= TOL


class _Outer:
    """An outer VLM class over the language model. Like the real ones, it
    numbers every text chunk from 0 and passes the positions in."""

    def __init__(self, lm):
        self.language_model = lm

    def __call__(self, x, cache=None, **kwargs):
        b, n = x.shape
        pos = mx.broadcast_to(mx.arange(n)[None, None, :], (3, b, n))
        return self.language_model(x, cache=cache, position_ids=pos)


def _drive(outer, gen):
    while True:
        try:
            _step(outer, lambda: next(gen))
        except StopIteration as stop:
            return stop.value


def test_a_full_logits_decision_on_a_kept_prefix_reads_its_own_positions():
    lm = _lm()
    outer = _Outer(lm)
    prefix, tail = _ids(150), _ids(6, seed=9)
    ref = _one_shot(lm, prefix + tail)[-1][mx.array(LETTERS)]
    reader = ar_reader.LetterReader(outer, LETTERS)
    reader.path = "logits"

    def decision(kept=None):
        def run():
            cache = kept if kept is not None else (yield from reader.prefill(prefix))
            scores = yield from reader.tails(cache, prefix, [tail])
            return cache, scores[0]
        return run()

    cache, cold = _drive(outer, decision())
    _, warm = _drive(outer, decision(cache))
    for got in (cold, warm):
        assert mx.max(mx.abs(mx.array(got) - ref)).item() <= TOL
