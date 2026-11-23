"""GMLX_SPEC_VERIFY_ONE_SUBMIT: the verify graph carries its command buffer
boundaries and one async_eval submits it under lifted caps. Tokens, target
cache, drafter ring and RNG are those of the stock caps, through full rounds
and a stop inside a round, alone and with the early draft."""

import mlx.core as mx
import pytest

import gmlx.spec.verify_submit as vs

from test_dflash2_drafter import SEED
from test_early_draft import _plain_rounds, _run, _same  # noqa: F401

kq = pytest.importorskip("mlx_kquant")

pytestmark = pytest.mark.skipif(
    not hasattr(kq, "cb_commit_mark") or mx.default_device() != mx.gpu,
    reason="kq build without cb_commit_mark, or no Metal")


def _one(monkeypatch, on, early=False, **kw):
    """A round loop walk on the tiny qwen3_5 target. Its module is given the
    hook slot so the lever engages; its layers never call the hook, so this
    covers the submit under lifted caps (the whole verify in one buffer).
    The marks are covered on the qwen4_exp model below."""
    import sys
    from test_dflash2_drafter import _target
    mod = sys.modules[type(_target()).__module__]
    monkeypatch.setattr(mod, "_VERIFY_LAYER_HOOK", None, raising=False)
    monkeypatch.setattr(vs, "_ON", on)
    monkeypatch.setattr(vs, "_MARKS", frozenset({1, 2}))
    return _run(monkeypatch, early, **kw)


def _sampler():
    mx.random.seed(SEED)
    return lambda logits: mx.random.categorical(logits * 1.0)


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("sampled", [False, True])
def test_same_tokens_and_state(monkeypatch, early, sampled):
    caps = kq.get_cb_caps()
    want = _one(monkeypatch, False, early,
                sampler=_sampler() if sampled else None)
    subs = vs._state["submits"]
    got = _one(monkeypatch, True, early,
               sampler=_sampler() if sampled else None)
    _same(want, got)
    assert vs._state["submits"] > subs         # one submit per verify
    assert kq.get_cb_caps() == caps            # caps put back


@pytest.mark.parametrize("fused", [False, True])
def test_marks_on_the_qwen4_exp_verify(monkeypatch, fused):
    """The layer hook puts the marks in a qwen4_exp verify; hiddens and cache
    states equal the unmarked ones, plain and with the fused records."""
    import numpy as np
    import gmlx.models.qwen4_exp.model as q4
    from test_qwen4exp_gdn_verify_v2 import _lm
    if fused and not hasattr(kq, "fuse_graph"):
        pytest.skip("kq build without fuse_graph")
    monkeypatch.setattr(q4, "_VERIFY_BLOCKED", False)
    monkeypatch.setattr(q4, "_VERIFY_FUSED_QSA", True)
    monkeypatch.setattr(q4, "_VERIFY_FUSED", fused)
    monkeypatch.setattr(vs, "_ON", True)
    monkeypatch.setattr(vs, "_MARKS", frozenset({1, 2}))
    kq.env_cache(fused)
    caps = kq.get_cb_caps()
    rounds = [(4, 4)] * 4 + [(4, 2)] * 3 + [(3, 3), (3, 1), (4, 4)]

    def walk(marked):
        lm, args = _lm(True)
        assert vs.enabled(lm)
        mx.random.seed(5)
        cache = lm.make_cache()
        mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, 9)),
                   cache=cache).logits)
        out = []
        for S, kept in rounds:
            block = mx.random.randint(0, args.vocab_size, (1, S))
            if marked:
                with vs.marks(lm):
                    hid, _, sink = lm.speculative_verify_hidden(block, cache)
                assert q4._VERIFY_LAYER_HOOK is None
                vs.submit(hid)
            else:
                hid, _, sink = lm.speculative_verify_hidden(block, cache)
            lm.rollback_speculative_cache(cache, sink, kept - 1, S)
            mx.eval(hid, [c.state for c in cache])
            out.append(np.array(hid.astype(mx.float32)))
        out.extend(np.array(a.astype(mx.float32))
                   for c in cache for a in c.state if isinstance(a, mx.array))
        return out

    try:
        want = walk(False)
        made, n = vs._state["made"], kq.cb_commit_marks()
        got = walk(True)
    finally:
        kq.env_cache(False)
    assert vs._state["made"] - made == 2 * len(rounds)
    assert kq.cb_commit_marks() - n == 2 * len(rounds)
    assert kq.get_cb_caps() == caps
    assert len(want) == len(got)
    for x, y in zip(want, got):
        assert x.shape == y.shape and np.array_equal(x, y)


@pytest.mark.parametrize("stop", [1, 2, 4, 7])
@pytest.mark.parametrize("early", [False, True])
def test_a_stop_inside_a_round(monkeypatch, stop, early):
    want = _one(monkeypatch, False, False, stop_after=stop)
    got = _one(monkeypatch, True, early, stop_after=stop)
    _same(want, got)


def test_a_job_hold_keeps_the_stock_caps(monkeypatch):
    from gmlx.serve import cb_phase
    caps = kq.get_cb_caps()
    seen = []
    real = kq.set_cb_caps
    monkeypatch.setattr(vs._kq(), "set_cb_caps",
                        lambda *a: (seen.append(a), real(*a))[1])
    monkeypatch.setitem(cb_phase._state, "holds", 1)
    want = _one(monkeypatch, False)
    plain = vs._state["plain"]
    got = _one(monkeypatch, True)
    _same(want, got)
    assert vs._state["plain"] > plain and not seen
    assert kq.get_cb_caps() == caps


def test_the_lever_is_on_by_default_and_needs_the_slot(monkeypatch):
    import os
    assert os.environ.get("GMLX_SPEC_VERIFY_ONE_SUBMIT") is None and vs._ON

    class NoSlot:
        pass

    assert not vs.enabled(NoSlot())
    assert vs._parse_marks("1, 12,x,") == frozenset({1, 12})


def test_overlapping_marks_put_the_hook_back(monkeypatch):
    # Two engines of one model module build verifies on two threads at
    # once: the hook stays while either builds and goes when both are out.
    import sys
    import types

    mod = types.ModuleType("_fake_verify_module")
    mod._VERIFY_LAYER_HOOK = None
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    lm = type("LM", (), {"__module__": mod.__name__})()
    a, b = vs.marks(lm), vs.marks(lm)
    a.__enter__()
    b.__enter__()
    a.__exit__(None, None, None)
    assert mod._VERIFY_LAYER_HOOK is vs._hook
    b.__exit__(None, None, None)
    assert mod._VERIFY_LAYER_HOOK is None
