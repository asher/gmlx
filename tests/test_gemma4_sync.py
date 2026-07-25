"""gemma4_sync: host-sync-free masks/offsets must not change any decision
or any number on the int-offset (B=1) paths, and must skip the offset probe
on array-offset caches.

The full-model pins run the same forwards before and after install and
require bit-identical logits: for int offsets the patched _make_masks makes
the same string/array decisions and the Attention rope sees the same offset
value (int instead of a 0-d wrap), so any drift is a real defect.
"""

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.gemma4.language")
from mlx_vlm.models.gemma4 import language as g4

from gmlx import gemma4_sync

from test_forward_contract_pins import _g4_lm, _ids, PROMPT

_orig = (g4.Gemma4TextModel._make_masks, g4.Attention.__call__)


def teardown_module(module):
    g4.Gemma4TextModel._make_masks = _orig[0]
    g4.Attention.__call__ = _orig[1]
    gemma4_sync._installed = False


def _install():
    assert gemma4_sync.install_gemma4_nosync()


def test_cache_has_prefix_decision():
    class C:
        pass

    c = C()
    c.offset = 0
    assert gemma4_sync._cache_has_prefix(c) is False
    c.offset = 7
    assert gemma4_sync._cache_has_prefix(c) is True
    c.offset = mx.array([0, 0])  # array offsets: never probe, assume prefix
    assert gemma4_sync._cache_has_prefix(c) is True
    c.offset = mx.array(3)
    assert gemma4_sync._cache_has_prefix(c) is True


def test_install_idempotent_and_killable(monkeypatch):
    g4.Gemma4TextModel._make_masks = _orig[0]
    g4.Attention.__call__ = _orig[1]
    gemma4_sync._installed = False

    monkeypatch.setenv("GMLX_G4_NOSYNC", "0")
    assert gemma4_sync.install_gemma4_nosync() is False
    assert g4.Gemma4TextModel._make_masks is _orig[0]

    monkeypatch.delenv("GMLX_G4_NOSYNC", raising=False)
    _install()
    patched = (g4.Gemma4TextModel._make_masks, g4.Attention.__call__)
    assert patched[0] is not _orig[0]
    assert patched[0]._gmlx_orig is _orig[0]
    assert patched[1]._gmlx_orig is _orig[1]
    _install()  # second install: no double wrap
    assert g4.Gemma4TextModel._make_masks is patched[0]
    assert g4.Attention.__call__ is patched[1]


def test_mask_decisions_match_stock_int_offsets():
    lm = _g4_lm()
    tm = lm.model
    ids = _ids(*PROMPT)

    def compare(h, cache):
        got = g4.Gemma4TextModel._make_masks(tm, h, cache)
        want = _orig[0](tm, h, cache)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert type(g) is type(w)
            if isinstance(w, mx.array):
                assert mx.array_equal(g, w).item()
            else:
                assert g == w

    _install()
    h = mx.zeros((1, len(PROMPT), 4))
    fresh = lm.make_cache()
    fresh = fresh + [None] * (len(tm.layers) - len(fresh))
    compare(h, fresh)  # offset 0, qL > 1

    warm = lm.make_cache()
    mx.eval(lm(ids, cache=warm).logits)
    warm_p = warm + [None] * (len(tm.layers) - len(warm))
    compare(mx.zeros((1, 3, 4)), warm_p)  # offset > 0, verify width
    compare(mx.zeros((1, 1, 4)), warm_p)  # decode width


def test_logits_bit_identical_before_and_after():
    lm = _g4_lm()
    prompt = _ids(*PROMPT)
    block = _ids(5, 7, 9)

    def run():
        c = lm.make_cache()
        out_p = lm(prompt, cache=c).logits
        out_d = lm(_ids(4), cache=c).logits
        out_v = lm(block, cache=c).logits
        mx.eval(out_p, out_d, out_v)
        return out_p, out_d, out_v

    g4.Gemma4TextModel._make_masks = _orig[0]
    g4.Attention.__call__ = _orig[1]
    gemma4_sync._installed = False
    stock = run()
    _install()
    patched = run()
    for s, p in zip(stock, patched):
        assert mx.array_equal(s, p).item()
