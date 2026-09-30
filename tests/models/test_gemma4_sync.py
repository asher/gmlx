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

pytest.importorskip("gmlx._vendor.mlx_vlm.models.gemma4.language")
from gmlx._vendor.mlx_vlm.models.gemma4 import language as g4
from gmlx._vendor.mlx_vlm.models.gemma4_unified import gemma4_unified as g4u

import gmlx.models.gemma4.sync as gemma4_sync

from test_forward_contract_pins import _g4_lm, _ids, PROMPT

_orig = (g4.Gemma4TextModel._make_masks, g4.Attention.__call__)


def teardown_module(module):
    g4.Gemma4TextModel._make_masks = _orig[0]
    g4.Attention.__call__ = _orig[1]
    gemma4_sync._installed = False


def _install():
    assert gemma4_sync.install_gemma4_nosync()


def _stock(fn):
    while getattr(fn, "_gmlx_orig", None) is not None:
        fn = fn._gmlx_orig
    return fn


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
    # Start the chunking policies from stock: the kill switch must still
    # install their wraps.
    policies = [(g4.LanguageModel, "chunked_prefill_policy"),
                (g4u.Model, "_should_disable_chunked_prefill")]
    stock = []
    for cls, name in policies:
        f = getattr(cls, name)
        stock.append(_stock(f))
        monkeypatch.setattr(cls, name, stock[-1])

    monkeypatch.setenv("GMLX_G4_NOSYNC", "0")
    assert gemma4_sync.install_gemma4_nosync() is False
    # The kill switch keeps the stock body, run under image_masks.
    killed = g4.Gemma4TextModel._make_masks
    assert killed._gmlx_orig is _orig[0] and killed._gmlx_body is _orig[0]
    assert g4.Attention.__call__ is _orig[1]
    for (cls, name), f in zip(policies, stock):
        assert getattr(cls, name)._gmlx_orig is f

    monkeypatch.delenv("GMLX_G4_NOSYNC", raising=False)
    _install()
    patched = (g4.Gemma4TextModel._make_masks, g4.Attention.__call__)
    assert patched[0] is not _orig[0]
    assert patched[0]._gmlx_orig is _orig[0]
    assert patched[0]._gmlx_body is not _orig[0]
    assert patched[1]._gmlx_orig is _orig[1]
    _install()  # second install: no double wrap
    assert g4.Gemma4TextModel._make_masks is patched[0]
    assert g4.Attention.__call__ is patched[1]


def test_installs_see_through_another_wrap(monkeypatch):
    """A linked wrap from another installer on top of a stock function hid
    it. The chunking policy took any link for its own and skipped its
    install, and the nosync builder recorded the image masks layer under
    that wrap as the stock builder."""
    stock_masks = _stock(g4.Gemma4TextModel._make_masks)
    monkeypatch.setattr(g4.Gemma4TextModel, "_make_masks", stock_masks)
    monkeypatch.setattr(g4.Attention, "__call__", _stock(g4.Attention.__call__))
    monkeypatch.setattr(gemma4_sync, "_installed", False)
    policy = _stock(g4.LanguageModel.chunked_prefill_policy)

    def other(self, **kwargs):
        return policy(self, **kwargs)

    other.__dict__["_gmlx_orig"] = policy
    monkeypatch.setattr(g4.LanguageModel, "chunked_prefill_policy", other)
    monkeypatch.setenv("GMLX_G4_NOSYNC", "0")
    assert gemma4_sync.install_gemma4_nosync() is False
    assert g4.LanguageModel.chunked_prefill_policy._gmlx_orig is other

    masks = g4.Gemma4TextModel._make_masks
    assert masks._gmlx_orig is stock_masks

    def other_masks(self, *args, **kwargs):
        return masks(self, *args, **kwargs)

    other_masks.__dict__["_gmlx_orig"] = masks
    g4.Gemma4TextModel._make_masks = other_masks
    monkeypatch.delenv("GMLX_G4_NOSYNC")
    assert gemma4_sync.install_gemma4_nosync()
    assert g4.Gemma4TextModel._make_masks._gmlx_orig is stock_masks


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


def test_array_offsets_snapshotted_ints_pass_through():
    """rope must never receive the cache's live offset array object:
    update_and_fetch advances it with an in-place += between the key rope
    and the query rope, so an aliased offset rotates queries one position
    ahead of keys and gated B>1 decode degenerates (gate-cert 2026-07-25).
    Int offsets keep passing through unwrapped -- that skip is the patch's
    sync win."""
    _install()
    lm = _g4_lm()
    attn = lm.model.layers[0].self_attn

    seen = []
    real_rope = attn.rope

    class SpyRope:
        def __call__(self, x, offset=0):
            seen.append(offset)
            return real_rope(x, offset=0)  # rotation content irrelevant

    attn.rope = SpyRope()

    class FakeCache:
        left_padding = None

        def __init__(self, off):
            self.offset = off

        def update_and_fetch(self, k, v):
            return k, v

    hidden = lm.model.layers[0].self_attn.q_proj.weight.shape[1]

    cache = FakeCache(mx.array([5, 6, 7]))
    attn(mx.zeros((3, 1, hidden)), mask=None, cache=cache)
    assert len(seen) == 2  # keys, then queries
    for off in seen:
        assert isinstance(off, mx.array)
        assert off is not cache.offset  # snapshot, not the live object
        assert mx.array_equal(off, cache.offset).item()

    seen.clear()
    cache_int = FakeCache(9)
    attn(mx.zeros((1, 1, hidden)), mask=None, cache=cache_int)
    assert seen == [9, 9]  # ints stay unwrapped

    attn.rope = real_rope


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
