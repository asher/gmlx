"""Owned gemma4 mask/attention classes: selection, construction pairing,
identity against the patched oracle, mirror drift alarms, and the owned
SDPA dispatch composing the hd512 row route.

The numeric oracle is the patched path (gemma4_sync installed on the
stock classes), which carries the certified production numerics; owned
and oracle bodies are line-identical modulo the substitution tables, so
the identity asserts are bit-exact, not allclose.
"""

import ast
import inspect
import textwrap

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.gemma4.language")

from mlx_vlm.models.gemma4 import language as _G
from mlx_vlm.models.gemma4.config import TextConfig

import gmlx.upstream.attn_hd512 as attn_hd512
import gmlx.models.gemma4.batched_sdpa as gemma4_batched_sdpa
import gmlx.models.gemma4.owned as gemma4_owned
import gmlx.models.gemma4.sync as gemma4_sync
from gmlx.models.gemma4.owned import (
    OwnedGemma4Attention,
    OwnedGemma4LanguageModel,
    OwnedGemma4TextModel,
    is_owned_language_model,
)


def _cfg():
    return TextConfig(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=6,
        intermediate_size=128,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        num_key_value_heads=2,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=0,
        sliding_window=32,
        sliding_window_pattern=3,
        tie_word_embeddings=True,
    )


def _pair():
    """Seed-identical (owned, stock) LanguageModels with the patch oracle
    installed on the stock classes."""
    gemma4_sync.install_gemma4_nosync()
    mx.random.seed(13)
    owned = OwnedGemma4LanguageModel(_cfg())
    mx.random.seed(13)
    stock = _G.LanguageModel(_cfg())
    mx.eval(owned.parameters(), stock.parameters())
    return owned, stock


def _ids(*toks):
    return mx.array([list(toks)])


PROMPT = (3, 17, 42, 99, 7, 63, 5, 28)


# ---------------------------------------------------------------------------
# selection and construction
# ---------------------------------------------------------------------------


def test_loader_selects_owned_by_default(monkeypatch):
    import gmlx.load.loader as loader

    monkeypatch.delenv("GMLX_GEMMA_OWNED", raising=False)
    cls, build = loader._mtp_target_classes("gemma4_text")
    assert cls is OwnedGemma4LanguageModel


def test_loader_env_reverts_to_stock(monkeypatch):
    import gmlx.load.loader as loader

    monkeypatch.setenv("GMLX_GEMMA_OWNED", "0")
    cls, build = loader._mtp_target_classes("gemma4_text")
    assert cls is _G.LanguageModel


def test_construction_pair_matches_stock():
    owned, stock = _pair()
    from mlx.utils import tree_flatten

    po = dict(tree_flatten(owned.parameters()))
    ps = dict(tree_flatten(stock.parameters()))
    assert set(po) == set(ps)
    for k in po:
        assert mx.array_equal(po[k], ps[k]), f"weight drift at {k}"
    assert set(vars(owned)) == set(vars(stock))
    assert set(vars(owned.model)) == set(vars(stock.model))


def test_owned_tree_classes():
    owned, stock = _pair()
    assert isinstance(owned.model, OwnedGemma4TextModel)
    for layer in owned.model.layers:
        # attention is rebound; the layer body stays the stock class so
        # the fused-MoE swap eligibility (type name + module) keeps firing
        assert type(layer.self_attn) is OwnedGemma4Attention
        assert isinstance(layer.self_attn, _G.Attention)
        assert type(layer) is _G.DecoderLayer
    assert is_owned_language_model(owned)
    assert not is_owned_language_model(stock)


def test_owned_inherits_mtp_hooks():
    for hook in (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_draft_hidden",
        "chunked_prefill_policy",
    ):
        assert hasattr(OwnedGemma4LanguageModel, hook)
        assert getattr(OwnedGemma4LanguageModel, hook) is getattr(
            _G.LanguageModel, hook
        )


# ---------------------------------------------------------------------------
# identity vs the patched oracle
# ---------------------------------------------------------------------------


def test_prefill_decode_identity_and_engagement():
    owned, stock = _pair()
    co, cs = owned.make_cache(), stock.make_cache()

    m0 = gemma4_owned.owned_mask_call_count()
    s0 = gemma4_owned.owned_sdpa_call_count()

    oo = owned(_ids(*PROMPT), cache=co)
    os_ = stock(_ids(*PROMPT), cache=cs)
    assert mx.array_equal(oo.logits, os_.logits)

    # decode steps: int offsets > 0, qL == 1
    tok = int(mx.argmax(oo.logits[:, -1]).item())
    for _ in range(4):
        oo = owned(_ids(tok), cache=co)
        os_ = stock(_ids(tok), cache=cs)
        assert mx.array_equal(oo.logits, os_.logits)
        tok = int(mx.argmax(oo.logits[:, -1]).item())

    # second prefill chunk: qL > 1 against a warm cache exercises the
    # sliding-branch prefix decision the owned _make_masks carries
    oo = owned(_ids(9, 21, 33), cache=co)
    os_ = stock(_ids(9, 21, 33), cache=cs)
    assert mx.array_equal(oo.logits, os_.logits)

    assert gemma4_owned.owned_mask_call_count() > m0
    assert gemma4_owned.owned_sdpa_call_count() > s0


def test_capture_and_sink_identity():
    owned, stock = _pair()
    oo = owned(_ids(*PROMPT), return_hidden=True, return_shared_kv=True)
    os_ = stock(_ids(*PROMPT), return_hidden=True, return_shared_kv=True)
    assert mx.array_equal(oo.logits, os_.logits)
    assert len(oo.hidden_states) == len(os_.hidden_states) == 1
    assert mx.array_equal(oo.hidden_states[-1], os_.hidden_states[-1])
    assert set(oo.shared_kv_states) == set(os_.shared_kv_states)
    for k in oo.shared_kv_states:
        for a, b in zip(oo.shared_kv_states[k], os_.shared_kv_states[k]):
            assert mx.array_equal(a, b)


class _ArrayOffsetCache:
    """Batched-cache stand-in: array offset advanced in place by
    update_and_fetch, the aliasing hazard the snapshot guards."""

    def __init__(self, offsets):
        self.offset = mx.array(offsets)

    def update_and_fetch(self, keys, values):
        self.offset += 1
        return keys, values


def test_array_offset_snapshot_identity():
    owned, stock = _pair()
    attn_o = owned.model.layers[0].self_attn
    attn_s = stock.model.layers[0].self_attn
    x = mx.random.normal((2, 1, 64))
    mx.eval(x)

    co, cs = _ArrayOffsetCache([5, 3]), _ArrayOffsetCache([5, 3])
    yo, kv_o, off_o = attn_o(x, mask=None, cache=co)
    ys, kv_s, off_s = attn_s(x, mask=None, cache=cs)
    assert mx.array_equal(yo, ys)
    # the returned offset is a pre-advance snapshot, not the live handle
    assert off_o is not co.offset
    assert mx.array_equal(off_o, mx.array([5, 3]))
    assert mx.array_equal(co.offset, mx.array([6, 4]))


def test_mask_objects_shared_per_layer_type():
    owned, _ = _pair()
    model = owned.model
    cache = owned.make_cache()
    cache = cache + [None] * (len(model.layers) - len(cache))
    h = mx.random.normal((1, 4, 64))
    masks = model._make_masks(h, cache, None)
    by_type = {}
    for layer, m in zip(model.layers, masks):
        if layer.layer_type in by_type:
            assert m is by_type[layer.layer_type]
        else:
            by_type[layer.layer_type] = m
    assert set(by_type) == {"sliding_attention", "full_attention"}


def test_mask_decisions_match_patched_oracle():
    owned, stock = _pair()

    def scenarios(lm):
        outs = []
        mx.random.seed(5)
        for qL, stamp_pads in ((8, False), (1, False), (4, False),
                               (1, True), (4, True)):
            cache = lm.make_cache()
            if stamp_pads:
                for c in cache:
                    c.left_padding = mx.array([0])
            padded = cache + [None] * (len(lm.model.layers) - len(cache))
            h = mx.random.normal((1, qL, 64))
            outs.append(lm.model._make_masks(h, padded, None))
        return outs

    for mo, ms in zip(scenarios(owned), scenarios(stock)):
        for a, b in zip(mo, ms):
            assert type(a) is type(b)
            if isinstance(a, mx.array):
                assert mx.array_equal(a, b)
            else:
                assert a == b


# ---------------------------------------------------------------------------
# owned SDPA dispatch
# ---------------------------------------------------------------------------


def test_sdpa_dispatch_claim_gating(monkeypatch):
    sentinel = mx.zeros((1, 1, 1, 1))
    calls = []

    def _fake_claim(q, k, v, cache, scale, mask, sinks):
        calls.append(1)
        return sentinel

    monkeypatch.setattr(gemma4_batched_sdpa, "_claim", _fake_claim)
    monkeypatch.setattr(attn_hd512, "_installed", True)
    q = mx.random.normal((2, 2, 1, 16))
    k = mx.random.normal((2, 2, 8, 16))
    v = mx.random.normal((2, 2, 8, 16))

    out = gemma4_owned._sdpa_dispatch(q, k, v, cache=None, scale=1.0,
                                      mask=None)
    assert out is sentinel and len(calls) == 1

    monkeypatch.setenv("GMLX_G4_BATCHED_SDPA", "0")
    out = gemma4_owned._sdpa_dispatch(q, k, v, cache=None, scale=1.0,
                                      mask=None)
    assert out is not sentinel and len(calls) == 1  # kill switch live

    monkeypatch.delenv("GMLX_G4_BATCHED_SDPA")
    monkeypatch.setattr(attn_hd512, "_installed", False)
    out = gemma4_owned._sdpa_dispatch(q, k, v, cache=None, scale=1.0,
                                      mask=None)
    assert out is not sentinel and len(calls) == 1  # hd512 precondition


class _FakeBatchCache:
    def __init__(self, pads):
        self.left_padding = mx.array(pads)
        self._right_padding = None


def test_sdpa_dispatch_composes_real_row_route(monkeypatch):
    assert attn_hd512.install_hd512_sdpa()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    mx.random.seed(7)
    pads = [0, 4]
    q = mx.random.normal((2, 4, 1, 512))
    k = mx.random.normal((2, 2, 64, 512))
    v = mx.random.normal((2, 2, 64, 512))
    mx.eval(q, k, v)
    cache = _FakeBatchCache(pads)

    n0 = gemma4_batched_sdpa.claims()
    got = gemma4_owned._sdpa_dispatch(q, k, v, cache=cache, scale=0.125,
                                      mask=None)
    assert gemma4_batched_sdpa.claims() == n0 + 1

    pos = mx.arange(64)[None, None, None, :]
    ref_mask = pos >= mx.array(pads)[:, None, None, None]
    ref = gemma4_owned._base_sdpa(q, k, v, cache=cache, scale=0.125,
                                  mask=ref_mask)
    assert mx.abs(got - ref).max().item() < 1e-4


# ---------------------------------------------------------------------------
# mirror drift alarms
# ---------------------------------------------------------------------------


def _norm(fn) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    f = tree.body[0]
    if (
        f.body
        and isinstance(f.body[0], ast.Expr)
        and isinstance(f.body[0].value, ast.Constant)
        and isinstance(f.body[0].value.value, str)
    ):
        f.body = f.body[1:]
    return ast.unparse(tree)


def _apply(text: str, table) -> str:
    for old, new, count in table:
        found = text.count(old)
        assert found == count, (
            f"substitution {old!r} matched {found} times, expected {count} "
            f"- upstream drifted or the table is stale"
        )
        text = text.replace(old, new)
    return text


def _assert_mirror(owned_fn, upstream_fn, table, name):
    expected = _apply(_norm(upstream_fn), table)
    assert _norm(owned_fn) == expected, (
        f"owned mirror of {name} drifted from upstream beyond its "
        f"substitution table - re-mirror and update the table"
    )


def _upstream(fn):
    """The pristine upstream body: gemma4_sync may already be installed in
    this process, and its replacements carry the original as _gmlx_orig."""
    return getattr(fn, "_gmlx_orig", fn)


def test_make_masks_mirror():
    _assert_mirror(
        gemma4_owned._owned_make_masks,
        _upstream(_G.Gemma4TextModel._make_masks),
        [
            ("def _make_masks(self, h, cache, mm_token_type_ids",
             "def _owned_make_masks(self, h, cache, mm_token_type_ids", 1),
            ("(int(mx.max(mx.array(c.offset)).item()) > 0)",
             "_cache_has_prefix(c)", 1),
            ("create_attention_mask(", "_G.create_attention_mask(", 2),
            ("create_causal_mask(", "_G.create_causal_mask(", 1),
        ],
        "Gemma4TextModel._make_masks",
    )


def test_attention_call_mirror():
    _assert_mirror(
        gemma4_owned._owned_attention_call,
        _upstream(_G.Attention.__call__),
        [
            ("def __call__(self, x",
             "def _owned_attention_call(self, x", 1),
            ("offset = mx.array(cache.offset) if cache is not None else 0",
             "offset = cache.offset if cache is not None else 0\n"
             "        if isinstance(offset, mx.array):\n"
             "            offset = mx.array(offset)", 1),
            ("output = scaled_dot_product_attention(",
             "output = _sdpa_dispatch(", 1),
        ],
        "Attention.__call__",
    )
