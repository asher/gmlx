"""Owned qwen3.5 model-level forward: identity vs stock + wiring.

The owned forward must be a drop-in for the stock mlx-vlm
``Qwen3_5Model.__call__`` on every live route: plain B=1, single-row
batch cache (where the stock class takes its extract/recurse/merge
shortcut and the owned class does not), batched left-padded prefill
(per-row on both arms), and the verify-shaped sink path. Identity is
greedy-token equality plus a logits bound; routes that legitimately
differ (shortcut removed) get the contract-pin tolerance, routes that
run the same ops get a tight one.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_vlm.models.qwen3_5 import language as _L
from mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel as Q35LanguageModel

from gmlx import qwen35_owned
from gmlx.gdn_patches import (
    _patch_gated_delta_tiled_v,
    _patch_mlxvlm_gated_delta_tiled_v,
)
from gmlx.loader import _mtp_target_classes

ATOL = 2e-3  # differing-route bound (shortcut removed / kernel path)
TIGHT_ATOL = 1e-5  # same-ops bound


@pytest.fixture(scope="module", autouse=True)
def _tiled_oracle():
    # Every qwen3.5 GGUF load installs the mlx-lm tiled rebind (plus
    # the vlm rebind on the stock fallback); the stock oracle arm must
    # match. With both arms tiled and in eval mode (see _pair) the GDN
    # scan runs the same mlx-lm kernel on both sides, so the same-ops
    # routes stay bit-tight.
    _patch_gated_delta_tiled_v()
    _patch_mlxvlm_gated_delta_tiled_v()

# The qwen3_5 GDN forward dispatches Metal-only kernels.
_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")

PROMPT = (3, 17, 42, 99, 7, 63, 5, 28)


def _cfg():
    return Q35TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        # GDN head dims 32: the eval-mode mlx-lm gated-delta kernels
        # template on Dk/Dv and reject 16 at build (zero-length array).
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        },
        full_attention_interval=4,
    )


def _top():
    # get_rope_index dereferences vision config + multimodal token ids on
    # every fresh text forward; text-only construction needs the stubs.
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
    )


def _pair():
    """Stock and owned LanguageModel with identical weights.

    Beyond the weight check, the attribute-set and module-key comparisons
    guard the mirrored constructors: an upstream field or submodule added
    to LanguageModel, Qwen3_5Model or Qwen3_5DecoderLayer __init__ shows
    up as a key-set mismatch here (and as a seam-pin failure), not as a
    silent behavioral drift. The mirrored constructors also draw
    parameters in the stock order, which the weight check certifies.

    Both arms run in eval mode: the production loaders call
    ``model.eval()``, and in training mode the two arms take different
    gated-delta scan implementations (mlx-lm ops vs the vlm chunked
    scan) whose float error breaks the same-ops tolerance.
    """
    from mlx.utils import tree_flatten

    mx.random.seed(11)
    stock = Q35LanguageModel(_cfg(), _top())
    mx.eval(stock.parameters())
    mx.random.seed(11)
    owned = qwen35_owned.OwnedQwen3_5LanguageModel(_cfg(), _top())
    mx.eval(owned.parameters())

    s_params = dict(tree_flatten(stock.parameters()))
    o_params = dict(tree_flatten(owned.parameters()))
    assert set(s_params) == set(o_params), "parameter tree keys diverged"
    for k, v in s_params.items():
        assert mx.array_equal(v, o_params[k]).item(), f"weight diverged: {k}"
    for s_mod, o_mod, name in (
        (stock, owned, "LanguageModel"),
        (stock.model, owned.model, "Qwen3_5Model"),
        *(
            (s_layer, o_layer, f"Qwen3_5DecoderLayer[{i}]")
            for i, (s_layer, o_layer) in enumerate(
                zip(stock.model.layers, owned.model.layers)
            )
        ),
    ):
        assert set(vars(s_mod)) == set(vars(o_mod)), (
            f"{name} instance attribute set diverged (mirrored constructor "
            f"drifted from upstream)"
        )
        # Module is a dict: parameter-less submodules live in keys() but in
        # neither vars() nor the parameter tree.
        assert set(s_mod.keys()) == set(o_mod.keys()), (
            f"{name} module key set diverged (mirrored constructor drifted "
            f"from upstream)"
        )
    stock.eval()
    owned.eval()
    return stock, owned


def _batch_caches(lm, pads):
    return [
        ArraysCache(size=2, left_padding=list(pads))
        if layer.is_linear
        else BatchKVCache(list(pads))
        for layer in lm.layers
    ]


def _close(a, b, atol):
    return (
        mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() < atol
    )


def _greedy_chain(lm, ids, cache, steps):
    toks = []
    logits = lm(ids, cache=cache).logits
    for _ in range(steps):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        toks.append(nxt)
        logits = lm(nxt[:, None], cache=cache).logits
    return mx.stack(toks, axis=1), logits


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_loader_selects_owned_by_default(monkeypatch):
    monkeypatch.delenv("GMLX_QWEN_OWNED", raising=False)
    cls, _build = _mtp_target_classes("qwen3_5")
    assert cls is qwen35_owned.OwnedQwen3_5LanguageModel


def test_loader_env_reverts_to_stock(monkeypatch):
    monkeypatch.setenv("GMLX_QWEN_OWNED", "0")
    cls, _build = _mtp_target_classes("qwen3_5")
    assert cls is Q35LanguageModel


def test_owned_moe_class_shape():
    cls = qwen35_owned.language_model_class("qwen3_5_moe")
    from mlx_vlm.models.qwen3_5_moe.language import (
        LanguageModel as MoeLanguageModel,
    )

    assert issubclass(cls, MoeLanguageModel)
    with pytest.raises(ValueError):
        qwen35_owned.language_model_class("gemma4_text")


def test_owned_inherits_mtp_hooks():
    for hook in (
        "speculative_verify_hidden",
        "speculative_verify_logits",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "rollback_speculative_cache",
        "chunked_prefill_policy",
    ):
        assert hasattr(qwen35_owned.OwnedQwen3_5LanguageModel, hook)


# ---------------------------------------------------------------------------
# identity vs stock
# ---------------------------------------------------------------------------


@_NEEDS_GPU
def test_b1_plain_cache_identity():
    stock, owned = _pair()
    ids = mx.array([list(PROMPT)])
    before = qwen35_owned.owned_call_count()
    toks_s, logits_s = _greedy_chain(stock, ids, stock.make_cache(), 6)
    mid = qwen35_owned.owned_call_count()
    toks_o, logits_o = _greedy_chain(owned, ids, owned.make_cache(), 6)
    after = qwen35_owned.owned_call_count()

    assert mid == before, "stock arm engaged the owned forward"
    assert after > mid, "owned arm did not engage the owned forward"
    assert mx.array_equal(toks_s, toks_o).item()
    assert _close(logits_s, logits_o, TIGHT_ATOL)


@_NEEDS_GPU
def test_b1_single_row_batch_cache_identity():
    # Stock takes the extract/recurse/merge shortcut here; owned takes the
    # direct batched path. pads=[0] is the one value where the two routes
    # agree (see the padded test below for the value where they must not).
    stock, owned = _pair()
    ids = mx.array([list(PROMPT)])
    toks_s, logits_s = _greedy_chain(stock, ids, _batch_caches(stock, [0]), 5)
    toks_o, logits_o = _greedy_chain(owned, ids, _batch_caches(owned, [0]), 5)
    assert mx.array_equal(toks_s, toks_o).item()
    assert _close(logits_s, logits_o, ATOL)


@_NEEDS_GPU
def test_b1_padded_batch_cache_fixes_stock():
    # The stock B=1 shortcut extracts a row cache that DROPS left_padding
    # while recursing with the full unsliced input, so pad tokens are
    # attended as content. The owned direct path honors the pads. Owned
    # must match an unpadded reference; stock must NOT - the anti-identity
    # assertion pins the upstream defect, so if a future mlx-vlm fixes the
    # shortcut this test flags that the fix claim can be retired.
    stock, owned = _pair()
    real = list(PROMPT[:5])
    pad = 3
    ref_ids = mx.array([real])
    padded_ids = mx.array([[0] * pad + real])

    # Eval each arm before the next forward runs: an unevaluated logits
    # graph reads through later in-place cache mutations (the same
    # aliasing class as the campaign's offset bug) and turns to NaN.
    ref = stock(ref_ids, cache=stock.make_cache()).logits[:, -1, :]
    mx.eval(ref)
    got_stock = stock(
        padded_ids, cache=_batch_caches(stock, [pad])
    ).logits[:, -1, :]
    mx.eval(got_stock)
    got_owned = owned(
        padded_ids, cache=_batch_caches(owned, [pad])
    ).logits[:, -1, :]
    mx.eval(got_owned)

    assert _close(got_owned, ref, TIGHT_ATOL), (
        "owned padded B=1 diverges from the unpadded reference"
    )
    delta = mx.abs(got_stock.astype(mx.float32) - ref.astype(mx.float32))
    delta = delta.max().item()
    assert delta > 1e-2, (
        f"stock shortcut now honors left padding (delta {delta:.4f}); "
        f"the fix framing in the changelog can be retired"
    )


@_NEEDS_GPU
def test_batched_padded_prefill_identity():
    stock, owned = _pair()
    pads = [2, 0, 1]
    rows = [
        [0, 0, 3, 17, 42, 99, 7, 63],
        [3, 17, 42, 99, 7, 63, 5, 28],
        [0, 3, 17, 42, 99, 7, 63, 5],
    ]
    ids = mx.array(rows)
    toks_s, logits_s = _greedy_chain(stock, ids, _batch_caches(stock, pads), 4)
    toks_o, logits_o = _greedy_chain(owned, ids, _batch_caches(owned, pads), 4)
    assert mx.array_equal(toks_s, toks_o).item()
    assert _close(logits_s, logits_o, ATOL)


@_NEEDS_GPU
def test_fully_padded_row_is_structural():
    # A row whose padding consumes the entire chunk recurses at S=0. The
    # stock class needs gmlx's guard patch for this; the owned forward
    # handles it structurally. Install the guard so the stock arm can be
    # compared at all.
    from gmlx.gdn_patches import _patch_qwen35_empty_sequence_guard

    _patch_qwen35_empty_sequence_guard()
    stock, owned = _pair()
    S = 4
    pads = [S, 0]
    rows = [[0] * S, [3, 17, 42, 99]]
    ids = mx.array(rows)
    out_s = stock(ids, cache=_batch_caches(stock, pads)).logits
    out_o = owned(ids, cache=_batch_caches(owned, pads)).logits
    assert out_s.shape == out_o.shape
    assert _close(out_s[1:], out_o[1:], ATOL)


@_NEEDS_GPU
def test_s0_direct_call_needs_no_guard():
    _stock, owned = _pair()
    empty = mx.zeros((1, 0, 64), dtype=owned.model.embed_tokens.weight.dtype)
    out = owned.model(mx.zeros((1, 0), dtype=mx.int32), inputs_embeds=empty)
    assert out.shape == (1, 0, 64)


@_NEEDS_GPU
def test_verify_shaped_sink_path_identity():
    stock, owned = _pair()
    pads = [0, 0]
    ids = mx.array([list(PROMPT), list(PROMPT[::-1])])
    cache_s = _batch_caches(stock, pads)
    cache_o = _batch_caches(owned, pads)
    stock(ids, cache=cache_s)
    owned(ids, cache=cache_o)

    block = mx.array([[5, 28, 3], [17, 42, 99]])
    out_s = stock(
        block,
        cache=cache_s,
        capture_layer_ids=[],
        return_hidden=True,
        return_shared_kv=True,
    )
    out_o = owned(
        block,
        cache=cache_o,
        capture_layer_ids=[],
        return_hidden=True,
        return_shared_kv=True,
    )
    assert _close(out_s.hidden_states[-1], out_o.hidden_states[-1], ATOL)
    assert _close(out_s.logits, out_o.logits, ATOL)
    # gdn_states entries are per-GDN-layer tuples mixing arrays and None.
    assert len(out_s.gdn_states) == len(out_o.gdn_states)
    assert out_s.gdn_states, "verify-shaped call captured no GDN states"
    for entry_s, entry_o in zip(out_s.gdn_states, out_o.gdn_states):
        assert len(entry_s) == len(entry_o)
        for a, b in zip(entry_s, entry_o):
            if isinstance(a, mx.array) or isinstance(b, mx.array):
                assert _close(a, b, ATOL)
            else:
                assert a == b


@_NEEDS_GPU
def test_hidden_capture_layers_match():
    stock, owned = _pair()
    ids = mx.array([list(PROMPT)])
    out_s = stock(ids, cache=stock.make_cache(), capture_layer_ids=[1, 3])
    out_o = owned(ids, cache=owned.make_cache(), capture_layer_ids=[1, 3])
    assert len(out_s.hidden_states) == len(out_o.hidden_states)
    for a, b in zip(out_s.hidden_states, out_o.hidden_states):
        assert _close(a, b, ATOL)


# ---------------------------------------------------------------------------
# Helper parity: the owned copies against the upstream originals.
# These certify the in-tree bodies stay equal to the pinned release and
# pin the shared memo-attr protocol the stock layers consume; they run on
# both streams (no GDN kernels involved).
# ---------------------------------------------------------------------------


def _same(a, b):
    if isinstance(a, mx.array) or isinstance(b, mx.array):
        return (
            isinstance(a, mx.array)
            and isinstance(b, mx.array)
            and a.shape == b.shape
            and mx.array_equal(a, b).item()
        )
    return a == b


def test_parity_pad_row_time():
    x = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8)
    for pad, target in ((0, 5), (-1, 5), (2, 5), (2, 3), (4, 7)):
        got = qwen35_owned._pad_row_time(x, pad, target)
        ref = _L._pad_row_time(x, pad, target)
        assert _same(got, ref), f"pad={pad} target={target}"


def test_parity_extract_row_cache():
    def arrays_pair():
        pair = []
        for _ in range(2):
            c = ArraysCache(size=2)
            c.cache = [mx.arange(12, dtype=mx.float32).reshape(3, 4), None]
            c.lengths = mx.array([5, 7, 9])
            pair.append(c)
        return pair

    c_o, c_s = arrays_pair()
    row_o = qwen35_owned._extract_row_cache(c_o, 1)
    row_s = _L._extract_row_cache(c_s, 1)
    assert type(row_o) is type(row_s)
    assert _same(row_o.cache[0], row_s.cache[0])
    assert row_o.cache[1] is None and row_s.cache[1] is None
    assert _same(row_o.lengths, row_s.lengths)

    def batch_pair(fill):
        pair = []
        for _ in range(2):
            c = BatchKVCache([1, 0])
            if fill:
                k = mx.arange(64, dtype=mx.float32).reshape(2, 2, 4, 4)
                c.update_and_fetch(k, k + 1)
            pair.append(c)
        return pair

    c_o, c_s = batch_pair(fill=True)
    row_o = qwen35_owned._extract_row_cache(c_o, 0)
    row_s = _L._extract_row_cache(c_s, 0)
    assert type(row_o) is type(row_s)
    assert row_o.offset == row_s.offset
    for a, b in zip(row_o.state, row_s.state):
        assert _same(a, b)

    # Empty batch cache with left_padding: both arms hand back a bare
    # KVCache (the dropped-pads mechanism behind the stock B=1 defect;
    # the owned forward never routes here for B=1, this pins the body).
    c_o, c_s = batch_pair(fill=False)
    row_o = qwen35_owned._extract_row_cache(c_o, 0)
    row_s = _L._extract_row_cache(c_s, 0)
    assert type(row_o) is KVCache and type(row_s) is KVCache

    plain = KVCache()
    assert qwen35_owned._extract_row_cache(plain, 0) is plain
    assert _L._extract_row_cache(plain, 0) is plain


def test_parity_attention_mask():
    h1 = mx.zeros((2, 1, 8))
    h4 = mx.zeros((2, 4, 8))

    assert _same(
        qwen35_owned._create_qwen3_5_attention_mask(h4, None),
        _L._create_qwen3_5_attention_mask(h4, None),
    )

    def twin():
        return BatchKVCache([2, 0]), BatchKVCache([2, 0])

    # S>1 delegates on both arms.
    c_o, c_s = twin()
    assert _same(
        qwen35_owned._create_qwen3_5_attention_mask(h4, c_o),
        _L._create_qwen3_5_attention_mask(h4, c_s),
    )

    # S=1 with live pads: sentinel + decode-pad attr, stale attr cleared.
    c_o, c_s = twin()
    c_o._qwen3_5_decode_left_padding = [9, 9]
    c_s._qwen3_5_decode_left_padding = [9, 9]
    got = qwen35_owned._create_qwen3_5_attention_mask(h1, c_o)
    ref = _L._create_qwen3_5_attention_mask(h1, c_s)
    assert got == ref == "left_padded_decode"
    assert (
        c_o._qwen3_5_decode_left_padding
        == c_s._qwen3_5_decode_left_padding
        == [2, 0]
    )
    # Memo keyed on left_padding identity survives a second call.
    memo = c_o._qwen3_5_left_padding_cache
    qwen35_owned._create_qwen3_5_attention_mask(h1, c_o)
    assert c_o._qwen3_5_left_padding_cache is memo

    # S=1 all-zero pads: None on both arms, no decode-pad attr.
    z_o, z_s = BatchKVCache([0, 0]), BatchKVCache([0, 0])
    assert qwen35_owned._create_qwen3_5_attention_mask(h1, z_o) is None
    assert _L._create_qwen3_5_attention_mask(h1, z_s) is None
    assert not hasattr(z_o, "_qwen3_5_decode_left_padding")
    assert not hasattr(z_s, "_qwen3_5_decode_left_padding")


def test_parity_ssm_mask():
    h4 = mx.zeros((2, 4, 8))

    assert qwen35_owned._create_qwen3_5_ssm_mask(h4, None) is None
    assert _L._create_qwen3_5_ssm_mask(h4, None) is None

    def twin(pads):
        return (
            ArraysCache(size=2, left_padding=list(pads)),
            ArraysCache(size=2, left_padding=list(pads)),
        )

    # Zero pads: None plus the no-mask memo on both arms.
    c_o, c_s = twin([0, 0])
    assert qwen35_owned._create_qwen3_5_ssm_mask(h4, c_o) is None
    assert _L._create_qwen3_5_ssm_mask(h4, c_s) is None
    assert (
        c_o._qwen3_5_ssm_no_mask_batch_size
        == c_s._qwen3_5_ssm_no_mask_batch_size
        == 2
    )

    # Live pads: equal masks, no-mask memo absent.
    c_o, c_s = twin([2, 0])
    got = qwen35_owned._create_qwen3_5_ssm_mask(h4, c_o)
    ref = _L._create_qwen3_5_ssm_mask(h4, c_s)
    assert _same(got, ref)
    assert got is not None
    assert not hasattr(c_o, "_qwen3_5_ssm_no_mask_batch_size")
    # Memoized pads give the same answer on a second call.
    assert _same(qwen35_owned._create_qwen3_5_ssm_mask(h4, c_o), ref)


def test_parity_set_decode_left_padding():
    layers = [
        SimpleNamespace(is_linear=flag) for flag in (True, False, False, True)
    ]

    def twin():
        return (
            [BatchKVCache([1, 2]) for _ in layers],
            [BatchKVCache([1, 2]) for _ in layers],
        )

    caches_o, caches_s = twin()
    qwen35_owned._set_qwen3_5_decode_left_padding(caches_o, layers, [1, 2])
    _L._set_qwen3_5_decode_left_padding(caches_s, layers, [1, 2])
    for got, ref, layer in zip(caches_o, caches_s, layers):
        assert hasattr(got, "_qwen3_5_decode_left_padding") == (
            not layer.is_linear
        )
        assert hasattr(got, "_qwen3_5_decode_left_padding") == hasattr(
            ref, "_qwen3_5_decode_left_padding"
        )

    qwen35_owned._set_qwen3_5_decode_left_padding(caches_o, layers, None)
    _L._set_qwen3_5_decode_left_padding(caches_s, layers, None)
    for got, ref in zip(caches_o, caches_s):
        assert not hasattr(got, "_qwen3_5_decode_left_padding")
        assert not hasattr(ref, "_qwen3_5_decode_left_padding")

    # None caches: no-op on both arms.
    qwen35_owned._set_qwen3_5_decode_left_padding(None, layers, [1, 2])
    _L._set_qwen3_5_decode_left_padding(None, layers, [1, 2])


def test_memo_protocol_interop():
    """The memo attrs are a shared format with stock layer code.

    The stock GDN layer advances _qwen3_5_left_padding_info and
    _qwen3_5_lengths_info after every step; an owned writer must produce
    memos the stock advancers mutate correctly and vice versa.
    """
    cache = BatchKVCache([3, 1])
    pads, max_pad = qwen35_owned._qwen3_5_left_padding_info(cache)
    assert (pads, max_pad) == ((3, 1), 3)
    _L._qwen3_5_advance_left_padding_info(cache, 2)
    pads, max_pad = qwen35_owned._qwen3_5_left_padding_info(cache)
    assert (pads, max_pad) == ((1, -1), 1)

    # Reverse direction: stock writes, owned reads the same memo.
    other = BatchKVCache([4, 0])
    ref = _L._qwen3_5_left_padding_info(other)
    memo = other._qwen3_5_left_padding_info
    got = qwen35_owned._qwen3_5_left_padding_info(other)
    assert got == ref
    assert other._qwen3_5_left_padding_info is memo

    lc = ArraysCache(size=2)
    lc.lengths = mx.array([5, 7])
    assert qwen35_owned._qwen3_5_lengths_info(lc) == 5
    _L._qwen3_5_advance_lengths_info(lc, 2)
    assert qwen35_owned._qwen3_5_lengths_info(lc) == 3


def test_copies_match_upstream_source():
    """Every owned helper body is a literal copy of the upstream original.

    The behavioral parity tests certify the routes they drive; this
    covers every branch at once (including the ones no behavioral case
    reaches) by asserting normalized source equality. Retires with the
    helpers when the layer classes are owned.
    """
    import inspect

    def norm(fn):
        return [
            line.rstrip()
            for line in inspect.getsource(fn).splitlines()
            if line.strip()
        ]

    for name in (
        "_qwen3_5_left_padding_info",
        "_qwen3_5_lengths_info",
        "_qwen3_5_set_left_padding_info",
        "_qwen3_5_advance_left_padding_info",
        "_qwen3_5_advance_lengths_info",
        "_create_qwen3_5_ssm_mask",
        "_create_qwen3_5_attention_mask",
        "_set_qwen3_5_decode_left_padding",
        "_extract_row_cache",
        "_pad_row_time",
    ):
        assert norm(getattr(qwen35_owned, name)) == norm(getattr(_L, name)), (
            f"owned copy of {name} drifted from the upstream original"
        )


def test_forward_on_the_cpu_device_skips_the_fused_rope(monkeypatch):
    """Stock MLX raises when the fused MRoPE kernel is dispatched on the
    CPU device; the owned forward takes the cos/sin route there. A fresh
    model for the CPU arm keeps its compiled-apply memo empty, so a fused
    dispatch cannot hide behind a cached kernel."""
    from gmlx import qwen35_rope

    ids = mx.array([[1, 2, 3, 4]])
    _, owned = _pair()
    ref = owned(ids, cache=owned.make_cache()).logits
    mx.eval(ref)

    def _no_kernel(*args, **kwargs):
        raise AssertionError("fused apply dispatched on the CPU device")

    monkeypatch.setattr(qwen35_rope, "_compiled_mrope_apply", _no_kernel)
    _, owned_cpu = _pair()
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        out = owned_cpu(ids, cache=owned_cpu.make_cache()).logits
        mx.eval(out)
    finally:
        mx.set_default_device(prev)
    assert _close(ref, out, 1e-3)
