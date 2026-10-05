"""Owned GatedDeltaNet: rebind wiring + route identity vs the patched path.

The oracle for every numerics test is the patched path (class patch +
vlm gated_delta rebind), which carries the certified numerics. Both
arms use the stock LanguageModel class so the comparison isolates the
GDN treatment from the model-level ownership. Toy GDN dims are 32 so
the fused decode and fused verify routes engage at toy scale (the
real-weights probes cover them at real shapes).
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm.models.qwen3_5.language")

from gmlx._vendor.mlx_vlm.models.cache import ArraysCache, BatchKVCache
from gmlx._vendor.mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from gmlx.models.qwen35.language import LanguageModel as Q35LanguageModel
from gmlx.models.qwen35.language import Qwen3_5GatedDeltaNet

import gmlx.upstream.gdn_patches as gdn_patches
import gmlx.models.qwen35.gdn as qwen35_gdn

ATOL = 2e-3
_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")

PROMPT = (3, 17, 42, 99, 7, 63, 5, 28)


def _cfg(**over):
    return Q35TextConfig(**{
        **dict(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
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
        # The base-built arm reads its head order here; the owned arm
        # scans through the patched mlx-lm module.
        kv_head_layout="tiled",
        ),
        **over,
    })


def _top():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
    )


def _lm():
    mx.random.seed(11)
    lm = Q35LanguageModel(_cfg(), _top())
    # Eval mode mirrors production loads and is required here: in
    # training mode the two arms take different scans (the chunked ops
    # scan of the base class, the gmlx.tune.gdn scan of the owned one),
    # whose float error breaks the tolerance.
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def _gdns(lm):
    return [layer.linear_attn for layer in lm.model.layers if layer.is_linear]


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
    out = mx.stack(toks, axis=1)
    # Evaluate before the other arm runs: an unevaluated graph reads
    # through later in-place cache mutations (the campaign aliasing
    # class) and turns to NaN.
    mx.eval(out, logits)
    return out, logits


def _arms(monkeypatch=None, fused=True):
    """(patched-stock lm, owned-rebound lm) with identical weights.

    The patched arm installs the same patch set an MTP load installs;
    the owned arm goes through rebind_gdn. Caller must restore
    Qwen3_5GatedDeltaNet.__call__ (see _saved_call fixture).
    """
    if not fused:
        os.environ["GMLX_FUSED_GDN"] = "0"
    patched = _lm()
    gdn_patches._patch_gated_delta_tiled_v()
    gdn_patches._patch_gated_delta_fused_verify(patched)
    owned = _lm()
    qwen35_gdn.rebind_gdn(owned)
    return patched, owned


@pytest.fixture
def _saved_call(monkeypatch):
    # Restoring __call__ alone is not enough: the install-once ClassPatch
    # latch would turn the next test's install into a no-op and its
    # "patched" arm would silently run stock (test_envflags convention).
    saved = Qwen3_5GatedDeltaNet.__call__
    saved_installed = gdn_patches._FUSED_VERIFY_PATCH.installed
    saved_stock = gdn_patches._FUSED_VERIFY_PATCH.stock
    gdn_patches._FUSED_VERIFY_PATCH.installed = False
    monkeypatch.delenv("GMLX_FUSED_GDN", raising=False)
    yield
    Qwen3_5GatedDeltaNet.__call__ = saved
    gdn_patches._FUSED_VERIFY_PATCH.installed = saved_installed
    gdn_patches._FUSED_VERIFY_PATCH.stock = saved_stock
    os.environ.pop("GMLX_FUSED_GDN", None)


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_owned_gdn_active():
    for mt in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        assert qwen35_gdn.owned_gdn_active(mt)
    assert not qwen35_gdn.owned_gdn_active("gemma4")
    assert not qwen35_gdn.owned_gdn_active(None)


def test_rebind_engagement(monkeypatch):
    monkeypatch.delenv("GMLX_FUSED_GDN", raising=False)
    lm = _lm()
    n = qwen35_gdn.rebind_gdn(lm)
    gdns = _gdns(lm)
    assert n == len(gdns) == 3
    for g in gdns:
        assert type(g) is qwen35_gdn.OwnedQwen3_5GatedDeltaNet
        assert g._gdn_owned_fused
        # The stock rollback reads per-position states, not records.
        assert not g._gdn_records
    # Second walk finds no stock instances left.
    assert qwen35_gdn.rebind_gdn(lm) == 0


def test_rebind_respects_fused_kill(monkeypatch):
    monkeypatch.setenv("GMLX_FUSED_GDN", "0")
    lm = _lm()
    qwen35_gdn.rebind_gdn(lm)
    assert all(not g._gdn_owned_fused for g in _gdns(lm))


# ---------------------------------------------------------------------------
# identity vs the patched path
# ---------------------------------------------------------------------------


def _identity_prefill_and_decode(patched, owned, pads):
    ids = mx.array([list(PROMPT)] * len(pads))
    toks_p, log_p = _greedy_chain(
        patched, ids, _batch_caches(patched, pads), 6
    )
    toks_o, log_o = _greedy_chain(owned, ids, _batch_caches(owned, pads), 6)
    assert mx.array_equal(toks_p, toks_o).item()
    assert _close(log_p, log_o, ATOL)


@_NEEDS_GPU
@pytest.mark.parametrize("fused", (True, False), ids=("fused", "unfused"))
def test_prefill_decode_identity(_saved_call, fused):
    patched, owned = _arms(fused=fused)
    _identity_prefill_and_decode(patched, owned, [0])
    _identity_prefill_and_decode(patched, owned, [2, 0])


@_NEEDS_GPU
@pytest.mark.parametrize("fused", (True, False), ids=("fused", "unfused"))
def test_verify_sink_identity(_saved_call, fused):
    patched, owned = _arms(fused=fused)
    ids = mx.array([list(PROMPT)])
    block = mx.array([[5, 9, 21]])

    outs = []
    for lm in (patched, owned):
        cache = _batch_caches(lm, [0])
        lm(ids, cache=cache)
        out = lm(
            block,
            cache=cache,
            capture_layer_ids=[],
            return_hidden=True,
            return_shared_kv=True,
        )
        mx.eval(out.logits)
        outs.append(out)
    out_p, out_o = outs

    assert _close(out_p.logits, out_o.logits, ATOL)
    assert len(out_p.gdn_states) == len(out_o.gdn_states) > 0
    for entry_p, entry_o in zip(out_p.gdn_states, out_o.gdn_states):
        assert len(entry_p) == len(entry_o)
        for a, b in zip(entry_p, entry_o):
            if isinstance(a, mx.array) or isinstance(b, mx.array):
                assert _close(a, b, ATOL)
            else:
                assert a == b


# ---------------------------------------------------------------------------
# K->V head pairing with fewer K heads than V heads
# ---------------------------------------------------------------------------


def _gdn_layer(hk, hv):
    """The first GDN of a toy tree with ``hk`` K heads and ``hv`` V heads,
    on the owned class."""
    mx.random.seed(11)
    # No order of its own: the layer pairs as the scan in the process does.
    lm = Q35LanguageModel(
        _cfg(linear_num_key_heads=hk, linear_num_value_heads=hv,
             kv_head_layout="grouped"), _top())
    lm.eval()
    mx.eval(lm.parameters())
    qwen35_gdn.rebind_gdn(lm)
    return _gdns(lm)[0]


def _paired_copy(gdn, heads):
    """A copy of ``gdn`` with one K head per V head, K head j being
    ``gdn``'s K head ``heads[j]``, and the conv channel index that maps
    ``gdn``'s channels onto the copy's."""
    ref = _gdn_layer(len(heads), len(heads))
    dk, kd = gdn.head_k_dim, gdn.key_dim
    qk = mx.concatenate([mx.arange(h * dk, (h + 1) * dk) for h in heads])
    idx = mx.concatenate([qk, kd + qk, 2 * kd + mx.arange(gdn.value_dim)])
    for name in ("in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "norm"):
        getattr(ref, name).weight = getattr(gdn, name).weight
    ref.A_log, ref.dt_bias = gdn.A_log, gdn.dt_bias
    ref.in_proj_qkv.weight = gdn.in_proj_qkv.weight[idx]
    ref.conv1d.weight = gdn.conv1d.weight[idx]
    ref._gdn_ba_weight = None
    ref._gdn_owned_fused = False
    return ref, idx


@_NEEDS_GPU
@pytest.mark.parametrize("hk,hv", ((2, 6), (4, 8)), ids=("2k6v", "4k8v"))
@pytest.mark.parametrize("tiled", (False, True), ids=("grouped", "tiled"))
@pytest.mark.parametrize("route", ("fused_decode", "fused_verify",
                                   "unfused_verify"))
def test_fewer_k_heads_pair_as_the_scan_pairs_them(monkeypatch, tiled, route,
                                                   hk, hv):
    """V head hv reads the K head the scan in this process pairs it with:
    hv % Hk once the GGUF tiled-V fixup is in, hv // (Hv / Hk) before it.
    The reference is a copy with Hv K heads that holds that pairing in its
    weights, so it has no pairing to choose. Neither shape has Hk equal to
    Hv / Hk, so a pairing that swaps the two (hv // Hk, or hv % (Hv / Hk))
    fails."""
    from gmlx._vendor.mlx_lm.models import gated_delta as gd

    monkeypatch.setattr(gd, "_gmlx_tiled_v_patched", tiled, raising=False)
    calls = []
    for name in ("_gdn_fused_decode_body", "_gdn_fused_verify_body"):
        orig = getattr(gdn_patches, name)
        monkeypatch.setattr(gdn_patches, name, lambda *a, _o=orig, _n=name,
                            **k: calls.append(_n) or _o(*a, **k))
    gdn = _gdn_layer(hk, hv)
    gdn._gdn_owned_fused = route != "unfused_verify"
    gdn._gdn_records = False
    r = hv // hk
    ref, idx = _paired_copy(
        gdn, [h % hk if tiled else h // r for h in range(hv)])
    S = 1 if route == "fused_decode" else 3
    mx.random.seed(3)
    x = mx.random.normal((1, S, 64)) * 0.5
    conv = mx.random.normal((1, 3, gdn.conv_dim)) * 0.5
    state = mx.random.normal((1, hv, 32, 32)) * 0.1
    outs = []
    for layer, c in ((gdn, conv), (ref, conv[..., idx])):
        cache = ArraysCache(size=2, left_padding=[0])
        cache[0], cache[1] = c, state
        y = layer(x, cache=cache,
                  gdn_sink=None if route == "fused_decode" else [])
        mx.eval(y, cache[1])
        outs.append((y, cache[1]))
    assert calls == {"fused_decode": ["_gdn_fused_decode_body"],
                     "fused_verify": ["_gdn_fused_verify_body"],
                     "unfused_verify": []}[route]
    (y, s), (y_ref, s_ref) = outs
    assert mx.abs(y - y_ref).max().item() < 1e-2 * mx.abs(y_ref).max().item()
    assert mx.abs(s - s_ref).max().item() < 1e-2 * mx.abs(s_ref).max().item()
