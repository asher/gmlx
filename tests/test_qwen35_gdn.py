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

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.cache import ArraysCache, BatchKVCache
from mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel as Q35LanguageModel
from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

from gmlx import gdn_patches, qwen35_gdn

ATOL = 2e-3
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
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
    )


def _lm():
    mx.random.seed(11)
    lm = Q35LanguageModel(_cfg(), _top())
    # Eval mode mirrors production loads and is load-bearing here: in
    # training mode the stock prefill takes vlm's own chunked ops scan,
    # which the tiled-V patch does not cover, so the arms diverge.
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
    gdn_patches._patch_mlxvlm_gated_delta_tiled_v()
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


def test_owned_gdn_active(monkeypatch):
    monkeypatch.delenv("GMLX_QWEN_OWNED", raising=False)
    for mt in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        assert qwen35_gdn.owned_gdn_active(mt)
    assert not qwen35_gdn.owned_gdn_active("gemma4")
    assert not qwen35_gdn.owned_gdn_active(None)
    monkeypatch.setenv("GMLX_QWEN_OWNED", "0")
    assert not qwen35_gdn.owned_gdn_active("qwen3_5")


def test_rebind_engagement(monkeypatch):
    monkeypatch.delenv("GMLX_FUSED_GDN", raising=False)
    lm = _lm()
    n = qwen35_gdn.rebind_gdn(lm)
    gdns = _gdns(lm)
    assert n == len(gdns) == 3
    for g in gdns:
        assert type(g) is qwen35_gdn.OwnedQwen3_5GatedDeltaNet
        assert g._gdn_owned_fused
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
