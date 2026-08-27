"""Owned layer assembly: drift tripwires, construction pairs, MoE identity.

The dispatch-only ``__call__`` bodies and the two forward helpers are
verbatim upstream copies (source-equality-tested); the mirrored
constructors must draw weights in the stock order and produce the same
attribute/module-key sets; the owned MoE tree must be greedy-identical
to a weight-identical stock tree on the live routes.
"""

import inspect
import os
import textwrap
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5_moe.language")

from mlx_vlm.models.cache import ArraysCache, BatchKVCache
from mlx_vlm.models.qwen3_5 import language as _L
from mlx_vlm.models.qwen3_5_moe import language as _ML
from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoeTextConfig
from mlx_vlm.models.qwen3_5_moe.language import LanguageModel as MoeStock
from mlx_vlm.models import activations as _A

import gmlx.models.qwen35.layers as qwen35_layers
import gmlx.models.qwen35.owned as qwen35_owned
from gmlx.upstream.gdn_patches import (
    _patch_gated_delta_tiled_v,
    _patch_mlxvlm_gated_delta_tiled_v,
)

ATOL = 2e-3
_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")

PROMPT = (3, 17, 42, 99, 7, 63, 5, 28)


@pytest.fixture(scope="module", autouse=True)
def _tiled_oracle():
    # Same posture as test_qwen35_owned: production always installs the
    # mlx-lm tiled rebind, and the stock-oracle arm gets the vlm rebind.
    _patch_gated_delta_tiled_v()
    _patch_mlxvlm_gated_delta_tiled_v()


def _norm(fn):
    return [
        line.rstrip()
        for line in textwrap.dedent(inspect.getsource(fn)).splitlines()
        if line.strip()
    ]


def test_copies_match_upstream_source():
    block_cls, layer_cls = qwen35_layers.moe_layer_classes()
    for owned, upstream, name in (
        (qwen35_layers.swiglu, _A.swiglu, "swiglu"),
        (
            qwen35_layers._target_verify_switch_glu,
            _ML._target_verify_switch_glu,
            "_target_verify_switch_glu",
        ),
        (
            qwen35_layers.OwnedQwen3_5DecoderLayer.__call__,
            _L.Qwen3_5DecoderLayer.__call__,
            "Qwen3_5DecoderLayer.__call__",
        ),
        (
            layer_cls.__call__,
            _ML.Qwen3_5MoeDecoderLayer.__call__,
            "Qwen3_5MoeDecoderLayer.__call__",
        ),
    ):
        assert _norm(owned) == _norm(upstream), (
            f"owned copy of {name} drifted from the upstream original"
        )


# ---------------------------------------------------------------------------
# MoE construction pair + identity
# ---------------------------------------------------------------------------


def _cfg():
    return MoeTextConfig(
        model_type="qwen3_5_moe",
        hidden_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        num_experts=8,
        num_experts_per_tok=2,
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


def _pair():
    from mlx.utils import tree_flatten

    owned_cls = qwen35_owned.language_model_class("qwen3_5_moe")
    mx.random.seed(11)
    stock = MoeStock(_cfg(), _top())
    mx.eval(stock.parameters())
    mx.random.seed(11)
    owned = owned_cls(_cfg(), _top())
    mx.eval(owned.parameters())

    s_params = dict(tree_flatten(stock.parameters()))
    o_params = dict(tree_flatten(owned.parameters()))
    assert set(s_params) == set(o_params), "parameter tree keys diverged"
    for k, v in s_params.items():
        assert mx.array_equal(v, o_params[k]).item(), f"weight diverged: {k}"
    mods = [
        (stock, owned, "LanguageModel"),
        (stock.model, owned.model, "Qwen3_5MoeModel"),
    ]
    for i, (s_layer, o_layer) in enumerate(
        zip(stock.model.layers, owned.model.layers)
    ):
        mods.append((s_layer, o_layer, f"Qwen3_5MoeDecoderLayer[{i}]"))
        mods.append((s_layer.mlp, o_layer.mlp, f"Qwen3_5MoeSparseMoeBlock[{i}]"))
    for s_mod, o_mod, name in mods:
        assert set(vars(s_mod)) == set(vars(o_mod)), (
            f"{name} instance attribute set diverged (mirrored constructor "
            f"drifted from upstream)"
        )
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
    return mx.allclose(
        a.astype(mx.float32), b.astype(mx.float32), atol=atol, rtol=0
    ).item()


def test_constructed_tree_is_owned():
    _stock, owned = _pair()
    block_cls, layer_cls = qwen35_layers.moe_layer_classes()
    from gmlx.models.qwen35.attn import OwnedQwen3_5Attention
    from gmlx.models.qwen35.gdn import OwnedQwen3_5GatedDeltaNet

    assert all(type(layer) is layer_cls for layer in owned.model.layers)
    assert all(type(layer.mlp) is block_cls for layer in owned.model.layers)
    assert type(owned.model.layers[0].linear_attn) is OwnedQwen3_5GatedDeltaNet
    assert type(owned.model.layers[3].self_attn) is OwnedQwen3_5Attention
    assert type(owned.model.layers[0].mlp.shared_expert) is (
        qwen35_layers.OwnedQwen3_5MLP
    )


@_NEEDS_GPU
def test_moe_b1_greedy_identity():
    stock, owned = _pair()
    ids = mx.array([list(PROMPT)])
    cs, co = stock.make_cache(), owned.make_cache()
    ls = stock(ids, cache=cs).logits
    mx.eval(ls)
    lo = owned(ids, cache=co).logits
    mx.eval(lo)
    assert _close(ls, lo, ATOL)
    ts, to = mx.argmax(ls[:, -1, :], -1), mx.argmax(lo[:, -1, :], -1)
    for _ in range(6):
        assert int(ts.item()) == int(to.item())
        ls = stock(ts[None], cache=cs).logits
        mx.eval(ls)
        lo = owned(to[None], cache=co).logits
        mx.eval(lo)
        ts, to = mx.argmax(ls[:, -1, :], -1), mx.argmax(lo[:, -1, :], -1)


@_NEEDS_GPU
def test_moe_verify_shaped_sink_identity():
    """The verify route drives the sparse block's target_verify path
    (owned _target_verify_switch_glu + verify_linear gates)."""
    stock, owned = _pair()
    pads = [0, 0]
    ids = mx.array([list(PROMPT), list(PROMPT[::-1])])
    cache_s = _batch_caches(stock, pads)
    cache_o = _batch_caches(owned, pads)
    out = stock(ids, cache=cache_s)
    mx.eval(out.logits)
    out = owned(ids, cache=cache_o)
    mx.eval(out.logits)

    block = mx.array([[5, 28, 3], [17, 42, 99]])
    out_s = stock(
        block,
        cache=cache_s,
        capture_layer_ids=[],
        return_hidden=True,
        return_shared_kv=True,
    )
    mx.eval(out_s.logits)
    out_o = owned(
        block,
        cache=cache_o,
        capture_layer_ids=[],
        return_hidden=True,
        return_shared_kv=True,
    )
    mx.eval(out_o.logits)
    assert _close(out_s.logits, out_o.logits, ATOL)
    assert out_s.gdn_states and len(out_s.gdn_states) == len(out_o.gdn_states)
