"""The sliding layers of Flint, Muse Glimmer, Muse Spark and Gemma 4 run a
prefill chunk on kq's window kernel and give the logits stock attention
gives under the band mask. A Gemma 4 chunk that carries image tokens, whose
sliding mask has the bidirectional block overlay, stays on stock attention.
A call the route declines reaches the model module's own SDPA global, where
training attention and the decode routes patch in.

Tiny random bf16 models at the kernel's head dims, prefilled in three
chunks over a 96-position window, against the same weights run in float32.
The MoE models route every token to every expert, so a rounding difference
in attention cannot flip a choice. At Gemma 4's head dim 256, stock bf16
attention is the less precise of the two.
"""

from __future__ import annotations

import sys

import mlx.core as mx
import pytest

import gmlx.cache.window_sdpa as window_sdpa

_WINDOW = 96
_CHUNKS = (128, 128, 64)
_VOCAB = 48


def _flint():
    from gmlx.models.flint.model import Model
    from test_flint import _LAYER_TYPES, _tiny_args

    args = _tiny_args(head_dim=128, sliding_window=_WINDOW, num_experts=4,
                      num_experts_per_tok=4)
    return Model(args), _LAYER_TYPES


def _glimmer():
    from gmlx.models.muse_glimmer.model import Model, ModelArgs

    types = ["sliding_attention", "full_attention", "sliding_attention",
             "sliding_attention"]
    args = ModelArgs(model_type="muse_glimmer", hidden_size=64,
                     intermediate_size=96, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2,
                     head_dim=128, vocab_size=_VOCAB, layer_types=types,
                     sliding_window=_WINDOW)
    return Model(args), types


def _spark():
    from gmlx.models.muse_spark.model import Model
    from test_muse_spark import _LAYER_TYPES, _tiny_args

    args = _tiny_args(head_dim=128, sliding_window=_WINDOW,
                      num_experts_per_tok=8)
    return Model(args), _LAYER_TYPES


def _gemma4():
    from gmlx._vendor.mlx_vlm.models.gemma4.config import TextConfig
    from gmlx.models.gemma4.language import LanguageModel

    types = ["sliding_attention", "sliding_attention", "full_attention",
             "sliding_attention"]
    cfg = TextConfig(model_type="gemma4_text", hidden_size=64,
                     num_hidden_layers=4, intermediate_size=128,
                     num_attention_heads=4, head_dim=256, global_head_dim=128,
                     num_key_value_heads=2, num_kv_shared_layers=0,
                     hidden_size_per_layer_input=0, vocab_size=_VOCAB,
                     sliding_window=_WINDOW, layer_types=types,
                     use_bidirectional_attention="vision")
    lm = LanguageModel(cfg)
    # Gemma 4 attends at scale 1.0 on normed queries. At unit query gains a
    # random model's attention is near one-hot, and its bf16 forward is far
    # from float32 on either path.
    for layer in lm.model.layers:
        norm = layer.self_attn.q_norm
        norm.weight = mx.full(norm.weight.shape, 0.1)
    return lm, types


@pytest.fixture
def calls(monkeypatch):
    """The window kernel's calls, by query rows."""
    kernel = window_sdpa._kq_window_kernel()
    if kernel is None or not window_sdpa._on_gpu():
        pytest.skip("needs an mlx_kquant build with sdpa_prefill_window on NAX")
    rows = []

    def counted(*args, **kwargs):
        rows.append(args[0].shape[2])
        return kernel(*args, **kwargs)

    monkeypatch.setattr(window_sdpa, "_kq_window_kernel", lambda: counted)
    return rows


def _built(build):
    mx.random.seed(0)
    model, types = build()
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model, types


def _prefill(model, ids, token_types=None):
    cache = model.make_cache()
    out, col = [], 0
    for L in _CHUNKS:
        kw = {}
        if token_types is not None:
            kw["mm_token_type_ids"] = token_types[:, col:col + L]
        logits = model(ids[:, col:col + L], cache=cache, **kw)
        logits = getattr(logits, "logits", logits)
        mx.eval(logits)
        out.append(logits)
        col += L
    return mx.concatenate(out, axis=1).astype(mx.float32)


def _errs(model, ids, monkeypatch, token_types=None):
    """The routed and stock bf16 prefills' errors against float32."""
    routed = _prefill(model, ids, token_types)
    monkeypatch.setenv("GMLX_SWA_KQ", "0")
    stock = _prefill(model, ids, token_types)
    model.set_dtype(mx.float32)
    want = _prefill(model, ids, token_types)
    scale = mx.abs(want).max()
    return [(mx.abs(got - want).max() / scale).item() for got in (routed, stock)]


@pytest.mark.parametrize("build", [_flint, _glimmer, _spark, _gemma4],
                         ids=["flint", "muse_glimmer", "muse_spark", "gemma4"])
def test_sliding_layers_take_the_window_kernel(build, calls, monkeypatch):
    model, types = _built(build)
    sliding = types.count("sliding_attention")
    ids = mx.random.randint(0, _VOCAB, (1, sum(_CHUNKS)), key=mx.random.key(3))
    routed, stock = _errs(model, ids, monkeypatch)
    assert calls == [L for L in _CHUNKS for _ in range(sliding)]
    assert routed <= stock * 1.1 + 1e-3


def test_gemma4_chunk_with_image_tokens_stays_on_stock(calls, monkeypatch):
    model, types = _built(_gemma4)
    sliding = types.count("sliding_attention")
    ids = mx.random.randint(0, _VOCAB, (1, sum(_CHUNKS)), key=mx.random.key(4))
    token_types = mx.zeros(ids.shape, dtype=mx.int32)
    token_types[:, 150:200] = 1
    routed, stock = _errs(model, ids, monkeypatch, token_types)
    assert calls == [L for L in (_CHUNKS[0], _CHUNKS[2]) for _ in range(sliding)]
    assert routed <= stock * 1.1 + 1e-3


@pytest.mark.parametrize("build", [_flint, _glimmer, _spark, _gemma4],
                         ids=["flint", "muse_glimmer", "muse_spark", "gemma4"])
def test_declined_calls_reach_the_model_modules_sdpa(build, monkeypatch):
    model, types = _built(build)
    mod = sys.modules[type(model).__module__]
    seen = []
    base = mod.scaled_dot_product_attention

    def spy(*args, **kwargs):
        seen.append(args[0].shape[2])
        return base(*args, **kwargs)

    monkeypatch.setattr(mod, "scaled_dot_product_attention", spy)
    mx.eval(model(mx.array([[1, 2, 3]]), cache=model.make_cache()))
    assert seen == [3] * len(types)
