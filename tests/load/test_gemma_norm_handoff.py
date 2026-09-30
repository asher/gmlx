"""Norm handoff in the fused gemma-4 decoder layer: each residual add also
produces the norm that reads it (kq.add_rmsnorm_norm), the closing add the
next layer's input norm. Decode logits must equal the stock composition bit
for bit, the next layer must use the handed-off norm only when its input is
the same array, and a handoff must never outlive the call that pops it.
Real kq kernels on the GPU."""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("mlx_vlm.models.gemma4.language")

from mlx_vlm.models.gemma4 import language as _G
from mlx_vlm.models.gemma4.config import TextConfig

import gmlx.load.modules as modules

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the fused gemma-4 layer needs a real GPU and a GPU default device",
                allow_module_level=True)

import mlx_kquant as kq

if not hasattr(kq, "add_rmsnorm_norm"):
    pytest.skip("mlx-kquant build without add_rmsnorm_norm", allow_module_level=True)

N_LAYERS = 4
PROMPT = [[3, 17, 42, 99, 7, 63, 5, 28]]


def _cfg(moe):
    extra = dict(enable_moe_block=True, num_experts=4, top_k_experts=2,
                 moe_intermediate_size=64) if moe else {}
    return TextConfig(
        model_type="gemma4_text",
        hidden_size=128,
        num_hidden_layers=N_LAYERS,
        intermediate_size=256,
        num_attention_heads=4,
        head_dim=32,
        global_head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        num_key_value_heads=2,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=32,
        sliding_window_pattern=2,
        tie_word_embeddings=True,
        **extra,
    )


# Distinct eps per norm, large enough to change bf16 outputs, so a handoff
# that reads the wrong norm's eps shows in the logits.
EPS = {"input_layernorm": 0.3, "post_attention_layernorm": 0.02,
       "pre_feedforward_layernorm": 0.1, "post_feedforward_layernorm": 0.05}


def _model(moe=False, fused=True, distinct_eps=False):
    """bf16 model with non-trivial norm weights and layer scalars; the same
    weights for every call with the same ``moe``."""
    mx.random.seed(7)
    m = _G.LanguageModel(_cfg(moe))
    m.set_dtype(mx.bfloat16)
    for _, mod in m.named_modules():
        if isinstance(mod, nn.RMSNorm):
            mod.weight = (1 + 0.3 * mx.random.normal(mod.weight.shape)).astype(
                mx.bfloat16)
    if distinct_eps:
        for layer in m.model.layers:
            for name, eps in EPS.items():
                getattr(layer, name).eps = eps
            if moe:
                # Fusion eligibility needs the MoE pre-norms and the router
                # to share the pre-feedforward eps.
                pre_eps = EPS["pre_feedforward_layernorm"]
                layer.pre_feedforward_layernorm_2.eps = pre_eps
                layer.router.eps = pre_eps
    for layer in m.model.layers:
        layer.layer_scalar = mx.random.uniform(0.5, 1.5, (1,)).astype(mx.bfloat16)
    mx.eval(m.parameters())
    m.eval()
    if fused:
        assert modules.install_fused_moe_glu(m) >= N_LAYERS
        assert all(getattr(layer, "_kq_fused_gemma_layer", False)
                   for layer in m.model.layers)
    return m


def _decode(m, steps=4, prompt=PROMPT):
    """Prompt logits, then greedy decode logits, one call per step."""
    cache = m.make_cache()
    out = [m(mx.array(prompt), cache=cache).logits[:, -1]]
    for _ in range(steps):
        tok = mx.argmax(out[-1], axis=-1).reshape(-1, 1)
        out.append(m(tok, cache=cache).logits[:, -1])
    mx.eval(out)
    return out


def _input_norm_uses(m, monkeypatch):
    """Record which layers' input_layernorm weights reach mx.fast.rms_norm."""
    ids = {id(layer.input_layernorm.weight): i
           for i, layer in enumerate(m.model.layers)}
    seen = []
    real = mx.fast.rms_norm

    def spy(x, w, eps, **kw):
        if w is not None and id(w) in ids:
            seen.append(ids[id(w)])
        return real(x, w, eps, **kw)

    monkeypatch.setattr(mx.fast, "rms_norm", spy)
    return seen


def test_linked_successors():
    m = _model()
    layers = m.model.layers
    for a, b in zip(layers, layers[1:]):
        assert a.__dict__.get("_kq_next") is b
    assert "_kq_next" not in layers[-1].__dict__


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_logits_match_without_handoff(moe, monkeypatch):
    m = _model(moe)
    on = _decode(m)
    monkeypatch.setattr(modules, "_NORM_HANDOFF_ENABLED", False)
    off = _decode(m)
    for a, b in zip(on, off):
        assert mx.array_equal(a, b)


def test_dense_logits_match_stock():
    fused = _decode(_model(fused=True))
    stock = _decode(_model(fused=False))
    for a, b in zip(fused, stock):
        assert mx.array_equal(a, b)


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_each_seam_uses_its_own_eps(moe, monkeypatch):
    m = _model(moe, distinct_eps=True)
    on = _decode(m)
    monkeypatch.setattr(modules, "_NORM_HANDOFF_ENABLED", False)
    off = _decode(m)
    for a, b in zip(on, off):
        assert mx.array_equal(a, b)


def test_batch_rows_match_without_handoff(monkeypatch):
    m = _model()
    prompt = [PROMPT[0], PROMPT[0][::-1]]
    on = _decode(m, prompt=prompt)
    monkeypatch.setattr(modules, "_NORM_HANDOFF_ENABLED", False)
    off = _decode(m, prompt=prompt)
    for a, b in zip(on, off):
        assert a.shape[0] == 2
        assert mx.array_equal(a, b)


def test_promoted_attention_output_skips_the_handoff(monkeypatch):
    # An f32 o_proj makes the attention output f32 under bf16 activations,
    # which kq.add_rmsnorm_norm would reject. Both seams must fall back.
    m = _model()
    for layer in m.model.layers:
        o = layer.self_attn.o_proj
        o.weight = o.weight.astype(mx.float32)
    mx.eval(m.parameters())
    on = _decode(m)
    monkeypatch.setattr(modules, "_NORM_HANDOFF_ENABLED", False)
    off = _decode(m)
    for a, b in zip(on, off):
        assert mx.array_equal(a, b)


class _Bf16Out(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x):
        return self.inner(x).astype(mx.bfloat16)


def test_other_activation_dtype_skips_the_handoff(monkeypatch):
    # The glue is set up for the first call's dtype (bf16). A later fp16
    # call whose attention output is still bf16 must not reach the kq op
    # with an fp16 residual.
    m = _model()
    l0 = m.model.layers[0]
    l0.self_attn.o_proj = _Bf16Out(l0.self_attn.o_proj)
    x = mx.random.normal((1, 1, 128))
    l0(x.astype(mx.bfloat16))
    got = l0(x.astype(mx.float16))[0]
    monkeypatch.setattr(modules, "_NORM_HANDOFF_ENABLED", False)
    assert mx.array_equal(got, l0(x.astype(mx.float16))[0])


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_handoff_replaces_input_norms(moe, monkeypatch):
    m = _model(moe)
    calls = []
    real = kq.add_rmsnorm_norm

    def count(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(kq, "add_rmsnorm_norm", count)
    seen = _input_norm_uses(m, monkeypatch)
    mx.eval(m(mx.array(PROMPT), cache=m.make_cache()).logits)
    # Post-attention in every layer, the close in every layer with a successor.
    assert len(calls) == 2 * N_LAYERS - 1
    assert seen == [0]
    assert not any("_kq_handoff" in layer.__dict__ for layer in m.model.layers)


def test_other_input_recomputes_the_norm(monkeypatch):
    m = _model()
    l0, l1 = m.model.layers[0], m.model.layers[1]
    x0 = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    x1 = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    out0 = l0(x0)[0]
    assert l1.__dict__["_kq_handoff"][0] is out0
    seen = _input_norm_uses(m, monkeypatch)
    got = l1(x1)[0]
    assert seen == [1]
    assert "_kq_handoff" not in l1.__dict__
    monkeypatch.setattr(modules, "_NORM_HANDOFF_ENABLED", False)
    assert mx.array_equal(got, l1(x1)[0])


def test_stock_path_drops_the_handoff():
    m = _model()
    l0, l1 = m.model.layers[0], m.model.layers[1]
    l0(mx.random.normal((1, 1, 128)).astype(mx.bfloat16))
    assert "_kq_handoff" in l1.__dict__
    # 64 rows takes the stock body, which must still pop the pair.
    l1(mx.random.normal((1, 64, 128)).astype(mx.bfloat16))
    assert "_kq_handoff" not in l1.__dict__
