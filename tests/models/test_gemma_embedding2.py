"""The EmbeddingGemma 2 text encoder.

A tiny random backbone recorded with transformers 5.22.2
(scripts/embedder_oracle/embeddinggemma2_tiny.py) runs on the CPU stream in
float32, with 16-row query blocks so a 45-token input crosses block edges
and the sliding band. The other tests cover the GGUF config synthesis, the
tensor names and the per-layer input slices of a quantized projection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.models.embedders import gemma_embedding2 as eg2

_ORACLE = Path(__file__).resolve().parents[1] / "fixtures" / "embeddinggemma2_oracle"


@pytest.fixture
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


@pytest.fixture(scope="module")
def oracle():
    with np.load(_ORACLE / "text.npz") as f:
        return {k: f[k] for k in f.files}


def _model(oracle) -> eg2.Model:
    config = json.loads(str(oracle["config"]))
    model = eg2.Model(eg2.ModelArgs.from_dict({"text_config": config}))
    model.load_weights([(k[2:], mx.array(v)) for k, v in oracle.items()
                        if k.startswith("w.")], strict=True)
    mx.eval(model.parameters())
    return model


def _rel_l2(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("case", ["short", "long", "soft"])
def test_matches_transformers(oracle, cpu, monkeypatch, case):
    monkeypatch.setattr(eg2, "Q_BLOCK", 16)
    model = _model(oracle)
    if f"{case}.ids" in oracle:
        out = model(mx.array(oracle[f"{case}.ids"][None]))
    else:
        out = model(inputs_embeds=mx.array(oracle[f"{case}.embeds"][None]))
    hidden = np.array(out.last_hidden_state[0])
    vec = np.array(out.text_embeds[0])
    # Ten times the reference's own float32-vs-float64 gap, at least 1e-5.
    tol = np.maximum(10 * oracle[f"{case}.gap"], 1e-5)
    assert _rel_l2(hidden, oracle[f"{case}.hidden"]) < tol[0]
    assert _rel_l2(vec, oracle[f"{case}.vec"]) < tol[1]


def test_block_size_does_not_change_the_vector(oracle, cpu, monkeypatch):
    model = _model(oracle)
    ids = mx.array(oracle["long.ids"][None])
    whole = np.array(model(ids).text_embeds)
    for block in (4, 16, 33):
        monkeypatch.setattr(eg2, "Q_BLOCK", block)
        np.testing.assert_allclose(np.array(model(ids).text_embeds), whole,
                                   rtol=0, atol=2e-6)


def test_layer_types_and_head_shapes_follow_the_config(oracle):
    model = _model(oracle)
    shapes = [(a.self_attn.is_sliding, a.self_attn.head_dim, a.self_attn.n_kv)
              for a in model.layers]
    assert shapes == [(True, 16, 2), (True, 16, 2), (False, 32, 1)] * 2


def _q8_linear(n_out, n_in, rng):
    from gguf import GGMLQuantizationType as GT
    from gguf import quants
    from mlx_kquant.nn import KQuantLinear

    w = (rng.standard_normal((n_out, n_in)) * 0.1).astype(np.float32)
    lin = KQuantLinear(in_dims=n_in, out_dims=n_out, bias=False, codec="q8_0")
    lin.weight = mx.array(quants.quantize(w, GT.Q8_0).astype(np.uint8))
    return lin


@pytest.mark.parametrize("kind", ["float", "affine", "kquant"])
def test_per_layer_input_equals_a_slice_of_the_full_projection(kind, cpu):
    args = eg2.TextArgs(hidden_size=64, num_hidden_layers=3,
                        hidden_size_per_layer_input=32)
    ple = eg2.PerLayerInputs(args)
    rng = np.random.default_rng(0)
    if kind == "affine":
        nn.quantize(ple, group_size=32, bits=8)
    elif kind == "kquant":
        ple.per_layer_model_projection = _q8_linear(96, 64, rng)
    x = mx.array(rng.standard_normal((1, 5, 64)).astype(np.float32))
    full = ple.per_layer_model_projection(x) * ple.scale
    for i in range(3):
        want = ple.per_layer_projection_norm(full[..., 32 * i:32 * (i + 1)])
        got = ple.layer_input(x, i)
        assert got.dtype == want.dtype
        np.testing.assert_allclose(np.array(got.astype(mx.float32)),
                                   np.array(want.astype(mx.float32)),
                                   rtol=1e-5, atol=1e-5)


_ARCH = "gemma-embedding2"


def _meta(**over) -> dict:
    """The real file's header, at six layers."""
    a = _ARCH
    meta = {
        "general.architecture": a, f"{a}.block_count": 6,
        f"{a}.context_length": 262144, f"{a}.embedding_length": 512,
        f"{a}.feed_forward_length": 2048, f"{a}.attention.head_count": 4,
        f"{a}.attention.head_count_kv": [2, 2, 1, 2, 2, 1],
        f"{a}.rope.freq_base": 1e6, f"{a}.rope.freq_base_swa": 1e4,
        f"{a}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{a}.attention.key_length": 512, f"{a}.attention.value_length": 512,
        f"{a}.attention.key_length_swa": 256,
        f"{a}.attention.value_length_swa": 256,
        f"{a}.attention.sliding_window": 1024,
        f"{a}.attention.shared_kv_layers": 0,
        f"{a}.embedding_length_per_layer_input": 512,
        f"{a}.attention.sliding_window_pattern": [True, True, False] * 2,
        f"{a}.rope.dimension_count": 512, f"{a}.rope.dimension_count_swa": 256,
        f"{a}.embedding_length_out": 768, f"{a}.attention.causal": False,
        f"{a}.pooling_type": 1, "tokenizer.ggml.tokens": ["a"] * 64,
    }
    meta.update({f"{a}.{k}": v for k, v in over.items()})
    return meta


def test_gguf_header_synthesizes_the_transformers_config():
    from gmlx.load.config_synth import synthesize_config

    config = synthesize_config(_meta(), tensor_shapes={})
    assert config["model_type"] == "gemma_embedding2"
    assert config["pooling_type"] == 1
    text = eg2.ModelArgs.from_dict(config).text
    # The file stores the full window; transformers keeps the radius.
    assert text.sliding_window == 512
    assert text.layer_types == ["sliding_attention", "sliding_attention",
                                "full_attention"] * 2
    assert [text.layer(i) for i in range(6)] == [(256, 2), (256, 2), (512, 1)] * 2
    assert text.hidden_size_per_layer_input == 512
    assert text.embedding_dim == 768
    assert text.rope_parameters["sliding_attention"]["rope_theta"] == 1e4
    assert text.rope_parameters["full_attention"]["rope_theta"] == 1e6
    assert eg2.ModelArgs.from_dict(config).max_position_embeddings == 8192


@pytest.mark.parametrize("over, words", [
    ({"attention.causal": True}, "bidirectional"),
    ({"rope.dimension_count": 128}, "full-width rope"),
    ({"rope.dimension_count_swa": 64}, "full-width rope"),
    ({"attention.shared_kv_layers": 2}, "shared KV"),
    ({"attention.value_length_swa": 128}, "value_length_swa"),
])
def test_gguf_header_outside_the_port_is_refused(over, words):
    from gmlx.load.config_synth import synthesize_config

    with pytest.raises(ValueError, match=words):
        synthesize_config(_meta(**over), tensor_shapes={})


def test_every_gguf_tensor_maps_to_a_model_parameter():
    from gmlx.load.config_synth import synthesize_config
    from gmlx.load.remap import parse_gguf_name

    config = synthesize_config(_meta(), tensor_shapes={})
    config["text_config"].update(hidden_size=64, intermediate_size=96,
                                 vocab_size=64, embedding_dim=48,
                                 hidden_size_per_layer_input=16)
    model = eg2.Model(eg2.ModelArgs.from_dict(config))
    params = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
    names = ["token_embd.weight", "output_norm.weight", "output.weight",
             "per_layer_model_proj.weight", "per_layer_proj_norm.weight"]
    for i in range(6):
        names += [f"blk.{i}.{t}.weight" for t in (
            "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
            "attn_q_norm", "attn_k_norm", "post_attention_norm", "ffn_norm",
            "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm", "inp_gate",
            "proj", "post_norm")]
        names.append(f"blk.{i}.layer_output_scale.weight")
    decisions = [parse_gguf_name(_ARCH, n) for n in names]
    assert [d.kind for d in decisions] == ["map"] * len(names)
    assert {d.transform for d in decisions} == {"passthrough"}
    assert {d.hf_name for d in decisions} == params


_GPU = (mx.default_device() == mx.Device(mx.gpu)
        and not os.environ.get("KQUANT_FORCE_CPU"))

# The real attention shapes, which the hd512 wrapper keys on; other widths
# stay small.
_REAL_HEADS = dict(
    vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=256,
    global_head_dim=512, num_global_key_value_heads=1,
    layer_types=["sliding_attention", "full_attention"],
    hidden_size_per_layer_input=16, embedding_dim=48)


def _real_heads_model() -> eg2.Model:
    model = eg2.Model(eg2.ModelArgs.from_dict({"text_config": _REAL_HEADS}))
    rng = np.random.default_rng(3)
    weights = []
    for name, p in nn.utils.tree_flatten(model.parameters()):
        if p.ndim == 1:
            w = 1.0 + 0.3 * rng.standard_normal(p.shape)
        else:
            w = rng.standard_normal(p.shape) / p.shape[-1] ** 0.5
        weights.append((name, mx.array(w.astype(np.float32)).astype(mx.bfloat16)))
    model.load_weights(weights)
    mx.eval(model.parameters())
    return model


# Item length -> the route of the full layer's last query tile, which has
# length % 512 rows. Its 4 query heads share one KV head, so the wrapper's
# key ceiling is 16,384 at 1-2 rows, 8,192 at 3-4 and 4,096 at 5-6.
_LAST_TILE = {
    514: "sdpa_vector",       # 2 rows
    519: "stock",             # 7 rows: wider than the vector kernel
    600: "chunked_prefill",   # 88 rows
    1030: "sdpa_vector",      # 6 rows, keys under 4,096
    3073: "1-row",            # 1 row, keys at the NAX decode floor
    4614: "stock",            # 6 rows, keys past 4,096
    7684: "sdpa_vector",      # 4 rows, keys under 8,192
}


@pytest.mark.skipif(not _GPU, reason="the hd512 routes are Metal kernels")
@pytest.mark.parametrize("length", sorted(_LAST_TILE))
def test_hd512_wrapper_routes_match_stock_attention(length, monkeypatch):
    from gmlx.upstream import attn_hd512
    from gmlx.upstream.patching import under

    stock = under(mx.fast.scaled_dot_product_attention, attn_hd512._FLAG)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", stock)
    monkeypatch.setattr(attn_hd512, "_installed", False)
    monkeypatch.setattr(attn_hd512, "_orig_sdpa", attn_hd512._orig_sdpa)
    model = _real_heads_model()
    ids = mx.array(np.random.default_rng(length).integers(3, 64, length)[None])
    want = model(ids)
    mx.eval(want.text_embeds, want.last_hidden_state)

    assert attn_hd512.install_hd512_sdpa()
    before = attn_hd512.route_counts()
    got = model(ids)
    mx.eval(got.text_embeds, got.last_hidden_state)
    after = attn_hd512.route_counts()
    moved = {k: after[k] - before.get(k, 0) for k in after
             if after[k] > before.get(k, 0)}

    tail = length % 512
    route = _LAST_TILE[length]
    if route == "1-row":
        # On NAX GPUs one KV head reaches sdpa_decode_gqa from 3,072 keys.
        route = "gqa_decode" if attn_hd512._GQA_NAX512 else "sdpa_vector"
    assert {k[0] for k in moved if k[1:3] == (512, tail)} == {route}, moved
    assert {k[0] for k in moved if k[1:3] == (512, 512)} == {"chunked_prefill"}
    h_got = np.array(got.last_hidden_state.astype(mx.float32))
    h_want = np.array(want.last_hidden_state.astype(mx.float32))
    assert _rel_l2(h_got, h_want) < 1e-2
    assert float(np.array(got.text_embeds)[0] @ np.array(want.text_embeds)[0]) > 0.9999
