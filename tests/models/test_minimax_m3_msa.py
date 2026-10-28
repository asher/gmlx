#!/usr/bin/env python3
"""MiniMax Sparse Attention (MSA) for minimax-m3.

Numerical anchors on a tiny random-init model (llama.cpp PR #24908
semantics): below topk*block key blocks MSA is exactly dense attention;
beyond it the decode gather path and the prefill masked path must agree with
each other (same selection, different execution), chunked prefill must match
one-shot prefill (indexer cache append correctness), and the sparse path must
actually diverge from dense. Plus the plumbing: remap spellings, synth
arming, sanitize gating, cache construction, and the mm:think tag fixes.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.minimax_m3 import MSAKVCache, ModelArgs, Model
from gmlx.load.remap import parse_gguf_name, RemapDecision
from gmlx._vendor.mlx_lm.models.cache import KVCache

MAP = RemapDecision.KIND_MAP


def _args(**over) -> ModelArgs:
    base = dict(
        model_type="minimax_m3",
        hidden_size=64,
        intermediate_size=32,          # expert FFN width
        dense_intermediate_size=48,
        shared_intermediate_size=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=3,
        num_local_experts=4,
        num_experts_per_tok=2,
        rms_norm_eps=1e-6,
        rope_theta=5e6,
        rotary_dim=4,                  # partial (< head_dim 8)
        vocab_size=100,
        head_dim=8,
        mlp_layer_types=["dense", "sparse", "sparse"],
        use_sparse_attention=True,
        sparse_index_dim=8,
        sparse_num_index_heads=2,      # == num_key_value_heads
        sparse_topk_blocks=2,
        sparse_block_size=8,
        sparse_local_block=1,
    )
    base.update(over)
    return ModelArgs(**base)


def _model(seed=0, **over) -> Model:
    mx.random.seed(seed)
    return Model(_args(**over))


def _dense_twin(sparse: Model, seed: int) -> Model:
    """A use_sparse_attention=False model sharing ``sparse``'s trunk weights
    (init order differs between the two, so the trunk must be copied)."""
    mx.random.seed(seed)
    dense = Model(_args(use_sparse_attention=False))
    flat = [(k, v) for k, v in tree_flatten(sparse.parameters())
            if "index_" not in k]
    dense.update(tree_unflatten(flat))
    return dense


def _logits(model, tokens, chunks=None):
    """Run ``tokens`` [1, L] through ``model`` with a fresh cache; return the
    full-sequence logits. ``chunks`` optionally splits the prompt to exercise
    chunked prefill against the same cache."""
    cache = model.make_cache()
    if chunks is None:
        return model(tokens, cache=cache), cache
    outs = []
    at = 0
    for n in chunks:
        outs.append(model(tokens[:, at : at + n], cache=cache))
        at += n
    return mx.concatenate(outs, axis=1), cache


def _tokens(n, seed=1):
    mx.random.seed(seed)
    return mx.random.randint(0, 100, (1, n))


# -- cache construction ------------------------------------------------------

def test_make_cache_types_follow_layer_sparsity():
    m = _model()
    caches = m.make_cache()
    assert type(caches[0]) is KVCache          # dense-lead layer: no indexer
    assert type(caches[1]) is MSAKVCache
    assert type(caches[2]) is MSAKVCache


def test_make_cache_all_plain_when_msa_off():
    m = _model(use_sparse_attention=False)
    assert all(type(c) is KVCache for c in m.make_cache())


def test_msa_cache_refuses_quantization():
    c = MSAKVCache()
    assert c.kv_quant_unsupported
    try:
        c.to_quantized()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("MSAKVCache.to_quantized must refuse")


# -- M3.1 layout: MSA on every layer that carries the indexer -----------------

_M31 = dict(msa_layers=[0, 1, 2])   # indexer on the dense-lead layer too


def test_m31_dense_lead_layer_gets_msa():
    m = _model(**_M31)
    assert all(type(c) is MSAKVCache for c in m.make_cache())
    names = {k for k, _ in tree_flatten(m.parameters())}
    assert "model.layers.0.self_attn.index_q_proj.weight" in names
    assert "model.layers.0.self_attn.index_k_norm.weight" in names


def test_m3_layout_keeps_dense_lead_layer_dense():
    # msa_layers from an M3 GGUF (MoE layers only) and None (older configs)
    # both leave the dense-lead layer on a plain cache.
    for over in ({"msa_layers": [1, 2]}, {}):
        caches = _model(**over).make_cache()
        assert type(caches[0]) is KVCache
        assert type(caches[1]) is MSAKVCache


def test_sanitize_drops_indexer_outside_msa_layers():
    w = {
        "model.layers.0.self_attn.index_q_proj.weight": mx.zeros((16, 64)),
        "model.layers.1.self_attn.index_q_proj.weight": mx.zeros((16, 64)),
        "language_model.model.layers.0.self_attn.index_k_proj.weight":
            mx.zeros((8, 64)),
    }
    m3 = _model(msa_layers=[1, 2]).sanitize(dict(w))
    assert set(m3) == {"model.layers.1.self_attn.index_q_proj.weight"}
    m31 = _model(**_M31).sanitize(dict(w))
    assert set(m31) == {
        "model.layers.0.self_attn.index_q_proj.weight",
        "model.layers.1.self_attn.index_q_proj.weight",
        "model.layers.0.self_attn.index_k_proj.weight",
    }


def test_m31_dense_lead_layer_runs_sparse_attention():
    # Same weights, with and without the layer-0 indexer armed. Below
    # topk*block keys both are dense-exact; past it layer 0's selection
    # must change the output.
    toks = _tokens(64)
    m31 = _model(seed=3, **_M31)
    mx.random.seed(3)
    m3 = Model(_args(msa_layers=[1, 2]))
    flat = [(k, v) for k, v in tree_flatten(m31.parameters())
            if not k.startswith("model.layers.0.self_attn.index_")]
    m3.update(tree_unflatten(flat))
    l31, _ = _logits(m31, toks)
    l3, _ = _logits(m3, toks)
    assert mx.abs(l31[:, :16] - l3[:, :16]).max().item() < 1e-5
    assert mx.abs(l31[:, 16:] - l3[:, 16:]).max().item() > 1e-2


def test_m31_decode_gather_matches_prefill_mask_path():
    m = _model(seed=5, **_M31)
    toks = _tokens(65)
    full, _ = _logits(m, toks)
    _, cache = _logits(m, toks[:, :64])
    step = m(toks[:, 64:65], cache=cache)
    assert mx.allclose(step[:, 0], full[:, 64], atol=1e-2, rtol=1e-2).item()


# -- numerics ----------------------------------------------------------------

def test_short_context_msa_equals_dense():
    # topk*block = 16 keys: at L=16 every block is selectable, so MSA must
    # reduce to exact dense attention. Observed bit-exact (the module
    # short-circuits to the same SDPA); the indexer weights must not leak
    # into the output in this regime.
    toks = _tokens(16)
    sparse = _model(seed=3)
    dense = _dense_twin(sparse, seed=3)
    ls, _ = _logits(sparse, toks)
    ld, _ = _logits(dense, toks)
    assert mx.allclose(ls, ld, atol=1e-6).item()


def test_long_context_msa_diverges_from_dense_after_selection_engages():
    # 9 blocks > topk 2. Queries at pos < topk*block still see every visible
    # block (selection keeps all) -> dense-equal; later queries discard
    # blocks -> the outputs must clearly diverge.
    toks = _tokens(64)
    sparse = _model(seed=3)
    dense = _dense_twin(sparse, seed=3)
    ls, _ = _logits(sparse, toks)
    ld, _ = _logits(dense, toks)
    early = mx.abs(ls[:, :16] - ld[:, :16]).max().item()
    late = mx.abs(ls[:, 16:] - ld[:, 16:]).max().item()
    assert early < 1e-5
    assert late > 1e-2


def test_decode_gather_matches_prefill_mask_path():
    # The same selection runs through two executions: prefill builds a -inf
    # block mask over the full cache, decode gathers the selected rows.
    # Verified (selection spy) that both paths pick identical blocks; the
    # residual is kernel reduction-order fp between masked-full SDPA and the
    # gathered form (~2e-3 on these logits), hence the tolerance.
    m = _model(seed=5)
    toks = _tokens(65)
    full, _ = _logits(m, toks)                      # 65-token prefill
    pre, cache = _logits(m, toks[:, :64])           # 64 + 1 decode
    step = m(toks[:, 64:65], cache=cache)
    assert mx.allclose(step[:, 0], full[:, 64], atol=1e-2, rtol=1e-2).item()


def test_chunked_prefill_matches_single_shot():
    # Indexer keys append through the same offset bookkeeping as K/V; a
    # split prefill reproduces one-shot logits up to matmul-tiling fp in the
    # scores (near-tie block ranks can flip under it; ~2e-3 observed here).
    m = _model(seed=7)
    toks = _tokens(60)
    one, _ = _logits(m, toks)
    two, _ = _logits(m, toks, chunks=[23, 37])
    assert mx.allclose(one, two, atol=1e-2, rtol=1e-2).item()


def test_local_block_always_attended():
    # Adversarial indexer: index projections zeroed -> all block scores tie.
    # The local-force bias must still guarantee each query attends its own
    # block; the output for the last token must depend on the last block's
    # values (perturbing a value inside the local block changes the output).
    m = _model(seed=9)
    zeros = {
        k: mx.zeros_like(v)
        for k, v in m.parameters().items()
        if "index_q_proj" in k or "index_k_proj" in k
    }
    m.update(zeros)
    toks = _tokens(64)
    base, _ = _logits(m, toks)

    # Perturb the token right before the last one (same local block as the
    # final query at pos 63: block 7 covers 56..63).
    toks2 = mx.array(toks)
    toks2[0, 62] = (toks[0, 62] + 1) % 100
    pert, _ = _logits(m, toks2)
    assert not mx.allclose(base[:, 63], pert[:, 63], atol=1e-6).item()


def test_dense_fallback_on_plain_cache_warns_not_crashes():
    # A non-MSA cache (batch merge, snapshot restore of an old cache) must
    # fall back to dense, not crash.
    m = _model(seed=11)
    caches = [KVCache() for _ in range(3)]
    out = m(_tokens(20), cache=caches)
    assert out.shape == (1, 20, 100)


# -- fp16 indexer keys, causal kernel, int32 tile cap -------------------------

def test_indexer_key_cache_is_fp16():
    m = _model(seed=5)
    _, cache = _logits(m, _tokens(64))
    assert cache[1].ik.dtype == mx.float16


def test_decode_scores_do_not_promote_the_key_cache(monkeypatch):
    # A mixed f32 x fp16 score matmul copies the whole key cache to f32
    # every decode step.
    monkeypatch.delenv("GMLX_MSA_F32_SCORES", raising=False)
    m = _model(seed=5)
    _, cache = _logits(m, _tokens(64))
    calls = []
    real = mx.matmul

    def spy(a, b, *rest, **kw):
        calls.append((a.dtype, b.dtype, b.ndim))
        return real(a, b, *rest, **kw)

    monkeypatch.setattr(mx, "matmul", spy)
    m(_tokens(1, seed=2), cache=cache)
    scores = [c for c in calls if c[2] == 4]
    assert scores
    assert all(a == b == mx.float16 for a, b, _ in scores), scores


@pytest.mark.parametrize("L", [1, 24])
def test_f32_scores_lever_matches_llama_form(monkeypatch, L):
    # Default: fp16 x fp16 scores (no cast of the key cache). With
    # GMLX_MSA_F32_SCORES=1 the scores are f32 values of the f32 query
    # against the fp16 keys, as llama.cpp computes them.
    monkeypatch.delenv("GMLX_MSA_F32_SCORES", raising=False)
    attn = _model().layers[1].self_attn
    mx.random.seed(4)
    iq = 8 * mx.random.normal((1, 2, L, 8))
    IK = (8 * mx.random.normal((1, 1, 200, 8))).astype(mx.float16)
    ref = (np.asarray(iq, dtype=np.float64)
           @ np.asarray(IK.astype(mx.float32), dtype=np.float64).swapaxes(-1, -2))
    IKt32 = IK.swapaxes(-1, -2).astype(mx.float32) if L > 1 else None
    assert attn._scores(iq, IK, IKt32).dtype == mx.float16
    monkeypatch.setenv("GMLX_MSA_F32_SCORES", "1")
    got = attn._scores(iq, IK, IKt32)
    assert got.dtype == mx.float32
    # f32 GEMMs run at TF32 on M5 GPUs (about 4e-4 relative here).
    err = np.abs(np.asarray(got, dtype=np.float64) - ref).max()
    assert err / np.abs(ref).max() < 2e-3


def test_short_context_causal_kernel_matches_explicit_mask():
    attn = _model().layers[1].self_attn
    mx.random.seed(2)
    offset, L = 5, 7
    S = offset + L
    q = mx.random.normal((1, 8, L, 8))
    K = mx.random.normal((1, 2, S, 8))
    V = mx.random.normal((1, 2, S, 8))
    ref = mx.fast.scaled_dot_product_attention(
        q, K, V, scale=attn.scale, mask=attn._causal(offset, L, S)[None, None])
    got = attn._dense_sdpa(q, K, V, offset, L, S)
    assert mx.allclose(got, ref, atol=1e-5).item()


def test_prefill_tile_capped_under_element_limit(monkeypatch):
    from gmlx.models.minimax_m3 import MiniMaxM3Attention

    m = _model(seed=7)
    toks = _tokens(60)
    ref, _ = _logits(m, toks)
    calls = []
    real = MiniMaxM3Attention._causal

    def counting(offset, L, S):
        calls.append(L)
        return real(offset, L, S)

    monkeypatch.setattr(MiniMaxM3Attention, "_causal", staticmethod(counting))
    monkeypatch.setenv("GMLX_MSA_MAX_ELEMS", str(64 * 16))  # 16-row tiles
    got, _ = _logits(m, toks)
    assert calls and max(calls) == 16
    assert mx.allclose(got, ref, atol=1e-5).item()


def test_gathered_prefill_matches_dense_masked_prefill(monkeypatch):
    # Past GMLX_MSA_GATHER_MIN, prefill attends over each query's gathered
    # rows instead of the masked full cache. Same selection, so the logits
    # agree up to SDPA reduction order. 60 tokens leave a partial tail block;
    # 5-query chunks leave a ragged last chunk.
    from gmlx.models.minimax_m3 import MiniMaxM3Attention

    m = _model(seed=7)
    toks = _tokens(60)
    ref_one, _ = _logits(m, toks)
    ref_two, _ = _logits(m, toks, chunks=[23, 37])
    calls = []
    real = MiniMaxM3Attention._msa_prefill_gather

    def spy(self, *a):
        calls.append(a[-2])
        return real(self, *a)

    monkeypatch.setattr(MiniMaxM3Attention, "_msa_prefill_gather", spy)
    monkeypatch.setenv("GMLX_MSA_GATHER_MIN", "0")
    monkeypatch.setenv("GMLX_MSA_GATHER_CHUNK", "5")
    one, _ = _logits(m, toks)
    two, _ = _logits(m, toks, chunks=[23, 37])
    assert 60 in calls and 23 in calls
    assert mx.allclose(one, ref_one, atol=1e-3, rtol=1e-3).item()
    assert mx.allclose(two, ref_two, atol=1e-3, rtol=1e-3).item()


def test_gathered_rows_capped_under_element_limit(monkeypatch):
    # The gathered K rows per chunk (C * Hkv * topk*blk * D) stay under the
    # element limit even when GMLX_MSA_GATHER_CHUNK asks for more.
    m = _model(seed=7)
    toks = _tokens(60)
    ref, _ = _logits(m, toks)
    widths = []
    real = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        if q.shape[1] == 1:                  # the gather path's batch layout
            widths.append(q.shape[0] // 2)   # queries per chunk (Hkv 2)
        return real(q, k, v, **kw)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    monkeypatch.setenv("GMLX_MSA_GATHER_MIN", "0")
    monkeypatch.setenv("GMLX_MSA_GATHER_CHUNK", "256")
    # Rows per query 2 * 16 * 8 = 256, so 1280 elements allow 5 queries.
    monkeypatch.setenv("GMLX_MSA_MAX_ELEMS", "1280")
    got, _ = _logits(m, toks)
    assert widths and max(widths) == 5
    assert mx.allclose(got, ref, atol=1e-3, rtol=1e-3).item()


def test_input_embeddings_replace_token_lookup():
    # The VLM container passes merged image embeddings; MSA prefill must run
    # on them and match the token path when the embeddings are the lookup.
    m = _model(seed=7, **_M31)
    toks = _tokens(60)
    ref, _ = _logits(m, toks)
    emb = m.model.embed_tokens(toks)
    got = m(toks, cache=m.make_cache(), input_embeddings=emb)
    assert mx.allclose(got, ref, atol=1e-5).item()
    other = m(toks, cache=m.make_cache(), input_embeddings=emb * 0.5)
    assert not mx.allclose(other, ref, atol=1e-3).item()


# -- sanitize / config -------------------------------------------------------

def test_sanitize_keeps_indexer_only_when_armed():
    on = _model()
    off = _model(use_sparse_attention=False)
    w = {
        "model.layers.1.self_attn.index_q_proj.weight": mx.zeros((16, 64)),
        "model.layers.1.self_attn.q_proj.weight": mx.zeros((64, 64)),
    }
    kept = on.sanitize(dict(w))
    assert "model.layers.1.self_attn.index_q_proj.weight" in kept
    dropped = off.sanitize(dict(w))
    assert "model.layers.1.self_attn.index_q_proj.weight" not in dropped
    assert "model.layers.1.self_attn.q_proj.weight" in dropped


def test_sanitize_narrows_f32_indexer_projections_to_bf16():
    # Some MSA GGUFs ship the indexer projections as F32; the source
    # checkpoint is BF16, so the narrow is a bit-exact recovery. Norms and
    # trunk weights keep their stored dtype.
    m = _model()
    w = {
        "model.layers.1.self_attn.index_q_proj.weight": mx.zeros((16, 64)),
        "model.layers.1.self_attn.index_q_norm.weight": mx.zeros((8,)),
        "model.layers.1.self_attn.q_proj.weight": mx.zeros((64, 64)),
        "model.layers.1.self_attn.index_k_proj.weight": mx.zeros(
            (16, 64), dtype=mx.bfloat16
        ),
    }
    out = m.sanitize(dict(w))
    assert out["model.layers.1.self_attn.index_q_proj.weight"].dtype == mx.bfloat16
    assert out["model.layers.1.self_attn.index_q_norm.weight"].dtype == mx.float32
    assert out["model.layers.1.self_attn.q_proj.weight"].dtype == mx.float32
    assert out["model.layers.1.self_attn.index_k_proj.weight"].dtype == mx.bfloat16


def test_msa_disable_env_drops_indexer_weights(monkeypatch):
    # GMLX_MSA_DISABLE skips building the indexer submodules, so sanitize
    # must drop the indexer weights or a strict load would fail.
    monkeypatch.setenv("GMLX_MSA_DISABLE", "1")
    m = _model()
    assert not any(getattr(layer.self_attn, "msa", False)
                   for layer in m.layers)
    w = {"model.layers.1.self_attn.index_q_proj.weight": mx.zeros((16, 64))}
    assert m.sanitize(dict(w)) == {}
    assert all(type(c) is KVCache for c in m.make_cache())


def test_quant_predicate_protects_indexer():
    pred = _model().quant_predicate
    assert pred("model.layers.1.self_attn.index_q_proj", None) is False
    assert pred("model.layers.1.self_attn.q_proj", None) is True


# -- sidecar resolution ------------------------------------------------------

def test_sidecar_resolver_discovery_and_gates(tmp_path, monkeypatch):
    from gmlx.load.loader import _resolve_indexer_sidecar

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    sidecar = tmp_path / "m3-indexer.gguf"
    sidecar.write_bytes(b"")
    # directory glob finds the sidecar for an indexless minimax-m3
    assert _resolve_indexer_sidecar(str(gguf), "minimax-m3", {}) == str(sidecar)
    # other archs and native MSA GGUFs never merge
    assert _resolve_indexer_sidecar(str(gguf), "llama", {}) is None
    native = {"blk.3.indexer.q_proj.weight": (128, 6144)}
    assert _resolve_indexer_sidecar(str(gguf), "minimax-m3", native) is None
    # GMLX_MSA_DISABLE skips the sidecar entirely (sanitize would drop it)
    monkeypatch.setenv("GMLX_MSA_DISABLE", "1")
    assert _resolve_indexer_sidecar(str(gguf), "minimax-m3", {}) is None


# -- remap -------------------------------------------------------------------

def test_remap_indexer_both_spellings():
    for src, tgt in (
        ("blk.3.indexer.q_proj.weight", "index_q_proj"),
        ("blk.3.indexer.k_proj.weight", "index_k_proj"),
        ("blk.3.index_q_proj.weight", "index_q_proj"),
        ("blk.3.index_k_proj.weight", "index_k_proj"),
    ):
        r = parse_gguf_name("minimax-m3", src)
        assert r.kind == MAP and r.transform == "passthrough"
        assert r.hf_name == f"model.layers.3.self_attn.{tgt}.weight"


def test_remap_indexer_norms_unbaked():
    # Sidecars/native MSA GGUFs bake gemma +1 into the indexer norms like
    # every other minimax-m3 norm; the unbake transform must fire on them.
    for src in (
        "blk.3.indexer.q_norm.weight",
        "blk.3.indexer.k_norm.weight",
        "blk.3.index_q_norm.weight",
        "blk.3.index_k_norm.weight",
    ):
        r = parse_gguf_name("minimax-m3", src)
        assert r.kind == MAP and r.transform == "gemma_norm_minus_one"


# -- config synth ------------------------------------------------------------

def _m3_meta_and_shapes(with_indexer_kvs: bool, with_tensor: bool):
    arch = "minimax-m3"
    meta = {
        "general.architecture": arch,
        f"{arch}.block_count": 3,
        f"{arch}.context_length": 4096,
        f"{arch}.embedding_length": 64,
        f"{arch}.feed_forward_length": 48,
        f"{arch}.attention.head_count": 8,
        f"{arch}.attention.head_count_kv": 2,
        f"{arch}.attention.key_length": 8,
        f"{arch}.attention.value_length": 8,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{arch}.rope.freq_base": 5e6,
        f"{arch}.rope.dimension_count": 4,
        f"{arch}.leading_dense_block_count": 1,
        f"{arch}.expert_count": 4,
        f"{arch}.expert_used_count": 2,
        f"{arch}.expert_feed_forward_length": 32,
        f"{arch}.expert_shared_count": 1,
        f"{arch}.expert_gating_func": 2,
        f"{arch}.expert_weights_scale": 2.0,
        "tokenizer.ggml.tokens": ["a"] * 100,
    }
    if with_indexer_kvs:
        meta[f"{arch}.attention.indexer.head_count"] = 2
        meta[f"{arch}.attention.indexer.key_length"] = 8
        meta[f"{arch}.attention.indexer.top_k"] = 2
        meta[f"{arch}.attention.indexer.block_size"] = 8
        meta[f"{arch}.attention.indexer.local_blocks"] = 1
    shapes = {
        "output.weight": [64, 100],
        "blk.0.attn_q_norm.weight": [8],
        "blk.1.ffn_gate_shexp.weight": [64, 24],
    }
    if with_tensor:
        shapes["blk.1.indexer.q_proj.weight"] = [64, 16]
    return meta, shapes


def test_synth_arms_msa_from_indexer_tensor_and_kvs():
    from gmlx.load.config_synth import synthesize_config

    meta, shapes = _m3_meta_and_shapes(True, True)
    cfg = synthesize_config(meta, shapes)
    assert cfg["use_sparse_attention"] is True
    assert cfg["sparse_num_index_heads"] == 2
    assert cfg["sparse_index_dim"] == 8
    assert cfg["sparse_topk_blocks"] == 2
    assert cfg["sparse_block_size"] == 8
    assert cfg["sparse_local_block"] == 1


def test_synth_defaults_when_tensors_present_without_kvs():
    from gmlx.load.config_synth import synthesize_config

    meta, shapes = _m3_meta_and_shapes(False, True)
    cfg = synthesize_config(meta, shapes)
    assert cfg["use_sparse_attention"] is True
    assert cfg["sparse_topk_blocks"] == 16
    assert cfg["sparse_block_size"] == 128


def test_synth_msa_layers_from_indexer_tensor_presence():
    from gmlx.load.config_synth import synthesize_config

    meta, shapes = _m3_meta_and_shapes(True, True)
    assert synthesize_config(meta, shapes)["msa_layers"] == [1]
    for i in (0, 2):  # M3.1: the dense-lead layer carries one too
        shapes[f"blk.{i}.indexer.q_proj.weight"] = [64, 16]
    assert synthesize_config(meta, shapes)["msa_layers"] == [0, 1, 2]


def test_synth_dense_when_no_indexer_tensors():
    from gmlx.load.config_synth import synthesize_config

    meta, shapes = _m3_meta_and_shapes(False, False)
    cfg = synthesize_config(meta, shapes)
    assert cfg["use_sparse_attention"] is False


# -- mm:think tags -----------------------------------------------------------

class _FakeTok:
    """Tokenizer double: both legacy and mm think tags are single vocab
    entries (like the real MiniMax-M3 vocab), template uses mm only."""

    chat_template = "{{ bos }}<mm:think>{{ '</mm:think>' if x }}"
    _vocab = {
        "<think>": 10, "</think>": 11,
        "<mm:think>": 12, "</mm:think>": 13, "\n": 14,
    }

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        return [1, 2]

    def decode(self, ids):
        rev = {v: k for k, v in self._vocab.items()}
        return "".join(rev.get(i, "?") for i in ids)


def test_thinking_seqs_prefer_template_spelling():
    from gmlx.gen.thinking_budget import _thinking_token_seqs

    start, end = _thinking_token_seqs(_FakeTok())
    assert end == (13,)   # </mm:think>, not </think>
    assert start == (12,)


def test_thinking_budget_forces_mm_close():
    from gmlx.gen.thinking_budget import make_thinking_budget_processor

    p = make_thinking_budget_processor(_FakeTok(), 0)
    assert p is not None
    assert p.forced_ids[-1] == 13


def test_prompt_opens_thinking_mm():
    from gmlx.gen.thinking_budget import prompt_opens_thinking

    assert prompt_opens_thinking("...<mm:think>reasoning", tokenizer=_FakeTok())
    assert not prompt_opens_thinking(
        "...<mm:think>x</mm:think>done", tokenizer=_FakeTok()
    )


def test_reasoning_filter_mm_markers():
    from gmlx.tui.reasoning import ReasoningFilter

    f = ReasoningFilter()
    spans = f.feed("<mm:think>plan</mm:think>Hello") + f.flush()
    assert ("plan", "reason") in spans
    assert ("Hello", "answer") in spans
