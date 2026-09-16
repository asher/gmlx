"""DeepSeek-V4.1-Flash model class.

The oracles are the reference ``inference/model.py`` and
``inference/engram.py``: the n-gram hash is checked against a numpy port of
``NgramHashState.forward``, and the cached paths against the same weights
run in one pass. CPU-only.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.models.deepseek_v41.model import (
    EngramHash,
    Model,
    ModelArgs,
    SharedStreams,
    _latent_qat,
)

_TABLE_ROWS = 3 * 2 * 7


def _args(**over) -> ModelArgs:
    base = dict(
        vocab_size=64, hidden_size=32, moe_intermediate_size=16,
        num_hidden_layers=6, num_attention_heads=4, head_dim=16,
        q_lora_rank=8, qk_rope_head_dim=4, o_groups=2, o_lora_rank=4,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        index_n_heads=2, index_head_dim=8, index_topk=4,
        sliding_window=4, hc_mult=4, rms_norm_eps=1e-6,
        compress_ratios=[0, 0, 2, 2, 1, 1],
        kv_source_layers=[2, 4], index_key_layers=[2, 4],
        index_source_layers=[2, 4, 5],
        engram_layer_ids=[1, 3],
        engram_table_rows=[_TABLE_ROWS, _TABLE_ROWS],
        engram_max_ngram_size=4, engram_n_heads=2, engram_head_dim=8,
        engram_pad_id=2,
        engram_multipliers=[3, 5, 7, 11, 13, 17, 19, 23],
        engram_primes=[7] * 12,
        engram_offsets=[0, 7, 14, 21, 28, 35] * 2,
        engram_token_map=[(i * 7) % 11 for i in range(64)],
        rope_scaling={"type": "yarn", "factor": 4.0,
                      "original_max_position_embeddings": 64,
                      "beta_fast": 32, "beta_slow": 1},
    )
    base.update(over)
    return ModelArgs(**base)


def _randomized(model):
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if node.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.normal(node.shape) * 0.2
        return node

    model.update(walk(model.parameters()))
    mx.eval(model.parameters())
    return model


# --- engram hash -----------------------------------------------------------


def _reference_rows(chunks, args, masks=None):
    """Port of the reference ``NgramHashState.forward``, cache and all."""
    dead = -1
    n_tab = len(args.engram_layer_ids)
    n_gram = args.engram_max_ngram_size
    heads = args.engram_n_heads
    token_map = np.array(args.engram_token_map, dtype=np.int64)
    mult = np.array(args.engram_multipliers, dtype=np.int64).reshape(n_tab, n_gram)
    primes = np.array(args.engram_primes, dtype=np.int64).reshape(
        n_tab, n_gram - 1, heads)
    offsets = np.array(args.engram_offsets, dtype=np.int64).reshape(
        n_tab, (n_gram - 1) * heads)
    pad_id = int(token_map[args.engram_pad_id])

    total = sum(len(c) for c in chunks)
    cache = np.zeros((1, total), dtype=np.int64)
    out, start = [], 0
    for chunk_idx, chunk in enumerate(chunks):
        length = len(chunk)
        compressed = token_map[np.array(chunk, dtype=np.int64)][None]
        if masks is not None:
            compressed = np.where(np.array(masks[chunk_idx])[None], compressed, dead)
        cache[:, start:start + length] = compressed
        positions = np.arange(start, start + length)[None]
        blocked = np.zeros_like(positions, dtype=bool)
        tokens = []
        for shift in range(n_gram):
            src = np.take_along_axis(
                cache, np.clip(positions - shift, 0, None), axis=1)
            blocked = blocked | (positions < shift) | (src == dead)
            tokens.append(np.where(blocked, pad_id, src))
        tokens = np.stack(tokens, axis=-1)
        products = tokens[:, :, None, :] * mult
        rolling, hashes = products[..., 0], []
        for i in range(1, n_gram):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % primes[:, i - 1])
        out.append(np.concatenate(hashes, axis=-1) + offsets)
        start += length
    return np.concatenate(out, axis=1)


@pytest.mark.parametrize("chunks", [
    [[5, 7, 9, 11, 13, 15, 17]],
    [[5, 7, 9], [11, 13], [15], [17]],
    [[5], [7], [9], [11], [13], [15], [17]],
])
def test_engram_hash_matches_the_reference_port(chunks):
    args = _args()
    hasher = EngramHash(args)
    history, got = None, []
    for chunk in chunks:
        rows, history = hasher(mx.array([chunk]), history)
        mx.eval(rows, history)
        got.append(np.array(rows))
    got = np.concatenate(got, axis=1)
    assert np.array_equal(got, _reference_rows(chunks, args))


def test_engram_hash_blocks_lookback_across_a_dead_token():
    args = _args()
    hasher = EngramHash(args)
    chunk = [5, 7, 9, 11, 13]
    mask = [True, True, False, True, True]
    rows, _ = hasher(mx.array([chunk]), None, mx.array([mask]))
    mx.eval(rows)
    assert np.array_equal(np.array(rows), _reference_rows([chunk], args, [mask]))


def test_engram_row_ids_stay_inside_their_table():
    args = _args()
    hasher = EngramHash(args)
    rows, _ = hasher(mx.array([list(range(64))]), None)
    mx.eval(rows)
    rows = np.array(rows)
    assert rows.min() >= 0
    assert rows.max() < _TABLE_ROWS


# --- forward ---------------------------------------------------------------


def test_prefill_and_decode_agree():
    mx.random.seed(0)
    model = _randomized(Model(_args()))
    toks = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

    full = model(mx.array([toks]))
    mx.eval(full)

    cache = model.make_cache()
    for tok in toks:
        step = model(mx.array([[tok]]), cache=cache)
        mx.eval(step)
    assert mx.allclose(full[:, -1], step[:, -1], atol=2e-5)


def test_chunked_prefill_matches_one_pass():
    mx.random.seed(0)
    model = _randomized(Model(_args()))
    toks = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

    full = model(mx.array([toks]))
    cache = model.make_cache()
    model(mx.array([toks[:8]]), cache=cache)
    tail = model(mx.array([toks[8:]]), cache=cache)
    mx.eval(full, tail)
    assert mx.allclose(full[:, -1], tail[:, -1], atol=2e-5)


def test_sparse_regime_runs_when_the_pool_outgrows_index_topk():
    """Layer 4 pools one row per token, so 14 tokens put it past
    index_topk=4 and into the gathered path."""
    args = _args()
    model = _randomized(Model(args))
    cache = model.make_cache()
    out = model(mx.array([[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]]),
                cache=cache)
    mx.eval(out)
    pool = list(cache[4])[1]
    assert pool.size() > args.index_topk


def test_sparse_decode_compiled_route_matches_eager(monkeypatch):
    """A decode step on a full window and an outgrown pool takes the
    compiled gathered-attention core; its logits match the eager chain."""
    from gmlx.models.deepseek_v4 import model as v4

    args = _args()
    model = _randomized(Model(args))
    prompt = mx.array([[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]])
    tok = mx.array([[33]])
    calls = []
    real = v4._sparse_gathered_attention_c

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    outs = []
    for compiled in (True, False):
        monkeypatch.setattr(v4, "_COMPILE_SPARSE", compiled)
        monkeypatch.setattr(v4, "_sparse_gathered_attention_c", spy)
        cache = model.make_cache()
        mx.eval(model(prompt, cache=cache))
        n = len(calls)
        out = model(tok, cache=cache)
        mx.eval(out)
        outs.append(out)
        assert (len(calls) - n > 0) == compiled
    assert mx.allclose(outs[0], outs[1], atol=1e-4, rtol=1e-4)


def test_only_kv_source_layers_own_pools():
    args = _args()
    model = Model(args)
    for idx, entry in enumerate(model.make_cache()):
        members = list(entry) if hasattr(entry, "__getitem__") else [entry]
        pools = [m for m in members if hasattr(m, "accumulate_windows")]
        assert len(pools) == (2 if idx in args.kv_source_layers else 0)


def test_engram_history_rides_the_first_engram_layer():
    args = _args()
    model = Model(args)
    cache = model.make_cache()
    slot = list(cache[args.engram_layer_ids[0]])[-1]
    assert hasattr(slot, "cache")
    model(mx.array([[5, 7, 9]]), cache=cache)
    assert slot[0] is not None
    assert slot[0].shape == (1, args.engram_max_ngram_size - 1)


# --- QAT -------------------------------------------------------------------


def test_latent_qat_uses_groups_of_16_with_an_e4m3_scale():
    """The reference calls fp4_act_quant(latent, 16, scale_dtype=e4m3), so
    the scale is an FP8 value, not a power of two."""
    x = mx.random.normal((2, 3, 32)) * 4.0
    out = _latent_qat(x)
    mx.eval(out)
    groups = np.array(out).reshape(-1, 16)
    src = np.array(x).reshape(-1, 16)
    for row, ref in zip(groups, src):
        scale = max(np.abs(ref).max(), 6.0 * 2.0**-9) / 6.0
        # every value is a multiple of the group scale's fp4 grid
        assert np.all(np.abs(row) <= 6.0 * scale * 1.5 + 1e-6)
    assert not np.array_equal(groups, src)


def test_qat_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("GMLX_DS41_QAT", "0")
    x = mx.random.normal((2, 3, 32))
    assert mx.array_equal(_latent_qat(x), x)


# --- weights ---------------------------------------------------------------


def test_sanitize_reshapes_wo_a_and_drops_drafter_tensors():
    args = _args()
    model = Model(args)
    heads, groups = args.num_attention_heads, args.o_groups
    flat = mx.zeros((groups * args.o_lora_rank, heads * args.head_dim // groups))
    cleaned = model.sanitize({
        "model.layers.0.attn.wo_a.weight": flat,
        "mtp.0.attn.wq_a.weight": mx.zeros((4, 4)),
        f"model.layers.{args.num_hidden_layers}.attn.wq_a.weight": mx.zeros((4, 4)),
    })
    assert "mtp.0.attn.wq_a.weight" not in cleaned
    assert f"model.layers.{args.num_hidden_layers}.attn.wq_a.weight" not in cleaned
    assert cleaned["model.layers.0.attn.wo_a.weight"].shape == (
        groups, args.o_lora_rank, heads * args.head_dim // groups)


def test_cast_predicate_matches_the_loader_fp32_row():
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE

    predicate = Model(_args()).cast_predicate
    for key in _FP32_KEEP_BY_MODEL_TYPE["deepseek_v41"]:
        assert not predicate(f"model.layers.0{key}weight")
    assert predicate("model.layers.0.attn.wq_a.weight")


# --- serve-side pricing ----------------------------------------------------


def test_engram_layers_price_as_window_capped():
    """A layer whose cache list is a rotating window plus the engram token
    history is still window-capped; only pool and plain KV members grow."""
    from gmlx.serve.mem_preflight import stack_geometry

    args = _args()
    model = Model(args)
    geo = stack_geometry(model, model.make_cache())
    for idx in args.engram_layer_ids:
        assert geo[idx].window == args.sliding_window
    for idx in args.kv_source_layers:
        assert geo[idx].window is None  # the pooled rows grow with context


def test_prompt_cache_restore_rejects_a_changed_layout():
    from gmlx.cache.prefix_cache import _CACHELIST_TAG, _restore_entry

    args = _args()
    model = Model(args)
    entry = model.make_cache()[args.kv_source_layers[0]]
    with pytest.raises(ValueError, match="layout changed"):
        _restore_entry(entry, (_CACHELIST_TAG, [None]))


# --- engram SSD offload ----------------------------------------------------


def test_engram_tables_are_the_declared_streamable_components():
    """One table per engram layer, addressed by layer index, and the wire
    names the header-only pricing paths match on."""
    import gmlx.stream.table_stream as ts

    args = _args()
    model = Model(args)
    tabs = ts.streamable_tables_for(model)
    assert [t.gguf_name for t, _ in tabs] == [
        f"blk.{i}.engram_embd.weight" for i in args.engram_layer_ids]
    assert [m for _, m in tabs] == [
        model.model.layers[i].engram.embed for i in args.engram_layer_ids]
    names = ts.streamable_table_names("deepseek41")
    assert all(names.fullmatch(t.gguf_name) for t, _ in tabs)
    assert not names.fullmatch("blk.1.engram_wkv.weight")


def test_streamed_engram_tables_do_not_change_the_output():
    import gmlx.stream.table_stream as ts

    args = _args()
    model = Model(args)
    mx.eval(model.parameters())
    toks = mx.array(
        np.random.default_rng(3).integers(0, args.vocab_size, (1, 9)),
        dtype=mx.int32)
    ref = model(toks, cache=model.make_cache())
    mx.eval(ref)
    offloaded, names = ts.install_table_streaming(model)
    assert offloaded == ts.table_bytes(model) > 0
    assert names == [f"blk.{i}.engram_embd.weight"
                     for i in args.engram_layer_ids]
    got = model(toks, cache=model.make_cache())
    mx.eval(got)
    assert bool(mx.all(got == ref))


# --- fused hyper-connection route ------------------------------------------


def _wide_bf16_model():
    from mlx.utils import tree_flatten, tree_unflatten

    mx.random.seed(3)
    args = _args(hidden_size=1024, moe_intermediate_size=64,
                 num_attention_heads=8, head_dim=64, q_lora_rank=64,
                 o_lora_rank=32, index_head_dim=32)
    model = _randomized(Model(args))
    model.eval()
    keep = model.cast_predicate
    model.update(tree_unflatten([
        (k, v.astype(mx.bfloat16) if keep(k) and v.dtype == mx.float32 else v)
        for k, v in tree_flatten(model.parameters())]))
    mx.eval(model.parameters())
    return model


@pytest.mark.skipif(mx.default_device() != mx.gpu,
                    reason="the fused hyper-connection kernels are Metal-only")
@pytest.mark.parametrize("li", [0, 2, 5])
def test_fused_step_matches_the_ops_block(li):
    """At decode width the block runs the two-dispatch hyper-connection
    route; its stream, pre and next-layer carry agree with the ops
    block to rounding."""
    from gmlx.cache.compat import cache_types
    from gmlx.models.deepseek_v4.hyper_connection import hc_expand_m1

    model = _wide_bf16_model()
    inner = model.model
    layer = inner.layers[li]
    toks = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    h = (mx.random.normal((1, 1, 4, 1024)) * 0.5).astype(mx.bfloat16)
    pre = mx.softmax(mx.random.normal((1, 1, 4)), axis=-1)
    ids = mx.array([[33]])
    assert layer.attn_hc.m1_fused_ok(h)

    def fresh():
        cache = model.make_cache()
        mx.eval(model(mx.array([toks]), cache=cache))
        return inner._split_cache(li, cache[li], cache_types("CacheList"))

    h_ops, pre_ops = layer(h, pre, None, fresh(), len(toks), SharedStreams(),
                           ids)
    _, pre_f, carry = layer.fused_step(h, pre, None, None, fresh(), len(toks),
                                       SharedStreams(), ids)
    h_f = hc_expand_m1(*carry)
    mx.eval(h_ops, pre_ops, h_f, pre_f)
    scale = mx.abs(h_ops.astype(mx.float32)).max().item()
    dh = mx.abs(h_ops.astype(mx.float32) - h_f.astype(mx.float32))
    assert dh.max().item() < 0.03 * scale
    assert dh.mean().item() < 0.01 * scale
    assert mx.abs(pre_ops - pre_f).max().item() < 1e-2


@pytest.mark.skipif(mx.default_device() != mx.gpu,
                    reason="the fused hyper-connection kernels are Metal-only")
def test_fused_route_engages_at_decode_width_only(monkeypatch):
    import gmlx.models.deepseek_v41.model as m41

    model = _wide_bf16_model()
    calls = []
    orig = m41.DeepseekV41Block.fused_step

    def spy(self, *a, **k):
        calls.append(self.layer_idx)
        return orig(self, *a, **k)

    monkeypatch.setattr(m41.DeepseekV41Block, "fused_step", spy)
    cache = model.make_cache()
    mx.eval(model(mx.array([[5, 7, 9, 11, 13, 15, 17, 19, 21]]), cache=cache))
    assert calls == []                      # prefill: ops route
    mx.eval(model(mx.array([[23]]), cache=cache))
    assert calls == list(range(6))          # decode: every layer
    calls.clear()
    monkeypatch.setenv("GMLX_DS41_HC_FUSED", "0")
    mx.eval(model(mx.array([[25]]), cache=cache))
    assert calls == []
