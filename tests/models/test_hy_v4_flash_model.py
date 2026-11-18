"""Hy4-flash model: hash, pooled selection and cache consistency on a tiny
random config."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.models.hy_v4_flash.model import (
    EngramHash,
    Model,
    ModelArgs,
    rope_at,
    split_cache,
)



@pytest.fixture(autouse=True)
def _on_cpu():
    # Exact fp32: the GPU's matmul precision varies with the chunk shape,
    # which hides a cache bug behind float noise on random weights.
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


N_LAYER = 6
SWA = [False, False, True, True, False, True]
VOCAB = 97
ENGRAM_LAYERS = [1, 4]
NGRAM, N_HEADS, HEAD_DIM = 3, 2, 8
COLS = (NGRAM - 1) * N_HEADS
PRIMES = [11, 13, 17, 19, 23, 29, 31, 37]
OFFSETS = list(np.concatenate([[0], np.cumsum(PRIMES)[:-1]]).astype(int))
# Two multipliers whose product with a mapped id passes 2**63, as two of
# the released model's do.
MULT = [(1 << 62) + 12345, 39690085502883, 24229915018165,
        19457948045719, (1 << 63) + 99, 84779352922315]


def _args(**kw) -> ModelArgs:
    base = dict(
        model_type="hy_v4_flash", vocab_size=VOCAB, hidden_size=32,
        intermediate_size=48, moe_intermediate_size=16,
        num_hidden_layers=N_LAYER,
        num_attention_heads=[4 if s else 2 for s in SWA],
        q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8,
        qk_rope_head_dim=8, v_head_dim=8, n_routed_experts=8,
        num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
        rms_norm_eps=1e-5, hc_mult=4, hc_eps=1e-6, hc_magnitude=2.0,
        sliding_window=6, sliding_window_pattern=SWA,
        rope_theta=1e7, rope_theta_swa=1e4, moe_latent_size=16,
        index_n_heads=2, index_head_dim=16, index_topk=8, index_kpool=4,
        index_select_tail=True, index_types=[not s for s in SWA],
        routed_scaling_factor=2.0, swiglu_limit=10.0,
        engram_layer_ids=ENGRAM_LAYERS, engram_max_ngram_size=NGRAM,
        engram_n_heads=N_HEADS, engram_head_dim=HEAD_DIM,
        engram_multipliers=MULT, engram_primes=PRIMES,
        engram_offsets=OFFSETS, engram_pad_id=VOCAB - 1,
        engram_token_map=[(7 * i + 3) % VOCAB for i in range(VOCAB)],
        engram_table_rows=int(OFFSETS[-1] + PRIMES[-1]),
    )
    base.update(kw)
    return ModelArgs(**base)


def _model(**kw) -> Model:
    mx.random.seed(7)
    model = Model(_args(**kw))
    # Random values for the raw arrays nn.Module init leaves at 0 or 1, so
    # every term of the forward is exercised.
    params = dict(tree_flatten(model.parameters()))
    upd = [(k, mx.random.normal(v.shape, scale=0.5).astype(v.dtype))
           for k, v in params.items()]
    model.load_weights(upd)
    mx.eval(model.parameters())
    return model


def test_rope_at_matches_the_rope_module():
    x = mx.random.normal((1, 2, 5, 8))
    ref = nn.RoPE(8, traditional=True, base=1e7)(x, offset=11)
    got = rope_at(x, mx.arange(11, 16), 1e7)
    assert mx.allclose(ref, got, atol=1e-5).item()


def _hash_reference(args: ModelArgs, ids: list[int]) -> np.ndarray:
    tmap = args.engram_token_map
    pad = tmap[args.engram_pad_id]
    n_e = len(args.engram_layer_ids)
    out = np.zeros((len(ids), n_e, COLS), dtype=np.int64)
    mask = (1 << 64) - 1
    for i in range(len(ids)):
        toks = [tmap[ids[i - s]] if i - s >= 0 else pad
                for s in range(NGRAM)]
        for e in range(n_e):
            mult = args.engram_multipliers[e * NGRAM:(e + 1) * NGRAM]
            rolling = (toks[0] * mult[0]) & mask
            for o in range(1, NGRAM):
                rolling ^= (toks[o] * mult[o]) & mask
                for h in range(N_HEADS):
                    col = (o - 1) * N_HEADS + h
                    out[i, e, col] = (rolling % PRIMES[e * COLS + col]
                                      + OFFSETS[e * COLS + col])
    return out


def test_hash_is_the_uint64_reference_in_one_call_and_across_calls():
    args = _args()
    ids = [int(v) for v in np.random.default_rng(0).integers(0, VOCAB, 23)]
    want = _hash_reference(args, ids)
    h = EngramHash(args)
    got, live, _ = h(mx.array([ids]), None)
    assert np.array_equal(np.array(got[0]), want)
    assert bool(live.all().item())
    # Fed in pieces, the history carries the look-back.
    hist, parts = None, []
    for a, b in ((0, 1), (1, 9), (9, 10), (10, 23)):
        part, _, hist = h(mx.array([ids[a:b]]), hist)
        parts.append(np.array(part[0]))
    assert np.array_equal(np.concatenate(parts), want)


def test_a_dead_token_is_masked_and_reads_as_pad_in_later_ngrams():
    args = _args()
    h = EngramHash(args)
    ids = [5, 9, 20, 31, 44]
    mask = mx.array([[True, True, False, True, True]])
    got, live, _ = h(mx.array([ids]), None, mask)
    assert np.array(live[0]).tolist() == [True, True, False, True, True]
    # The reference with the dead token's mapped id replaced by the pad id.
    tmap = list(args.engram_token_map)
    tmap[20] = tmap[args.engram_pad_id]
    want = _hash_reference(_args(engram_token_map=tmap), ids)
    assert np.array_equal(np.array(got[0]), want)


def _logits(model, ids, splits):
    cache = model.make_cache()
    outs, a = [], 0
    for b in splits:
        outs.append(model(mx.array([ids[a:b]]), cache))
        mx.eval(outs[-1], [c.state for c in cache])
        a = b
    return mx.concatenate(outs, axis=1)


@pytest.mark.parametrize("sparse", [True, False])
def test_chunked_prefill_and_decode_match_one_pass(sparse, monkeypatch):
    if not sparse:
        monkeypatch.setenv("GMLX_HY4_SPARSE_DISABLE", "1")
    model = _model()
    ids = [int(v) for v in np.random.default_rng(1).integers(0, VOCAB, 41)]
    one = _logits(model, ids, [41])
    # A prefill in uneven chunks, then single-token decode steps.
    split = _logits(model, ids, [5, 18, 19, 30] + list(range(31, 42)))
    assert mx.allclose(one, split, atol=1e-4, rtol=1e-4).item()


def test_a_table_that_reads_by_slot_gives_the_one_gather_logits():
    model = _model()
    ids = [int(v) for v in np.random.default_rng(2).integers(0, VOCAB, 23)]
    splits = [9, 20, 21, 22, 23]
    want = _logits(model, ids, splits)
    table = model.model.engram_embed
    shapes = []

    class BySlot:
        """The file-backed table's face: no gather of every slot at once."""

        def rows_by_slot(self, x):
            shapes.append(tuple(x.shape))
            return [table(x[:, :, s]) for s in range(x.shape[2])]

    model.model.engram_embed = BySlot()
    got = _logits(model, ids, splits)
    assert shapes == [(1, n, len(ENGRAM_LAYERS), COLS) for n in (9, 11, 1, 1, 1)]
    assert mx.array_equal(got, want)


def test_a_sliding_window_layer_holds_its_window_alone(monkeypatch):
    model = _model()
    ids = [int(v) for v in np.random.default_rng(4).integers(0, VOCAB, 41)]
    splits = [30] + list(range(31, 42))
    cache = model.make_cache()
    outs, a = [], 0
    for b in splits:
        outs.append(model(mx.array([ids[a:b]]), cache))
        mx.eval(outs[-1], [c.state for c in cache])
        a = b
    window = model.args.sliding_window
    for i, swa in enumerate(SWA):
        rows = split_cache(cache[i])[0].state[0].shape[2]
        assert rows == (window if swa else 41), i
    # The same steps over caches that keep every key.
    monkeypatch.setenv("GMLX_HY4_SWA_ROLL", "0")
    full = _logits(model, ids, splits)
    rolled = mx.concatenate(outs, axis=1)
    assert mx.allclose(rolled, full, atol=1e-4, rtol=1e-4).item()


def test_the_selection_binds_past_the_key_budget():
    # 41 tokens against an 8-key budget: the sparse forward must differ
    # from the dense one, and agree with it up to the budget.
    import os

    model = _model()
    ids = [int(v) for v in np.random.default_rng(2).integers(0, VOCAB, 41)]
    sparse = _logits(model, ids, [41])
    os.environ["GMLX_HY4_SPARSE_DISABLE"] = "1"
    try:
        dense = _logits(model, ids, [41])
    finally:
        del os.environ["GMLX_HY4_SPARSE_DISABLE"]
    assert not mx.allclose(sparse[:, -1], dense[:, -1], atol=1e-4).item()
    short = _logits(model, ids[:8], [8])
    assert mx.allclose(short, dense[:, :8], atol=1e-4, rtol=1e-4).item()
    assert mx.allclose(short, sparse[:, :8], atol=1e-4, rtol=1e-4).item()


def test_sparse_keys_are_the_reference_set():
    # Brute force over one query: top (budget / kpool - 1) pools that end
    # before the query's group, plus the group up to the query.
    model = _model()
    attn = model.layers[0].self_attn
    r, select_k = 4, 8 // 4 - 1
    for q_abs in (9, 10, 11, 12, 20):
        sel = mx.array([[q_abs // r - 1][:select_k]])
        tok, keep = attn._sparse_keys(sel, q_abs, 1)
        got = sorted(int(t) for t, k in zip(np.array(tok[0]),
                                            np.array(keep[0])) if k)
        pool = q_abs // r - 1
        want = list(range(pool * r, pool * r + r)) + list(
            range(q_abs // r * r, q_abs + 1))
        assert got == want
        mask = attn._sparse_mask(sel[None], q_abs, 1, q_abs + 1)
        assert sorted(np.flatnonzero(np.array(mask[0, 0])).tolist()) == want


def test_gathered_sparse_chunk_matches_the_masked_dense_form():
    # The chunk path gathers each query's keys; the same selection as a
    # visibility mask over every key is the reference.
    model = _model()
    attn = model.layers[0].self_attn
    B, L, S, offset = 1, 23, 37, 14
    H, R = attn.num_heads, attn.kv_lora_rank
    q_nope = mx.random.normal((B, H, L, attn.qk_nope_head_dim))
    q_pe = mx.random.normal((B, H, L, attn.qk_rope_head_dim))
    latent = mx.random.normal((B, 1, S, R))
    k_pe = mx.random.normal((B, 1, S, attn.qk_rope_head_dim))
    n_pool = S // attn.kpool
    # Distinct pools per query, as a top-k hands out.
    rng = np.random.default_rng(3)
    sel = mx.array(np.stack([[rng.permutation(n_pool)[:3]
                              for _ in range(L)]]).astype(np.int32))
    got = attn._sparse_prefill(q_nope, q_pe, latent, k_pe, sel, offset)
    mask = attn._sparse_mask(sel, offset, L, S)[:, None]
    want = attn._attend(q_nope, q_pe, latent, k_pe, mask)
    assert mx.allclose(got, want, atol=1e-4, rtol=1e-4).item()
