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


def _held_table(table):
    """The file-backed table's face over ``table``, holding the rows of
    one call: the chunk widths it was asked to hold, and per read whether
    the held rows were this call's."""
    held, served = [], []

    class Held:
        ahead = None

        def hold_ahead(self, x):
            held.append(x.shape[1])
            self.ahead = x

        def rows_by_slot(self, x):
            ahead, self.ahead = self.ahead, None
            n = x.shape[1]
            served.append(ahead is not None and n <= ahead.shape[1]
                          and mx.array_equal(x, ahead[:, :n]).item())
            return [table(x[:, :, s]) for s in range(x.shape[2])]

    return Held(), held, served


def test_a_planned_chunk_reads_its_rows_during_the_chunk_before():
    model = _model()
    ids = [int(v) for v in np.random.default_rng(3).integers(0, VOCAB, 30)]
    splits = [8, 16, 21, 29, 30]
    want = _logits(model, ids, splits)
    model.model.engram_embed, held, served = _held_table(
        model.model.engram_embed)
    cache, outs, lo = model.make_cache(), [], 0
    for hi in splits:
        # As a prefill loop plans: a step of 8, the last token no chunk's.
        rest = len(ids) - hi - 1
        if rest >= 1:
            model.table_rows_ahead(
                mx.array([ids[hi:hi + min(8, rest)]]), hi - lo)
        outs.append(model(mx.array([ids[lo:hi]]), cache))
        mx.eval(outs[-1], [c.state for c in cache])
        lo = hi
    got = mx.concatenate(outs, axis=1)
    # Chunks of 8, 8, 5 (short of its plan), 8, then the last token.
    assert held == [8, 8, 8]
    assert served == [False, True, True, True, False]
    assert mx.array_equal(got, want)

    # A plan for a call of another length is dropped, not read.
    model.table_rows_ahead(mx.array([ids[:4]]), 7)
    model(mx.array([ids[:3]]), model.make_cache())
    assert held == [8, 8, 8]


def test_generate_reads_each_chunk_s_rows_a_chunk_ahead(monkeypatch):
    # The prefill loop of run and chat: every chunk after the first finds
    # its rows held, and the token it keeps for the first step reads its
    # own.
    from gmlx._vendor.mlx_lm.generate import generate_step

    model = _model()
    ids = [int(v) for v in np.random.default_rng(4).integers(0, VOCAB, 30)]

    def first_token():
        return next(generate_step(mx.array(ids), model, max_tokens=1,
                                  prefill_step_size=8))[0]

    want = first_token()
    model.model.engram_embed, held, served = _held_table(
        model.model.engram_embed)
    assert first_token() == want
    # Chunks of 8, 8, 8, 5, then the decode steps.
    assert held == [8, 8, 5]
    assert served[:4] == [False, True, True, True] and not any(served[4:])

    monkeypatch.setenv("GMLX_TABLE_READAHEAD", "0")
    del held[:], served[:]
    assert first_token() == want
    assert held == [] and not any(served)


def test_a_deep_prefill_chunk_is_sized_by_the_pooled_scores(monkeypatch):
    # The server sizes each prefill chunk by the score transient. A head
    # count per layer failed that sizing, and the dense model (64 heads
    # against the whole depth) would cut a 128k-deep chunk to a few rows.
    import gmlx.gen.prefill_decay as pd
    import gmlx.models.hy_v4_flash.model as hy

    assert pd._SCORE_PROFILES["hy_v4_flash"] is hy._prefill_score_profile
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "6")
    model = _model()
    cache = model.make_cache()
    kpool = model.args.index_kpool
    heads = pd.score_heads(model.args)
    assert heads == max(model.args.num_attention_heads)

    # The ops path: 512 queries at a time, every indexer head in fp32.
    ops = pd.resolve_score_profile(model, cache)
    assert ops == pd.ScoreTransientProfile(
        heads=model.args.index_n_heads + 1, bytes_per_elem=4,
        depth_divisor=kpool, rows=512)
    # kq's chunk kernel: one score per query and pool.
    monkeypatch.setattr(hy, "_kq_indexer_kernels", lambda: (None, None))
    monkeypatch.setattr(hy, "_kq_indexer_chunk_kernel", lambda: object())
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu))
    kq = pd.resolve_score_profile(model, cache)
    assert kq == pd.ScoreTransientProfile(
        heads=1, bytes_per_elem=4, depth_divisor=kpool)

    depth = 131072
    for prof in (ops, kq):
        assert pd.decayed_step(2048, depth, 64, profile=prof) == 2048
    assert pd.decayed_step(2048, depth, 64) < 512
    assert pd.score_transient_bytes(model, cache, depth) < 0.5e9


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


def test_window_keys_from_the_dequantized_stack_match_the_gather(monkeypatch):
    """A prefill chunk's per-head keys: the K-quant stack dequantized and
    multiplied in one matmul, against kq's untransposed gather."""
    kq = pytest.importorskip("mlx_kquant")
    from mlx_kquant.nn import KQuantMultiLinear

    from gmlx.models.hy_v4_flash.model import _head_keys

    heads, nope, rank, rows = 4, 64, 96, 37
    mx.random.seed(3)
    wire = kq.quantize(
        mx.random.normal((heads * rank, nope)) * 0.05, "q8_0")
    wire = wire[0] if isinstance(wire, (list, tuple)) else wire
    emb = KQuantMultiLinear(nope, rank, heads, "q8_0")
    emb.weight = wire.reshape(heads, rank, -1)
    latent = mx.random.normal((2, 1, rows, rank)).astype(mx.bfloat16)

    want = np.array(emb(latent, transpose=False).astype(mx.float32))
    got = _head_keys(emb, latent)
    assert got.shape == (2, heads, rows, nope) and got.dtype == mx.bfloat16
    # The matmul reads the weights rounded to bf16, the gather exact.
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), want, rtol=0,
        atol=0.02 * np.abs(want).max())

    monkeypatch.setenv("GMLX_HY4_KEYS_DENSE", "0")
    np.testing.assert_array_equal(
        np.array(_head_keys(emb, latent).astype(mx.float32)), want)


# The released file's tensor names, by the parts a block is made of.
_ATTN_TENSORS = (
    "attn_gate", "attn_k_b", "attn_kv_a_mqa", "attn_kv_a_norm", "attn_norm",
    "attn_output", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_sinks",
    "attn_v_b", "ffn_norm", "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
    "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale", "post_attention_norm",
    "post_ffw_norm")
_DENSE_TENSORS = ("ffn_down", "ffn_gate", "ffn_up")
_MOE_TENSORS = (
    "ffn_down_exps", "ffn_down_shexp", "ffn_gate_exps", "ffn_gate_inp",
    "ffn_gate_shexp", "ffn_routed_down", "ffn_routed_up", "ffn_up_exps",
    "ffn_up_shexp")
_INDEXER_TENSORS = (
    "indexer.attn_k", "indexer.attn_q_b", "indexer.k_norm", "indexer.proj",
    "indexer_compressor_ape", "indexer_compressor_gate",
    "indexer_compressor_norm", "indexer_compressor_value")
_ENGRAM_TENSORS = (
    "engram_gate", "engram_hc_scale", "engram_kv", "engram_kv_b",
    "engram_lookup_norm", "engram_pre_gamma")
_GLOBAL_TENSORS = (
    "output", "output_hc_base", "output_hc_fn", "output_hc_scale",
    "output_norm", "token_embd", "per_layer_token_embd")


def test_the_file_s_tensor_names_fill_the_model_and_nothing_else():
    """Every kind of block the released file has: the dense first block,
    sparse and window blocks with and without engram, and the draft block
    after the trunk, which a load without MTP drops."""
    from gmlx.load.wire import remap_arrays

    # Engram on a sparse layer (1) and on a window layer (3), as blocks 1
    # and 11 of the file.
    args = _args(engram_layer_ids=[1, 3])
    model = Model(args)
    params = {k for k, _ in tree_flatten(model.parameters())}

    names = [f"{t}.weight" for t in _GLOBAL_TENSORS]
    for i in range(N_LAYER):
        parts = _ATTN_TENSORS + (_DENSE_TENSORS if i == 0 else _MOE_TENSORS)
        if not SWA[i]:
            parts += _INDEXER_TENSORS
        if i in (1, 3):
            parts += _ENGRAM_TENSORS
        names += [f"blk.{i}.{t}.weight" for t in parts]
        if i:
            names.append(f"blk.{i}.exp_probs_b.bias")
    trunk = len(names)
    draft = _ATTN_TENSORS[:12] + _ATTN_TENSORS[18:] + _MOE_TENSORS \
        + _INDEXER_TENSORS
    names += [f"blk.{N_LAYER}.{t}.weight" for t in draft]
    names.append(f"blk.{N_LAYER}.exp_probs_b.bias")
    block = len(names) - trunk
    # The draft block's own four, which only the MTP load reads.
    nextn = [f"blk.{N_LAYER}.nextn.{t}.weight" for t in (
        "eh_proj", "enorm", "hnorm", "shared_head_norm")]

    weights, _, stats = remap_arrays(
        {n: mx.zeros((4, 4)) for n in names + nextn}, {}, "hy_v4_flash",
        n_head=2, n_head_kv=1)
    assert stats["mapped"] == len(names) and stats["failed"] == 0
    assert stats["skipped"] == len(nextn)
    kept = set(model.sanitize(weights))
    assert len(weights) - len(kept) == block
    assert kept == params, (
        f"only in the file: {sorted(kept - params)}; only in the model: "
        f"{sorted(params - kept)}")
