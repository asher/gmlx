"""Batched decode on the qwen4exp QSA layers against each row decoded
alone.

Past its indexer budget the model's attention is sparse: a row alone
attends its top blocks plus its incomplete tail block. A row in a decode
batch must compute that same attention.

Model level, teacher-forced: tiny qwen4exp with the real indexer budget
(2048 tokens, 512 blocks of 4, so a row is sparse from 2052 tokens), two
rows past it. Each row prefills alone, as serve prefills a prompt, then
the same forced tokens go into each row alone and into the two-row batch,
and the log-probabilities are compared per position. The bound is twice
the fixture's own noise, the larger of two runs that compute the model's
own sparse attention: the row alone with only its prefill chunk changed,
and the row beside an identical copy of itself in the model's own
two-row cache, which holds rows of one length and runs the sparse path
for both. The second carries the rounding of a two-row batch's
projections, which a batched row cannot avoid. Today's dense batch path,
each row over its whole context, runs beside them as the control; it
must sit above the bound, or the comparison has no power.

Layer level, on the same inputs: a batched row's attention core is the
single-row core on that row's slice of the batched projections, so it
equals the row alone bit for bit, through the kq kernels and through the
stock-op route, while rows join and leave the batch.

The hc prefill kernels and the kq kernels are Metal only; the stock-op
core case runs anywhere."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from helpers import _real_apple_gpu
from test_config_synth import _QWEN4EXP_SHAPES, _qwen4exp_meta

import gmlx.models.qwen4_exp.model as q4
from gmlx._vendor.mlx_lm.generate import _merge_caches
from gmlx.cache.kv_cache import BatchKVCache
from gmlx.load.config_synth import synthesize_config
from gmlx.models.qwen4_exp.model import (
    Attention,
    BatchQSAKVCache,
    Model,
    ModelArgs,
    QSAKVCache,
    ensure_registered,
)

gpu_only = pytest.mark.skipif(
    not _real_apple_gpu(), reason="the hc prefill kernels are Metal only")
kq_only = pytest.mark.skipif(
    not _real_apple_gpu(), reason="the kq attention kernels are Metal only")

# Prompt lengths: both rows past 2052 tokens, of different lengths mod 4.
_ROWS = (4002, 2903)
_STEPS = 6
_ALT_CHUNK = 1024


def _model():
    """Tiny qwen4exp in bf16 (hc inject float32, as the GGUF ships it) with
    the real indexer budget. Unit q and k norms of 3 on the QSA layer make
    its attention peaked, so the context it leaves out matters."""
    ensure_registered()
    meta = _qwen4exp_meta(True, True)
    meta["qwen4exp.attention.indexer.top_k"] = 2048
    cfg = synthesize_config(meta, _QWEN4EXP_SHAPES)
    m = Model(ModelArgs.from_dict(cfg))
    mx.random.seed(0)
    m.load_weights([(k, (mx.random.normal(v.shape) * 0.1).astype(v.dtype))
                    for k, v in nn.utils.tree_flatten(m.parameters())],
                   strict=False)
    for layer in m.layers:
        if not layer.is_linear and layer.self_attn.ratio > 0:
            attn = layer.self_attn
            attn.q_norm.weight = mx.full(attn.q_norm.weight.shape, 3.0)
            attn.k_norm.weight = mx.full(attn.k_norm.weight.shape, 3.0)
            assert attn.indexer.block_topk == 512
    m.set_dtype(mx.bfloat16)
    for layer in m.layers:
        for hc in (layer.hc_attn, layer.hc_ffn):
            hc.inject.weight = hc.inject.weight.astype(mx.float32)
    m.eval()
    mx.eval(m.parameters())
    return m


def _logprobs(out):
    return nn.log_softmax(out.astype(mx.float32), axis=-1)


def _prefill(m, prompt, chunk=None):
    """A row alone: the prompt to n-1 in chunks of ``chunk`` (one chunk by
    default), then the last column alone, as serve prefills."""
    cache = m.make_cache()
    head = prompt[:-1]
    step = chunk or len(head)
    for a in range(0, len(head), step):
        mx.eval(m(mx.array([head[a:a + step]]), cache=cache),
                [c.state for c in cache])
    mx.eval(m(mx.array([prompt[-1:]]), cache=cache), [c.state for c in cache])
    return cache


def _alone(m, prompt, forced, chunk=None):
    """[steps, vocab] log-probabilities of one row decoding ``forced``."""
    cache = _prefill(m, prompt, chunk)
    out = []
    for t in forced:
        lp = _logprobs(m(mx.array([[t]]), cache=cache)[0, -1])
        mx.eval(lp, [c.state for c in cache])
        out.append(lp)
    return mx.stack(out)


def _pair(m, prompt, forced):
    """[steps, vocab] log-probabilities of a row decoding ``forced`` beside
    an identical copy of itself, both rows in the model's own caches."""
    cache = m.make_cache()
    for cols in (prompt[:-1], prompt[-1:]):
        mx.eval(m(mx.array([cols, cols]), cache=cache), [c.state for c in cache])
    out = []
    for t in forced:
        lp = _logprobs(m(mx.array([[t], [t]]), cache=cache)[0, -1])
        mx.eval(lp, [c.state for c in cache])
        out.append(lp)
    return mx.stack(out)


def _dense_merge(rows):
    """Today's batch lift for the QSA layers: keys and values only, left
    padded, the indexer stream dropped."""
    out = []
    for i in range(len(rows[0])):
        layer = [r[i] for r in rows]
        if isinstance(layer[0], QSAKVCache):
            out.append(BatchKVCache.merge(layer))
        else:
            out.append(layer[0].merge(layer))
    return out


def _batched(m, prompts, forced, merge):
    """[rows, steps, vocab] log-probabilities of the rows decoding
    ``forced`` together after each prefilled alone."""
    batch = merge([_prefill(m, p) for p in prompts])
    out = []
    for step in zip(*forced):
        lp = _logprobs(m(mx.array([[t] for t in step]), cache=batch)[:, -1])
        mx.eval(lp, [c.state for c in batch])
        out.append(lp)
    return mx.stack(out, axis=1)


def _gap(ref, got):
    """Max abs log-probability difference and max per-position KL(ref || got)."""
    d = float(mx.abs(ref - got).max())
    kl = float((mx.exp(ref) * (ref - got)).sum(axis=-1).max())
    return d, kl


@pytest.fixture(scope="module")
def arms():
    m = _model()
    V = m.args.vocab_size
    mx.random.seed(41)
    prompts = [mx.random.randint(3, V, (n,)).tolist() for n in _ROWS]
    forced = [mx.random.randint(3, V, (_STEPS,)).tolist() for _ in _ROWS]
    alone = [_alone(m, p, f) for p, f in zip(prompts, forced)]
    alt = [_alone(m, p, f, chunk=_ALT_CHUNK) for p, f in zip(prompts, forced)]
    pair = [_pair(m, p, f) for p, f in zip(prompts, forced)]
    batch = _batched(m, prompts, forced, _merge_caches)
    dense = _batched(m, prompts, forced, _dense_merge)
    out = {"chunk": [], "pair": [], "batch": [], "dense": []}
    for i in range(len(_ROWS)):
        out["chunk"].append(_gap(alone[i], alt[i]))
        out["pair"].append(_gap(alone[i], pair[i]))
        out["batch"].append(_gap(alone[i], batch[i]))
        out["dense"].append(_gap(alone[i], dense[i]))
    print(f"\n[batch-decode] rows {_ROWS}, {_STEPS} forced steps; "
          "max abs logprob, max KL per row:")
    for k, v in out.items():
        print(f"[batch-decode]   {k}: "
              + "; ".join(f"{d:.4g}, {kl:.3g}" for d, kl in v))
    return out


def _bound(arms):
    return 2 * max(d for k in ("chunk", "pair") for d, _ in arms[k])


@gpu_only
def test_the_dense_control_sits_above_the_noise_floor(arms):
    """Power check: a dense batch row moves past the bound."""
    assert min(d for d, _ in arms["dense"]) > _bound(arms), arms


@gpu_only
def test_a_batched_row_decodes_as_the_row_alone(arms):
    """Each batched row's log-probabilities stay within twice the
    fixture's noise of the row decoded alone."""
    # A pair arm bit-equal to the row alone leaves the bound at the chunk
    # arm's 0: the fixture lost its two-row rounding, not the batch its rows.
    assert max(d for d, _ in arms["pair"]) > 0, arms
    assert max(d for d, _ in arms["batch"]) <= _bound(arms), arms


# The attention core and its wiring, one layer. A batched row runs the
# single-row core on its slice of the batched projections, so the core is
# bit-exact against the row alone on the same inputs; the projections are
# not (a GEMM over the batch rounds unlike a one-row GEMV).


def _layer(head_dim, dtype, budget):
    args = ModelArgs(hidden_size=128, num_hidden_layers=1,
                     num_attention_heads=12, num_key_value_heads=2,
                     head_dim=head_dim, indexer_budget=budget,
                     compress_ratios=[4], layer_types=["full_attention"])
    mx.random.seed(13)
    layer = Attention(args, 0)
    layer.set_dtype(dtype)
    for lin in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                layer.indexer.q_proj, layer.indexer.k_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.05).astype(dtype)
    return layer


def _prefilled(layer, n, seed):
    """Two caches of one row of ``n`` tokens, filled the same way."""
    dtype = layer.q_proj.weight.dtype
    mx.random.seed(seed)
    x = mx.random.normal((1, n, 128)).astype(dtype)
    out = []
    for _ in range(2):
        c = QSAKVCache(ratio=4)
        mx.eval(layer(x, mask="causal", cache=c))
        out.append(c)
    return out


def _route(monkeypatch, kernels):
    """``kernels=False`` takes the stock-op route: scores by matmul,
    argpartition, the take gather and stock SDPA."""
    if not kernels:
        for name in ("_kq_topk", "_kq_score", "_kq_paged", "_kq_paged_rows",
                     "_kv_gather", "_norm_rope"):
            monkeypatch.setattr(q4, name, lambda: None)


_CORE_CASES = {
    # rows of every length mod 4; the last crosses the sparse boundary
    "kernels": dict(head_dim=256, dtype=mx.bfloat16, budget=2048,
                    rows=(2301, 2350, 2419, 2044)),
    "ops": dict(head_dim=64, dtype=mx.float32, budget=8,
                rows=(41, 47, 58, 9)),
}


def _core_case(monkeypatch, route):
    case = _CORE_CASES[route]
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", "8")
    _route(monkeypatch, route == "kernels")
    layer = _layer(case["head_dim"], case["dtype"], case["budget"])
    pairs = [_prefilled(layer, n, 50 + i) for i, n in enumerate(case["rows"])]
    return layer, [p[0] for p in pairs], [p[1] for p in pairs]


@pytest.mark.parametrize("route", [
    pytest.param("kernels", marks=kq_only), "ops"])
def test_a_batched_row_core_is_the_row_alone(monkeypatch, route):
    """Each step, each batched row's attention core equals the core of the
    row alone on a copy of its cache, given the same projections. Rows
    join and leave the batch on the way, and no row's buffers are padded
    or widened for another."""
    layer, alone, rows = _core_case(monkeypatch, route)
    dtype = layer.q_proj.weight.dtype
    live = [0, 1, 2]
    batch = BatchQSAKVCache.merge([rows[i] for i in live])
    mx.random.seed(71)
    for step in range(14):
        if step == 5:
            live.append(3)
            batch.extend(rows[3])
        if step == 9:
            live.remove(1)
            batch.filter([0, 2, 3])
        x = mx.random.normal((len(live), 1, 128)).astype(dtype)
        q, _, k, v, ik = layer._project(x)
        got = layer._attend_rows(x, q, k, v, ik, batch, None)
        for j, i in enumerate(live):
            s = slice(j, j + 1)
            want = layer._attend(x[s], q[s], k[s], v[s], ik[s], None,
                                 alone[i], None)
            assert mx.array_equal(got[s], want).item(), (route, step, i)
            assert batch.rows[j] is rows[i]
            assert rows[i].offset == alone[i].offset
            assert rows[i]._kb.shape == alone[i]._kb.shape
    # the row that crossed the boundary in the batch went sparse
    assert rows[3].offset // 4 > layer.indexer.block_topk


@kq_only
def test_the_layer_on_a_batch_wires_the_rows_cores(monkeypatch):
    """The layer on a batch: the batched projections, each row's core on
    its own cache, then the batched gate and output projection."""
    layer, alone, rows = _core_case(monkeypatch, "kernels")
    batch = BatchQSAKVCache.merge(rows)
    mx.random.seed(73)
    for _ in range(3):
        x = mx.random.normal((4, 1, 128)).astype(mx.bfloat16)
        got = layer(x, mask=None, cache=batch)
        q, gate, k, v, ik = layer._project(x)
        cores = [layer._attend(x[i:i + 1], q[i:i + 1], k[i:i + 1],
                               v[i:i + 1], ik[i:i + 1], None, alone[i], None)
                 for i in range(4)]
        out = mx.concatenate(cores, axis=0).transpose(0, 2, 1, 3)
        want = layer.o_proj(out.reshape(4, 1, -1) * mx.sigmoid(gate))
        assert mx.array_equal(got, want).item()


def test_a_model_whose_first_attention_layer_is_qsa_beside_a_plain_one_decodes_batched():
    """The batch mask comes from the first attention layer's cache. Where
    that layer is QSA and a later one is plain, the QSA rows join the padded
    batch class, so the plain layer masks its padding, and two rows decode
    together as each does alone. Both rows here are restored: their QSA
    caches are rebuilt from state, as a lookup rebuilds them, before the
    last prompt token runs. The rows sit inside the indexer budget, where
    the QSA attention is dense either way."""
    ensure_registered()
    meta = _qwen4exp_meta(True, False)
    meta["qwen4exp.full_attention_interval"] = 2
    meta["qwen4exp.attention.compress_ratios"] = [0, 4, 0, 0]
    m = Model(ModelArgs.from_dict(synthesize_config(meta, _QWEN4EXP_SHAPES)))
    mx.random.seed(0)
    m.load_weights([(k, mx.random.normal(v.shape) * 0.1)
                    for k, v in nn.utils.tree_flatten(m.parameters())],
                   strict=False)
    mx.eval(m.parameters())
    V = m.args.vocab_size
    mx.random.seed(3)
    prompts = [mx.random.randint(3, V, (n,)).tolist() for n in (6, 4)]
    nxt = mx.random.randint(3, V, (2,)).tolist()

    def restored(p):
        cache = m.make_cache()
        m(mx.array([p[:-1]]), cache=cache)
        q = cache[1]
        cache[1] = QSAKVCache.from_state(q.state, q.meta_state)
        m(mx.array([p[-1:]]), cache=cache)
        return cache

    alone = [m(mx.array([[t]]), cache=restored(p))[0, -1]
             for p, t in zip(prompts, nxt)]
    batch = _merge_caches([restored(p) for p in prompts])
    got = m(mx.array([[t] for t in nxt]), cache=batch)[:, -1]
    for i in range(2):
        scale = float(mx.abs(alone[i]).max())
        # the batched projections round unlike a one-row GEMV
        assert float(mx.abs(got[i] - alone[i]).max()) < 2e-3 * scale
    assert isinstance(batch[1], BatchKVCache)
    fresh = m.make_cache()
    assert [type(c) for c in fresh[1::2]] == [QSAKVCache, q4.KVCache]
    assert fresh[1].dense_lift
