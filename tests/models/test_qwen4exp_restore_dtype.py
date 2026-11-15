"""A turn after a checkpoint restore against the same turn on a fresh
prefill, tiny qwen4exp in bf16 with the hc inject kept float32 as the GGUF
ships it.

Turn 1, one live cache: prefill A[:64], a boundary store at 64 (no KV
tail), A[64:69], the last prompt column alone, 4 decode steps, and a
retire store at 74 (a KV tail of 10 rows). Turn 2's prompt B = A + G + S
(86 tokens) prefills as the serve path does: one chunk to n-1, then the
last column alone.

Arms: fresh (B from an empty cache), warm grid (the 64 record restored),
warm tail (the 74 record restored) and live (turn 1's own cache
continued). Decode leaves the conv states float32 and the retire record
keeps them. The prefill after the warm tail restore casts them to the
activation dtype, so it runs bf16 as fresh does and the hc prefill
kernels engage. The last column alone runs float32 off the hc residual;
its rows must not widen the restored QSA streams, nor the decode tails it
opens.

Turn 1's prompt record, two ways: the p=N store at 70 after the last
column, and the N-1 replay at 69 standing in for it as a boundary. Turn 2
restored from each must answer as fresh does.

Batched decode: two rows joined into a decode batch decode off float32
conv states too, and each row's QSA keys must keep their dtype in the
row's own buffers.

The hc prefill kernels are Metal only, so the file runs on a real Apple
GPU."""

from __future__ import annotations

import functools

import mlx.core as mx
import mlx.nn as nn
import pytest

from helpers import _real_apple_gpu
from test_config_synth import _QWEN4EXP_SHAPES, _qwen4exp_meta

import gmlx.models.qwen4_exp.model as q4
from gmlx._vendor.mlx_lm.generate import _merge_caches
from gmlx._vendor.mlx_vlm.apc import APCManager
from gmlx.cache.kv_cache import ArraysCache
from gmlx.cache.snapshot import (
    _assemble_from_record,
    _ckpt_records,
    ckpt_layout,
    ckpt_lookup,
    ckpt_replay_stands_in,
    ckpt_store,
)
from gmlx.load.config_synth import synthesize_config
from gmlx.models.qwen4_exp.model import Model, ModelArgs, ensure_registered

pytestmark = pytest.mark.skipif(
    not _real_apple_gpu(), reason="the hc prefill kernels are Metal only")

BS = 16


def _model():
    ensure_registered()
    cfg = synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES)
    m = Model(ModelArgs.from_dict(cfg))
    mx.random.seed(0)
    m.load_weights([(k, (mx.random.normal(v.shape) * 0.1).astype(v.dtype))
                    for k, v in nn.utils.tree_flatten(m.parameters())],
                   strict=False)
    m.set_dtype(mx.bfloat16)
    for layer in m.model.layers:
        for hc in (layer.hc_attn, layer.hc_ffn):
            hc.inject.weight = hc.inject.weight.astype(mx.float32)
    m.eval()
    mx.eval(m.parameters())
    return m


def _fwd(m, toks, cache):
    out = m(mx.array([toks]), cache=cache)
    mx.eval(out, [c.state for c in cache])
    return out[0, -1]


def _leaves(cache):
    """{name: array} over every layer's state."""
    out = {}
    for i, c in enumerate(cache):
        names = (("k", "v", "ik", "pos") if hasattr(c, "update_and_fetch_qsa")
                 else ("k", "v") if hasattr(c, "keys") else None)
        for j, a in enumerate(c.state):
            if a is not None:
                nm = names[j] if names else f"s{j}"
                out[f"L{i}.{type(c).__name__}.{nm}"] = a
    return out


def _cmp(a_cache, a_logit, b_cache, b_logit):
    la, lb = _leaves(a_cache), _leaves(b_cache)
    meta, diffs = [], {}
    for k in sorted(set(la) | set(lb)):
        x, y = la.get(k), lb.get(k)
        mx_ = None if x is None else (x.dtype, x.shape, x.nbytes)
        my_ = None if y is None else (y.dtype, y.shape, y.nbytes)
        if mx_ != my_:
            meta.append((k, mx_, my_))
        elif x is not None:
            diffs[k] = float(mx.abs(x.astype(mx.float32)
                                    - y.astype(mx.float32)).max())
    a32, b32 = a_logit.astype(mx.float32), b_logit.astype(mx.float32)
    logits = {"max_abs": float(mx.abs(a32 - b32).max()),
              "scale": float(mx.abs(b32).max()),
              "argmax_eq": int(mx.argmax(a_logit)) == int(mx.argmax(b_logit))}
    return {"meta": meta, "diffs": diffs, "logits": logits}


@pytest.fixture(scope="module")
def model():
    return _model()


def _qsa_layers(m):
    return [i for i, layer in enumerate(m.layers)
            if not layer.is_linear and layer.self_attn.ratio > 0]


@pytest.fixture(scope="module")
def arms(model):
    m = model
    V = m.args.vocab_size
    mx.random.seed(7)
    A = mx.random.randint(3, V, (70,)).tolist()
    G = mx.random.randint(3, V, (4,)).tolist()
    S = mx.random.randint(3, V, (12,)).tolist()
    B = A + G + S
    n = len(B)

    man = APCManager(num_blocks=64, block_size=BS)
    # Turn 1's prompt record: the p=N store (man_n) or the N-1 replay
    # standing in for it (man_r).
    man_n = APCManager(num_blocks=64, block_size=BS)
    man_r = APCManager(num_blocks=64, block_size=BS)
    live = m.make_cache()
    _fwd(m, A[:64], live)
    assert ckpt_store(man, A[:64], live, kind="boundary",
                      skeleton_disk=False) == 64
    _fwd(m, A[64:69], live)
    for mm in (man_n, man_r):
        assert ckpt_store(mm, A[:69], live, kind="replay",
                          skeleton_disk=False) == 69
    _fwd(m, A[69:70], live)
    assert ckpt_store(man_n, A, live, skeleton_disk=False) == 70
    assert ckpt_replay_stands_in(man_r, A, None)
    for t in G:
        _fwd(m, [t], live)
    assert ckpt_store(man, A + G, live, kind="retire",
                      skeleton_disk=False) == 74
    recs = {r.p: r for r in _ckpt_records(man).values()}
    layout = ckpt_layout(live, BS)

    warm_tail, p_tail = ckpt_lookup(man, B, layout=layout)
    assert p_tail == 74
    _fwd(m, B[74:n - 1], warm_tail)
    lt = _fwd(m, B[n - 1:], warm_tail)
    tails = [(str(warm_tail[i]._kt.dtype), str(warm_tail[i]._kb.dtype),
              str(warm_tail[i]._it.dtype), str(warm_tail[i]._ib.dtype))
             for i in _qsa_layers(m)]

    turn2 = {}
    for name, mm, want in (("p_n", man_n, 70), ("n_1", man_r, 69)):
        warm, got = ckpt_lookup(mm, B, layout=layout)
        assert got == want, name
        _fwd(m, B[got:n - 1], warm)
        turn2[name] = (warm, _fwd(m, B[n - 1:], warm))

    warm_grid = _assemble_from_record(man, recs[64])
    _fwd(m, B[64:n - 1], warm_grid)
    lg = _fwd(m, B[n - 1:], warm_grid)

    _fwd(m, B[74:n - 1], live)
    ll = _fwd(m, B[n - 1:], live)

    fresh = m.make_cache()
    _fwd(m, B[:64], fresh)
    _fwd(m, B[64:n - 1], fresh)
    lf = _fwd(m, B[n - 1:], fresh)

    return {"warm_tail_vs_fresh": _cmp(warm_tail, lt, fresh, lf),
            "warm_grid_vs_fresh": _cmp(warm_grid, lg, fresh, lf),
            "warm_tail_vs_live": _cmp(warm_tail, lt, live, ll),
            "warm_tail_tails": tails,
            "p_n_vs_fresh": _cmp(*turn2["p_n"], fresh, lf),
            "n_1_vs_fresh": _cmp(*turn2["n_1"], fresh, lf),
            "n_1_vs_p_n": _cmp(*turn2["n_1"], *turn2["p_n"])}


# Logit bounds relative to the logit scale. Measured on the M5 Max: about
# 7.5e-5 of the scale against fresh, 2.5e-7 between the two records.
_VS_FRESH = 2e-3
_VS_RECORD = 1e-4


def _within(lg, rel):
    return lg["argmax_eq"] and lg["max_abs"] <= rel * lg["scale"]


def test_warm_grid_matches_fresh(arms):
    """Control: a restore with no KV tail runs the same chunks as fresh
    from 64 on, so the cache and the logits match bit for bit."""
    c = arms["warm_grid_vs_fresh"]
    assert c["meta"] == []
    assert all(d == 0.0 for d in c["diffs"].values())
    assert c["logits"]["max_abs"] == 0.0


def test_warm_tail_matches_fresh(arms):
    """A record stored after decode, restored, then the follow-up
    prefilled: every cache leaf keeps the dtype, shape and bytes of a
    fresh prefill, and the logits stay within 2e-3 of fresh's logit
    scale."""
    c = arms["warm_tail_vs_fresh"]
    assert c["meta"] == []
    assert _within(c["logits"], _VS_FRESH), c["logits"]


def test_warm_tail_matches_live(arms):
    """The restored cache continues exactly as the cache that stored the
    record does."""
    c = arms["warm_tail_vs_live"]
    assert c["meta"] == []
    assert c["logits"]["max_abs"] == 0.0


def test_warm_tail_decode_tails_keep_the_base_dtype(arms):
    """The last column after the warm tail restore opens the decode tails
    off the float32 hc residual. They take the base's dtype, so
    gather_kv's fused kernel stays on for the turn."""
    for kt, kb, it, ib in arms["warm_tail_tails"]:
        assert (kt, it) == (kb, ib)


@pytest.mark.parametrize("pair,rel", [("p_n_vs_fresh", _VS_FRESH),
                                      ("n_1_vs_fresh", _VS_FRESH),
                                      ("n_1_vs_p_n", _VS_RECORD)])
def test_turn_two_from_the_prompt_record(arms, pair, rel):
    """Turn 2 restored from the p=N record, from the N-1 replay standing in
    for it, and fresh: one argmax. Each record's logits are within 2e-3
    of fresh's logit scale (the warm path rounds through other chunks),
    and the two records within 1e-4 of the scale of each other, since
    both restore the same prompt state."""
    assert _within(arms[pair]["logits"], rel), arms[pair]["logits"]


def test_turn_two_from_the_stand_in_matches_fresh_leaves(arms):
    """The stand-in was stored mid-prefill, so turn 2 off it runs the
    prefill dtype path: every cache leaf keeps the dtype, shape and bytes
    of a fresh prefill."""
    assert arms["n_1_vs_fresh"]["meta"] == []


def test_batched_decode_keeps_the_qsa_dtype(model):
    """Two rows joined into a decode batch, two decode steps off the
    float32 conv states the last column leaves: each row's QSA keys and
    values, in its base and its decode tail, keep the dtype they had at
    the join."""
    m = model
    V = m.args.vocab_size
    mx.random.seed(11)
    rows = []
    for n in (40, 33):
        P = mx.random.randint(3, V, (n,)).tolist()
        c = m.make_cache()
        _fwd(m, P[:-1], c)
        _fwd(m, P[-1:], c)
        rows.append(c)
    qsa = _qsa_layers(m)
    assert qsa
    joined = {(i, r): (rows[r][i]._kb.dtype, rows[r][i]._vb.dtype)
              for i in qsa for r in range(2)}
    batch = _merge_caches(rows)
    tok = mx.array([[3], [4]])
    for _ in range(2):
        out = m(tok, cache=batch)
        mx.eval(out, [c.state for c in batch])
        tok = mx.argmax(out[:, -1:, :], axis=-1)
    for i in qsa:
        for r, row in enumerate(batch[i].rows):
            assert row is rows[r][i] and row.offset == (40, 33)[r] + 2
            k_dt, v_dt = joined[i, r]
            assert {row._kb.dtype, row._kt.dtype} == {k_dt}
            assert {row._vb.dtype, row._vt.dtype} == {v_dt}


@pytest.fixture(scope="module")
def retired(model):
    """A retire record at 74 stored after 4 decode steps, and turn 2's
    prompt."""
    m = model
    V = m.args.vocab_size
    mx.random.seed(13)
    A = mx.random.randint(3, V, (70,)).tolist()
    G = mx.random.randint(3, V, (4,)).tolist()
    B = A + G + mx.random.randint(3, V, (12,)).tolist()
    man = APCManager(num_blocks=64, block_size=BS)
    live = m.make_cache()
    _fwd(m, A[:-1], live)
    _fwd(m, A[-1:], live)
    for t in G:
        _fwd(m, [t], live)
    assert ckpt_store(man, A + G, live, kind="retire",
                      skeleton_disk=False) == 74
    return man, B, ckpt_layout(live, BS)


def _conv_states(cache):
    """(layer cache, slot) of every conv state: the GDN's, and the PLE's
    in a layer that carries one."""
    return [(c, j) for c in cache if isinstance(c, ArraysCache)
            for j in ((0, 2) if len(c.cache) == 4 else (0,))]


def test_prefill_after_a_retire_restore_runs_in_the_activation_dtype(
        model, retired, monkeypatch):
    """The retire record keeps the float32 conv states decode left (on
    this fixture's eager hc path every layer's after the first, whose
    input is the bf16 embedding; on the real model the PLE's). The
    follow-up prefill casts them to the activation dtype, so it runs bf16
    as a fresh prefill does and every hc combine takes the prefill
    kernel."""
    m = model
    man, B, layout = retired
    warm, p = ckpt_lookup(man, B, layout=layout)
    assert p == 74
    f32 = [j for c, j in _conv_states(warm) if c[j].dtype == mx.float32]
    assert 0 in f32 and 2 in f32, "a GDN and the PLE conv state are float32"
    seen = []
    kern = q4._hc_combine_kern

    @functools.wraps(kern)
    def spy(*a, **k):
        y = kern(*a, **k)
        seen.append(y is not None)
        return y

    monkeypatch.setattr(q4, "_hc_combine_kern", spy)
    out = _fwd(m, B[p:-1], warm)
    assert out.dtype == mx.bfloat16
    assert seen and all(seen)


def test_the_cast_equals_the_float32_state_cast_elementwise(model, retired):
    """The follow-up prefill off the record's float32 conv states answers
    bit for bit as the same prefill off those states cast to bf16 first,
    leaf for leaf and logit for logit."""
    m = model
    man, B, layout = retired
    runs = []
    for cast in (False, True):
        warm, p = ckpt_lookup(man, B, layout=layout)
        assert p == 74
        if cast:
            for c, j in _conv_states(warm):
                c[j] = c[j].astype(mx.bfloat16)
        runs.append((warm, _fwd(m, B[p:-1], warm)))
    c = _cmp(*runs[0], *runs[1])
    assert c["meta"] == []
    assert all(d == 0.0 for d in c["diffs"].values())
    assert c["logits"]["max_abs"] == 0.0
