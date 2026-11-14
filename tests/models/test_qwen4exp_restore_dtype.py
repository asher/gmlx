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
continued). Decode leaves the conv states float32, so the prefill after
the warm tail restore runs float32; its rows must not widen the restored
QSA streams. The hc prefill kernels are Metal only, so the file runs on a
real Apple GPU."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from helpers import _real_apple_gpu
from test_config_synth import _QWEN4EXP_SHAPES, _qwen4exp_meta

from gmlx._vendor.mlx_vlm.apc import APCManager
from gmlx.cache.snapshot import (
    _assemble_from_record,
    _ckpt_records,
    ckpt_layout,
    ckpt_lookup,
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
def arms():
    m = _model()
    V = m.args.vocab_size
    mx.random.seed(7)
    A = mx.random.randint(3, V, (70,)).tolist()
    G = mx.random.randint(3, V, (4,)).tolist()
    S = mx.random.randint(3, V, (12,)).tolist()
    B = A + G + S
    n = len(B)

    man = APCManager(num_blocks=64, block_size=BS)
    live = m.make_cache()
    _fwd(m, A[:64], live)
    assert ckpt_store(man, A[:64], live, kind="boundary",
                      skeleton_disk=False) == 64
    _fwd(m, A[64:69], live)
    _fwd(m, A[69:70], live)
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
            "warm_tail_vs_live": _cmp(warm_tail, lt, live, ll)}


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
    fresh prefill, and the logits stay within bf16 noise of it."""
    c = arms["warm_tail_vs_fresh"]
    assert c["meta"] == []
    lg = c["logits"]
    assert lg["argmax_eq"]
    assert lg["max_abs"] <= 0.02 * max(lg["scale"], 1.0)


def test_warm_tail_matches_live(arms):
    """The restored cache continues exactly as the cache that stored the
    record does."""
    c = arms["warm_tail_vs_live"]
    assert c["meta"] == []
    assert c["logits"]["max_abs"] == 0.0
