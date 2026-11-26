"""GMLX_Q4_VERIFY_FUSED_WHOLE: an attention layer of a verify past the
sparse boundary replays as one record, its rope offset, append offsets,
base length and block counts read from a parameter block. The hiddens and
the cache states carry the bits of the plain path through full accepts,
rollbacks, folds of the tail and rebuilds of the block base; a call the
record does not cover takes the split form or the plain call."""

import functools

import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest

import gmlx.models.qwen4_exp.model as q4
from gmlx.models.qwen4_exp import qsa_whole
from gmlx.models.qwen4_exp.mtp import Qwen4ExpSpecLM

from helpers import _real_apple_gpu
from test_qwen4exp_mtp import _args, _randomize

pytestmark = pytest.mark.skipif(
    not (hasattr(kq, "fused_bind") and _real_apple_gpu()),
    reason="needs a kq build with parameter blocks and a real Apple GPU")

# (rows, kept rows) of one pass; 36 tokens are kept a pass.
PASS = [(4, 4), (4, 4), (4, 4), (4, 2), (3, 3), (3, 1), (8, 8), (8, 3),
        (2, 2), (5, 5)]
PROMPT = 2301


@pytest.fixture(autouse=True)
def _levers(monkeypatch):
    monkeypatch.setattr(q4, "_VERIFY_BLOCKED", False)
    monkeypatch.setattr(q4, "_VERIFY_FUSED", False)
    monkeypatch.setattr(q4, "_VERIFY_FUSED_QSA", True)
    monkeypatch.setattr(q4, "_VERIFY_FUSED_CAP", True)
    monkeypatch.setattr(q4, "_FUSED_WHOLE", False)
    monkeypatch.setattr(q4, "_FUSED_LIMITS", (4096, 256 << 20))
    yield
    kq.env_cache(False)


def _lm(dtype):
    args = _args(head_dim=256, indexer_budget=2048, indexer_n_heads=4,
                 indexer_head_dim=128, mrope_section=[22, 22, 22])
    lm = Qwen4ExpSpecLM(args)
    _randomize(lm)
    lm.set_dtype(dtype)
    mx.eval(lm.parameters())
    q4.prepare_runtime(lm)
    for m in lm.modules():
        if isinstance(m, q4.GatedDeltaNet):
            m._verify_v2 = True
    lm.eval()
    return lm, args


def _walk(monkeypatch, mode, dtype=mx.bfloat16, passes=6, tail=64,
          rounds=None, steps_at=(), prompt=PROMPT):
    """mode: "plain", "split" (two records and the Python middle) or
    "whole"."""
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", str(tail))
    monkeypatch.setattr(q4, "_VERIFY_FUSED", mode != "plain")
    monkeypatch.setattr(q4, "_FUSED_WHOLE", mode == "whole")
    kq.env_cache(mode != "plain")
    lm, args = _lm(dtype)
    lm.whole_calls = 0
    if mode == "whole":
        inner = q4._fused_qsa_whole

        @functools.wraps(inner)
        def counted(*a, **k):
            out = inner(*a, **k)
            lm.whole_calls += out is not None
            return out

        monkeypatch.setattr(q4, "_fused_qsa_whole", counted)
    mx.random.seed(5)
    cache = lm.make_cache()
    mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, prompt)),
               cache=cache).logits)
    out = []
    for i, (S, kept) in enumerate(rounds or PASS * passes):
        if i in steps_at:
            tok = mx.random.randint(0, args.vocab_size, (1, 1))
            out.append(np.array(lm(tok, cache=cache).logits.astype(mx.float32)))
        block = mx.random.randint(0, args.vocab_size, (1, S))
        hid, _, sink = lm.speculative_verify_hidden(block, cache)
        lm.rollback_speculative_cache(cache, sink, kept - 1, S)
        mx.eval(hid, [c.state for c in cache])
        out.append(np.array(hid.astype(mx.float32)))
    out.extend(np.array(a.astype(mx.float32))
               for c in cache for a in c.state if isinstance(a, mx.array))
    return lm, out


def _equal(a, b):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape and np.array_equal(x, y), i


def _whole(lm):
    """(records, layer calls the whole form ran, whole-form primitives that refused
    a direct replay) of the whole-layer records."""
    recs = [r for layer in lm.model.layers if not layer.is_linear
            for r in vars(layer).get("_verify_fused", (0, {}, []))[1].values()
            if isinstance(r, tuple) and r[2] == "whole"]
    infos = [kq.fused_direct_info(r[0]) for r in recs]
    # The tiny model's hyper-connection norm and float MoE are not on the
    # direct list, so its records run their primitives; none of the
    # primitives the whole form adds may be a reason for that.
    bad = sorted({n for i in infos for n in i["primitives"]
                  if n.split()[0] in _WHOLE_PRIMS})
    return len(recs), lm.whole_calls, bad


_WHOLE_PRIMS = {
    "KQuantKVAppend", "RoPE", "View", "Gather", "Slice", "Squeeze",
    "CustomKernel", "KQDsaIndexerScoreDecode", "KQDsaTopKIndices",
    "KQuantSDPAPagedRows", "LessEqual", "ScaledDotProductAttention",
    "Concatenate", "KQuantSDPAGQA", "Multiply", "Broadcast"}


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
def test_whole_layer_records_equal_the_plain_path(monkeypatch, dtype):
    _, plain = _walk(monkeypatch, "plain", dtype)
    lm, whole = _walk(monkeypatch, "whole", dtype)
    n, used, refused = _whole(lm)
    assert not refused
    # Three folds of the tail in the walk: a layout a width and a fold.
    assert n >= 8 and used >= 20, (n, used)
    _equal(plain, whole)
    _, split = _walk(monkeypatch, "split", dtype)
    _equal(plain, split)


def test_decode_steps_between_whole_layer_rounds(monkeypatch):
    # A one-row step runs the plain attention and leaves the block tail in
    # the form the record reads.
    at = (7, 8, 19, 33, 34, 35)
    _, plain = _walk(monkeypatch, "plain", passes=4, steps_at=at)
    lm, whole = _walk(monkeypatch, "whole", passes=4, steps_at=at)
    assert _whole(lm)[1] >= 8
    _equal(plain, whole)


def test_a_small_tail_folds_often(monkeypatch):
    _, plain = _walk(monkeypatch, "plain", passes=3, tail=16)
    lm, whole = _walk(monkeypatch, "whole", passes=3, tail=16)
    assert not _whole(lm)[2]
    _equal(plain, whole)


def test_a_call_outside_the_record_takes_the_other_paths(monkeypatch):
    lm, _ = _walk(monkeypatch, "whole", passes=1)
    layer = next(m for m in lm.model.layers if not m.is_linear)
    attn = layer.self_attn
    c = q4.QSAKVCache(ratio=attn.ratio)
    # No cache rows, a dense depth, one row, nine rows: not covered.
    assert qsa_whole.state(attn, c, 4) is None
    lm2, args = _lm(mx.bfloat16)
    cache = lm2.make_cache()
    mx.eval(lm2(mx.random.randint(0, args.vocab_size, (1, 40)),
                cache=cache).logits)
    qc = next(c for c in cache if type(c) is q4.QSAKVCache)
    a2 = next(m for m in lm2.model.layers if not m.is_linear).self_attn
    assert qsa_whole.state(a2, qc, 4) is None


def test_a_block_outside_the_ranges_is_refused_by_the_record(monkeypatch):
    lm, _ = _walk(monkeypatch, "whole", passes=2)
    layer = next(m for m in lm.model.layers if not m.is_linear)
    table = vars(layer)["_verify_fused"][1]
    key, rec = next((k, r) for k, r in table.items()
                    if isinstance(r, tuple) and r[2] == "whole")
    L, cap = key[1][1], 64
    kb_shape, vb_shape, ib_shape, nbb = key[3], key[5], key[6], key[7]
    dt = key[4]
    h = mx.zeros(key[1], key[2])
    kt = mx.zeros(kb_shape[:2] + (cap, kb_shape[3]), dt)
    it = mx.zeros((1, cap, ib_shape[2]), dt)
    bt = mx.zeros((1, cap // 4 + qsa_whole.MAX_NEW, ib_shape[2]), dt)
    ins = [h, kt, kt, it, bt, mx.zeros(kb_shape, dt), mx.zeros(vb_shape, dt),
           mx.zeros(ib_shape, dt), mx.zeros((1, nbb, ib_shape[2]), dt)]
    good = [2400, 8, 2392, 2, 600]
    kq.fused_call(rec[0], ins + [mx.array(good, dtype=mx.int32)])
    for slot, bad in ((0, 2000), (1, cap - L + 1), (2, kb_shape[2] + 1),
                      (3, cap // 4 + 1), (4, -1)):
        vals = list(good)
        vals[slot] = bad
        with pytest.raises(ValueError, match="outside the range"):
            kq.fused_call(rec[0], ins + [mx.array(vals, dtype=mx.int32)])


# Dense depth: widths of 1 to 5 rows take the dense record, 6 to 8 the
# split form (mlx builds the unfused attention there).
DENSE = [(4, 4), (5, 2), (3, 3), (1, 1), (5, 5), (8, 3), (2, 1), (4, 2),
         (6, 6), (5, 3), (3, 1), (2, 2)]


@pytest.mark.parametrize("prompt,least", [(700, 12), (985, 4), (1010, 0),
                                          (1990, 4)])
def test_dense_depth_records_equal_the_plain_path(monkeypatch, prompt, least):
    # 985 and 1010 reach 1024 keys, where mlx changes its kernel and the
    # split form takes over; 1990 crosses into the sparse depth.
    kw = dict(rounds=DENSE * 3, tail=256, prompt=prompt)
    _, plain = _walk(monkeypatch, "plain", **kw)
    lm, whole = _walk(monkeypatch, "whole", **kw)
    n, used, refused = _whole(lm)
    assert not refused
    assert used >= least, (n, used)
    _equal(plain, whole)


def test_a_process_that_starts_at_dense_depth_reaches_the_sparse_depth(monkeypatch):
    # The dense record makes its copy kernel first; the sparse record's
    # kernels must still be built when a later verify is at sparse depth
    # (a served session that began on a short prompt raised KeyError
    # there).
    monkeypatch.setattr(qsa_whole, "_KERNELS", {})
    _walk(monkeypatch, "whole", rounds=DENSE * 2, tail=256, prompt=700)
    assert set(qsa_whole._KERNELS) == {"cat"}
    _, whole = _walk(monkeypatch, "whole", passes=2)
    assert {"cat", "finish", "queries"} <= set(qsa_whole._KERNELS)
    _, plain = _walk(monkeypatch, "plain", passes=2)
    _equal(plain, whole)


def test_dense_depth_over_1024_keys_takes_the_split_form(monkeypatch):
    # On the real model a 2 row verify over 1024 keys is not bit equal
    # under a capped dense record, so that depth has none.
    kw = dict(rounds=DENSE * 2, tail=256, prompt=1500)
    lm, whole = _walk(monkeypatch, "whole", **kw)
    assert lm.whole_calls == 0
    _, split = _walk(monkeypatch, "split", **kw)
    _equal(split, whole)


def test_dense_depth_in_float16(monkeypatch):
    kw = dict(rounds=DENSE * 2, tail=64, prompt=700, dtype=mx.float16)
    _, plain = _walk(monkeypatch, "plain", **kw)
    lm, whole = _walk(monkeypatch, "whole", **kw)
    assert lm.whole_calls >= 8
    _equal(plain, whole)


def test_one_row_rounds_at_sparse_depth(monkeypatch):
    # A verify of one row at each remainder of the offset, between wider
    # rounds, rollbacks and folds.
    rounds = [(1, 1), (4, 4), (1, 1), (1, 1), (3, 2), (1, 1), (8, 8),
              (1, 1), (1, 1), (1, 1), (1, 1), (5, 1)] * 5
    for dtype in (mx.bfloat16, mx.float16):
        kw = dict(rounds=rounds, tail=64, dtype=dtype)
        _, plain = _walk(monkeypatch, "plain", **kw)
        lm, whole = _walk(monkeypatch, "whole", **kw)
        n, used, refused = _whole(lm)
        assert not refused
        assert used >= 30, (n, used)
        _equal(plain, whole)


def test_a_copied_cache_keeps_its_rows(monkeypatch):
    # A prompt cache entry and its deep copy share arrays. The copy is
    # trimmed and continues; the entry must still hold its rows after.
    import copy

    def run(mode):
        monkeypatch.setenv("GMLX_Q4_KV_TAIL", "64")
        monkeypatch.setattr(q4, "_VERIFY_FUSED", mode != "plain")
        monkeypatch.setattr(q4, "_FUSED_WHOLE", mode == "whole")
        kq.env_cache(mode != "plain")
        lm, args = _lm(mx.bfloat16)
        mx.random.seed(7)
        entry = lm.make_cache()
        mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, PROMPT)),
                   cache=entry).logits)
        out = []

        def rounds(cache, n):
            for _ in range(n):
                for S, kept in ((4, 4), (5, 2), (3, 3), (4, 4)):
                    block = mx.random.randint(0, args.vocab_size, (1, S))
                    hid, _, sink = lm.speculative_verify_hidden(block, cache)
                    lm.rollback_speculative_cache(cache, sink, kept - 1, S)
                    mx.eval(hid)
                    out.append(np.array(hid.astype(mx.float32)))

        rounds(entry, 3)
        clone = copy.deepcopy(entry)
        for c in clone:
            if type(c) is q4.QSAKVCache:
                c.trim(9)
        rounds(clone, 3)
        rounds(entry, 3)
        return out

    _equal(run("plain"), run("whole"))


def test_a_handout_before_the_rollback_keeps_its_rows(monkeypatch):
    # A holder that took the tail arrays after a verify appended and
    # before its rollback still reads the rows it took.
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", "64")
    monkeypatch.setattr(q4, "_VERIFY_FUSED", True)
    monkeypatch.setattr(q4, "_FUSED_WHOLE", True)
    kq.env_cache(True)
    lm, args = _lm(mx.bfloat16)
    mx.random.seed(11)
    cache = lm.make_cache()
    mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, PROMPT)),
               cache=cache).logits)
    held = []
    for i in range(12):
        block = mx.random.randint(0, args.vocab_size, (1, 4))
        hid, _, sink = lm.speculative_verify_hidden(block, cache)
        mx.eval(hid, [c._kt for c in cache if type(c) is q4.QSAKVCache])
        if i >= 4:
            for c in cache:
                if type(c) is q4.QSAKVCache:
                    for a in (c._kt, c._vt):
                        held.append((a, c._tl, np.array(
                            a[..., :c._tl, :].astype(mx.float32))))
        lm.rollback_speculative_cache(cache, sink, 0, 4)
    assert held
    for a, tl, want in held:
        assert np.array_equal(
            np.array(a[..., :tl, :].astype(mx.float32)), want)
