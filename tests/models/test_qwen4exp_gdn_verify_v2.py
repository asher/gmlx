"""One-kernel GDN verify (v2): bits against the two-kernel form, and the
records rollback of qwen4exp at partial accepts.

The v2 kernel runs the float ops of the records-form scan and the gate
kernel in their order, so its output, records and replayed states equal
theirs bit for bit, and a qwen4exp verify, rollback and next step equal the
states form's.
"""

import os

import mlx.core as mx
import pytest

import gmlx.models.qwen4_exp.model as q4
import gmlx.upstream.gdn_patches as gp
from gmlx.models.qwen4_exp.mtp import Qwen4ExpSpecLM

from test_qwen4exp_mtp import _args, _randomize

pytestmark = [
    pytest.mark.skipif(
        bool(os.environ.get("KQUANT_FORCE_CPU")),
        reason="the fused GDN kernels are Metal-only"),
    pytest.mark.skipif(gp._gdn_verify_v2_kernel is None, reason="no Metal device"),
]


# -- kernel bits ---------------------------------------------------------


class _Gdn:
    """The attributes _gdn_verify_v2 reads, around random weights."""

    def __init__(self, Hk, Hv, Dk, Dv, K, tiled, sigmoid):
        self.num_k_heads, self.num_v_heads = Hk, Hv
        self.head_k_dim, self.head_v_dim = Dk, Dv
        self.conv_kernel_size = K
        self.conv_dim = 2 * Hk * Dk + Hv * Dv
        self.gdn_kv_tiled = bool(tiled)
        self.gdn_gate_sigmoid = bool(sigmoid)
        self.conv1d = _Box(weight=mx.random.normal((self.conv_dim, K, 1)) * 0.5)
        self.A_log = mx.random.uniform(-1, 1, (Hv,))
        self.dt_bias = mx.random.uniform(-1, 1, (Hv,))
        self.norm = _Box(weight=mx.random.uniform(0.5, 1.5, (Dv,)))
        self.out_proj = lambda y: y


class _Box:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Cache(list):
    """A two-slot cache without row lengths."""

    lengths = None


def _two_kernel(g, conv_input, ba, z, state):
    """The records-form scan and the gate kernel on a concatenated conv
    input: the gated output and the records."""
    B, S = z.shape[:2]
    Hk, Hv, Dk, Dv = g.num_k_heads, g.num_v_heads, g.head_k_dim, g.head_v_dim
    SGS, RPS = gp._GDN_VERIFY_SGS, gp._GDN_VERIFY_RPS
    cw = g.conv1d.weight[:, :, 0].T.astype(mx.float32)
    pre, _, rd, rk, rg = gp._gdn_verify_rec_kernel(
        inputs=[conv_input, cw, ba, g.A_log, g.dt_bias, state, S,
                *gp._no_records(), gp.replay_steps(0, B)],
        template=[("InT", conv_input.dtype), ("StT", state.dtype), ("Dk", Dk),
                  ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("K_size", g.conv_kernel_size),
                  ("SGS", SGS), ("RPS", RPS), ("TILED", gp._kv_heads_tiled(g)),
                  ("CONV_BF16", 1), ("REPLAY", 0)],
        grid=(32, SGS, B * Hv * (Dv // gp._GDN_VERIFY_ROWS)),
        threadgroup=(32, SGS, 1),
        output_shapes=[(B, S, Hv, Dv), (1,), (B, S, Hv, Dv), (B, S, Hv, Dk),
                       (B, S, Hv)],
        output_dtypes=[mx.float32] * 5)
    nrows = B * S * Hv
    (y,) = gp._gdn_verify_gate_kernel(
        inputs=[pre, z, g.norm.weight, nrows],
        template=[("InT", conv_input.dtype), ("Dv", Dv),
                  ("GATE_SIGMOID", int(g.gdn_gate_sigmoid))],
        grid=(32, nrows, 1), threadgroup=(32, 8, 1),
        output_shapes=[(B, S, Hv, Dv)], output_dtypes=[conv_input.dtype])
    return y, (rd, rk, rg)


SHAPES = [
    # Hk, Hv, Dk, Dv, tiled, sigmoid gate
    (16, 48, 128, 128, 1, 1),   # Qwen3.8-Flash-Next
    (4, 4, 128, 128, 0, 0),     # equal heads, silu gate
    (2, 6, 64, 128, 0, 0),      # grouped heads
    (2, 4, 32, 32, 1, 1),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("cap", [False, True])
def test_rounds_equal_the_two_kernel_form(shape, dtype, cap):
    """Rounds of several widths with partial accepts of every size, two
    rows that keep different counts: each round's output and records, and
    the state and conv window each rollback leaves, equal the two-kernel
    form's on the materialized state. Every other round the kernel computes
    b and a itself, from x and the b/a weight. In the fixed row capacity
    form the records and conv rows have 8 or 16 rows whatever the width,
    and their first S rows carry the same bits."""
    Hk, Hv, Dk, Dv, tiled, sigmoid = shape
    K, B = 4, 2
    mx.random.seed(3)
    g = _Gdn(Hk, Hv, Dk, Dv, K, tiled, sigmoid)
    g._verify_row_cap = cap
    cache = _Cache([(mx.random.normal((B, K - 1, g.conv_dim)) * 0.5).astype(dtype),
                    mx.random.normal((B, Hv, Dv, Dk)) * 0.05])
    conv_ref, state_ref = cache[0], cache[1]
    rounds = [(2, (1, 2)), (8, (3, 8)), (3, (2, 1)), (5, (0, 0)), (16, (16, 5)),
              (9, (1, 9)), (4, (4, 4)), (8, (7, 2))]
    for i, (S, kept) in enumerate(rounds):
        qkv = (mx.random.normal((B, S, g.conv_dim)) * 0.5).astype(dtype)
        # A width that is no multiple of the GEMV's 64-wide step.
        x = mx.random.normal((B, S, 200)).astype((mx.float32, dtype)[i % 4 // 2])
        w = (mx.random.normal((2 * Hv, 200)) * 0.07).astype(mx.bfloat16)
        ba = gp._f16_head_gemv(x, w)
        z = mx.random.normal((B, S, Hv, Dv)).astype(dtype)
        sink: list = []
        if i % 2:
            y = gp._gdn_verify_v2(g, x, qkv, None, z, None, cache, sink, ba_weight=w)
        else:
            y = gp._gdn_verify_v2(g, x, qkv, ba, z, None, cache, sink)
        assert y is not None
        conv, rec = sink[-1][9], sink[-1][11]
        assert isinstance(conv, gp.GdnConvRows) and isinstance(rec, gp.GdnRecords)

        conv_input = mx.concatenate([conv_ref, qkv], axis=1)
        y_ref, rec_ref = _two_kernel(g, conv_input, ba, z, state_ref)
        assert mx.array_equal(y.reshape(y_ref.shape), y_ref)
        rows = (8 if S <= 8 else 16) if cap else S
        assert conv.rows.shape == (B, rows, g.conv_dim)
        assert (conv.rows is qkv) is not cap
        assert mx.array_equal(conv.rows[:, :S], qkv)
        for got, want in zip(rec.records, rec_ref, strict=True):
            assert got.shape[1] == rows
            assert mx.array_equal(got[:, :S], want)
        assert mx.array_equal(rec.base, state_ref)
        assert mx.array_equal(conv.base, conv_ref)
        # Left as fully accepted.
        assert mx.array_equal(cache[0], conv_input[:, -(K - 1):])
        assert mx.array_equal(cache[1], gp.gdn_replay_state(state_ref, rec_ref, S))

        if kept != (0, 0):
            steps = mx.array(kept, dtype=mx.int32)
            gp.set_gdn_replay(cache, rec.base, rec.records, steps)
            gp.set_gdn_conv(cache, conv.base, conv.rows, steps)
        else:
            # A reader replaced both states: the pending ones are void.
            cache[0] = mx.array(cache[0]) + 0
            cache[1] = mx.array(cache[1]) + 0
            steps = mx.array([S, S], dtype=mx.int32)
        state_ref = gp.gdn_replay_state(state_ref, rec_ref, steps)
        idx = steps[:, None] + mx.arange(K - 1, dtype=mx.int32)[None, :]
        conv_ref = mx.take_along_axis(conv_input, idx[..., None], axis=1)
        mx.eval(state_ref, conv_ref)
        assert mx.array_equal(cache[1], state_ref)
        assert mx.array_equal(cache[0], conv_ref)


def test_declines_what_it_does_not_cover():
    g = _Gdn(2, 4, 32, 32, 4, 1, 1)
    C = g.conv_dim

    def call(S, conv_dtype=mx.float32, lengths=None, Dv=32):
        g.head_v_dim = Dv
        cache = _Cache([mx.zeros((1, 3, C), dtype=conv_dtype), mx.zeros((1, 4, Dv, 32))])
        cache.lengths = lengths
        qkv = mx.zeros((1, S, C))
        return gp._gdn_verify_v2(g, qkv, qkv, mx.zeros((1, S, 8)),
                                 mx.zeros((1, S, 4, Dv)), None, cache, [])

    assert call(4) is not None
    assert call(17) is None                               # more rows than its memory holds
    assert call(4, conv_dtype=mx.bfloat16) is None        # conv state of another dtype
    assert call(4, lengths=mx.array([4])) is None         # ragged rows
    assert call(4, Dv=24) is None                         # rows that do not tile


def test_lever(monkeypatch):
    """On from the M3 generation up, where its 256-thread groups are
    validated; the lever forces either way."""
    import gmlx.load.dtypes as dtypes

    monkeypatch.delenv("GMLX_GDN_VERIFY_V2", raising=False)
    for gen, want in ((13, False), (14, False), (15, True), (17, True), (0, True)):
        monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda g=gen: g)
        assert gp.gdn_verify_v2_default() is want
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 17)
    monkeypatch.setenv("GMLX_GDN_VERIFY_V2", "0")
    assert gp.gdn_verify_v2_default() is False
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 13)
    monkeypatch.setenv("GMLX_GDN_VERIFY_V2", "1")
    assert gp.gdn_verify_v2_default() is True


# -- qwen4exp verify and rollback ----------------------------------------


def _lm(v2):
    args = _args()
    lm = Qwen4ExpSpecLM(args)
    _randomize(lm)
    counts = q4.prepare_runtime(lm)
    assert counts["gdn_fused_verify"] > 0
    for m in lm.modules():
        if isinstance(m, q4.GatedDeltaNet):
            m._verify_v2 = v2
    return lm, args


def _gdn_caches(sink):
    return [e["cache"] for e in sink if e["kind"] == "gdn"]


def _walk(v2, rounds, prompt_len=9):
    """Prefill, then verify rounds of (rows, kept rows) with a rollback
    after each, a decode step, and one more verify. Returns every hidden,
    the step's logits and the GDN states at the end."""
    lm, args = _lm(v2)
    mx.random.seed(5)
    vocab = args.vocab_size
    prompt = mx.random.randint(0, vocab, (1, prompt_len))
    cache = lm.make_cache()
    mx.eval(lm(prompt, cache=cache).logits)
    out = []
    gdn = []
    for S, kept in rounds:
        block = mx.random.randint(0, vocab, (1, S))
        hid, _, sink = lm.speculative_verify_hidden(block, cache)
        gdn = _gdn_caches(sink)
        assert gdn
        for e in sink:
            if e["kind"] == "gdn":
                assert (e.get("records") is not None) == v2
                assert (e["inter"] is None) == v2
        lm.rollback_speculative_cache(cache, sink, kept - 1, S)
        if v2 and kept < S:
            for c in gdn:
                assert c._gdn_replay[0] is c[1] and c._gdn_conv[0] is c[0]
        mx.eval(hid)
        out.append(hid)
    step = lm(mx.random.randint(0, vocab, (1, 1)), cache=cache).logits
    hid, _, sink = lm.speculative_verify_hidden(mx.random.randint(0, vocab, (1, 4)), cache)
    states = [a for c in gdn for a in (c[0], c[1])]
    mx.eval(step, hid, states)
    return out + [step, hid] + states


ROUNDS = [
    [(4, 1), (4, 3), (4, 2)],
    [(2, 1), (8, 5), (8, 8), (3, 1)],
    [(16, 9), (8, 1), (16, 16), (2, 1)],
]


@pytest.mark.parametrize("rounds", ROUNDS)
def test_rounds_match_the_states_form(rounds):
    """Partial accepts of every size, full accepts among them: hiddens,
    the decode step after the last rollback, the next verify and the GDN
    states match the two-kernel states form bit for bit."""
    got, want = _walk(True, rounds), _walk(False, rounds)
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        assert a.shape == b.shape
        assert mx.array_equal(a, b)


@pytest.mark.parametrize("kept", [1, 2, 3])
def test_rollback_matches_a_prefix_forward(kept):
    """A verify block of 4 rolled back to its first rows leaves the GDN
    caches as a forward over the prompt and the kept tokens does, and the
    next decode step agrees."""
    lm, args = _lm(True)
    vocab = args.vocab_size
    mx.random.seed(1)
    prompt = mx.random.randint(0, vocab, (1, 9))
    block = mx.random.randint(0, vocab, (1, 4))
    nxt = mx.random.randint(0, vocab, (1, 1))

    cache_a = lm.make_cache()
    lm(prompt, cache=cache_a)
    _, _, sink = lm.speculative_verify_hidden(block, cache_a)
    lm.rollback_speculative_cache(cache_a, sink, kept - 1, 4)
    logits_a = lm(nxt, cache=cache_a).logits

    cache_b = lm.make_cache()
    lm(mx.concatenate([prompt, block[:, :kept]], axis=1), cache=cache_b)
    logits_b = lm(nxt, cache=cache_b).logits
    mx.eval(logits_a, logits_b)
    assert mx.abs(logits_a - logits_b).max() < 2e-3
    gdn = set(map(id, _gdn_caches(sink)))
    n = 0
    for ca, cb in zip(cache_a, cache_b):
        if id(ca) not in gdn:
            continue
        n += 1
        for x, y in ((ca[0], cb[0]), (ca[1], cb[1])):
            assert x.shape == y.shape
            assert mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max() < 2e-4
    assert n == len(gdn)


def test_a_gated_round_restores_the_pending_states():
    """A verify built and discarded (a snapshot restore) leaves the pending
    replay and conv window in place for the next verify."""
    import gmlx.spec.speculative as sp

    lm, args = _lm(True)
    mx.random.seed(2)
    vocab = args.vocab_size
    cache = lm.make_cache()
    mx.eval(lm(mx.random.randint(0, vocab, (1, 9)), cache=cache).logits)
    block = mx.random.randint(0, vocab, (1, 4))
    _, _, sink = lm.speculative_verify_hidden(block, cache)
    lm.rollback_speculative_cache(cache, sink, 1, 4)
    snap = sp._cache_snapshot(cache)
    assert snap is not None
    lm.speculative_verify_hidden(block, cache)
    sp._cache_restore(snap)
    for c in _gdn_caches(sink):
        assert c._gdn_replay[0] is c[1] and c._gdn_conv[0] is c[0]
