"""Inkling 2 forward against a float32 numpy reference written from
llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp.

The tiny model is loaded from a float32 GGUF at float32 activations. Its
prompt is longer than the sliding window and the relative extent, and
the log scale floor is short, so every distance rule and the query scale
are in play.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
import inkling2_tiny as T  # noqa: E402
from f32_exact import f32_exact  # noqa: E402

L = 40
IDS = np.random.default_rng(7).integers(2, len(T.tokens()), L).tolist()
EPS = 1e-6
FLOOR, ALPHA, QK_CLIP, REL_CLIP = 8, 0.1, 22.627417, 8.0
SCALE, SOFTCAP = 4.0, 30.0


def _rms(x, w=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return y if w is None else y * w


def _mix(blocks, cur, query):
    """[L, nq, D]: softmax over depth of the rms-normed states."""
    states = _rms(np.stack([*blocks, cur], axis=1))          # [L, n, D]
    q = query.reshape(-1, T.D)
    scores = states @ q.T / np.sqrt(T.D)                      # [L, n, nq]
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return np.einsum("lnq,lnd->lqd", p, states)


def _clip(x, scale, threshold):
    if scale is not None:
        x = x * scale
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + 1e-6
    return x * np.clip(threshold / norm, 0.0, 1.0)


def _convolve(x, w, dynamic):
    """Causal conv of width TAPS, w [C, TAPS] with w[:, -1] on the current
    token; dynamic [L, heads * TAPS] taps with lag 0 on the current token."""
    n, c = x.shape
    window = np.concatenate([np.zeros((T.TAPS - 1, c), np.float32), x])
    out = sum(window[k:k + n] * w[:, k] for k in range(T.TAPS))
    if dynamic is not None:
        heads = dynamic.shape[1] // T.TAPS
        taps = dynamic.reshape(n, heads, T.TAPS)
        for lag in range(T.TAPS):
            past = window[T.TAPS - 1 - lag:T.TAPS - 1 - lag + n]
            out = out + (past.reshape(n, heads, c // heads)
                         * taps[:, :, lag:lag + 1]).reshape(n, c)
    return (x + out) / np.sqrt((2 if dynamic is not None else 1) * T.TAPS + 1)


def _control(v, layer, c, cur, msg):
    p = f"blk.{layer}.loop.{c}."
    g = np.concatenate([_rms(cur), _rms(msg)], axis=-1) @ v[p + "down"].T
    gain = np.tanh(g @ v[p + "gain"].T)
    forget = np.tanh(g @ v[p + "forget"].T)
    return cur - forget * cur + gain * msg


def _attention(q, k, val, r, sink, table, swa):
    """q [L, H, HD], k/val [L, KV, HD], r [L, H, R], sink [L, H]. The bias
    R . B[d] - sink joins the scores after the 1/sqrt(HD) scale, and the
    softmax has an extra zero logit."""
    n = q.shape[0]
    table = np.concatenate([table, np.zeros((1, T.R), np.float32)])
    out = np.zeros_like(q)
    group = T.HEADS // T.KV
    for i in range(n):
        for h in range(T.HEADS):
            kh = h // group
            logits = []
            for j in range(i + 1):
                d = i - j
                if swa and d >= T.WINDOW:
                    logits.append(-np.inf)
                    continue
                row = table[d if d < T.EXTENT else T.EXTENT]
                bias = r[i, h] @ row - sink[i, h]
                logits.append(q[i, h] @ k[j, kh] / np.sqrt(T.HD) + bias)
            logits = np.array(logits)
            m = max(logits.max(), 0.0)
            e = np.exp(logits - m)
            out[i, h] = (e[:, None] * val[:i + 1, kh]).sum(0) / (
                e.sum() + np.exp(-m))
    return out


def _moe(v, p, x):
    scores = 1 / (1 + np.exp(-(x @ v[p + "ffn_gate_inp.weight"].T)))
    select = scores + v[p + "exp_probs_b.bias"]
    out = np.zeros_like(x)
    gu_w, down_w = v[p + "ffn_gate_up_exps.weight"], v[p + "ffn_down_exps.weight"]
    for t in range(x.shape[0]):
        top = np.argsort(-select[t], kind="stable")[:T.USED]
        w = scores[t, top] / scores[t, top].sum() * SCALE
        for e, we in zip(top, w):
            gu = gu_w[e] @ x[t]
            g, u = gu[:T.FF], gu[T.FF:]
            act = 4 * np.tanh(g / 4) / (1 + np.exp(-g)) * 25 * np.tanh(u / 25)
            out[t] += we * (down_w[e] @ act)
    return out


def _layer(v, il, cur, blocks, tau, msg=None):
    p = f"blk.{il}."
    swa = T.SWA[il]
    inp_sa = cur
    if il:
        mixed = _mix(blocks, cur, v[p + "attn_res_query.weight"])
        inputs = [mixed[:, 0 if swa else j] for j in range(3)]
    else:
        inputs = [cur] * 3
    if msg is not None:
        inputs = [_control(v, il, c, x, msg[:, j])
                  for j, (c, x) in enumerate(zip(("q", "k", "v"), inputs))]
    if swa:
        inputs = [_rms(x, v[p + "attn_norm.weight"]) for x in inputs]
    else:
        inputs = [_rms(x, v[p + f"attn_{n}_inp_norm.weight"])
                  for x, n in zip(inputs, "qkv")]
    lin = lambda x, n: x @ v[p + n + ".weight"].T  # noqa: E731
    q = lin(inputs[0], "attn_q")
    k = _convolve(lin(inputs[1], "attn_k"), v[p + "attn_k_conv.weight"],
                  None if swa else lin(inputs[1], "attn_k_conv_proj"))
    val = _convolve(lin(inputs[2], "attn_v"), v[p + "attn_v_conv.weight"],
                    None if swa else lin(inputs[2], "attn_v_conv_proj"))
    r = lin(inputs[0], "attn_rel")
    n = cur.shape[0]
    q = _clip(q.reshape(n, T.HEADS, T.HD), v[p + "attn_q_clip.weight"], QK_CLIP)
    k = _clip(k.reshape(n, T.KV, T.HD), None, QK_CLIP)
    val = _rms(val.reshape(n, T.KV, T.HD))
    r = _clip(r.reshape(n, T.HEADS, T.R), v[p + "attn_rel_clip.weight"],
              REL_CLIP)
    if not swa:
        q = q * tau[:, None, None]
        r = r * tau[:, None, None]
    sink = lin(inputs[0], "attn_sink_proj")
    out = _attention(q, k, val, r, sink, v[p + "attn_rel_bias.weight"], swa)
    out = out.reshape(n, -1) / (1 + np.exp(-lin(inputs[0], "attn_gate")))
    out = _convolve(lin(out, "attn_output"), v[p + "attn_post_conv.weight"],
                    None)
    if il % T.BLOCK == 0:
        blocks.append(inp_sa)
        cur = out
    else:
        cur = inp_sa + out
    ffn_inp = cur
    x = _mix(blocks, cur, v[p + "ffn_res_query.weight"])[:, 0]
    if msg is not None:
        x = _control(v, il, "ffn", x, msg[:, 3])
    x = _moe(v, p, _rms(x, v[p + "ffn_norm.weight"]))
    x = _convolve(x, v[p + "ffn_post_conv.weight"], None)
    return ffn_inp + x


def reference_logits(v, ids):
    v = {k: a.astype(np.float64) for k, a in v.items()}
    pos = np.arange(len(ids), dtype=np.float64)
    tau = 1 + ALPHA * np.log(np.maximum(1.0, (pos + 1) / FLOOR))
    cur = _rms(v["token_embd.weight"][ids], v["token_embd_norm.weight"])
    blocks = []
    for il in range(T.START):
        cur = _layer(v, il, cur, blocks, tau)
    prelude, prelude_blocks = cur, list(blocks)
    for pas in range(T.PASSES):
        msgs = None
        if pas:
            bank = blocks[len(prelude_blocks) + 1:]
            bank_out = cur
            cur, blocks = prelude, list(prelude_blocks)
        for il in range(T.START, T.END):
            if pas:
                msgs = _mix(bank, bank_out, v[f"blk.{il}.loop_msg_query"])
            cur = _layer(v, il, cur, blocks, tau, msgs)
    for il in range(T.END, T.LAYERS):
        cur = _layer(v, il, cur, blocks, tau)
    x = _rms(_mix(blocks, cur, v["output_res_query.weight"])[:, 0],
             v["output_norm.weight"])
    logits = x @ v["output.weight"].T
    logits = SOFTCAP * np.tanh(logits / SOFTCAP)
    return logits[:, :len(T.tokens())]


def _with_msg_queries(v):
    """``v`` plus the per-layer message queries the reference reads."""
    out = dict(v)
    q, kv = v["loop_res_query.weight"], v["loop_kv_res_query.weight"]
    for j in range(T.END - T.START):
        out[f"blk.{T.START + j}.loop_msg_query"] = np.stack(
            [q[2 * j], kv[2 * j], kv[2 * j + 1], q[2 * j + 1]])
    return out


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import gmlx

    old = os.environ.get("GMLX_ACTIVATION_DTYPE")
    os.environ["GMLX_ACTIVATION_DTYPE"] = "float32"
    try:
        path = tmp_path_factory.mktemp("inkling2_ref") / "tiny.gguf"
        T.write(path, "root", T.values(seed=3), experts_q8=False)
        m = gmlx.load_model(str(path), verbose=False)[0]
    finally:
        if old is None:
            os.environ.pop("GMLX_ACTIVATION_DTYPE", None)
        else:
            os.environ["GMLX_ACTIVATION_DTYPE"] = old
    return m


@pytest.fixture(scope="module")
def reference():
    return reference_logits(_with_msg_queries(T.values(seed=3)), IDS)


def _logits(model, ids, cache=None):
    return np.array(model(mx.array([ids]), cache=cache)[0].astype(mx.float32))


@f32_exact
def test_prefill_matches_the_reference(model, reference):
    got = _logits(model, IDS)
    np.testing.assert_allclose(got, reference, atol=2e-4, rtol=2e-4)


@f32_exact
def test_split_queries_match_the_reference(model, reference, monkeypatch):
    # A small bias budget splits each attention call into query blocks.
    import gmlx.models.inkling2.model as im

    monkeypatch.setattr(im, "_BIAS_ELEMENTS", 3 * T.HEADS * L)
    got = _logits(model, IDS)
    np.testing.assert_allclose(got, reference, atol=2e-4, rtol=2e-4)


@f32_exact
def test_chunked_prefill_and_decode_match_the_reference(model, reference):
    # Chunks of 7, then one token at a time: the sliding caches wrap, and
    # the distances come from the ring order.
    cache = model.make_cache()
    rows = []
    for i0 in range(0, 21, 7):
        rows.append(_logits(model, IDS[i0:i0 + 7], cache))
    for i in range(21, L):
        rows.append(_logits(model, IDS[i:i + 1], cache))
    got = np.concatenate(rows)
    np.testing.assert_allclose(got, reference, atol=2e-4, rtol=2e-4)


def test_visit_commits_leave_the_logits_unchanged(model, monkeypatch):
    # Decode and verify widths commit every few visits; the GPU then runs
    # those visits early, and the logits must not move.
    from gmlx.models.inkling2 import model as M

    def run(commit):
        monkeypatch.setattr(M, "_COMMIT_VISITS", commit)
        cache = model.make_cache()
        rows = [_logits(model, IDS[:7], cache)]
        rows += [_logits(model, IDS[i:i + 4], cache) for i in (7, 11)]
        rows += [_logits(model, IDS[i:i + 1], cache) for i in (15, 16)]
        return np.concatenate(rows)

    calls = []
    real = mx.async_eval
    monkeypatch.setattr(mx, "async_eval", lambda *a: calls.append(1) or real(*a))
    on = run(3)
    assert len(calls) == 5 * (len(T.visits()) // 3)
    np.testing.assert_array_equal(on, run(0))


def test_an_eval_forward_leaves_out_the_gradient_rules(model, monkeypatch):
    # The row kernels' gradient rules are graph nodes only training needs.
    # A forward of a model in eval mode builds without them, to the bit.
    import io

    from gmlx.models.inkling2 import row_kernels

    if not row_kernels.enabled():
        pytest.skip("the row kernels run on the GPU")

    def run():
        cache = model.make_cache()
        _logits(model, IDS[:7], cache)
        out = model(mx.array([IDS[7:8]]), cache=cache)
        f = io.StringIO()
        mx.export_to_dot(f, out)
        return np.array(out.astype(mx.float32)), f.getvalue().count("CustomTransforms")

    assert not model.training
    got, n_eval = run()
    real = row_kernels.held
    monkeypatch.setattr(row_kernels, "held", lambda grad=True: real(True))
    want, n_rules = run()
    assert n_eval == 0 < n_rules
    np.testing.assert_array_equal(got, want)


def test_route_counts_name_each_visit_kind(model):
    # A route check at depth reads these tags: the model builds its own
    # bias, so the wrapper alone counts every call as stock.
    from gmlx.upstream import attn_hd512

    def tags():
        out = {}
        for k, n in attn_hd512.route_counts().items():
            if k[0].startswith("inkling2_"):
                out[k[0]] = out.get(k[0], 0) + n
        return out

    before = tags()
    cache = model.make_cache()
    model(mx.array([IDS[:7]]), cache=cache)
    model(mx.array([IDS[7:8]]), cache=cache)
    grown = {k: n - before.get(k, 0) for k, n in tags().items()
             if n > before.get(k, 0)}
    # Ring caches are not in time order, so sliding layers take the dense
    # route; full layers take the banded one.
    assert set(grown) == {"inkling2_banded_full", "inkling2_dense_sliding"}


@f32_exact
def test_a_ragged_batch_matches_each_row_alone(model):
    # Left padding shifts the batch rows: the relative distances, the
    # query scale of the full layers and the convolution tails must all
    # ignore it.
    from gmlx._vendor.mlx_lm.generate import BatchGenerator, generate_step
    from gmlx._vendor.mlx_lm.models.cache import make_prompt_cache

    def greedy(x):
        return mx.argmax(x, axis=-1)

    rng = np.random.default_rng(0)
    prompts = [rng.integers(2, len(T.tokens()), n).tolist() for n in (5, 23, 37)]
    alone = []
    for ids in prompts:
        steps = generate_step(mx.array(ids), model, max_tokens=8, sampler=greedy,
                              prompt_cache=make_prompt_cache(model))
        alone.append([(int(t), np.array(lp)) for t, lp in steps])

    gen = BatchGenerator(model, sampler=greedy)
    uids = gen.insert(prompts, [8] * len(prompts))
    batched = {u: [] for u in uids}
    while responses := gen.next_generated():
        for r in responses:
            batched[r.uid].append((int(r.token), np.array(r.logprobs)))
    gen.close()

    for want, uid in zip(alone, uids):
        got = batched[uid]
        assert [t for t, _ in got] == [t for t, _ in want]
        for (_, a), (_, b) in zip(got, want):
            np.testing.assert_allclose(a, b, atol=1e-4)


@f32_exact
def test_the_prompt_cache_checkpoint_round_trips(model, reference):
    # The model takes the checkpoint tier: whole convolution states and
    # sliding windows beside the block chain of the full layers.
    from gmlx._vendor.mlx_vlm.apc import APCManager
    from gmlx.cache.snapshot import ckpt_layout, ckpt_lookup, ckpt_store
    from gmlx.spec.engine import _ckpt_active

    assert _ckpt_active(model, "exact", 16)
    manager = APCManager(num_blocks=256, block_size=16)
    try:
        cache = model.make_cache()
        _logits(model, IDS[:32], cache)
        assert ckpt_store(manager, IDS[:32], cache, skeleton_disk=False) == 32
        layout = tuple(ckpt_layout(model.make_cache(), 16))
        warm, prefix = ckpt_lookup(manager, IDS, layout=layout)
        assert prefix == 32
        got = _logits(model, IDS[prefix:], warm)
    finally:
        manager.close()
    np.testing.assert_allclose(got, reference[prefix:], atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@f32_exact
def test_speculative_rollback_matches_a_plain_run(model, reference, accepted):
    # Verify a 4-token block past the sliding window, keep accepted + 1
    # of its tokens, then continue: the logits must match a run that never
    # saw the rejected tokens.
    from types import SimpleNamespace

    from gmlx.models.inkling2.mtp import Inkling2SpecLM
    from gmlx.spec.helpers import _buffer_mtp_target_cache

    def run(ids, cache):
        h = spec.model(mx.array([ids]), cache)
        return np.array(spec.head(h)[0].astype(mx.float32))

    spec, plain = model, type(model)
    spec.__class__ = Inkling2SpecLM
    try:
        cache = spec.make_cache()
        run(IDS[:30], cache)
        drafter = SimpleNamespace(config=SimpleNamespace(block_size=4))
        _buffer_mtp_target_cache(cache, drafter, 4)
        _, _, sink = spec.speculative_verify_hidden(mx.array([IDS[30:34]]),
                                                    cache)
        spec.rollback_speculative_cache(cache, sink, accepted, 4)
        keep = 31 + accepted
        got = run(IDS[keep:], cache)
    finally:
        spec.__class__ = plain
    np.testing.assert_allclose(got, reference[keep:], atol=2e-4, rtol=2e-4)


def test_recorded_visits_match_the_plain_call(model, monkeypatch):
    # Verify and decode visits replay kq records from a layout's third use,
    # past the sliding window and across rollbacks. The logits match the
    # plain call's to the bit.
    from types import SimpleNamespace

    kq = pytest.importorskip("mlx_kquant")
    if not (hasattr(kq, "fuse_graph") and hasattr(kq, "env_cache")):
        pytest.skip("kq build without records")

    from gmlx.models.inkling2 import records
    from gmlx.models.inkling2.mtp import Inkling2SpecLM
    from gmlx.spec.helpers import _buffer_mtp_target_cache

    def logits(h):
        return np.array(spec.head(h)[0].astype(mx.float32))

    def run(on):
        monkeypatch.setattr(records, "_ON", [on])
        records.invalidate()
        cache = spec.make_cache()
        spec.model(mx.array([IDS[:20]]), cache)
        _buffer_mtp_target_cache(
            cache, SimpleNamespace(config=SimpleNamespace(block_size=4)), 4)
        out, pos = [], 20
        for accepted in (1, 0, 2, 3):
            h, _, sink = spec.speculative_verify_hidden(
                mx.array([IDS[pos:pos + 4]]), cache)
            out.append(logits(h))
            spec.rollback_speculative_cache(cache, sink, accepted, 4)
            pos += accepted + 1
        for t in IDS[pos:pos + 4]:
            out.append(logits(spec.model(mx.array([[t]]), cache)))
        return out

    # The record machinery on any kq, direct replay or not, and unblocked
    # by a weight swap in an earlier test (feeder_common.swapped_weights).
    monkeypatch.setattr(records, "DIRECT_ONLY", [False])
    monkeypatch.setattr(records, "_BLOCKED", [False])
    spec, plain = model, type(model)
    spec.__class__ = Inkling2SpecLM
    try:
        want = run(False)
        replayed = records.STATS["replayed"]
        got = run(True)
    finally:
        spec.__class__ = plain
    assert records.STATS["replayed"] > replayed
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_a_cut_verify_keeps_the_rows_it_keeps(model, monkeypatch):
    # Under verify_row_keep the rows past the keep route to the experts of
    # the last kept row. The kept rows' logits, and every step after a
    # rollback into them, match the plain verify's to the bit.
    from types import SimpleNamespace

    import gmlx.models.inkling2.model as im
    from gmlx.models.inkling2.mtp import Inkling2SpecLM
    from gmlx.spec.helpers import _buffer_mtp_target_cache

    seen: list = []
    real = im.MoE._fused_down

    def fused_down(self, x, inds):
        if inds.shape[-2] == 4:
            seen.append(np.array(inds).reshape(4, -1))
        return real(self, x, inds)

    monkeypatch.setattr(im.MoE, "_fused_down", fused_down)

    def logits(h):
        return np.array(spec.head(h)[0].astype(mx.float32))

    def run(cut):
        cache = spec.make_cache()
        spec.model(mx.array([IDS[:20]]), cache)
        _buffer_mtp_target_cache(
            cache, SimpleNamespace(config=SimpleNamespace(block_size=4)), 4)
        out, ids, pos = [], [], 20
        for keep, accepted in ((1, 1), (0, 0), (2, 1), (3, 3)):
            block = mx.array([IDS[pos:pos + 4]])
            seen.clear()
            if cut:
                with spec.verify_row_keep(mx.array(keep)):
                    h, _, sink = spec.speculative_verify_hidden(block, cache)
            else:
                h, _, sink = spec.speculative_verify_hidden(block, cache)
            out.append(logits(h)[:keep + 1])
            ids.append((keep, list(seen)))
            spec.rollback_speculative_cache(cache, sink, accepted, 4)
            pos += accepted + 1
        for t in IDS[pos:pos + 4]:
            out.append(logits(spec.model(mx.array([[t]]), cache)))
        return out, ids

    spec, plain = model, type(model)
    spec.__class__ = Inkling2SpecLM
    try:
        # BF16 experts take no fused gathers: cut rows reuse row keep's ids.
        assert not spec.row_cut_saves(4)
        want, plain_ids = run(False)
        got, cut_ids = run(True)
    finally:
        spec.__class__ = plain
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)
    moved = False
    for (keep, cut), (_, whole) in zip(cut_ids, plain_ids):
        assert cut and len(cut) == len(whole)
        for c, w in zip(cut, whole):
            np.testing.assert_array_equal(c[:keep + 1], w[:keep + 1])
            assert (c[keep + 1:] == c[keep]).all()
            moved |= not (w[keep + 1:] == w[keep]).all()
    assert moved


@pytest.mark.parametrize("experts", ["Q4_0", "Q8_0"])
def test_a_cut_verify_reads_no_experts_past_the_keep(tmp_path, monkeypatch,
                                                     experts):
    # On K-quant expert stacks the rows past the keep name no expert (kq
    # absent=True), on the qf (Q4_0) and generic (Q8_0) kernels. The kept
    # rows' logits, and every step after a rollback into them, match the
    # plain verify's to the bit.
    from types import SimpleNamespace

    import mlx_kquant as kq

    import gmlx
    from gmlx.models.inkling2.mtp import Inkling2SpecLM
    from gmlx.spec.helpers import _buffer_mtp_target_cache

    if "absent (bool)" not in (kq.gather_qmv_mix_ns_kq.__doc__ or ""):
        pytest.skip("this kq build has no absent expert ids")
    path = tmp_path / "tiny.gguf"
    T.write(path, "root", T.values(seed=3), experts_type=experts)
    spec = gmlx.load_model(str(path), verbose=False)[0]
    spec.__class__ = Inkling2SpecLM
    assert spec.row_cut_saves(4)
    calls: list = []
    real = kq.gather_qmv_mix_ns_kq

    def mix_ns(*a, **kw):
        calls.append(kw.get("absent", False))
        return real(*a, **kw)

    monkeypatch.setattr(kq, "gather_qmv_mix_ns_kq", mix_ns)

    def logits(h):
        return np.array(spec.head(h)[0].astype(mx.float32))

    def run(cut):
        cache = spec.make_cache()
        spec.model(mx.array([IDS[:20]]), cache)
        _buffer_mtp_target_cache(
            cache, SimpleNamespace(config=SimpleNamespace(block_size=4)), 4)
        out, pos = [], 20
        for keep, accepted in ((1, 1), (0, 0), (2, 1), (3, 3)):
            block = mx.array([IDS[pos:pos + 4]])
            if cut:
                with spec.verify_row_keep(mx.array(keep)):
                    h, _, sink = spec.speculative_verify_hidden(block, cache)
            else:
                h, _, sink = spec.speculative_verify_hidden(block, cache)
            out.append(logits(h)[:keep + 1])
            spec.rollback_speculative_cache(cache, sink, accepted, 4)
            pos += accepted + 1
        for t in IDS[pos:pos + 4]:
            out.append(logits(spec.model(mx.array([[t]]), cache)))
        return out

    want = run(False)
    assert calls and not any(calls)
    calls.clear()
    got = run(True)
    assert any(calls)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_drafter_taps_are_visit_inputs(model):
    # Tap i is the input of visit i, and tap len(visits) the normed final
    # hidden state. A pass start reads the prelude output, as the first
    # core visit does.
    from gmlx.models.inkling2.mtp import Inkling2SpecLM

    n = len(T.visits())
    core = T.END - T.START
    starts = [T.START, T.START + core, T.START + 2 * core]
    ids = [0, *starts, n - 1, n]
    caps: list = []
    h = model.model(mx.array([IDS]), capture_layer_ids=ids, hidden_sink=caps)
    assert len(caps) == len(ids)
    first, *others = (np.array(caps[ids.index(s)]) for s in starts)
    for other in others:
        np.testing.assert_array_equal(first, other)
    np.testing.assert_array_equal(np.array(caps[-1]), np.array(h))
    assert not np.allclose(np.array(caps[-2]), np.array(caps[-1]))

    plain = type(model)
    model.__class__ = Inkling2SpecLM
    try:
        with pytest.raises(ValueError, match="outside"):
            model.set_dflash_capture([1, n + 1])
        model.set_dflash_capture(ids)
        normed, packed = model._forward(mx.array([IDS]), None)
        assert packed.shape[-1] == (len(ids) + 1) * normed.shape[-1]
        np.testing.assert_array_equal(np.array(model._dflash_trunk(packed)),
                                      np.array(normed))
    finally:
        object.__setattr__(model, "_dflash_capture", None)
        model.__class__ = plain


@pytest.mark.parametrize("n_queries,n_keys,padded", [
    (40, 40, False), (40, 40, True), (5, 60, False), (1, 70, True)])
def test_the_banded_mask_matches_the_dense_one(model, n_queries, n_keys,
                                                padded):
    # Full and sliding layers, a fresh prefill, a chunk at depth and a
    # decode step; the banded route must give the dense route's output.
    rng = np.random.default_rng(n_keys)
    B, H, D = 2, T.HEADS, T.HD
    q = mx.array(rng.standard_normal((B, H, n_queries, D)), dtype=mx.float32)
    k = mx.array(rng.standard_normal((B, T.KV, n_keys, D)), dtype=mx.float32)
    v = mx.array(rng.standard_normal((B, T.KV, n_keys, D)), dtype=mx.float32)
    pad = None
    if padded:
        keep = np.ones((B, 1, n_queries, n_keys), dtype=bool)
        keep[1, ..., :3] = False  # row 1 is left-padded by three keys
        pad = mx.array(keep)
    for lyr in model.model.layers[:2]:  # one sliding, one full layer
        attn = lyr.self_attn
        r = mx.array(rng.standard_normal((B, n_queries, H, attn.rel_length)),
                     dtype=mx.float32)
        sink = mx.array(rng.standard_normal((B, n_queries, H)),
                        dtype=mx.float32)
        got = attn._attend_banded(q, k, v, r, sink, None, pad)
        want = attn._attend_dense(q, k, v, r, sink, None, pad)
        if attn.window is None:
            np.testing.assert_array_equal(np.array(got), np.array(want))
        else:
            # The sliding layer drops the keys outside its window, so the
            # kernel sums fewer terms; the values agree to rounding.
            np.testing.assert_allclose(np.array(got), np.array(want),
                                       atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("filled", [3, 30])
def test_the_fused_attention_matches_the_dense_route(model, filled,
                                                     monkeypatch):
    # At verify width, attention over the real caches runs one fused
    # kernel: a full layer's growing cache, and a sliding layer's ring
    # before and after it wraps. It must give the dense route's output.
    from gmlx.cache.kv_cache import KVCache, RotatingKVCache
    from gmlx.models.inkling2 import row_kernels

    if not row_kernels.enabled():
        pytest.skip("the fused kernel runs on the GPU")
    calls = []
    real = row_kernels.attend
    monkeypatch.setattr(row_kernels, "attend",
                        lambda *a: calls.append(a) or real(*a))
    rng = np.random.default_rng(filled)
    B, L, D = 1, 4, 128

    def rows(*shape, dtype=mx.bfloat16):
        return mx.array(rng.standard_normal(shape), dtype=dtype)

    for lyr in model.model.layers[:2]:  # one sliding, one full layer
        attn = lyr.self_attn
        cache = (RotatingKVCache(max_size=attn.window) if attn.window
                 else KVCache())
        cache.update_and_fetch(rows(B, T.KV, filled, D),
                               rows(B, T.KV, filled, D))
        k, v = cache.update_and_fetch(rows(B, T.KV, L, D),
                                      rows(B, T.KV, L, D))
        q = rows(B, T.HEADS, L, D)
        r = rows(B, L, T.HEADS, attn.rel_length, dtype=mx.float32)
        sink = rows(B, L, T.HEADS, dtype=mx.float32)
        got = attn.attend(q, k, v, r, sink, cache, None)
        f32 = [a.astype(mx.float32) for a in (q, k, v)]
        want = attn._attend_dense(*f32, r, sink, cache, None)
        np.testing.assert_allclose(np.array(got.astype(mx.float32)),
                                   np.array(want), atol=2e-2, rtol=2e-2)
    assert len(calls) == 2


def test_the_compiled_mask_follows_changing_shapes():
    """The mask build is compiled shapeless, so its first trace serves
    every later shape: batch, query rows, key count and a padding mask that
    adds axes. Each call must equal the eager build."""
    from gmlx.models.inkling2.model import _masker

    extent, n_heads, rank = 6, 3, 4
    rng = np.random.default_rng(0)
    table = mx.array(rng.standard_normal((extent + 1, rank)), mx.float32)

    def eager(r, sink, dist, valid):
        scores = r @ table.T - sink[..., None]
        index = mx.where((dist >= 0) & (dist < extent), dist, extent)
        B, L, H, _ = r.shape
        index = mx.broadcast_to(index[None, :, None, :],
                                (B, L, H, dist.shape[1]))
        bias = mx.take_along_axis(scores, index, axis=-1)
        return mx.where(valid, bias.transpose(0, 2, 1, 3),
                        -mx.inf).astype(mx.bfloat16)

    run = _masker(extent, mx.bfloat16)
    for B, L, K, padded in [(1, 1, 5, False), (2, 3, 9, True),
                            (1, 4, 12, False), (3, 1, 2, True)]:
        r = mx.array(rng.standard_normal((B, L, n_heads, rank)), mx.float32)
        sink = mx.array(rng.standard_normal((B, L, n_heads)), mx.float32)
        dist = (mx.arange(K)[None, :] * -1 + K - L
                + mx.arange(L)[:, None]).astype(mx.int32)
        valid = dist >= 0
        if padded:
            pad = mx.array(rng.random((B, 1, L, K)) > 0.3)
            valid = valid & pad
        got, want = run(r, sink, table, dist, valid), eager(r, sink, dist,
                                                            valid)
        assert got.shape == want.shape == (B, n_heads, L, K)
        assert mx.array_equal(got, want, equal_nan=True).item()


def test_a_new_bias_table_reaches_the_mask(model):
    # The padded table is cached per weight array: a replaced weight (a
    # training step, a load) must not read the old one.
    attn = next(layer.self_attn for layer in model.model.layers)
    first = attn._padded_table()
    assert attn._padded_table() is first
    old = attn.rel_bias
    try:
        attn.rel_bias = old + 1.0
        np.testing.assert_array_equal(np.array(attn._padded_table()[:-1]),
                                      np.array(old + 1.0))
    finally:
        attn.rel_bias = old
    np.testing.assert_array_equal(np.array(attn._padded_table()[:-1]),
                                  np.array(old))


# The attention projections as the 4-bit files hold them: Q8_0, with the
# sink and the conv projections left float.
_Q8_ATTN = ("attn_q.weight", "attn_gate.weight", "attn_rel.weight",
            "attn_k.weight", "attn_v.weight")


def _load_tiny(tmp_path, name, monkeypatch, fuse, **write_args):
    import gmlx

    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float32")
    monkeypatch.setenv("GMLX_GEMV_FUSE", "1" if fuse else "0")
    path = tmp_path / f"{name}.gguf"
    T.write(path, "root", T.values(seed=3), experts_q8=False, **write_args)
    return gmlx.load_model(str(path), verbose=False)[0]


def _groups(model):
    return [{slot: names for slot, (names, _) in
             layer.self_attn._row_groups.items()
             if layer.self_attn.get(slot) is not None}
            for layer in model.model.layers]


def test_row_fusion_joins_same_codec_projections(tmp_path, monkeypatch):
    """On a Q8_0 layout, q, gate and rel run as one mat-vec on every layer
    and k with v on sliding layers. The float sink and conv projections
    stay alone. Decode steps match an unfused load exactly; a prefill
    chunk may round differently, since its kernel follows the row count."""
    fused = _load_tiny(tmp_path, "fused", monkeypatch, True, q8=_Q8_ATTN)
    plain = _load_tiny(tmp_path, "plain", monkeypatch, False, q8=_Q8_ATTN)
    q = ("q_proj", "gate_proj", "rel_proj")
    want = [{"_q_fused": q, "_kv_fused": ("k_proj", "v_proj")} if swa
            else {"_q_fused": q} for swa in T.SWA]
    assert _groups(fused) == want
    assert _groups(plain) == [{}] * T.LAYERS

    import gmlx.models.inkling2.model as im

    calls = []
    split = im.split_rows
    monkeypatch.setattr(im, "split_rows",
                        lambda *a: calls.append(1) or split(*a))
    caches = fused.make_cache(), plain.make_cache()
    a, b = (_logits(m, IDS[:20], c) for m, c in zip((fused, plain), caches))
    # q splits on every visit. k and v split on sliding layers outside the
    # loop passes, where each physical layer runs once; the passes give k
    # and v separate inputs.
    assert len(calls) == len(T.visits()) + sum(T.SWA)
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
    for t in IDS[20:]:
        a, b = (_logits(m, [t], c) for m, c in zip((fused, plain), caches))
        np.testing.assert_array_equal(a, b)


def test_clearing_the_fused_slots_falls_back_to_the_members(
        tmp_path, monkeypatch):
    """An adapter install clears the fused slots (drop_fused_children), and
    the members, which are views of the fused rows, serve alone."""
    from gmlx.load.modules import drop_fused_children

    model = _load_tiny(tmp_path, "float", monkeypatch, True)
    assert all(g.get("_q_fused") for g in _groups(model))
    before = _logits(model, IDS)
    assert drop_fused_children(model) > 0
    assert _groups(model) == [{}] * T.LAYERS
    np.testing.assert_allclose(_logits(model, IDS), before, atol=1e-5,
                               rtol=1e-5)


@pytest.mark.skipif(not mx.metal.is_available(), reason="kq kernels need Metal")
def test_the_skinny_router_routes_a_verify_like_decode(monkeypatch):
    """With GMLX_INKLING2_SKINNY_ROUTER, each row of a 4-row router call
    has the logits of that row alone, to float32 rounding."""
    import mlx.nn as nn

    import gmlx.models.inkling2.model as im

    monkeypatch.setattr(im, "_SKINNY_ROUTER", True)
    rng = np.random.default_rng(3)
    gate = nn.Linear(256, 128, bias=False)
    gate.weight = mx.array(rng.standard_normal((128, 256)) * 0.05,
                           dtype=mx.float32)
    x = mx.array(rng.standard_normal((1, 4, 256)), dtype=mx.float32)
    rows = im._gate_logits(gate, x)
    for i in range(4):
        one = im._gate_logits(gate, x[:, i:i + 1])
        np.testing.assert_allclose(np.array(rows[:, i:i + 1]),
                                   np.array(one), rtol=0, atol=2e-6)
