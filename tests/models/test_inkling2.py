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


def test_prefill_matches_the_reference(model, reference):
    got = _logits(model, IDS)
    np.testing.assert_allclose(got, reference, atol=2e-4, rtol=2e-4)


def test_split_queries_match_the_reference(model, reference, monkeypatch):
    # A small bias budget splits each attention call into query blocks.
    import gmlx.models.inkling2.model as im

    monkeypatch.setattr(im, "_BIAS_ELEMENTS", 3 * T.HEADS * L)
    got = _logits(model, IDS)
    np.testing.assert_allclose(got, reference, atol=2e-4, rtol=2e-4)


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
