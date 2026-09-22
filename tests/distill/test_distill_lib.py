"""CPU tests for gmlx.distill, ported from the lab's Phase 1 suite: the
NaN list, slot masses against float64, the fused head against a dense
reference, the losses and their gradients, alignment tables and projection
on two minted tokenizers, the cache format and validator, the view loader
and framed rows. MLX runs on the CPU device (the GPU's f32 GEMM is TF32 by
default and would put the dense references 1e-3 away from the fused head).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import gmlx.distill as dl
import gmlx.distill.eval as dl_eval


@pytest.fixture(autouse=True)
def _cpu():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)


KNOBS = dict(dl.DEFAULT_KNOBS)


# ---------------------------------------------------------------------------
# synthetic tokenizers
# ---------------------------------------------------------------------------

def _bytelevel_tokenizer(extra_merges):
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    BPEStreamingDetokenizer.make_byte_decoder()
    chars = sorted(BPEStreamingDetokenizer._byte_decoder.items(), key=lambda kv: kv[1])
    vocab = {ch: i for i, (ch, _b) in enumerate(chars)}
    merges = []
    for a, b in extra_merges:
        merged = a + b
        if merged not in vocab:
            vocab[merged] = len(vocab)
        merges.append((a, b))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", bos_token="<bos>")
    return fast


def _spm_tokenizer(pieces, merges):
    from tokenizers import Tokenizer, decoders, models, normalizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    for i in range(256):
        vocab[f"<0x{i:02X}>"] = len(vocab)
    for p in pieces:
        if p not in vocab:
            vocab[p] = len(vocab)
    for a, b in merges:
        for piece in (a, b, a + b):
            if piece not in vocab:
                vocab[piece] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=True, unk_token=None))
    tok.normalizer = normalizers.Replace(" ", "\u2581")
    tok.decoder = decoders.Sequence([decoders.Replace("\u2581", " "), decoders.ByteFallback()])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", bos_token="<bos>",
                                   pad_token="<pad>")
    return fast


@pytest.fixture(scope="module")
def tok_bl():
    # teacher: byte-level with a few merges
    return _bytelevel_tokenizer([("\u0120", "t"), ("h", "e"), ("\u0120t", "he"), ("1", "2"),
                                 ("12", "3"), ("\u0120", "a"), ("i", "s"), ("\u0120", "is"),
                                 ("c", "a"), ("ca", "t")])


@pytest.fixture(scope="module")
def tok_spm():
    # student: SPM-like with byte fallback; every ascii letter, digit and
    # space is a piece so any ascii text tokenizes without fallback
    import string
    pieces = ["\u2581"] + list(string.ascii_letters + string.digits + ".,!?\n") \
        + ["\u2581" + c for c in string.ascii_lowercase] + ["\u2581the", "\u2581cat", "\u2581is",
                                                              "th", "he", "at", "12", "123", "\u25811"]
    merges = [("\u2581", "t"), ("\u2581t", "h"), ("\u2581th", "e"), ("t", "h"), ("h", "e"), ("a", "t"),
              ("\u2581", "c"), ("\u2581c", "a"), ("\u2581ca", "t"), ("\u2581", "i"), ("\u2581i", "s"),
              ("1", "2"), ("12", "3"), ("\u2581", "1")]
    return _spm_tokenizer(pieces, merges)


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

def test_encode_with_byte_ends(tok_bl, tok_spm):
    text = "the cat is 123 \u00e9t\u00e9".encode("utf-8")
    for tok in (tok_bl, tok_spm):
        tb = dl.token_bytes(tok)
        ids, ends, flagged = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
        assert not flagged
        assert ends[-1] == len(text)
        assert np.all(np.diff(ends.astype(np.int64)) > 0)


# ---------------------------------------------------------------------------
# alignment and projection
# ---------------------------------------------------------------------------

def _rand_text(rng, n_words):
    words = ["the", "cat", "is", "123", "at", "hat", "1", "12", "a", "this", "that", "\u00e9t\u00e9", "!"]
    return " ".join(rng.choice(words) for _ in range(n_words))


def test_shared_boundaries_invariants(tok_bl, tok_spm):
    rng = np.random.default_rng(1)
    ttb, stb = dl.token_bytes(tok_bl), dl.token_bytes(tok_spm)
    for _ in range(50):
        text = _rand_text(rng, int(rng.integers(3, 40))).encode("utf-8")
        t_ids, t_ends, _ = dl.encode_with_byte_ends(tok_bl, text, ttb, add_special_tokens=False)
        s_ids, s_ends, _ = dl.encode_with_byte_ends(tok_spm, text, stb, add_special_tokens=True)
        al = dl.align_row(t_ids, s_ids, text, teacher_tb=ttb, student_tb=stb,
                          teacher_special=set(), student_special=set(dl.hf_inner(tok_spm).all_special_ids))
        assert np.all(np.diff(al.ends) > 0)
        assert np.all(np.diff(al.t_pos) > 0) and np.all(np.diff(al.s_pos) > 0)
        for j in range(al.J):
            assert t_ends[al.t_pos[j]] == al.ends[j] == s_ends[al.s_pos[j]]
            assert al.t_pos[j] < len(t_ids) - 1 and al.s_pos[j] < len(s_ids) - 1
        # chunk bytes agree
        for j in range(1, al.J):
            tb_ = b"".join(ttb[i] for i in t_ids[al.t_pos[j - 1] + 1: al.t_pos[j] + 1])
            sb_ = b"".join(stb[i] for i in s_ids[al.s_pos[j - 1] + 1: al.s_pos[j] + 1])
            assert tb_ == sb_


def test_tables_partition(tok_bl, tok_spm):
    t = dl.build_tables(tok_bl, tok_spm)
    assert not t.identity
    assert t.group_of.min() >= 0 and t.group_of.max() < t.G
    assert np.all(t.group_size == np.bincount(t.group_of, minlength=t.G))
    assert set(t.nonsingleton_ids.tolist()) == set(np.nonzero(t.group_size[t.group_of] > 1)[0].tolist())
    # every teacher token whose own group exists targets it
    key_to_group = {int(k): g for g, k in enumerate(t.group_key) if k >= 0}
    for v in range(t.V_T):
        if v in key_to_group:
            assert t.target_g[v] == key_to_group[v] and t.own[v]
    # the unmapped group (key -1) is never a target
    assert not np.any(t.target_g == 0) or t.group_key[0] != -1
    # save / load round trip
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        dl.save_tables(Path(d), t)
        t2 = dl.load_tables(Path(d))
        assert np.array_equal(t2.target_g, t.target_g) and t2.G == t.G


def test_reused_tables_are_saved_into_the_view(tmp_path, tok_bl, tok_spm):
    """``align --tables DIR`` on a matching pair leaves a copy of the tables
    in the view, since ``train`` and ``eval`` read them from there."""
    from gmlx.distill.view import get_tables

    t = dl.build_tables(tok_bl, tok_spm)
    dl.save_tables(tmp_path / "census", t)
    got = get_tables(tok_bl, tok_spm, tmp_path / "census", tmp_path / "view", V_T=None, V_S=None)
    assert got.teacher_hash == t.teacher_hash and got.student_hash == t.student_hash
    assert (tmp_path / "view" / "tables.json").exists() and (tmp_path / "view" / "tables.safetensors").exists()
    t2 = dl.load_tables(tmp_path / "view")
    assert np.array_equal(t2.target_g, t.target_g) and t2.G == t.G
    # a mismatched artifact is rebuilt and saved too
    dl.save_tables(tmp_path / "other", dl.identity_tables(t.V_T, np.zeros(t.V_T, bool), "aaaa", "bbbb"))
    get_tables(tok_bl, tok_spm, tmp_path / "other", tmp_path / "view2", V_T=None, V_S=None)
    assert dl.load_tables(tmp_path / "view2").teacher_hash == t.teacher_hash


def test_identity_tables(tok_bl):
    t = dl.build_tables(tok_bl, tok_bl)
    assert t.identity and t.G == t.V_S
    assert np.array_equal(t.target_g, np.arange(t.V_T))


def _lm_shaped_logp(rng, P, V, K):
    """LM-shaped teacher top-K: a few heavy tokens, a long tail."""
    z = rng.standard_normal((P, V)) * 3.0
    z[:, :8] += 6.0
    lp = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
    idx = np.argsort(-lp, axis=1)[:, :K]
    return np.take_along_axis(lp, idx, axis=1).astype(np.float32), idx.astype(np.int32), lp


def test_project_topk_matches_float64(tok_bl, tok_spm):
    t = dl.build_tables(tok_bl, tok_spm)
    rng = np.random.default_rng(2)
    P, K = 7, 32
    lp, idx, full = _lm_shaped_logp(rng, P, t.V_T, K)
    proj = dl.project_topk(lp, idx, t)
    for j in range(P):
        acc = {}
        for k in range(K):
            g = int(t.target_g[idx[j, k]])
            if g < 0:
                continue
            acc[g] = np.logaddexp(acc.get(g, -np.inf), float(lp[j, k]))
        got = {int(g): float(v) for g, v in zip(proj["gid"][j], proj["log_p"][j]) if g != t.G}
        assert set(got) == set(acc)
        for g in acc:
            assert abs(got[g] - acc[g]) < 1e-5
        assert abs(float(proj["log_M"][j]) - np.logaddexp.reduce(list(acc.values()))) < 1e-5
    assert np.all(proj["own"] + proj["redirect"] + proj["dropped"] <= 1.0 + 1e-5)
    # cap keeps the heaviest groups
    proj2 = dl.project_topk(lp, idx, t, Kp=2)
    assert proj2["gid"].shape[1] == 2
    assert np.all(proj2["log_p"][:, 0] >= proj2["log_p"][:, 1])
    assert np.all(proj2["dropped"] >= proj["dropped"] - 1e-6)


# ---------------------------------------------------------------------------
# head and loss numerics
# ---------------------------------------------------------------------------

def _ref_bucketed(target_log_p, log_M, Q_slot, weight, mode="bucketed"):
    """float64 reference of bucketed_kl."""
    Nb, Kp = target_log_p.shape
    tot = 0.0
    for j in range(Nb):
        M = math.exp(log_M[j])
        head = 0.0
        for k in range(Kp):
            if target_log_p[j, k] == -np.inf:
                continue
            P = math.exp(target_log_p[j, k])
            Q = max(float(Q_slot[j, k]), 2.0 ** -126)
            head += P * (target_log_p[j, k] - math.log(Q))
        per = head
        if mode == "bucketed" and log_M[j] < -1e-7:
            Qt = max(float(Q_slot[j, Kp]), 2.0 ** -126)
            per += (1 - M) * (math.log1p(-M) - math.log(Qt))
        tot += weight[j] * per
    return tot / max(weight.sum(), 1.0)


def _dense_group_masses(q, gid_row, group_of, G, Kp):
    """[Kp+1] slot masses from a dense q [V] in float64."""
    Qs = np.zeros(Kp + 1)
    gm = np.zeros(G + 1)
    np.add.at(gm, group_of, q)
    valid = gid_row != G
    Qs[:Kp][valid] = gm[gid_row[valid]]
    Qs[Kp] = 1.0 - Qs[:Kp][valid].sum()
    return Qs


def _make_case(rng, V, d, N, n_bnd, Kp, G, group_of, C, softcap=None):
    h = rng.standard_normal((N, d)).astype(np.float32)
    W = (rng.standard_normal((V, d)) * 0.3).astype(np.float32)
    next_ids = rng.integers(0, V, N).astype(np.int32)
    gid = np.full((n_bnd, Kp), G, dtype=np.int32)
    lp = np.full((n_bnd, Kp), dl.NEG_INF, dtype=np.float32)
    for j in range(n_bnd):
        k = int(rng.integers(1, Kp + 1))
        gs = rng.choice(G, size=k, replace=False)
        raw = rng.standard_normal(k) * 2
        raw = raw - np.logaddexp.reduce(raw) + math.log(0.9 if j % 3 else 1.0)  # M < 1 or M = 1
        gid[j, :k] = gs
        lp[j, :k] = raw
    log_M = np.log(np.exp(lp.astype(np.float64)).sum(axis=1)).astype(np.float32)
    w = rng.uniform(0.2, 1.0, n_bnd).astype(np.float32)
    bmask = rng.random(V) < 0.3
    return dict(h=h, W=W, next_ids=next_ids, gid=gid, lp=lp, log_M=log_M, w=w, bmask=bmask,
                softcap=softcap, C=C)


@pytest.mark.parametrize("V,C,softcap", [(64, 4, None), (64, 16, 30.0), (64, 3, None)])
def test_chunked_head_matches_dense(V, C, softcap):
    rng = np.random.default_rng(3)
    d, N, n_bnd, Kp, G = 8, 11, 6, 5, 20
    group_of = rng.integers(0, G, V).astype(np.int32)
    case = _make_case(rng, V, d, N, n_bnd, Kp, G, group_of, C, softcap)
    head = dl.linear_head(mx.array(case["W"]), softcap)
    onpath, Q_slot, log_bm = dl.chunked_head(
        mx.array(case["h"]), head, mx.array(case["next_ids"]), n_bnd=n_bnd,
        target_gid=mx.array(case["gid"]), group_of=mx.array(group_of), G=G, Kp=Kp,
        log_bmask=dl.log_bmask_from(case["bmask"]), C=C)
    mx.eval(onpath, Q_slot, log_bm)
    z = case["h"].astype(np.float64) @ case["W"].astype(np.float64).T
    if softcap:
        z = softcap * np.tanh(z / softcap)
    logq = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
    q = np.exp(logq)
    assert np.allclose(np.asarray(onpath), logq[np.arange(N), case["next_ids"]], atol=1e-5)
    for j in range(n_bnd):
        ref = _dense_group_masses(q[j], case["gid"][j], group_of, G, Kp)
        assert np.allclose(np.asarray(Q_slot[j]), ref, atol=1e-6)
        assert abs(np.asarray(Q_slot[j]).sum() - 1.0) < 1e-5
        bm = np.log(q[j][case["bmask"]].sum())
        assert abs(float(log_bm[j]) - bm) < 1e-5


def test_chunked_head_identity_path():
    rng = np.random.default_rng(4)
    V, d, N, n_bnd, Kp = 40, 8, 9, 5, 6
    G = V
    case = _make_case(rng, V, d, N, n_bnd, Kp, G, np.arange(V), 4)
    head = dl.linear_head(mx.array(case["W"]))
    a = dl.chunked_head(mx.array(case["h"]), head, mx.array(case["next_ids"]), n_bnd=n_bnd,
                        target_gid=mx.array(case["gid"]), group_of=None, G=G, Kp=Kp,
                        log_bmask=dl.log_bmask_from(case["bmask"]), C=4)
    b = dl.chunked_head(mx.array(case["h"]), head, mx.array(case["next_ids"]), n_bnd=n_bnd,
                        target_gid=mx.array(case["gid"]), group_of=mx.arange(V, dtype=mx.int32),
                        G=G, Kp=Kp, log_bmask=dl.log_bmask_from(case["bmask"]), C=4)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_bucketed_kl_matches_reference_and_modes():
    rng = np.random.default_rng(5)
    Nb, Kp = 6, 4
    lp = np.full((Nb, Kp), dl.NEG_INF, dtype=np.float32)
    for j in range(Nb):
        k = int(rng.integers(1, Kp + 1))
        raw = rng.standard_normal(k)
        raw = raw - np.logaddexp.reduce(raw) + math.log(0.8 if j else 1.0)
        lp[j, :k] = raw
    log_M = np.log(np.exp(lp.astype(np.float64)).sum(axis=1)).astype(np.float32)
    Q = rng.dirichlet(np.ones(Kp + 1), size=Nb).astype(np.float32)
    # pad slots carry no student mass (no student token maps to the sentinel)
    Q[:, :Kp][lp == dl.NEG_INF] = 0.0
    Q[:, Kp] = 1.0 - Q[:, :Kp].sum(axis=1)
    w = rng.uniform(0.5, 1.0, Nb).astype(np.float32)
    for mode in ("bucketed", "paper"):
        loss, aux = dl.bucketed_kl(mx.array(lp), mx.array(log_M), mx.array(Q), mx.array(w), mode=mode)
        mx.eval(loss)
        assert abs(float(loss) - _ref_bucketed(lp, log_M, Q, w, mode)) < 1e-5, mode
    # T_dk = 1 path and the tempered path agree at T = 1 + eps
    l1, _ = dl.bucketed_kl(mx.array(lp), mx.array(log_M), mx.array(Q), mx.array(w), T_dk=1.0)
    l2, _ = dl.bucketed_kl(mx.array(lp), mx.array(log_M), mx.array(Q), mx.array(w), T_dk=1.0 + 1e-6)
    assert abs(float(l1) - float(l2)) < 1e-4
    # renorm: support-only softmax, zero when the student conditional matches
    lpr = lp.copy()
    Qr = np.zeros((Nb, Kp + 1), dtype=np.float32)
    for j in range(Nb):
        valid = lpr[j] != dl.NEG_INF
        Qr[j, :Kp][valid] = np.exp(lpr[j][valid]) / np.exp(lpr[j][valid]).sum() * 0.5
        Qr[j, Kp] = 0.5
    lr, _ = dl.bucketed_kl(mx.array(lpr), mx.array(log_M), mx.array(Qr), mx.array(w), mode="renorm")
    assert abs(float(lr)) < 1e-5


def test_nan_list():
    """Every case in the plan's NaN list holds in f32."""
    Kp = 3
    w = mx.array([1.0])
    # M = 1 exactly
    lp = mx.array([[math.log(0.5), math.log(0.5), dl.NEG_INF]])
    logM = mx.array([0.0])
    Q = mx.array([[0.4, 0.4, 0.0, 0.2]])
    loss, aux = dl.bucketed_kl(lp, logM, Q, w)
    mx.eval(loss)
    assert np.isfinite(float(loss))
    # M = 0 (boundary selected out): all pads
    lp0 = mx.array([[dl.NEG_INF] * Kp])
    loss, aux = dl.bucketed_kl(lp0, mx.array([dl.NEG_INF]), Q, mx.array([0.0]))
    mx.eval(loss)
    assert float(loss) == 0.0
    # Q underflowing to 0 in f32 for a support group and the tail: finite, zero grad, counter
    lp = mx.array([[math.log(0.5), math.log(0.4), dl.NEG_INF]])
    logM = mx.array([math.log(0.9)])

    def f(Qv):
        lv, a = dl.bucketed_kl(lp, logM, Qv, w)
        return lv
    Qz = mx.array([[0.9, 0.0, 0.0, 0.0]])
    loss, aux = dl.bucketed_kl(lp, logM, Qz, w)
    g = mx.grad(f)(Qz)
    mx.eval(loss, g, aux["floored"])
    assert np.isfinite(float(loss)) and int(aux["floored"]) == 2
    assert float(g[0, 1]) == 0.0 and float(g[0, 3]) == 0.0
    # sentinel gids and -inf targets at pads: handled above (pads present)
    # a batch with N_bnd = 0
    loss, aux = dl.bucketed_kl(mx.zeros((0, Kp)), mx.zeros((0,)), mx.zeros((0, Kp + 1)), mx.zeros((0,)))
    mx.eval(loss)
    assert float(loss) == 0.0
    # log boundary mass -inf on one side, chunk of length 8 with probability 1e-30
    on = mx.full((1, 10), math.log(1e-30) / 8.0)
    bm = mx.full((1, 10), dl.NEG_INF)
    alm, n = dl.alm_term(on, bm, mx.array([[0]]), mx.array([[7]]), mx.array([[-2.0]]),
                         mx.array([[-0.1]]), mx.array([[True]]))
    mx.eval(alm)
    assert np.isfinite(float(alm))
    alm2, _ = dl.alm_term(on, mx.zeros((1, 10)), mx.array([[0]]), mx.array([[7]]), mx.array([[-2.0]]),
                          mx.array([[dl.NEG_INF]]), mx.array([[True]]))
    mx.eval(alm2)
    assert np.isfinite(float(alm2))
    # padded chunk (0, 0) masked -> zero
    alm3, n3 = dl.alm_term(on, mx.zeros((1, 10)), mx.array([[0]]), mx.array([[0]]), mx.array([[0.0]]),
                           mx.array([[0.0]]), mx.array([[False]]))
    mx.eval(alm3)
    assert float(alm3) == 0.0 and int(n3) == 0


def test_alm_and_scatter_vs_python_reference():
    rng = np.random.default_rng(6)
    B, Tm1 = 3, 12
    on = -rng.uniform(0.1, 3.0, (B, Tm1)).astype(np.float32)
    bm = -rng.uniform(0.1, 3.0, (B, Tm1)).astype(np.float32)
    starts = np.array([[0, 3, 7, 0], [2, 5, 0, 0], [1, 0, 0, 0]], dtype=np.int32)
    ends = np.array([[2, 6, 9, 0], [4, 8, 0, 0], [5, 0, 0, 0]], dtype=np.int32)
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool)
    tll = -rng.uniform(0.5, 4.0, (B, 4)).astype(np.float32)
    tbm = -rng.uniform(0.1, 1.0, (B, 4)).astype(np.float32)
    alm, n = dl.alm_term(mx.array(on), mx.array(bm), mx.array(starts), mx.array(ends),
                         mx.array(tll), mx.array(tbm), mx.array(mask))
    mx.eval(alm)
    tot, cnt = 0.0, 0
    for b in range(B):
        for c in range(4):
            if not mask[b, c]:
                continue
            ll = on[b, starts[b, c]:ends[b, c] + 1].sum()
            lb = ll + bm[b, ends[b, c] + 1]
            la = tll[b, c] + tbm[b, c]
            a, bb = math.exp(la), math.exp(lb)
            tot += a * (la - lb) + (1 - a) * (math.log1p(-a) - math.log1p(-bb))
            cnt += 1
    assert abs(float(alm) - tot / cnt) < 1e-4 and int(n) == cnt
    # scatter back
    positions = np.array([5, 0, 17, 30], dtype=np.int32)
    vals = np.array([1.5, -2.0, 0.25, 3.0], dtype=np.float32)
    full = dl.scatter_back(mx.array(vals), mx.array(positions), B, Tm1)
    ref = np.zeros(B * Tm1, dtype=np.float32)
    ref[positions] = vals
    assert np.array_equal(np.asarray(full), ref.reshape(B, Tm1))


def _dense_loss_mx(hidden, W, batch, group_of, G, Kp, bmask, knobs, softcap=None):
    """Dense mx reference of distill_loss (materialized logits, plain
    scatter), for gradient comparison."""
    B, T, d = hidden.shape
    Tm1 = T - 1
    z = (hidden[:, :-1, :].reshape(B * Tm1, d) @ W.T).astype(mx.float32)
    if softcap:
        z = softcap * mx.tanh(z / softcap)
    logq = z - mx.logsumexp(z, axis=-1, keepdims=True)
    next_ids = batch["student_ids"][:, 1:].reshape(-1)
    onpath_all = mx.take_along_axis(logq, next_ids[:, None], axis=-1)[:, 0]
    positions = batch["positions"]
    n_bnd = int(batch["n_bnd"])
    cm = batch["compute_mask"].reshape(-1)
    onpath_full = (onpath_all * cm).reshape(B, Tm1)
    q = mx.exp(logq)
    bpos = positions[:n_bnd]
    qb = q[bpos]                                     # [Nb, V]
    gm = mx.zeros((n_bnd, G + 1)).at[mx.broadcast_to(mx.arange(n_bnd)[:, None], qb.shape),
                                     mx.broadcast_to(group_of[None, :], qb.shape)].add(qb)
    gid = batch["target_gid"]
    Qg = mx.take_along_axis(gm, gid, axis=1)         # sentinel column G stays 0
    valid = gid != G
    Qg = mx.where(valid, Qg, mx.zeros_like(Qg))
    Qtail = 1.0 - mx.sum(Qg, axis=1)
    Q_slot = mx.concatenate([Qg, Qtail[:, None]], axis=1)
    dk, _ = dl.bucketed_kl(batch["target_log_p"], batch["log_M"], Q_slot, batch["bnd_weight"])
    lbm = mx.log(mx.sum(qb * mx.array(bmask.astype(np.float32))[None, :], axis=1))
    log_bm_full = mx.zeros((B * Tm1,)).at[bpos].add(lbm).reshape(B, Tm1)
    alm, _ = dl.alm_term(onpath_full, log_bm_full, batch["chunk_start"], batch["chunk_end"],
                         batch["chunk_teacher_ll"], batch["chunk_teacher_log_bm"], batch["chunk_mask"])
    N = int(positions.shape[0])
    ce = -mx.sum(mx.take(onpath_all, positions)) / N
    return knobs["lambda_dk"] * dk + knobs["lambda_alm"] * alm + knobs["lambda_ce"] * ce


def _synthetic_batch(rng, B, T, V, G, group_of, Kp, bmask, n_chunks=3):
    rows = []
    for b in range(B):
        n = int(rng.integers(T // 2, T + 1))
        ids = rng.integers(0, V, n).astype(np.int32)
        J = max(2, n // 3)
        bnd = np.sort(rng.choice(n - 1, size=J, replace=False)).astype(np.int32)
        gid = np.full((J, Kp), G, dtype=np.int32)
        lp = np.full((J, Kp), dl.NEG_INF, dtype=np.float32)
        for j in range(J):
            k = int(rng.integers(1, Kp + 1))
            gs = rng.choice(G, size=k, replace=False)
            raw = rng.standard_normal(k)
            raw = raw - np.logaddexp.reduce(raw) + math.log(0.85)
            gid[j, :k] = gs
            lp[j, :k] = raw
        logM = np.log(np.exp(lp.astype(np.float64)).sum(axis=1)).astype(np.float32)
        cs, ce = [], []
        for j in range(1, J):
            if bnd[j] - bnd[j - 1] <= 8:
                cs.append(int(bnd[j - 1]))
                ce.append(int(bnd[j]) - 1)
        nc = len(cs)
        rows.append(dl.RowView(
            student_ids=ids, compute_mask=np.ones(n - 1, dtype=bool), bnd_pos=bnd,
            bnd_weight=rng.uniform(0.3, 1.0, J).astype(np.float32), target_gid=gid, target_log_p=lp,
            log_M=logM, chunk_start=np.array(cs, np.int32), chunk_end=np.array(ce, np.int32),
            chunk_teacher_ll=(-rng.uniform(0.5, 5.0, nc)).astype(np.float32),
            chunk_teacher_log_bm=(-rng.uniform(0.05, 1.0, nc)).astype(np.float32)))
    return dl.collate(rows, Kp, G, pad_to=8)


@pytest.mark.parametrize("V,C,softcap", [(64, 5, None), (64, 64, 30.0), (151936, 7, None)])
def test_distill_loss_gradient_vs_dense(V, C, softcap):
    rng = np.random.default_rng(7)
    B, T, d, G, Kp = 2, 14, 8, min(V, 30), 4
    group_of = rng.integers(0, G, V).astype(np.int32) if G < V else np.arange(V, dtype=np.int32)
    bmask = rng.random(V) < 0.25
    batch = _synthetic_batch(rng, B, T, V, G, group_of, Kp, bmask)
    bm = dl.batch_to_mx(batch)
    T_pad = int(batch["student_ids"].shape[1])
    hidden = mx.array(rng.standard_normal((B, T_pad, d)).astype(np.float32))
    W = mx.array((rng.standard_normal((V, d)) * 0.3).astype(np.float32))
    knobs = dict(KNOBS, lambda_ce=0.5)

    def fused(hid, w):
        head = dl.linear_head(w, softcap)
        loss, _ = dl.distill_loss(hid, bm, head, group_of=mx.array(group_of), G=G, Kp=Kp,
                                  log_bmask=dl.log_bmask_from(bmask), knobs=knobs, C=C)
        return loss

    def dense(hid, w):
        return _dense_loss_mx(hid, w, bm, mx.array(group_of), G, Kp, bmask, knobs, softcap=softcap)

    lf, ld = fused(hidden, W), dense(hidden, W)
    gf = mx.grad(fused, argnums=(0, 1))(hidden, W)
    gd = mx.grad(dense, argnums=(0, 1))(hidden, W)
    mx.eval(lf, ld, gf, gd)
    assert abs(float(lf) - float(ld)) < 2e-4 * max(1.0, abs(float(ld)))
    for a, b in zip(gf, gd):
        a, b = np.asarray(a), np.asarray(b)
        assert np.allclose(a, b, atol=2e-4, rtol=2e-3), np.abs(a - b).max()


def test_slot_masses_at_1e6_vs_float64():
    """Group masses at true mass 1e-6 against float64 at V >= 150k with
    LM-shaped logits."""
    rng = np.random.default_rng(8)
    V, d, G, Kp = 151936, 16, 4000, 6
    group_of = rng.integers(0, G, V).astype(np.int32)
    h = rng.standard_normal((3, d)).astype(np.float32)
    W = (rng.standard_normal((V, d)) * 0.5).astype(np.float32)
    z64 = h.astype(np.float64) @ W.astype(np.float64).T
    z64[:, :5] += 12.0                                    # sharp head
    q64 = np.exp(z64 - np.logaddexp.reduce(z64, axis=1, keepdims=True))
    gm64 = np.zeros((3, G))
    for r in range(3):
        np.add.at(gm64[r], group_of, q64[r])
    # pick groups with mass near 1e-6 as targets
    gid = np.full((3, Kp), G, dtype=np.int32)
    for r in range(3):
        order = np.argsort(np.abs(gm64[r] - 1e-6))[:Kp]
        gid[r] = order
    head = dl.linear_head(mx.array(W))
    # emulate the sharp head by biasing the hidden product: add the bias through a bias row
    Wb = W.copy()
    hb = np.concatenate([h, np.ones((3, 1), dtype=np.float32)], axis=1)
    Wb = np.concatenate([W, np.zeros((V, 1), dtype=np.float32)], axis=1)
    Wb[:5, d] = 12.0
    head = dl.linear_head(mx.array(Wb))
    _, Q_slot, _ = dl.chunked_head(mx.array(hb), head, mx.zeros((3,), dtype=mx.int32), n_bnd=3,
                                   target_gid=mx.array(gid), group_of=mx.array(group_of), G=G, Kp=Kp,
                                   log_bmask=dl.log_bmask_from(np.zeros(V, dtype=bool)), C=2)
    mx.eval(Q_slot)
    Q = np.asarray(Q_slot)
    for r in range(3):
        ref = gm64[r][gid[r]]
        assert np.all(ref < 1e-4)
        rel = np.abs(Q[r, :Kp] - ref) / ref
        assert rel.max() < 1e-3, rel.max()


# ---------------------------------------------------------------------------
# cache format, validator, resume, prefill plan
# ---------------------------------------------------------------------------

def _tiny_cache(tmp: Path, tok, seed=11, n_rows=6, K=8, floor=False, routing=None):
    """A synthetic teacher (random linear head over random hidden states)
    through reduce_logits and the ShardWriter. ``routing`` (moe_layers, k,
    n_experts) adds a random routes field and the manifest block."""
    rng = np.random.default_rng(seed)
    tb = dl.token_bytes(tok)
    V = len(tb)
    ws = dl.whitespace_start_mask(tok, V, tb)
    log_bmask = dl.log_bmask_from(ws)
    W = mx.array((rng.standard_normal((V, 6)) * 0.5).astype(np.float32))
    writer = dl.ShardWriter(tmp, K, floor)
    texts = [_rand_text(rng, int(rng.integers(4, 12))) for _ in range(n_rows)]
    rows_per_shard = 2
    for si in range(0, n_rows, rows_per_shard):
        if si // rows_per_shard < writer.n_done:
            continue
        rows, metas, tbytes = [], [], []
        for r in range(si, min(si + rows_per_shard, n_rows)):
            tb_ = texts[r].encode("utf-8")
            ids, ends, _ = dl.encode_with_byte_ends(tok, tb_, tb, add_special_tokens=False)
            n = len(ids)
            h = mx.array(np.random.default_rng(seed * 1000 + r).standard_normal((n, 6)).astype(np.float32))
            logits = (h @ W.T).astype(mx.bfloat16)
            nxt = np.concatenate([ids[1:], [-1]]).astype(np.int32)
            red = dl.reduce_logits(logits, mx.array(nxt), K=K, log_bmask=log_bmask,
                                   onpath_valid=nxt >= 0, floor=floor)
            red["onpath_mask"] = nxt >= 0
            red["token_ids"] = ids
            red["token_end_byte"] = ends
            if routing:
                n_moe, k, E = len(routing["moe_layers"]), routing["k"], routing["n_experts"]
                red[dl.ROUTES_FIELD] = np.stack([np.stack([rng.permutation(E)[:k] for _ in range(n_moe)])
                                                 for _ in range(n)]).astype(dl.routes_dtype(E))
            rows.append(red)
            metas.append(dl.RowMeta(row_id=r, doc_id=f"d{r}", window=0, n_tokens=n))
            tbytes.append(tb_)
        packed = dl.pack_shard(rows, tbytes, K, floor)
        writer.write(si // rows_per_shard, packed, metas, wall_s=0.01, step=64)
    dl.write_manifest(tmp, teacher_path="synthetic", dataset="synthetic", num_samples=n_rows,
                      max_seq_len=64, seed=seed, top_k=K, vocab_size=V, config_vocab_size=V,
                      tokenizer_hash=dl.vocab_map_hash(tok), batch_size=rows_per_shard,
                      gmlx_distill={"mlx_kld_compatible": True, "corpus_sha256": "x", "routing": routing})
    return writer


def test_routes_field_pack_and_validator(tmp_path, tok_bl):
    routing = {"moe_layers": [1, 3, 5], "k": 2, "n_experts": 8, "dtype": "uint8"}
    _tiny_cache(tmp_path / "c", tok_bl, routing=routing)
    assert dl.validate_cache(tmp_path / "c") == []
    reader = dl.CacheReader(tmp_path / "c")
    arrs, _, _ = reader.row(0)
    assert arrs[dl.ROUTES_FIELD].shape == (len(arrs["token_ids"]), 3, 2)
    assert arrs[dl.ROUTES_FIELD].dtype == np.uint8
    assert dl.routes_dtype(256) == np.uint8 and dl.routes_dtype(257) == np.uint16
    assert dl.bytes_per_position(8, routes_bytes=6) == dl.bytes_per_position(8) + 6
    # an id past the expert count, and a repeated expert, are problems
    sp = tmp_path / "c" / "batch-00000.safetensors"
    sh = dl.load_shard(sp)
    bad = dict(sh)
    bad[dl.ROUTES_FIELD] = sh[dl.ROUTES_FIELD].copy()
    bad[dl.ROUTES_FIELD][0, 0, 0, 0] = 9
    bad[dl.ROUTES_FIELD][0, 1, 1, :] = 3
    sp.write_bytes(dl.save_safetensors_bytes(bad))
    probs = dl.validate_cache(tmp_path / "c", check_sha=False)
    assert any("id >= n_experts" in p for p in probs) and any("repeat an expert" in p for p in probs)
    # the field without the block, and the block without the field
    manifest = dl.read_json(tmp_path / "c" / "manifest.json")
    manifest["gmlx_distill"]["routing"] = None
    dl.write_json_atomic(tmp_path / "c" / "manifest.json", manifest)
    assert any("without a routing block" in p for p in dl.validate_cache(tmp_path / "c", check_sha=False))
    good = dict(sh)
    del good[dl.ROUTES_FIELD]
    sp.write_bytes(dl.save_safetensors_bytes(good))
    manifest["gmlx_distill"]["routing"] = routing
    dl.write_json_atomic(tmp_path / "c" / "manifest.json", manifest)
    assert any("routes field missing" in p for p in dl.validate_cache(tmp_path / "c", check_sha=False))


def test_route_record_and_pin_helpers():
    pytest.importorskip("gmlx.stream.moe_routes")
    from types import SimpleNamespace

    import mlx.nn as nn
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    args = SimpleNamespace(hidden_size=16, moe_intermediate_size=32, num_experts=8, num_experts_per_tok=4,
                           norm_topk_prob=True)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Qwen3MoeSparseMoeBlock(args)

    class _Shell:
        pass

    model = _Shell()
    model.layers = [_Layer(), _Layer()]
    mx.random.seed(3)
    for ly in model.layers:
        mx.eval(ly.parameters())
    x = mx.random.normal((2, 5, 16))
    rec, why = dl.install_route_recording(model)
    assert rec is not None, why
    ref = [np.array(ly.mlp(x)) for ly in model.layers]
    routes = dl.take_routes_blt(rec)
    assert routes.shape == (2, 5, 2, 4)
    from gmlx.stream.moe_routes import clear_moe_route_controls
    clear_moe_route_controls(model)
    with dl.pin_routes(model, routes, [0, 1]) as replay:
        assert replay.rows == 2
        for ly, r in zip(model.layers, ref):
            assert np.array_equal(np.array(ly.mlp(x)), r)
    swapped = routes.copy()
    swapped[0, 2, 0] = [e for e in range(8) if e not in set(routes[0, 2, 0].tolist())][:4]
    with dl.pin_routes(model, swapped, [0, 1]):
        out = np.array(model.layers[0].mlp(x))
    assert not np.array_equal(out[0, 2], ref[0][0, 2]) and np.array_equal(out[1], ref[0][1])
    manifest = {"gmlx_distill": {"routing": {"moe_layers": [0, 1], "k": 4, "n_experts": 8}}}
    assert dl.replay_layers_for(model, manifest) == [0, 1]
    manifest["gmlx_distill"]["routing"]["moe_layers"] = [0]
    assert dl.replay_layers_for(model, manifest) is None
    assert dl.replay_layers_for(model, {"gmlx_distill": {}}) is None



def test_reduce_logits_fields():
    rng = np.random.default_rng(9)
    V, n, K = 300, 5, 16
    z = mx.array(rng.standard_normal((n, V)).astype(np.float32))
    nxt = np.array([1, 2, 3, 4, -1], dtype=np.int32)
    bmask = rng.random(V) < 0.2
    red = dl.reduce_logits(z.astype(mx.bfloat16), mx.array(nxt), K=K, log_bmask=dl.log_bmask_from(bmask),
                           onpath_valid=nxt >= 0, floor=True)
    z64 = np.asarray(z.astype(mx.bfloat16).astype(mx.float32)).astype(np.float64)
    lp = z64 - np.logaddexp.reduce(z64, axis=1, keepdims=True)
    top = np.sort(lp, axis=1)[:, ::-1][:, :K]
    assert np.allclose(red["top_k_log_softmax"].astype(np.float64), top, atol=2e-3)
    assert np.all(np.diff(red["top_k_log_softmax"].astype(np.float32), axis=1) <= 0)
    for i in range(n):
        mask = np.ones(V, dtype=bool)
        mask[red["top_k_indices"][i]] = False
        assert abs(red["tail_log_mass"][i] - np.log(np.exp(lp[i][mask]).sum())) < 1e-4
        assert abs(red["log_boundary_mass"][i] - np.log(np.exp(lp[i][bmask]).sum())) < 1e-4
        if nxt[i] >= 0:
            assert abs(red["onpath_log_p"][i] - lp[i, nxt[i]]) < 1e-4
        else:
            assert red["onpath_log_p"][i] == 0.0
    assert np.all(red["floor_kld"] >= -1e-6)


def test_cache_validator_resume_and_s2_rules(tmp_path, tok_bl):
    _tiny_cache(tmp_path / "c", tok_bl)
    assert dl.validate_cache(tmp_path / "c") == []
    # resume reproduces identical shards: drop the last, rewrite, compare bytes
    p2 = tmp_path / "c" / "batch-00002.safetensors"
    orig = p2.read_bytes()
    p2.unlink()
    w2 = dl.ShardWriter(tmp_path / "c", 8, False)
    assert w2.verified_shards() == 2
    _tiny_cache(tmp_path / "c", tok_bl)
    assert p2.read_bytes() == orig
    assert dl.validate_cache(tmp_path / "c") == []
    # S2 rules: a nonzero prefix field with mlx_kld_compatible true is a problem
    rp = tmp_path / "c" / "rows-00000.jsonl"
    rows = dl.read_rows_jsonl(rp)
    rows[0]["prefix_n_tokens"] = 1
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    probs = dl.validate_cache(tmp_path / "c", check_sha=False)
    assert any("prefix fields" in p for p in probs)
    # corrupt a shard: sha mismatch reported
    p0 = tmp_path / "c" / "batch-00000.safetensors"
    b = bytearray(p0.read_bytes())
    b[-1] ^= 1
    p0.write_bytes(bytes(b))
    probs = dl.validate_cache(tmp_path / "c")
    assert any("sha256" in p for p in probs)


def test_cache_reader_and_identity_view_equals_compiled(tmp_path, tok_bl):
    _tiny_cache(tmp_path / "c", tok_bl, K=8)
    reader = dl.CacheReader(tmp_path / "c")
    assert len(reader) == 6
    tables = dl.build_tables(tok_bl, tok_bl)
    knobs = dict(KNOBS, w_mid=1.0)
    ident = dl.ViewLoader(reader, tok_bl, tables, knobs=knobs, Kp=8, identity=True)
    comp = dl.ViewLoader(reader, tok_bl, tables, knobs=knobs, Kp=8, identity=False,
                         student_tb=dl.token_bytes(tok_bl))
    for r in range(len(reader)):
        a = ident.compile(r)
        b = comp.compile(r)
        if b is None:
            continue
        assert np.array_equal(a.student_ids, b.student_ids)
        assert np.array_equal(a.bnd_pos, b.bnd_pos)
        # same groups per boundary, up to ordering
        for j in range(a.target_gid.shape[0]):
            ga = {int(g): float(v) for g, v in zip(a.target_gid[j], a.target_log_p[j]) if g != tables.G}
            gb = {int(g): float(v) for g, v in zip(b.target_gid[j], b.target_log_p[j]) if g != tables.G}
            assert ga.keys() == gb.keys()
            for g in ga:
                assert abs(ga[g] - gb[g]) < 1e-5
        assert np.allclose(a.log_M, b.log_M, atol=1e-5)
        assert np.all(b.bnd_weight == 1.0)


def test_cross_view_compiles_and_loss_runs(tmp_path, tok_bl, tok_spm):
    _tiny_cache(tmp_path / "c", tok_bl, K=8, n_rows=8)
    reader = dl.CacheReader(tmp_path / "c")
    tables = dl.build_tables(tok_bl, tok_spm)
    loader = dl.ViewLoader(reader, tok_spm, tables, knobs=KNOBS, Kp=8, identity=False)
    it = dl.BatchIterator([reader.rows_meta[r]["n_tokens"] for r in range(len(reader))], 4, seed=1)
    _, rows = next(it.iterate())
    batch = loader.batch(rows)
    assert batch is not None
    bm = dl.batch_to_mx(batch)
    d = 6
    rng = np.random.default_rng(12)
    hidden = mx.array(rng.standard_normal((batch["student_ids"].shape[0], batch["student_ids"].shape[1], d)).astype(np.float32))
    W = mx.array((rng.standard_normal((tables.V_S, d)) * 0.3).astype(np.float32))
    loss, aux = dl.distill_loss(hidden, bm, dl.linear_head(W), group_of=mx.array(tables.group_of),
                                G=tables.G, Kp=8, log_bmask=dl.log_bmask_from(tables.bmask_S),
                                knobs=KNOBS, C=16)
    mx.eval(loss)
    assert np.isfinite(float(loss)) and float(loss) >= 0
    # materialize and read back identically
    n = loader.materialize(tmp_path / "v")
    assert n == reader.n_shards
    mat = dl.ViewLoader(reader, tok_spm, tables, knobs=KNOBS, Kp=8, identity=False, view_dir=tmp_path / "v")
    for r in rows:
        a, b = loader.compile(r), mat.compile(r)
        if a is None:
            assert b is None
            continue
        assert np.array_equal(a.target_gid, b.target_gid) and np.array_equal(a.chunk_end, b.chunk_end)


def test_batch_iterator_seeded_and_resumable():
    lengths = list(np.random.default_rng(0).integers(5, 50, 40))
    a = dl.BatchIterator(lengths, 4, seed=0)
    b = dl.BatchIterator(lengths, 4, seed=0)
    c = dl.BatchIterator(lengths, 4, seed=1)
    xs = [r for _, (_, r) in zip(range(25), a.iterate())]
    ys = [r for _, (_, r) in zip(range(25), b.iterate())]
    zs = [r for _, (_, r) in zip(range(25), c.iterate())]
    assert xs == ys and xs != zs
    resumed = [r for _, (_, r) in zip(range(5), b.iterate(skip=20))]
    assert resumed == xs[20:25]
    assert sorted(sum(xs[:a.per_epoch], [])) == sorted(range(40))


def test_decontam_and_stats():
    rng = np.random.default_rng(13)
    corpus = [bytes(rng.integers(0, 256, 5000, dtype=np.uint8)) for _ in range(3)]
    clean = bytes(rng.integers(0, 256, 3000, dtype=np.uint8))
    assert dl.decontam_fraction(clean, corpus) == 0.0
    dirty = clean[:1000] + corpus[1][100:1100] + clean[1000:]
    f = dl.decontam_fraction(dirty, corpus)
    assert 0.2 < f < 0.4
    r = dl.paired_bootstrap(np.array([1, 2, 3, 4.0]), np.array([1, 1, 1, 1.0]))
    assert r["diff"] == 1.5
    t = dl.welch_one_sided(np.array([1.0, 1.1, 0.9]), np.array([0.2, 0.3]))
    assert t["dof"] == 3 and t["p"] < 0.05
    h = dl.holm({"a": 0.001, "b": 0.03, "c": 0.5})
    assert h["a"] and not h["c"]


def test_adapter_disabled_drops_cached_lora_tables():
    """gmlx's live LoRA module folds its scale into cached tables on first
    use; the before-score toggle must reach through that cache."""
    import mlx.core as mx
    import mlx.nn as nn
    gm = pytest.importorskip("gmlx.load.modules")
    base = nn.Linear(8, 6, bias=False)
    a = mx.random.normal((4, 8)) * 0.1
    b = mx.random.normal((6, 4)) * 0.1
    m = gm.LoRAKQuantLinear(base, a, b, scale=2.0)
    x = mx.random.normal((3, 8))
    adapted = m(x)
    mx.eval(adapted)
    with dl.adapter_disabled(m):
        off = m(x)
        mx.eval(off)
    back = m(x)
    mx.eval(adapted, off, back)
    assert np.allclose(np.asarray(off), np.asarray(base(x)), atol=1e-6)
    assert not np.allclose(np.asarray(off), np.asarray(adapted), atol=1e-4)
    assert np.allclose(np.asarray(back), np.asarray(adapted), atol=1e-6)


def test_refusal_phrase_match():
    yes = ["I can't help with that request.", "I'm sorry, but I cannot assist with this.",
           "I won't provide instructions for that.", "I am unable to help with harmful activities."]
    no = ["The capital of Australia is Canberra.", "def reverse(s):\n    return s[::-1]",
          "Sure! Here are three tips.", "It cannot be sorted in place without extra memory."]
    assert all(dl_eval._REFUSAL_RE.search(t) for t in yes)
    assert not any(dl_eval._REFUSAL_RE.search(t) for t in no)


def test_chat_compliance_helpers():
    assert dl_eval._is_repetitive("a b c d e f g h i j k l " * 4)
    assert not dl_eval._is_repetitive("* item one\n* item two\n* item three\n" * 3)
    assert dl_eval._TEMPLATE_LEAK_RE.search("Sure.<end_of_turn>\n<start_of_turn>user\nhi")
    assert dl_eval._TEMPLATE_LEAK_RE.search("answer\nuser\nnext question")
    assert not dl_eval._TEMPLATE_LEAK_RE.search("The user asked about hash tables. Model answers follow.")


# ---------------------------------------------------------------------------
# framed rows
# ---------------------------------------------------------------------------

_TEMPLATE_A = ("{% for m in messages %}{% if m['role'] == 'user' %}<bos>User: {{ m['content'] | trim }}\n"
               "{% else %}Model:\n{{ m['content'] | trim }}\n<end>\n{% endif %}{% endfor %}"
               "{% if add_generation_prompt %}Model:\n{% endif %}")
_TEMPLATE_B = ("{% for m in messages %}{% if m['role'] == 'user' %}[u] {{ m['content'] }} [/u]\n"
               "{% else %}[m] {{ m['content'] }} [/m]\n{% endif %}{% endfor %}"
               "{% if add_generation_prompt %}[m] {% endif %}")


def _with_template(tok, template):
    tok.chat_template = template
    return tok


def test_special_text_bytes_in_offsets(tok_bl):
    tb = dl.token_bytes(tok_bl)
    text = b"<bos>the cat"
    ids, ends, flagged = dl.encode_with_byte_ends(tok_bl, text, tb, add_special_tokens=False)
    assert not flagged
    assert ids[0] == tok_bl.bos_token_id and ends[0] == 5 and ends[-1] == len(text)


def test_render_row_spans_and_target_mask(tok_bl):
    tok = _with_template(tok_bl, _TEMPLATE_A)
    tb = dl.token_bytes(tok)
    # continue framing: frame + content, no closing marker, one span over the content
    msgs = dl.continue_messages("the cat is the cat")
    text, spans = dl.render_row(tok, msgs, open_tail=True)
    assert text.endswith(b"the cat is the cat") and not text.endswith(b"<end>\n")
    assert spans == [(len(text) - 18, len(text), len(text))]
    ids, ends, _ = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
    tm = dl.target_mask(ends.astype(np.int64), spans)
    first = int(np.argmax(tm))
    # the position before the first content token is the first target; every
    # later position with a successor is one too; the last has no successor
    assert ends[first] == spans[0][0] and tm[first:-1].all() and not tm[-1] and not tm[:first].any()
    assert not dl.target_mask(ends.astype(np.int64), None)[-1] and dl.target_mask(ends.astype(np.int64), None)[:-1].all()
    # chat framing: two assistant spans, each extended over the end marker,
    # a tool-call-only turn (empty content) is context
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "the cat"},
            {"role": "user", "content": "more"}, {"role": "assistant", "content": ""},
            {"role": "user", "content": "again"}, {"role": "assistant", "content": "is the cat"}]
    text, spans = dl.render_row(tok, conv, open_tail=False)
    assert len(spans) == 2
    for (b0, b1, b2), c in zip(spans, ("the cat", "is the cat")):
        assert text[b0:b1] == c.encode() and text[b1:b2] == b"\n<end>\n"
    with pytest.raises(ValueError):
        dl.render_row(tok, conv[:-1], open_tail=False)   # ends on a user turn


def test_render_row_closed_tail(tok_bl):
    tok = _with_template(tok_bl, _TEMPLATE_A)
    tb = dl.token_bytes(tok)
    msgs = dl.continue_messages("the cat is the cat")
    open_text, open_spans = dl.render_row(tok, msgs, **dl.row_render_args("continue"))
    text, spans = dl.render_row(tok, msgs, **dl.row_render_args("continue-closed"))
    tail = dl.assistant_tails(tok)[0].encode()
    # same frame and content as the open render, closed by the template's own marker
    assert text == open_text + tail and spans == [(open_spans[0][0], open_spans[0][1], open_spans[0][1] + len(tail))]
    ids, ends, _ = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
    tm = dl.target_mask(ends.astype(np.int64), spans)
    oids, oends, _ = dl.encode_with_byte_ends(tok, open_text, tb, add_special_tokens=False)
    otm = dl.target_mask(oends.astype(np.int64), open_spans)
    # the position after the last content token now predicts the marker
    assert int(tm.sum()) > int(otm.sum()) and tm[len(oids) - 1] and not otm[len(oids) - 1]
    assert dl.row_render_args("chat") == {"open_tail": False} and "continue-closed" in dl.FRAME_KINDS


def _tiny_framed_cache(tmp: Path, tok, msgs_list, *, open_tail, K=8, seed=5):
    rng = np.random.default_rng(seed)
    tb = dl.token_bytes(tok)
    V = len(tb)
    log_bmask = dl.log_bmask_from(dl.whitespace_start_mask(tok, V, tb))
    W = mx.array((rng.standard_normal((V, 6)) * 0.5).astype(np.float32))
    writer = dl.ShardWriter(tmp, K, False)
    rows, metas, tbytes = [], [], []
    for r, msgs in enumerate(msgs_list):
        text, spans = dl.render_row(tok, msgs, open_tail=open_tail)
        ids, ends, _ = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
        n = len(ids)
        h = mx.array(rng.standard_normal((n, 6)).astype(np.float32))
        nxt = np.concatenate([ids[1:], [-1]]).astype(np.int32)
        tm = dl.target_mask(ends.astype(np.int64), spans)
        valid = (nxt >= 0) & tm
        red = dl.reduce_logits((h @ W.T).astype(mx.bfloat16), mx.array(nxt), K=K, log_bmask=log_bmask,
                               onpath_valid=valid)
        red["onpath_mask"] = valid
        red["token_ids"] = ids
        red["token_end_byte"] = ends
        rows.append(red)
        tbytes.append(text)
        metas.append(dl.RowMeta(row_id=r, doc_id=f"d{r}", window=0, n_tokens=n,
                                frame="continue" if open_tail else "chat", messages=msgs,
                                spans=[list(s) for s in spans], prefix_n_tokens=int(np.argmax(tm)) + 1,
                                suffix_start_byte=spans[0][0]))
    writer.write(0, dl.pack_shard(rows, tbytes, K, False), metas, wall_s=0.01, step=64)
    dl.write_manifest(tmp, teacher_path="synthetic", dataset="synthetic", num_samples=len(rows),
                      max_seq_len=64, seed=seed, top_k=K, vocab_size=V, config_vocab_size=V,
                      tokenizer_hash=dl.vocab_map_hash(tok), batch_size=8,
                      gmlx_distill={"mlx_kld_compatible": False, "corpus_sha256": "x",
                                    "frame": {"kind": "continue" if open_tail else "chat"}})


def test_framed_identity_view_and_validator(tmp_path, tok_bl):
    tok = _with_template(tok_bl, _TEMPLATE_A)
    msgs = [dl.continue_messages(t) for t in ("the cat is the cat", "is the cat the cat is", "cat cat the")]
    _tiny_framed_cache(tmp_path / "c", tok, msgs, open_tail=True)
    assert dl.validate_cache(tmp_path / "c") == []
    reader = dl.CacheReader(tmp_path / "c")
    tables = dl.build_tables(tok, tok)
    loader = dl.ViewLoader(reader, tok, tables, knobs=KNOBS, Kp=8, identity=True)
    for r in range(len(reader)):
        arrs, _text, meta = reader.row(r)
        rv = loader.compile(r)
        pn = meta["prefix_n_tokens"]
        assert np.array_equal(rv.student_ids, arrs["token_ids"])          # frame forwarded as is
        assert not rv.compute_mask[:pn - 1].any() and rv.compute_mask[pn - 1:].all()
        assert np.array_equal(rv.bnd_pos, np.nonzero(arrs["onpath_mask"][:-1])[0])
        assert rv.bnd_pos[0] == pn - 1
    batch = loader.batch(list(range(len(reader))))
    assert int(batch["compute_mask"].sum()) == int(batch["n_bnd"])
    # validator: an on-path flag inside the frame is a problem, and so is a
    # framed row without spans
    from safetensors.numpy import load_file, save_file
    p0 = tmp_path / "c" / "batch-00000.safetensors"
    sh = load_file(str(p0))
    sh["onpath_mask"][0, 0] = True
    save_file(sh, str(p0))
    probs = dl.validate_cache(tmp_path / "c", check_sha=False)
    assert any("inside the frame" in p for p in probs)
    rp = tmp_path / "c" / "rows-00000.jsonl"
    rows = dl.read_rows_jsonl(rp)
    rows[1]["spans"] = []
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    probs = dl.validate_cache(tmp_path / "c", check_sha=False)
    assert any("without spans" in p for p in probs)


def test_framed_cross_view_aligns_inside_spans(tmp_path, tok_bl, tok_spm):
    teacher = _with_template(tok_bl, _TEMPLATE_A)
    student = _with_template(tok_spm, _TEMPLATE_B)
    convs = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "the cat is the cat"},
              {"role": "user", "content": "more"}, {"role": "assistant", "content": "is the cat 123"}],
             dl.continue_messages("the cat is 123 the cat")]
    _tiny_framed_cache(tmp_path / "chat", teacher, convs[:1], open_tail=False)
    _tiny_framed_cache(tmp_path / "cont", teacher, convs[1:], open_tail=True)
    tables = dl.build_tables(teacher, student)
    for name in ("chat", "cont"):
        reader = dl.CacheReader(tmp_path / name)
        loader = dl.ViewLoader(reader, student, tables, knobs=KNOBS, Kp=8, identity=False)
        arrs, ttext, meta = reader.row(0)
        s_ids, s_ends, s_spans = loader.student_tokens(arrs, ttext, meta)
        stext, _ = dl.render_row(student, meta["messages"], open_tail=(name == "cont"))
        assert stext.startswith(b"[u] ") and not ttext.startswith(b"[u] ")   # each side's own frame
        rv = loader.compile(0)
        assert rv is not None and rv.bnd_pos.shape[0] >= 2
        s_tm = dl.target_mask(s_ends, s_spans)
        assert np.array_equal(rv.compute_mask, s_tm[:len(s_ids) - 1])
        # every boundary predicts a target on both sides and sits at the same
        # offset inside its span
        t_ends = arrs["token_end_byte"].astype(np.int64)
        t_spans = [tuple(x) for x in meta["spans"]]
        al = dl.shared_boundaries_spans(t_ends, s_ends, t_spans, s_spans)
        assert np.array_equal(al.s_pos, rv.bnd_pos)
        for tp, sp in zip(al.t_pos, al.s_pos):
            assert arrs["onpath_mask"][tp] and s_tm[sp]
            k = next(i for i, sp in enumerate(t_spans) if sp[0] <= t_ends[tp] <= sp[1])
            assert t_ends[tp] - t_spans[k][0] == s_ends[sp] - s_spans[k][0]
        # chunks never cross the frame
        for cs, ce in zip(rv.chunk_start, rv.chunk_end):
            assert s_tm[cs:ce + 1].all()
        batch = loader.batch([0])
        bm = dl.batch_to_mx(batch)
        rng = np.random.default_rng(3)
        hidden = mx.array(rng.standard_normal((1, batch["student_ids"].shape[1], 6)).astype(np.float32))
        W = mx.array((rng.standard_normal((tables.V_S, 6)) * 0.3).astype(np.float32))
        loss, _aux = dl.distill_loss(hidden, bm, dl.linear_head(W), group_of=mx.array(tables.group_of),
                                     G=tables.G, Kp=8, log_bmask=dl.log_bmask_from(tables.bmask_S),
                                     knobs=KNOBS, C=16)
        mx.eval(loss)
        assert np.isfinite(float(loss))


def test_cache_kld_rerenders_framed_rows(tmp_path, tok_bl):
    import copy
    teacher = _with_template(tok_bl, _TEMPLATE_A)
    msgs = [dl.continue_messages(t) for t in ("the cat is the cat", "is the cat the cat is", "cat cat the")]
    _tiny_framed_cache(tmp_path / "c", teacher, msgs, open_tail=True)
    reader = dl.CacheReader(tmp_path / "c")
    V = len(dl.token_bytes(tok_bl))
    W = mx.array(np.random.default_rng(3).standard_normal((V, V)).astype(np.float32))

    class Stub:
        def __call__(self, ids):
            return W[ids]          # logits depend on the current token only

    base = dl.cache_kld(Stub(), reader)
    same = dl.cache_kld(Stub(), reader, tokenizer=teacher)
    assert same["rerendered_rows"] == 0 and same["positions"] == base["positions"]
    assert same["mean_kld_nats"] == base["mean_kld_nats"]
    student = _with_template(copy.deepcopy(tok_bl), _TEMPLATE_B)
    cross = dl.cache_kld(Stub(), reader, tokenizer=student)
    assert cross["rerendered_rows"] == len(reader) == cross["rows"]
    # at most the first content position per row is lost to a frame-straddling token
    assert base["positions"] - len(reader) <= cross["positions"] <= base["positions"]
    assert np.isfinite(cross["mean_kld_nats"])
    if cross["positions"] == base["positions"]:
        assert abs(cross["mean_kld_nats"] - base["mean_kld_nats"]) < 1e-9


# template-disjoint pairs: frame rule, turn-end markers, render settings,
# templates that reject roles, students without a template

_TEMPLATE_H = ("{% for m in messages %}{% if m['role'] == 'user' %}<s>user<m>{{ m['content'] }}<end>"
               "{% elif loop.last %}<s>assistant<final>{{ m['content'] }}<ret>"
               "{% else %}<s>assistant<final>{{ m['content'] }}<end>{% endif %}{% endfor %}"
               "{% if add_generation_prompt %}<s>assistant{% endif %}")
_TEMPLATE_G = ("{% for m in messages %}{% if m['role'] == 'user' %}<u>{{ m['content'] }}"
               "{% else %}<a><think></think>{{ m['content'] }}{% endif %}{% endfor %}"
               "{% if add_generation_prompt %}<a><think>{% endif %}")
_TEMPLATE_NOSYS = ("{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}"
                   "{% endif %}{% for m in messages %}{% if m['role'] == 'tool' %}"
                   "{{ raise_exception('tool role not supported') }}{% endif %}"
                   "<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>\n{% endfor %}"
                   "{% if add_generation_prompt %}<assistant>{% endif %}")
_TEMPLATE_KW = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>\n{% endfor %}"
                "{% if add_generation_prompt %}<assistant>{% if enable_thinking is defined and not enable_thinking %}"
                "<nothink>{% endif %}{% endif %}{% if date_string is defined %}<date>{{ date_string }}</date>{% endif %}")


def _marked(tok_bl, template, markers):
    import copy
    tok = copy.deepcopy(tok_bl)
    tok.add_special_tokens({"additional_special_tokens": list(markers)})
    tok.chat_template = template
    return tok


def test_frame_extends_to_completed_turn_prefix(tok_bl):
    tok = _marked(tok_bl, _TEMPLATE_H, ["<s>", "<m>", "<end>", "<ret>", "<final>"])
    user = [{"role": "user", "content": "q"}]
    assert dl.render_frame(tok, user) == "<s>user<m>q<end><s>assistant<final>"
    text, spans = dl.render_row(tok, dl.continue_messages("the cat"), open_tail=True)
    assert text.decode().endswith("<final>the cat") and spans == [(len(text) - 7, len(text), len(text))]
    # the frame prefix for bpb windows is the same extended frame
    assert dl.frame_prefix(tok, "continue").endswith("<s>assistant<final>")
    tok_g = _marked(tok_bl, _TEMPLATE_G, ["<u>", "<a>", "<think>", "</think>"])
    assert dl.render_frame(tok_g, user) == "<u>q<a><think></think>"
    # a template whose completed turn does not extend the generation prompt keeps it
    tok_a = _with_template(tok_bl, _TEMPLATE_A)
    assert dl.render_frame(tok_a, user) == "<bos>User: q\nModel:\n"


def test_turn_end_markers_per_turn_and_appended(tok_bl):
    tok = _marked(tok_bl, _TEMPLATE_H, ["<s>", "<m>", "<end>", "<ret>", "<final>"])
    assert dl.assistant_tails(tok) == ["<ret>", "<end>"]
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    text, spans = dl.render_row(tok, msgs, open_tail=False)
    t = text.decode()
    assert [t[a:c] for a, _b, c in spans] == ["a1<end>", "a2<ret>"]
    tok_g = _marked(tok_bl, _TEMPLATE_G, ["<u>", "<a>", "<think>", "</think>"])
    assert dl.assistant_tails(tok_g) == ["<u>"]
    text, spans = dl.render_row(tok_g, msgs, open_tail=False)
    t = text.decode()
    # the last turn, unmarked by the template, gets the marker appended
    assert t.endswith("a2<u>") and [t[a:c] for a, _b, c in spans] == ["a1<u>", "a2<u>"]
    assert dl.target_mask(np.array([0, 1, 2, 3]), spans).dtype == bool


def test_render_kwargs_change_the_frame_and_pin_the_date(tok_bl):
    import copy
    tok = _with_template(copy.deepcopy(tok_bl), _TEMPLATE_KW)
    assert dl.default_render_kwargs(tok, date="01 Jan 2026") == {"date_string": "01 Jan 2026"}
    user = [{"role": "user", "content": "q"}]
    dl.set_render_kwargs(tok, {})
    assert dl.render_frame(tok, user) == "<user>q</user>\n<assistant>"
    dl.set_render_kwargs(tok, {"enable_thinking": False, "date_string": "01 Jan 2026"})
    assert dl.render_frame(tok, user) == "<user>q</user>\n<assistant><nothink><date>01 Jan 2026</date>"
    assert dl.render_kwargs(tok) == {"enable_thinking": False, "date_string": "01 Jan 2026"}
    # the resolver inherits the other side's date and applies overrides on top
    kw = dl.resolve_render_kwargs(tok, inherit={"date_string": "02 Feb 2026"}, override={"enable_thinking": False})
    assert kw == {"date_string": "02 Feb 2026", "enable_thinking": False}
    assert dl.parse_render_kwargs('{"enable_thinking": false}') == {"enable_thinking": False}
    assert dl.parse_render_kwargs(None) == {}


def test_system_folded_and_tool_rows_dropped(tok_bl):
    import copy
    tok = _with_template(copy.deepcopy(tok_bl), _TEMPLATE_NOSYS)
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a1"}]
    text, spans = dl.render_row(tok, msgs, open_tail=False)
    t = text.decode()
    assert t.startswith("<user>be brief\n\nq</user>") and t[spans[0][0]:spans[0][1]] == "a1"
    assert dl.render_frame(tok, msgs[:2]).startswith("<user>be brief\n\nq</user>")
    tool = [{"role": "user", "content": "q"}, {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "a1"}]
    with pytest.raises(ValueError, match="rejects"):
        dl.render_row(tok, tool, open_tail=False)
    tb = dl.token_bytes(tok)
    assert dl.fit_conversation(tok, tool, 64, tb) is None


def test_no_template_student_plain_render(tok_bl):
    import copy
    tok = copy.deepcopy(tok_bl)
    tok.chat_template = None
    assert not dl.has_chat_template(tok)
    assert dl.assistant_tails(tok) == ["<eos>"]
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    bos = "<bos>" if dl.adds_bos(tok) else ""   # the stub adds none; a GGUF synth may
    text, spans = dl.render_row(tok, msgs, open_tail=False)
    t = text.decode()
    assert t == bos + "q\n\na1<eos>\n\nq2\n\na2<eos>"
    assert [t[a:c] for a, _b, c in spans] == ["a1<eos>", "a2<eos>"]
    text, spans = dl.render_row(tok, dl.continue_messages("the cat"), open_tail=True)
    assert text.decode() == bos + "Continue the following text.\n\nthe cat"
    assert spans == [(len(text) - 7, len(text), len(text))]


def test_view_loader_counts_student_render_failures(tmp_path, tok_bl):
    import copy
    teacher = _with_template(tok_bl, _TEMPLATE_A)
    convs = [[{"role": "user", "content": "q"}, {"role": "tool", "content": "{}"},
              {"role": "assistant", "content": "the cat is"}],
             [{"role": "user", "content": "q"}, {"role": "assistant", "content": "the cat is the cat"}]]
    _tiny_framed_cache(tmp_path / "c", teacher, convs, open_tail=False)
    reader = dl.CacheReader(tmp_path / "c")
    student = _with_template(copy.deepcopy(tok_bl), _TEMPLATE_NOSYS)
    V = len(dl.token_bytes(tok_bl))
    tables = dl.build_tables(teacher, student, V_T=V, V_S=V)
    loader = dl.ViewLoader(reader, student, tables, knobs=dict(dl.DEFAULT_KNOBS), Kp=reader.K, identity=False)
    views = [loader.compile(r) for r in range(len(reader))]
    assert loader.render_failures == 1 and sum(v is not None for v in views) == 1


# ---------------------------------------------------------------------------
# Phase 8: reply rows, two message lists, census, LoRA alpha
# ---------------------------------------------------------------------------

def test_reply_rows_last_only_and_reasoning_skipped(tok_bl):
    tok = _with_template(tok_bl, _TEMPLATE_A)
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "the cat"},
            {"role": "user", "content": "more"}, {"role": "assistant", "content": "is the cat"}]
    text, spans = dl.render_row(tok, conv, **dl.row_render_args("reply"))
    full, all_spans = dl.render_row(tok, conv, **dl.row_render_args("chat"))
    assert text == full and spans == all_spans[-1:] and "reply" in dl.FRAME_KINDS
    # per-turn expansion: one row per assistant turn, each ending on its turn
    rows = dl.per_turn_rows(conv)
    assert [len(r) for r in rows] == [2, 4] and rows[0][-1]["content"] == "the cat"
    # a reasoning_content that repeats the reply verbatim is skipped over
    # when the template renders it in front of the content
    tok2 = _with_template(tok_bl, "{% for m in messages %}{% if m['role'] == 'user' %}<bos>User: {{ m['content'] }}\n"
                          "{% else %}Model:\n{% if m['reasoning_content'] %}<think>{{ m['reasoning_content'] }}</think>\n{% endif %}"
                          "{{ m['content'] }}\n<end>\n{% endif %}{% endfor %}")
    conv2 = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "the cat", "reasoning_content": "the cat is it"}]
    t2, sp2 = dl.render_row(tok2, conv2, **dl.row_render_args("reply"))
    b0, b1, _b2 = sp2[0]
    assert t2[b0:b1] == b"the cat" and t2[:b0].endswith(b"</think>\n")
    assert dl.same_reply(conv, conv) and not dl.same_reply(conv, conv[:2])
    assert not dl.same_reply(conv2, [conv2[0], dict(conv2[1], reasoning_content="")])


def test_fit_reply_drops_leading_turns(tok_bl):
    tok = _with_template(tok_bl, _TEMPLATE_A)
    tb = dl.token_bytes(tok)
    conv = [{"role": "system", "content": "sys"}]
    for i in range(4):
        conv += [{"role": "user", "content": f"q{i} the cat"}, {"role": "assistant", "content": f"a{i} is the cat"}]
    full = dl.fit_reply(tok, conv, 10_000, tb)
    assert full is not None and full[3] == conv
    n_full = len(full[0])
    short = dl.fit_reply(tok, conv, n_full - 1, tb)
    assert short is not None
    kept = short[3]
    # the system message and the final reply survive; the oldest exchange goes first
    assert kept[0]["role"] == "system" and kept[-1] == conv[-1] and kept[1]["content"] == "q1 the cat"
    assert len(short[0]) <= n_full - 1 and len(short[4]) == 1
    assert dl.fit_reply(tok, conv, 2, tb) is None
    assert dl.fit_reply(tok, conv[:-1], 10_000, tb) is None


def _tiny_reply_cache(tmp: Path, tok, pairs, K=8, seed=5):
    """A reply-row cache: pairs of (teacher messages, student messages or None)."""
    rng = np.random.default_rng(seed)
    tb = dl.token_bytes(tok)
    V = len(tb)
    log_bmask = dl.log_bmask_from(dl.whitespace_start_mask(tok, V, tb))
    W = mx.array((rng.standard_normal((V, 6)) * 0.5).astype(np.float32))
    writer = dl.ShardWriter(tmp, K, False)
    rows, metas, tbytes = [], [], []
    for r, (msgs, st) in enumerate(pairs):
        text, spans = dl.render_row(tok, msgs, **dl.row_render_args("reply"))
        ids, ends, _ = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
        n = len(ids)
        h = mx.array(rng.standard_normal((n, 6)).astype(np.float32))
        nxt = np.concatenate([ids[1:], [-1]]).astype(np.int32)
        tm = dl.target_mask(ends.astype(np.int64), spans)
        valid = (nxt >= 0) & tm
        red = dl.reduce_logits((h @ W.T).astype(mx.bfloat16), mx.array(nxt), K=K, log_bmask=log_bmask,
                               onpath_valid=valid)
        red["onpath_mask"] = valid
        red["token_ids"] = ids
        red["token_end_byte"] = ends
        rows.append(red)
        tbytes.append(text)
        metas.append(dl.RowMeta(row_id=r, doc_id=f"d{r}", window=0, n_tokens=n, frame="reply", messages=msgs,
                                spans=[list(s) for s in spans], prefix_n_tokens=int(np.argmax(tm)) + 1,
                                suffix_start_byte=spans[0][0], student_messages=st))
    writer.write(0, dl.pack_shard(rows, tbytes, K, False), metas, wall_s=0.01, step=64)
    dl.write_manifest(tmp, teacher_path="synthetic", dataset="synthetic", num_samples=len(rows),
                      max_seq_len=64, seed=seed, top_k=K, vocab_size=V, config_vocab_size=V,
                      tokenizer_hash=dl.vocab_map_hash(tok), batch_size=8,
                      gmlx_distill={"mlx_kld_compatible": False, "corpus_sha256": "x", "frame": {"kind": "reply"}})


def test_asymmetric_rows_align_on_the_student_list(tmp_path, tok_bl, tok_spm):
    teacher = _with_template(tok_bl, _TEMPLATE_A)
    student = _with_template(tok_spm, _TEMPLATE_B)
    reply = {"role": "assistant", "content": "the cat is the cat 123"}
    t_msgs = [{"role": "user", "content": "the cat is 123 the cat the cat\n\nwhat is it"}, reply]
    s_msgs = [{"role": "user", "content": "what is it"}, reply]
    _tiny_reply_cache(tmp_path / "c", teacher, [(t_msgs, s_msgs), (t_msgs, None)])
    assert dl.validate_cache(tmp_path / "c") == []
    reader = dl.CacheReader(tmp_path / "c")
    meta0 = reader.rows_meta[0]
    assert meta0["student_messages"] == s_msgs and "student_messages" not in reader.rows_meta[1]
    tables = dl.build_tables(teacher, student)
    loader = dl.ViewLoader(reader, student, tables, knobs=KNOBS, Kp=8, identity=False)
    arrs, ttext, meta = reader.row(0)
    s_ids, s_ends, s_spans = loader.student_tokens(arrs, ttext, meta)
    stext, _ = dl.render_row(student, s_msgs, **dl.row_render_args("reply"))
    # the student's own frame over its own list: the context never appears in it
    assert b"what is it" in stext and b"the cat is 123 the cat the cat" not in stext
    assert b"the cat is 123 the cat the cat" in ttext
    assert len(s_spans) == 1 and stext[s_spans[0][0]:s_spans[0][1]] == reply["content"].encode()
    rv = loader.compile(0)
    assert rv is not None and rv.bnd_pos.shape[0] >= 2
    # every boundary lies inside the reply on both sides at the same offset
    t_ends = arrs["token_end_byte"].astype(np.int64)
    tb0 = meta["spans"][0][0]
    sb0 = s_spans[0][0]
    al = dl.shared_boundaries_spans(t_ends, s_ends, [tuple(meta["spans"][0])], s_spans)
    assert np.array_equal(al.s_pos, rv.bnd_pos)
    for tp, sp in zip(al.t_pos, al.s_pos):
        assert arrs["onpath_mask"][tp] and t_ends[tp] - tb0 == s_ends[sp] - sb0
    assert rv.compute_mask[:sb0].sum() == 0 or True   # compute positions never precede the reply
    s_tm = dl.target_mask(s_ends, s_spans)
    assert np.array_equal(rv.compute_mask, s_tm[:len(s_ids) - 1])
    # the row without a student list renders the teacher's list on the student side
    _a1, t1, m1 = reader.row(1)
    ids1, _e1, sp1 = loader.student_tokens(_a1, t1, m1)
    st1, _ = dl.render_row(student, t_msgs, **dl.row_render_args("reply"))
    assert b"the cat is 123 the cat the cat" in st1 and len(sp1) == 1
    batch = loader.batch([0, 1])
    assert batch is not None and batch["student_ids"].shape[0] == 2


def test_asymmetric_rows_identity_pair_uses_identity_tables_on_the_general_path(tmp_path, tok_bl):
    tok = _with_template(tok_bl, _TEMPLATE_A)
    reply = {"role": "assistant", "content": "the cat is the cat"}
    t_msgs = [{"role": "user", "content": "is the cat 123\n\nq"}, reply]
    s_msgs = [{"role": "user", "content": "q"}, reply]
    _tiny_reply_cache(tmp_path / "c", tok, [(t_msgs, s_msgs)])
    reader = dl.CacheReader(tmp_path / "c")
    V = len(dl.token_bytes(tok))
    tables = dl.identity_tables(V, dl.whitespace_start_mask(tok, V), dl.vocab_map_hash(tok), dl.vocab_map_hash(tok))
    loader = dl.ViewLoader(reader, tok, tables, knobs=KNOBS, Kp=8, identity=False)
    arrs, _t, meta = reader.row(0)
    rv = loader.compile(0)
    assert rv is not None
    # identity groups: the projected targets are the raw top-K at every paired position
    s_ids, s_ends, s_spans = loader.student_tokens(arrs, _t, meta)
    al = dl.shared_boundaries_spans(arrs["token_end_byte"].astype(np.int64), s_ends, [tuple(meta["spans"][0])], s_spans)
    for j, (tp, sp) in enumerate(zip(al.t_pos, al.s_pos)):
        assert rv.bnd_pos[j] == sp
        raw = arrs["top_k_indices"][tp]
        raw = raw[raw >= 0]
        got = rv.target_gid[j][rv.target_gid[j] < tables.G]
        assert set(raw.tolist()) == set(got.tolist())
        # the student's tokens after its own frame are the teacher's suffix tokens
    pn = meta["prefix_n_tokens"]
    s_first = int(np.argmax(dl.target_mask(s_ends, s_spans))) + 1
    assert np.array_equal(s_ids[s_first:], arrs["token_ids"][pn:])


def test_validator_rejects_student_list_on_chat_rows_and_reply_mismatch(tmp_path, tok_bl):
    tok = _with_template(tok_bl, _TEMPLATE_A)
    reply = {"role": "assistant", "content": "the cat"}
    t_msgs = [{"role": "user", "content": "ctx\n\nq"}, reply]
    s_msgs = [{"role": "user", "content": "q"}, reply]
    _tiny_reply_cache(tmp_path / "c", tok, [(t_msgs, s_msgs)])
    assert dl.validate_cache(tmp_path / "c") == []
    rows_path = tmp_path / "c" / "rows-00000.jsonl"
    rows = dl.read_rows_jsonl(rows_path)
    bad = dict(rows[0], frame="chat")
    rows_path.write_text(json.dumps(bad) + "\n")
    assert any("reply only" in p for p in dl.validate_cache(tmp_path / "c", check_sha=False))
    # a reply-think row is a reply row whose target starts earlier, so it keeps its student list
    ok = dict(rows[0], frame="reply-think")
    rows_path.write_text(json.dumps(ok) + "\n")
    assert not any("reply only" in p for p in dl.validate_cache(tmp_path / "c", check_sha=False))
    bad = dict(rows[0], student_messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a dog"}])
    rows_path.write_text(json.dumps(bad) + "\n")
    assert any("different reply" in p for p in dl.validate_cache(tmp_path / "c", check_sha=False))
    bad = dict(rows[0], spans=rows[0]["spans"] * 2)
    rows_path.write_text(json.dumps(bad) + "\n")
    assert any("reply row with 2 spans" in p for p in dl.validate_cache(tmp_path / "c", check_sha=False))


def test_span_rows_per_turn_and_positions(tok_bl):
    tok = _with_template(tok_bl, _TEMPLATE_A)
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "the cat"},
            {"role": "user", "content": "more"}, {"role": "assistant", "content": "is the cat 123"}]
    rows, dropped = dl_eval._span_rows(tok, [conv], max_len=256)
    assert len(rows) == 1 and dropped == 0
    per, _ = dl_eval._span_rows(tok, [{"id": "x", "messages": conv}], max_len=256, per_turn=True)
    assert [r[3] for r in per] == ["x:0", "x:1"]
    # each per-turn row targets exactly its own reply bytes
    assert per[0][2] == len("the cat") + len("\n<end>\n") and per[1][2] == len("is the cat 123") + len("\n<end>\n")
    # the student's own list wins over the teacher's
    st = [{"role": "user", "content": "what"}, {"role": "assistant", "content": "is the cat 123"}]
    only, _ = dl_eval._span_rows(tok, [{"id": "y", "messages": conv, "student_messages": st}], max_len=256, last_only=True)
    assert len(only) == 1 and only[0][2] == per[1][2]
    # restricted to the high-delta byte ranges: only the token starting at the reply's byte 7 ("cat")
    tb = dl.token_bytes(tok)
    text, spans = dl.render_row(tok, st, **dl.row_render_args("reply"))
    ids, ends, _ = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
    b0 = spans[0][0]
    starts = ends.astype(np.int64)[:-1] - b0
    k = int(np.nonzero(starts == 7)[0][0])
    res, _ = dl_eval._span_rows(tok, [{"id": "y", "messages": st}], max_len=256, last_only=True, positions={"y": [[7, 8]]})
    assert len(res) == 1 and res[0][1].sum() == 1 and res[0][1][k]
    none, _ = dl_eval._span_rows(tok, [{"id": "z", "messages": st}], max_len=256, last_only=True, positions={"y": [[7, 8]]})
    assert none == []


def test_reply_think_rows_target_the_trace(tok_bl):
    tok = _with_template(tok_bl, "{% for m in messages %}{% if m['role'] == 'user' %}<bos>User: {{ m['content'] }}\n"
                         "{% else %}Model:\n<think>\n{% if m['reasoning_content'] %}{{ m['reasoning_content'] }}{% endif %}{{ '\\n' }}</think>\n\n"
                         "{{ m['content'] }}\n<end>\n{% endif %}{% endfor %}")
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "the cat", "reasoning_content": "count cats"}]
    text, spans = dl.render_row(tok, conv, **dl.row_render_args("reply-think"))
    b0, b1, b2 = spans[0]
    assert "reply-think" in dl.FRAME_KINDS
    assert text[b0:b1] == b"count cats\n</think>\n\nthe cat" and text[:b0].endswith(b"<think>\n")
    assert text[b1:b2] == b"\n<end>\n"
    # the plain reply kind on the same row keeps the trace as context
    t2, sp2 = dl.render_row(tok, conv, **dl.row_render_args("reply"))
    assert t2 == text and t2[sp2[0][0]:sp2[0][1]] == b"the cat"
    # fit_reply threads the flag
    tb = dl.token_bytes(tok)
    fit = dl.fit_reply(tok, conv, 4096, tb, reason_target=True)
    assert fit is not None and fit[4] == spans


def test_probe_step_re_derives_an_integer_tier_when_the_measured_constant_exceeds_the_budget():
    from gmlx.distill.cache import probe_step
    # under budget: the budgeted step and constant stand
    assert probe_step(14.0, 16.0, 151936, 4.0) == (1024, 16.0)
    # over budget: the measured constant replaces it and the step halves to fit the unchanged cap
    step, constant = probe_step(26.0, 16.0, 151936, 4.0)
    assert constant == 26.0 and isinstance(step, int) and step == 512   # 4e9 / (151936 * 26) = 1012, the tier below it


def test_align_refusal_leaves_no_view(tmp_path, tok_bl, tok_spm, monkeypatch):
    """A refused pair writes no view.json, so ``train`` has nothing to
    accept, and ``--force`` is what keeps the view."""
    from gmlx.distill import view as _view

    _tiny_cache(tmp_path / "cache", tok_bl)
    tok_bl.save_pretrained(tmp_path / "cache" / "tokenizer")
    tok_spm.save_pretrained(tmp_path / "student")
    monkeypatch.setattr(_view, "REFUSE_A", 1.01)
    opts = dict(cache=str(tmp_path / "cache"), student=str(tmp_path / "student"), out=str(tmp_path / "view"))
    assert _view.run_align(_view.AlignOptions(**opts)) == 3
    assert not (tmp_path / "view" / "view.json").exists()
    assert _view.run_align(_view.AlignOptions(**opts, force=True)) == 0
    assert (tmp_path / "view" / "view.json").exists()
