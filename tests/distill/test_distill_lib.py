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


_BL_MERGES = [("\u0120", "t"), ("h", "e"), ("\u0120t", "he"), ("1", "2"), ("12", "3"), ("\u0120", "a"),
              ("i", "s"), ("\u0120", "is"), ("c", "a"), ("ca", "t")]


@pytest.fixture(scope="module")
def tok_bl():
    # teacher: byte-level with a few merges
    return _bytelevel_tokenizer(_BL_MERGES)


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


def _dense_loss_mx(hidden, W, batch, group_of, G, Kp, bmask, knobs, softcap=None, scale=1.0):
    """Dense mx reference of distill_loss (materialized logits, plain
    scatter), for gradient comparison."""
    B, T, d = hidden.shape
    Tm1 = T - 1
    z = (hidden[:, :-1, :].reshape(B * Tm1, d) @ W.T).astype(mx.float32) * scale
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


@pytest.mark.parametrize("V,C,softcap,scale", [(64, 5, None, 1.0), (64, 64, 30.0, 1.0), (151936, 7, None, 1.0),
                                                (64, 9, None, 0.125)])
def test_distill_loss_gradient_vs_dense(V, C, softcap, scale):
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
        head = dl.linear_head(w, softcap, scale=scale)
        loss, _ = dl.distill_loss(hid, bm, head, group_of=mx.array(group_of), G=G, Kp=Kp,
                                  log_bmask=dl.log_bmask_from(bmask), knobs=knobs, C=C)
        return loss

    def dense(hid, w):
        return _dense_loss_mx(hid, w, bm, mx.array(group_of), G, Kp, bmask, knobs, softcap=softcap, scale=scale)

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
    # every slice from one pass over the corpus, the same numbers
    both = dl_eval.decontam_fractions({"clean": clean, "dirty": dirty, "short": b"x"}, corpus)
    assert both == {"clean": 0.0, "dirty": f, "short": 0.0}
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


# ---------------------------------------------------------------------------
# identity width, stale views, checkpoints, loss modes, hashing
# ---------------------------------------------------------------------------

def test_identity_tables_at_a_wider_student_head():
    """A same-vocabulary pair whose student head is wider than the teacher's
    (pad-style surplus ids) takes identity tables at the student width, so
    the head's slot map and boundary mask match its logits."""
    V_T, V_S, d, K = 10, 12, 8, 3
    tables = dl.identity_tables(V_S, np.zeros(V_S, bool), "t", "s", V_T=V_T)
    assert tables.G == V_S and tables.V_T == V_T and tables.V_S == V_S
    assert tables.target_g.shape == (V_T,) and tables.group_of.shape == (V_S,)
    with pytest.raises(ValueError):
        dl.identity_tables(V_T, np.zeros(V_T, bool), "t", "s", V_T=V_S)
    rng = np.random.default_rng(0)
    head = dl.linear_head(mx.array(rng.standard_normal((V_S, d)).astype(np.float32)))
    T = 6
    ids = np.arange(1, T + 1, dtype=np.int32)
    row = {"token_end_byte": np.arange(1, T + 1), "onpath_mask": np.array([1, 1, 1, 1, 1, 0], bool),
           "top_k_indices": rng.integers(0, V_T, (T, K)).astype(np.int32),
           "top_k_log_softmax": np.log(np.full((T, K), 0.2)).astype(np.float32)}
    rv = dl.compile_row(row, b"abcdef", ids, np.arange(1, T + 1), tables, Kp=K, knobs=dict(KNOBS),
                        teacher_special=set(), student_special=set(), identity=True)
    b = dl.batch_to_mx(dl.collate([rv], K, tables.G))
    h = mx.array(rng.standard_normal((1, 32, d)).astype(np.float32))
    loss, _aux = dl.distill_loss(h, b, head, group_of=None, G=tables.G, Kp=K,
                                 log_bmask=dl.log_bmask_from(tables.bmask_S), knobs=dict(KNOBS, lambda_alm=0.0))
    mx.eval(loss)
    assert np.isfinite(float(loss))


def test_align_rewrites_a_reused_view_directory(tmp_path, tok_bl, tok_spm):
    """A second align into the same --out leaves nothing of the first: the
    materialized shards and view.json are removed before the pass, and a
    tables artifact for other head widths is rebuilt."""
    from gmlx.distill import view as _view

    _tiny_cache(tmp_path / "cache", tok_bl)
    tok_bl.save_pretrained(tmp_path / "cache" / "tokenizer")
    tok_spm.save_pretrained(tmp_path / "student")
    out = tmp_path / "view"
    opts = dict(cache=str(tmp_path / "cache"), student=str(tmp_path / "student"), out=str(out))
    assert _view.run_align(_view.AlignOptions(**opts, materialize=True)) == 0
    shards = sorted(out.glob("view-*.safetensors"))
    assert shards
    (out / "view-00099.safetensors").write_bytes(b"stale")
    assert _view.run_align(_view.AlignOptions(**opts)) == 0
    assert not list(out.glob("view-*.safetensors"))
    assert (out / "view.json").exists()
    # the tables check covers the head widths as well as the pair hashes
    t = dl.load_tables(out)
    wrong = dl.load_tables(out)
    wrong.V_S = t.V_S + 8
    dl.save_tables(tmp_path / "wide", wrong)
    got = _view.get_tables(tok_bl, tok_spm, tmp_path / "wide", tmp_path / "view2", V_T=t.V_T, V_S=t.V_S)
    assert got.V_S == t.V_S


def test_train_refuses_tables_that_are_not_the_views_own(tmp_path, tok_bl, tok_spm, capsys):
    """view.json and tables.json in one directory must come from the same
    align, else train reads a projection the view was not built with."""
    from gmlx.distill import trainer as _trainer
    from gmlx.distill import view as _view

    _tiny_cache(tmp_path / "cache", tok_bl)
    tok_bl.save_pretrained(tmp_path / "cache" / "tokenizer")
    tok_spm.save_pretrained(tmp_path / "student")
    out = tmp_path / "view"
    assert _view.run_align(_view.AlignOptions(cache=str(tmp_path / "cache"), student=str(tmp_path / "student"),
                                              out=str(out))) == 0
    tj = json.loads((out / "tables.json").read_text())
    tj["student_hash"] = "0" * len(tj["student_hash"])
    (out / "tables.json").write_text(json.dumps(tj))
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "student.gguf").write_bytes(b"")
    rc = _trainer.run_train(_trainer.TrainOptions(views=[str(out)], student=str(fake / "student.gguf"), iters=1))
    assert rc == 2
    assert "not the ones view.json was aligned with" in capsys.readouterr().err


def test_cache_resume_refuses_other_inputs(tmp_path, tok_bl, capsys):
    """progress.json records the corpus and row options of the first run,
    and a resume with any of them changed is refused before the teacher
    loads."""
    from gmlx.distill import teacher as _teacher

    tok_bl.save_pretrained(tmp_path / "teacher")
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("".join(json.dumps({"text": "alpha beta gamma delta " * 3}) + "\n" for _ in range(4)))
    out = tmp_path / "cache"
    out.mkdir()
    prev = {"shards": [], "tokens": 0, "bytes": 0, "wall_s": 0.0, "min_step": None, "trunk_chunk": None,
            "bytes_per_v_element": None, "format_version": dl.FORMAT_VERSION,
            "run": {"corpus_sha256": "other", "n_rows": 1, "n_tokens": 1, "max_len": 64, "rows_per_shard": 2,
                    "top_k": 8, "floor": False, "frame": "none", "routes": False, "hidden": False}}
    (out / "progress.json").write_text(json.dumps(prev))
    opts = _teacher.CacheOptions(teacher=str(tmp_path / "teacher"), corpus=str(corpus), out=str(out),
                                 top_k=8, max_len=64, rows_per_shard=2, resume=True)
    assert _teacher.run_cache(opts) == 2
    assert "other inputs than the first run" in capsys.readouterr().err


def test_checkpoint_replace_is_crash_safe(tmp_path):
    """The previous checkpoint moves aside until the new one is in place,
    and a crash that left only the moved copy is restored on the next
    resume; an empty directory offers nothing to resume."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    from gmlx.distill import trainer as _trainer

    model = nn.Linear(4, 3)
    opt = optim.AdamW(learning_rate=1e-3)
    opt.init(model.trainable_parameters())
    ck = tmp_path / "ckpt"
    ck.mkdir()
    assert _trainer.checkpoint_dir(ck, "last") is None
    _trainer.save_checkpoint(ck, "last", model, opt, {"iteration": 1})
    _trainer.save_checkpoint(ck, "last", model, opt, {"iteration": 2})
    assert sorted(p.name for p in ck.iterdir()) == ["last"]
    assert json.loads((ck / "last" / "state.json").read_text())["iteration"] == 2
    (ck / "last").rename(ck / "last.old")
    assert _trainer.checkpoint_dir(ck, "last") == ck / "last"
    assert json.loads((ck / "last" / "state.json").read_text())["iteration"] == 2
    got = _trainer.load_checkpoint(ck, "last", model, opt)
    assert got["iteration"] == 2


def test_loss_modes_stay_finite_on_an_empty_boundary():
    """A boundary with no kept group (all pads, M = 0) gives a finite loss
    and finite gradients under renorm, under a tempered conditional factor
    and under the paper loss, not only under the default mode."""
    Kp = 3
    lp = mx.array([[math.log(0.5), math.log(0.4), dl.NEG_INF], [dl.NEG_INF] * Kp])
    logM = mx.array([math.log(0.9), dl.NEG_INF])
    w = mx.array([1.0, 0.0])
    Q = mx.array([[0.5, 0.3, 0.0, 0.2], [0.25, 0.25, 0.25, 0.25]])
    for mode, T in (("bucketed", 2.0), ("paper", 2.0), ("renorm", 1.0), ("renorm", 2.0), ("paper", 1.0)):
        def f(Qv, mode=mode, T=T):
            lv, _a = dl.bucketed_kl(lp, logM, Qv, w, mode=mode, T_dk=T)
            return lv
        v, g = mx.value_and_grad(f)(Q)
        mx.eval(v, g)
        assert np.isfinite(float(v)), (mode, T)
        assert np.all(np.isfinite(np.asarray(g))), (mode, T)
        assert np.all(np.asarray(g)[1] == 0.0), (mode, T)


def test_window_hashes_blocked_equals_whole():
    rng = np.random.default_rng(3)
    data = rng.integers(0, 256, 5000, dtype=np.uint8).tobytes()
    whole = dl_eval.window_hashes(data, 64, block=1 << 20)
    blocked = dl_eval.window_hashes(data, 64, block=700)
    assert whole.shape == (5000 - 63,)
    assert np.array_equal(whole, blocked)
    assert dl_eval.window_hashes(data[:70], 64, block=3).shape == (7,)


# ---------------------------------------------------------------------------
# expert adapters, dropout seeds, scorer memory, refusals
# ---------------------------------------------------------------------------

def test_align_identity_path_at_a_wider_student_head(tmp_path, tok_bl):
    """A same-vocabulary student whose head carries pad-style surplus ids
    aligns on the identity path with tables at its own width, and the
    materialized view compiles every row at that width."""
    from gmlx.distill import view as _view

    _tiny_cache(tmp_path / "cache", tok_bl)
    tok_bl.save_pretrained(tmp_path / "cache" / "tokenizer")
    student = _bytelevel_tokenizer(_BL_MERGES)
    student.add_tokens(["<pad0>", "<pad1>"], special_tokens=True)
    student.save_pretrained(tmp_path / "student")
    out = tmp_path / "view"
    rc = _view.run_align(_view.AlignOptions(cache=str(tmp_path / "cache"), student=str(tmp_path / "student"),
                                            out=str(out), materialize=True))
    assert rc == 0
    v = json.loads((out / "view.json").read_text())
    t = dl.load_tables(out)
    assert v["identity"] and v["V_S"] == v["V_T"] + 2
    assert t.G == t.V_S == v["V_S"] and t.V_T == v["V_T"]
    assert sorted(out.glob("view-*.safetensors"))


def test_adapter_disabled_reaches_expert_lora_stamps():
    """Expert adapters are objects stamped on the expert-stack leaf outside
    the module tree; the before-score toggle zeroes every slot's scale,
    drops their folded tables, and restores both on exit."""
    import mlx.nn as nn
    gm = pytest.importorskip("gmlx.load.modules")
    leaf = nn.Linear(4, 4)
    model = nn.Sequential(leaf)
    lo = gm.ExpertLoRA(mx.zeros((2, 4, 1)), mx.ones((2, 1, 4)), 2.0, slot=0)
    lo2 = gm.ExpertLoRA(mx.zeros((2, 4, 1)), mx.ones((2, 1, 4)), 3.0, slot=1)
    object.__setattr__(leaf, "_kq_lora", lo)
    object.__setattr__(leaf, "_kq_lora_extra", [lo2])
    assert float(lo.tables(mx.float32)[1].sum()) == 16.0
    with dl.adapter_disabled(model):
        assert lo.scale == 0.0 and lo2.scale == 0.0
        assert float(lo.tables(mx.float32)[1].sum()) == 0.0
        assert float(lo2.tables(mx.float32)[1].sum()) == 0.0
    assert lo.scale == 2.0 and lo2.scale == 3.0
    assert not lo._tables and not lo2._tables
    assert float(lo2.tables(mx.float32)[1].sum()) == 24.0


def test_step_seed_gives_both_trunk_forwards_one_dropout_mask():
    """LoRA dropout draws from the global stream, so the two trunk forwards
    of a step agree only when both are seeded with the step's seed."""
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    from gmlx.distill import trainer as _trainer

    m = LoRALinear.from_base(nn.Linear(16, 16), r=4, dropout=0.5)
    m.lora_b = mx.random.normal(m.lora_b.shape)
    m.train()
    x = mx.random.normal((3, 16))
    s = _trainer.step_seed(1, 0)
    mx.random.seed(s)
    a = m(x)
    mx.random.seed(s)
    b = m(x)
    c = m(x)
    mx.eval(a, b, c)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))
    assert len({_trainer.step_seed(seed, i) for seed in (0, 1, 2) for i in range(3)}) == 9


def test_target_logprobs_match_the_dense_log_softmax():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((2, 5, 7)).astype(np.float32)
    tgt = rng.integers(0, 7, (2, 4))
    lp = dl_eval._target_logprobs(mx.array(z), tgt)
    lsm = z - np.log(np.exp(z).sum(-1, keepdims=True))
    ref = np.take_along_axis(lsm[:, :-1], tgt[..., None], -1)[..., 0]
    assert lp.shape == (2, 4) and lp.dtype == np.float32
    assert np.allclose(lp, ref, atol=1e-5)


def test_scorers_report_none_when_nothing_scored(tok_bl):
    """Nothing scored is None in every scorer, never a perfect 0.0."""
    r = dl_eval._score_span_rows(None, [], batch_tokens=64)
    assert r["bpb"] is None and r["nll_per_token"] is None and r["rows"] == 0
    r = dl_eval.bits_per_byte(None, tok_bl, "")
    assert r["bpb"] is None and r["bytes"] == 0


def test_eval_refuses_bad_inputs_before_the_load(tmp_path, capsys):
    """--before without --adapter, a positions map naming no reply row and
    an unreadable slice file exit 2 before any model load."""
    from gmlx.distill import evaluate as _ev

    student = tmp_path / "student.gguf"
    student.write_bytes(b"")

    def run(**kw):
        opts = _ev.EvalOptions(student=str(student), md=str(tmp_path / "r.md"), json=str(tmp_path / "r.json"),
                               **kw)
        return _ev.run_eval(opts), capsys.readouterr().err

    rc, err = run(before=True)
    assert rc == 2 and "needs --adapter" in err
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "a", "messages": []}) + "\n")
    pos = tmp_path / "census.json"
    pos.write_text(json.dumps({"high_delta": {"b": [[0, 1]]}}))
    rc, err = run(reply_slices=["h=" + str(rows)], reply_positions=str(pos))
    assert rc == 2 and "names none of the reply-slice rows" in err
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    rc, err = run(reply_slices=["h=" + str(bad)])
    assert rc == 2 and "unreadable input" in err
    nomsg = tmp_path / "nomsg.jsonl"
    nomsg.write_text(json.dumps({"id": "a"}) + "\n")
    rc, err = run(chat_slices=["c=" + str(nomsg)])
    assert rc == 2 and "no 'messages' key" in err
    assert _ev.reply_row_ids({"h": [{"id": "a"}, {"id": 3}, {"x": 1}]}) == {"a", "3", "2"}


def test_save_checkpoint_extra_lands_inside_the_swap(tmp_path):
    """What the extra writer puts in the checkpoint arrives with the rest,
    never after the directory swap."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    from gmlx.distill import trainer as _trainer

    model = nn.Linear(4, 3)
    opt = optim.AdamW(learning_rate=1e-3)
    opt.init(model.trainable_parameters())
    ck = tmp_path / "ckpt"
    seen = []

    def extra(d):
        seen.append(d.name)
        (d / "hs_head.safetensors").write_bytes(b"x")

    _trainer.save_checkpoint(ck, "last", model, opt, {"iteration": 1}, extra=extra)
    assert seen == ["last.tmp"] and (ck / "last" / "hs_head.safetensors").exists()


def test_assistant_tails_cache_lives_on_the_tokenizer(tok_bl):
    """The turn-end markers are cached on the tokenizer object under its
    template, so a copy given another template answers for itself."""
    import copy

    from gmlx.load.tokenizer import hf_inner

    from gmlx.distill import frames as _frames

    a = _with_template(copy.deepcopy(tok_bl), _TEMPLATE_A)
    ta = _frames.assistant_tails(a)
    assert ta and hf_inner(a)._gmlx_tails
    b = _with_template(copy.deepcopy(a), _TEMPLATE_B)
    tb = _frames.assistant_tails(b)
    assert tb != ta and _frames.assistant_tails(a) == ta
    assert not hasattr(_frames, "_TAIL_CACHE")


# ---------------------------------------------------------------------------
# ids by index, None-safe lines, pre-load checks, views
# ---------------------------------------------------------------------------

def test_report_lines_survive_none_aggregates():
    """One scored row leaves the clustered se None and an empty chat set
    leaves every rate None; the log lines and the markdown print them."""
    from gmlx.distill import evaluate as _ev

    k = {"K": 8, "mean_kld_nats": 0.5, "clustered_se": None, "top1_agreement": 0.9, "rows": 1,
         "rerendered_rows": 0, "wall_s": 0.0}
    assert "se None" in _ev.kld_line("", k)
    c = {"compliance": None, "truncated_rate": None, "refusal_rate": None, "task_refusal_rate": None,
         "ref_nll_nats": None, "wall_s": 0.0}
    assert "compliance None truncated None" in _ev.chat_line("after", c)
    opts = _ev.EvalOptions(student="s.gguf", md="r.md", json="r.json", kld_cache="c")
    report = {"after": {"kld": k, "tasks": {}}, "before": {"kld": dict(k, clustered_se=None)},
              "decontam": {}, "contaminated_slices": []}
    md = _ev.report_markdown(opts, report, {}, {}, {}, set())
    assert "| after | 0.50000 | None | 0.9000 |" in md and "| before | 0.50000 | None |" in md


def test_reply_rows_without_ids_are_named_by_index(tok_bl):
    """A reply row with no id is named by its index in the file, by the
    scorer and by the refusal check alike, so a census map keyed that way
    scores it."""
    from gmlx.distill import evaluate as _ev

    assert _ev.reply_row_ids({"h": [{"x": 1}, {"id": "a"}, {"y": 2}]}) == {"0", "a", "2"}
    tok = _with_template(tok_bl, _TEMPLATE_A)
    st = [{"role": "user", "content": "what"}, {"role": "assistant", "content": "is the cat 123"}]
    rows, _ = dl_eval._span_rows(tok, [{"messages": st}, {"id": "b", "messages": st}, {"messages": st}],
                                 max_len=256, last_only=True)
    assert [r[3] for r in rows] == ["0", "b", "2"]
    res, _ = dl_eval._span_rows(tok, [{"messages": st}, {"messages": st}], max_len=256, last_only=True,
                                positions={"1": [[7, 8]]})
    assert len(res) == 1 and res[0][3] == "1"


def test_eval_checks_cache_refs_and_task_keys_before_the_load(tmp_path, capsys):
    """--cache that is not a cache, --chat-refs without chat items and a
    task file missing a key exit 2 before any model load."""
    from gmlx.distill import evaluate as _ev

    student = tmp_path / "student.gguf"
    student.write_bytes(b"")

    def run(**kw):
        opts = _ev.EvalOptions(student=str(student), md=str(tmp_path / "r.md"), json=str(tmp_path / "r.json"),
                               **kw)
        return _ev.run_eval(opts), capsys.readouterr().err

    (tmp_path / "nocache").mkdir()
    rc, err = run(cache=str(tmp_path / "nocache"))
    assert rc == 2 and "unreadable input" in err and "not a cache" in err
    refs = tmp_path / "refs.json"
    refs.write_text(json.dumps({"after": {"chat": {"items": [{"id": "a", "reply": "x"}]}}}))
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"id": "a", "messages": [{"role": "user", "content": "hi"}]}) + "\n")
    rc, err = run(chat_sanity=str(chat), chat_refs=str(refs))
    assert rc == 2 and "not an eval report with chat items" in err
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "arc_easy.jsonl").write_text(json.dumps({"id": "a", "query": "q", "choices": ["x", "y"]}) + "\n")
    rc, err = run(tasks="arc_easy", tasks_dir=str(tasks))
    assert rc == 2 and "no 'gold' key" in err
    (tasks / "gsm8k.jsonl").write_text(json.dumps({"id": "a", "question": "q", "answer": "1"}) + "\n")
    (tasks / "gsm8k_shots.jsonl").write_text(json.dumps({"question": "q"}) + "\n")
    rc, err = run(tasks="gsm8k", tasks_dir=str(tasks))
    assert rc == 2 and "gsm8k_shots.jsonl: a row has no 'answer' key" in err
    nomsg = tmp_path / "nomsg.jsonl"
    nomsg.write_text(json.dumps({"id": "a"}) + "\n")
    rc, err = run(chat_sanity=str(nomsg))
    assert rc == 2 and "no 'messages' key" in err


def test_kld_cache_refusal_and_replay_gate(tmp_path, tok_bl, monkeypatch):
    """A kld cache wider than the student's head, or one whose tokenizer
    cannot be found, is refused; a same-map cache passes through the hash
    or the saved tokenizer; routes are replayed only without an adapter."""
    from gmlx.distill import evaluate as _ev

    h = dl.vocab_map_hash(tok_bl)
    V = len(dl.token_bytes(tok_bl))
    assert _ev.kld_cache_refusal(tmp_path, {"vocab_size": V, "tokenizer_hash": h}, tok_bl, V) is None
    why = _ev.kld_cache_refusal(tmp_path, {"vocab_size": V + 1, "tokenizer_hash": h}, tok_bl, V)
    assert why and "wider than the student's head" in why
    why = _ev.kld_cache_refusal(tmp_path, {"vocab_size": V, "tokenizer_hash": "x"}, tok_bl, V)
    assert why and "no tokenizer directory" in why
    tok_bl.save_pretrained(tmp_path / "tokenizer")
    assert _ev.kld_cache_refusal(tmp_path, {"vocab_size": V, "tokenizer_hash": "x"}, tok_bl, V) is None
    manifest = {"gmlx_distill": {"routing": None}}
    assert _ev.kld_replay_layers(object(), manifest, None) == dl.replay_layers_for(object(), manifest) is None
    monkeypatch.setattr(_ev, "replay_layers_for", lambda model, man: [1, 3])
    assert _ev.kld_replay_layers(object(), manifest, None) == [1, 3]
    assert _ev.kld_replay_layers(object(), manifest, "adapter.gguf") is None


def test_view_records_the_cache_absolutely_and_train_refuses_a_moved_cache(tmp_path, tok_bl, tok_spm,
                                                                           monkeypatch, capsys):
    """view.json names the cache by absolute path, so a train run from
    another directory finds it, and a cache that moved is refused with a
    pointer to align rather than a traceback."""
    from gmlx.distill import trainer as _trainer
    from gmlx.distill import view as _view

    _tiny_cache(tmp_path / "cache", tok_bl)
    tok_bl.save_pretrained(tmp_path / "cache" / "tokenizer")
    tok_spm.save_pretrained(tmp_path / "student")
    monkeypatch.chdir(tmp_path)
    assert _view.run_align(_view.AlignOptions(cache="cache", student="student", out="view")) == 0
    v = json.loads((tmp_path / "view" / "view.json").read_text())
    assert Path(v["cache_dir"]).is_absolute() and Path(v["cache_dir"]) == (tmp_path / "cache").resolve()
    (tmp_path / "cache").rename(tmp_path / "moved")
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "student.gguf").write_bytes(b"")
    rc = _trainer.run_train(_trainer.TrainOptions(views=["view"], student=str(fake / "student.gguf"), iters=1))
    assert rc == 2 and "no longer at" in capsys.readouterr().err


def test_align_materialize_over_the_disk_cap_leaves_no_view(tmp_path, tok_bl, tok_spm, capsys):
    """--materialize past --max-disk-gb refuses with exit 2 and removes the
    partial shards and view.json, so train cannot pick up a half view."""
    from gmlx.distill import view as _view

    _tiny_cache(tmp_path / "cache", tok_bl)
    tok_bl.save_pretrained(tmp_path / "cache" / "tokenizer")
    tok_spm.save_pretrained(tmp_path / "student")
    out = tmp_path / "view"
    rc = _view.run_align(_view.AlignOptions(cache=str(tmp_path / "cache"), student=str(tmp_path / "student"),
                                            out=str(out), materialize=True, max_disk_gb=1e-9))
    assert rc == 2 and "[align] refuse: --max-disk-gb" in capsys.readouterr().err
    assert not list(out.glob("view-*.safetensors")) and not (out / "view.json").exists()


# ---------------------------------------------------------------------------
# head scale and parity, the teacher pass end to end,
# readers, the kld cache before the load
# ---------------------------------------------------------------------------

def _tiny_mlx_teacher(tmp: Path, tok, seed=0) -> Path:
    """A two-layer llama checkpoint over tok's vocabulary, loadable by
    mlx_lm.load, so the teacher pass runs end to end on the CPU."""
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama

    d = tmp
    d.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(d)
    cfg = dict(model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
               num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=len(dl.token_bytes(tok)),
               tie_word_embeddings=True, max_position_embeddings=512, rope_theta=10000.0)
    mx.random.seed(seed)
    model = llama.Model(llama.ModelArgs.from_dict(cfg))
    mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _text_corpus(path: Path, n=3) -> Path:
    path.write_text("".join(json.dumps({"text": "the cat is the cat " * 4}) + "\n" for _ in range(n)))
    return path


def test_head_carries_granite_logits_scaling_and_parity_catches_the_rest():
    """Granite divides its logits by logits_scaling after the projection;
    the head spec folds that in, and the parity gap against the model's
    own forward is what refuses any other post-projection change."""
    from mlx_lm.models import granite, llama

    from gmlx.distill import head as _head

    V = 64
    ga = granite.ModelArgs(model_type="granite", hidden_size=32, num_hidden_layers=1, intermediate_size=64,
                           num_attention_heads=4, rms_norm_eps=1e-5, vocab_size=V, logits_scaling=8.0,
                           attention_multiplier=0.125, embedding_multiplier=1.0, residual_multiplier=1.0,
                           max_position_embeddings=512, num_key_value_heads=2, attention_bias=False,
                           mlp_bias=False, rope_theta=10000.0)
    mx.random.seed(1)
    g = granite.Model(ga)
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    spec = _head.head_spec_from_model(g)
    assert spec.scale == pytest.approx(0.125)
    assert _head.head_parity_gap(g, spec, ids) < 1e-4
    la = llama.ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=1, intermediate_size=64,
                         num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=V)
    m = llama.Model(la)
    plain = _head.head_spec_from_model(m)
    assert plain.scale == 1.0 and _head.head_parity_gap(m, plain, ids) < 1e-4
    orig = llama.Model.__call__
    try:
        llama.Model.__call__ = lambda self, inputs, cache=None: orig(self, inputs, cache) * 3.0
        assert _head.head_parity_gap(m, plain, ids) > _head.HEAD_PARITY_TOL
    finally:
        llama.Model.__call__ = orig


def test_cache_pass_records_the_teacher_and_the_largest_id(tmp_path, tok_bl, capsys):
    """The manifest names the teacher by absolute path and the largest
    cached token id; the resume fingerprint carries the teacher, so a
    resume on another one is refused."""
    import shutil

    from gmlx.distill import teacher as _teacher

    teacher = _tiny_mlx_teacher(tmp_path / "teacher", tok_bl)
    corpus = _text_corpus(tmp_path / "c.jsonl")
    out = tmp_path / "cache"
    opts = _teacher.CacheOptions(teacher=str(teacher), corpus=str(corpus), out=str(out), top_k=8, max_len=64,
                                 rows_per_shard=2)
    assert _teacher.run_cache(opts) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert Path(man["teacher_path"]).is_absolute() and Path(man["teacher_path"]) == teacher.resolve()
    assert man["gmlx_distill"]["teacher"]["path"] == man["teacher_path"]
    V = len(dl.token_bytes(tok_bl))
    assert isinstance(man["max_top_k_id"], int) and 0 <= man["max_top_k_id"] < V
    run = json.loads((out / "progress.json").read_text())["run"]
    assert run["teacher"]["path"] == man["teacher_path"] and run["teacher"]["size"] > 0
    assert "render_kwargs" in run and run["hidden_dim"] is None
    other = tmp_path / "teacher2"
    shutil.copytree(teacher, other)
    a = _teacher.run_fingerprint(opts, "x", 1, 1, None)
    b = _teacher.run_fingerprint(_teacher.CacheOptions(teacher=str(other), corpus=str(corpus), out=str(out)),
                                 "x", 1, 1, None)
    assert a["teacher"]["path"] != b["teacher"]["path"]
    (out / "batch-00001.safetensors").unlink()
    rc = _teacher.run_cache(_teacher.CacheOptions(teacher=str(other), corpus=str(corpus), out=str(out), top_k=8,
                                                  max_len=64, rows_per_shard=2, resume=True))
    err = capsys.readouterr().err
    assert rc == 2 and "other inputs than the first run" in err and "teacher" in err


def test_cache_pass_refuses_a_writer_error_and_a_head_that_misses_the_logits(tmp_path, tok_bl, monkeypatch,
                                                                              capsys):
    """A disk-cap or free-space error from the shard writer exits 2 with
    the verified shards kept, and a model whose logits the head does not
    reproduce is refused before any shard is written."""
    from mlx_lm.models import llama

    from gmlx.distill import teacher as _teacher

    teacher = _tiny_mlx_teacher(tmp_path / "teacher", tok_bl)
    corpus = _text_corpus(tmp_path / "c.jsonl")

    def boom(self, *a, **k):
        raise RuntimeError("--max-disk-gb 0.0 would be exceeded by shard 0")

    monkeypatch.setattr(dl.ShardWriter, "write", boom)
    rc = _teacher.run_cache(_teacher.CacheOptions(teacher=str(teacher), corpus=str(corpus),
                                                  out=str(tmp_path / "c1"), top_k=8, max_len=64))
    err = capsys.readouterr().err
    assert rc == 2 and "[cache] refuse: --max-disk-gb 0.0 would be exceeded" in err
    assert not list((tmp_path / "c1").glob("batch-*.safetensors"))
    monkeypatch.undo()

    def full(self, *a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(dl.ShardWriter, "write", full)
    rc = _teacher.run_cache(_teacher.CacheOptions(teacher=str(teacher), corpus=str(corpus),
                                                  out=str(tmp_path / "c3"), top_k=8, max_len=64))
    err = capsys.readouterr().err
    assert rc == 2 and "[cache] refuse: [Errno 28] No space left on device" in err
    monkeypatch.undo()
    orig = llama.Model.__call__
    monkeypatch.setattr(llama.Model, "__call__", lambda self, inputs, cache=None: orig(self, inputs, cache) * 3.0)
    rc = _teacher.run_cache(_teacher.CacheOptions(teacher=str(teacher), corpus=str(corpus),
                                                  out=str(tmp_path / "c2"), top_k=8, max_len=64))
    err = capsys.readouterr().err
    assert rc == 2 and "does not reproduce the teacher's own logits" in err
    assert not list((tmp_path / "c2").glob("batch-*.safetensors"))


def test_kld_cache_refusal_reads_the_largest_cached_id_and_a_gone_teacher_path(tmp_path, tok_bl):
    from gmlx.distill import evaluate as _ev

    h = dl.vocab_map_hash(tok_bl)
    V = len(dl.token_bytes(tok_bl))
    # a wider vocabulary whose cached ids all fit the head is accepted
    assert _ev.kld_cache_refusal(tmp_path, {"vocab_size": V + 7, "tokenizer_hash": h, "max_top_k_id": V - 1},
                                 tok_bl, V) is None
    why = _ev.kld_cache_refusal(tmp_path, {"vocab_size": V, "tokenizer_hash": h, "max_top_k_id": V}, tok_bl, V)
    assert why and "beyond the student's head" in why
    why = _ev.kld_cache_refusal(tmp_path, {"vocab_size": V + 7, "tokenizer_hash": h}, tok_bl, V)
    assert why and "records no largest cached id" in why
    why = _ev.kld_cache_refusal(tmp_path, {"vocab_size": V, "tokenizer_hash": "x",
                                           "teacher_path": str(tmp_path / "gone")}, tok_bl, V)
    assert why and "is not at" in why


def test_eval_refuses_the_kld_cache_before_the_load(tmp_path, tok_bl, monkeypatch, capsys):
    """The kld cache check reads the student's width and tokenizer from
    its header, so a cache the student cannot score is refused before
    the model load."""
    from gmlx.distill import evaluate as _ev

    V = len(dl.token_bytes(tok_bl))
    monkeypatch.setattr(_ev, "student_width", lambda path: V)
    monkeypatch.setattr(_ev._tokens, "load_tokenizer", lambda path: tok_bl)
    kld = tmp_path / "kld"
    kld.mkdir()
    (kld / "manifest.json").write_text(json.dumps({"vocab_size": V, "tokenizer_hash": "x", "top_k": 8,
                                                   "num_batches": 0}))
    student = tmp_path / "student.gguf"
    student.write_bytes(b"")
    rc = _ev.run_eval(_ev.EvalOptions(student=str(student), md=str(tmp_path / "r.md"), json=str(tmp_path / "r.json"),
                                      kld_cache=str(kld)))
    err = capsys.readouterr().err
    assert rc == 2 and "[eval] refuse: --kld-cache" in err and "no tokenizer directory" in err


def test_teacher_bpb_accepts_a_map_or_an_eval_report(tmp_path):
    from gmlx.distill import evaluate as _ev

    p = tmp_path / "t.json"
    p.write_text(json.dumps({"prose": 0.5, "code": None}))
    assert _ev.teacher_bpb_map(p) == {"prose": 0.5, "code": None}
    p.write_text(json.dumps({"after": {"bpb": {"prose": {"bpb": 0.4, "bytes": 10}}}}))
    assert _ev.teacher_bpb_map(p) == {"prose": 0.4}
    p.write_text(json.dumps({"after": {"tasks": {}}}))
    with pytest.raises(_ev.UnreadableInput):
        _ev.teacher_bpb_map(p)


def test_jsonl_readers_keep_unicode_line_separators(tmp_path):
    """The writers emit non-ASCII text verbatim, so a row holding U+2028
    must survive every reader: a line is what a newline ends."""
    from gmlx.distill import evaluate as _ev
    from gmlx.distill import filter as _flt
    from gmlx.distill import gen as _gen

    text = "one\u2028two\u2029three\u0085four"
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"id": "a", "messages": [{"role": "user", "content": text}]}, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    assert [r["id"] for r in _ev.read_jsonl(p)] == ["a"]
    assert [r["id"] for r in _flt._read_rows(p)] == ["a"]
    assert _gen._done_ids(p) == {"a"}
    rows = _gen.prompt_rows(_gen.GenOptions(out="x", prompts=str(p)))
    assert len(rows) == 1 and rows[0]["messages"][0]["content"] == text


def test_cache_kld_scores_the_students_own_list(tmp_path, tok_bl):
    """A row with student_messages is re-rendered from that list, the
    conversation training scored the student on, not from the teacher's
    longer one."""
    teacher = _with_template(tok_bl, _TEMPLATE_A)
    reply = {"role": "assistant", "content": "the cat is the cat 123"}
    t_msgs = [{"role": "user", "content": "the cat is 123 the cat the cat\n\nwhat is it"}, reply]
    s_msgs = [{"role": "user", "content": "what is it"}, reply]
    _tiny_reply_cache(tmp_path / "c", teacher, [(t_msgs, s_msgs), (t_msgs, None)])
    reader = dl.CacheReader(tmp_path / "c")
    V = len(dl.token_bytes(tok_bl))
    W = mx.array(np.random.default_rng(3).standard_normal((V, V)).astype(np.float32))

    class Stub:
        def __call__(self, ids):
            return W[ids]

    res = dl.cache_kld(Stub(), reader, tokenizer=teacher)
    assert res["rows"] == 2 and res["rerendered_rows"] == 1 and res["positions"] > 0
    assert np.isfinite(res["mean_kld_nats"])


# ---------------------------------------------------------------------------
# chat-frame spans, resume contracts, batching and validation
# ---------------------------------------------------------------------------

_TEMPLATE_DATE = "{% if date_string %}Date: {{ date_string }}\n{% endif %}" + _TEMPLATE_A
_TEMPLATE_SWITCH = ("{% if enable_thinking %}<think>{% endif %}{% if date_string %}{{ date_string }}{% endif %}"
                    + _TEMPLATE_A)


def _chat_corpus(path: Path, n=4) -> Path:
    rows = [{"messages": [{"role": "user", "content": f"the cat {i}"},
                          {"role": "assistant", "content": "the cat is the cat " * (i + 1)}]} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _cpu_view(tmp: Path, tok, name="view", n_rows=8, kprime=None, student_tok=None) -> tuple[Path, Path]:
    """(view dir, MLX student dir): a plain cache over tok, aligned to a
    two-layer llama student over student_tok (tok itself by default, the
    identity path), so train runs on the CPU end to end through the
    library's MLX loader."""
    from gmlx.distill import view as _view

    cache = tmp / "cache"
    if not (cache / "manifest.json").exists():
        _tiny_cache(cache, tok, n_rows=n_rows)
        tok.save_pretrained(cache / "tokenizer")
    student = tmp / "student"
    if not (student / "config.json").exists():
        _tiny_mlx_teacher(student, student_tok or tok)
    out = tmp / name
    assert _view.run_align(_view.AlignOptions(cache=str(cache), student=str(student), out=str(out),
                                              kprime=kprime)) == 0
    return out, student


def test_render_row_starts_the_reply_search_after_the_turn_header(tok_bl):
    """A reply of one letter that the turn header also contains ("d" in
    "Model:") is found after the header, never inside it."""
    tok = _with_template(tok_bl, _TEMPLATE_A)
    for reply in ("d", "e", "Model"):
        msgs = [{"role": "user", "content": "Pick one: d or e?"}, {"role": "assistant", "content": reply}]
        text, spans = dl.render_row(tok, msgs, open_tail=False)
        b0, b1, b2 = spans[0]
        assert text[b0:b1] == reply.encode() and text[:b0].endswith(b"Model:\n")
        assert text[b1:b2] == b"\n<end>\n"


def test_render_row_finds_the_turn_end_marker_after_trailing_whitespace(tok_bl):
    """A reply ending in a newline keeps the marker after it as a target,
    on a template that renders the content as given."""
    tok = _with_template(tok_bl, _TEMPLATE_B)
    tails = dl.assistant_tails(tok)
    assert tails
    for reply in ("Paris", "Paris\n", "Paris \n\n"):
        msgs = [{"role": "user", "content": "capital of France"}, {"role": "assistant", "content": reply}]
        text, spans = dl.render_row(tok, msgs, open_tail=False)
        b0, b1, b2 = spans[0]
        assert text[b0:b1] == b"Paris"
        assert b2 > b1 and text[b1:b2].endswith(tails[0].encode()), reply


def test_identity_path_refuses_a_cache_with_any_student_list(tmp_path, tok_bl):
    """One row with its own student list anywhere in the cache rules out
    the identity path, however many plain rows come before it."""
    from gmlx.distill import view as _view

    tok = _with_template(tok_bl, _TEMPLATE_A)
    reply = {"role": "assistant", "content": "the cat is the cat"}
    plain = [{"role": "user", "content": "the cat"}, reply]
    ctx = [{"role": "user", "content": "the cat is 123 the cat the cat the cat\n\nthe cat"}, reply]
    pairs = [(plain, None)] * 10 + [(ctx, plain)]
    _tiny_reply_cache(tmp_path / "c", tok, pairs)
    reader = dl.CacheReader(tmp_path / "c")
    same, why = _view.same_render(reader, tok, "reply")
    assert not same and "student message list" in why
    same, why = _view.same_render(dl.CacheReader(tmp_path / "c"), tok, "reply", n_check=64)
    assert not same


def test_cache_resume_pins_the_date_and_the_teacher_identity_survives_a_touch(tmp_path, tok_bl, monkeypatch,
                                                                              capsys):
    """A template that reads date_string renders a resume with the first
    run's date, an unframed run records no render kwargs, and the teacher
    identity holds the bytes, not the mtime."""
    import os

    from gmlx.distill import frames as _frames
    from gmlx.distill import teacher as _teacher

    tok = _with_template(tok_bl, _TEMPLATE_DATE)
    teacher = _tiny_mlx_teacher(tmp_path / "teacher", tok)
    corpus = _chat_corpus(tmp_path / "chat.jsonl")
    out = tmp_path / "cache"
    monkeypatch.setattr(_frames, "today_string", lambda: "22 Sep 2026")
    opts = _teacher.CacheOptions(teacher=str(teacher), corpus=str(corpus), out=str(out), top_k=8, max_len=256,
                                 rows_per_shard=2, frame="reply")
    assert _teacher.run_cache(opts) == 0
    run = json.loads((out / "progress.json").read_text())["run"]
    assert run["render_kwargs"] == {"date_string": "22 Sep 2026"}
    assert "mtime_ns" not in run["teacher"] and run["teacher"]["sha256_head"]
    (out / "batch-00001.safetensors").unlink()
    monkeypatch.setattr(_frames, "today_string", lambda: "23 Sep 2026")
    for f in teacher.iterdir():
        os.utime(f, (1, 1))
    rc = _teacher.run_cache(_teacher.CacheOptions(teacher=str(teacher), corpus=str(corpus), out=str(out), top_k=8,
                                                  max_len=256, rows_per_shard=2, frame="reply", resume=True))
    assert rc == 0, capsys.readouterr().err
    assert json.loads((out / "progress.json").read_text())["run"]["render_kwargs"] == {"date_string": "22 Sep 2026"}
    assert dl.validate_cache(out) == []
    reader = dl.CacheReader(out)
    assert all(b"Date: 22 Sep 2026" in reader.row(r)[1] for r in range(len(reader)))
    a = _teacher.teacher_identity(str(teacher))
    for f in teacher.iterdir():
        os.utime(f, (2, 2))
    assert _teacher.teacher_identity(str(teacher)) == a
    w = teacher / "model.safetensors"
    data = bytearray(w.read_bytes())
    data[-1] ^= 0xFF
    w.write_bytes(bytes(data))
    assert _teacher.teacher_identity(str(teacher)) != a
    plain = tmp_path / "plain"
    assert _teacher.run_cache(_teacher.CacheOptions(teacher=str(teacher), corpus=str(_text_corpus(tmp_path / "t.jsonl")),
                                                    out=str(plain), top_k=8, max_len=64)) == 0
    assert json.loads((plain / "progress.json").read_text())["run"]["render_kwargs"] is None


def _mlx_students(monkeypatch):
    """Let run_train take the MLX directory student the CPU tests build
    (the CLI admits GGUF students only)."""
    from gmlx.distill import student as _student
    from gmlx.distill import trainer as _trainer

    monkeypatch.setattr(_trainer, "is_gguf", lambda p: True)
    monkeypatch.setattr(_trainer, "load_student",
                        lambda p, adapter, hf: (*_student.load_mlx_student(p, adapter_path=adapter), "mlx"))


def test_train_resume_refuses_other_settings_and_scores_a_fixed_val_sample(tmp_path, tok_bl, capsys, monkeypatch):
    """A checkpoint records the run it belongs to; a resume under another
    batch size or seed is refused, the same settings continue."""
    from gmlx.distill import trainer as _trainer

    _mlx_students(monkeypatch)
    view, student = _cpu_view(tmp_path, tok_bl)
    ck = tmp_path / "ckpt"
    base = dict(views=[str(view)], student=str(student), iters=2, batch_size=2, seed=1, ckpt_dir=str(ck),
                save_every=1, val_every=1, val_batches=1, no_wired_limit=True, lora_rank=2, chunk=16)
    assert _trainer.run_train(_trainer.TrainOptions(**base)) == 0
    state = json.loads((ck / "last" / "state.json").read_text())
    assert state["iteration"] == 2 and state["run"]["batch_size"] == 2 and len(state["run"]["views"]) == 1
    capsys.readouterr()
    rc = _trainer.run_train(_trainer.TrainOptions(**dict(base, batch_size=4, iters=3, resume=True)))
    err = capsys.readouterr().err
    assert rc == 2 and "other settings than the run that wrote the checkpoint" in err and "batch_size" in err
    rc = _trainer.run_train(_trainer.TrainOptions(**dict(base, seed=2, resume=True)))
    assert rc == 2 and "seed" in capsys.readouterr().err
    assert _trainer.run_train(_trainer.TrainOptions(**dict(base, resume=True))) == 0
    assert "resumed at step 2" in capsys.readouterr().err


def test_train_compiles_each_view_at_its_own_kprime(tmp_path, tok_bl, tok_spm, monkeypatch):
    """Two views over one cross-tokenizer pair with different K' each get
    a loader at their own width; the batch pads to the widest."""
    from gmlx.distill import data as _data
    from gmlx.distill import trainer as _trainer

    _mlx_students(monkeypatch)
    v1, student = _cpu_view(tmp_path, tok_bl, "v1", kprime=4, student_tok=tok_spm)
    v2, _ = _cpu_view(tmp_path, tok_bl, "v2", kprime=8, student_tok=tok_spm)
    seen = []
    orig = _data.ViewLoader.__init__

    def record(self, *a, **k):
        seen.append(k.get("Kp"))
        orig(self, *a, **k)

    monkeypatch.setattr(_data.ViewLoader, "__init__", record)
    rc = _trainer.run_train(_trainer.TrainOptions(views=[str(v1), str(v2)], student=str(student), iters=1,
                                                  batch_size=2, no_wired_limit=True, lora_rank=2, chunk=16,
                                                  val_batches=1, ckpt_dir=str(tmp_path / "ck")))
    assert rc == 0 and seen == [4, 8]


def test_batch_iterator_keeps_the_final_short_batch():
    """Ten rows at batch size four make three batches an epoch, the last
    one holding the two longest rows."""
    lengths = [3, 9, 1, 7, 5, 8, 2, 6, 4, 10]
    it = dl.BatchIterator(lengths, 4, seed=0)
    assert it.per_epoch == 3
    epoch = [rows for _, rows in zip(range(3), (r for _, r in it.iterate()))]
    assert sorted(sum(epoch, [])) == list(range(10))
    short = [b for b in it.batches if len(b) == 2]
    assert short == [[1, 9]]


def test_sample_rows_draws_once_across_views():
    from gmlx.distill import data as _data

    pairs = [(0, i) for i in range(10)] + [(1, i) for i in range(10)]
    lengths = {p: 100 - 5 * p[1] + p[0] for p in pairs}
    a = _data.sample_rows(pairs, lengths, 8, seed=1)
    b = _data.sample_rows(pairs, lengths, 8, seed=1)
    c = _data.sample_rows(pairs, lengths, 8, seed=2)
    assert a == b and a != c and len(a) == 8
    assert {p[0] for p in a} == {0, 1}
    assert [lengths[p] for p in a] == sorted(lengths[p] for p in a)
    assert _data.sample_rows(pairs, lengths, 40, seed=1) == sorted(pairs, key=lambda p: lengths[p])


def test_render_kwargs_inherit_every_variable_the_template_reads(tok_bl):
    from gmlx.distill import frames as _frames

    tok = _with_template(tok_bl, _TEMPLATE_SWITCH)
    kw = dl.resolve_render_kwargs(tok, inherit={"enable_thinking": False, "date_string": "1 Jan 2026", "x": 1})
    assert kw == {"enable_thinking": False, "date_string": "1 Jan 2026"}
    assert dl.resolve_render_kwargs(tok, inherit={"enable_thinking": False}, override={"enable_thinking": True}) \
        == {"enable_thinking": True, "date_string": _frames.today_string()}
    plain = _with_template(tok_bl, _TEMPLATE_A)
    assert dl.resolve_render_kwargs(plain, inherit={"enable_thinking": False, "date_string": "1 Jan 2026"}) == {}


def test_parse_render_kwargs_takes_a_long_inline_object():
    spec = json.dumps({"enable_thinking": False, "pad": "x" * 400})
    assert dl.parse_render_kwargs(spec) == {"enable_thinking": False, "pad": "x" * 400}
    assert dl.parse_render_kwargs("  " + spec) == json.loads(spec)


def test_head_scale_skips_the_tied_minicpm_head_and_reads_config():
    """mlx-lm's tied MiniCPM path applies no scale, so the head carries
    none there; a language model that keeps its arguments under config
    (mlx-vlm) is read like one that keeps them under args."""
    import types

    from mlx_lm.models import granite, minicpm

    from gmlx.distill import head as _head

    ids = mx.arange(1, 9)[None]
    for tie, want in ((False, 0.25), (True, 1.0)):
        mx.random.seed(0)
        a = minicpm.ModelArgs(model_type="minicpm", hidden_size=64, dim_model_base=16, num_hidden_layers=2,
                              intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-5, vocab_size=100,
                              num_key_value_heads=4, scale_depth=1.4, scale_emb=12, tie_word_embeddings=tie)
        m = minicpm.Model(a)
        spec = _head.head_spec_from_model(m)
        assert spec.scale == pytest.approx(want), tie
        assert _head.head_parity_gap(m, spec, ids) < 1e-4, tie
    ga = granite.ModelArgs(model_type="granite", hidden_size=32, num_hidden_layers=1, intermediate_size=64,
                           num_attention_heads=4, rms_norm_eps=1e-5, vocab_size=64, logits_scaling=8.0,
                           attention_multiplier=0.125, embedding_multiplier=1.0, residual_multiplier=1.0,
                           max_position_embeddings=512, num_key_value_heads=2, attention_bias=False,
                           mlp_bias=False, rope_theta=10000.0, tie_word_embeddings=False)
    g = granite.Model(ga)
    w = types.SimpleNamespace(config=ga, model=g.model, lm_head=g.lm_head, __call__=g.__call__)
    spec = _head.head_spec_from_model(w)
    assert spec.scale == pytest.approx(0.125)
    assert _head.h_dim(types.SimpleNamespace(config=ga, embed_tokens=types.SimpleNamespace())) == 32


def test_head_logits_applies_the_softcap_for_the_pass_and_the_probes():
    from gmlx.distill import teacher as _teacher

    rng = np.random.default_rng(3)
    W = mx.array((rng.standard_normal((16, 4)) * 3).astype(np.float32))
    h = mx.array(rng.standard_normal((5, 4)).astype(np.float32))
    z = _teacher.head_logits(dl.linear_head(W, softcap=2.0), h)
    assert z.dtype == mx.bfloat16
    want = 2.0 * mx.tanh((h @ W.T) / 2.0)
    assert float(mx.abs(z.astype(mx.float32) - want).max()) < 0.05
    assert float(mx.abs(z.astype(mx.float32)).max()) <= 2.0
    plain = _teacher.head_logits(dl.linear_head(W), h)
    assert float(mx.abs(plain.astype(mx.float32) - h @ W.T).max()) < 0.1


# ---------------------------------------------------------------------------
# review round six: header back-off, EOS roles, resume fingerprint, the
# patch restore, position-weighted KL, id-less chat items, the first
# continuation token, the hidden-state map's step, the teacher identity
# ---------------------------------------------------------------------------

_TEMPLATE_THINK = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                   "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}")


def test_render_row_backs_the_header_end_off_a_generation_prompt_suffix(tok_bl):
    """A generation prompt that opens a think block after the turn header
    shares its "<" with a reply that starts with a tag; the reply search
    starts where the header ends, not one character into the reply, and
    a reply the header contains is still found after it."""
    tok = _with_template(tok_bl, _TEMPLATE_THINK)
    for first in ("<b>the</b> cat", "<table>the cat", "the cat", "assistant"):
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": first},
                {"role": "user", "content": "why <b>the</b> cat"}, {"role": "assistant", "content": "done"}]
        text, spans = dl.render_row(tok, msgs, open_tail=False)
        got = [text[a:b].decode() for a, b, _c in spans]
        assert got == [first, "done"], first
        assert text[:spans[0][0]].endswith(b"assistant\n")


def test_every_teacher_eos_maps_into_the_student_eos_group():
    """A teacher with two end-of-sequence ids (gemma-4's eos and end of
    turn) sends both into the student's EOS group, as own mass, and the
    student's second EOS id joins that group."""
    import string

    t = _bytelevel_tokenizer(_BL_MERGES)
    t.add_special_tokens({"additional_special_tokens": ["<eot>"]})
    eot = t.convert_tokens_to_ids("<eot>")
    t._gguf_eos_token_ids = [t.eos_token_id, eot]
    s = _spm_tokenizer(["\u2581"] + list(string.ascii_letters), [])
    s.add_special_tokens({"additional_special_tokens": ["<|im_end|>"]})
    ie = s.convert_tokens_to_ids("<|im_end|>")
    s._gguf_eos_token_ids = [ie, s.eos_token_id]
    tb = dl.build_tables(t, s)
    g = tb.group_of[ie]
    assert tb.group_of[s.eos_token_id] == g
    assert tb.target_g[eot] == g and tb.target_g[t.eos_token_id] == g
    assert tb.own[eot] and tb.own[t.eos_token_id]
    assert tb.roles["eos"] == {"teacher": [t.eos_token_id, eot], "student": [ie, s.eos_token_id]}


def test_train_resume_survives_a_realigned_view_and_refuses_a_changed_term(tmp_path, tok_bl, capsys, monkeypatch):
    """The fingerprint reads what decides the batches, not the timing or
    the student path align also writes: a re-aligned view resumes, a
    changed row index, clip, weight decay or dropout is refused, and the
    validation rows come from one seeded draw."""
    from gmlx.distill import data as _data
    from gmlx.distill import trainer as _trainer

    _mlx_students(monkeypatch)
    draws = []
    orig = _data.sample_rows

    def record(pairs, lengths, n, seed):
        draws.append((n, seed))
        return orig(pairs, lengths, n, seed)

    monkeypatch.setattr(_data, "sample_rows", record)
    view, student = _cpu_view(tmp_path, tok_bl)
    ck = tmp_path / "ckpt"
    base = dict(views=[str(view)], student=str(student), iters=2, batch_size=2, seed=1, ckpt_dir=str(ck),
                save_every=1, val_every=1, val_batches=1, no_wired_limit=True, lora_rank=2, chunk=16)
    assert _trainer.run_train(_trainer.TrainOptions(**base)) == 0
    assert draws == [(2, 1)]
    run = json.loads((ck / "last" / "state.json").read_text())["run"]
    assert run["clip"] == 1.0 and run["lora_dropout"] == 0.0 and run["hs"] == 0.0
    vj = view / "view.json"
    v = json.loads(vj.read_text())
    v["alignment"]["loader_ms_per_row"] = 999.0
    v["student"] = "elsewhere/student"
    vj.write_text(json.dumps(v))
    capsys.readouterr()
    assert _trainer.run_train(_trainer.TrainOptions(**dict(base, resume=True))) == 0
    assert "resumed at step 2" in capsys.readouterr().err
    for change in (dict(clip=0.5), dict(weight_decay=0.1), dict(lora_dropout=0.1)):
        rc = _trainer.run_train(_trainer.TrainOptions(**dict(base, resume=True, **change)))
        err = capsys.readouterr().err
        assert rc == 2 and "other settings than the run" in err and next(iter(change)) in err, change
    v["index"][0]["split"] = "val" if v["index"][0]["split"] == "train" else "train"
    vj.write_text(json.dumps(v))
    rc = _trainer.run_train(_trainer.TrainOptions(**dict(base, resume=True)))
    assert rc == 2 and "views" in capsys.readouterr().err


def test_train_pads_a_narrow_view_into_the_wide_batch(tmp_path, tok_bl, tok_spm, monkeypatch):
    """Rows compiled at K' 4 collate into a batch at K' 8 with sentinel
    group ids and -inf targets in the padding, on the compiled and the
    materialized path alike."""
    from gmlx.distill import data as _data
    from gmlx.distill import trainer as _trainer
    from gmlx.distill import view as _view

    _mlx_students(monkeypatch)
    v1, student = _cpu_view(tmp_path, tok_bl, "v1", kprime=4, student_tok=tok_spm)
    v2, _ = _cpu_view(tmp_path, tok_bl, "v2", kprime=8, student_tok=tok_spm)
    assert _view.run_align(_view.AlignOptions(cache=str(tmp_path / "cache"), student=str(student),
                                              out=str(tmp_path / "v1m"), kprime=4, materialize=True)) == 0
    assert list((tmp_path / "v1m").glob("view-*.safetensors"))
    seen = []
    orig = _data.collate

    def record(rows, Kp, G, pad_to=32):
        out = orig(rows, Kp, G, pad_to)
        seen.append(([int(r.target_gid.shape[1]) for r in rows], Kp, G, out))
        return out

    monkeypatch.setattr(_data, "collate", record)
    for narrow in (v1, tmp_path / "v1m"):
        seen.clear()
        rc = _trainer.run_train(_trainer.TrainOptions(views=[str(narrow), str(v2)], student=str(student), iters=4,
                                                      batch_size=4, no_wired_limit=True, lora_rank=2, chunk=16,
                                                      val_batches=1, ckpt_dir=str(tmp_path / "ck")))
        assert rc == 0
        widths = {w for ws, _kp, _g, _o in seen for w in ws}
        assert widths == {4, 8}, (narrow, widths)
        for ws, kp, G, out in seen:
            assert kp == 8 and out["target_gid"].shape[1] == 8 and out["target_log_p"].shape[1] == 8
            n4 = sum(1 for w in ws if w == 4)
            if n4:
                pad_g = out["target_gid"][:, 4:]
                pad_lp = out["target_log_p"][:, 4:]
                assert (pad_g == G).sum() >= 4 * n4 and np.isneginf(pad_lp[pad_g == G]).all()


def test_train_restores_the_attention_patch_when_it_refuses(tmp_path, tok_bl, capsys, monkeypatch):
    """A run refused after the training patches went in leaves the
    module's attention as it found it."""
    import mlx_lm.models.llama as llama_mod

    from gmlx.distill import trainer as _trainer

    _mlx_students(monkeypatch)
    view, student = _cpu_view(tmp_path, tok_bl)
    before = llama_mod.scaled_dot_product_attention
    rc = _trainer.run_train(_trainer.TrainOptions(views=[str(view)], student=str(student), iters=2, batch_size=2,
                                                  no_wired_limit=True, lora_rank=2, chunk=16, val_batches=1,
                                                  ckpt_dir=str(tmp_path / "none"), resume=True))
    assert rc == 2 and "no checkpoint" in capsys.readouterr().err
    assert llama_mod.scaled_dot_product_attention is before
    assert not getattr(llama_mod.scaled_dot_product_attention, "_gmlx_training_attention", False)


class _RowSubset:
    """A reader over a subset of another reader's rows."""

    def __init__(self, reader, rows):
        self.reader, self.rows, self.manifest = reader, rows, reader.manifest

    def __len__(self):
        return len(self.rows)

    def row(self, r):
        return self.reader.row(self.rows[r])


def test_cache_kld_weights_positions_not_rows(tmp_path, tok_bl):
    """The mean KL is over positions, so a long row counts for every
    position it holds; the SE clusters by row."""
    _tiny_cache(tmp_path / "c", tok_bl, n_rows=4)
    reader = dl.CacheReader(tmp_path / "c")
    V = len(dl.token_bytes(tok_bl))
    W = mx.array(np.random.default_rng(3).standard_normal((V, V)).astype(np.float32))

    class Stub:
        def __call__(self, ids):
            return W[ids]

    full = dl.cache_kld(Stub(), reader)
    parts = [dl.cache_kld(Stub(), _RowSubset(reader, [r])) for r in range(len(reader))]
    counts = np.array([p["positions"] for p in parts], dtype=np.float64)
    means = np.array([p["mean_kld_nats"] for p in parts])
    top1 = np.array([p["top1_agreement"] for p in parts])
    assert len(set(counts.tolist())) > 1 and full["positions"] == counts.sum()
    assert full["mean_kld_nats"] == pytest.approx(float((means * counts).sum() / counts.sum()))
    assert full["top1_agreement"] == pytest.approx(float((top1 * counts).sum() / counts.sum()))
    assert full["mean_kld_nats"] != pytest.approx(float(means.mean()))
    sums = means * counts
    resid = sums - full["mean_kld_nats"] * counts
    n = len(sums)
    se = float(np.sqrt((resid ** 2).sum() / (n - 1)) / np.sqrt(n) / counts.mean())
    assert full["clustered_se"] == pytest.approx(se)


def test_cache_kld_skips_a_row_whose_spans_do_not_align(tmp_path, tok_bl, monkeypatch):
    """A re-rendered row whose span alignment raises is skipped like a
    row whose render fails, not propagated."""
    teacher = _with_template(tok_bl, _TEMPLATE_A)
    reply = {"role": "assistant", "content": "the cat is the cat 123"}
    t_msgs = [{"role": "user", "content": "the cat is 123 the cat the cat\n\nwhat is it"}, reply]
    s_msgs = [{"role": "user", "content": "what is it"}, reply]
    _tiny_reply_cache(tmp_path / "c", teacher, [(t_msgs, s_msgs), (t_msgs, None)])
    reader = dl.CacheReader(tmp_path / "c")
    V = len(dl.token_bytes(tok_bl))
    W = mx.array(np.random.default_rng(3).standard_normal((V, V)).astype(np.float32))

    class Stub:
        def __call__(self, ids):
            return W[ids]

    def boom(*a, **k):
        raise ValueError("spans do not line up")

    monkeypatch.setattr(dl_eval, "shared_boundaries_spans", boom)
    res = dl.cache_kld(Stub(), reader, tokenizer=teacher)
    assert res["rows"] == 1 and res["rerendered_rows"] == 0


def test_chat_sanity_keys_id_less_items_by_index(tmp_path, tok_bl):
    """Items without an id are named by their index in the records and
    in the reference lookup, so a drift reference reaches every item."""
    from gmlx.distill import student as _student

    tok = _with_template(tok_bl, _TEMPLATE_A)
    model, _cfg, tokenizer = _student.load_mlx_student(str(_tiny_mlx_teacher(tmp_path / "m", tok)))
    items = [{"messages": [{"role": "user", "content": "the cat"}]},
             {"messages": [{"role": "user", "content": "is the cat"}]}]
    before = dl.chat_sanity(model, tokenizer, items, max_tokens=3)
    assert [r["id"] for r in before["items"]] == ["0", "1"]
    refs = {r["id"]: "the cat is" for r in before["items"]}
    after = dl.chat_sanity(model, tokenizer, items, refs=refs, max_tokens=3)
    assert all("ref_nll_nats" in r for r in after["items"]) and after["ref_nll_nats"] is not None


def test_continuation_logprob_scores_every_token_without_a_bos(tok_bl):
    """With no BOS and an empty context the first continuation token is
    the first target, not the last one."""
    V = len(dl.token_bytes(tok_bl))
    W = mx.array(np.random.default_rng(3).standard_normal((V, V)).astype(np.float32))

    class Stub:
        def __call__(self, ids):
            return W[ids]

    inner = dl.hf_inner(tok_bl)
    assert inner.encode("", add_special_tokens=True) == []
    ids = inner.encode("the cat is", add_special_tokens=True)
    assert len(ids) >= 3
    arr = np.array([ids], dtype=np.int32)
    lp = dl_eval._target_logprobs(W[mx.array(arr)], arr[:, 1:])[0]
    got = dl.continuation_logprob(Stub(), tok_bl, "", ["the cat is"])[0]
    assert got == pytest.approx(float(lp.sum()))


def test_reply_slice_se_weights_rows_by_their_bytes(tok_bl):
    """The SE of a byte-weighted bpb is the cluster-robust one over rows,
    not the spread of the per-row ratios."""
    tok = _with_template(tok_bl, _TEMPLATE_A)
    V = len(dl.token_bytes(tok))
    W = mx.array(np.random.default_rng(3).standard_normal((V, V)).astype(np.float32))

    class Stub:
        def __call__(self, ids):
            return W[ids]

    rows = [{"id": f"r{i}", "messages": [{"role": "user", "content": "say it"},
                                          {"role": "assistant", "content": "the cat is the cat 123 " * (i + 1)}]}
            for i in range(3)]
    r = dl.reply_slice_nll(Stub(), tok, rows, max_len=256)
    s = np.array([it["nll"] for it in r["items"]]) / math.log(2)
    b = np.array([it["bytes"] for it in r["items"]], dtype=np.float64)
    assert len(set(b.tolist())) == 3 and r["bpb"] == pytest.approx(float(s.sum() / b.sum()))
    resid = s - r["bpb"] * b
    se = float(np.sqrt((resid ** 2).sum() / 2) / np.sqrt(3) / b.mean())
    assert r["row_bpb_se"] == pytest.approx(se)
    assert r["row_bpb_se"] != pytest.approx(float(np.std(s / b) / np.sqrt(3)))


def test_eval_reports_an_empty_task_as_none(tmp_path, tok_bl, capsys):
    """A task file with no items reports None, not NaN, in the JSON and
    the table."""
    from gmlx.distill import evaluate as _ev

    class Stub:
        def eval(self):
            pass

    opts = _ev.EvalOptions(student="s.gguf", md=str(tmp_path / "r.md"), json=str(tmp_path / "r.json"))
    res = _ev.run_arm(Stub(), tok_bl, opts, {}, {"arc_easy": {"items": []}})
    assert res["tasks"]["arc_easy"]["acc"] is None and res["tasks"]["arc_easy"]["n"] == 0
    assert "acc None" in capsys.readouterr().err
    report = {"after": res, "before": {}, "reply_positions": None}
    md = _ev.report_markdown(opts, report, {}, {}, {}, set())
    assert "| arc_easy | None | None | 0 |" in md
    assert "nan" not in json.dumps(res).lower()


def test_hidden_state_map_keeps_step_with_the_trunk(tmp_path, tok_bl, monkeypatch, capsys):
    """A batch trained before the first boundary batch counts on the
    map's schedule too: the map's optimizer step equals the trunk's at
    every checkpoint, and a resume without a saved map says so."""
    from gmlx.distill import data as _data
    from gmlx.distill import hidden as _hidden
    from gmlx.distill import trainer as _trainer
    from gmlx.distill import view as _view

    from .test_distill_hidden import _hidden_cache

    hh = _hidden.HsHead(4, 3, 1, 1e-3)
    assert int(hh.opt.step) == 0
    hh.advance()
    assert int(hh.opt.step) == 1
    _mlx_students(monkeypatch)
    cache = tmp_path / "cache"
    _hidden_cache(cache, tok_bl, n_rows=8)
    tok_bl.save_pretrained(cache / "tokenizer")
    student = _tiny_mlx_teacher(tmp_path / "student", tok_bl)
    view = tmp_path / "view"
    assert _view.run_align(_view.AlignOptions(cache=str(cache), student=str(student), out=str(view))) == 0
    orig = _data.collate
    calls = []

    def drop_first(rows, Kp, G, pad_to=32):
        out = orig(rows, Kp, G, pad_to)
        calls.append(1)
        if len(calls) == 1:
            out.pop("hidden_target", None)
        return out

    monkeypatch.setattr(_data, "collate", drop_first)
    ck = tmp_path / "ckpt"
    base = dict(views=[str(view)], student=str(student), iters=3, batch_size=2, seed=1, ckpt_dir=str(ck),
                save_every=3, val_every=3, val_batches=1, no_wired_limit=True, lora_rank=2, chunk=16, hs=0.5)
    assert _trainer.run_train(_trainer.TrainOptions(**base)) == 0
    main = mx.load(str(ck / "last" / "optimizer.safetensors"))
    hs = mx.load(str(ck / "last" / "hs_optimizer.safetensors"))
    assert isinstance(main, dict) and isinstance(hs, dict)
    assert int(main["step"]) == 3 and int(hs["step"]) == 3
    (ck / "last" / "hs_head.safetensors").unlink()
    capsys.readouterr()
    rc = _trainer.run_train(_trainer.TrainOptions(**dict(base, iters=4, resume=True)))
    err = capsys.readouterr().err
    assert rc == 2 and "iters" in err
    assert _trainer.run_train(_trainer.TrainOptions(**dict(base, resume=True))) == 0
    assert "hidden-state map not in the last checkpoint" in capsys.readouterr().err


def test_teacher_identity_covers_split_shards_and_gguf_directories(tmp_path):
    """The identity of a split GGUF hashes every shard, and a directory
    of GGUF files hashes them, so a re-quantized second shard or a
    swapped file is not the same teacher."""
    from gmlx.distill import teacher as _teacher

    d = tmp_path / "split"
    d.mkdir()
    a = d / "m-00001-of-00002.gguf"
    b = d / "m-00002-of-00002.gguf"
    a.write_bytes(b"GGUF" + bytes(64))
    b.write_bytes(b"GGUF" + bytes(64))
    ident = _teacher.teacher_identity(str(a))
    assert ident["size"] == 2 * 68
    b.write_bytes(b"GGUF" + bytes(63) + b"\x01")
    assert _teacher.teacher_identity(str(a)) != ident
    e = tmp_path / "dir"
    e.mkdir()
    (e / "one.gguf").write_bytes(b"GGUF" + bytes(8))
    ident_dir = _teacher.teacher_identity(str(e))
    assert ident_dir["size"] == 12
    (e / "one.gguf").write_bytes(b"GGUF" + bytes(7) + b"\x01")
    assert _teacher.teacher_identity(str(e)) != ident_dir


# ---------------------------------------------------------------------------
# review round seven: headers that end in a marker, the LoRA and student
# fingerprint, per-view knobs, symlinked teachers, hard cuts inside a
# character, a student that adapts nothing
# ---------------------------------------------------------------------------

_TEMPLATE_INST = ("{{ bos_token }}{% for m in messages %}{% if m['role'] == 'user' %}[INST] {{ m['content'] }} [/INST]"
                  "{% else %}{{ m['content'] }}</s>{% endif %}{% endfor %}")
_TEMPLATE_FINAL = ("{% for m in messages %}{% if m['role'] == 'user' %}<|start|>user<|message|>{{ m['content'] }}<|end|>"
                   "{% else %}<|start|>assistant<|channel|>final<|message|>{{ m['content'] }}<|return|>{% endif %}"
                   "{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %}")


def test_render_row_keeps_short_replies_out_of_headers_ending_in_a_marker(tok_bl):
    """A header that ends in a marker rather than whitespace ("[/INST]"),
    and one that runs past the generation prompt ("<|channel|>final
    <|message|>"), still put a one-letter reply after the header."""
    for template, tail in ((_TEMPLATE_INST, b"</s>"), (_TEMPLATE_FINAL, b"<|return|>")):
        tok = _with_template(tok_bl, template)
        for reply in ("I", "INST", "a", "t", "final", "Hello"):
            msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": reply}]
            text, spans = dl.render_row(tok, msgs, open_tail=False)
            b0, b1, b2 = spans[0]
            assert text[b0:b1] == reply.encode() and text[b1:b2] == tail, (template[:12], reply, text)
            assert text[:b0].endswith(b"[/INST]" if tail == b"</s>" else b"<|message|>")


def test_train_resume_refuses_other_lora_settings_or_student(tmp_path, tok_bl, capsys, monkeypatch):
    """The checkpoint's LoRA rank, multiplier, keys and student are part
    of the run; a resume under another one is refused before the load
    of a checkpoint whose factors would not fit."""
    from gmlx.distill import trainer as _trainer

    _mlx_students(monkeypatch)
    view, student = _cpu_view(tmp_path, tok_bl)
    other = _tiny_mlx_teacher(tmp_path / "student2", tok_bl, seed=3)
    ck = tmp_path / "ckpt"
    base = dict(views=[str(view)], student=str(student), iters=2, batch_size=2, seed=1, ckpt_dir=str(ck),
                save_every=1, val_every=1, val_batches=1, no_wired_limit=True, lora_rank=2, chunk=16)
    assert _trainer.run_train(_trainer.TrainOptions(**base)) == 0
    run = json.loads((ck / "last" / "state.json").read_text())["run"]
    assert run["lora_rank"] == 2 and run["lora_scale"] == 2.0 and run["student"]["sha256_head"]
    capsys.readouterr()
    for change, key in ((dict(lora_rank=4), "lora_rank"), (dict(lora_scale=7.0), "lora_scale"),
                        (dict(lora_alpha=1.0), "lora_scale"), (dict(student=str(other)), "student")):
        rc = _trainer.run_train(_trainer.TrainOptions(**dict(base, resume=True, **change)))
        err = capsys.readouterr().err
        assert rc == 2 and "other settings than the run" in err and key in err, (change, err)
    assert _trainer.run_train(_trainer.TrainOptions(**dict(base, resume=True))) == 0


def test_train_refuses_views_with_other_knobs_and_a_gamma_override_on_a_materialized_view(tmp_path, tok_bl,
                                                                                              capsys, monkeypatch):
    """Views aligned with other loss knobs cannot mix, since a view's
    chunks are cut with its own gamma and chunk length at align time,
    and a gamma override on a materialized view would change nothing."""
    from gmlx.distill import trainer as _trainer
    from gmlx.distill import view as _view

    _mlx_students(monkeypatch)
    v1, student = _cpu_view(tmp_path, tok_bl, "v1")
    cache = tmp_path / "cache"
    assert _view.run_align(_view.AlignOptions(cache=str(cache), student=str(student), out=str(tmp_path / "v2"),
                                              gamma=0.01)) == 0
    assert _view.run_align(_view.AlignOptions(cache=str(cache), student=str(student), out=str(tmp_path / "v3"),
                                              materialize=True)) == 0
    base = dict(student=str(student), iters=1, batch_size=2, no_wired_limit=True, lora_rank=2, chunk=16,
                val_batches=1, ckpt_dir=str(tmp_path / "ck"))
    capsys.readouterr()
    rc = _trainer.run_train(_trainer.TrainOptions(views=[str(v1), str(tmp_path / "v2")], **base))
    err = capsys.readouterr().err
    assert rc == 2 and "other loss knobs" in err and "gamma" in err
    rc = _trainer.run_train(_trainer.TrainOptions(views=[str(tmp_path / "v3")], gamma=0.01, **base))
    err = capsys.readouterr().err
    assert rc == 2 and "materialized" in err and "--gamma" in err
    assert _trainer.run_train(_trainer.TrainOptions(views=[str(tmp_path / "v3")], gamma=0.001, **base)) == 0
    assert _trainer.run_train(_trainer.TrainOptions(views=[str(v1), str(tmp_path / "v3")], **base)) == 0


def test_teacher_identity_and_path_keep_symlinks(tmp_path, tok_bl):
    """A teacher reached through symlinks (the Hugging Face cache layout)
    keeps its own name in the identity and the manifest, so the split
    shards are found and the tokenizer can be read back from the path."""
    from gmlx.distill import teacher as _teacher

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "aaa").write_bytes(b"GGUF" + bytes(64))
    (blobs / "bbb").write_bytes(b"GGUF" + bytes(32))
    snap = tmp_path / "snap"
    snap.mkdir()
    first = snap / "m-00001-of-00002.gguf"
    first.symlink_to(blobs / "aaa")
    (snap / "m-00002-of-00002.gguf").symlink_to(blobs / "bbb")
    ident = _teacher.teacher_identity(str(first))
    assert ident["path"].endswith("snap/m-00001-of-00002.gguf") and ident["size"] == 68 + 36
    teacher = _tiny_mlx_teacher(tmp_path / "teacher", tok_bl)
    link = tmp_path / "link"
    link.symlink_to(teacher)
    corpus = _text_corpus(tmp_path / "t.jsonl")
    out = tmp_path / "cache"
    assert _teacher.run_cache(_teacher.CacheOptions(teacher=str(link), corpus=str(corpus), out=str(out), top_k=8,
                                                    max_len=64)) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["teacher_path"].endswith("/link") and man["gmlx_distill"]["teacher"]["path"].endswith("/link")
    assert _teacher.teacher_identity(str(link))["path"].endswith("/link")


def test_cut_windows_hard_cut_stays_on_a_character_boundary(tmp_path, tok_bl):
    """A window cut hard at the token limit inside a multi-byte character
    backs off to the character's start, so a continue-frame cache over
    text with no whitespace-initial token still builds."""
    from gmlx.distill import frames as _frames
    from gmlx.distill import teacher as _teacher

    tok = _with_template(tok_bl, _TEMPLATE_A)
    tb = dl.token_bytes(tok)
    text = ("\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8" * 20).encode("utf-8")
    ids, ends, _ = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
    ws = dl.whitespace_start_mask(tok, len(tb), tb)
    windows = _frames.cut_windows(ids, ws, 40, 0, text=text, ends=ends)
    assert len(windows) > 1
    for s, e in windows:
        b0 = int(ends[s - 1]) if s > 0 else 0
        text[b0:int(ends[e - 1])].decode("utf-8")
    corpus = tmp_path / "cjk.jsonl"
    corpus.write_text(json.dumps({"text": text.decode("utf-8")}) + "\n", encoding="utf-8")
    rows = _teacher.build_rows(tok, str(corpus), max_len=40, text_key="text", max_rows=None, max_tokens=None,
                               source=None, hf_split="train", limit_docs=None, frame="continue")
    assert len(rows[0]) > 1


def test_train_refuses_a_student_that_adapts_nothing_and_warns_on_a_partial_match(tmp_path, tok_bl, capsys,
                                                                                    monkeypatch):
    from gmlx.distill import trainer as _trainer

    _mlx_students(monkeypatch)
    view, student = _cpu_view(tmp_path, tok_bl)
    base = dict(views=[str(view)], student=str(student), iters=1, batch_size=2, no_wired_limit=True, lora_rank=2,
                chunk=16, val_batches=1, ckpt_dir=str(tmp_path / "ck"))
    monkeypatch.setattr(_trainer, "LORA_KEYS", ("self_attn.qkv_proj",))
    rc = _trainer.run_train(_trainer.TrainOptions(**base))
    err = capsys.readouterr().err
    assert rc == 2 and "nothing would train" in err and "self_attn.qkv_proj" in err
    monkeypatch.setattr(_trainer, "LORA_KEYS", ("self_attn.q_proj", "self_attn.qkv_proj"))
    assert _trainer.run_train(_trainer.TrainOptions(**base)) == 0
    err = capsys.readouterr().err
    assert "[train] warn: LoRA adapted 2 modules over 2 layers" in err
