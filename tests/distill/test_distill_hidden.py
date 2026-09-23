#!/usr/bin/env python3
"""The hidden-state option on minted caches: the sketch matrix, the
hidden field through the shard packer, the validator rules, the row
view's hidden targets on identity and projected views, the materialized
round trip, and the hidden-state pass's loss and gradients. CPU only."""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import gmlx.distill as dl
from gmlx.distill import hidden as hs
from gmlx.distill.format import HIDDEN_FIELD

from .test_distill_lib import KNOBS, _bytelevel_tokenizer, _rand_text, _spm_tokenizer

DIM = 16


@pytest.fixture(scope="module")
def tok_bl():
    return _bytelevel_tokenizer([("\u0120", "t"), ("h", "e"), ("\u0120t", "he"), ("1", "2"),
                                 ("12", "3"), ("\u0120", "a"), ("i", "s"), ("\u0120", "is"),
                                 ("c", "a"), ("ca", "t")])


@pytest.fixture(scope="module")
def tok_spm():
    import string
    pieces = ["\u2581"] + list(string.ascii_letters + string.digits + ".,!?\n") \
        + ["\u2581" + c for c in string.ascii_lowercase] + ["\u2581the", "\u2581cat", "\u2581is",
                                                              "th", "he", "at", "12", "123", "\u25811"]
    merges = [("\u2581", "t"), ("\u2581t", "h"), ("\u2581th", "e"), ("t", "h"), ("h", "e"), ("a", "t"),
              ("\u2581", "c"), ("\u2581c", "a"), ("\u2581ca", "t"), ("\u2581", "i"), ("\u2581i", "s"),
              ("1", "2"), ("12", "3"), ("\u2581", "1")]
    return _spm_tokenizer(pieces, merges)


def _hidden_cache(tmp: Path, tok, *, seed=11, n_rows=4, K=8, dim=DIM):
    """A synthetic teacher through reduce_logits with a hidden sketch of
    its own random hidden states, so the field equals h @ R exactly."""
    rng = np.random.default_rng(seed)
    tb = dl.token_bytes(tok)
    V = len(tb)
    log_bmask = dl.log_bmask_from(dl.whitespace_start_mask(tok, V, tb))
    d_model = 6
    W = mx.array((rng.standard_normal((V, d_model)) * 0.5).astype(np.float32))
    R = mx.array(hs.projection_matrix(d_model, dim, seed))
    writer = dl.ShardWriter(tmp, K, False)
    rows, metas, tbytes, sketches = [], [], [], []
    for r in range(n_rows):
        text = _rand_text(rng, int(rng.integers(4, 12))).encode("utf-8")
        ids, ends, _ = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
        n = len(ids)
        h = mx.array(rng.standard_normal((n, d_model)).astype(np.float32))
        nxt = np.concatenate([ids[1:], [-1]]).astype(np.int32)
        red = dl.reduce_logits((h @ W.T).astype(mx.bfloat16), mx.array(nxt), K=K, log_bmask=log_bmask,
                               onpath_valid=nxt >= 0)
        red["onpath_mask"] = nxt >= 0
        red["token_ids"] = ids
        red["token_end_byte"] = ends
        sk = hs.sketch(h[None], R)[0]
        red[HIDDEN_FIELD] = sk
        sketches.append(sk)
        rows.append(red)
        tbytes.append(text)
        metas.append(dl.RowMeta(row_id=r, doc_id=f"d{r}", window=0, n_tokens=n))
    writer.write(0, dl.pack_shard(rows, tbytes, K, False), metas, wall_s=0.01, step=64)
    dl.write_manifest(tmp, teacher_path="synthetic", dataset="synthetic", num_samples=n_rows,
                      max_seq_len=64, seed=seed, top_k=K, vocab_size=V, config_vocab_size=V,
                      tokenizer_hash=dl.vocab_map_hash(tok), batch_size=8,
                      gmlx_distill={"mlx_kld_compatible": True, "corpus_sha256": "x",
                                    "hidden": hs.hidden_block(d_model, dim, seed)})
    return sketches


def test_projection_and_loss_closed_forms():
    R = hs.projection_matrix(6, DIM, 3)
    assert R.shape == (6, DIM) and R.dtype == np.float32
    assert np.array_equal(R, hs.projection_matrix(6, DIM, 3))
    assert not np.array_equal(R, hs.projection_matrix(6, DIM, 4))
    # a Gaussian sketch keeps norms in expectation: the mean squared row
    # norm of a unit-row input is about 1
    x = np.eye(6, dtype=np.float32)
    assert 0.3 < float(np.mean(np.sum((x @ R) ** 2, axis=1))) < 3.0
    a = mx.array(np.random.default_rng(0).standard_normal((5, DIM)).astype(np.float32))
    assert float(hs.hs_loss(a, a, "cosine")) == pytest.approx(0.0, abs=1e-5)
    assert float(hs.hs_loss(a, a, "mse")) == pytest.approx(0.0, abs=1e-5)
    assert float(hs.hs_loss(a, -a, "cosine")) == pytest.approx(2.0, abs=1e-5)
    assert float(hs.hs_loss(3.0 * a, a, "cosine")) == pytest.approx(0.0, abs=1e-5)
    assert hs.hidden_block(6, DIM, 3)["layer"] == "final"


def test_hidden_field_packs_validates_and_reads(tmp_path, tok_bl):
    sketches = _hidden_cache(tmp_path / "c", tok_bl)
    assert dl.validate_cache(tmp_path / "c") == []
    reader = dl.CacheReader(tmp_path / "c")
    for r in range(len(reader)):
        arrs, _t, _m = reader.row(r)
        assert arrs[HIDDEN_FIELD].shape == (len(arrs["token_ids"]), DIM)
        assert np.array_equal(arrs[HIDDEN_FIELD], sketches[r])
    # the estimate counts the sketch
    assert dl.estimate_cache_bytes(10, 8, hidden_bytes=2 * DIM) == dl.estimate_cache_bytes(10, 8) + 10 * 2 * DIM
    # the validator ties the field to the manifest block, both ways
    from safetensors.numpy import load_file, save_file
    man = dl.read_json(tmp_path / "c" / "manifest.json")
    blk = man["gmlx_distill"].pop("hidden")
    dl.write_json_atomic(tmp_path / "c" / "manifest.json", man)
    probs = dl.validate_cache(tmp_path / "c", check_sha=False)
    assert any("hidden field present without a hidden block" in p for p in probs)
    man["gmlx_distill"]["hidden"] = dict(blk, dim=DIM + 1)
    dl.write_json_atomic(tmp_path / "c" / "manifest.json", man)
    probs = dl.validate_cache(tmp_path / "c", check_sha=False)
    assert any("hidden shape" in p for p in probs)
    man["gmlx_distill"]["hidden"] = blk
    dl.write_json_atomic(tmp_path / "c" / "manifest.json", man)
    p0 = tmp_path / "c" / "batch-00000.safetensors"
    sh = load_file(str(p0))
    sh[HIDDEN_FIELD][0, 0, 0] = np.float16(np.nan)
    save_file(sh, str(p0))
    probs = dl.validate_cache(tmp_path / "c", check_sha=False)
    assert any("hidden not finite" in p for p in probs)


def test_view_carries_hidden_targets(tmp_path, tok_bl, tok_spm):
    sketches = _hidden_cache(tmp_path / "c", tok_bl)
    reader = dl.CacheReader(tmp_path / "c")
    # identity: one target per on-path position, the sketch at that position
    tables = dl.build_tables(tok_bl, tok_bl)
    loader = dl.ViewLoader(reader, tok_bl, tables, knobs=KNOBS, Kp=8, identity=True)
    rv = loader.compile(0)
    assert rv is not None and rv.hidden_target is not None
    assert rv.hidden_target.shape == (len(rv.bnd_pos), DIM)
    assert np.array_equal(rv.hidden_target, sketches[0][rv.bnd_pos])
    batch = loader.batch(list(range(len(reader))))
    assert batch["hidden_target"].shape == (int(batch["n_bnd"]), DIM)
    assert batch["hidden_target"].dtype == np.float16
    # the batch order is row by row, boundary by boundary
    first = loader.compile(0)
    assert np.array_equal(batch["hidden_target"][:len(first.bnd_pos)], first.hidden_target)
    # cross-tokenizer: one target per shared boundary at its teacher position
    tables2 = dl.build_tables(tok_bl, tok_spm)
    loader2 = dl.ViewLoader(reader, tok_spm, tables2, knobs=KNOBS, Kp=8, identity=False)
    for r in range(len(reader)):
        rv2 = loader2.compile(r)
        if rv2 is None:
            continue
        assert rv2.hidden_target is not None and rv2.hidden_target.shape == (len(rv2.bnd_pos), DIM)
        rows_in = {tuple(x) for x in sketches[r].tolist()}
        assert all(tuple(x) in rows_in for x in rv2.hidden_target.tolist())
    # materialized views keep the field and read it back
    n = loader2.materialize(tmp_path / "v")
    assert n == 1
    loader3 = dl.ViewLoader(reader, tok_spm, tables2, knobs=KNOBS, Kp=8, identity=False, view_dir=tmp_path / "v")
    for r in range(len(reader)):
        a, b = loader2.compile(r), loader3.compile(r)
        if a is None:
            assert b is None
            continue
        assert np.array_equal(a.hidden_target, b.hidden_target)
    # a cache without the field yields no target and no batch key
    _hidden_cache(tmp_path / "p", tok_bl)
    man = dl.read_json(tmp_path / "p" / "manifest.json")
    man["gmlx_distill"]["hidden"] = None
    dl.write_json_atomic(tmp_path / "p" / "manifest.json", man)
    plain = dl.CacheReader(tmp_path / "p")
    arrs, _t, _m = plain.row(0)
    arrs.pop(HIDDEN_FIELD)
    s_ids, s_ends, _ = loader.student_tokens(arrs, _t, _m)
    rv0 = dl.compile_row(arrs, _t, s_ids, s_ends, tables, Kp=8, knobs=KNOBS, teacher_special=set(),
                         student_special=set(), identity=True)
    assert rv0.hidden_target is None
    assert "hidden_target" not in dl.collate([rv0], 8, tables.G)


def test_hs_pass_gradients_and_a_step_reduces_the_loss():
    rng = np.random.default_rng(5)
    d_s, n = 12, 20
    hg = mx.array(rng.standard_normal((n, d_s)).astype(np.float32))
    target = mx.array(rng.standard_normal((n, DIM)).astype(np.float16))
    head = hs.HsHead(d_s, DIM, seed=1, lr=0.05)
    loss0, dh, dW = hs.hs_pass(hg, target, head, "cosine")
    assert dh.shape == (n, d_s) and dh.dtype == mx.float32
    assert float(loss0) > 0.5
    # the gradient with respect to the hidden states matches a float64
    # finite difference of the same loss written in numpy (the GPU's f32
    # GEMM is not exact enough for a difference of two losses near 1)
    W = np.asarray(head.module.weight).astype(np.float64)
    h64, t64 = np.asarray(hg).astype(np.float64), np.asarray(target).astype(np.float64)

    def ref(h):
        pn = h @ W.T
        pn = pn / np.maximum(np.linalg.norm(pn, axis=-1, keepdims=True), 1e-6)
        tn = t64 / np.maximum(np.linalg.norm(t64, axis=-1, keepdims=True), 1e-6)
        return float(np.mean(1.0 - np.sum(pn * tn, axis=-1)))

    assert ref(h64) == pytest.approx(float(loss0), rel=1e-3)
    v = rng.standard_normal((n, d_s))
    eps = 1e-5
    fd = (ref(h64 + eps * v) - ref(h64 - eps * v)) / (2 * eps)
    assert float(np.sum(np.asarray(dh).astype(np.float64) * v)) == pytest.approx(fd, rel=1e-2, abs=1e-4)
    # optimizer steps on the map alone bring the loss down
    for _ in range(30):
        loss, _dh, dW = hs.hs_pass(hg, target, head, "cosine")
        head.update(dW)
    assert float(loss) < float(loss0) * 0.8
    # save and load round trip
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        head.save(Path(d))
        other = hs.HsHead(d_s, DIM, seed=9, lr=0.05)
        assert other.load(Path(d))
        assert np.array_equal(np.asarray(other.module.weight), np.asarray(head.module.weight))
        assert not hs.HsHead(d_s, DIM, seed=9, lr=0.05).load(Path(d) / "nope")


def test_hs_head_init_is_keyed_and_leaves_the_stream_alone():
    """The map's initial weights come from their own key: two heads at one
    seed agree, and building one neither reseeds nor advances the global
    random stream."""
    mx.random.seed(3)
    a = mx.random.normal((4,))
    mx.random.seed(3)
    h1 = hs.HsHead(6, DIM, seed=1, lr=0.1)
    b = mx.random.normal((4,))
    mx.eval(a, b)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    h2 = hs.HsHead(6, DIM, seed=1, lr=0.1)
    h3 = hs.HsHead(6, DIM, seed=2, lr=0.1)
    assert np.array_equal(np.asarray(h1.module.weight), np.asarray(h2.module.weight))
    assert not np.array_equal(np.asarray(h1.module.weight), np.asarray(h3.module.weight))
    y = h1.module(mx.ones((2, 6)))
    mx.eval(y)
    assert y.shape == (2, DIM) and h1.module.weight.shape == (DIM, 6)
