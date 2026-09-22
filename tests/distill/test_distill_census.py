#!/usr/bin/env python3
"""``gmlx distill census`` on minted reply caches: the coarsened KL and
residual closed forms against small hand references, pairing across two
caches whose prompts differ but whose replies agree, the high_delta map
keyed by corpus id, reply mismatches, the refusals, and the eval side's
restriction of a reply row to the census byte ranges. CPU only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import gmlx.distill as dl
import gmlx.distill.eval as dl_eval
from gmlx.distill import census as cs

from .test_distill_lib import _TEMPLATE_A, _bytelevel_tokenizer, _with_template


@pytest.fixture(scope="module")
def tok():
    t = _bytelevel_tokenizer([("\u0120", "t"), ("h", "e"), ("\u0120t", "he"), ("\u0120", "a"),
                              ("i", "s"), ("\u0120", "is"), ("c", "a"), ("ca", "t")])
    return _with_template(t, _TEMPLATE_A)


def _reply_cache(tmp: Path, tok, convs: list[list[dict]], *, doc_prefix: str, boost: dict | None = None,
                 K: int = 8, seed: int = 5):
    """A reply-frame cache over convs (each ending with the assistant reply)
    from a synthetic head. boost maps a reply-relative byte offset (one at
    which every row has a token boundary, 0 is always one) to the nats
    added to the on-path token's logit at the position that predicts the
    token starting there, so a context cache can be made to move the
    teacher at chosen positions. Hidden states depend on the byte offset
    from the reply start only, so two caches with the same seed agree
    wherever no boost applies."""
    tb = dl.token_bytes(tok)
    V = len(tb)
    log_bmask = dl.log_bmask_from(dl.whitespace_start_mask(tok, V, tb))
    W = mx.array((np.random.default_rng(seed).standard_normal((V, 6)) * 0.5).astype(np.float32))
    writer = dl.ShardWriter(tmp, K, False)
    rows, metas, tbytes = [], [], []
    for r, msgs in enumerate(convs):
        text, spans = dl.render_row(tok, msgs, open_tail=False, last_only=True)
        ids, ends, _ = dl.encode_with_byte_ends(tok, text, tb, add_special_tokens=False)
        n = len(ids)
        b0 = spans[-1][0]
        # hidden states keyed by the byte offset from the reply start, so
        # reply positions agree across caches whose frames differ in length
        h = np.zeros((n, 6), dtype=np.float32)
        for t in range(n):
            h[t] = np.random.default_rng(seed * 1000 + int(ends[t]) - b0 + 500).standard_normal(6)
        nxt = np.concatenate([ids[1:], [-1]]).astype(np.int32)
        logits = (mx.array(h) @ W.T)
        if boost:
            e = ends.astype(np.int64)
            for off, nats in boost.items():
                t = int(np.nonzero(e[:-1] - b0 == off)[0][0])
                logits[t, int(nxt[t])] = logits[t, int(nxt[t])] + nats
        tm = dl.target_mask(ends.astype(np.int64), spans)
        valid = (nxt >= 0) & tm
        red = dl.reduce_logits(logits.astype(mx.bfloat16), mx.array(nxt), K=K, log_bmask=log_bmask,
                               onpath_valid=valid)
        red["onpath_mask"] = valid
        red["token_ids"] = ids
        red["token_end_byte"] = ends
        rows.append(red)
        tbytes.append(text)
        metas.append(dl.RowMeta(row_id=r, doc_id=f"{doc_prefix}:{r}", window=0, n_tokens=n, frame="reply",
                                messages=msgs, spans=[list(s) for s in spans],
                                prefix_n_tokens=int(np.argmax(tm)) + 1, suffix_start_byte=b0))
    writer.write(0, dl.pack_shard(rows, tbytes, K, False), metas, wall_s=0.01, step=64)
    dl.write_manifest(tmp, teacher_path="synthetic", dataset="synthetic", num_samples=len(rows),
                      max_seq_len=64, seed=seed, top_k=K, vocab_size=V, config_vocab_size=V,
                      tokenizer_hash=dl.vocab_map_hash(tok), batch_size=8,
                      gmlx_distill={"mlx_kld_compatible": False, "corpus_sha256": "x",
                                    "frame": {"kind": "reply"}})
    return tmp


def _conv(prompt: str, reply: str) -> list[dict]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]


REPLIES = ["the cat is the cat", "is the cat a cat", "a cat is a cat is a cat"]


def test_coarsened_kl_and_residual_closed_forms():
    ids = np.array([3, 7, 9])
    lp = np.log(np.array([0.5, 0.3, 0.1]))
    assert cs.coarsened_kl(ids, lp, ids, lp) == pytest.approx(0.0, abs=1e-12)
    # q holds two of p's ids: shared terms plus one rest bucket, by hand
    q_ids = np.array([7, 3, 11])
    q_lp = np.log(np.array([0.2, 0.4, 0.3]))
    p_in, q_in = 0.5 + 0.3, 0.4 + 0.2
    ref = 0.5 * math.log(0.5 / 0.4) + 0.3 * math.log(0.3 / 0.2) \
        + (1 - p_in) * math.log((1 - p_in) / (1 - q_in))
    assert cs.coarsened_kl(ids, lp, q_ids, q_lp) == pytest.approx(ref, rel=1e-9)
    # identical contexts leave no residual; disjoint supports leave a lot
    assert cs.residual_kl([(ids, lp), (ids, lp)]) == pytest.approx(0.0, abs=1e-12)
    far = cs.residual_kl([(ids, lp), (np.array([20, 21, 22]), lp)])
    assert far > 0.3
    assert cs.row_key("a.jsonl:4", "line") == "4" and cs.row_key("a.jsonl:4", "doc") == "a.jsonl:4"


def test_census_pairs_rows_and_writes_high_delta(tmp_path, tok):
    convs_without = [_conv(f"say it {i}", r) for i, r in enumerate(REPLIES)]
    convs_with = [_conv(f"with the long context text here {i}, say it {i}", r) for i, r in enumerate(REPLIES)]
    without = _reply_cache(tmp_path / "without", tok, convs_without, doc_prefix="a.jsonl")
    # the context moves the teacher at the first reply token of every row
    # (it gains 6 nats) and nowhere else
    with_ = _reply_cache(tmp_path / "with", tok, convs_with, doc_prefix="b.jsonl", boost={0: 6.0})
    assert dl.validate_cache(without) == [] and dl.validate_cache(with_) == []
    corpus = tmp_path / "prompts.jsonl"
    corpus.write_text("".join(json.dumps({"id": f"p{i}", "messages": c}) + "\n"
                              for i, c in enumerate(convs_without)), encoding="utf-8")
    out, md = tmp_path / "census.json", tmp_path / "census.md"
    rc = cs.run_census(cs.CensusOptions(without=str(without), with_=[str(with_)], out=str(out), md=str(md),
                                        corpus=str(corpus)))
    assert rc == 0
    s = json.loads(out.read_text())
    assert s["rows"] == 3 and s["rows_mismatched"] == 0 and s["positions"] > 3
    assert s["pair_by"] == "line" and s["caches"]["with"] == [str(with_)]
    # exactly one high-delta position per row, keyed by the corpus id, whose
    # range starts at the boosted offset and spans that token's bytes
    assert set(s["high_delta"]) == {"p0", "p1", "p2"}
    for rid, ranges in s["high_delta"].items():
        assert len(ranges) == 1 and ranges[0][0] == 0 and ranges[0][1] > 0
    assert s["teacher_high_delta"]["positions"] == 3
    assert s["teacher_high_delta"]["nll_nats_with"] < s["teacher_high_delta"]["nll_nats_without"]
    assert s["high_delta_fraction"] == pytest.approx(3 / s["positions"])
    assert s["mean_onpath_delta_nats"] > 0 and s["distillable_effect_kl_nats"] > 0
    assert s["residual_kl_nats"] is None
    assert sum(s["delta_histogram"]["counts"]) == s["positions"]
    for row in s["per_row"]:
        assert row["high_delta_positions"] == 1 and row["id"].startswith("p")
    text = md.read_text()
    assert "| paired reply rows | 3 (0 reply mismatches skipped) |" in text and "Delta histogram" in text
    # a second context identical to the first: residual zero, mismatch counted
    # for a row whose reply differs
    convs_other = [_conv(f"other {i}", r) for i, r in enumerate(REPLIES)]
    convs_other[1] = _conv("other 1", "is the cat a dog")
    other = _reply_cache(tmp_path / "other", tok, convs_other, doc_prefix="c.jsonl", boost={0: 6.0})
    out2 = tmp_path / "census2.json"
    rc = cs.run_census(cs.CensusOptions(without=str(without), with_=[str(with_), str(other)], out=str(out2)))
    assert rc == 0
    s2 = json.loads(out2.read_text())
    assert s2["rows"] == 2 and s2["rows_mismatched"] == 1
    assert s2["residual_kl_nats"] == pytest.approx(0.0, abs=1e-6)
    assert set(s2["high_delta"]) == {"a.jsonl:0", "a.jsonl:2"}


def test_census_refusals(tmp_path, tok):
    convs = [_conv(f"say it {i}", r) for i, r in enumerate(REPLIES)]
    a = _reply_cache(tmp_path / "a", tok, convs, doc_prefix="a.jsonl")
    out = tmp_path / "c.json"
    assert cs.run_census(cs.CensusOptions(without=str(a), with_=[str(tmp_path / "nope")], out=str(out))) == 2
    # pairing by the full doc_id finds nothing across differently named corpora
    b = _reply_cache(tmp_path / "b", tok, convs, doc_prefix="b.jsonl")
    assert cs.run_census(cs.CensusOptions(without=str(a), with_=[str(b)], out=str(out), pair_by="doc")) == 2
    assert not out.exists()
    assert cs.run_census(cs.CensusOptions(without=str(a), with_=[str(b)], out=str(out), max_rows=1)) == 0
    assert json.loads(out.read_text())["rows"] == 1


def test_eval_reply_rows_restricted_to_census_positions(tok):
    tb = dl.token_bytes(tok)
    row = {"id": "p0", "messages": _conv("say it", REPLIES[0])}
    full, dropped = dl_eval._span_rows(tok, [row], max_len=64, last_only=True)
    assert dropped == 0 and len(full) == 1
    _ids, _tm, nbytes_full, rid = full[0]
    assert rid == "p0"
    # the reply is "the cat is the cat"; bytes 4..7 hold "cat" (a range
    # relative to the content start, as the census writes them)
    ranges = {"p0": [[4, 7]]}
    part, _ = dl_eval._span_rows(tok, [row], max_len=64, last_only=True, positions=ranges)
    assert len(part) == 1
    ids, tm, nbytes, _ = part[0]
    assert 0 < nbytes < nbytes_full
    e = np.array(dl.encode_with_byte_ends(tok, dl.render_row(tok, row["messages"], open_tail=False,
                                                              last_only=True)[0], tb,
                                          add_special_tokens=False)[1]).astype(np.int64)
    text, spans = dl.render_row(tok, row["messages"], open_tail=False, last_only=True)
    b0 = spans[-1][0]
    starts = e[:-1] - b0
    kept = np.nonzero(tm[:-1])[0]
    assert kept.size and all(4 <= starts[t] < 7 for t in kept)
    # a row the map does not name scores nothing and is left out
    none, _ = dl_eval._span_rows(tok, [row], max_len=64, last_only=True, positions={})
    assert none == []
