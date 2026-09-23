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
                 K: int = 8, seed: int = 5, docs: list[tuple[str, int]] | None = None,
                 boost_rows: set[int] | None = None, content_offset: int = 0):
    """A reply-frame cache over convs (each ending with the assistant reply)
    from a synthetic head. boost maps a reply-relative byte offset (one at
    which every row has a token boundary, 0 is always one) to the nats
    added to the on-path token's logit at the position that predicts the
    token starting there, so a context cache can be made to move the
    teacher at chosen positions. Hidden states depend on the byte offset
    from the reply start only, so two caches with the same seed agree
    wherever no boost applies. docs, when given, names each row's (doc_id,
    window) instead of one window-0 document per row. boost_rows limits
    the boost to those row indexes."""
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
        if boost and (boost_rows is None or r in boost_rows):
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
        doc_id, window = docs[r] if docs else (f"{doc_prefix}:{r}", 0)
        metas.append(dl.RowMeta(row_id=r, doc_id=doc_id, window=window, n_tokens=n, frame="reply",
                                messages=msgs, spans=[list(s) for s in spans],
                                prefix_n_tokens=int(np.argmax(tm)) + 1, suffix_start_byte=b0,
                                content_start=b0 + content_offset))
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


def test_high_delta_keeps_the_last_window_of_a_document(tmp_path, tok):
    """A per-turn cache holds several windows of one document under one
    id; the high_delta map keeps the last window's ranges, the final turn
    that eval's reply slice scores, whatever order the rows were cached in."""
    # window 1 is cached first; its reply opens with a 3-byte token where
    # window 0's opens with a 1-byte one, so the kept range tells them apart
    convs = [_conv("second turn", "cat is cat"), _conv("first turn", "the cat")]
    docs = [("a.jsonl:0", 1), ("a.jsonl:0", 0)]
    without = _reply_cache(tmp_path / "without", tok, convs, doc_prefix="a.jsonl", docs=docs)
    with_ = _reply_cache(tmp_path / "with", tok, convs, doc_prefix="b.jsonl", docs=docs, boost={0: 6.0})
    out = tmp_path / "census.json"
    assert cs.run_census(cs.CensusOptions(without=str(without), with_=[str(with_)], out=str(out))) == 0
    s = json.loads(out.read_text())
    assert s["rows"] == 2 and [r["high_delta_positions"] for r in s["per_row"]] == [1, 1]
    assert s["high_delta"] == {"a.jsonl:0": [[0, 3]]}


def test_high_delta_is_the_last_windows_or_nothing(tmp_path, tok):
    """A document whose last window has no high-delta position gets no
    map, whatever an earlier window held, and a document cut by max_rows
    before its last window gets none either."""
    convs = [_conv("second turn", "cat is cat"), _conv("first turn", "the cat"), _conv("third turn", "the cat")]
    docs = [("a.jsonl:0", 1), ("a.jsonl:0", 0), ("a.jsonl:0", 2)]
    without = _reply_cache(tmp_path / "without", tok, convs, doc_prefix="a.jsonl", docs=docs)
    flat_last = _reply_cache(tmp_path / "flat", tok, convs, doc_prefix="b.jsonl", docs=docs, boost={0: 6.0},
                             boost_rows={0, 1})
    out = tmp_path / "census.json"
    assert cs.run_census(cs.CensusOptions(without=str(without), with_=[str(flat_last)], out=str(out))) == 0
    s = json.loads(out.read_text())
    assert [r["high_delta_positions"] for r in s["per_row"]] == [1, 1, 0]
    assert s["high_delta"] == {}
    every = _reply_cache(tmp_path / "every", tok, convs, doc_prefix="c.jsonl", docs=docs, boost={0: 6.0})
    assert cs.run_census(cs.CensusOptions(without=str(without), with_=[str(every)], out=str(out))) == 0
    assert json.loads(out.read_text())["high_delta"] == {"a.jsonl:0": [[0, 1]]}
    assert cs.run_census(cs.CensusOptions(without=str(without), with_=[str(every)], out=str(out),
                                          max_rows=2)) == 0
    assert json.loads(out.read_text())["high_delta"] == {}


def test_corpus_ids_fall_back_to_the_non_blank_row_index(tmp_path):
    """A row without an id is named by its index among the non-blank rows,
    the way eval names a reply-slice row, while the pairing key stays the
    cache's line number."""
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"messages": []}) + "\n\n" + json.dumps({"messages": []}) + "\n"
                      + json.dumps({"id": "z", "messages": []}) + "\n")
    assert cs.corpus_ids(corpus, "line") == {"0": "0", "2": "1", "3": "z"}


def test_corpus_ids_keep_unicode_line_separators(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"id": "a", "text": "x\u2028y"}, ensure_ascii=False) + "\n"
                      + json.dumps({"id": "b", "text": "z"}) + "\n", encoding="utf-8")
    assert cs.corpus_ids(corpus, "line") == {"0": "a", "1": "b"}


def test_census_records_the_frame_and_eval_refuses_a_mismatch(tmp_path, tok, capsys):
    """The positions map says which frame measured it, since its byte
    ranges start at the trace on a reply-think cache and at the reply
    otherwise; eval refuses the map under the other setting."""
    from gmlx.distill import evaluate as _ev

    convs_without = [_conv(f"say it {i}", r) for i, r in enumerate(REPLIES)]
    convs_with = [_conv(f"with context {i}, say it {i}", r) for i, r in enumerate(REPLIES)]
    without = _reply_cache(tmp_path / "without", tok, convs_without, doc_prefix="a.jsonl")
    with_ = _reply_cache(tmp_path / "with", tok, convs_with, doc_prefix="b.jsonl", boost={0: 6.0})
    out = tmp_path / "census.json"
    assert cs.run_census(cs.CensusOptions(without=str(without), with_=[str(with_)], out=str(out))) == 0
    s = json.loads(out.read_text())
    assert s["frame"] == "reply"
    student = tmp_path / "student.gguf"
    student.write_bytes(b"")
    reply = tmp_path / "reply.jsonl"
    reply.write_text(json.dumps({"id": "a.jsonl:0", "messages": convs_without[0]}) + "\n")

    def run(**kw):
        opts = _ev.EvalOptions(student=str(student), md=str(tmp_path / "r.md"), json=str(tmp_path / "r.json"),
                               reply_positions=str(out), reply_slices=[f"held={reply}"], **kw)
        return _ev.run_eval(opts), capsys.readouterr().err

    rc, err = run(reply_think=True)
    assert rc == 2 and "measured on a reply cache" in err and "--reply-think" in err
    s["frame"] = "reply-think"
    out.write_text(json.dumps(s))
    rc, err = run()
    assert rc == 2 and "measured on a reply-think cache" in err
    (tmp_path / "nocache").mkdir()
    rc, err = run(reply_think=True, cache=str(tmp_path / "nocache"))
    assert rc == 2 and "unreadable input" in err and "measured on a" not in err


def test_census_walks_the_rows_shard_by_shard():
    """Cache rows are sorted by length across shards while the pairing keys
    sort by id, so the walk is ordered by shard to load each one once."""
    class _Reader:
        index = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]

    rows = {("d", 0): 0, ("a", 0): 1, ("c", 0): 2, ("b", 0): 3, ("e", 0): 4}
    keys = sorted(rows)
    assert cs.walk_order(_Reader(), rows, keys) == [("d", 0), ("c", 0), ("a", 0), ("e", 0), ("b", 0)]


def test_census_anchors_content_ranges_at_the_content_start(tmp_path, tok):
    """On a reply-think cache the target span opens at the trace, and the
    gap between trace and content differs between templates; the map
    keeps content positions relative to the content start and trace
    positions, separately, relative to the trace start."""
    convs_without = [_conv(f"say it {i}", r) for i, r in enumerate(REPLIES)]
    convs_with = [_conv(f"with context {i}, say it {i}", r) for i, r in enumerate(REPLIES)]
    # the reply "the cat is the cat" opens with the one-byte token "t"
    # and has boundaries at 7 and 10 around " is"; with the content
    # declared to start at byte 3 the first token is trace and " is" is
    # content bytes 4..7
    without = _reply_cache(tmp_path / "without", tok, convs_without, doc_prefix="a.jsonl", content_offset=3)
    with_ = _reply_cache(tmp_path / "with", tok, convs_with, doc_prefix="b.jsonl", boost={0: 6.0, 7: 6.0},
                         boost_rows={0}, content_offset=3)
    out = tmp_path / "census.json"
    assert cs.run_census(cs.CensusOptions(without=str(without), with_=[str(with_)], out=str(out))) == 0
    s = json.loads(out.read_text())
    assert s["high_delta"] == {"a.jsonl:0": [[4, 7]]}
    assert s["high_delta_trace"] == {"a.jsonl:0": [[0, 1]]}


_TEMPLATE_TRACE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
                   "{% if m['role'] == 'assistant' and m['reasoning_content'] %}<think>{{ m['reasoning_content'] }}"
                   "</think>\n{% endif %}{{ m['content'] }}<|im_end|>\n{% endfor %}")


def test_eval_reply_rows_apply_trace_and_content_ranges_from_their_own_anchors(tok):
    tb = dl.token_bytes(tok)
    tk = _with_template(tok, _TEMPLATE_TRACE)
    row = {"id": "p0", "messages": [{"role": "user", "content": "say it"},
                                    {"role": "assistant", "content": "is the cat", "reasoning_content": "the cat"}]}
    text, spans = dl.render_row(tk, row["messages"], open_tail=False, last_only=True, reason_target=True)
    b0, b1, _b2 = spans[-1]
    cstart = b1 - len(b"is the cat")
    assert text[b0:b0 + 7] == b"the cat" and text[cstart:b1] == b"is the cat" and cstart - b0 > 7
    e = np.array(dl.encode_with_byte_ends(tk, text, tb, add_special_tokens=False)[1]).astype(np.int64)
    rows, _ = dl_eval._span_rows(tk, [row], max_len=256, last_only=True, reason_target=True,
                                 positions={"p0": [[0, 2]]}, trace_positions={"p0": [[4, 7]]})
    assert len(rows) == 1
    _ids, tm, nbytes, _ = rows[0]
    kept = np.nonzero(tm[:-1])[0]
    starts = e[:-1]
    assert kept.size == 2
    assert all((cstart <= starts[t] < cstart + 2) or (b0 + 4 <= starts[t] < b0 + 7) for t in kept)
    only_trace, _ = dl_eval._span_rows(tk, [row], max_len=256, last_only=True, reason_target=True,
                                       positions={}, trace_positions={"p0": [[0, 3]]})
    assert len(only_trace) == 1
