#!/usr/bin/env python3
"""MemoryStore tests through the embed/rerank seams - deterministic fake
vectors, a tmp sqlite file, no server. Covers cosine recall + the similarity
floor, server-side rerank of the shortlist (and its fallback), dedup,
persistence across reopen, self-disable without embeddings, and the
dimension-change guard. The talk_client embed/rerank HTTP wrappers are
exercised via the monkeypatched ``_http_post`` seam."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gmlx import talk_client
from gmlx.talk_client import TalkClientError
from gmlx.talk_memory import MemoryStore, default_memory_path

# Fixed fake embedding space: axis 0 = tea, 1 = bikes, 2 = weather. remember()
# prefixes rows with "user: ", so match on substrings.
_VECS = {"tea": [1.0, 0.1, 0.0], "bike": [0.0, 1.0, 0.1],
         "rain": [0.1, 0.0, 1.0]}


def _fake_embed(texts):
    out = []
    for t in texts:
        for key, v in _VECS.items():
            if key in t:
                out.append(list(v))
                break
        else:
            out.append([0.0, 0.0, 0.0])
    return out


def _store(tmp_path, _cls=MemoryStore, **kw):
    kw.setdefault("embed", _fake_embed)
    kw.setdefault("rerank", False)
    kw.setdefault("warn", lambda m: kw.setdefault("_warned", []) or None)
    return _cls(base_url="http://h:1/v1",
                path=str(tmp_path / "mem.db"), **kw)


def test_remember_recall_and_similarity_floor(tmp_path):
    m = _store(tmp_path)
    m.remember("I like green tea", "Noted.")
    m.remember("my bike is red", "Nice bike.")
    assert m.count() == 2

    got = m.recall("what tea do I drink?")
    assert len(got) == 1                          # bike fails the sim floor
    assert "green tea" in got[0] and got[0].startswith("user:")
    assert "assistant: Noted." in got[0]
    assert m.recall("zzz unrelated") == []        # zero vector -> nothing
    m.close()


def test_dedup_and_short_text_skipped(tmp_path):
    m = _store(tmp_path)
    m.remember("I like green tea", "Noted.")
    m.remember("I like green tea", "Noted.")     # exact repeat
    m.remember("ok", "sure")                     # too short to matter
    assert m.count() == 1
    m.close()


def test_persists_across_reopen(tmp_path):
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    m.close()
    m2 = _store(tmp_path)
    assert m2.count() == 1
    assert "green tea" in m2.recall("tea?")[0]
    m2.close()


def test_corrupt_db_sidelined_not_fatal(tmp_path):
    # A garbage file at the store path (unclean shutdown, stray file) must
    # not crash every future session: quarantine it and start fresh.
    junk = b"definitely not a sqlite database\n" * 8
    (tmp_path / "mem.db").write_bytes(junk)
    warned = []
    m = _store(tmp_path, warn=warned.append)
    m.remember("I like green tea", "Noted.")
    assert m.count() == 1
    m.close()
    assert (tmp_path / "mem.db.corrupt").read_bytes() == junk
    assert any("corrupt" in w for w in warned)


def test_prune_time_corruption_also_quarantined(tmp_path, monkeypatch):
    # Data-page corruption can pass the schema check and only surface in the
    # open-time prune; construction must still self-heal, not abort.
    import sqlite3

    from gmlx import talk_memory

    def bad_prune(self):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(talk_memory.MemoryStore, "_prune", bad_prune)
    warned = []
    m = _store(tmp_path, warn=warned.append)
    m.remember("I like green tea", "Noted.")
    assert m.count() == 1
    m.close()
    assert (tmp_path / "mem.db.corrupt").exists()
    assert any("corrupt" in w for w in warned)


def test_rerank_orders_shortlist_and_falls_back(tmp_path):
    calls = []

    def fake_rerank(query, docs, top_n):
        calls.append((query, list(docs), top_n))
        return list(range(len(docs)))[::-1][:top_n]   # reverse order

    m = _store(tmp_path, rerank=fake_rerank, top_k=1)
    m.remember("green tea daily", "")
    m.remember("tea with milk", "")
    got = m.recall("tea?")
    assert len(calls) == 1 and calls[0][2] == 1
    assert len(got) == 1
    assert got[0] == calls[0][1][-1]              # reversed pick honored

    def broken(query, docs, top_n):
        raise TalkClientError("no rerank route")

    m2 = _store(tmp_path, rerank=broken, top_k=1)
    got = m2.recall("tea?")
    assert len(got) == 1                          # cosine fallback
    assert m2._rerank_ok is False                 # and it stops retrying
    m.close(), m2.close()


def test_embed_failure_disables_store_once(tmp_path):
    warned = []

    def dead_embed(texts):
        raise TalkClientError("HTTP 404")

    m = _store(tmp_path, embed=dead_embed, warn=warned.append)
    m.remember("I like green tea", "")
    m.remember("my bike is red", "")
    assert m.count() == 0 and m.recall("tea?") == []
    assert len(warned) == 1 and "memory disabled" in warned[0]
    m.close()


def test_list_all_newest_first(tmp_path):
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    m.remember("my bike is red", "")
    rows = m.list_all()
    assert len(rows) == 2
    assert {"id", "text", "created", "recalled"} <= set(rows[0])
    assert rows[0]["id"] > rows[1]["id"]          # same-second: id breaks tie
    assert "bike" in rows[0]["text"]
    assert m.list_all(limit=1) == rows[:1]
    m.close()


def test_delete_removes_and_invalidates_cache(tmp_path):
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    m.remember("my bike is red", "")
    assert "green tea" in m.recall("tea?")[0]     # warms the cache
    tea_id = next(r["id"] for r in m.list_all() if "tea" in r["text"])
    assert m.delete(tea_id) is True
    assert m.delete(tea_id) is False              # already gone
    assert m.count() == 1
    assert m.recall("tea?") == []                 # cache reloaded, row gone
    m.close()


def test_clear_counts_and_empties(tmp_path):
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    m.remember("my bike is red", "")
    m.recall("tea?")                              # warm the cache
    assert m.clear() == 2
    assert m.count() == 0 and m.list_all() == []
    assert m.recall("tea?") == []
    m.close()


def test_dimension_change_warns_and_returns_nothing(tmp_path):
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    m.close()
    warned = []
    m2 = _store(tmp_path, embed=lambda ts: [[1.0] * 8 for _ in ts],
                warn=warned.append)
    assert m2.recall("tea?") == []
    assert len(warned) == 1 and "dimension changed" in warned[0]
    m2.recall("tea?")                             # warns only once
    assert len(warned) == 1
    m2.close()


def test_default_memory_path_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_memory_path() == str(tmp_path / "gmlx" /
                                        "assistant-memory.db")


def test_default_store_migrates_old_talk_db(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    d = tmp_path / "gmlx"
    d.mkdir()
    m = MemoryStore(base_url="http://h:1/v1", path=str(d / "talk-memory.db"),
                    embed=_fake_embed, rerank=False, warn=lambda m: None)
    m.remember("I like green tea", "Noted.")
    m.close()

    warned = []
    m2 = MemoryStore(base_url="http://h:1/v1", embed=_fake_embed,
                     rerank=False, warn=warned.append)
    assert m2.path == str(d / "assistant-memory.db")
    assert m2.count() == 1                       # rows adopted, not restarted
    assert not (d / "talk-memory.db").exists()
    assert any("migrated" in w for w in warned)
    m2.close()

    m3 = MemoryStore(base_url="http://h:1/v1", embed=_fake_embed,
                     rerank=False, warn=lambda m: None)
    assert m3.count() == 1                       # migration is one-time
    m3.close()


# -- the talk_client HTTP wrappers ------------------------------------------
def test_embed_texts_parses_openai_shape(monkeypatch):
    seen = {}

    def fake_post(url, data, headers, timeout):
        seen["url"], seen["body"] = url, json.loads(data)
        return json.dumps({"data": [
            {"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]},
        ]}).encode()

    monkeypatch.setattr(talk_client, "_http_post", fake_post)
    out = talk_client.embed_texts("http://h:1/v1", ["a", "b"])
    assert seen["url"].endswith("/v1/embeddings")
    assert seen["body"] == {"input": ["a", "b"]}   # no model field
    assert out == [[0.1], [0.2]]                   # index order restored


def test_rerank_documents_parses_results(monkeypatch):
    def fake_post(url, data, headers, timeout):
        assert url.endswith("/v1/rerank")
        assert json.loads(data)["return_documents"] is False
        return json.dumps({"results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
        ]}).encode()

    monkeypatch.setattr(talk_client, "_http_post", fake_post)
    assert talk_client.rerank_documents("http://h:1/v1", "q",
                                        ["a", "b", "c"], top_n=2) == [2, 0]


def test_embed_texts_errors_wrap(monkeypatch):
    import urllib.error

    def boom(url, data, headers, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(talk_client, "_http_post", boom)
    with pytest.raises(TalkClientError, match="embeddings failed"):
        talk_client.embed_texts("http://h:1/v1", ["a"])


def test_recall_caches_matrix_and_tracks_inserts(tmp_path):
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    m.recall("tea?")                              # builds the cache
    mat_before = m._cache[2]
    loads = []
    real_execute = m._db.execute

    def counting_execute(sql, *a):
        if sql.lstrip().startswith("SELECT text, vec"):
            loads.append(sql)
        return real_execute(sql, *a)

    m._db = type("DB", (), {"execute": staticmethod(counting_execute),
                            "executemany": m._db.executemany,
                            "commit": m._db.commit, "close": m._db.close})()
    m.remember("my bike is red", "")              # append, not reload
    got = m.recall("what bike do I ride?")
    assert "bike" in got[0]                       # new row visible via cache
    assert loads == []                            # sqlite never re-scanned
    assert m._cache[2] is mat_before and m._cache[3] == 2
    m.close()


def test_cache_capacity_doubles(tmp_path, monkeypatch):
    from gmlx import talk_memory
    monkeypatch.setattr(talk_memory, "_INITIAL_CAP", 1)
    m = _store(tmp_path)
    m.remember("green tea daily", "")
    m.recall("tea?")                              # cache: capacity 1, n 1
    m.remember("tea with milk", "")               # grow to 2
    m.remember("tea with lemon", "")              # grow to 4
    assert m._cache[3] == 3 and m._cache[2].shape[0] == 4
    assert len(m.recall("tea?")) >= 1
    m.close()


def test_recall_correct_after_growth(tmp_path, monkeypatch):
    # The promise behind the doubling rule: recall stays CORRECT after the
    # cache matrix has grown - rows appended through growth are recallable.
    from gmlx import talk_memory
    monkeypatch.setattr(talk_memory, "_INITIAL_CAP", 1)
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    m.recall("tea?")                              # cache built at capacity 1
    m.remember("my bike is red", "")              # grow to 2
    m.remember("rain is coming", "")              # grow to 4
    got = m.recall("what bike do I ride?")
    assert len(got) == 1 and "bike" in got[0]
    assert "rain" in m.recall("rain today?")[0]   # tail row also recallable
    m.close()


def test_rerank_failure_not_retried(tmp_path):
    # Sibling of the _rerank_ok latch test: a broken rerank endpoint is tried
    # exactly once; later recalls go straight to the cosine fallback.
    calls = []

    def broken(query, docs, top_n):
        calls.append(query)
        raise TalkClientError("no rerank route")

    m = _store(tmp_path, rerank=broken, top_k=1)
    m.remember("green tea daily", "")
    m.remember("tea with milk", "")
    got1 = m.recall("tea?")
    got2 = m.recall("tea?")
    assert len(calls) == 1                        # one attempt, never retried
    assert len(got1) == 1 and len(got2) == 1      # fallback still answers
    m.close()


# -- managed storage: extraction / consolidation / TTL / cap -----------------
def test_extracted_facts_stored_instead_of_transcript(tmp_path):
    m = _store(tmp_path, extract=lambda u, a: ["user drinks green tea",
                                               "user rides a red bike"])
    m.remember("what tea should I buy?", "Try sencha.")
    m.flush()
    assert m.count() == 2                         # facts, not the exchange
    got = m.recall("tea?")
    assert got == ["user drinks green tea"]       # stored verbatim, no prefix
    m.close()


def test_extract_none_stores_nothing(tmp_path):
    m = _store(tmp_path, extract=lambda u, a: [])
    m.remember("what time is it?", "It's 3pm.")
    m.flush()
    assert m.count() == 0
    m.close()


def test_extract_failure_falls_back_to_raw_exchange(tmp_path):
    warned = []

    def boom(u, a):
        raise TalkClientError("HTTP 500")

    m = _store(tmp_path, extract=boom, warn=warned.append)
    m.remember("I like green tea", "Noted.")
    m.remember("my bike is red", "Nice.")
    m.flush()
    assert m.count() == 2
    assert len(warned) == 1                       # warns once, keeps working
    assert "extraction failed" in warned[0]
    assert m.recall("tea?")[0].startswith("user:")
    m.close()


def test_near_duplicate_fact_consolidates(tmp_path):
    outs = [["I like green tea"], ["I love strong green tea daily"]]
    m = _store(tmp_path, extract=lambda u, a: outs.pop(0))
    m.remember("tea question one", "x")
    m.flush()
    m.remember("tea question two", "y")
    m.flush()
    assert m.count() == 1                         # replaced, not appended
    assert m.recall("tea?") == ["I love strong green tea daily"]
    m.close()


def test_ttl_prunes_expired_rows_at_open(tmp_path):
    import sqlite3
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    m.close()
    db = sqlite3.connect(str(tmp_path / "mem.db"))
    db.execute("UPDATE memories SET created = created - 10 * 86400")
    db.commit(), db.close()
    m2 = _store(tmp_path, ttl_days=7.0)
    assert m2.count() == 0
    m2.close()


def test_cap_evicts_least_recalled_oldest(tmp_path):
    m = _store(tmp_path, max_items=2)
    m.remember("I like green tea", "")
    m.remember("my bike is red", "")
    assert len(m.recall("tea?")) == 1             # bumps the tea row
    m.remember("rain is coming", "")              # over cap -> evict
    assert m.count() == 2
    texts = {r[0] for r in m._db.execute("SELECT text FROM memories")}
    assert any("tea" in t for t in texts)         # recalled row survives
    assert not any("bike" in t for t in texts)    # never-recalled oldest goes
    assert len(m.recall("rain today?")) == 1      # cache rebuilt post-evict
    m.close()


def test_old_schema_gains_recalled_column(tmp_path):
    import sqlite3
    db = sqlite3.connect(str(tmp_path / "mem.db"))
    db.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY,"
               " text TEXT NOT NULL UNIQUE, created REAL NOT NULL,"
               " dim INTEGER NOT NULL, vec BLOB NOT NULL)")
    db.commit(), db.close()
    m = _store(tmp_path)
    m.remember("I like green tea", "")
    assert len(m.recall("tea?")) == 1             # bump uses the new column
    m.close()


def test_make_extractor_prompts_and_parses(monkeypatch):
    from gmlx import talk_memory
    seen = {}

    def fake_stream(base_url, *, model, messages, max_tokens,
                    api_key=None, tools=None, timeout=600.0):
        seen["model"], seen["messages"] = model, messages
        yield {"reasoning": "thinking..."}        # ignored
        yield {"content": "- likes green tea\n"}
        yield {"content": "NONE\n2) rides a red bike"}
        yield {"_finish": "stop"}

    monkeypatch.setattr(talk_memory.talk_client, "stream_chat", fake_stream)
    extract = talk_memory.make_extractor("http://h:1/v1", "m1")
    assert extract("I like tea", "noted") == ["likes green tea",
                                              "rides a red bike"]
    assert seen["model"] == "m1"
    assert seen["messages"][0]["role"] == "system"
    assert "I like tea" in seen["messages"][1]["content"]


def test_extractor_strips_reasoning_from_facts(monkeypatch):
    """A thinking model's chain-of-thought (reasoning deltas or inline
    <think> spans) must not be stored as facts."""
    from gmlx import talk_client
    from gmlx.talk_memory import make_extractor

    def fake_stream_chat(base_url, **kw):
        yield {"reasoning": "the user seems to hate cats"}
        yield {"content": "<think>secret plan"}
        yield {"content": " about cats</think>"}
        yield {"content": "User's dog is named Rex\n"}
        yield {"content": "User lives in Lisbon"}

    monkeypatch.setattr(talk_client, "stream_chat", fake_stream_chat)
    extract = make_extractor("http://x/v1", "m")
    facts = extract("hi", "hello")
    assert facts == ["User's dog is named Rex", "User lives in Lisbon"]


def test_matrix_reads_cache_once(tmp_path):
    """`/memory forget` (main loop) rebinds _cache to None while recall() is in
    _matrix on the turn thread. _matrix must snapshot _cache, not re-read it
    after the None check - the second read used to raise TypeError mid-turn."""

    class _Vanishing(MemoryStore):
        """_cache goes None right after the first read: the delete() interleave,
        made deterministic instead of depending on a one-bytecode window."""

        @property
        def _cache(self):
            v = self.__dict__.get("_cache_val")
            self.__dict__["_cache_val"] = None
            return v

        @_cache.setter
        def _cache(self, v):
            self.__dict__["_cache_val"] = v

    m = _store(tmp_path, _cls=_Vanishing)
    m.remember("I like green tea", "Noted.")
    m.flush()
    m._matrix(3)                      # populate the cache
    texts, mat = m._matrix(3)         # used to raise TypeError on the 2nd read
    assert len(texts) == 1 and mat.shape[0] == 1


def test_invalidation_is_not_lost_to_a_concurrent_append(tmp_path):
    """`/memory clear` on the main loop must not be overwritten by the
    extraction worker's in-flight _cache_append republishing its tuple."""
    import threading

    m = _store(tmp_path)
    m.remember("I like green tea", "Noted.")
    m.flush()
    m._matrix(3)
    dim, texts, mat, n = m._cache

    class _Trap(list):
        """Fires delete()'s invalidation while the append holds _cache_lock."""

        def append(self, x):
            t = threading.Thread(target=m.clear)   # the /memory clear thread
            t.start()
            t.join(0.2)                       # blocks until the append publishes
            self.thread = t
            super().append(x)

    trap = _Trap(texts)
    m._cache = (dim, trap, mat, n)
    m._cache_append("new fact", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    trap.thread.join(2.0)
    assert m._cache is None                   # the clear won, not the append
    assert m.count() == 0


def test_matrix_reload_does_not_overwrite_a_concurrent_invalidation(
        tmp_path, monkeypatch):
    """clear() landing while _matrix rebuilds from sqlite must not have its
    invalidation overwritten by the reload's publish, or the cache serves the
    cleared rows until something else invalidates it. The publish therefore
    stays inside the sqlite lock, ordered against clear()'s commit."""
    import threading

    m = _store(tmp_path)
    m.remember("I like green tea", "Noted.")
    m.flush()

    real_stack = np.stack
    fired = {}

    def stack_hook(*a, **kw):
        if "t" not in fired:              # inside the reload's matrix build
            t = threading.Thread(target=m.clear)
            t.start()
            t.join(0.5)   # pre-fix the sqlite lock was free here: clear() lands
            fired["t"] = t
        return real_stack(*a, **kw)

    monkeypatch.setattr(np, "stack", stack_hook)
    m._matrix(3)
    fired["t"].join(2.0)
    assert m._cache is None               # the clear won, not the stale reload


class _AppendDuringUpdate:
    """The worker thread landing an insert while _consolidate is mid-UPDATE."""

    def __init__(self, db, store):
        self._db = db
        self._store = store

    def execute(self, sql, *args):
        if sql.startswith("UPDATE"):
            self._store._cache_append(
                "worker fact", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return self._db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._db, name)


def test_consolidate_drops_the_cache_when_its_matrix_detached(tmp_path):
    """An append that GROWS the matrix while _consolidate is inside the sqlite
    UPDATE republishes a fresh array; _consolidate's `mat` view is detached by
    then, so the write-through would vanish: invalidate instead, or recall
    keeps serving the pre-consolidation fact."""
    m = _store(tmp_path)
    m.remember("I like green tea", "Noted.")
    m.flush()
    m._matrix(3)
    dim, texts, mat, n = m._cache
    m._cache = (dim, texts, mat[:n].copy(), n)  # full: the next append grows

    m._db = _AppendDuringUpdate(m._db, m)
    assert m._consolidate("I love strong green tea daily",
                          np.array([1.0, 0.0, 0.0], dtype=np.float32)) is True
    assert m._cache is None                   # reload, don't trust a stale view


def test_consolidate_writes_through_a_non_growing_concurrent_append(tmp_path):
    """An append that fits the capacity shares the matrix array (and the texts
    list), so the write-through is still live - the cache must survive it, not
    be dropped on mere tuple identity."""
    m = _store(tmp_path)
    m.remember("I like green tea", "Noted.")
    m.flush()
    m._matrix(3)

    m._db = _AppendDuringUpdate(m._db, m)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert m._consolidate("I love strong green tea daily", vec) is True
    dim, texts, mat, n = m._cache             # kept: append + write-through
    assert n == 2 and texts[0] == "I love strong green tea daily"
    assert np.allclose(mat[0], vec)


def test_concurrent_first_remember_spawns_one_worker(tmp_path):
    """The worker bootstrap is locked: N threads racing the first remember()
    must produce exactly one worker/queue, and every queued exchange must be
    distilled (an unlocked bootstrap rebound the queue, orphaning items)."""
    import threading

    seen = []
    m = _store(tmp_path, extract=lambda u, a: seen.append(u) or [])
    barrier = threading.Barrier(8)

    def turn(i):
        barrier.wait()
        m.remember(f"exchange number {i}", "Noted.")

    threads = [threading.Thread(target=turn, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    m.flush()
    assert len(seen) == 8
    workers = [t for t in threading.enumerate() if t.name == "talk-memory"]
    assert len(workers) == 1
