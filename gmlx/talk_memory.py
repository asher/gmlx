"""Long-term memory for the built-in assistant brain.

A local RAG store over the server's own retrieval endpoints: remembered
content is embedded via ``POST /v1/embeddings`` and kept in a sqlite file
(text + float32 vector blob). ``recall`` embeds the query, ranks everything
by cosine over an in-memory matrix (a personal memory store is thousands of
rows, not millions - numpy over all of it is microseconds), shortlists, and
lets ``POST /v1/rerank`` order the shortlist when the server offers it.

What gets stored is managed, not append-only - raw transcripts at voice
cadence would drown recall in near-duplicate noise within months:

- **Fact extraction** (default): after each turn a background thread asks the
  chat model to distill the exchange into up to ``_MAX_FACTS`` durable facts
  (or NONE), and stores those instead of the transcript. Off the turn thread,
  so it never adds to voice latency. Extraction failure falls back to storing
  the raw exchange (one warning).
- **Consolidation**: a new fact whose embedding is nearly identical to an
  existing one (cosine >= ``_CONSOLIDATE_SIM``) replaces that row instead of
  piling up beside it.
- **TTL + cap**: rows older than ``ttl_days`` are pruned at open; the store
  never exceeds ``max_items`` (eviction: least-recalled, then oldest -
  ``recall`` bumps a usage counter so memories that earn their keep stay).

Fits the assistant brain's memory seam (``recall(text) -> [str]`` /
``remember(user_text, answer)`` / ``close()``). Degradation is self-managed:
a server without ``/v1/embeddings`` disables the store after one warning
(recall returns [], remember is a no-op); a missing ``/v1/rerank`` just
falls back to cosine order. Vectors are tied to the server's configured
embedder - rows whose dimension no longer matches are left in place but
unreachable until the embedder comes back (one warning).
"""

from __future__ import annotations

import os
import queue
import re
import sqlite3
import threading
import time

import numpy as np

from . import talk_client
from .talk_client import TalkClientError

# Similarity floor for injection: below this a memory is unrelated noise and
# saying nothing beats saying something irrelevant.
MIN_SIM = 0.30
# How many cosine candidates the reranker gets to reorder.
_SHORTLIST_FACTOR = 4
# Answers are memory context, not transcripts - keep the gist.
_ANSWER_SNIPPET_CHARS = 240
# First allocation of the in-RAM vector matrix (rows); grows by doubling.
_INITIAL_CAP = 1024
# A new fact at least this cosine-similar to a stored one replaces it.
_CONSOLIDATE_SIM = 0.92
# Facts kept per exchange (the extraction prompt asks for the same number).
_MAX_FACTS = 3

_EXTRACT_SYSTEM = (
    "You maintain an assistant's long-term memory of its user. From the "
    "voice-chat exchange below, extract up to 3 short, self-contained facts "
    "worth remembering in future conversations: preferences, people, "
    "projects, decisions, corrections. One fact per line, no preamble. If "
    "the exchange contains nothing worth keeping long-term, reply with "
    "exactly: NONE")


def _parse_facts(text: str) -> list:
    """Model output -> fact list: strip bullets/numbering, drop NONE/empty."""
    facts = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-*•]*(?:\d+[.)])?\s*", "", line).strip()
        if not line or line.upper().rstrip(".") == "NONE":
            continue
        facts.append(line)
        if len(facts) == _MAX_FACTS:
            break
    return facts


def make_extractor(base_url: str, model: str, *,
                   api_key: str | None = None, max_tokens: int = 512):
    """Fact-extraction seam backed by the server's own chat model - one extra
    completion per remembered exchange, run on the store's background thread.
    Returns ``extract(user_text, answer) -> [fact, ...]``."""
    def extract(user_text: str, answer: str) -> list:
        from .reasoning import ReasoningFilter

        prompt = (f"user: {user_text}\n"
                  f"assistant: {answer[:_ANSWER_SNIPPET_CHARS]}")
        # Same filter ServerChatBrain applies to its own stream: a thinking
        # model's chain-of-thought must not be stored as facts.
        rf = ReasoningFilter()
        parts = []
        for d in talk_client.stream_chat(
                base_url, model=model, max_tokens=max_tokens, api_key=api_key,
                messages=[{"role": "system", "content": _EXTRACT_SYSTEM},
                          {"role": "user", "content": prompt}]):
            if d.get("reasoning"):
                continue
            c = d.get("content")
            if isinstance(c, str):
                parts.extend(span for span, mode in rf.feed(c)
                             if mode == "answer")
        parts.extend(span for span, mode in rf.flush() if mode == "answer")
        return _parse_facts("".join(parts))
    return extract


def default_memory_path() -> str:
    # `or`, not a .get default: XDG_DATA_HOME="" must also fall back, or the
    # sqlite DB lands relative to whatever cwd the process started in.
    base = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return os.path.join(os.path.expanduser(base), "gmlx",
                        "assistant-memory.db")


class MemoryStore:
    """See the module docstring. Recall runs on the turn thread, extraction on
    the worker, and ``/memory forget|clear`` on the main loop, so ``_cache`` is
    published as an immutable tuple (readers snapshot it, writers rebind it) and
    sqlite calls hold ``_lock``."""

    def __init__(self, *, base_url: str, api_key: str | None = None,
                 path: str | None = None, top_k: int = 4,
                 min_sim: float = MIN_SIM, embed=None, rerank=None,
                 extract=None, ttl_days: float | None = None,
                 max_items: int = 20000, warn=None):
        self.path = os.path.expanduser(path or default_memory_path())
        self.top_k = top_k
        self.min_sim = min_sim
        self.ttl_days = ttl_days
        self.max_items = max_items
        self._embed = embed or (lambda texts: talk_client.embed_texts(
            base_url, texts, api_key=api_key))
        self._rerank = rerank if rerank is not None else (
            lambda query, docs, top_n: talk_client.rerank_documents(
                base_url, query, docs, top_n=top_n, api_key=api_key))
        if rerank is False:                    # cosine-only, no server rerank
            self._rerank = None
        self._warn = warn or (lambda msg: print(f"[talk] {msg}",
                                                flush=True))
        self._dead = False              # embeddings unavailable -> inert
        self._rerank_ok = True          # flips off on the first rerank error
        self._dim_warned = False
        self._cache = None    # (dim, texts, mat(capacity), n) - see _matrix
        self._lock = threading.Lock()          # sqlite
        self._cache_lock = threading.Lock()    # _cache_append vs itself
        self._worker_lock = threading.Lock()   # one-shot worker bootstrap
        self._extract = extract         # (user, answer) -> [fact] | raises
        self._extract_warned = False
        self._queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # One-time migration: the default db was talk-memory.db before the
        # assistant rename; adopt it rather than starting an empty store.
        old = os.path.join(os.path.dirname(default_memory_path()),
                           "talk-memory.db")
        if (self.path == default_memory_path()
                and not os.path.exists(self.path) and os.path.exists(old)):
            os.rename(old, self.path)
            self._warn(f"memory: migrated {old} -> {self.path}")
        self._db = self._open()
        try:
            self._prune()
        except sqlite3.DatabaseError as e:
            # Data-page corruption that passed the schema check still must
            # not abort construction (server startup runs through here).
            self._db.close()
            self._db = self._quarantine_and_reopen(e)

    def _open(self) -> sqlite3.Connection:
        """Connect + ensure schema. A corrupt file at the store path (unclean
        shutdown, a stray non-sqlite file) would fail here on every future
        session - sideline it and start fresh instead."""
        db = sqlite3.connect(self.path, check_same_thread=False)
        try:
            self._ensure_schema(db)
        except sqlite3.DatabaseError as e:
            db.close()
            db = self._quarantine_and_reopen(e)
        return db

    def _quarantine_and_reopen(self, e: Exception) -> sqlite3.Connection:
        quarantine = self.path + ".corrupt"
        os.replace(self.path, quarantine)
        self._warn(f"memory: store corrupt ({e}) - sidelined to "
                   f"{quarantine}, starting fresh")
        db = sqlite3.connect(self.path, check_same_thread=False)
        self._ensure_schema(db)
        return db

    @staticmethod
    def _ensure_schema(db: sqlite3.Connection) -> None:
        db.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            " id INTEGER PRIMARY KEY,"
            " text TEXT NOT NULL UNIQUE,"
            " created REAL NOT NULL,"
            " dim INTEGER NOT NULL,"
            " vec BLOB NOT NULL,"
            " recalled INTEGER NOT NULL DEFAULT 0)")
        cols = {r[1] for r in db.execute("PRAGMA table_info(memories)")}
        if "recalled" not in cols:      # store predates the usage counter
            db.execute("ALTER TABLE memories ADD COLUMN "
                       "recalled INTEGER NOT NULL DEFAULT 0")
        db.commit()

    # -- embedding plumbing --------------------------------------------------
    def _embed_one(self, text: str) -> np.ndarray | None:
        """Embed or die quietly: the voice loop must keep working on a server
        without /v1/embeddings, so the first failure disables the store."""
        if self._dead:
            return None
        try:
            vec = np.asarray(self._embed([text])[0], dtype=np.float32)
        except (TalkClientError, IndexError, TypeError, ValueError) as e:
            self._dead = True
            self._warn(f"memory disabled: {e}")
            return None
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _matrix(self, dim: int):
        """(texts, unit-vector matrix) for every stored row of ``dim`` -
        loaded from sqlite once, then maintained incrementally by
        ``remember``. A reload per recall would cost seconds at heavy-use row
        counts (hundreds of thousands of rows) and land straight in the
        end-of-speech -> first-audio latency; the cached matmul stays in the
        low milliseconds.

        The returned ``texts`` is the live cache list and ``mat[:n]`` a view of
        the cache matrix: :meth:`_consolidate` writes a replaced fact through
        both. Rows past ``n`` are unpublished capacity."""
        # Snapshot once: delete()/clear() rebind _cache to None from another
        # thread, so re-reading self._cache after the check can subscript None.
        cache = self._cache
        if cache is not None and cache[0] == dim:
            _, texts, mat, n = cache
            return texts, mat[:n]
        # Publish while still holding the sqlite lock: delete()/clear() commit
        # under it and invalidate after releasing it, so any deletion this
        # SELECT missed lands its invalidation after this publish - a bare
        # publish here could overwrite that invalidation with the stale rows.
        with self._lock:
            rows = self._db.execute(
                "SELECT text, vec FROM memories WHERE dim = ? ORDER BY id",
                (dim,)).fetchall()
            texts = [t for t, _ in rows]
            n = len(rows)
            mat = np.zeros((max(_INITIAL_CAP, n), dim), dtype=np.float32)
            if n:
                stacked = np.stack([np.frombuffer(v, dtype=np.float32)
                                    for _, v in rows])
                norms = np.linalg.norm(stacked, axis=1, keepdims=True)
                mat[:n] = stacked / np.maximum(norms, 1e-9)
            with self._cache_lock:
                self._cache = (dim, texts, mat, n)
        return texts, mat[:n]

    def _invalidate(self) -> None:
        """Drop the cache so the next recall reloads from sqlite. Held under
        ``_cache_lock``: a bare ``self._cache = None`` can be overwritten by a
        concurrent ``_cache_append`` publishing its tuple a moment later, which
        would resurrect the rows this invalidation exists to forget."""
        with self._cache_lock:
            self._cache = None

    def _cache_append(self, text: str, vec: np.ndarray) -> None:
        """Keep the cached matrix in step with an insert (amortized O(1) via
        capacity doubling). No cache yet -> the next recall loads from sqlite
        and picks the row up anyway.

        Serialized against other appends, and published as a whole new tuple so
        a concurrent ``_matrix`` snapshot never sees a half-updated cache: its
        ``mat[:n]`` predates row ``n``, and a grown ``mat`` is a fresh array."""
        with self._cache_lock:
            cache = self._cache
            if cache is None:
                return
            if cache[0] != len(vec):
                self._cache = None            # embedder changed mid-run
                return
            dim, texts, mat, n = cache
            if n == mat.shape[0]:
                grown = np.zeros((mat.shape[0] * 2, dim), dtype=np.float32)
                grown[:n] = mat[:n]
                mat = grown
            mat[n] = vec
            texts.append(text)
            self._cache = (dim, texts, mat, n + 1)

    # -- the seam -------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            return int(self._db.execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0])

    def list_all(self, limit: int | None = None) -> list:
        """Newest-first rows as {id, text, created, recalled} dicts."""
        sql = ("SELECT id, text, created, recalled FROM memories "
               "ORDER BY created DESC, id DESC")
        args: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            args = (int(limit),)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [{"id": r[0], "text": r[1], "created": r[2], "recalled": r[3]}
                for r in rows]

    def delete(self, mem_id: int) -> bool:
        """Remove one row by id. True when a row was deleted."""
        with self._lock:
            cur = self._db.execute("DELETE FROM memories WHERE id = ?",
                                   (int(mem_id),))
            self._db.commit()
        if cur.rowcount > 0:
            self._invalidate()
            return True
        return False

    def clear(self) -> int:
        """Remove every stored memory; returns the count removed."""
        with self._lock:
            n = int(self._db.execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0])
            self._db.execute("DELETE FROM memories")
            self._db.commit()
        self._invalidate()
        return n

    def recall(self, text: str) -> list:
        if self._dead or not text.strip() or self.count() == 0:
            return []
        q = self._embed_one(text)
        if q is None:
            return []
        texts, mat = self._matrix(len(q))
        if not texts:
            if not self._dim_warned:
                self._dim_warned = True
                self._warn("memory: stored embeddings don't match the "
                           "server's embedder (dimension changed) - old "
                           "memories are unreachable until it's restored")
            return []
        sims = mat @ q
        keep = [int(i) for i in np.argsort(sims)[::-1]
                if sims[i] >= self.min_sim][:self.top_k * _SHORTLIST_FACTOR]
        if not keep:
            return []
        shortlist = [texts[i] for i in keep]
        picked = shortlist[:self.top_k]
        if (self._rerank_ok and self._rerank is not None
                and len(shortlist) > self.top_k):
            try:
                order = self._rerank(text, shortlist, self.top_k)
                picked = [shortlist[i] for i in order]
            except TalkClientError:
                self._rerank_ok = False       # cosine order from here on
        if picked:                            # earn-their-keep counter (see
            with self._lock:                  # eviction order in _prune)
                self._db.executemany(
                    "UPDATE memories SET recalled = recalled + 1 "
                    "WHERE text = ?", [(t,) for t in picked])
                self._db.commit()
        return picked

    def remember(self, user_text: str, answer: str) -> None:
        user_text = (user_text or "").strip()
        if self._dead or len(user_text) < 4:
            return
        answer = (answer or "").strip()
        if self._extract is not None:
            self._ensure_worker()
            self._queue.put((user_text, answer))
            return
        self._insert(self._exchange_text(user_text, answer),
                     consolidate=False)

    def flush(self) -> None:
        """Block until every queued exchange has been distilled + stored."""
        if self._queue is not None:
            self._queue.join()

    def close(self) -> None:
        if self._worker is not None:
            self._queue.put(None)             # drain, then stop
            self._worker.join(timeout=10.0)
        with self._lock:
            self._db.close()

    # -- managed storage ------------------------------------------------------
    @staticmethod
    def _exchange_text(user_text: str, answer: str) -> str:
        text = f"user: {user_text}"
        if answer:
            text += f"\nassistant: {answer[:_ANSWER_SNIPPET_CHARS]}"
        return text

    def _ensure_worker(self) -> None:
        # Locked: concurrent first remember() calls (served-assistant turn
        # threads share one store) must not each spawn a worker - a rebound
        # queue would orphan the loser's items and leak its thread on close.
        with self._worker_lock:
            if self._worker is None:
                self._queue = queue.Queue()
                self._worker = threading.Thread(target=self._work,
                                                args=(self._queue,),
                                                name="talk-memory", daemon=True)
                self._worker.start()

    def _work(self, q: queue.Queue) -> None:
        while True:
            item = q.get()
            try:
                if item is None:
                    return
                self._distill(*item)
            except Exception as e:            # keep the worker alive
                self._warn(f"memory: {e}")
            finally:
                q.task_done()

    def _distill(self, user_text: str, answer: str) -> None:
        """Extraction happens here, off the turn thread - the extra completion
        overlaps with speech playback instead of delaying the return to
        listening. Extraction failure -> store the raw exchange (the
        pre-extraction behavior), so a busy/limited server degrades to more
        noise, never to amnesia."""
        try:
            facts = self._extract(user_text, answer)
        except Exception as e:
            if not self._extract_warned:
                self._extract_warned = True
                self._warn(f"memory: fact extraction failed ({e}) - "
                           f"storing raw exchanges")
            self._insert(self._exchange_text(user_text, answer),
                         consolidate=False)
            return
        for fact in facts[:_MAX_FACTS]:
            fact = str(fact).strip()
            if len(fact) >= 4:
                self._insert(fact, consolidate=True)

    def _insert(self, text: str, *, consolidate: bool) -> None:
        """Store one row: exact-dedup, embed, near-dup consolidation (facts
        only - raw exchanges are timestamped records, distinct by intent),
        then insert + cap check."""
        with self._lock:
            exists = self._db.execute(
                "SELECT 1 FROM memories WHERE text = ?", (text,)).fetchone()
        if exists:
            return
        vec = self._embed_one(text)
        if vec is None:
            return
        if consolidate and self._consolidate(text, vec):
            return
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO memories (text, created, dim, vec) "
                "VALUES (?, ?, ?, ?)",
                (text, time.time(), len(vec), vec.tobytes()))
            self._db.commit()
        self._cache_append(text, vec)
        self._enforce_cap()

    def _consolidate(self, text: str, vec: np.ndarray) -> bool:
        """Replace a near-identical stored fact (keeps its recalled count -
        the fact earned it) instead of accumulating restatements."""
        texts, mat = self._matrix(len(vec))
        if not texts:
            return False
        sims = mat @ vec
        i = int(np.argmax(sims))
        if float(sims[i]) < _CONSOLIDATE_SIM:
            return False
        with self._lock:
            self._db.execute(
                "UPDATE memories SET text = ?, created = ?, vec = ? "
                "WHERE text = ?",
                (text, time.time(), vec.tobytes(), texts[i]))
            self._db.commit()
        with self._cache_lock:
            # `mat` is a view of the published matrix iff its base array is
            # still the cache's. Tuple identity is too strict: an append
            # republishes (fine, `texts` and the array are shared) - while a
            # growth or an invalidation detaches `mat`, and writing through it
            # would vanish. Reload on the next recall instead.
            cache = self._cache
            if cache is not None and cache[2] is mat.base:
                texts[i] = text               # cache list + matrix view
                mat[i] = vec                  # write through in place
            else:
                self._cache = None
        return True

    def _enforce_cap(self) -> None:
        with self._lock:
            n = int(self._db.execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0])
            excess = n - self.max_items
            if excess <= 0:
                return
            self._db.execute(
                "DELETE FROM memories WHERE id IN (SELECT id FROM memories "
                "ORDER BY recalled ASC, created ASC LIMIT ?)", (excess,))
            self._db.commit()
        self._invalidate()                    # rare: next recall reloads

    def _prune(self) -> None:
        """At open: TTL expiry, then the size cap (eviction order: never/
        least recalled first, oldest first within a tie)."""
        if self.ttl_days:
            with self._lock:
                self._db.execute(
                    "DELETE FROM memories WHERE created < ?",
                    (time.time() - self.ttl_days * 86400.0,))
                self._db.commit()
        self._enforce_cap()
