"""The batch loader: compiles a view of the cache for one student on the fly
(or reads a materialized one) and yields the target tensors the loss
consumes, with length buckets and a seeded order."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from gmlx.load.tokenizer import hf_inner, token_bytes

from .align import Tables, project_topk, shared_boundaries, tokenization_bias_check
from .constants import GB, NEG_INF
from .format import (
    ROUTES_FIELD,
    load_shard,
    read_json,
    read_rows_jsonl,
    shard_texts,
    write_bytes_atomic,
)
from .frames import render_row, row_render_args, shared_boundaries_spans, target_mask
from .tokens import adds_bos, bos_id, encode_with_byte_ends

# ---------------------------------------------------------------------------
# view compilation and batches
# ---------------------------------------------------------------------------

@dataclass
class RowView:
    """One row's compiled targets (numpy). Positions are student positions
    t in [0, T-1) predicting token t+1."""
    student_ids: np.ndarray        # [T]
    compute_mask: np.ndarray       # [T-1] bool
    bnd_pos: np.ndarray            # [J] boundary positions (subset of compute positions)
    bnd_weight: np.ndarray         # [J] f32
    target_gid: np.ndarray         # [J, Kp] int32 (sentinel G at pads)
    target_log_p: np.ndarray       # [J, Kp] f32
    log_M: np.ndarray              # [J] f32
    chunk_start: np.ndarray        # [Nc] positions inclusive
    chunk_end: np.ndarray          # [Nc]
    chunk_teacher_ll: np.ndarray   # [Nc]
    chunk_teacher_log_bm: np.ndarray  # [Nc]
    stats: dict = field(default_factory=dict)


def compile_row(cache_row: dict[str, np.ndarray], text: bytes, student_ids: np.ndarray,
                student_ends: np.ndarray, tables: Tables, *, Kp: int | None,
                knobs: dict, teacher_special: set[int], student_special: set[int],
                identity: bool = False, t_spans: list | None = None,
                s_spans: list | None = None) -> RowView | None:
    """Compile one cache row against one student tokenization.

    cache_row holds the unpadded per-position teacher arrays (token_ids,
    token_end_byte, top_k_log_softmax, top_k_indices, onpath_log_p,
    onpath_mask, log_boundary_mass). Returns None for rows with fewer than
    two shared boundaries (counted by the caller)."""
    t_ends = cache_row["token_end_byte"].astype(np.int64)
    Ts = len(student_ids)
    if identity:
        # every on-path position is a boundary (all of them on an unframed
        # row, the target spans on a framed one); targets are the raw top-K
        onm = cache_row["onpath_mask"][:Ts - 1].astype(bool)
        compute_mask = onm.copy()
        pos = np.nonzero(onm)[0].astype(np.int32)
        J = len(pos)
        gid = cache_row["top_k_indices"][pos].astype(np.int32)
        lp = cache_row["top_k_log_softmax"][pos].astype(np.float32)
        gid = np.where(gid >= 0, gid, tables.G)
        lp = np.where(gid < tables.G, lp, NEG_INF).astype(np.float32)
        with np.errstate(divide="ignore"):
            log_M = np.log(np.exp(lp.astype(np.float64)).sum(axis=1)).astype(np.float32)
        return RowView(student_ids=student_ids.astype(np.int32), compute_mask=compute_mask,
                       bnd_pos=pos, bnd_weight=np.ones(J, dtype=np.float32),
                       target_gid=gid, target_log_p=lp, log_M=log_M,
                       chunk_start=np.zeros(0, np.int32), chunk_end=np.zeros(0, np.int32),
                       chunk_teacher_ll=np.zeros(0, np.float32),
                       chunk_teacher_log_bm=np.zeros(0, np.float32),
                       stats={"J": J, "own": 1.0, "redirect": 0.0, "singleton": 1.0,
                              "dropped": 0.0, "bias_ok": 1.0, "bias_cov": 1.0, "n_chunks": 0})
    if t_spans is not None:
        al = shared_boundaries_spans(t_ends, student_ends.astype(np.int64), t_spans, s_spans)
        compute_mask = target_mask(student_ends.astype(np.int64), s_spans)[:max(Ts - 1, 0)]
    else:
        al = shared_boundaries(t_ends, student_ends.astype(np.int64))
        compute_mask = np.ones(max(Ts - 1, 0), dtype=bool)
    if al.J < 2:
        return None
    proj = project_topk(cache_row["top_k_log_softmax"][al.t_pos].astype(np.float32),
                        cache_row["top_k_indices"][al.t_pos], tables, Kp)
    # boundary weights: 1 word-initial, w_mid intra-word, 0 heavy redirect
    nxt = student_ids[al.s_pos + 1]
    word_initial = tables.bmask_S[nxt]
    w = np.where(word_initial, 1.0, knobs["w_mid"]).astype(np.float32)
    w = np.where(proj["redirect"] > knobs["redirect_cut"], 0.0, w).astype(np.float32)
    # tokenization-bias diagnostic: mass projected onto the student's next-token group
    # against the teacher's on-path probability at the same boundary
    t_next = cache_row["token_ids"][al.t_pos + 1]
    in_topk = (cache_row["top_k_indices"][al.t_pos] == t_next[:, None]).any(axis=1)
    bias_ok, bias_cov = tokenization_bias_check(proj, tables.group_of[nxt].astype(np.int32),
                                                cache_row["onpath_log_p"][al.t_pos].astype(np.float32), in_topk)
    # chunks between consecutive boundaries
    cs, ce, tll, tbm = [], [], [], []
    onp = cache_row["onpath_log_p"].astype(np.float64)
    onm = cache_row["onpath_mask"]
    lbm = cache_row["log_boundary_mass"].astype(np.float32)
    log_gamma = math.log(knobs["gamma"])
    for j in range(1, al.J):
        s0, s1 = int(al.s_pos[j - 1]), int(al.s_pos[j])
        t0, t1 = int(al.t_pos[j - 1]), int(al.t_pos[j])
        ls, lt = s1 - s0, t1 - t0
        if ls > knobs["max_chunk_len"] or lt > knobs["max_chunk_len"]:
            continue
        if lbm[t1] <= log_gamma:
            continue
        if not np.all(onm[t0:t1]):
            continue
        cs.append(s0)
        ce.append(s1 - 1)
        tll.append(float(onp[t0:t1].sum()))
        tbm.append(float(lbm[t1]))
    return RowView(student_ids=student_ids.astype(np.int32), compute_mask=compute_mask,
                   bnd_pos=al.s_pos.astype(np.int32), bnd_weight=w,
                   target_gid=proj["gid"], target_log_p=proj["log_p"], log_M=proj["log_M"],
                   chunk_start=np.array(cs, dtype=np.int32), chunk_end=np.array(ce, dtype=np.int32),
                   chunk_teacher_ll=np.array(tll, dtype=np.float32),
                   chunk_teacher_log_bm=np.array(tbm, dtype=np.float32),
                   stats={"J": al.J, "own": float(proj["own"].mean()),
                          "redirect": float(proj["redirect"].mean()),
                          "singleton": float(proj["singleton"].mean()),
                          "dropped": float(proj["dropped"].mean()),
                          "M_K": float(proj["M_K"].mean()),
                          "n_groups_max": int(proj["n_groups"].max()),
                          "bias_ok": bias_ok, "bias_cov": bias_cov,
                          "n_chunks": len(cs)})


def collate(rows: list[RowView], Kp: int, G: int, pad_to: int = 32) -> dict[str, np.ndarray]:
    """Batch tensors in the format doc's table. positions lists the N_bnd
    boundary positions first, then the other compute positions."""
    B = len(rows)
    T = max(int(r.student_ids.shape[0]) for r in rows)
    T = pad_to * ((T + pad_to - 1) // pad_to)
    Nc = max([int(r.chunk_start.shape[0]) for r in rows] + [1])
    student_ids = np.zeros((B, T), dtype=np.int32)
    compute_mask = np.zeros((B, T - 1), dtype=bool)
    chunk_start = np.zeros((B, Nc), dtype=np.int32)
    chunk_end = np.zeros((B, Nc), dtype=np.int32)
    chunk_tll = np.zeros((B, Nc), dtype=np.float32)
    chunk_tbm = np.zeros((B, Nc), dtype=np.float32)
    chunk_mask = np.zeros((B, Nc), dtype=bool)
    bnd_flat, bnd_w, gids, lps, logMs = [], [], [], [], []
    for b, r in enumerate(rows):
        n = int(r.student_ids.shape[0])
        student_ids[b, :n] = r.student_ids
        compute_mask[b, :n - 1] = r.compute_mask[:n - 1]
        nc = int(r.chunk_start.shape[0])
        chunk_start[b, :nc] = r.chunk_start
        chunk_end[b, :nc] = r.chunk_end
        chunk_tll[b, :nc] = r.chunk_teacher_ll
        chunk_tbm[b, :nc] = r.chunk_teacher_log_bm
        chunk_mask[b, :nc] = True
        bnd_flat.append(b * (T - 1) + r.bnd_pos.astype(np.int64))
        bnd_w.append(r.bnd_weight)
        J, kp = r.target_gid.shape
        g = np.full((J, Kp), G, dtype=np.int32)
        lp = np.full((J, Kp), NEG_INF, dtype=np.float32)
        g[:, :min(kp, Kp)] = r.target_gid[:, :Kp]
        lp[:, :min(kp, Kp)] = r.target_log_p[:, :Kp]
        gids.append(g)
        lps.append(lp)
        logMs.append(r.log_M)
    bnd_positions = np.concatenate(bnd_flat).astype(np.int32) if bnd_flat else np.zeros(0, np.int32)
    is_bnd = np.zeros(B * (T - 1), dtype=bool)
    is_bnd[bnd_positions] = True
    other = np.nonzero(compute_mask.reshape(-1) & ~is_bnd)[0].astype(np.int32)
    positions = np.concatenate([bnd_positions, other])
    return {
        "student_ids": student_ids, "compute_mask": compute_mask,
        "positions": positions, "n_bnd": np.int32(len(bnd_positions)),
        "bnd_weight": np.concatenate(bnd_w).astype(np.float32) if bnd_w else np.zeros(0, np.float32),
        "target_gid": np.concatenate(gids) if gids else np.zeros((0, Kp), np.int32),
        "target_log_p": np.concatenate(lps) if lps else np.zeros((0, Kp), np.float32),
        "log_M": np.concatenate(logMs).astype(np.float32) if logMs else np.zeros(0, np.float32),
        "chunk_start": chunk_start, "chunk_end": chunk_end,
        "chunk_teacher_ll": chunk_tll, "chunk_teacher_log_bm": chunk_tbm,
        "chunk_mask": chunk_mask,
    }


def batch_to_mx(batch: dict[str, np.ndarray]) -> dict:
    import mlx.core as mx
    out = {}
    for k, v in batch.items():
        if k == "n_bnd":
            out[k] = int(v)
        else:
            out[k] = mx.array(v)
    return out


# ---------------------------------------------------------------------------
# data iteration
# ---------------------------------------------------------------------------

class BatchIterator:
    """mlx-lm's batching semantics with our own unconditional seeding.

    Rows are sorted by length and cut into consecutive batches; each epoch
    permutes the batch order with np.random.default_rng([seed, epoch]), so
    seed 0 is a seed and resume re-derives the permutation and skips the
    first `skip` batches without loading them."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int):
        if len(lengths) < batch_size:
            raise ValueError(f"need at least batch_size={batch_size} rows, have {len(lengths)}")
        idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [idx[i:i + batch_size] for i in range(0, len(idx) - batch_size + 1, batch_size)]
        self.seed = int(seed)

    @property
    def per_epoch(self) -> int:
        return len(self.batches)

    def order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, int(epoch)])
        return rng.permutation(len(self.batches))

    def iterate(self, skip: int = 0) -> Iterator[tuple[int, list[int]]]:
        """Yields (global iteration, row indices) forever, starting at
        iteration `skip`."""
        it = 0
        epoch = 0
        while True:
            for bi in self.order(epoch):
                if it >= skip:
                    yield it, self.batches[int(bi)]
                it += 1
            epoch += 1


class CacheReader:
    """Row access over a cache directory with a small shard LRU."""

    def __init__(self, cache_dir: Path, keep: int = 4):
        self.dir = Path(cache_dir)
        self.manifest = read_json(self.dir / "manifest.json")
        self.K = int(self.manifest["top_k"])
        self.n_shards = int(self.manifest["num_batches"])
        self.keep = keep
        self._shards: dict[int, dict] = {}
        self._order: list[int] = []
        self.index: list[tuple[int, int]] = []   # row -> (shard, slot)
        self.rows_meta: list[dict] = []
        for i in range(self.n_shards):
            rows = read_rows_jsonl(self.dir / f"rows-{i:05d}.jsonl")
            for slot, r in enumerate(rows):
                self.index.append((i, slot))
                self.rows_meta.append(r)

    def __len__(self) -> int:
        return len(self.index)

    def shard(self, i: int) -> dict:
        sh = self._shards.get(i)
        if sh is None:
            sh = load_shard(self.dir / f"batch-{i:05d}.safetensors")
            sh["_texts"] = shard_texts(sh)
            self._shards[i] = sh
            self._order.append(i)
            while len(self._order) > self.keep:
                old = self._order.pop(0)
                self._shards.pop(old, None)
        return sh

    def row(self, r: int) -> tuple[dict[str, np.ndarray], bytes, dict]:
        i, slot = self.index[r]
        sh = self.shard(i)
        n = int(sh["attention_mask"][slot].sum())
        arrs = {k: sh[k][slot, :n] for k in ("token_ids", "token_end_byte", "onpath_log_p",
                                              "onpath_mask", "tail_log_mass", "log_boundary_mass",
                                              "top_k_log_softmax", "top_k_indices")}
        if ROUTES_FIELD in sh:
            arrs[ROUTES_FIELD] = sh[ROUTES_FIELD][slot, :n]
        return arrs, sh["_texts"][slot], self.rows_meta[r]


class ViewLoader:
    """Compiles batches on the fly (or reads materialized rows) for a
    cache, a student tokenizer and a tables artifact."""

    def __init__(self, reader: CacheReader, student_tok, tables: Tables, *, knobs: dict,
                 Kp: int | None, identity: bool, student_tb=None, view_dir: Path | None = None,
                 teacher_tok=None):
        self.reader = reader
        self.student_tok = student_tok
        self.tables = tables
        self.knobs = knobs
        self.Kp = Kp
        self.identity = identity
        si = hf_inner(student_tok)
        self.student_special = set(si.all_special_ids)
        self.stb = student_tb if student_tb is not None else (None if identity else token_bytes(student_tok, tables.V_S))
        self.bos = bos_id(student_tok) if adds_bos(student_tok) else None
        self.view_dir = Path(view_dir) if view_dir else None
        self._mat: dict[int, dict] = {}
        self.dropped = 0
        self.render_failures = 0

    def student_tokens(self, arrs: dict, text: bytes, meta: dict) -> tuple[np.ndarray, np.ndarray, list | None]:
        """(ids, end bytes, target spans or None). Identity rows forward the
        cached tokens, frame included. Framed rows on the general path are
        re-rendered through the student's own template, from the student's
        own message list when the row carries one (student_messages)."""
        if self.identity:
            return arrs["token_ids"].astype(np.int32), arrs["token_end_byte"].astype(np.int64), None
        msgs = meta.get("student_messages") or meta.get("messages")
        if msgs:
            stext, spans = render_row(self.student_tok, msgs, **row_render_args(meta.get("frame")))
            ids, ends, _flag = encode_with_byte_ends(self.student_tok, stext, self.stb, add_special_tokens=False)
            return ids, ends.astype(np.int64), spans
        ids, ends, _flag = encode_with_byte_ends(self.student_tok, text, self.stb, add_special_tokens=True)
        return ids, ends.astype(np.int64), None

    def compile(self, r: int) -> RowView | None:
        if self.view_dir is not None:
            rv = self._materialized(r)
            if rv is not None:
                return rv
        arrs, text, meta = self.reader.row(r)
        try:
            s_ids, s_ends, s_spans = self.student_tokens(arrs, text, meta)
        except ValueError:
            self.dropped += 1
            self.render_failures += 1
            return None
        t_spans = [tuple(x) for x in meta["spans"]] if (s_spans is not None and meta.get("spans")) else None
        rv = compile_row(arrs, text, s_ids, s_ends, self.tables, Kp=self.Kp, knobs=self.knobs,
                         teacher_special=set(), student_special=self.student_special,
                         identity=self.identity, t_spans=t_spans, s_spans=s_spans)
        if rv is None:
            self.dropped += 1
        return rv

    def batch(self, rows: list[int]) -> dict[str, np.ndarray] | None:
        views = [v for v in (self.compile(r) for r in rows) if v is not None]
        if not views:
            return None
        assert self.Kp is not None, "ViewLoader.batch needs K' chosen"
        return collate(views, self.Kp, self.tables.G)

    # materialized views: one safetensors per cache shard, arrays keyed by slot
    def materialize(self, out_dir: Path, max_disk_gb: float | None = None) -> int:
        from safetensors.numpy import save
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        total = 0
        for i in range(self.reader.n_shards):
            p = out_dir / f"view-{i:05d}.safetensors"
            if p.exists():
                continue
            arrays: dict[str, np.ndarray] = {}
            for r, (si, slot) in enumerate(self.reader.index):
                if si != i:
                    continue
                rv = self.compile(r)
                if rv is None:
                    continue
                for name in ("student_ids", "compute_mask", "bnd_pos", "bnd_weight", "target_gid",
                             "target_log_p", "log_M", "chunk_start", "chunk_end", "chunk_teacher_ll",
                             "chunk_teacher_log_bm"):
                    arrays[f"r{slot}.{name}"] = np.ascontiguousarray(getattr(rv, name))
            data = save(arrays)
            total += len(data)
            if max_disk_gb is not None and total > max_disk_gb * GB:
                raise RuntimeError(f"--max-disk-gb {max_disk_gb} exceeded while materializing shard {i}")
            write_bytes_atomic(p, data)
            written += 1
        return written

    def _materialized(self, r: int) -> RowView | None:
        i, slot = self.reader.index[r]
        sh = self._mat.get(i)
        if sh is None:
            p = self.view_dir / f"view-{i:05d}.safetensors"
            if not p.exists():
                return None
            from safetensors.numpy import load_file
            sh = load_file(str(p))
            self._mat = {i: sh}
        key = f"r{slot}.student_ids"
        if key not in sh:
            return None
        def g(n):
            return sh[f"r{slot}.{n}"]
        ids = g("student_ids")
        cm = sh.get(f"r{slot}.compute_mask")
        if cm is None:
            cm = np.ones(max(len(ids) - 1, 0), dtype=bool)
        return RowView(student_ids=ids, compute_mask=cm.astype(bool),
                       bnd_pos=g("bnd_pos"), bnd_weight=g("bnd_weight"), target_gid=g("target_gid"),
                       target_log_p=g("target_log_p"), log_M=g("log_M"), chunk_start=g("chunk_start"),
                       chunk_end=g("chunk_end"), chunk_teacher_ll=g("chunk_teacher_ll"),
                       chunk_teacher_log_bm=g("chunk_teacher_log_bm"))


