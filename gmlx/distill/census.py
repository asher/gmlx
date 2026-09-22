"""Context-delta census: how much a context the student never sees moves
the teacher's reply distribution, measured between two or more caches
over the same reply rows.

Every cache is cut with ``distill cache --frame reply`` (or reply-think)
from the same prompts, once without the context and once per context.
Rows pair by corpus line and window (``pair_by`` line, the default, since
the corpora hold the same prompts in the same order under different file
names) or by the full doc_id, and the reply bytes must agree. Positions
inside the reply line up by the byte offset of the predicted token from
the reply's content start. Per position the census records the on-path
delta (log p with the context minus without), the coarsened KL between
the two stored top-K distributions over the ids both hold plus one
bucket for everything else, and whether the top-1 moved. With several
contexts the residual is the mean KL of each context's distribution
against their mixture at the same position, the part no training
recovers. The JSON carries per-row means, a delta histogram, the
teacher's on-path nats over the high-delta positions with and without
the context, and ``high_delta``, byte ranges per row id above the
threshold, which ``distill eval --reply-positions`` reads."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import log
from .data import CacheReader
from .format import write_json_atomic

FLOOR = 1e-12
HIST_BINS = [-math.inf, -1, -0.1, 0.1, 0.5, 1, 2, 4, math.inf]


@dataclass
class CensusOptions:
    without: str
    with_: list[str]
    out: str
    md: str | None = None
    corpus: str | None = None
    delta_threshold: float = 1.0
    pair_by: str = "line"
    max_rows: int | None = None


def row_key(doc_id: str, pair_by: str) -> str:
    """The pairing key of a row: its corpus line for ``line``, else the
    whole doc_id."""
    return doc_id.rsplit(":", 1)[-1] if pair_by == "line" else doc_id


def reply_rows(cache: Path, pair_by: str) -> tuple[CacheReader, dict]:
    """(reader, {(key, window): row index}) over the cache's reply rows."""
    reader = CacheReader(cache, keep=2)
    out = {}
    for r, meta in enumerate(reader.rows_meta):
        if meta.get("frame") in ("reply", "reply-think") and meta.get("spans"):
            out[(row_key(str(meta["doc_id"]), pair_by), int(meta.get("window", 0)))] = r
    return reader, out


def reply_positions(arrs: dict, meta: dict) -> tuple[dict[int, int], int, int]:
    """{start byte of the predicted token relative to the reply's content
    start: position} over on-path positions, plus the reply's content
    start and target end."""
    b0, _b1, b2 = (int(x) for x in meta["spans"][-1])
    ends = arrs["token_end_byte"].astype(np.int64)
    om = arrs["onpath_mask"].astype(bool)
    keys = {}
    for t in range(len(ends) - 1):
        if om[t] and b0 <= ends[t] < b2:
            keys[int(ends[t] - b0)] = t
    return keys, b0, b2


def sparse(arrs: dict, t: int) -> tuple[np.ndarray, np.ndarray]:
    """The stored top-K ids and log-probs at position t, pads dropped."""
    ids = arrs["top_k_indices"][t].astype(np.int64)
    lp = arrs["top_k_log_softmax"][t].astype(np.float64)
    keep = ids >= 0
    return ids[keep], lp[keep]


def coarsened_kl(p_ids: np.ndarray, p_lp: np.ndarray, q_ids: np.ndarray, q_lp: np.ndarray) -> float:
    """KL(p || q) over the partition {ids both hold} + {everything else}."""
    _shared, pi, qi = np.intersect1d(p_ids, q_ids, assume_unique=True, return_indices=True)
    p = np.exp(p_lp[pi])
    kl = float(np.sum(p * (p_lp[pi] - q_lp[qi])))
    p_rest = max(1.0 - float(p.sum()), FLOOR)
    q_rest = max(1.0 - float(np.exp(q_lp[qi]).sum()), FLOOR)
    return kl + p_rest * (math.log(p_rest) - math.log(q_rest))


def residual_kl(dists: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mean over contexts of KL(p_d || mixture) on the union support plus
    a rest bucket. A context's mass outside its top-K counts as rest."""
    union = np.unique(np.concatenate([ids for ids, _ in dists]))
    P = np.zeros((len(dists), len(union) + 1))
    for d, (ids, lp) in enumerate(dists):
        P[d, np.searchsorted(union, ids)] = np.exp(lp)
        P[d, -1] = max(1.0 - P[d, :-1].sum(), FLOOR)
    m = np.maximum(P.mean(axis=0), FLOOR)
    Pc = np.maximum(P, FLOOR)
    return float(np.mean(np.sum(np.where(P > 0, P * (np.log(Pc) - np.log(m)), 0.0), axis=1)))


def corpus_ids(corpus: Path, pair_by: str) -> dict[str, str]:
    """{pairing key: the corpus row's id} so high_delta is keyed the way
    the eval's reply slice names its rows."""
    out = {}
    for i, line in enumerate(corpus.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            out[row_key(f"{corpus.name}:{i}", pair_by)] = str(json.loads(line).get("id", i))
    return out


def census(base: tuple[CacheReader, dict], ctx: list[tuple[CacheReader, dict]], *,
           delta_threshold: float, id_of: dict[str, str], max_rows: int | None = None) -> dict:
    """The summary dict over the rows every cache holds."""
    base_reader, base_rows = base
    common = set(base_rows)
    for _r, rows in ctx:
        common &= set(rows)
    keys = sorted(common)
    if max_rows:
        keys = keys[:max_rows]
    per_row = []
    deltas_all: list[float] = []
    kl_all: list[float] = []
    res_all: list[float] = []
    top1_all: list[bool] = []
    high: dict[str, list] = {}
    hd = {"without": 0.0, "with": 0.0, "bytes": 0, "positions": 0}
    mismatch = 0
    for key, window in keys:
        arrs0, text0, meta0 = base_reader.row(base_rows[(key, window)])
        doc_id = str(meta0["doc_id"])
        keys0, b0, _b2 = reply_positions(arrs0, meta0)
        reply0 = text0[b0:int(meta0["spans"][-1][1])]
        sides = []
        ok = True
        for reader, rows in ctx:
            arrs1, text1, meta1 = reader.row(rows[(key, window)])
            k1, c0, _c2 = reply_positions(arrs1, meta1)
            if text1[c0:int(meta1["spans"][-1][1])] != reply0:
                ok = False
                break
            sides.append((arrs1, k1))
        if not ok:
            mismatch += 1
            continue
        shared = set(keys0)
        for _a, k1 in sides:
            shared &= set(k1)
        rel = sorted(shared)
        if not rel:
            continue
        ends0 = arrs0["token_end_byte"].astype(np.int64)
        row_delta, row_kl, row_res, row_top1 = [], [], [], []
        ranges = []
        for r in rel:
            t0 = keys0[r]
            p0_ids, p0_lp = sparse(arrs0, t0)
            on0 = float(arrs0["onpath_log_p"][t0])
            arrs1, k1 = sides[0]
            t1 = k1[r]
            p1_ids, p1_lp = sparse(arrs1, t1)
            on1 = float(arrs1["onpath_log_p"][t1])
            delta = on1 - on0
            row_delta.append(delta)
            row_kl.append(coarsened_kl(p1_ids, p1_lp, p0_ids, p0_lp))
            row_top1.append(bool(len(p1_ids) and len(p0_ids) and int(p1_ids[0]) != int(p0_ids[0])))
            if len(sides) > 1:
                dists = [(p1_ids, p1_lp)] + [sparse(a, k[r]) for a, k in sides[1:]]
                row_res.append(residual_kl(dists))
            if delta > delta_threshold:
                nbytes = int(ends0[t0 + 1] - ends0[t0])
                ranges.append([r, r + nbytes])
                hd["without"] += -on0
                hd["with"] += -on1
                hd["bytes"] += nbytes
                hd["positions"] += 1
        rid = id_of.get(key, doc_id)
        if ranges:
            high[rid] = ranges
        per_row.append({"doc_id": doc_id, "window": window, "id": rid, "positions": len(rel),
                        "mean_delta": float(np.mean(row_delta)), "sum_delta": float(np.sum(row_delta)),
                        "mean_kl": float(np.mean(row_kl)), "top1_moved": float(np.mean(row_top1)),
                        "mean_residual": float(np.mean(row_res)) if row_res else None,
                        "high_delta_positions": len(ranges)})
        deltas_all += row_delta
        kl_all += row_kl
        top1_all += row_top1
        res_all += row_res
    deltas = np.array(deltas_all)
    hist = np.histogram(deltas, bins=HIST_BINS)[0].tolist() if deltas.size else []
    return {
        "rows": len(per_row), "rows_mismatched": mismatch, "positions": int(deltas.size),
        "delta_threshold": delta_threshold,
        "distillable_effect_kl_nats": float(np.mean(kl_all)) if kl_all else None,
        "mean_onpath_delta_nats": float(deltas.mean()) if deltas.size else None,
        "high_delta_fraction": float((deltas > delta_threshold).mean()) if deltas.size else None,
        "top1_moved_fraction": float(np.mean(top1_all)) if top1_all else None,
        "residual_kl_nats": float(np.mean(res_all)) if res_all else None,
        "delta_histogram": {"bins": [str(b) for b in HIST_BINS], "counts": hist},
        "teacher_high_delta": {"nll_nats_without": hd["without"], "nll_nats_with": hd["with"],
                               "bytes": hd["bytes"], "positions": hd["positions"]},
        "per_row": per_row, "high_delta": high}


def report_markdown(opts: CensusOptions, s: dict) -> str:
    hd = s["teacher_high_delta"]
    hist = s["delta_histogram"]["counts"]
    lines = ["# Context-delta census", "",
             f"Caches: without `{opts.without}`, with {', '.join(f'`{c}`' for c in opts.with_)}.", "",
             "| measure | value |", "|---|---|",
             f"| paired reply rows | {s['rows']} ({s['rows_mismatched']} reply mismatches skipped) |",
             f"| reply positions compared | {s['positions']} |",
             f"| distillable effect, mean coarsened KL (nats) | {s['distillable_effect_kl_nats']} |",
             f"| mean on-path delta (nats) | {s['mean_onpath_delta_nats']} |",
             f"| positions above {opts.delta_threshold} nats | {s['high_delta_fraction']} |",
             f"| top-1 moved | {s['top1_moved_fraction']} |",
             f"| residual across contexts (nats) | {s['residual_kl_nats']} |",
             f"| teacher nats on high-delta positions, without / with | {hd['nll_nats_without']:.1f} / "
             f"{hd['nll_nats_with']:.1f} over {hd['bytes']} bytes |", ""]
    if hist:
        lines.append("Delta histogram (nats): " + ", ".join(
            f"{HIST_BINS[i]}..{HIST_BINS[i + 1]}: {c}" for i, c in enumerate(hist)))
        lines.append("")
    return "\n".join(lines) + "\n"


def run_census(opts: CensusOptions) -> int:
    """Pair the caches, measure, and write the JSON (and Markdown).
    Returns 0, or 2 when a cache directory has no manifest or no rows
    pair across the caches."""
    caches = [Path(opts.without).expanduser()] + [Path(c).expanduser() for c in opts.with_]
    for c in caches:
        if not (c / "manifest.json").is_file():
            print(f"[census] refuse: no manifest in {c}", file=sys.stderr)
            return 2
    base = reply_rows(caches[0], opts.pair_by)
    ctx = [reply_rows(c, opts.pair_by) for c in caches[1:]]
    id_of = corpus_ids(Path(opts.corpus).expanduser(), opts.pair_by) if opts.corpus else {}
    log(f"[census] {len(base[1])} reply rows without, {[len(r) for _x, r in ctx]} with, "
        f"paired by {opts.pair_by}")
    s = census(base, ctx, delta_threshold=opts.delta_threshold, id_of=id_of, max_rows=opts.max_rows)
    if s["rows"] == 0:
        print("[census] refuse: no reply rows pair across the caches (same prompts, --frame reply, "
              "matching reply bytes)", file=sys.stderr)
        return 2
    summary = {"caches": {"without": str(caches[0]), "with": [str(c) for c in caches[1:]]},
               "pair_by": opts.pair_by, "corpus": opts.corpus, **s}
    out = Path(opts.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, summary)
    if opts.md:
        md = Path(opts.md).expanduser()
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(report_markdown(opts, s), encoding="utf-8")
    log(f"[census] effect {s['distillable_effect_kl_nats']} nats, delta {s['mean_onpath_delta_nats']}, "
        f"high-delta fraction {s['high_delta_fraction']}, residual {s['residual_kl_nats']}; wrote {out}")
    return 0
