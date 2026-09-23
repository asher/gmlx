"""The view compiler driver: one pass over a cache with the student
tokenizer that writes the tables, the alignment statistics and the row
index, and the materialized batch tensors on request. ``run_align`` is what
``gmlx distill align`` calls."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gmlx.load.tokenizer import (
    hf_inner,
    token_bytes,
    vocab_map_hash,
    whitespace_start_mask,
)

from . import align as _align
from . import frames as _frames
from . import tokens as _tokens
from .constants import DEFAULT_KNOBS, TABLES_VERSION, log
from .data import CacheReader, ViewLoader
from .format import manifest_sha256, write_json_atomic

# Projection gates on the own-group mass fraction a and the singleton mass
# fraction s, mass-weighted over the top-K at shared boundaries.
WARN_A, REFUSE_A, WARN_S = 0.90, 0.70, 0.50


@dataclass
class AlignOptions:
    cache: str
    student: str
    out: str
    tables: str | None = None
    kprime: int | None = None
    materialize: bool = False
    max_disk_gb: float | None = None
    force: bool = False
    val_fraction: float = 0.02
    seed: int = 1
    w_mid: float = DEFAULT_KNOBS["w_mid"]
    gamma: float = DEFAULT_KNOBS["gamma"]
    tau_alm: float = DEFAULT_KNOBS["tau_alm"]
    T_dk: float = DEFAULT_KNOBS["T_dk"]
    max_chunk_len: int = DEFAULT_KNOBS["max_chunk_len"]
    frame_kwargs: str | None = None


def student_width(path: str) -> int | None:
    """The logits width from a GGUF header or an MLX checkpoint's
    safetensors header, no model load; None when it cannot be read."""
    p = Path(path)
    try:
        if p.is_file() and p.suffix == ".gguf":
            return _tokens.logits_width_from_gguf(str(p))
        if p.is_dir():
            ggufs = sorted(p.glob("*.gguf"))
            if ggufs and not (p / "tokenizer.json").exists():
                return _tokens.logits_width_from_gguf(str(ggufs[0]))
            return _tokens.logits_width_from_mlx_dir(str(p))
    except (OSError, ValueError, KeyError) as e:
        log(f"[align] warn: student width not read: {e}")
    return None


def get_tables(teacher_tok, student_tok, tables_dir: Path | None, out_dir: Path, *,
               V_T: int | None, V_S: int | None) -> _align.Tables:
    """An existing tables artifact when its pair hashes match, else a fresh
    build. Either way the tables are saved under out_dir, since ``train``
    and ``eval`` read them from the view."""
    th, sh = vocab_map_hash(teacher_tok), vocab_map_hash(student_tok)
    if tables_dir and (tables_dir / "tables.json").exists():
        t = _align.load_tables(tables_dir)
        if t.teacher_hash == th and t.student_hash == sh:
            log(f"[align] tables from {tables_dir} (pair hashes match)")
            if tables_dir.resolve() != out_dir.resolve():
                _align.save_tables(out_dir, t)
            return t
        log(f"[align] tables at {tables_dir} are for another pair, rebuilding")
    t0 = time.perf_counter()
    t = _align.build_tables(teacher_tok, student_tok, V_T=V_T, V_S=V_S)
    log(f"[align] tables built in {time.perf_counter() - t0:.1f}s: G={t.G} "
        f"N_ns={t.nonsingleton_ids.shape[0]} identity={t.identity}")
    _align.save_tables(out_dir, t)
    return t


def same_render(reader: CacheReader, student_tok, kind: str, n_check: int = 8) -> tuple[bool, str]:
    """Whether the student's chat template reproduces the cached token ids
    on the first n_check framed rows."""
    stb = token_bytes(student_tok)
    checked = 0
    for r in range(min(len(reader), 4096)):
        meta = reader.rows_meta[r]
        if not meta.get("messages"):
            continue
        if meta.get("student_messages"):
            return False, f"row {r}: the student has its own message list"
        arrs, _text, _ = reader.row(r)
        try:
            stext, _spans = _frames.render_row(student_tok, meta["messages"],
                                               **_frames.row_render_args(meta.get("frame") or kind))
        except ValueError as e:
            return False, f"row {r}: {e}"
        ids, _ends, _f = _tokens.encode_with_byte_ends(student_tok, stext, stb, add_special_tokens=False)
        if len(ids) != len(arrs["token_ids"]) or not np.array_equal(ids, arrs["token_ids"]):
            return False, f"row {r}: {len(ids)} student tokens vs {len(arrs['token_ids'])} cached"
        checked += 1
        if checked >= n_check:
            break
    return checked > 0, f"{checked} rows checked"


def run_align(opts: AlignOptions) -> int:
    """Returns 0 on a written view, 2 when the cache is missing, 3 when the
    projection gate refuses the pair (``force`` keeps the view)."""
    cache = Path(opts.cache)
    if not (cache / "manifest.json").is_file():
        log(f"[align] refuse: no manifest.json in {cache}, run gmlx distill cache first")
        return 2
    if not Path(opts.student).expanduser().exists():
        log(f"[align] refuse: no student at {opts.student}")
        return 2
    if opts.tables and not (Path(opts.tables) / "tables.json").is_file():
        log(f"[align] refuse: no tables.json in {opts.tables}")
        return 2
    try:
        _frames.parse_render_kwargs(opts.frame_kwargs)
    except ValueError as e:
        log(f"[align] refuse: --frame-kwargs is not a JSON object: {e}")
        return 2
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = CacheReader(cache)
    manifest = reader.manifest
    teacher_tok = _tokens.load_tokenizer(str(cache / "tokenizer")) if (cache / "tokenizer").exists() \
        else _tokens.load_tokenizer(manifest["teacher_path"])
    student_tok = _tokens.load_tokenizer(opts.student)
    V_T = int(manifest["vocab_size"])
    V_S = student_width(opts.student) or len(hf_inner(student_tok))
    knobs = dict(DEFAULT_KNOBS, w_mid=opts.w_mid, gamma=opts.gamma, tau_alm=opts.tau_alm,
                 T_dk=opts.T_dk, max_chunk_len=opts.max_chunk_len)
    frame = (manifest.get("gmlx_distill", {}) or {}).get("frame")
    student_kw = _frames.resolve_render_kwargs(student_tok, inherit=(frame or {}).get("render_kwargs"),
                                               override=_frames.parse_render_kwargs(opts.frame_kwargs))
    _frames.set_render_kwargs(student_tok, student_kw)
    if frame:
        log(f"[align] student chat-template kwargs {student_kw or '{}'}, turn-end markers "
            f"{_frames.assistant_tails(student_tok)!r}"
            + ("" if _frames.has_chat_template(student_tok) else " (no chat template: plain render)"))
    prefixed = any(int(r.get("prefix_n_tokens", 0) or 0) for r in reader.rows_meta)
    ident, why = _tokens.identity_pair(teacher_tok, student_tok)
    same_vocab = ident and V_S >= V_T
    identity = same_vocab
    if ident and V_S < V_T:
        log(f"[align] identity maps but student head is narrower ({V_S} < {V_T}), general path")
    if identity and frame:
        # the identity path forwards the cached render, so the student's own
        # template must produce the same tokens; otherwise the general path
        # re-renders every row with the student's template
        same, why_f = same_render(reader, student_tok, frame["kind"])
        log(f"[align] framed cache ({frame['kind']}): student render {'matches' if same else 'differs'} ({why_f})")
        identity = same
    elif identity and prefixed:
        log("[align] prefixed rows without a frame block, general path")
        identity = False
    log(f"[align] path={'identity' if identity else 'general'} ({why}), V_T={V_T} V_S={V_S}")
    # an identical vocabulary keeps the identity tables even on the general
    # path (group_of is the identity, so the projection is exact)
    if same_vocab:
        tables = _align.identity_tables(V_T, whitespace_start_mask(student_tok, V_T), vocab_map_hash(teacher_tok),
                                        vocab_map_hash(student_tok))
        _align.save_tables(out, tables)
    else:
        tables = get_tables(teacher_tok, student_tok, Path(opts.tables) if opts.tables else None, out,
                            V_T=V_T, V_S=V_S)
    loader = ViewLoader(reader, student_tok, tables, knobs=knobs, Kp=opts.kprime, identity=identity)
    n = len(reader)
    stats: dict[str, list] = {"J": [], "own": [], "redirect": [], "singleton": [], "dropped": [], "M_K": [],
                              "n_groups_max": [], "bias_ok": [], "bias_cov": [], "n_chunks": [], "n_student": []}
    index = []
    chunk_hist: dict[int, int] = {}
    t0 = time.perf_counter()
    for r in range(n):
        rv = loader.compile(r)
        if rv is None:
            continue
        for k in stats:
            if k == "n_student":
                stats[k].append(int(rv.student_ids.shape[0]))
            elif k in rv.stats:
                stats[k].append(rv.stats[k])
        for cs, ce in zip(rv.chunk_start, rv.chunk_end):
            L = int(ce - cs + 1)
            chunk_hist[L] = chunk_hist.get(L, 0) + 1
        si, slot = reader.index[r]
        index.append({"row": r, "shard": si, "slot": slot, "source": reader.rows_meta[r].get("source", "human"),
                      "n_student_tokens": int(rv.student_ids.shape[0]), "n_boundaries": int(rv.bnd_pos.shape[0]),
                      "n_chunks": int(rv.chunk_start.shape[0])})
        if (r + 1) % 1000 == 0:
            log(f"[align] {r + 1}/{n} rows ({(time.perf_counter() - t0) / (r + 1) * 1000:.1f} ms/row)")
    wall = time.perf_counter() - t0
    Kp = opts.kprime or (max(stats["n_groups_max"]) if stats["n_groups_max"] else reader.K)
    if identity:
        Kp = reader.K
    a = float(np.mean(stats["own"])) if stats["own"] else 1.0
    s = float(np.mean(stats["singleton"])) if stats["singleton"] else 1.0
    red = float(np.mean(stats["redirect"])) if stats["redirect"] else 0.0
    rng = np.random.default_rng(opts.seed)
    val = set(rng.choice(len(index), size=max(1, int(len(index) * opts.val_fraction)),
                         replace=False).tolist()) if index else set()
    for i, e in enumerate(index):
        e["split"] = "val" if i in val else "train"
    retained = [1.0 - d for d in stats["dropped"]]
    jw = np.array(stats["J"], dtype=np.float64) * np.array(stats["bias_cov"], dtype=np.float64) \
        if stats["bias_ok"] else np.zeros(0)
    view = {
        "cache_manifest_sha256": manifest_sha256(cache), "cache_dir": str(cache),
        "teacher_hash": tables.teacher_hash, "student_hash": tables.student_hash,
        "V_T": V_T, "V_S": V_S, "tables_version": TABLES_VERSION, "identity": identity,
        "K": reader.K, "Kp": int(Kp), "knobs": knobs, "student": opts.student,
        "student_render_kwargs": student_kw, "student_render_failures": loader.render_failures,
        "alignment": {
            "rows": n, "rows_kept": len(index), "rows_dropped_lt2": loader.dropped,
            "a_own_mass_fraction": {"mean": a, "p05": float(np.percentile(stats["own"], 5)) if stats["own"] else 1.0},
            "redirected_mass_fraction": red,
            "singleton_mass_fraction": s,
            "dropped_mass_mean": float(np.mean(stats["dropped"])) if stats["dropped"] else 0.0,
            "retained_fraction_r": {"mean": float(np.mean(retained)) if retained else 1.0,
                                    "p05": float(np.percentile(retained, 5)) if retained else 1.0},
            "captured_mass_M_K_mean": float(np.mean(stats["M_K"])) if stats["M_K"] else None,
            "shared_boundary_fraction": float(np.sum(stats["J"]) / max(np.sum([x - 1 for x in stats["n_student"]]), 1))
            if stats["J"] else 1.0,
            "chunks_per_row_mean": float(np.mean(stats["n_chunks"])) if stats["n_chunks"] else 0.0,
            "chunk_length_histogram": {str(k): v for k, v in sorted(chunk_hist.items())},
            "n_groups_max": int(max(stats["n_groups_max"])) if stats["n_groups_max"] else reader.K,
            "tokenization_bias_ok": float(np.average(stats["bias_ok"], weights=jw)) if jw.sum() > 0 else 1.0,
            "onpath_in_topk_fraction": float(np.average(stats["bias_cov"], weights=stats["J"])) if stats["bias_cov"] else 1.0,
            "loader_ms_per_row": wall / max(n, 1) * 1000,
        },
        "index": index,
    }
    log(f"[align] {len(index)} rows kept, {loader.dropped} dropped, kprime={Kp}, a={a:.3f} s={s:.3f} "
        f"redirect={red:.3f} bias_ok={view['alignment']['tokenization_bias_ok']:.4f} "
        f"(on-path in top-K {view['alignment']['onpath_in_topk_fraction']:.3f}); "
        f"{wall / max(n, 1) * 1000:.2f} ms/row")
    if not identity and a < REFUSE_A and not opts.force:
        # refused before the write, so no view is left for train to accept
        log(f"[align] refuse: own-group fraction a={a:.3f} < {REFUSE_A}, the tokenizers diverge too far. "
            f"Pick a student from the teacher's family, or pass --force and expect a weaker result")
        return 3
    write_json_atomic(out / "view.json", view)
    if not identity and (a < WARN_A or s < WARN_S):
        log(f"[align] warn: own-group fraction a={a:.3f} (< {WARN_A}) or singleton fraction s={s:.3f} (< {WARN_S})")
    if opts.materialize:
        loader2 = ViewLoader(reader, student_tok, tables, knobs=knobs, Kp=int(Kp), identity=identity)
        t1 = time.perf_counter()
        nsh = loader2.materialize(out, opts.max_disk_gb)
        log(f"[align] materialized {nsh} view shards in {time.perf_counter() - t1:.1f}s")
    return 0
