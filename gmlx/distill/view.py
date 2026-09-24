"""The view compiler driver: one pass over a cache with the student
tokenizer that writes the tables, the alignment statistics and the row
index, and the materialized batch tensors on request. ``run_align`` is what
``gmlx distill align`` calls."""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gmlx.load.tokenizer import (
    eos_ids,
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
from .format import manifest_sha256, output_error, read_json, removes_empty_output, write_json_atomic

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


def student_identity(tokenizer) -> dict:
    """The student's fields the vocab hash leaves out and a view depends
    on: its end and start ids, whether it adds a start token, the strings
    of its special tokens, and its chat template's digest. train refuses
    a student whose values differ."""
    import hashlib
    import json
    tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tpl, dict):
        tpl = json.dumps(tpl, sort_keys=True)
    return {"eos": [int(v) for v in eos_ids(tokenizer)], "bos": _tokens.bos_id(tokenizer),
            "adds_bos": bool(_tokens.adds_bos(tokenizer)), "specials": _tokens.specials_digest(tokenizer),
            "chat_template_sha256": hashlib.sha256(tpl.encode("utf-8")).hexdigest() if isinstance(tpl, str) else None}


def get_tables(teacher_tok, student_tok, tables_dir: Path | None, out_dir: Path, *,
               V_T: int | None, V_S: int | None, save: bool = True) -> _align.Tables:
    """An existing tables artifact when its pair hashes, widths and special
    roles match, else a fresh build. With ``save`` the tables are written
    under out_dir, since ``train`` and ``eval`` read them from the view;
    ``run_align`` saves them itself once every refusal is behind it."""
    th, sh = vocab_map_hash(teacher_tok), vocab_map_hash(student_tok)
    roles = _align.special_roles(teacher_tok, student_tok)
    if tables_dir and (tables_dir / "tables.json").exists():
        try:
            version = read_json(tables_dir / "tables.json").get("tables_version")
        except (OSError, ValueError) as e:
            version = f"unreadable ({e})"
        if version != TABLES_VERSION:
            log(f"[align] tables at {tables_dir} are version {version}, this build writes {TABLES_VERSION}, "
                "rebuilding")
        else:
            try:
                t = _align.load_tables(tables_dir)
            except ValueError as e:
                log(f"[align] {e}, rebuilding")
            else:
                if t.teacher_hash == th and t.student_hash == sh and (V_T is None or t.V_T == V_T) \
                        and (V_S is None or t.V_S == V_S) and _align.same_roles(t.roles, roles):
                    log(f"[align] tables from {tables_dir} (pair hashes, widths and special roles match)")
                    if save and tables_dir.resolve() != out_dir.resolve():
                        _align.save_tables(out_dir, t)
                    return t
                elif t.teacher_hash == th and t.student_hash == sh and not _align.same_roles(t.roles, roles):
                    # the vocab hash leaves special ids out, so a base and an
                    # instruct student share it while their EOS ids differ
                    log(f"[align] tables at {tables_dir} carry other special roles (EOS or BOS ids or special "
                        "token strings differ), rebuilding")
                elif t.teacher_hash == th and t.student_hash == sh:
                    log(f"[align] tables at {tables_dir} are for other head widths ({t.V_T}, {t.V_S}), rebuilding")
                else:
                    log(f"[align] tables at {tables_dir} are for another pair, rebuilding")
    t0 = time.perf_counter()
    t = _align.build_tables(teacher_tok, student_tok, V_T=V_T, V_S=V_S)
    log(f"[align] tables built in {time.perf_counter() - t0:.1f}s: G={t.G} "
        f"N_ns={t.nonsingleton_ids.shape[0]} identity={t.identity}")
    if save:
        _align.save_tables(out_dir, t)
    return t


def same_encoding(reader: CacheReader, student_tok) -> tuple[bool, str]:
    """Whether the student's tokenizer reproduces the cached token ids of
    every unframed row from the row's bytes, its own special tokens
    added (the BOS policies were compared before). An equal vocabulary
    with other merges or another pre-tokenizer segments the same bytes
    differently, and the identity path would train the student on ids
    it never produces. The rows are read from their token ids and text
    alone, so the top-K arrays stay on disk."""
    stb = token_bytes(student_tok)
    for r, cached, text in reader.ids_and_texts():
        ids, _ends, _f = _tokens.encode_with_byte_ends(student_tok, text, stb, add_special_tokens=True)
        if len(ids) != len(cached) or not np.array_equal(ids, cached):
            return False, f"row {r}: {len(ids)} student tokens vs {len(cached)} cached"
    return True, f"{len(reader)} rows"


def same_render(reader: CacheReader, student_tok, kind: str, n_check: int | None = 8) -> tuple[bool, str]:
    """Whether the student's chat template reproduces the cached token ids
    on the framed rows: n_check rows of every conversation shape (the
    sequence of roles, and which turns carry a reasoning trace or tool
    calls), spread over the cache, or every framed row when n_check is
    None. A template can agree on plain exchanges and differ on a system
    turn, a tool call, or one row's content (a template that trims), so
    the identity path checks every row. Never when any row of the cache
    carries a student message list of its own: such a row renders for the
    student without the teacher's context, so the cached ids cannot be
    the student's input."""
    stb = token_bytes(student_tok)
    with_list = sum(1 for m in reader.rows_meta if m.get("student_messages"))
    if with_list:
        return False, f"{with_list} rows carry their own student message list"
    shapes: dict[tuple, list[int]] = {}
    for r in range(len(reader)):
        msgs = reader.rows_meta[r].get("messages")
        if msgs:
            key = tuple((m.get("role"), bool(m.get("reasoning_content")), bool(m.get("tool_calls"))) for m in msgs)
            shapes.setdefault(key, []).append(r)
    picked: list[int] = []
    for framed in shapes.values():
        if n_check is None:
            picked.extend(framed)
        else:
            # rows are length-sorted, so the sample spans short and long rows
            picked.extend(framed[k] for k in sorted(set(int(x) for x in np.linspace(0, len(framed) - 1, n_check))))
    picked.sort()
    cached = reader.token_ids(picked)
    checked = 0
    for r in picked:
        meta = reader.rows_meta[r]
        try:
            stext, _spans = _frames.render_row(student_tok, meta["messages"],
                                               **_frames.row_render_args(meta.get("frame") or kind))
        except ValueError as e:
            return False, f"row {r}: {e}"
        ids, _ends, _f = _tokens.encode_with_byte_ends(student_tok, stext, stb, add_special_tokens=False)
        if len(ids) != len(cached[r]) or not np.array_equal(ids, cached[r]):
            return False, f"row {r}: {len(ids)} student tokens vs {len(cached[r])} cached"
        checked += 1
    if len(shapes) > 1:
        return checked > 0, f"{checked} rows checked over {len(shapes)} conversation shapes"
    return checked > 0, f"{checked} rows checked"


def val_split(doc_ids: list, fraction: float, seed: int) -> set[int]:
    """Row indexes held for validation: whole documents in a seeded order
    until at least max(1, fraction * rows) rows are held, so a document's
    windows, and a conversation's turns, never sit on both sides. A
    document that would carry the held count past twice that budget is
    passed over while other documents exist; when none fits, the smallest
    document is held whole. A corpus of one document holds that document's
    last rows, so at least one row trains."""
    if not doc_ids:
        return set()
    docs: dict = {}
    for i, d in enumerate(doc_ids):
        docs.setdefault(d, []).append(i)
    names = list(docs)
    order = np.random.default_rng(seed).permutation(len(names))
    want = max(1, int(len(doc_ids) * fraction))
    val: set[int] = set()
    for j in order:
        if len(val) >= want:
            break
        rows = docs[names[int(j)]]
        if len(names) > 1 and len(val) + len(rows) > 2 * want:
            continue
        val.update(rows)
    if len(names) > 1 and (not val or len(val) >= len(doc_ids)):
        val = set(min(docs.values(), key=len))
    elif len(names) == 1:
        only = docs[names[0]]
        # a one-row cache trains its row and validates nothing
        val = set(only[-min(want, len(only) - 1):]) if len(only) > 1 else set()
    return val


def _log_first_failure(loader: ViewLoader) -> None:
    n = loader.render_failures
    if n:
        log(f"[align] {n} row{'s' if n != 1 else ''} failed to render or pair on the student side, "
            f"the first: {loader.first_failure}")


@removes_empty_output(lambda opts: opts.out)
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
        frame_kwargs = _frames.parse_render_kwargs(opts.frame_kwargs)
    except ValueError as e:
        log(f"[align] refuse: {e}")
        return 2
    err = output_error(opts.out, directory=True)
    if err:
        log(f"[align] refuse: cannot write --out {opts.out}: {err}")
        return 2
    # every input is read before an earlier view is removed, so a refusal
    # leaves the view directory as it found it
    try:
        reader = CacheReader(cache)
        manifest = reader.manifest
        V_T = int(manifest["vocab_size"])
        teacher_tok = _tokens.load_tokenizer(str(cache / "tokenizer")) if (cache / "tokenizer").exists() \
            else _tokens.load_tokenizer(manifest["teacher_path"])
    except (OSError, KeyError, TypeError, ValueError) as e:
        log(f"[align] refuse: cannot read the cache at {cache}: {e}")
        return 2
    try:
        student_tok = _tokens.load_tokenizer(opts.student)
    except (OSError, KeyError, ValueError) as e:
        log(f"[align] refuse: cannot load the student tokenizer at {opts.student}: {e}")
        return 2
    want = manifest.get("tokenizer_hash")
    have = vocab_map_hash(teacher_tok)
    if want and have != want:
        # the cached ids index the vocabulary the pass wrote with, and a
        # tokenizer with another map would project them wrong
        src = cache / "tokenizer" if (cache / "tokenizer").exists() else manifest.get("teacher_path")
        log(f"[align] refuse: the tokenizer at {src} has another vocab hash than the cache recorded "
            f"({have[:12]} vs {str(want)[:12]}), the teacher tokenizer changed since the cache was written")
        return 2
    out = Path(opts.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    V_S = student_width(opts.student) or len(hf_inner(student_tok))
    knobs = dict(DEFAULT_KNOBS, w_mid=opts.w_mid, gamma=opts.gamma, tau_alm=opts.tau_alm,
                 T_dk=opts.T_dk, max_chunk_len=opts.max_chunk_len)
    block = manifest.get("gmlx_distill", {}) or {}
    frame = block.get("frame")
    # the cache keeps the teacher's spelling of gen's thinking switch; the
    # student reads it under its own template's variable, as serve maps it
    gen_kw = _frames.generator_render_kwargs(student_tok, block.get("generator")) if frame else {}
    inherit = {**gen_kw, **((frame or {}).get("render_kwargs") or {})}
    student_kw = _frames.resolve_render_kwargs(student_tok, inherit=inherit, override=frame_kwargs)
    _frames.set_render_kwargs(student_tok, student_kw)
    if frame:
        why = _frames.template_problem(student_tok)
        if why:
            log(f"[align] refuse: the student's {why.removeprefix('the ')}")
            return 2
        log(f"[align] student chat-template kwargs {student_kw or '{}'}, turn-end markers "
            f"{_frames.assistant_tails(student_tok)!r}"
            + ("" if _frames.has_chat_template(student_tok) else " (no chat template: plain render)"))
    prefixed = any(int(r.get("prefix_n_tokens", 0) or 0) for r in reader.rows_meta)
    ident, why = _tokens.identity_pair(teacher_tok, student_tok)
    same_vocab = ident and V_S >= V_T
    identity = same_vocab
    if ident and V_S < V_T:
        log(f"[align] identity maps but student head is narrower ({V_S} < {V_T}), general path")
    t_bos, s_bos = _tokens.adds_bos(teacher_tok), _tokens.adds_bos(student_tok)
    if identity and (t_bos != s_bos or (s_bos and _tokens.bos_id(teacher_tok) != _tokens.bos_id(student_tok))):
        # the identity path forwards the cached ids as they are, so the
        # student would train without the BOS it gets at inference (or
        # with one the teacher wrote and it never adds)
        log(f"[align] BOS policies differ (teacher adds {'one' if t_bos else 'none'}, student adds "
            f"{'one' if s_bos else 'none'}), general path")
        identity = False
    if identity and frame:
        # the identity path forwards the cached render, so the student's own
        # template must produce the same tokens on every row; otherwise the
        # general path re-renders every row with the student's template
        same, why_f = same_render(reader, student_tok, frame["kind"], n_check=None)
        log(f"[align] framed cache ({frame['kind']}): student render {'matches' if same else 'differs'} ({why_f})")
        identity = same
    elif identity and prefixed:
        log("[align] prefixed rows without a frame block, general path")
        identity = False
    elif identity:
        # an unframed cache holds the teacher's own encoding of every row
        # and the identity path forwards it, so the student's tokenizer
        # must produce it: the vocab hash covers the map, not the merges
        # or the pre-tokenizer
        same, why_e = same_encoding(reader, student_tok)
        log(f"[align] unframed cache: student encoding {'matches' if same else 'differs'} ({why_e})")
        identity = same
    log(f"[align] path={'identity' if identity else 'general'} ({why}), V_T={V_T} V_S={V_S}")
    # an identical vocabulary keeps the identity tables even on the general
    # path (group_of is the identity, so the projection is exact)
    if same_vocab:
        # at the student's width: the head runs over V_S columns, and a wider
        # student head keeps the teacher's ids as a prefix
        # with the token lengths, so a boundary where only one side spells a
        # dummy-prefix space still gets no weight
        tables = _align.identity_tables(V_S, whitespace_start_mask(student_tok, V_S), vocab_map_hash(teacher_tok),
                                        vocab_map_hash(student_tok), V_T=V_T,
                                        t_len=_align.token_lengths(token_bytes(teacher_tok), V_T),
                                        s_len=_align.token_lengths(token_bytes(student_tok), V_S))
    else:
        tables = get_tables(teacher_tok, student_tok, Path(opts.tables) if opts.tables else None, out,
                            V_T=V_T, V_S=V_S, save=False)
    loader = ViewLoader(reader, student_tok, tables, knobs=knobs, Kp=opts.kprime, identity=identity)
    n = len(reader)
    stats: dict[str, list] = {"J": [], "own": [], "redirect": [], "singleton": [], "dropped": [], "capped": [],
                              "M_K": [],
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
    if not index:
        # an empty view would pass the own-group gate (no boundary to
        # average) and replace an earlier view with nothing for train
        log(f"[align] refuse: no row compiled ({n} rows, {loader.dropped} dropped, {loader.render_failures} "
            "student render failures), no view written")
        _log_first_failure(loader)
        return 2
    Kp = opts.kprime or (max(stats["n_groups_max"]) if stats["n_groups_max"] else reader.K)
    if identity:
        if opts.kprime and opts.kprime != reader.K:
            log(f"[align] --kprime {opts.kprime} ignored on the identity path, K' = K = {reader.K}")
        Kp = reader.K
    a = float(np.mean(stats["own"])) if stats["own"] else 1.0
    s = float(np.mean(stats["singleton"])) if stats["singleton"] else 1.0
    red = float(np.mean(stats["redirect"])) if stats["redirect"] else 0.0
    doc_ids = [str(reader.rows_meta[e["row"]].get("doc_id", e["row"])) for e in index]
    val = val_split(doc_ids, opts.val_fraction, opts.seed)
    for i, e in enumerate(index):
        e["split"] = "val" if i in val else "train"
    n_docs = len(set(doc_ids))
    log(f"[align] validation: {len(val)} of {len(index)} rows from {len({doc_ids[i] for i in val})} of "
        f"{n_docs} documents" + (", the only document is split" if n_docs == 1 and 0 < len(val) < len(index) else ""))
    retained = [1.0 - d - c for d, c in zip(stats["dropped"], stats["capped"])]
    jw = np.array(stats["J"], dtype=np.float64) * np.array(stats["bias_cov"], dtype=np.float64) \
        if stats["bias_ok"] else np.zeros(0)
    view = {
        "cache_manifest_sha256": manifest_sha256(cache), "cache_dir": str(cache.resolve()),
        "teacher_hash": tables.teacher_hash, "student_hash": tables.student_hash,
        "V_T": V_T, "V_S": V_S, "tables_version": TABLES_VERSION, "identity": identity,
        "K": reader.K, "Kp": int(Kp), "knobs": knobs, "student": opts.student,
        "student_identity": student_identity(student_tok),
        "student_render_kwargs": student_kw, "student_render_failures": loader.render_failures,
        "alignment": {
            "rows": n, "rows_kept": len(index), "rows_dropped_lt2": loader.dropped,
            "a_own_mass_fraction": {"mean": a, "p05": float(np.percentile(stats["own"], 5)) if stats["own"] else 1.0},
            "redirected_mass_fraction": red,
            "singleton_mass_fraction": s,
            "dropped_mass_mean": float(np.mean(stats["dropped"])) if stats["dropped"] else 0.0,
            "capped_mass_mean": float(np.mean(stats["capped"])) if stats["capped"] else 0.0,
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
    _log_first_failure(loader)
    if not identity and a < REFUSE_A and not opts.force:
        # refused before any write, so an earlier view in the directory
        # stays as it was and no new one is left for train to accept
        log(f"[align] refuse: own-group fraction a={a:.3f} < {REFUSE_A}, the tokenizers diverge too far. "
            f"Pick a student from the teacher's family, or pass --force and expect a weaker result")
        return 3
    staged = None
    if opts.materialize:
        # shards go to a staging directory first, so a materialization
        # that fails leaves an earlier view untouched
        loader2 = ViewLoader(reader, student_tok, tables, knobs=knobs, Kp=int(Kp), identity=identity)
        t1 = time.perf_counter()
        staging = out / "materialize.tmp"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            nsh = loader2.materialize(staging, opts.max_disk_gb)
        except (RuntimeError, OSError) as e:
            shutil.rmtree(staging, ignore_errors=True)
            log(f"[align] refuse: {e}, the partial view was removed")
            return 2
        staged = (staging, nsh, time.perf_counter() - t1)
    # every refusal is behind, and the tables are staged first, so the
    # earlier view goes only once every new file is on disk; a view
    # directory is then rewritten whole, since shards of an earlier align
    # over another student or other knobs would otherwise be reused by
    # train
    tables_tmp = out / "tables.tmp"
    shutil.rmtree(tables_tmp, ignore_errors=True)
    try:
        _align.save_tables(tables_tmp, tables)
    except OSError as e:
        shutil.rmtree(tables_tmp, ignore_errors=True)
        if staged is not None:
            shutil.rmtree(staged[0], ignore_errors=True)
        log(f"[align] refuse: cannot write the tables under {out}: {e}")
        return 2
    stale = sorted(out.glob("view-*.safetensors")) + [p for p in (out / "view.json",) if p.exists()]
    for p in stale:
        p.unlink()
    if stale:
        log(f"[align] removed {len(stale)} files of an earlier view in {out}")
    if staged is None:
        # a staging directory a killed --materialize left behind
        shutil.rmtree(out / "materialize.tmp", ignore_errors=True)
    for name in ("tables.safetensors", "tables.json"):
        # arrays first: a kill between the two leaves the earlier
        # tables.json, which names other arrays and refuses these
        os.replace(tables_tmp / name, out / name)
    shutil.rmtree(tables_tmp, ignore_errors=True)
    if staged is not None:
        staging, nsh, wall_m = staged
        for p in sorted(staging.glob("view-*.safetensors")):
            os.replace(p, out / p.name)
        shutil.rmtree(staging, ignore_errors=True)
        log(f"[align] materialized {nsh} view shards in {wall_m:.1f}s")
    write_json_atomic(out / "view.json", view)
    if not identity and (a < WARN_A or s < WARN_S):
        log(f"[align] warn: own-group fraction a={a:.3f} (< {WARN_A}) or singleton fraction s={s:.3f} (< {WARN_S})")
    return 0
