"""The teacher pass driver: rows from a corpus, the teacher load, the trunk
and head loop with the memory probe, shard writing and the manifest.
``run_cache`` is what ``gmlx distill cache`` calls."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gmlx.load.tokenizer import (
    token_bytes,
    vocab_map_hash,
    whitespace_start_mask,
)

from . import cache as _cache
from . import corpus as _corpus
from . import format as _format
from . import frames as _frames
from . import hidden as _hidden
from . import student as _student
from . import tokens as _tokens
from .constants import GB, log
from .head import HEAD_PARITY_TOL, HeadSpec, head_parity_gap, head_spec_from_model, log_bmask_from

CONTINUE_INSTRUCTION = _frames.CONTINUE_INSTRUCTION
FRAME_CHOICES = ("none", "continue", "chat", "reply", "reply-think")


@dataclass
class CacheOptions:
    """Everything the teacher pass reads. Sizes in decimal GB."""
    teacher: str
    corpus: str
    out: str
    top_k: int = 256
    max_len: int = 2048
    max_disk_gb: float | None = None
    cache_limit_gb: float = 8.0
    logits_cap_gb: float = 4.0
    floor: bool = False
    rows_per_shard: int = 64
    trunk: int | None = None
    resume: bool = False
    max_rows: int | None = None
    max_tokens: int | None = None
    limit_docs: int | None = None
    text_key: str = "text"
    hf_split: str = "train"
    source: str | None = None
    frame: str = "none"
    per_turn: bool = False
    student_messages_key: str = "student_messages"
    frame_instruction: str = CONTINUE_INSTRUCTION
    messages_key: str = "messages"
    close_final_windows: bool = False
    frame_kwargs: str | None = None
    hf_source: str | None = None
    require_feeder: bool = True
    no_wired_limit: bool = False
    stream_experts: bool = False
    expert_bytes_gb: float | None = None
    routes: bool = False
    hidden: bool = False
    hidden_dim: int = 256
    hidden_seed: int = 1
    extra: dict = field(default_factory=dict)


def _hash_doc(h, doc_id: str, payload: bytes) -> None:
    """One document into the corpus hash: its id and its bytes, each
    length-prefixed, so a corpus re-split at other document boundaries or
    renumbered by an inserted line hashes apart and a resume over it is
    refused (rows are keyed by doc id downstream)."""
    for part in (doc_id.encode("utf-8"), payload):
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)


def build_rows(tokenizer, corpus: str, *, max_len: int, text_key: str, max_rows: int | None,
               max_tokens: int | None, source: str | None, hf_split: str, limit_docs: int | None,
               frame: str = "none", instruction: str = CONTINUE_INSTRUCTION, messages_key: str = "messages",
               close_final: bool = False, per_turn: bool = False, student_key: str | None = "student_messages"):
    """Deterministic row list: (row_id, doc_id, window, ids, ends, text bytes,
    messages, spans, frame kind, student messages). The last four are None
    on unframed rows; student messages are None unless the corpus row
    carries a list under student_key.

    frame "continue" renders every text window as a one-turn conversation
    (user: instruction, model: the window) through the teacher's chat
    template with no closing marker, targets on the window; with
    close_final the window that ends its document is closed by the
    template's turn-end marker and that marker is a target too (row kind
    "continue-closed"). frame "chat" reads conversations under
    messages_key and targets every assistant span; a conversation longer
    than max_len tokens loses trailing turns until it fits, and one that
    never fits is dropped. frame "reply" targets the final assistant turn
    only and drops leading turns to fit; with per_turn (under chat or
    reply) each conversation becomes one reply row per assistant turn with
    the history before it. A row with a student list must be a reply row
    ending on the same reply as the teacher's list; a mismatch is dropped
    and counted."""
    tb = token_bytes(tokenizer)
    ws = whitespace_start_mask(tokenizer, len(tb), tb)
    bos = _tokens.bos_id(tokenizer) if _tokens.adds_bos(tokenizer) else None
    rows = []
    n_tokens = 0
    corpus_hash = hashlib.sha256()
    flagged = 0
    dropped = 0
    frame_tokens = 0
    closed = 0
    student_rows = 0
    mismatch = 0

    def full() -> bool:
        return bool((max_rows and len(rows) >= max_rows) or (max_tokens and n_tokens >= max_tokens))

    if frame in ("chat", "reply", "reply-think"):
        reply_kind = per_turn or frame in ("reply", "reply-think")
        reason_target = frame == "reply-think"
        for doc_id, msgs, st in _corpus.iter_conversations(corpus, key=messages_key, student_key=student_key,
                                                           limit=limit_docs, hf_split=hf_split):
            while msgs and msgs[-1].get("role") != "assistant":
                msgs = msgs[:-1]
            _hash_doc(corpus_hash, doc_id, json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode())
            variants = _corpus.per_turn_rows(msgs) if per_turn else [msgs]
            if st is not None:
                corpus_hash.update(json.dumps(st, sort_keys=True, ensure_ascii=False).encode())
                if not reply_kind:
                    raise ValueError(f"{doc_id} carries {student_key}; such rows need --frame reply "
                                     "(or --per-turn), since only the final reply is common to both renders")
                while st and st[-1].get("role") != "assistant":
                    st = st[:-1]
                if not _corpus.same_reply(msgs, st):
                    # counted in rows, the unit of every other counter
                    mismatch += len(variants)
                    continue
            st_variants = (_corpus.per_turn_rows(st) if per_turn else [st]) if st is not None else None
            if st_variants is not None and len(st_variants) != len(variants):
                mismatch += len(variants)
                continue
            for w, m in enumerate(variants):
                # per-turn rows pair by position: the pair must end on the
                # same reply, as the whole conversations were checked to
                if st_variants is not None and not _corpus.same_reply(m, st_variants[w]):
                    mismatch += 1
                    continue
                row = (_frames.fit_reply(tokenizer, m, max_len, tb, reason_target=reason_target) if reply_kind
                       else _frames.fit_conversation(tokenizer, m, max_len, tb))
                if row is None:
                    dropped += 1
                    continue
                ids, ends, text, m2, spans, flag = row
                st_row = None
                if st_variants is not None:
                    st_row = list(st_variants[w])
                    lost = len(m) - len(m2)
                    if lost:
                        # the turns fit_reply dropped go from the student's
                        # list too, counted on the bodies after any system
                        # turn, since one side may carry a system turn alone
                        th = 1 if m[0].get("role") == "system" else 0
                        sh = 1 if st_row and st_row[0].get("role") == "system" else 0
                        if len(st_row) - sh != len(m) - th:
                            mismatch += 1
                            continue
                        st_row = st_row[:sh] + st_row[sh + lost:]
                    student_rows += 1
                flagged += int(flag)
                rows.append((len(rows), doc_id, w, ids.astype(np.int32), ends.astype(np.uint32), text, m2, spans,
                             ("reply-think" if reason_target else "reply") if reply_kind else "chat", st_row,
                             len(variants) if per_turn else None))
                n_tokens += len(ids)
                if full():
                    break
            if full():
                break
    else:
        if frame == "continue":
            frame_text = _frames.render_frame(tokenizer, [{"role": "user", "content": instruction}])
            fids, _, _ = _tokens.encode_with_byte_ends(tokenizer, frame_text.encode("utf-8"), tb,
                                                       add_special_tokens=False)
            frame_tokens = len(fids)
            tail_tokens = 0
            if close_final:
                tails = _frames.assistant_tails(tokenizer)
                if tails:
                    tids, _, _ = _tokens.encode_with_byte_ends(tokenizer, tails[0].encode("utf-8"), tb,
                                                               add_special_tokens=False)
                    tail_tokens = len(tids)
            budget = max_len - frame_tokens - tail_tokens
            if budget < 8:
                raise ValueError(
                    f"--max-len {max_len} leaves {budget} tokens for a window beside the {frame_tokens}-token "
                    f"continue frame{' and its closing tail' if tail_tokens else ''}, at least 8 are needed")
        else:
            budget = max_len - (1 if bos is not None else 0)
        for doc_id, text in _corpus.iter_corpus(corpus, text_key=text_key, limit=limit_docs, hf_split=hf_split):
            tbytes = text.encode("utf-8")
            if not tbytes.strip():
                continue
            _hash_doc(corpus_hash, doc_id, tbytes)
            ids, ends, flag = _tokens.encode_with_byte_ends(tokenizer, tbytes, tb, add_special_tokens=False)
            flagged += int(flag)
            if len(ids) == 0:
                continue
            for w, (s, e) in enumerate(_frames.cut_windows(ids, ws, budget, 0, text=tbytes, ends=ends)):
                b0 = int(ends[s - 1]) if s > 0 else 0
                b1 = int(ends[e - 1])
                wtext = tbytes[b0:b1]
                kind = None
                if frame == "continue":
                    content = wtext.decode("utf-8").lstrip()
                    if not content:
                        continue
                    msgs = _frames.continue_messages(content, instruction)
                    kind = "continue-closed" if close_final and e == len(ids) else "continue"
                    closed += int(kind == "continue-closed")
                    wtext, spans = _frames.render_row(tokenizer, msgs, **_frames.row_render_args(kind))
                    wids, wends, f2 = _tokens.encode_with_byte_ends(tokenizer, wtext, tb, add_special_tokens=False)
                    flagged += int(f2)
                    wends = wends.astype(np.int64)
                else:
                    wids = ids[s:e]
                    wends = ends[s:e].astype(np.int64) - b0
                    msgs, spans = None, None
                    if bos is not None:
                        wids = np.concatenate([[bos], wids]).astype(np.int32)
                        wends = np.concatenate([[0], wends])
                if len(wids) < 2:
                    continue
                rows.append((len(rows), doc_id, w, wids.astype(np.int32), wends.astype(np.uint32), wtext, msgs,
                             spans, kind))
                n_tokens += len(wids)
                if full():
                    break
            if full():
                break
    # sort by length, ties by row_id, so batches are compact and the order
    # is reproducible on resume
    rows.sort(key=lambda r: (len(r[3]), r[0]))
    info = {"dropped": dropped, "frame_tokens": frame_tokens, "closed_rows": closed,
            "student_rows": student_rows, "reply_mismatch": mismatch}
    return rows, n_tokens, corpus_hash.hexdigest(), flagged, tb, ws, source, info


def row_meta(r, source: str, frame: str, generator_id: str = "", special: set[int] | None = None) -> _format.RowMeta:
    """Row sidecar entry; framed rows carry their messages, teacher-render
    spans and the frame length as the prefix fields, plus the student's
    own message list when the corpus row had one. A token that spans no
    bytes after a segment marker (the dummy prefix of a Llama-2 style
    tokenizer after a mid-row control or added token) is listed under
    zero_width for the validator; ``special`` is the marker set the byte
    ends were computed with."""
    meta = _format.RowMeta(row_id=r[0], doc_id=str(r[1]), window=r[2], n_tokens=len(r[3]), source=source,
                           generator_id=generator_id)
    if special:
        zw = _tokens.zero_width_indices(r[3], r[4], special)
        if zw:
            meta.zero_width = zw
    if r[6] is not None:
        tm = _frames.target_mask(r[4].astype(np.int64), r[7])
        first = int(np.argmax(tm)) if tm.any() else len(r[3]) - 1
        meta.frame = r[8] if len(r) > 8 and r[8] else frame
        meta.messages = r[6]
        meta.spans = [[int(x) for x in sp] for sp in r[7]]
        meta.prefix_n_tokens = first + 1
        meta.suffix_start_byte = int(r[7][0][0])
        # where the reply's content starts: the target span opens at the
        # reasoning trace on a reply-think row, and the census keys
        # content positions from the content itself
        if meta.frame in ("reply", "reply-think"):
            cs = (r[6][-1].get("content") or "").strip()
            meta.content_start = int(r[7][-1][1]) - len(cs.encode("utf-8"))
        else:
            meta.content_start = int(r[7][-1][0])
        if len(r) > 9 and r[9] is not None:
            meta.student_messages = r[9]
        if len(r) > 10 and r[10] is not None:
            # the census needs the final turn's number even when no cache
            # holds the final turn
            meta.turns = int(r[10])
    return meta


def forward_trunk(model, inputs):
    inner = getattr(model, "language_model", model)
    return _student.trunk_hidden(inner, inputs)


def _manifest_aside(out: Path, problems: list[str]) -> int:
    """Log the validator's problems, move the manifest to
    manifest.invalid.json so no view or resume takes the cache as
    finished, and return the pass's exit code."""
    for x in problems[:10]:
        log("[cache] validate: " + x)
    bad = out / "manifest.invalid.json"
    os.replace(out / "manifest.json", bad)
    log(f"[cache] error: validator failed with {len(problems)} problems, manifest moved to {bad}; "
        "a resume rewrites the manifest, a problem inside a shard needs a fresh --out")
    return 4


def teacher_head(model) -> HeadSpec:
    inner = getattr(model, "language_model", model)
    return head_spec_from_model(inner)


IDENTITY_HEAD_BYTES = 16 * 1024 * 1024
IDENTITY_BLOCK_BYTES = 1024 * 1024
IDENTITY_BLOCKS = 64


def teacher_identity(path: str) -> dict:
    """What a resume compares to know it continues on the same teacher:
    the absolute path as named (symlinks kept, so a Hugging Face snapshot
    keeps its shard names and its .gguf suffix), the size and a hash over
    the leading bytes plus 64 one-mebibyte blocks spread through the rest
    of a GGUF file and its split shards, or of a directory checkpoint's
    config and each weight file, in order. The same bytes written again,
    touched, or reached by another name still match; a resume compares
    the size and the hash, not the path."""
    from gmlx.load.preflight import find_split_shards

    p = Path(path).expanduser().absolute()
    if p.is_file():
        try:
            files = [Path(f) for f in find_split_shards(str(p))]
        except FileNotFoundError:
            files = [p]     # the load names the missing shard
    else:
        files = sorted(list(p.glob("config.json")) + list(p.glob("*.safetensors")) + list(p.glob("*.gguf")))
    h = hashlib.sha256()
    size = 0
    for j, f in enumerate(files):
        n = int(f.stat().st_size)
        size += n
        h.update(f"{j}:{n}:".encode())
        with open(f, "rb") as fh:
            h.update(fh.read(IDENTITY_HEAD_BYTES))
            if n > IDENTITY_HEAD_BYTES:
                starts = np.linspace(IDENTITY_HEAD_BYTES, max(n - IDENTITY_BLOCK_BYTES, IDENTITY_HEAD_BYTES),
                                     IDENTITY_BLOCKS)
                for off in sorted(set(int(x) for x in starts)):
                    fh.seek(off)
                    h.update(fh.read(IDENTITY_BLOCK_BYTES))
    return {"path": str(p), "size": size, "sha256_head": h.hexdigest()}


def head_logits(head: HeadSpec, h):
    """The teacher's half-precision logits for hidden states ``h``: the
    projection, then the softcap when the head carries one. f16 logits
    stay f16 (a cast to bf16 would drop three mantissa bits), anything
    else goes to bf16. The pass and both memory probes go through here,
    so they measure the same graph."""
    import mlx.core as mx
    z = head.fn(head.params, h)
    if head.softcap:
        z = head.softcap * mx.tanh(z / head.softcap)
    return z if z.dtype == mx.float16 else z.astype(mx.bfloat16)


def run_fingerprint(opts: CacheOptions, corpus_sha: str, n_rows: int, n_tokens: int, render_kw,
                    tokenizer=None, source: str | None = None, generator_id: str = "") -> dict:
    """The inputs a resumed pass must share with the first run: the corpus
    and row options, the corpus's generator sidecar, the teacher by size
    and leading bytes (its path may be spelled another way), its
    vocabulary and, when rows are framed, its chat template (a directory
    checkpoint keeps both outside the hashed files), the config source,
    the hidden sketch and the render kwargs. Any difference refuses the
    resume. A key this build added compares as None against a record
    that predates it."""
    template = _frames.template_text(tokenizer) if tokenizer is not None else ""
    return {"corpus_sha256": corpus_sha, "n_rows": n_rows, "n_tokens": n_tokens, "source": source,
            "generator_id": generator_id,
            "tokenizer_hash": vocab_map_hash(tokenizer) if tokenizer is not None else None,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest() if opts.frame != "none" else None,
            "max_len": opts.max_len, "rows_per_shard": opts.rows_per_shard, "top_k": opts.top_k,
            "floor": bool(opts.floor), "frame": opts.frame, "routes": bool(opts.routes),
            "hidden": bool(opts.hidden), "hf_source": opts.hf_source,
            "teacher": {k: v for k, v in teacher_identity(opts.teacher).items() if k != "path"},
            "hidden_dim": opts.hidden_dim if opts.hidden else None,
            "hidden_seed": opts.hidden_seed if opts.hidden else None,
            "render_kwargs": (render_kw or None) if opts.frame != "none" else None,
            "per_turn": bool(opts.per_turn),
            "close_final_windows": bool(opts.close_final_windows),
            "frame_instruction": opts.frame_instruction if opts.frame == "continue" else None}


def teacher_over_budget(model, budget: int | None = None) -> tuple[bool, int, int]:
    """(over, parameter bytes, budget): whether the model's parameters
    exceed the wired budget, nine tenths of the recommended working set
    as the expert streaming installer reads it. Never over when no
    device reports one."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    flat: Any = tree_flatten(model.parameters())
    total = sum(int(a.nbytes) for _k, a in flat)
    if budget is None:
        try:
            budget = int(0.9 * int(mx.device_info()["max_recommended_working_set_size"]))
        except Exception:  # noqa: BLE001
            return False, total, 0
    return total > budget, total, budget


def experts_streaming(model) -> bool:
    """Whether the expert streaming installer put this model's experts in
    streaming mode. Reads the verdict the installer records on the model;
    a model without one reports through ``moe_streaming_active``, which
    also counts experts a CPU-only codec keeps resident."""
    from gmlx.load.loader import moe_streaming_active
    flag = getattr(model, "_kq_streaming", None)
    return bool(flag) if flag is not None else bool(moe_streaming_active(model))


def stream_over_budget(model, gguf_path: str, total: int, budget: int) -> tuple[int, int]:
    """Place a teacher that is over the wired budget: the installer
    streams its experts, or a table that alone brings it under. Returns
    (expert stacks wrapped, expert bytes streamed), the bytes zero when
    the experts stay resident. A teacher with nothing to stream is
    refused, since a resident load would pin it under the wired limit."""
    from gmlx.stream.expert_streaming import install_expert_streaming
    for var in ("GMLX_ARENA_SPLIT_MAX_TOKENS", "GMLX_ARENA_STAGE_MAX_TOKENS"):
        os.environ.setdefault(var, "0")
    n, offloaded = install_expert_streaming(model, gguf_path=gguf_path)
    if n == 0:
        raise ValueError(f"the teacher is {total / GB:.1f} GB against a wired budget of {budget / GB:.1f} GB and "
                         "has no expert stacks to stream; use a smaller quantization")
    if experts_streaming(model):
        log(f"[cache] teacher over the wired budget: {n} expert stacks stream from disk")
        return n, offloaded
    log("[cache] teacher over the wired budget: a streamed table brings it under, the experts stay resident")
    return n, 0


def load_teacher(opts: CacheOptions):
    """(model, config, arch, streaming, expert_bytes). A GGUF teacher
    loads through gmlx; one over the wired budget streams its experts
    from disk (and runs the prefill feeder), and ``stream_experts``
    forces that on a MoE teacher that would otherwise sit resident. A
    directory is loaded as an MLX checkpoint."""
    teacher_is_gguf = Path(opts.teacher).is_file() and opts.teacher.endswith(".gguf")
    offloaded = 0
    if teacher_is_gguf:
        import gmlx.load.loadlog as loadlog
        from gmlx.load.loader import load_model
        from gmlx.load.preflight import preflight
        with loadlog.load_ui(False, opts.teacher):
            model, config, _tok = load_model(opts.teacher, hf_source=opts.hf_source)
        arch = preflight(opts.teacher).arch
        if opts.stream_experts:
            from gmlx.stream.expert_streaming import install_expert_streaming
            # Expert calls of at most GMLX_ARENA_SPLIT_MAX_TOKENS tokens
            # (default 256; the stage path below GMLX_ARENA_STAGE_MAX_TOKENS,
            # default 64) are served by the decode feeder's arena, whose
            # output differs from the prefill path. A teacher pass is
            # prefill only, so both go to the prefill feeder.
            for var in ("GMLX_ARENA_SPLIT_MAX_TOKENS", "GMLX_ARENA_STAGE_MAX_TOKENS"):
                os.environ.setdefault(var, "0")
            n, offloaded = install_expert_streaming(model, gguf_path=opts.teacher, force_stream=True)
            if n == 0:
                raise ValueError("--stream-experts on a teacher with no expert stacks")
        else:
            over, total, budget = teacher_over_budget(model)
            if over:
                _n, offloaded = stream_over_budget(model, opts.teacher, total, budget)
        streaming = experts_streaming(model)
    else:
        model, config, _tok = _student.load_mlx_student(opts.teacher)
        arch = config.get("model_type", "mlx")
        streaming = False
    model.eval()
    return model, config, arch, streaming, offloaded


def run_cache(opts: CacheOptions) -> int:
    """The pass. Returns 0 on a validated cache, 2 on a refusal before the
    teacher runs or on a shard the writer could not put down, 3 when the
    memory probe fails twice or the recorded routes do not match the
    chunk, 4 when the validator finds a problem in what was written."""
    import mlx.core as mx

    out = Path(opts.out)
    t0 = time.perf_counter()
    try:
        frame_kwargs = _frames.parse_render_kwargs(opts.frame_kwargs)
    except ValueError as e:
        log(f"[cache] refuse: --frame-kwargs is not a JSON object: {e}")
        return 2
    if not Path(opts.teacher).expanduser().exists():
        log(f"[cache] refuse: no teacher at {opts.teacher}")
        return 2
    c = opts.corpus
    if (c.endswith((".jsonl", ".txt")) or c.startswith((".", "/", "~"))) and not Path(c).expanduser().exists():
        log(f"[cache] refuse: no corpus at {c}")
        return 2
    teacher_is_gguf = Path(opts.teacher).is_file() and opts.teacher.endswith(".gguf")
    log(f"[cache] tokenizer from {opts.teacher}" + (" (header only)" if teacher_is_gguf else ""))
    if opts.stream_experts and not teacher_is_gguf:
        log("[cache] refuse: --stream-experts needs a GGUF teacher")
        return 2
    tokenizer = _tokens.load_tokenizer(opts.teacher)
    # a resume renders as the first run did: the stored kwargs carry the
    # date a template reads, which would otherwise move to today
    stored = _format.read_json(out / "progress.json").get("run") \
        if opts.resume and (out / "progress.json").is_file() else None
    render_kw = _frames.resolve_render_kwargs(tokenizer, inherit=(stored or {}).get("render_kwargs"),
                                              override=frame_kwargs)
    _frames.set_render_kwargs(tokenizer, render_kw)
    if opts.frame != "none":
        why = _frames.template_problem(tokenizer)
        if why:
            log(f"[cache] refuse: {why}")
            return 2
        if not _frames.has_chat_template(tokenizer):
            log("[cache] warn: teacher has no chat template, framed rows use the plain render "
                "(contents separated by blank lines)")
        log(f"[cache] frame {opts.frame}: chat-template kwargs {render_kw or '{}'}, turn-end markers "
            f"{_frames.assistant_tails(tokenizer)!r}")
    try:
        rows, n_tokens, corpus_sha, flagged, tb, ws, source, frame_info = build_rows(
            tokenizer, opts.corpus, max_len=opts.max_len, text_key=opts.text_key, max_rows=opts.max_rows,
            max_tokens=opts.max_tokens, source=opts.source, hf_split=opts.hf_split, limit_docs=opts.limit_docs,
            frame=opts.frame, instruction=opts.frame_instruction, messages_key=opts.messages_key,
            close_final=opts.close_final_windows, per_turn=opts.per_turn, student_key=opts.student_messages_key)
    except ValueError as e:
        log(f"[cache] refuse: {e}")
        return 2
    try:
        generator, generator_id = _corpus.generator_sidecar(opts.corpus)
    except ValueError as e:
        log(f"[cache] refuse: {e}")
        return 2
    source = opts.source or ("synthetic" if generator else "human")
    special = _tokens.segment_markers(tokenizer)
    # rows with a zero-width token are not ones mlx-kld's reader can pair
    # with its own offsets
    zero_width_rows = sum(1 for r in rows if _tokens.zero_width_indices(r[3], r[4], special))
    if generator:
        log(f"[cache] generator sidecar: {generator.get('model')} filter_version "
            f"{generator.get('filter_version')} ({generator_id})")
    est = _format.estimate_cache_bytes(n_tokens, opts.top_k, opts.floor,
                                       hidden_bytes=2 * opts.hidden_dim if opts.hidden else 0.0)
    log(f"[cache] {len(rows)} rows, {n_tokens} tokens, estimate {est / GB:.2f} GB "
        f"({flagged} rows on the offsets fallback)")
    if opts.frame != "none":
        log(f"[cache] frame {opts.frame}: {frame_info['frame_tokens']} frame tokens, "
            f"{frame_info['dropped']} rows dropped, {frame_info['student_rows']} rows with a "
            f"student list, {frame_info['reply_mismatch']} rows dropped for a reply mismatch")
    if not rows:
        log(f"[cache] refuse: {opts.corpus} yields no rows (blank texts and conversations without an "
            "assistant turn are skipped)")
        return 2
    targets_total = sum(int(_frames.target_mask(r[4].astype(np.int64), r[7]).sum()) for r in rows)
    if opts.max_disk_gb is not None and est > opts.max_disk_gb * GB:
        log(f"[cache] refuse: estimate exceeds --max-disk-gb {opts.max_disk_gb}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    writer = _format.ShardWriter(out, opts.top_k, opts.floor, opts.max_disk_gb)
    done = writer.verified_shards() if opts.resume else 0
    if not opts.resume and writer.n_done:
        log(f"[cache] refuse: {out} already has {writer.n_done} shards, pass --resume or a fresh --out")
        return 2
    # the space still needed: a resume has the verified shards on disk
    fb = _format.free_bytes(out)
    need = max(est - writer.progress["bytes"], 0)
    if need > fb * 0.9:
        log(f"[cache] refuse: estimate {est / GB:.2f} GB ({need / GB:.2f} GB still to write) against "
            f"{fb / GB:.2f} GB free")
        return 2
    # rows are sorted by length over the whole row set, so any change to the
    # corpus or the row options changes every shard's contents: a resume
    # must continue the same run
    run = run_fingerprint(opts, corpus_sha, len(rows), int(n_tokens), render_kw, tokenizer, source=source,
                          generator_id=generator_id)
    prev = writer.progress.get("run")
    if opts.resume and prev is not None and any(prev.get(k) != run[k] for k in run):
        diff = ", ".join(f"{k} {prev.get(k)!r} -> {run[k]!r}" for k in run if prev.get(k) != run[k])
        log(f"[cache] refuse: --resume with other inputs than the first run ({diff}), use a fresh --out")
        return 2
    writer.progress["run"] = run
    shards = [rows[i:i + opts.rows_per_shard] for i in range(0, len(rows), opts.rows_per_shard)]
    log(f"[cache] {len(shards)} shards of {opts.rows_per_shard} rows, {done} verified already")
    if opts.resume and done == len(shards) and (out / "manifest.json").is_file():
        # a rewritten manifest would change the hash every view carries,
        # so a finished cache is validated as it stands and left alone
        # every shard's hash was checked a moment ago, the contents remain
        problems = _format.validate_cache(out, check_sha=False)
        if problems:
            return _manifest_aside(out, problems)
        log(f"[cache] nothing to do: all {done} shards verified, the manifest is present and the validator passed")
        return 0
    if opts.resume and (out / "manifest.json").is_file():
        # the manifest describes shards this run rewrites; without it a
        # resume cut short cannot pass as finished
        (out / "manifest.json").unlink()
        log("[cache] removed the manifest of an unfinished cache, it is rewritten after the last shard")

    try:
        model, config, arch, streaming, offloaded = load_teacher(opts)
    except ValueError as e:
        log(f"[cache] refuse: {e}")
        return 2
    cfg = config.get("text_config", config) if isinstance(config, dict) else config
    head = teacher_head(model)
    gap = head_parity_gap(model, head, mx.array(rows[0][3][:8].astype(np.int32))[None])
    if gap > HEAD_PARITY_TOL:
        log(f"[cache] refuse: the head does not reproduce the teacher's own logits (relative gap {gap:.3f}), "
            "the model changes its logits after the projection in a way the distill head does not carry")
        return 2
    V = head.V
    if opts.top_k >= V:
        log(f"[cache] refuse: --top-k {opts.top_k} is not below the head width {V}")
        return 2
    hidden_blk = writer.progress.get("hidden") if opts.hidden else None
    R_hidden = None   # the sketch matrix, built from the first trunk chunk's width
    recorder, routing = None, None
    if opts.routes:
        recorder, why = _format.install_route_recording(getattr(model, "language_model", model))
        if recorder is None:
            log(f"[cache] refuse: --routes: {why}")
            return 2
        n_experts = next((int(cfg[k]) for k in ("num_experts", "n_routed_experts", "num_local_experts",
                                                  "moe_num_experts") if isinstance(cfg, dict) and cfg.get(k)), None)
        if n_experts is None:
            log("[cache] refuse: --routes: expert count not in the config (num_experts, n_routed_experts, "
                "num_local_experts, moe_num_experts)")
            return 2
        routing = {"moe_layers": list(recorder.layers), "k": None, "n_experts": n_experts,
                   "dtype": np.dtype(_format.routes_dtype(n_experts)).name, "layout": "[B, L, n_moe, k]",
                   "path": "streaming" if streaming else "resident", "seam": "gmlx.stream.moe_routes"}
        # k is learned from the first recorded chunk, so a resume that finds
        # every shard written takes it from progress.json
        if opts.resume and (writer.progress.get("routing") or {}).get("k") is not None:
            routing["k"] = int(writer.progress["routing"]["k"])
        log(f"[cache] routes: {len(recorder.layers)} MoE layers, {n_experts} experts, {routing['dtype']}")
    feeder = getattr(model, "_kq_feeder", None) if streaming else None
    if streaming:
        log(f"[cache] streaming teacher: feeder installed={feeder is not None} "
            f"expert bytes {offloaded / GB:.1f} GB")
        if feeder is None and opts.require_feeder:
            log("[cache] refuse: streaming teacher requested and no prefill feeder is installed")
            return 2
    if not streaming and not opts.no_wired_limit:
        try:
            mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
        except Exception as e:  # noqa: BLE001
            log(f"[cache] warn: wired limit not set: {e}")
    # shard rows pad to a per-shard length, so freed buffers of many sizes
    # would otherwise accumulate in MLX's cache up to the memory limit
    mx.set_cache_limit(int(opts.cache_limit_gb * GB))
    from gmlx.gen.prefill_plan import prefill_plan
    plan = prefill_plan(model, V=V, cap_gb=opts.logits_cap_gb, floor=opts.floor, streaming=streaming,
                        requested_trunk=opts.trunk)
    step = plan["step"]
    T = plan["trunk"]
    log(f"[cache] plan: V={V} step={step} trunk={T} streaming={streaming} "
        f"headroom={plan['headroom_gb']} bytes/V budget={plan['bytes_per_v']}")
    # the mask lives at the measured head width V (padded ids are never
    # whitespace-initial), not at the tokenizer's length
    ws = whitespace_start_mask(tokenizer, V, token_bytes(tokenizer, V))
    log_bmask = log_bmask_from(ws)
    baseline = mx.get_active_memory()
    constant = plan["bytes_per_v"]
    probed = False
    tokenizer_hash = vocab_map_hash(tokenizer)
    tokens_done = writer.progress["tokens"]
    tokens_run = 0
    reads_bytes = 0.0
    E_bytes = opts.expert_bytes_gb * GB if opts.expert_bytes_gb is not None else float(offloaded)
    row_len = opts.max_len
    trunk_wall = 0.0
    trunk_forwards = 0

    for si in range(done, len(shards)):
        shard_rows = shards[si]
        ts = time.perf_counter()
        reduced = []
        # trunk chunks of at most T tokens, rows on the batch axis
        rows_per_chunk = max(1, T // row_len)
        for ci in range(0, len(shard_rows), rows_per_chunk):
            chunk = shard_rows[ci:ci + rows_per_chunk]
            L = max(len(r[3]) for r in chunk)
            inputs = np.zeros((len(chunk), L), dtype=np.int32)
            for j, r in enumerate(chunk):
                inputs[j, :len(r[3])] = r[3]
            tt = time.perf_counter()
            hidden = forward_trunk(model, mx.array(inputs))
            mx.eval(hidden)
            trunk_wall += time.perf_counter() - tt
            trunk_forwards += 1
            hidden_sk = None
            if opts.hidden:
                if R_hidden is None:
                    d_model = int(hidden.shape[-1])
                    R_hidden = mx.array(_hidden.projection_matrix(d_model, opts.hidden_dim, opts.hidden_seed))
                    hidden_blk = _hidden.hidden_block(d_model, opts.hidden_dim, opts.hidden_seed)
                    writer.progress["hidden"] = hidden_blk
                    log(f"[cache] hidden: final state {d_model} -> {opts.hidden_dim} dims, "
                        f"seed {opts.hidden_seed}, float16")
                hidden_sk = _hidden.sketch(hidden, R_hidden)
            routes_blt = None
            if recorder is not None and routing is not None:
                routes_blt = _format.take_routes_blt(recorder)
                if routes_blt.shape[:2] != inputs.shape:
                    log(f"[cache] refuse: routes recorded as {routes_blt.shape[:2]} for a {inputs.shape} chunk")
                    return 3
                if routes_blt.shape[2] != len(recorder.layers):
                    log(f"[cache] refuse: routes recorded for {routes_blt.shape[2]} layers, "
                        f"{len(recorder.layers)} MoE layers installed")
                    return 3
                if routing["k"] is None:
                    routing["k"] = int(routes_blt.shape[-1])
                    writer.progress["routing"] = routing
                    per_pos = routes_blt.shape[2] * routing["k"] * np.dtype(routes_blt.dtype).itemsize
                    est = _format.estimate_cache_bytes(n_tokens, opts.top_k, opts.floor, routes_bytes=per_pos,
                                                       hidden_bytes=2 * opts.hidden_dim if opts.hidden else 0.0)
                    log(f"[cache] routes: k={routing['k']}, estimate with routes {est / GB:.2f} GB")
                    if opts.max_disk_gb is not None and est > opts.max_disk_gb * GB:
                        log(f"[cache] refuse: estimate with routes exceeds --max-disk-gb {opts.max_disk_gb}")
                        return 2
                routes_blt = routes_blt.astype(_format.routes_dtype(routing["n_experts"]))
            if streaming and E_bytes:
                reads_bytes += E_bytes
            # flatten valid positions
            lens = [len(r[3]) for r in chunk]
            flat_h = mx.concatenate([hidden[j, :lens[j]] for j in range(len(chunk))], axis=0)
            nxt = np.concatenate([np.concatenate([r[3][1:], [-1]]) for r in chunk]).astype(np.int32)
            # targets: every successor on an unframed row, the target spans on a framed one
            tmask = np.concatenate([_frames.target_mask(r[4].astype(np.int64), r[7]) for r in chunk])
            valid = (nxt >= 0) & tmask
            n = int(flat_h.shape[0])
            parts = []
            s = 0
            while s < n:
                e = min(s + step, n)
                if not probed:
                    mx.reset_peak_memory()
                logits = head_logits(head, flat_h[s:e])
                mx.eval(logits)
                red = _cache.reduce_logits(logits, mx.array(nxt[s:e]), K=opts.top_k, log_bmask=log_bmask,
                                           onpath_valid=valid[s:e], floor=opts.floor)
                del logits
                if not probed:
                    peak = mx.get_peak_memory()
                    measured = _cache.measure_bytes_per_v(peak, baseline, e - s, V)
                    new_step, constant = _cache.probe_step(measured, plan["bytes_per_v"], V,
                                                           opts.logits_cap_gb)
                    log(f"[cache] probe: measured {measured:.1f} B per V-element, budget "
                        f"{plan['bytes_per_v']}, step {step} -> {new_step}")
                    if new_step != step:
                        step = new_step
                        # second probe on the next sub-chunk confirms; a second failure refuses
                        mx.reset_peak_memory()
                        e2 = min(s + step, n)
                        logits = head_logits(head, flat_h[s:e2])
                        mx.eval(logits)
                        _ = _cache.reduce_logits(logits, mx.array(nxt[s:e2]), K=opts.top_k, log_bmask=log_bmask,
                                                 onpath_valid=valid[s:e2], floor=opts.floor)
                        del logits
                        again = _cache.measure_bytes_per_v(mx.get_peak_memory(), baseline, e2 - s, V)
                        if again * step * V > opts.logits_cap_gb * GB:
                            log(f"[cache] refuse: peak still over the cap at step {step} "
                                f"({again:.1f} B per V-element)")
                            return 3
                    try:
                        writer.set_constant("bytes_per_v_element", float(constant))
                        writer.set_constant("bytes_per_v_measured", float(measured))
                    except OSError as e:
                        log(f"[cache] refuse: cannot write progress.json: {e}, the {writer.n_done} verified "
                            "shards stay for --resume")
                        return 2
                    probed = True
                parts.append(red)
                s = e
            # stitch per row
            merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            off = 0
            for j, r in enumerate(chunk):
                m = lens[j]
                rr = {k: v[off:off + m] for k, v in merged.items()}
                rr["onpath_mask"] = valid[off:off + m]
                rr["token_ids"] = r[3]
                rr["token_end_byte"] = r[4]
                if routes_blt is not None:
                    rr[_format.ROUTES_FIELD] = routes_blt[j, :m]
                if hidden_sk is not None:
                    rr[_format.HIDDEN_FIELD] = hidden_sk[j, :m]
                reduced.append((r, rr))
                off += m
            # the chunk's hidden states would otherwise stay alive through
            # the next chunk's trunk forward
            del hidden, flat_h, hidden_sk
        packed = _format.pack_shard([rr for _, rr in reduced], [r[5] for r, _ in reduced], opts.top_k, opts.floor)
        metas = [row_meta(r, source, opts.frame, generator_id, special) for r, _ in reduced]
        wall = time.perf_counter() - ts
        try:
            entry = writer.write(si, packed, metas, wall_s=wall, step=step, trunk_chunk=T,
                                 max_id=int(packed["top_k_indices"].max()))
        except (RuntimeError, OSError) as e:
            log(f"[cache] refuse: {e}, the {writer.n_done} verified shards stay for --resume")
            return 2
        tokens_done += entry["tokens"]
        tokens_run += entry["tokens"]
        rate = entry["tokens"] / max(wall, 1e-9)
        log(f"[cache] shard {si + 1}/{len(shards)}: {entry['tokens']} tokens {wall:.1f}s "
            f"({rate:.0f} tok/s) {entry['bytes'] / GB:.3f} GB peak {mx.get_peak_memory() / GB:.1f} GB")

    captured = []
    entries = writer.progress["shards"]
    spread = sorted(set(int(x) for x in np.linspace(0, len(entries) - 1, 8))) if entries else []
    for e in (entries[k] for k in spread):
        sh = _format.load_shard(writer.shard_path(e["index"]))
        m = sh["attention_mask"]
        lp = sh["top_k_log_softmax"].astype(np.float32)[m]
        valid = (sh["top_k_indices"] >= 0)[m]
        captured.append(np.exp(np.where(valid, lp, -np.inf)).sum(axis=-1))
    captured = np.concatenate(captured) if captured else np.zeros(0)
    edges = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001]
    # float16 log-probs can sum past 1, and the top bin would drop them
    hist = np.histogram(np.minimum(captured, 1.0), bins=edges)[0].tolist() if captured.size else []
    wall_total = time.perf_counter() - t0
    template = _frames.template_text(tokenizer)
    teacher_abs = str(Path(opts.teacher).expanduser().absolute())
    try:
        _format.write_manifest(
            out, teacher_path=teacher_abs, dataset=opts.corpus, num_samples=len(rows),
            max_seq_len=opts.max_len, seed=0, top_k=opts.top_k, vocab_size=V,
            config_vocab_size=(cfg.get("vocab_size") if isinstance(cfg, dict) else None),
            tokenizer_hash=tokenizer_hash, batch_size=opts.rows_per_shard,
            gmlx_distill={
                "teacher": {"arch": arch, "path": teacher_abs, "feeder_installed": bool(feeder),
                            "streaming": streaming},
                "corpus": {"spec": opts.corpus, "rows": len(rows), "tokens": n_tokens,
                           "window_policy": "last whitespace-initial boundary",
                           "normalization": "NFC", "rows_offset_fallback": flagged,
                           "boundary_byte_set": " \\t\\n\\r\\x0b\\x0c", "source": source,
                           "zero_width_rows": int(zero_width_rows)},
                "corpus_sha256": corpus_sha,
                "generator": generator,
                "mlx_kld_compatible": opts.frame == "none" and not zero_width_rows,
                "routing": routing,
                "hidden": hidden_blk,
                "frame": None if opts.frame == "none" else {
                    "kind": opts.frame,
                    "instruction": opts.frame_instruction if opts.frame == "continue" else None,
                    "frame_tokens": frame_info["frame_tokens"],
                    "rows_dropped": frame_info["dropped"],
                    "close_final_windows": bool(opts.close_final_windows) if opts.frame == "continue" else None,
                    "closed_rows": frame_info.get("closed_rows", 0),
                    "per_turn": bool(opts.per_turn),
                    "student_messages_key": opts.student_messages_key,
                    "student_rows": frame_info.get("student_rows", 0),
                    "reply_mismatch": frame_info.get("reply_mismatch", 0),
                    "target_positions": targets_total,
                    "render_kwargs": render_kw,
                    "turn_end_markers": _frames.assistant_tails(tokenizer),
                    "has_template": _frames.has_chat_template(tokenizer),
                    "render_date": datetime.date.today().isoformat(),
                    "template_sha256": hashlib.sha256(template.encode()).hexdigest()},
                "prefill_plan": dict(plan, step_final=step, bytes_per_v_in_force=constant),
                "throughput": {"tok_s": tokens_done / max(writer.progress["wall_s"], 1e-9),
                               "wall_s": wall_total, "gb_written": writer.progress["bytes"] / GB,
                               "tb_read": reads_bytes / 1e12, "tokens_this_run": tokens_run,
                               "trunk_forwards": trunk_forwards, "trunk_wall_s": trunk_wall,
                               "trunk_tok_s": tokens_run / max(trunk_wall, 1e-9),
                               "expert_bytes_gb": E_bytes / GB,
                               "stream_bandwidth_gb_s": (E_bytes * trunk_forwards / max(trunk_wall, 1e-9) / GB)
                               if streaming and E_bytes else None},
                "peak_memory": {"peak_gb": mx.get_peak_memory() / GB,
                                "active_gb": mx.get_active_memory() / GB},
                "captured_mass_histogram": {"edges": edges[:-1] + [1.0], "counts": hist,
                                            "mean": float(captured.mean()) if captured.size else None},
                "serves": "any student tokenizer",
            })
    except OSError as e:
        log(f"[cache] refuse: cannot write the manifest under {out}: {e}, the shards stay for --resume")
        return 2
    problems = _format.validate_cache(out)
    if problems:
        return _manifest_aside(out, problems)
    if (out / "manifest.invalid.json").exists():
        (out / "manifest.invalid.json").unlink()
    log(f"[cache] done: {tokens_done} tokens, {writer.progress['bytes'] / GB:.2f} GB, validator passed")
    return 0
