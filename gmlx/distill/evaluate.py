"""The evaluator driver: a student with an optional adapter scored on
held-out bits per byte, chat and reply slices, sparse KL against a cache,
downstream tasks and the chat sanity set, before and after the adapter in
one process, with a Markdown and a JSON report. ``run_eval`` is what
``gmlx distill eval`` calls."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gmlx.load.tokenizer import token_bytes, vocab_map_hash

from . import eval as _eval
from . import frames as _frames
from . import tokens as _tokens
from .constants import GB, log
from .corpus import message_list, nfc, norm_messages
from .data import CacheReader
from .format import (read_json, replay_layers_for, routing_block, shard_texts, write_bytes_atomic,
                     write_json_atomic)
from .student import adapter_disabled
from .head import HEAD_PARITY_TOL, head_parity_gap, head_spec_from_model
from .trainer import load_student
from .view import student_width


@dataclass
class EvalOptions:
    student: str
    md: str
    json: str
    adapter: str | None = None
    cache: str | None = None
    slices: list[str] = field(default_factory=list)
    teacher_bpb: str | None = None
    tasks_dir: str | None = None
    tasks: str = ""
    task_limit: int | None = None
    gsm8k_max_tokens: int = 384
    before: bool = False
    chat_slices: list[str] = field(default_factory=list)
    chat_sanity: str | None = None
    chat_max_tokens: int = 256
    chat_refs: str | None = None
    chat_max_len: int = 2048
    chat_per_turn: bool = False
    reply_slices: list[str] = field(default_factory=list)
    reply_think: bool = False
    reply_positions: str | None = None
    kld_cache: str | None = None
    kld_rows: int | None = None
    frame_kwargs: str | None = None
    max_len: int = 512
    bpb_prefix: str | None = None
    batch_size: int = 8
    cache_limit_gb: float = 4.0
    decontam_threshold: float = 0.01
    hf_source: str | None = None


class UnreadableInput(Exception):
    """A slice, task or report file that cannot be read as what it should be."""


def read_slice(path: Path) -> str:
    """A --slice file's text. A file that is not UTF-8 is unreadable, since
    its bytes would score as replacement characters, and so is one the
    process cannot open."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise UnreadableInput(f"{path}: not UTF-8 (byte {e.start}), convert it") from None
    except OSError as e:
        raise UnreadableInput(f"{path}: {e.strerror or e}") from None


def read_jsonl(path: Path) -> list[dict]:
    """The rows of a jsonl file; a ``messages`` or ``student_messages``
    list is checked message by message and NFC-normalized."""
    rows: list[tuple[int, dict]] = []
    try:
        # newlines only: a row may hold U+2028 or another line separator
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if line.strip():
                rows.append((n, json.loads(line)))
    except (OSError, ValueError) as e:
        raise UnreadableInput(f"{path}: {e}") from e
    if not all(isinstance(r, dict) for _n, r in rows):
        raise UnreadableInput(f"{path}: every line must be a JSON object")
    for n, r in rows:
        for key in ("messages", "student_messages"):
            # a null student list means the row has none, as cache reads it
            if key in r and not (key == "student_messages" and r[key] is None):
                try:
                    r[key] = norm_messages(message_list(r, key, f"{path.name} line {n}"))
                except ValueError as e:
                    raise UnreadableInput(str(e)) from None
    return [r for _n, r in rows]


def read_conversations(path: Path) -> list:
    """The ``messages`` list of every row of a chat-slice jsonl."""
    try:
        return [r["messages"] for r in read_jsonl(path)]
    except KeyError as e:
        raise UnreadableInput(f"{path}: a row has no {e} key") from e


def read_report(path: Path) -> dict:
    try:
        d = read_json(path)
    except (OSError, ValueError) as e:
        raise UnreadableInput(f"{path}: {e}") from e
    if not isinstance(d, dict):
        raise UnreadableInput(f"{path}: not a JSON object")
    return d


def reply_row_ids(reply_slices: dict) -> set[str]:
    """The ids of every reply-slice row (its ``id``, else its index in the
    file, the way the scorer names it), the keys a census high_delta map
    must share with them for ``--reply-positions`` to score anything."""
    return {str(r.get("id", i)) for rows in reply_slices.values() for i, r in enumerate(rows)}


def check_keys(path, rows: list, keys: tuple) -> list:
    """rows, each of which must carry every key, else UnreadableInput."""
    for r in rows:
        for k in keys:
            if k not in r:
                raise UnreadableInput(f"{path}: a row has no {k!r} key")
    return rows


TASK_KEYS = {"gsm8k": ("id", "question", "answer")}
MC_KEYS = ("id", "query", "choices", "gold")


def check_gsm8k_items(path, rows: list, keys: tuple = TASK_KEYS["gsm8k"]) -> list:
    """rows of gsm8k (or its shots, with keys question and answer): the
    question and the answer are strings, else UnreadableInput (the
    extractor would raise after the first generation)."""
    for r in check_keys(path, rows, keys):
        for k in ("question", "answer"):
            if not isinstance(r[k], str):
                raise UnreadableInput(f"{path}: item {r.get('id')!r} has no string {k!r}")
    return rows


def check_mc_items(path, rows: list) -> list:
    """rows of a multiple-choice task: query is a string, choices a
    non-empty list of strings and gold an index into it, else
    UnreadableInput (the scorer would raise after the slices ran)."""
    for r in check_keys(path, rows, MC_KEYS):
        if not isinstance(r["query"], str):
            raise UnreadableInput(f"{path}: item {r.get('id')!r} has no string 'query'")
        ch = r["choices"]
        if not isinstance(ch, list) or not ch or not all(isinstance(c, str) for c in ch):
            raise UnreadableInput(f"{path}: item {r.get('id')!r} has no list of choice strings")
        g = r["gold"]
        if isinstance(g, bool) or not isinstance(g, int) or not 0 <= g < len(ch):
            raise UnreadableInput(f"{path}: item {r.get('id')!r} gold {g!r} is not an index into its "
                                  f"{len(ch)} choices")
    return rows


def teacher_bpb_map(path) -> dict:
    """{slice: bpb} from a --teacher-bpb file: a plain map, or an eval
    report whose after.bpb block is read."""
    d = read_report(path)
    after = d.get("after")
    if isinstance(after, dict) and isinstance(after.get("bpb"), dict):
        return {n: (r.get("bpb") if isinstance(r, dict) else None) for n, r in after["bpb"].items()}
    if d and all(v is None or isinstance(v, (int, float)) for v in d.values()):
        return d
    raise UnreadableInput(f"{path}: neither a {{slice: bpb}} map nor an eval report")


def chat_refs(path) -> dict:
    """{id: reply} of the compliant, non-empty replies in an earlier
    report, the anchors of the drift score."""
    items = (read_report(path).get("after") or {}).get("chat", {}).get("items", [])
    try:
        return {r["id"]: r["reply"] for r in items if r["compliant"] and r["reply"].strip()}
    except (KeyError, TypeError, AttributeError) as e:
        raise UnreadableInput(f"{path}: not an eval report with chat items ({e})") from e


def cache_sources(cache: Path) -> dict[str, int]:
    """Rows per source of a cache, read from its sidecars alone."""
    try:
        rows = CacheReader(cache, keep=1).rows_meta
    except (OSError, ValueError, KeyError) as e:
        raise UnreadableInput(f"{cache}: not a cache ({e})") from e
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("source", "human")] = counts.get(r.get("source", "human"), 0) + 1
    return counts


def kld_cache_refusal(kld_dir: Path, manifest: dict, tokenizer, head_width: int) -> str | None:
    """Why --kld-cache cannot score this student, or None: the cache's
    vocabulary must be the student's (equal maps, or pad-style surplus ids
    on one side) and every cached id must fall inside the student's head,
    since the sparse KL gathers the cache's token ids from the student's
    logits."""
    top = manifest.get("max_top_k_id")
    if top is not None:
        if int(top) >= head_width:
            return f"it holds token id {top}, beyond the student's head ({head_width})"
    elif int(manifest.get("vocab_size", 0)) > head_width:
        return (f"its vocabulary ({manifest.get('vocab_size')}) is wider than the student's head ({head_width}) "
                "and it records no largest cached id")
    if manifest.get("tokenizer_hash") == vocab_map_hash(tokenizer):
        return None
    # the hash is the fast path; a same-vocabulary pair whose maps differ
    # by pad-style surplus ids is what align accepts as identity too
    tok_dir = kld_dir / "tokenizer"
    src = str(tok_dir) if tok_dir.exists() else manifest.get("teacher_path")
    if not src:
        return "it carries no tokenizer directory and its manifest names no teacher_path"
    if not Path(src).expanduser().exists():
        return f"its tokenizer is not at {src} (no tokenizer directory, and the manifest's teacher_path is gone)"
    same, why = _tokens.identity_pair(_tokens.load_tokenizer(src), tokenizer)
    return None if same else f"it was cached with another tokenizer than the student's ({why})"


def kld_replay_layers(model, manifest: dict, adapter: str | None) -> list[int] | None:
    """The cache's MoE layers to replay recorded routes on: the model must
    carry the teacher's layers and run without an adapter, since an
    adapter's own routing changes are part of what the score measures."""
    if adapter:
        return None
    return replay_layers_for(getattr(model, "language_model", model), manifest)


def kld_line(label: str, k: dict) -> str:
    return (f"[eval] kld vs cache K={k['K']}{label}: {_fmt(k['mean_kld_nats'], 5)} nats se "
            f"{_fmt(k['clustered_se'], 5)} top-1 {_fmt(k['top1_agreement'])} over {k['rows']} rows, "
            f"{k['rerendered_rows']} re-rendered through the student template ({k['wall_s']:.0f}s)")


def chat_line(label: str, c: dict) -> str:
    return (f"[eval] chat {label}: compliance {_fmt(c['compliance'], 3)} truncated {_fmt(c['truncated_rate'], 3)} "
            f"refusal {_fmt(c['refusal_rate'], 3)} task refusal {_fmt(c['task_refusal_rate'], 3)} "
            f"ref nll {_fmt(c.get('ref_nll_nats'))} ({c['wall_s']:.0f}s)")


def corpus_texts(cache: Path):
    """The cached rows' text, read from the text fields alone so the
    top-K arrays of every shard stay on disk."""
    from safetensors import safe_open
    reader = CacheReader(cache, keep=1)
    for i in range(reader.n_shards):
        with safe_open(str(reader.dir / f"batch-{i:05d}.safetensors"), framework="np") as fh:
            sh = {k: fh.get_tensor(k) for k in ("text_bytes", "text_offsets")}
        for t in shard_texts(sh):
            yield t


def _named(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"expected name=path, got {spec!r}")
        if any(name == n for n, _p in out):
            raise ValueError(f"slice {name!r} is named twice")
        out.append((name, os.path.expanduser(path)))
    return out


def run_arm(model, tokenizer, opts: EvalOptions, slices: dict[str, str], tasks: dict,
            chat_slices: dict | None = None, reply_slices: dict | None = None,
            positions: dict | None = None, trace_positions: dict | None = None, head=None,
            label: str = "after") -> dict:
    """Every instrument on the loaded weights as they are. ``positions``
    and ``trace_positions``, a census high_delta map and its trace half,
    restrict every reply slice to the byte ranges they name. ``head`` is
    the student's head spec, so the cache KL runs the head over the
    scored positions only. ``label`` names the arm in every log line."""
    res: dict = {"bpb": {}, "tasks": {}, "chat_bpb": {}, "reply_bpb": {}}
    for name, convs in (chat_slices or {}).items():
        t0 = time.perf_counter()
        model.eval()
        r = _eval.chat_slice_nll(model, tokenizer, convs, max_len=opts.chat_max_len,
                                 batch_tokens=opts.batch_size * opts.max_len, per_turn=opts.chat_per_turn)
        r["wall_s"] = time.perf_counter() - t0
        res["chat_bpb"][name] = r
        log(f"[eval] {name} {label}: assistant-turn bpb {_fmt(r['bpb'])} ({_fmt(r['nll_per_token'])} nats/token) over "
            f"{r['rows']} {'turns' if opts.chat_per_turn else 'conversations'}, {r['dropped']} dropped "
            f"({r['wall_s']:.0f}s)")
    for name, rows in (reply_slices or {}).items():
        t0 = time.perf_counter()
        model.eval()
        r = _eval.reply_slice_nll(model, tokenizer, rows, max_len=opts.chat_max_len,
                                  batch_tokens=opts.batch_size * opts.max_len, positions=positions,
                                  reason_target=opts.reply_think, trace_positions=trace_positions)
        r["wall_s"] = time.perf_counter() - t0
        res["reply_bpb"][name] = r
        log(f"[eval] {name} {label}: reply bpb {_fmt(r['bpb'])} ({_fmt(r['nll_per_token'])} nats/token) over "
            f"{r['rows']} rows{' at the high-delta positions' if positions is not None else ''}, "
            f"{r['dropped']} dropped ({r['wall_s']:.0f}s)")
    if opts.kld_cache:
        t0 = time.perf_counter()
        model.eval()
        reader = CacheReader(Path(opts.kld_cache), keep=2)
        replay = kld_replay_layers(model, reader.manifest, opts.adapter)
        if replay is None and not opts.adapter and routing_block(reader.manifest):
            log("[eval] the cache's recorded routes are not replayed: the student's MoE layers are not the "
                "teacher's, or gmlx cannot replay their gates")
        k = _eval.cache_kld(model, reader, max_rows=opts.kld_rows, tokenizer=tokenizer, replay_layers=replay,
                            head=head)
        k["wall_s"] = time.perf_counter() - t0
        res["kld"] = k
        log(kld_line(f" {label}", k))
    tb = token_bytes(tokenizer)
    prefix = None
    if opts.bpb_prefix:
        prefix = _frames.frame_prefix(tokenizer, opts.bpb_prefix[1:]) if opts.bpb_prefix.startswith("@") \
            else literal_prefix(opts.bpb_prefix)
    res["bpb_prefix_text"] = prefix
    for name, text in slices.items():
        t0 = time.perf_counter()
        r = _eval.bits_per_byte(model, tokenizer, text, max_len=opts.max_len, batch_size=opts.batch_size, tb=tb,
                                window_prefix=prefix)
        r["wall_s"] = time.perf_counter() - t0
        res["bpb"][name] = r
        log(f"[eval] {name} {label}: bpb {_fmt(r['bpb'])} over {r['bytes']} bytes ({r['wall_s']:.0f}s)")
    model.eval()
    for tname, items in tasks.items():
        t0 = time.perf_counter()
        if tname == "gsm8k":
            per = _eval.score_gsm8k(model, tokenizer, items["items"], items["shots"],
                                    max_tokens=opts.gsm8k_max_tokens)
        else:
            per = _eval.score_multiple_choice(model, tokenizer, items["items"])
        acc = float(np.mean([p["correct"] for p in per])) if per else None
        res["tasks"][tname] = {"acc": acc, "n": len(per), "items": per, "wall_s": time.perf_counter() - t0}
        log(f"[eval] {tname} {label}: acc {_fmt(acc)} on {len(per)} items ({res['tasks'][tname]['wall_s']:.0f}s)")
    return res


def _fmt(v, digits=4):
    return "None" if v is None else f"{v:.{digits}f}"


def report_markdown(opts: EvalOptions, report: dict, slices: dict, chat_slices: dict, reply_slices: dict,
                    contaminated: set) -> str:
    md = [f"# Distill eval: {Path(opts.student).name}"
          + (f" + {Path(opts.adapter).name}" if opts.adapter else ""), ""]
    before = report.get("before", {})
    if slices:
        md += ["| slice | bpb after | bpb before | teacher bpb | decontam | gate |", "|---|---|---|---|---|---|"]
        for name in slices:
            a = report["after"]["bpb"][name]["bpb"]
            b = before.get("bpb", {}).get(name, {}).get("bpb")
            t = report.get("teacher_bpb", {}).get(name)
            d = report["decontam"].get(name)
            md.append(f"| {name} | {_fmt(a)} | {_fmt(b)} | {_fmt(t)} | {_fmt(d, 5)} | "
                      f"{'void' if name in contaminated else 'ok' if d is not None else 'unchecked'} |")
    if chat_slices:
        md += ["", "| chat slice | bpb after | bpb before | se | conversations |", "|---|---|---|---|---|"]
        for name in chat_slices:
            a = report["after"]["chat_bpb"][name]
            b = before.get("chat_bpb", {}).get(name, {}).get("bpb")
            md.append(f"| {name} | {_fmt(a['bpb'])} | {_fmt(b)} | {_fmt(a.get('row_bpb_se'))} | {a['rows']} |")
    if reply_slices:
        md += ["", "| reply slice | bpb after | bpb before | nats/token after | nats/token before | se | rows |",
               "|---|---|---|---|---|---|---|"]
        for name in reply_slices:
            a = report["after"]["reply_bpb"][name]
            b = before.get("reply_bpb", {}).get(name, {})
            md.append(f"| {name} | {_fmt(a['bpb'])} | {_fmt(b.get('bpb'))} | {_fmt(a['nll_per_token'])} | "
                      f"{_fmt(b.get('nll_per_token'))} | {_fmt(a.get('row_bpb_se'))} | {a['rows']} |")
    if report["after"]["tasks"]:
        md += ["", "| task | acc after | acc before | n |", "|---|---|---|---|"]
        for t in report["after"]["tasks"]:
            a = report["after"]["tasks"][t]["acc"]
            b = before.get("tasks", {}).get(t, {}).get("acc")
            md.append(f"| {t} | {_fmt(a)} | {_fmt(b)} | {report['after']['tasks'][t]['n']} |")
    if opts.kld_cache:
        ka = report["after"]["kld"]
        kb = before.get("kld")
        md += ["", f"| kld vs cache K={ka['K']} | nats | clustered se | top-1 |", "|---|---|---|---|",
               f"| after | {_fmt(ka['mean_kld_nats'], 5)} | {_fmt(ka['clustered_se'], 5)} | "
               f"{_fmt(ka['top1_agreement'])} |"]
        if kb:
            md.append(f"| before | {_fmt(kb['mean_kld_nats'], 5)} | {_fmt(kb['clustered_se'], 5)} | "
                      f"{_fmt(kb['top1_agreement'])} |")
    if opts.chat_sanity:
        md += ["", "| chat sanity | after | before |", "|---|---|---|"]
        ca = report["after"]["chat"]
        cb = before.get("chat", {})
        for key in ("compliance", "truncated_rate", "refusal_rate", "task_refusal_rate", "ref_nll_nats"):
            md.append(f"| {key} | {_fmt(ca.get(key), 3)} | {_fmt(cb.get(key), 3)} |")
    if "corpus_sources" in report:
        md += ["", "Corpus rows per source: " + ", ".join(f"{k}={v}" for k, v in report["corpus_sources"].items())]
    return "\n".join(md) + "\n"


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}


def literal_prefix(text: str) -> str:
    """A prefix given as text, with the escapes \\n, \\t, \\r and \\\\ decoded and
    every other character kept as typed."""
    import re
    return re.sub(r"\\([ntr\\])", lambda m: _ESCAPES[m.group(1)], text)


def run_eval(opts: EvalOptions) -> int:
    """Returns 0 with both reports written, 2 on a refusal: before the
    load, a missing or unreadable input, ``--before`` without ``--adapter``
    or a positions map naming none of the reply rows; after it, a kld cache
    the student's tokenizer or head cannot score, or a chat instrument
    asked of a student without a template."""
    import mlx.core as mx
    # batch shapes differ per slice, item and conversation, so freed buffers
    # of many sizes would otherwise accumulate in MLX's cache
    mx.set_cache_limit(int(opts.cache_limit_gb * GB))
    try:
        slice_specs = _named(opts.slices)
        chat_specs = _named(opts.chat_slices)
        reply_specs = _named(opts.reply_slices)
        frame_kwargs = _frames.parse_render_kwargs(opts.frame_kwargs)
    except ValueError as e:
        log(f"[eval] refuse: {e}")
        return 2
    for name, path in [*slice_specs, *chat_specs, *reply_specs]:
        if not Path(path).expanduser().is_file():
            log(f"[eval] refuse: slice {name}: no file at {path}")
            return 2
    task_files = []
    if opts.tasks:
        td = Path(opts.tasks_dir or ".")
        names = [t.strip() for t in opts.tasks.split(",") if t.strip()]
        task_files = [td / f"{t}.jsonl" for t in names] + ([td / "gsm8k_shots.jsonl"] if "gsm8k" in names else [])
    for label, path in [("--student", opts.student), ("--adapter", opts.adapter), ("--cache", opts.cache),
                        ("--reply-positions", opts.reply_positions), ("--chat-sanity", opts.chat_sanity),
                        ("--chat-refs", opts.chat_refs), ("--teacher-bpb", opts.teacher_bpb),
                        ("--kld-cache", opts.kld_cache)] + [("--tasks", str(f)) for f in task_files]:
        if path and not Path(path).expanduser().exists():
            log(f"[eval] refuse: {label}: nothing at {path}")
            return 2
    if opts.before and not opts.adapter:
        log("[eval] refuse: --before scores the adapter disabled in process, so it needs --adapter")
        return 2
    if opts.bpb_prefix and opts.bpb_prefix.startswith("@") and opts.bpb_prefix[1:] not in _frames.FRAME_PREFIX_KINDS:
        log(f"[eval] refuse: --bpb-prefix {opts.bpb_prefix} names no frame, the frames are "
            + ", ".join("@" + k for k in _frames.FRAME_PREFIX_KINDS) + " (or give the prefix text)")
        return 2
    # every input is read before the load, so a malformed file is refused
    # in seconds and never after minutes of scoring
    tasks: dict = {}
    kld_manifest: dict = {}
    corpus_sources = None
    try:
        slices = {name: nfc(read_slice(Path(path).expanduser())) for name, path in slice_specs}
        if opts.tasks:
            td = Path(opts.tasks_dir or ".")
            for t in opts.tasks.split(","):
                t = t.strip()
                if not t:
                    continue
                items = read_jsonl(td / f"{t}.jsonl")
                items = check_gsm8k_items(td / f"{t}.jsonl", items) if t == "gsm8k" \
                    else check_mc_items(td / f"{t}.jsonl", items)
                if opts.task_limit:
                    items = items[:opts.task_limit]
                tasks[t] = {"items": items}
                if t == "gsm8k":
                    tasks[t]["shots"] = check_gsm8k_items(td / "gsm8k_shots.jsonl",
                                                          read_jsonl(td / "gsm8k_shots.jsonl"),
                                                          ("question", "answer"))[:8]
        chat_slices = {name: read_conversations(Path(path).expanduser()) for name, path in chat_specs}
        reply_slices = {name: read_jsonl(Path(path).expanduser()) for name, path in reply_specs}
        positions = trace_positions = None
        if opts.reply_positions:
            census = read_report(Path(opts.reply_positions).expanduser())
            positions = census.get("high_delta") or {}
            trace_positions = census.get("high_delta_trace") or {}
            # the content ranges are relative to the content start and
            # the trace ranges to the trace start, which the reply row
            # has under --reply-think alone, so the eval must score the
            # reply under the frame that measured the map
            measured = census.get("frame")
            if measured in ("reply", "reply-think") and (measured == "reply-think") != bool(opts.reply_think):
                log(f"[eval] refuse: --reply-positions {opts.reply_positions} was measured on a {measured} cache, "
                    f"{'pass' if measured == 'reply-think' else 'drop'} --reply-think to score the same positions")
                return 2
        chat_items = (check_keys(Path(opts.chat_sanity), read_jsonl(Path(opts.chat_sanity)), ("messages",))
                      if opts.chat_sanity else [])
        refs_before = chat_refs(Path(opts.chat_refs)) if opts.chat_refs else None
        teacher_bpb = teacher_bpb_map(Path(opts.teacher_bpb)) if opts.teacher_bpb else None
        if opts.kld_cache:
            kld_manifest = read_report(Path(opts.kld_cache) / "manifest.json")
        if opts.cache:
            corpus_sources = cache_sources(Path(opts.cache))
    except UnreadableInput as e:
        log(f"[eval] refuse: unreadable input {e}")
        return 2
    if positions is not None and reply_slices and not ((set(positions) | set(trace_positions or {}))
                                                        & reply_row_ids(reply_slices)):
        log(f"[eval] refuse: --reply-positions {opts.reply_positions} names none of the reply-slice rows "
            "(a census run without --corpus keys its positions by cache row, not by corpus id)")
        return 2
    # the kld cache check needs the student's tokenizer and head width, both
    # readable from its header, so it runs before the load; a student whose
    # width the header does not give is checked after it
    kld_width = student_width(opts.student) if opts.kld_cache else None
    if opts.kld_cache and kld_width is not None:
        try:
            tok0 = _tokens.load_tokenizer(opts.student)
        except (OSError, ValueError) as e:
            log(f"[eval] refuse: cannot load the student tokenizer from {opts.student}: {e}")
            return 2
        why = kld_cache_refusal(Path(opts.kld_cache), kld_manifest, tok0, kld_width)
        if why:
            log(f"[eval] refuse: --kld-cache {opts.kld_cache}: {why}; the sparse KL is defined on a "
                "same-tokenizer cache only")
            return 2
    report: dict = {"student": opts.student, "adapter": opts.adapter, "slices": {}, "decontam": {},
                    "bpb_prefix": opts.bpb_prefix}
    contaminated = set()
    if opts.cache and slices:
        fractions = _eval.decontam_fractions({name: text.encode("utf-8") for name, text in slices.items()},
                                             corpus_texts(Path(opts.cache)))
        for name, f in fractions.items():
            report["decontam"][name] = f
            if f > opts.decontam_threshold:
                contaminated.add(name)
                log(f"[eval] warn: {name}: {f:.4f} of 64-byte windows in the cached corpus, bpb gate void")
            else:
                log(f"[eval] {name}: decontam fraction {f:.5f}")
    model, _cfg, tokenizer, _kind = load_student(opts.student, opts.adapter, opts.hf_source)
    import mlx.core as mx
    head = head_spec_from_model(getattr(model, "language_model", model))
    gap = head_parity_gap(model, head, mx.arange(1, 9)[None])
    if gap > HEAD_PARITY_TOL:
        log(f"[eval] refuse: the head does not reproduce the student's own logits (relative gap {gap:.3f}), "
            "the model changes its logits after the projection in a way the distill head does not carry")
        return 2
    if opts.kld_cache and kld_width is None:
        why = kld_cache_refusal(Path(opts.kld_cache), kld_manifest, tokenizer, head.V)
        if why:
            log(f"[eval] refuse: --kld-cache {opts.kld_cache}: {why}; the sparse KL is defined on a "
                "same-tokenizer cache only")
            return 2
    if (opts.chat_sanity or chat_specs) and not _frames.has_chat_template(tokenizer):
        # a checkpoint without its template renders every conversation as
        # plain text, and the chat numbers then measure another prompt format
        log(f"[eval] refuse: {opts.student} carries no chat template, drop --chat-sanity and --chat-slice")
        return 2
    inherit = ((kld_manifest.get("gmlx_distill") or {}).get("frame") or {}).get("render_kwargs")
    _frames.set_render_kwargs(tokenizer, _frames.resolve_render_kwargs(
        tokenizer, inherit=inherit, override=frame_kwargs))
    model.eval()
    if positions is not None:
        report["reply_positions"] = opts.reply_positions
        log(f"[eval] reply slices restricted to the high-delta positions of {len(positions)} rows")
    report["after"] = run_arm(model, tokenizer, opts, slices, tasks, chat_slices, reply_slices, positions,
                              trace_positions, head=head)
    if opts.before:
        with adapter_disabled(model):
            report["before"] = run_arm(model, tokenizer, opts, slices, tasks, chat_slices, reply_slices,
                                       positions, trace_positions, head=head, label="before")
    if opts.chat_sanity:
        items = chat_items
        refs = refs_before
        if refs is not None and opts.before:
            log(f"[eval] --chat-refs {opts.chat_refs} ignored: --before anchors the drift score on the "
                "adapter-off replies")
        elif refs is not None:
            log(f"[eval] chat drift references: {len(refs)} replies from {opts.chat_refs}")
        if opts.before:
            t0 = time.perf_counter()
            with adapter_disabled(model):
                before = _eval.chat_sanity(model, tokenizer, items, max_tokens=opts.chat_max_tokens)
            before["wall_s"] = time.perf_counter() - t0
            report["before"]["chat"] = before
            refs = {r["id"]: r["reply"] for r in before["items"] if r["compliant"] and r["reply"].strip()}
            log(chat_line("before", before))
        t0 = time.perf_counter()
        after = _eval.chat_sanity(model, tokenizer, items, refs=refs, max_tokens=opts.chat_max_tokens)
        after["wall_s"] = time.perf_counter() - t0
        after["refs_source"] = "before" if opts.before else opts.chat_refs
        report["after"]["chat"] = after
        log(chat_line("after", after))
    if teacher_bpb is not None:
        report["teacher_bpb"] = teacher_bpb
    report["contaminated_slices"] = sorted(contaminated)
    if corpus_sources is not None:
        report["corpus_sources"] = corpus_sources
    write_json_atomic(Path(opts.json), report)
    md = report_markdown(opts, report, slices, chat_slices, reply_slices, contaminated)
    write_bytes_atomic(Path(opts.md), md.encode("utf-8"))
    print(md, end="")
    return 0
