"""The evaluator driver: a student with an optional adapter scored on
held-out bits per byte, chat and reply slices, sparse KL against a cache,
downstream tasks and the chat sanity set, before and after the adapter in
one process, with a Markdown and a JSON report. ``run_eval`` is what
``gmlx distill eval`` calls."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gmlx.load.tokenizer import token_bytes

from . import eval as _eval
from . import frames as _frames
from .constants import GB, log
from .corpus import nfc
from .data import CacheReader
from .format import read_json, write_json_atomic
from .student import adapter_disabled
from .trainer import load_student


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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_texts(cache: Path):
    reader = CacheReader(cache, keep=1)
    for i in range(reader.n_shards):
        for t in reader.shard(i)["_texts"]:
            yield t


def _named(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"expected name=path, got {spec!r}")
        out.append((name, path))
    return out


def run_arm(model, tokenizer, opts: EvalOptions, slices: dict[str, str], tasks: dict,
            chat_slices: dict | None = None, reply_slices: dict | None = None,
            positions: dict | None = None) -> dict:
    """Every instrument on the loaded weights as they are. ``positions``,
    a census high_delta map, restricts every reply slice to the byte
    ranges it names."""
    res: dict = {"bpb": {}, "tasks": {}, "chat_bpb": {}, "reply_bpb": {}}
    for name, convs in (chat_slices or {}).items():
        t0 = time.perf_counter()
        model.eval()
        r = _eval.chat_slice_nll(model, tokenizer, convs, max_len=opts.chat_max_len,
                                 batch_tokens=opts.batch_size * opts.max_len, per_turn=opts.chat_per_turn)
        r["wall_s"] = time.perf_counter() - t0
        res["chat_bpb"][name] = r
        log(f"[eval] {name}: assistant-turn bpb {r['bpb']:.4f} ({r['nll_per_token']:.4f} nats/token) over "
            f"{r['rows']} {'turns' if opts.chat_per_turn else 'conversations'}, {r['dropped']} dropped "
            f"({r['wall_s']:.0f}s)")
    for name, rows in (reply_slices or {}).items():
        t0 = time.perf_counter()
        model.eval()
        r = _eval.reply_slice_nll(model, tokenizer, rows, max_len=opts.chat_max_len,
                                  batch_tokens=opts.batch_size * opts.max_len, positions=positions,
                                  reason_target=opts.reply_think)
        r["wall_s"] = time.perf_counter() - t0
        res["reply_bpb"][name] = r
        log(f"[eval] {name}: reply bpb {r['bpb']:.4f} ({r['nll_per_token']:.4f} nats/token) over "
            f"{r['rows']} rows{' at the census positions' if positions is not None else ''}, "
            f"{r['dropped']} dropped ({r['wall_s']:.0f}s)")
    if opts.kld_cache:
        t0 = time.perf_counter()
        model.eval()
        reader = CacheReader(Path(opts.kld_cache), keep=2)
        k = _eval.cache_kld(model, reader, max_rows=opts.kld_rows, tokenizer=tokenizer)
        k["wall_s"] = time.perf_counter() - t0
        res["kld"] = k
        log(f"[eval] kld vs cache K={k['K']}: {k['mean_kld_nats']:.5f} nats se {k['clustered_se']:.5f} "
            f"top-1 {k['top1_agreement']:.4f} over {k['rows']} rows, {k['rerendered_rows']} re-rendered "
            f"through the student template ({k['wall_s']:.0f}s)")
    tb = token_bytes(tokenizer)
    prefix = None
    if opts.bpb_prefix:
        prefix = _frames.frame_prefix(tokenizer, opts.bpb_prefix[1:]) if opts.bpb_prefix.startswith("@") \
            else opts.bpb_prefix.encode().decode("unicode_escape")
    res["bpb_prefix_text"] = prefix
    for name, text in slices.items():
        t0 = time.perf_counter()
        r = _eval.bits_per_byte(model, tokenizer, text, max_len=opts.max_len, batch_size=opts.batch_size, tb=tb,
                                window_prefix=prefix)
        r["wall_s"] = time.perf_counter() - t0
        res["bpb"][name] = r
        log(f"[eval] {name}: bpb {r['bpb']:.4f} over {r['bytes']} bytes ({r['wall_s']:.0f}s)")
    model.eval()
    for tname, items in tasks.items():
        t0 = time.perf_counter()
        if tname == "gsm8k":
            per = _eval.score_gsm8k(model, tokenizer, items["items"], items["shots"],
                                    max_tokens=opts.gsm8k_max_tokens)
        else:
            per = _eval.score_multiple_choice(model, tokenizer, items["items"])
        acc = float(np.mean([p["correct"] for p in per])) if per else float("nan")
        res["tasks"][tname] = {"acc": acc, "n": len(per), "items": per, "wall_s": time.perf_counter() - t0}
        log(f"[eval] {tname}: acc {acc:.4f} on {len(per)} items ({res['tasks'][tname]['wall_s']:.0f}s)")
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
            md.append(f"| {name} | {a:.4f} | {_fmt(b)} | {_fmt(t)} | {_fmt(d, 5)} | "
                      f"{'void' if name in contaminated else 'ok'} |")
    if chat_slices:
        md += ["", "| chat slice | bpb after | bpb before | se | conversations |", "|---|---|---|---|---|"]
        for name in chat_slices:
            a = report["after"]["chat_bpb"][name]
            b = before.get("chat_bpb", {}).get(name, {}).get("bpb")
            md.append(f"| {name} | {a['bpb']:.4f} | {_fmt(b)} | {_fmt(a.get('row_bpb_se'))} | {a['rows']} |")
    if reply_slices:
        md += ["", "| reply slice | bpb after | bpb before | nats/token after | se | rows |",
               "|---|---|---|---|---|---|"]
        for name in reply_slices:
            a = report["after"]["reply_bpb"][name]
            b = before.get("reply_bpb", {}).get(name, {}).get("bpb")
            md.append(f"| {name} | {a['bpb']:.4f} | {_fmt(b)} | {a['nll_per_token']:.4f} | "
                      f"{_fmt(a.get('row_bpb_se'))} | {a['rows']} |")
    if report["after"]["tasks"]:
        md += ["", "| task | acc after | acc before | n |", "|---|---|---|---|"]
        for t in report["after"]["tasks"]:
            a = report["after"]["tasks"][t]["acc"]
            b = before.get("tasks", {}).get(t, {}).get("acc")
            md.append(f"| {t} | {a:.4f} | {_fmt(b)} | {report['after']['tasks'][t]['n']} |")
    if opts.kld_cache:
        ka = report["after"]["kld"]
        kb = before.get("kld")
        md += ["", f"| kld vs cache K={ka['K']} | nats | clustered se | top-1 |", "|---|---|---|---|",
               f"| after | {ka['mean_kld_nats']:.5f} | {ka['clustered_se']:.5f} | {ka['top1_agreement']:.4f} |"]
        if kb:
            md.append(f"| before | {kb['mean_kld_nats']:.5f} | {kb['clustered_se']:.5f} | {kb['top1_agreement']:.4f} |")
    if opts.chat_sanity:
        md += ["", "| chat sanity | after | before |", "|---|---|---|"]
        ca = report["after"]["chat"]
        cb = before.get("chat", {})
        for key in ("compliance", "truncated_rate", "refusal_rate", "task_refusal_rate", "ref_nll_nats"):
            md.append(f"| {key} | {_fmt(ca.get(key), 3)} | {_fmt(cb.get(key), 3)} |")
    if "corpus_sources" in report:
        md += ["", "Corpus rows per source: " + ", ".join(f"{k}={v}" for k, v in report["corpus_sources"].items())]
    return "\n".join(md) + "\n"


def run_eval(opts: EvalOptions) -> int:
    """Returns 0 with both reports written, 2 on a refusal before the load
    or when a chat instrument is asked of a student without a template."""
    import mlx.core as mx
    # batch shapes differ per slice, item and conversation, so freed buffers
    # of many sizes would otherwise accumulate in MLX's cache
    mx.set_cache_limit(int(opts.cache_limit_gb * GB))
    try:
        slice_specs = _named(opts.slices)
        chat_specs = _named(opts.chat_slices)
        reply_specs = _named(opts.reply_slices)
    except ValueError as e:
        log(f"[eval] refuse: {e}")
        return 2
    slices = {name: nfc(Path(path).expanduser().read_text(encoding="utf-8", errors="replace"))
              for name, path in slice_specs}
    report: dict = {"student": opts.student, "adapter": opts.adapter, "slices": {}, "decontam": {},
                    "bpb_prefix": opts.bpb_prefix}
    contaminated = set()
    if opts.cache and slices:
        for name, text in slices.items():
            f = _eval.decontam_fraction(text.encode("utf-8"), corpus_texts(Path(opts.cache)))
            report["decontam"][name] = f
            if f > opts.decontam_threshold:
                contaminated.add(name)
                log(f"[eval] {name}: {f:.4f} of 64-byte windows in the cached corpus, bpb gate void")
            else:
                log(f"[eval] {name}: decontam fraction {f:.5f}")
    tasks: dict = {}
    if opts.tasks:
        td = Path(opts.tasks_dir or ".")
        for t in opts.tasks.split(","):
            t = t.strip()
            if not t:
                continue
            items = read_jsonl(td / f"{t}.jsonl")
            if opts.task_limit:
                items = items[:opts.task_limit]
            tasks[t] = {"items": items}
            if t == "gsm8k":
                tasks[t]["shots"] = read_jsonl(td / "gsm8k_shots.jsonl")[:8]
    model, _cfg, tokenizer, _kind = load_student(opts.student, opts.adapter, opts.hf_source)
    if (opts.chat_sanity or chat_specs) and not _frames.has_chat_template(tokenizer):
        # a checkpoint without its template renders every conversation as
        # plain text, and the chat numbers then measure another prompt format
        log(f"[eval] refuse: {opts.student} carries no chat template, drop --chat-sanity and --chat-slice")
        return 2
    inherit = None
    if opts.kld_cache:
        inherit = ((read_json(Path(opts.kld_cache) / "manifest.json").get("gmlx_distill") or {})
                   .get("frame") or {}).get("render_kwargs")
    _frames.set_render_kwargs(tokenizer, _frames.resolve_render_kwargs(
        tokenizer, inherit=inherit, override=_frames.parse_render_kwargs(opts.frame_kwargs)))
    model.eval()
    chat_slices = {name: [json.loads(line)["messages"] for line in
                          Path(path).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
                   for name, path in chat_specs}
    reply_slices = {name: read_jsonl(Path(path).expanduser()) for name, path in reply_specs}
    positions = None
    if opts.reply_positions:
        positions = read_json(Path(opts.reply_positions).expanduser()).get("high_delta") or {}
        report["reply_positions"] = opts.reply_positions
        log(f"[eval] reply slices restricted to the census positions of {len(positions)} rows")
    report["after"] = run_arm(model, tokenizer, opts, slices, tasks, chat_slices, reply_slices, positions)
    if opts.before and opts.adapter:
        with adapter_disabled(model):
            report["before"] = run_arm(model, tokenizer, opts, slices, tasks, chat_slices, reply_slices,
                                       positions)
    if opts.chat_sanity:
        items = read_jsonl(Path(opts.chat_sanity))
        refs = None
        if opts.chat_refs:
            ref_items = (read_json(Path(opts.chat_refs)).get("after") or {}).get("chat", {}).get("items", [])
            refs = {r["id"]: r["reply"] for r in ref_items if r["compliant"] and r["reply"].strip()}
            log(f"[eval] chat drift references: {len(refs)} replies from {opts.chat_refs}")
        if opts.before and opts.adapter:
            t0 = time.perf_counter()
            with adapter_disabled(model):
                before = _eval.chat_sanity(model, tokenizer, items, max_tokens=opts.chat_max_tokens)
            before["wall_s"] = time.perf_counter() - t0
            report["before"]["chat"] = before
            refs = {r["id"]: r["reply"] for r in before["items"] if r["compliant"] and r["reply"].strip()}
            log(f"[eval] chat before: compliance {before['compliance']:.3f} truncated "
                f"{before['truncated_rate']:.3f} refusal {before['refusal_rate']} "
                f"task refusal {before['task_refusal_rate']} ({before['wall_s']:.0f}s)")
        t0 = time.perf_counter()
        after = _eval.chat_sanity(model, tokenizer, items, refs=refs, max_tokens=opts.chat_max_tokens)
        after["wall_s"] = time.perf_counter() - t0
        report["after"]["chat"] = after
        log(f"[eval] chat after: compliance {after['compliance']:.3f} truncated {after['truncated_rate']:.3f} "
            f"refusal {after['refusal_rate']} task refusal {after['task_refusal_rate']} "
            f"ref nll {after['ref_nll_nats']} ({after['wall_s']:.0f}s)")
    if opts.teacher_bpb:
        report["teacher_bpb"] = read_json(Path(opts.teacher_bpb))
    report["contaminated_slices"] = sorted(contaminated)
    if opts.cache:
        reader = CacheReader(Path(opts.cache), keep=1)
        counts: dict[str, int] = {}
        for r in reader.rows_meta:
            counts[r.get("source", "human")] = counts.get(r.get("source", "human"), 0) + 1
        report["corpus_sources"] = counts
    write_json_atomic(Path(opts.json), report)
    md = report_markdown(opts, report, slices, chat_slices, reply_slices, contaminated)
    Path(opts.md).write_text(md, encoding="utf-8")
    print(md, end="")
    return 0
