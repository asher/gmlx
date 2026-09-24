"""Rejection filter for a generated corpus: keep the rows a student should
learn from, drop the rest with a reason, and stamp the filter version into
the corpus sidecar.

Checks run in a fixed order and the first failure names the reason:

    length   the reply did not reach its end of turn (finish_reason is not stop)
    budget   the thinking budget cut the trace (the gen block says so)
    empty    the answer has fewer than min_words units, a unit being a
             whitespace-separated word or one ideograph or kana character
    marker   a template marker string leaked into the reply or its trace
    repeat   repeated n-grams cover more than max_repeat of the reply or
             max_trace_repeat of the trace, or one line with a letter or
             digit repeats more than max_line_repeats times in a row
    ascii    non-ASCII characters above max_non_ascii (off unless set)
    tokens   the reply is longer than max_reply_tokens completion tokens
    verify   an external command rejected the row

The verify command reads the candidate rows as jsonl on stdin and prints
one line per row in order, ``ok`` or a reason word, so a task-specific
checker (a test runner, a SQL executor) plugs in without the filter
knowing the task. ``context`` rebuilds each kept row with a context the
replier never saw on the teacher's side, which is how an on-policy round
is prepared: a student answers the prompts without the context, and the
teacher then scores those replies with the context in its prompt."""
from __future__ import annotations

import hashlib
import json
import re
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .constants import log
from .corpus import message_list
from .format import write_json_atomic
from .gen import DEFAULT_CONTEXT_FORMAT, apply_context, context_format_error

FILTER_VERSION = "3"
MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<start_of_turn>", "<end_of_turn>", "<turn|>",
           "<|eot_id|>", "<|start_header_id|>", "<|channel|>", "<|message|>", "<|return|>", "<|user|>",
           "<|assistant|>", "<think>", "</think>")
REASONS = ("length", "budget", "empty", "marker", "repeat", "ascii", "tokens", "verify")
# the trace is delimited by the think tags, so those two are not a leak there
TRACE_MARKERS = tuple(m for m in MARKERS if m not in ("<think>", "</think>"))


@dataclass
class FilterOptions:
    inputs: list[str]
    out: str
    report: str | None = None
    rejects: str | None = None
    min_words: int = 16
    ngram: int = 8
    max_repeat: float = 0.2
    max_trace_repeat: float = 0.5
    max_line_repeats: int = 2
    max_non_ascii: float | None = None
    max_reply_tokens: int | None = None
    keep_budget_hit: bool = False
    verify: str | None = None
    context: str | None = None
    context_format: str = DEFAULT_CONTEXT_FORMAT


# whitespace and CJK punctuation separate units; ideographs and kana are
# written without spaces, so each character is a unit of its own, while
# Hangul, Latin, digits and code keep their whitespace tokens
_SEP = re.compile("[\\s\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]+")
_CJK = re.compile("([\u2e80-\u2fdf\u3040-\u30ff\u3100-\u312f\u31a0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
                  "\uf900-\ufaff\U00020000-\U0003ffff])")


def _units(text: str) -> list[str]:
    """The units the word count and the repeat check run over: whitespace
    tokens, split further into single characters inside ideograph and
    kana runs."""
    return [p for tok in _SEP.split(text) for p in _CJK.split(tok) if p]


def word_count(text: str) -> int:
    return len(_units(text))


def repeat_fraction(text: str, n: int) -> float:
    """Fraction of the text's n-grams over its units (see _units) that
    repeat an earlier n-gram."""
    toks = _units(text)
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    c = Counter(grams)
    dup = sum(v - 1 for v in c.values())
    return dup / len(grams)


def max_line_run(text: str) -> int:
    """The longest run of one line repeated back to back, over lines that
    hold a letter or digit (closing brackets of nested code and JSON
    repeat without being a loop)."""
    run = best = 0
    prev = None
    for line in (ln.strip() for ln in text.splitlines()):
        if not line or not any(ch.isalnum() for ch in line):
            continue
        run = run + 1 if line == prev else 1
        best = max(best, run)
        prev = line
    return best


def reason(row: dict, opts: FilterOptions) -> str | None:
    """The first check the row fails, or None when it passes every local
    check (the verify command runs separately, over the survivors)."""
    g = row.get("gen") or {}
    reply = row["messages"][-1].get("content") or ""
    trace = row["messages"][-1].get("reasoning_content") or ""
    if g.get("finish_reason") != "stop":
        return "length"
    if g.get("budget_hit") and not opts.keep_budget_hit:
        return "budget"
    if word_count(reply) < opts.min_words:
        return "empty"
    if any(m in reply for m in MARKERS) or any(m in trace for m in TRACE_MARKERS):
        return "marker"
    for text, cap in ((reply, opts.max_repeat), (trace, opts.max_trace_repeat)):
        if text and (repeat_fraction(text, opts.ngram) > cap or max_line_run(text) > opts.max_line_repeats):
            return "repeat"
    if opts.max_non_ascii is not None and reply:
        na = sum(1 for ch in reply if ord(ch) > 126) / len(reply)
        if na > opts.max_non_ascii:
            return "ascii"
    if opts.max_reply_tokens is not None and int(g.get("completion_tokens") or 0) > opts.max_reply_tokens:
        return "tokens"
    return None


def run_verify(command: str, rows: list[dict]) -> list[str]:
    """Run the verify command once over ``rows`` (jsonl on stdin) and return
    one verdict per row: ``ok`` or the reason word it printed."""
    if not rows:
        return []
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    proc = subprocess.run(command, shell=True, input=data, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"verify command exited {proc.returncode}: {tail}")
    verdicts = [ln.strip() for ln in proc.stdout.decode("utf-8", "replace").splitlines() if ln.strip()]
    if len(verdicts) != len(rows):
        raise RuntimeError(f"verify command printed {len(verdicts)} verdicts for {len(rows)} rows")
    return verdicts


def recontext_row(row: dict, context: str, fmt: str = DEFAULT_CONTEXT_FORMAT) -> dict:
    """The row with ``context`` on the teacher's side: ``messages`` gets
    the context applied to the final user turn and ``student_messages``
    keeps the prompt as given. The row must end on a user turn followed
    by the reply and must not already carry a student list."""
    msgs = row["messages"]
    if row.get("student_messages"):
        raise ValueError(f"row {row.get('id')} already carries student_messages")
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant" or msgs[-2].get("role") != "user":
        raise ValueError(f"row {row.get('id')} must end on a user turn followed by the reply")
    out = dict(row)
    out["messages"] = apply_context(msgs[:-1], context, fmt) + [dict(msgs[-1])]
    out["student_messages"] = msgs
    gen = dict(row.get("gen") or {})
    gen["context"] = True
    out["gen"] = gen
    return out


def _read_rows(path: Path) -> list[dict]:
    """The corpus rows of a jsonl file, or a ValueError naming the file
    when it cannot be read or is not UTF-8, or naming the first line that
    is not JSON or not a row (an object with a non-empty list of message
    objects whose contents are strings)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not UTF-8 (byte {e.start}), convert it") from None
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e.strerror or e}") from None
    rows = []
    for n, ln in enumerate(text.split("\n"), 1):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {n}: not JSON ({e.msg})") from None
        if (not isinstance(obj, dict) or not isinstance(obj.get("messages"), list) or not obj["messages"]
                or not all(isinstance(m, dict) for m in obj["messages"])):
            raise ValueError(f"{path} line {n}: not a corpus row (an object with a list of message objects)")
        message_list(obj, "messages", f"{path} line {n}")
        if obj.get("student_messages") is not None:
            message_list(obj, "student_messages", f"{path} line {n}")
        rows.append(obj)
    return rows


_GEN_KEYS = ("gen_version", "model", "served_model_id", "sampling", "seed", "chat_template_kwargs", "thinking",
             "thinking_budget", "context", "context_format", "serve_args")
# recorded by a finished gen only; compared when both sidecars carry them
_GEN_KEYS_IF_PRESENT = ("instruction", "prefix_chars")


def _read_sidecar(path: Path) -> dict | None:
    """The input's sidecar, None when it has none, or a ValueError when the
    file is not a JSON object."""
    side = path.with_suffix(path.suffix + ".gen.json")
    if not side.exists():
        return None
    try:
        obj = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{side} is not a JSON object ({e})") from None
    if not isinstance(obj, dict):
        raise ValueError(f"{side} is not a JSON object")
    return obj


def _settings_diff(a: dict, b: dict) -> list[str]:
    """The keys of _GEN_KEYS on which two sidecars disagree; a model or
    context named two ways for one file agrees, and so does a served
    model id one side does not know (a rerun that contacted no server)."""
    from .gen import same_name
    return [k for k in _GEN_KEYS if a.get(k) != b.get(k)
            and not (k in ("model", "context") and same_name(a.get(k), b.get(k)))
            and not (k == "served_model_id" and None in (a.get(k), b.get(k)))] + \
        [k for k in _GEN_KEYS_IF_PRESENT if k in a and k in b and a[k] != b[k]]


def run_filter(opts: FilterOptions) -> int:
    """Filter the inputs in order into ``out`` with its sidecar. Returns 0,
    or 2 when an input is missing or its sidecar unreadable, the inputs
    were generated with other settings than each other or filtered
    differently, no row carries a gen block, a row cannot take the
    context, or the verify command fails."""
    inputs = [Path(p).expanduser() for p in opts.inputs]
    for p in inputs:
        if not p.is_file():
            print(f"[filter] refuse: no such file: {p}", file=sys.stderr)
            return 2
    try:
        sides = [_read_sidecar(p) for p in inputs]
    except ValueError as e:
        print(f"[filter] refuse: {e}", file=sys.stderr)
        return 2
    first_i = next((i for i, s in enumerate(sides) if s is not None), None)
    first = sides[first_i] if first_i is not None else None
    first_path = inputs[first_i] if first_i is not None else None
    # a sidecar without a served model id agrees with any, so the id is
    # checked against the first sidecar that names one
    served: tuple[str, Path] | None = None
    for p, s in zip(inputs, sides):
        sid = s.get("served_model_id") if s is not None else None
        if sid is not None and served is not None and sid != served[0]:
            print(f"[filter] refuse: {p} was generated with other settings than {served[1]} (served_model_id), "
                  "filter each file on its own", file=sys.stderr)
            return 2
        if sid is not None and served is None:
            served = (sid, p)
        if s is not None and first is not None and _settings_diff(s, first):
            diff = ", ".join(_settings_diff(s, first))
            print(f"[filter] refuse: {p} was generated with other settings than {first_path} ({diff}), "
                  "filter each file on its own", file=sys.stderr)
            return 2
        if s is not None and first is not None and s.get("filter_version") != first.get("filter_version"):
            # the output sidecar records one filter history for every row
            print(f"[filter] refuse: {p} was filtered differently than {first_path} "
                  f"({s.get('filter_version')!r} vs {first.get('filter_version')!r}), filter the unfiltered "
                  "inputs first or join unfiltered files", file=sys.stderr)
            return 2
        if s is not None and first is not None and isinstance(s.get("filter"), dict) \
                and isinstance(first.get("filter"), dict):
            pa = dict(s["filter"].get("params") or {}, verify=s["filter"].get("verify"))
            pb = dict(first["filter"].get("params") or {}, verify=first["filter"].get("verify"))
            diff = ", ".join(sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)))
            if diff:
                print(f"[filter] refuse: {p} was filtered with other settings than {first_path} ({diff}), "
                      "filter the unfiltered inputs together", file=sys.stderr)
                return 2
    for p, s in zip(inputs, sides):
        if s is None and first is not None:
            # the output sidecar labels every row it holds
            print(f"[filter] warn: {p} has no sidecar, its rows are written under the settings of {first_path}",
                  file=sys.stderr)
    if opts.context and not Path(opts.context).expanduser().is_file():
        print(f"[filter] refuse: no such file: {opts.context}", file=sys.stderr)
        return 2
    fmt_err = context_format_error(opts.context_format) if opts.context else None
    if fmt_err:
        print(f"[filter] refuse: {fmt_err}", file=sys.stderr)
        return 2
    out = Path(opts.out).expanduser()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        context = Path(opts.context).expanduser().read_text(encoding="utf-8") if opts.context else None
    except UnicodeDecodeError as e:
        print(f"[filter] refuse: {opts.context}: not UTF-8 (byte {e.start}), convert it", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"[filter] refuse: {e}", file=sys.stderr)
        return 2
    if context is not None and not context.strip():
        print(f"[filter] refuse: context file {opts.context} is blank", file=sys.stderr)
        return 2
    counts: Counter = Counter()
    rejects: list[dict] = []
    survivors: list[dict] = []
    seen: dict[str, Path] = {}
    for p in inputs:
        try:
            rows = _read_rows(p)
        except ValueError as e:
            print(f"[filter] refuse: {e}", file=sys.stderr)
            return 2
        if rows and not any(isinstance(row.get("gen"), dict) for row in rows):
            # every row would drop as length and the output would be empty
            print(f"[filter] refuse: no row of {p} carries a gen block, this is not a generated corpus",
                  file=sys.stderr)
            return 2
        for row in rows:
            rid = row.get("id")
            if rid is not None:
                # rows are keyed by id downstream (a gen resume, census pairs)
                if str(rid) in seen:
                    print(f"[filter] refuse: id {str(rid)!r} appears in {p} and in {seen[str(rid)]}, "
                          "give the inputs distinct ids", file=sys.stderr)
                    return 2
                seen[str(rid)] = p
            why = reason(row, opts)
            if why:
                counts[why] += 1
                rejects.append({"id": row.get("id"), "reason": why})
                continue
            survivors.append(row)
    if opts.verify:
        try:
            verdicts = run_verify(opts.verify, survivors)
        except RuntimeError as e:
            print(f"[filter] error: {e}", file=sys.stderr)
            return 2
        kept_rows = []
        for row, v in zip(survivors, verdicts):
            if v == "ok":
                kept_rows.append(row)
            else:
                counts["verify"] += 1
                rejects.append({"id": row.get("id"), "reason": "verify", "detail": v})
        survivors = kept_rows
    if context is not None:
        try:
            survivors = [recontext_row(r, context, opts.context_format) for r in survivors]
        except ValueError as e:
            print(f"[filter] refuse: {e}", file=sys.stderr)
            return 2
    unenforced = sum(1 for r in survivors if (r.get("gen") or {}).get("budget_unenforced"))
    if unenforced:
        log(f"[filter] warn: {unenforced} kept rows carry budget_unenforced, a trace the server never cut "
            "(a drafter drops the budget when requests batch)")
    # written beside the final name and moved into place, so a failed
    # write leaves the corpus that was there
    tmp = out.with_name(out.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as ofh:
            for row in survivors:
                ofh.write(json.dumps(row, ensure_ascii=False) + "\n")
            ofh.flush()
            os.fsync(ofh.fileno())
        os.replace(tmp, out)
    except OSError as e:
        print(f"[filter] refuse: cannot write {out}: {e}", file=sys.stderr)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return 2
    kept = len(survivors)
    sidecar: dict = dict(first) if first is not None else {"gen_version": None}
    if served is not None:
        sidecar["served_model_id"] = served[0]
    prev = sidecar.get("filter_version")
    sidecar["filter_version"] = f"{prev}+{FILTER_VERSION}" if prev else FILTER_VERSION
    params = {k: getattr(opts, k) for k in ("min_words", "ngram", "max_repeat", "max_trace_repeat",
                                            "max_line_repeats", "max_non_ascii", "max_reply_tokens",
                                            "keep_budget_hit")}
    if isinstance(sidecar.get("filter"), dict):
        # every earlier pass stays on record, oldest first
        sidecar["filter_history"] = [*(sidecar.get("filter_history") or []), sidecar["filter"]]
    sidecar["filter"] = {"params": params, "verify": opts.verify, "kept": kept, "dropped": dict(counts),
                         "budget_unenforced": unenforced, "inputs": [str(p) for p in inputs]}
    joined = [(p, s) for p, s in zip(inputs, sides) if s is not None]
    if joined:
        # the run block keeps what gen measured (a join adds the inputs'
        # counts up), and the survivors' totals sit beside it under kept;
        # an input whose sidecar holds no run block (a run cut short) is
        # counted from its rows
        from .gen import row_totals
        runs = [dict(s["run"]) if isinstance(s.get("run"), dict) else {**row_totals(p), "wall_s": 0.0}
                for p, s in joined]
        run = dict(runs[0])
        if len(joined) > 1:
            for k in ("completed", "failed", "generated_tokens", "stops", "budget_hits", "budget_unenforced"):
                if any(k in r for r in runs):
                    run[k] = sum(int(r.get(k) or 0) for r in runs)
            run["wall_s"] = sum(float(r.get("wall_s") or 0.0) for r in runs)
            if "stops" in run and "completed" in run:
                run["stop_fraction"] = run["stops"] / max(run["completed"], 1)
            if any("longest_stopped_reply_tokens" in r for r in runs):
                run["longest_stopped_reply_tokens"] = max(int(r.get("longest_stopped_reply_tokens") or 0)
                                                          for r in runs)
            run.pop("tok_s_aggregate", None)     # one rate cannot describe two runs
        run["kept"] = row_totals(out)
        sidecar["run"] = run
    if len(joined) > 1:
        # one sidecar describes every input: the prompt fields list each
        # input's
        sidecar["prompts"] = sum(int(s.get("prompts") or 0) for _, s in joined)
        sidecar["prompt_source"] = [s.get("prompt_source") for _, s in joined]
        sidecar["prompt_set_sha256"] = hashlib.sha256(
            "\n".join(str(s.get("prompt_set_sha256")) for _, s in joined).encode()).hexdigest()
        sidecar["joined"] = [str(p) for p, _ in joined]
    if opts.context is not None:
        ctx_name = str(Path(opts.context).expanduser().absolute())
        sidecar.update({"context": ctx_name, "shared_context": ctx_name, "context_format": opts.context_format,
                        "recontext_from": [str(p) for p in inputs], "prompts": kept})
    write_json_atomic(out.with_suffix(out.suffix + ".gen.json"), sidecar)
    summary = {"inputs": [str(p) for p in inputs], "out": str(out), "kept": kept, "dropped": dict(counts),
               "budget_unenforced": unenforced, "filter_version": sidecar["filter_version"]}
    log(f"[filter] kept {kept}, dropped {dict(counts)} -> {out}")
    if opts.report:
        write_json_atomic(Path(opts.report).expanduser(), summary)
    if opts.rejects:
        rp = Path(opts.rejects).expanduser()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejects), encoding="utf-8")
    return 0
