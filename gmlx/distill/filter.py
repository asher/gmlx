"""Rejection filter for a generated corpus: keep the rows a student should
learn from, drop the rest with a reason, and stamp the filter version into
the corpus sidecar.

Checks run in a fixed order and the first failure names the reason:

    length   the reply did not reach its end of turn (finish_reason is not stop)
    budget   the thinking budget cut the trace (the gen block says so)
    empty    the answer has fewer than min_words whitespace-separated words
    marker   a template marker string leaked into the reply
    repeat   repeated n-grams cover more than max_repeat of the reply, or one
             line repeats more than max_line_repeats times in a row
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

import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .constants import log
from .format import write_json_atomic
from .gen import DEFAULT_CONTEXT_FORMAT, apply_context

FILTER_VERSION = "2"
MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<start_of_turn>", "<end_of_turn>", "<turn|>",
           "<|eot_id|>", "<|start_header_id|>", "<|channel|>", "<|message|>", "<|return|>", "<|user|>",
           "<|assistant|>", "<think>", "</think>")
REASONS = ("length", "budget", "empty", "marker", "repeat", "ascii", "tokens", "verify")


@dataclass
class FilterOptions:
    inputs: list[str]
    out: str
    report: str | None = None
    rejects: str | None = None
    min_words: int = 16
    ngram: int = 8
    max_repeat: float = 0.2
    max_line_repeats: int = 2
    max_non_ascii: float | None = None
    max_reply_tokens: int | None = None
    keep_budget_hit: bool = False
    verify: str | None = None
    context: str | None = None
    context_format: str = DEFAULT_CONTEXT_FORMAT


def repeat_fraction(text: str, n: int) -> float:
    """Fraction of the reply's whitespace-token n-grams that repeat an
    earlier n-gram."""
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    c = Counter(grams)
    dup = sum(v - 1 for v in c.values())
    return dup / len(grams)


def max_line_run(text: str) -> int:
    run = best = 0
    prev = None
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
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
    if g.get("finish_reason") != "stop":
        return "length"
    if g.get("budget_hit") and not opts.keep_budget_hit:
        return "budget"
    if len(reply.split()) < opts.min_words:
        return "empty"
    if any(m in reply for m in MARKERS):
        return "marker"
    if repeat_fraction(reply, opts.ngram) > opts.max_repeat or max_line_run(reply) > opts.max_line_repeats:
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
    """The corpus rows of a jsonl file, or a ValueError naming the first
    line that is not JSON or not a row (an object with a non-empty list
    of message objects)."""
    rows = []
    for n, ln in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {n}: not JSON ({e.msg})") from None
        if (not isinstance(obj, dict) or not isinstance(obj.get("messages"), list) or not obj["messages"]
                or not all(isinstance(m, dict) for m in obj["messages"])):
            raise ValueError(f"{path} line {n}: not a corpus row (an object with a list of message objects)")
        rows.append(obj)
    return rows


_GEN_KEYS = ("model", "sampling", "seed", "chat_template_kwargs", "thinking", "thinking_budget",
             "context", "context_format")


def _read_sidecar(path: Path) -> dict | None:
    side = path.with_suffix(path.suffix + ".gen.json")
    return json.loads(side.read_text(encoding="utf-8")) if side.exists() else None


def _gen_settings(side: dict) -> dict:
    """The generator settings every row of a file shares, which two inputs
    must agree on before their rows are joined under one sidecar."""
    return {k: side.get(k) for k in _GEN_KEYS}


def _settings_diff(a: dict, b: dict) -> list[str]:
    """The keys of _GEN_KEYS on which two sidecars disagree; a model or
    context named two ways for one file agrees."""
    from .gen import same_name
    return [k for k in _GEN_KEYS if a.get(k) != b.get(k)
            and not (k in ("model", "context") and same_name(a.get(k), b.get(k)))]


def run_filter(opts: FilterOptions) -> int:
    """Filter the inputs in order into ``out`` with its sidecar. Returns 0,
    or 2 when an input is missing, the inputs were generated with other
    settings than each other, a row cannot take the context, or the
    verify command fails."""
    inputs = [Path(p).expanduser() for p in opts.inputs]
    for p in inputs:
        if not p.is_file():
            print(f"[filter] refuse: no such file: {p}", file=sys.stderr)
            return 2
    sides = [_read_sidecar(p) for p in inputs]
    first_i = next((i for i, s in enumerate(sides) if s is not None), None)
    first = sides[first_i] if first_i is not None else None
    first_path = inputs[first_i] if first_i is not None else None
    for p, s in zip(inputs, sides):
        if s is not None and first is not None and _settings_diff(s, first):
            diff = ", ".join(_settings_diff(s, first))
            print(f"[filter] refuse: {p} was generated with other settings than {first_path} ({diff}), "
                  "filter each file on its own", file=sys.stderr)
            return 2
    for p, s in zip(inputs, sides):
        if s is None and first is not None:
            # the output sidecar labels every row it holds
            print(f"[filter] warn: {p} has no sidecar, its rows are written under the settings of {first_path}",
                  file=sys.stderr)
    if opts.context and not Path(opts.context).expanduser().is_file():
        print(f"[filter] refuse: no such file: {opts.context}", file=sys.stderr)
        return 2
    out = Path(opts.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    context = Path(opts.context).expanduser().read_text(encoding="utf-8") if opts.context else None
    counts: Counter = Counter()
    rejects: list[dict] = []
    survivors: list[dict] = []
    for p in inputs:
        try:
            rows = _read_rows(p)
        except ValueError as e:
            print(f"[filter] refuse: {e}", file=sys.stderr)
            return 2
        for row in rows:
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
    prev = sidecar.get("filter_version")
    sidecar["filter_version"] = f"{prev}+{FILTER_VERSION}" if prev else FILTER_VERSION
    params = {k: getattr(opts, k) for k in ("min_words", "ngram", "max_repeat", "max_line_repeats",
                                            "max_non_ascii", "max_reply_tokens", "keep_budget_hit")}
    sidecar["filter"] = {"params": params, "verify": opts.verify, "kept": kept, "dropped": dict(counts),
                         "inputs": [str(p) for p in inputs]}
    if opts.context is not None:
        ctx_name = str(Path(opts.context).expanduser().absolute())
        sidecar.update({"context": ctx_name, "shared_context": ctx_name, "context_format": opts.context_format,
                        "recontext_from": [str(p) for p in inputs], "prompts": kept})
    write_json_atomic(out.with_suffix(out.suffix + ".gen.json"), sidecar)
    summary = {"inputs": [str(p) for p in inputs], "out": str(out), "kept": kept, "dropped": dict(counts),
               "filter_version": sidecar["filter_version"]}
    log(f"[filter] kept {kept}, dropped {dict(counts)} -> {out}")
    if opts.report:
        write_json_atomic(Path(opts.report).expanduser(), summary)
    if opts.rejects:
        rp = Path(opts.rejects).expanduser()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejects), encoding="utf-8")
    return 0
