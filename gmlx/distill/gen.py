"""Corpus generation for the teacher pass: run a teacher to its own end of
turn over a prompt set through ``gmlx serve``, with request fan-out, and
write the replies as a conversation corpus with a generator sidecar.

The output is one ``{"messages": [...]}`` row per line with the teacher's
reply as the last assistant turn, plus ``<out>.gen.json``, which ``cache``
copies into the manifest's generator block and ``filter`` extends. A row
also carries a ``gen`` block (finish reason, token counts, whether the
thinking budget cut the trace) for the filter. The server reports no
reasoning token count, so with a budget the verb counts the trace's
tokens with the teacher's tokenizer, loaded from the GGUF header. When a prompt carries a
context the teacher reads and the student never sees, the row keeps two
lists: ``messages`` with the context in front of the last user turn and
``student_messages`` with the prompt as given, both ending on the same
reply. Every request is a plain chat completion, so what the teacher sees
is what any serve client would send."""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from . import corpus as _corpus
from .constants import log
from .format import write_bytes_atomic, write_json_atomic
from .frames import CONTINUE_INSTRUCTION

GEN_VERSION = "4"
# tokens a forced close spends when the tokenizer gives no better count
# (run_gen measures the wrap phrase plus the closing marker)
CLOSE_ALLOWANCE = 4
# a cut trace re-tokenized by gen can lose this many tokens to merges
RETOKENIZE_SLACK = 2
# a trace this far past the budget was never cut: the server ignored it
UNENFORCED_MARGIN = 32
# serve flags that install a drafter, whose budget close differs
DRAFTER_FLAGS = ("--native-mtp", "--speculative", "--draft-gguf")
# the template variables serve's thinking switch is mapped onto; the
# mapped switch overwrites a same-named kwarg, so gen owns them
THINKING_KWARGS = ("enable_thinking", "thinking", "thinking_mode")


# serve flags that change what the teacher is prompted with, which the
# rows would not record, each with the reason gen gives when it refuses one
PROMPT_FLAGS = {
    "--thinking": "is gen's own flag, sent with every request and recorded in the sidecar, not a --serve-arg",
    "--thinking-budget": "is gen's own flag, sent with every request, not a --serve-arg",
    "--chat-template-config": "is set from gen's --chat-template-kwargs, which the sidecar records, "
                              "not a --serve-arg",
    "--reasoning-effort": "changes the teacher's prompt with no record in the rows, set the template's "
                          "variable with --chat-template-kwargs instead",
    "--system-prompt": "changes the teacher's prompt with no record in the rows, put a system turn in "
                       "the prompt rows instead",
    "--chat-template": "renders the teacher's prompt with a template that cache does not use, so cache "
                       "would score other bytes than the teacher saw",
    "--profile": "sets template variables through its reasoning intents, with no record in the rows, set "
                 "sampling with gen's own flags and variables with --chat-template-kwargs instead",
}


def drafter_flag(arg: str) -> bool:
    """True for a serve argument that installs a drafter, by its full
    spelling, an abbreviation argparse accepts, or either with =value."""
    name = arg.split("=", 1)[0]
    return len(name) > 3 and any(f.startswith(name) for f in DRAFTER_FLAGS)


def prompt_flag(arg: str) -> str | None:
    """The PROMPT_FLAGS entry a serve argument sets, by its full spelling,
    an abbreviation argparse accepts, or either with =value."""
    name = arg.split("=", 1)[0]
    if name in PROMPT_FLAGS:
        return name
    if len(name) > 3:
        return next((f for f in PROMPT_FLAGS if f.startswith(name)), None)
    return None
DEFAULT_CONTEXT_FORMAT = "{context}\n\n{prompt}"


@dataclass
class GenOptions:
    out: str
    prompts: str | None = None
    corpus: str | None = None
    teacher: str | None = None
    base_url: str | None = None
    host: str = "127.0.0.1"
    port: int = 8093
    text_key: str = "text"
    hf_split: str = "train"
    prefix_chars: int = 1500
    min_chars: int = 2000
    docs: int = 0
    instruction: str = CONTINUE_INSTRUCTION
    chat_template_kwargs: str | None = None
    context: str | None = None
    context_format: str = DEFAULT_CONTEXT_FORMAT
    thinking: bool = False
    thinking_budget: int | None = None
    tokenizer: str | None = None
    close_tokens: int = CLOSE_ALLOWANCE      # set by run_gen from the tokenizer
    serve_arg: list[str] = field(default_factory=list)
    startup_timeout: float = 900.0
    concurrency: int = 8
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None
    min_p: float | None = None
    seed: int = 1
    timeout: float = 1800.0
    report_every: int = 50


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def apply_context(messages: list[dict], context: str, fmt: str = DEFAULT_CONTEXT_FORMAT) -> list[dict]:
    """The teacher's copy of ``messages`` with ``context`` placed in front
    of the last user turn through ``fmt``."""
    teacher = [dict(m) for m in messages]
    teacher[-1]["content"] = fmt.format(context=context.rstrip(), prompt=messages[-1]["content"])
    return teacher


def _read_side(side: Path) -> dict:
    """The sidecar as a dict, or a ValueError naming the file when it is
    not a JSON object (a torn write)."""
    try:
        obj = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{side} is not a JSON object ({e})") from None
    if not isinstance(obj, dict):
        raise ValueError(f"{side} is not a JSON object")
    return obj


def prompt_rows(opts: GenOptions) -> list[dict]:
    """``[{id, messages, student_messages?, ...}]`` from the prompt file or
    built from a text corpus. ``messages`` is the teacher's list with any
    context applied; ``student_messages`` is the prompt as given and is
    present only when a context was applied."""
    if opts.context and not Path(opts.context).expanduser().is_file():
        raise ValueError(f"no context file at {opts.context}")
    shared = _corpus.read_utf8(Path(opts.context).expanduser()) if opts.context else None
    if shared is not None and not shared.strip():
        raise ValueError(f"context file {opts.context} is blank")
    rows: list[dict] = []
    if opts.prompts:
        path = Path(opts.prompts).expanduser()
        if not path.is_file():
            raise ValueError(f"no prompts file at {path}")
        for i, line in enumerate(_corpus.read_utf8(path).split("\n")):
            if not line.strip():
                continue
            where = f"prompt {i} of {path.name}"
            try:
                r = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{where}: not JSON ({e})") from None
            if not isinstance(r, dict):
                raise ValueError(f"{where}: not a JSON object")
            if "student_messages" in r:
                raise ValueError(f"{where}: student_messages is written by gen, not read")
            msgs = _corpus.message_list(r, "messages", where)
            if not msgs or msgs[-1].get("role") != "user":
                raise ValueError(f"{where}: messages must end on a user turn")
            own = r.get("context")
            if own is not None and (not isinstance(own, str) or not own.strip()):
                raise ValueError(f"{where}: context must be a non-empty string")
            ctx = own if own is not None else shared
            extra = {k: v for k, v in r.items() if k not in ("id", "messages", "context", "student_messages")}
            row = {"id": str(r.get("id", i)), "messages": msgs, **extra}
            if ctx:
                row = {"id": row["id"], "messages": apply_context(msgs, ctx, opts.context_format),
                       "student_messages": msgs, **extra}
            rows.append(row)
        return _unique_ids(rows)
    assert opts.corpus is not None
    n = 0
    for doc_id, text in _corpus.iter_corpus(opts.corpus, text_key=opts.text_key, limit=None,
                                           hf_split=opts.hf_split):
        text = text.strip()
        if len(text) < opts.min_chars:
            continue
        prefix = text[:opts.prefix_chars]
        if len(text) > opts.prefix_chars:
            cut = prefix.rfind(" ")
            if cut > opts.prefix_chars // 2:
                prefix = prefix[:cut]
        msgs = [{"role": "user", "content": f"{opts.instruction}\n\n{prefix}"}]
        row = {"id": str(doc_id), "messages": msgs}
        if shared:
            row = {"id": row["id"], "messages": apply_context(msgs, shared, opts.context_format),
                   "student_messages": msgs}
        rows.append(row)
        n += 1
        if opts.docs and n >= opts.docs:
            break
    return _unique_ids(rows)


def _unique_ids(rows: list[dict]) -> list[dict]:
    """The rows, or a ValueError naming the first id two rows share: a
    resume skips rows by id, so a shared id would drop one of them."""
    seen: set[str] = set()
    for r in rows:
        if r["id"] in seen:
            raise ValueError(f"prompt id {r['id']!r} appears twice, ids must be unique")
        seen.add(r["id"])
    return rows


def prompt_set_sha256(rows: list[dict]) -> str:
    text = "\n".join(json.dumps(r["messages"], sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

class PortInUse(RuntimeError):
    """The port has a listener already, a refusal rather than a failure."""


class ServerError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status   # the HTTP status, when the server answered with one


def _get_json(url: str, timeout: float):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local server)
        return json.loads(r.read())


def _post_json(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local server)
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise ServerError(f"HTTP {e.code} from {url}: {body}", status=e.code) from None


def port_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def spawn_server(opts: GenOptions, log_path: Path):
    """Start ``gmlx serve`` on the teacher through the serve lifecycle
    (detached, runfile written, so ``gmlx stop`` can find it). Returns the
    process. Refuses a port that already has a listener: a stranger there
    would answer the requests as the teacher."""
    from gmlx.serve import lifecycle
    if port_listening(opts.host, opts.port):
        raise PortInUse(f"port {opts.port} already has a listener; stop it (gmlx stop --port {opts.port}) "
                          "or pass another --port")
    # serve maps its own thinking switch onto whatever variable the
    # teacher's template reads, so the switch is sent as that, not as a
    # template kwarg of one family's name
    serve_args = [opts.teacher, "--thinking", "on" if opts.thinking else "off"]
    if opts.chat_template_kwargs:
        serve_args += ["--chat-template-config", opts.chat_template_kwargs]
    serve_args += list(opts.serve_arg)
    spawned = lifecycle.start_background_nowait(serve_args, host=opts.host, port=opts.port, log=str(log_path))
    if spawned is None:
        raise PortInUse(f"a server already holds {opts.host}:{opts.port}; pass --base-url to use it "
                          "or --port for a free one")
    proc, _ = spawned
    return proc


def pick_model_id(ids: list[str], teacher: str | None) -> str:
    """The served id gen sends: the only one listed or, among several,
    the one named like ``--teacher`` (its path, file name or stem).
    Raises ServerError otherwise, since a request to the first id
    listed would go to a model the sidecar does not name."""
    if len(ids) == 1:
        return ids[0]
    names: set[str] = set()
    if teacher:
        p = Path(teacher).expanduser()
        names = {teacher, p.name, p.stem}
    hits = [i for i in ids if i in names or Path(i).name in names or Path(i).stem in names]
    if len(hits) == 1:
        return hits[0]
    raise ServerError(f"the server lists {len(ids)} models ({', '.join(ids)}) and "
                      f"{'several' if hits else 'none'} match --teacher {teacher!r}; point --base-url at a "
                      "server holding the teacher alone, or name the served model as --teacher")


def wait_ready(base_url: str, proc, timeout: float, teacher: str | None = None) -> str:
    """Poll ``/models`` until it lists a model, then require one 1-token
    completion (the model list answers while the model still preloads,
    and a server still loading answers the completion with a 5xx, which
    is polled again). Returns the served model id, chosen by
    ``pick_model_id`` when the server lists several. A list in which none
    or several match the teacher, and a 4xx other than 408 or 429, end
    the wait at once."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise ServerError(f"server exited with code {proc.returncode} before becoming ready")
        try:
            ids = [str(d["id"]) for d in _get_json(base_url + "/models", timeout=3).get("data") or []]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            ids, last = [], f"GET /models: {e}"
        if ids:
            model_id = pick_model_id(ids, teacher)
            body = {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
            try:
                _post_json(base_url + "/chat/completions", body, timeout=120)
                return model_id
            except ServerError as e:
                if e.status is not None and 400 <= e.status < 500 and e.status not in (408, 429):
                    # a request the server refuses (no such model, a bad
                    # template) does not clear by waiting
                    raise
                last = str(e)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                last = str(e)
        time.sleep(2)
    raise ServerError(f"server not ready after {timeout:.0f}s" + (f" (last error: {last})" if last else ""))


def stop_server(opts: GenOptions, proc) -> None:
    if proc is None:
        return
    from gmlx.serve import lifecycle
    try:
        lifecycle.stop(opts.host, opts.port)
    except Exception as e:  # noqa: BLE001 - the process is killed below in any case
        log(f"[gen] warn: gmlx stop failed ({e}), terminating the server process")
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------

def _sampling(opts: GenOptions, seed: int) -> dict:
    body: dict = {"temperature": opts.temperature, "top_p": opts.top_p, "seed": seed, "stream": False}
    if opts.top_k is not None:
        body["top_k"] = opts.top_k
    if opts.min_p is not None:
        body["min_p"] = opts.min_p
    if opts.thinking_budget:
        body["thinking_budget"] = opts.thinking_budget
    # a server gen did not start (--base-url) only sees what the request
    # carries, so the thinking switch and the template kwargs ride along.
    # The switch is serve's own thinking control, which it maps onto the
    # variable the teacher's template reads, and it goes both ways: a
    # template whose default is thinking would otherwise think without
    # --thinking while the sidecar says off
    body["thinking"] = "on" if opts.thinking else "off"
    if opts.chat_template_kwargs:
        body["chat_template_kwargs"] = json.loads(opts.chat_template_kwargs)
    return body


def _reasoning_tokens(obj: dict) -> int | None:
    """Reasoning tokens as the server reports them, from the usage details
    or the timings block, or None when it reports neither."""
    usage = obj.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    for block in (details, usage, obj.get("timings") or {}):
        v = block.get("reasoning_tokens")
        if isinstance(v, int):
            return v
    return None


def trace_tokens(tokenizer, reasoning: str) -> int:
    return len(tokenizer.encode(reasoning, add_special_tokens=False)) if reasoning else 0


def wrap_closed(reasoning: str) -> bool:
    """True when the trace ends with the phrase a gmlx server injects
    ahead of a forced close."""
    from ..gen.thinking_budget import BUDGET_WRAP_PHRASE
    return bool(reasoning) and reasoning.rstrip().endswith(BUDGET_WRAP_PHRASE.strip())


def close_tokens(tokenizer, *, spawned: bool) -> int:
    """Tokens the request budget leaves for a forced close. The server gen
    spawns is never drafted (run_gen refuses that) and closes a cut trace
    with a newline and the end marker, which CLOSE_ALLOWANCE covers with
    the over-budget token and an opening marker. A server behind
    --base-url may be drafted, and a drafted close is the wrap phrase plus
    the marker, so that count is used there and an answer from a plain
    server can run that many tokens past --max-tokens."""
    if spawned:
        return CLOSE_ALLOWANCE
    from ..gen.thinking_budget import budget_close_tokens
    return max(budget_close_tokens(tokenizer), CLOSE_ALLOWANCE)


def complete(base_url: str, model_id: str, messages: list[dict], opts: GenOptions, seed: int,
             tokenizer=None) -> dict:
    """One chat completion. With a thinking budget, ``budget_hit`` is True
    when the trace reached the budget and the server forced it closed,
    and ``budget_unenforced`` when the trace ran so far past it that the
    server cannot have applied it (the trace is then whole, not cut):
    the server's reasoning token count when it reports one, else the
    trace re-tokenized with ``tokenizer``. None without a budget or a
    way to count."""
    t0 = time.perf_counter()
    body = dict(_sampling(opts, seed), model=model_id, messages=messages, max_tokens=answer_budget(opts))
    obj = _post_json(base_url + "/chat/completions", body, timeout=opts.timeout)
    ch = obj["choices"][0]
    msg = ch.get("message") or {}
    usage = obj.get("usage") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    rt = _reasoning_tokens(obj)
    reported = rt is not None
    if rt is None and tokenizer is not None and opts.thinking_budget:
        rt = trace_tokens(tokenizer, reasoning)
    budget_hit: bool | None = None
    unenforced = False
    if opts.thinking_budget and rt is not None:
        # a drafted gmlx server ends every trace it cuts with its wrap
        # phrase, so that alone marks a hit whatever the count (it
        # overshoots by a draft block). Elsewhere the count decides: the
        # plain path closes with a newline and the marker, the server forces
        # the close once the count reaches the budget, its own count is
        # exact, a trace re-tokenized here can come out a merge or two
        # short, and a trace far past the budget plus its close was never
        # cut (a drafter at concurrency above 1 drops the budget with a
        # note in the server log)
        wrapped = wrap_closed(reasoning)
        unenforced = not wrapped and rt > opts.thinking_budget + opts.close_tokens + UNENFORCED_MARGIN
        budget_hit = wrapped or (not unenforced
                                 and rt >= opts.thinking_budget - (0 if reported else RETOKENIZE_SLACK))
    return {"content": msg.get("content") or "", "reasoning": reasoning,
            "finish_reason": ch.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "reasoning_tokens": rt,
            "budget_hit": budget_hit, "budget_unenforced": unenforced, "wall_s": time.perf_counter() - t0}


def reply_row(r: dict, c: dict, seed: int) -> dict:
    reply = {"role": "assistant", "content": c["content"]}
    if c["reasoning"]:
        reply["reasoning_content"] = c["reasoning"]
    row = {"id": r["id"], "messages": r["messages"] + [reply],
           **{k: v for k, v in r.items() if k not in ("id", "messages", "student_messages")}}
    if r.get("student_messages"):
        row["student_messages"] = r["student_messages"] + [reply]
    row["gen"] = {"finish_reason": c["finish_reason"], "completion_tokens": c["completion_tokens"],
                  "prompt_tokens": c["prompt_tokens"], "reasoning_tokens": c["reasoning_tokens"],
                  "reasoning_chars": len(c["reasoning"]), "budget_hit": c["budget_hit"],
                  "budget_unenforced": bool(c.get("budget_unenforced")),
                  "context": bool(r.get("student_messages")), "seed": seed, "wall_s": round(c["wall_s"], 3)}
    return row


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def answer_budget(opts: GenOptions) -> int:
    """max_tokens for a request: the server counts the reasoning trace in
    it, so a thinking budget is added on top, plus the tokens a forced
    close spends inside the trace (the wrap phrase and the closing
    marker), and the answer keeps the --max-tokens budget. A thinking
    reply without a budget shares --max-tokens with its trace."""
    if opts.thinking and opts.thinking_budget:
        return opts.max_tokens + opts.thinking_budget + opts.close_tokens
    return opts.max_tokens


def sidecar_settings(opts: GenOptions, rows: list, prompt_hash: str, with_context: bool, base_url: str,
                     model_id: str | None) -> dict:
    """The sidecar's settings block, everything but the run totals:
    what a resume compares, the prompt set and the context."""
    return {"gen_version": GEN_VERSION, "model": teacher_name(opts) or base_url, "served_model_id": model_id,
            **run_settings(opts),
            "prompt_source": opts.prompts or opts.corpus,
            "prompt_set_sha256": prompt_hash, "prompts": len(rows),
            "instruction": opts.instruction if opts.corpus else None,
            "prefix_chars": opts.prefix_chars if opts.corpus else None, "filter_version": None,
            "context": shared_context(opts) or ("per-prompt" if with_context else None),
            "shared_context": shared_context(opts),
            "context_format": opts.context_format if with_context else None}


def row_totals(out: Path) -> dict:
    """The run totals of every reply row in ``out``, read from the rows'
    ``gen`` blocks, so the totals describe the file after any number of
    resumes and interrupts rather than the runs that wrote it."""
    completed = stops = tokens = budget_hits = longest = unenforced = 0
    if out.exists():
        with open(out, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    g = json.loads(line).get("gen") or {}
                except (ValueError, AttributeError):
                    continue
                completed += 1
                ct = int(g.get("completion_tokens") or 0)
                tokens += ct
                budget_hits += int(bool(g.get("budget_hit")))
                unenforced += int(bool(g.get("budget_unenforced")))
                if g.get("finish_reason") == "stop":
                    stops += 1
                    longest = max(longest, ct)
    return {"completed": completed, "generated_tokens": tokens, "stops": stops,
            "stop_fraction": stops / max(completed, 1), "budget_hits": budget_hits,
            "budget_unenforced": unenforced, "longest_stopped_reply_tokens": longest}


def _done_ids(out: Path) -> dict[str, list | None]:
    """The ids already in ``out`` with the prompt each one answered (its
    messages without the reply, None when the row has none). A torn
    final line (a kill mid-write) is cut off and logged, a complete
    final row with no newline after it gets one, and a bad line
    anywhere raises ValueError."""
    done: dict[str, list | None] = {}
    if not out.exists():
        return done
    # bytes, not text: a kill mid-write can leave a partial multi-byte
    # character, which a text read would raise on before the cut
    data = out.read_bytes()
    lines = data.split(b"\n")
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        # the last element is a line only when no newline ends the data
        unterminated = k == len(lines) - 1
        try:
            row = json.loads(line.decode("utf-8"))
        except ValueError as e:
            if unterminated:
                write_bytes_atomic(out, data[:len(data) - len(line)])
                log(f"[gen] dropped a torn last line of {out} ({len(line)} bytes)")
                break
            raise ValueError(f"{out}: line {k + 1} is not a JSON row ({e})") from e
        try:
            msgs = row.get("messages")
            done[str(row["id"])] = list(msgs[:-1]) if isinstance(msgs, list) and msgs else None
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"{out}: line {k + 1} is not a JSON row with an id ({e})") from e
        if unterminated:
            # the next row would otherwise be written onto this line
            write_bytes_atomic(out, data + b"\n")
            log(f"[gen] ended the last row of {out} with the newline it lacked")
    return done


def prompt_conflict(done: dict[str, list | None], rows: list[dict], out: Path, source: str) -> str | None:
    """Why the rows already in ``out`` cannot be resumed against these
    prompt rows: a done id whose prompt is not this row's, or one that
    the prompt set no longer holds. Ids from line numbers shift when a
    line is inserted or removed, so both cases mean another prompt set."""
    by_id = {r["id"]: r["messages"] for r in rows}
    gone = sorted(i for i in done if i not in by_id)
    changed = sorted(i for i, p in done.items() if i in by_id and p is not None and p != by_id[i])
    if not gone and not changed:
        return None
    what = []
    if changed:
        what.append(f"{len(changed)} answered another prompt under the same id (first {changed[0]!r})")
    if gone:
        what.append(f"{len(gone)} have ids not in {source} (first {gone[0]!r})")
    return (f"of the {len(done)} replies in {out}, " + " and ".join(what)
            + "; ids taken from line numbers shift when a line is added or removed, pass a fresh --out")


def context_format_error(fmt: str) -> str | None:
    """Why ``fmt`` cannot combine a context and a prompt: a field other
    than the two, or one of the two missing."""
    try:
        probe = fmt.format(context="\x00C\x00", prompt="\x00P\x00")
    except (KeyError, IndexError, ValueError, AttributeError) as e:
        return f"--context-format takes the fields {{context}} and {{prompt}} only ({e!r})"
    if "\x00C\x00" not in probe or "\x00P\x00" not in probe:
        return "--context-format must place both {context} and {prompt}"
    return None


def run_settings(opts: GenOptions) -> dict:
    """The settings every row of one output file shares, recorded in the
    sidecar; a resume must match them."""
    return {"sampling": {"temperature": opts.temperature, "top_p": opts.top_p, "top_k": opts.top_k,
                         "min_p": opts.min_p, "max_tokens": opts.max_tokens},
            "seed": opts.seed,
            "chat_template_kwargs": json.loads(opts.chat_template_kwargs) if opts.chat_template_kwargs else None,
            "thinking": bool(opts.thinking), "thinking_budget": opts.thinking_budget,
            "request_max_tokens": answer_budget(opts),
            "serve_args": list(opts.serve_arg)}


def teacher_name(opts: GenOptions) -> str | None:
    """The teacher file as an absolute path, or None when a server is
    named instead."""
    return str(Path(opts.teacher).expanduser().absolute()) if opts.teacher else None


def shared_context(opts: GenOptions) -> str | None:
    """The --context file as an absolute path, or None."""
    return str(Path(opts.context).expanduser().absolute()) if opts.context else None


def model_label(opts: GenOptions) -> str:
    """What the sidecar records as the model: the teacher file, else the
    server the replies came from."""
    return teacher_name(opts) or (opts.base_url.rstrip("/") if opts.base_url
                                  else f"http://{opts.host}:{opts.port}/v1")


def same_name(a, b) -> bool:
    """Two model or context names agree when they are one URL or one
    file under two spellings of its path."""
    def norm(s):
        if not isinstance(s, str):
            return s
        if s.startswith(("http://", "https://")):
            return s.rstrip("/")
        return str(Path(s).expanduser().absolute())
    return norm(a) == norm(b)


def resume_conflict(prev: dict, opts: GenOptions, *, with_context: bool = False) -> str | None:
    """What differs between an earlier run's sidecar and this run's
    settings, model and context, or None. The context format is compared
    as the sidecar records it (None when no row got a context); a context
    the rows carry themselves is recorded as per-prompt and is not a flag
    to compare."""
    now: dict = dict(run_settings(opts), gen_version=GEN_VERSION,
                     context_format=opts.context_format if with_context else None)
    if "shared_context" in prev:
        now["shared_context"] = shared_context(opts)
    elif prev.get("context") != "per-prompt":
        now["context"] = opts.context
    diffs = [f"{k} {prev.get(k)!r} -> {v!r}" for k, v in now.items()
             if k in prev and prev.get(k) != v
             and not (k in ("context", "shared_context") and same_name(prev.get(k), v))]
    if "shared_context" not in prev and prev.get("context") == "per-prompt" and opts.context:
        # a sidecar from before the shared-context key recorded per-prompt
        # only when no --context was given
        diffs.append(f"context 'per-prompt' -> {opts.context!r}")
    if "model" in prev and not same_name(prev["model"], model_label(opts)):
        diffs.append(f"model {prev['model']!r} -> {model_label(opts)!r}")
    return ", ".join(diffs) or None


def run_gen(opts: GenOptions) -> int:
    """Generate the corpus. Returns 0 when every request succeeded, 1 when
    some failed (the rows that succeeded are written and a rerun picks up
    the rest), 2 on a refusal before any request."""
    if not opts.prompts and not opts.corpus:
        print("[gen] refuse: --prompts or --corpus is required", file=sys.stderr)
        return 2
    if opts.teacher and not opts.base_url and not Path(opts.teacher).expanduser().exists():
        print(f"[gen] refuse: no model at {opts.teacher}", file=sys.stderr)
        return 2
    if not opts.teacher and not opts.base_url:
        print("[gen] refuse: --teacher or --base-url is required", file=sys.stderr)
        return 2
    if opts.chat_template_kwargs:
        try:
            template_kw = json.loads(opts.chat_template_kwargs)
            if not isinstance(template_kw, dict):
                raise ValueError("not an object")
        except ValueError as e:
            print(f"[gen] refuse: --chat-template-kwargs is not a JSON object: {e}", file=sys.stderr)
            return 2
        named = [k for k in THINKING_KWARGS if k in template_kw]
        if named:
            print(f"[gen] refuse: --chat-template-kwargs sets {', '.join(named)}; the thinking switch gen sends "
                  "with every request is mapped onto that variable and wins over the kwarg, so drop the key "
                  "and use --thinking", file=sys.stderr)
            return 2
    if opts.thinking_budget and not opts.thinking:
        print("[gen] refuse: --thinking-budget needs --thinking", file=sys.stderr)
        return 2
    if opts.thinking_budget and any(drafter_flag(a) for a in opts.serve_arg):
        # a drafted teacher overshoots the budget by a draft block and
        # drops it when requests batch, so its cut is not the one the
        # sidecar records
        print("[gen] refuse: --thinking-budget with a drafter (--native-mtp, --speculative or --draft-gguf in "
              "--serve-arg) "
              "is not enforced per request, serve the teacher without the drafter", file=sys.stderr)
        return 2
    for a in opts.serve_arg:
        # a server-wide budget would cap every request under what the
        # sidecar records, and the other flags change the prompt the rows
        # hold for cache to render
        flag = prompt_flag(a)
        if flag:
            name = a.split("=", 1)[0]
            given = "" if name == flag else f" (given as {name})"
            print(f"[gen] refuse: {flag}{given} {PROMPT_FLAGS[flag]}", file=sys.stderr)
            return 2
    tokenizer = None
    if opts.thinking_budget:
        tok_path = opts.tokenizer or opts.teacher
        if not tok_path:
            print("[gen] refuse: --thinking-budget with --base-url needs --tokenizer to count the trace",
                  file=sys.stderr)
            return 2
        from .tokens import load_tokenizer
        try:
            tokenizer = load_tokenizer(tok_path)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"[gen] refuse: cannot load the tokenizer from {tok_path}: {e}", file=sys.stderr)
            return 2
        opts.close_tokens = close_tokens(tokenizer, spawned=not opts.base_url)
    if opts.corpus is not None:
        c = opts.corpus
        if (c.endswith((".jsonl", ".txt")) or c.startswith((".", "/", "~"))) and not Path(c).expanduser().exists():
            print(f"[gen] refuse: no corpus at {c}", file=sys.stderr)
            return 2
    fmt_err = context_format_error(opts.context_format)
    if fmt_err:
        print(f"[gen] refuse: {fmt_err}", file=sys.stderr)
        return 2
    out = Path(opts.out).expanduser()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = prompt_rows(opts)
    except (OSError, ValueError) as e:
        # an --out gen cannot write, a missing prompt file, or a Hugging
        # Face id the datasets library cannot find (an OSError subclass)
        print(f"[gen] refuse: {e}", file=sys.stderr)
        return 2
    if not rows:
        # a run with nothing to generate would exit 0 and write no --out,
        # and the next step would fail one step away from the cause
        print(f"[gen] refuse: {opts.prompts or opts.corpus} yields no prompts"
              + ("" if opts.prompts else f" (blank documents and those under --min-chars {opts.min_chars} "
                 "are skipped)"), file=sys.stderr)
        return 2
    prompt_hash = prompt_set_sha256(rows)
    with_context = any(r.get("student_messages") for r in rows)
    side = out.with_suffix(out.suffix + ".gen.json")
    # the settings check comes before the torn-line cut; the prompt and
    # server checks below run after it, and the cut only drops a torn
    # last line
    if out.exists() and out.stat().st_size > 0 and side.exists():
        try:
            conflict = resume_conflict(_read_side(side), opts, with_context=with_context)
        except ValueError as e:
            print(f"[gen] refuse: {e}", file=sys.stderr)
            return 2
        if conflict:
            print(f"[gen] refuse: {out} was generated with other settings ({conflict}), pass a fresh --out",
                  file=sys.stderr)
            return 2
    try:
        done = _done_ids(out)
    except (OSError, ValueError) as e:
        print(f"[gen] refuse: {e}", file=sys.stderr)
        return 2
    conflict = prompt_conflict(done, rows, out, opts.prompts or opts.corpus or "the prompt set")
    if conflict:
        print(f"[gen] refuse: {conflict}", file=sys.stderr)
        return 2
    todo = [(i, r) for i, r in enumerate(rows) if r["id"] not in done]
    log(f"[gen] {len(rows)} prompts (sha256 {prompt_hash[:12]}), {len(done)} done, {len(todo)} to run at "
        f"concurrency {opts.concurrency}, max_tokens {opts.max_tokens}, T {opts.temperature} top_p {opts.top_p}")
    base_url = opts.base_url.rstrip("/") if opts.base_url else f"http://{opts.host}:{opts.port}/v1"
    if not todo:
        try:
            prev = _read_side(side) if side.exists() else None
        except ValueError:
            prev = None
        if prev is None or not isinstance(prev.get("run"), dict):
            # every prompt is answered by a run that was cut short (its
            # sidecar holds no run block) or that left no sidecar: the
            # totals still describe the rows, and filter or cache then
            # read the corpus as generated
            sidecar = dict(prev) if prev else sidecar_settings(opts, rows, prompt_hash, with_context, base_url,
                                                              None)
            sidecar["run"] = {**row_totals(out), "failed": 0, "wall_s": 0.0, "tok_s_aggregate": 0.0,
                              "concurrency": opts.concurrency}
            write_json_atomic(side, sidecar)
            log(f"[gen] nothing to run, sidecar {side} written from the {len(done)} rows")
        return 0
    proc = None
    try:
        if not opts.base_url:
            proc = spawn_server(opts, out.with_suffix(out.suffix + ".server.log"))
        model_id = wait_ready(base_url, proc, opts.startup_timeout, teacher=opts.teacher)
        log(f"[gen] server ready: model {model_id}")
        if done and side.exists():
            try:
                prev_id = _read_side(side).get("served_model_id")
            except ValueError as e:
                print(f"[gen] refuse: {e}", file=sys.stderr)
                return 2
            if prev_id not in (None, model_id):
                print(f"[gen] refuse: the server now serves {model_id}, the {len(done)} replies in {out} came "
                      f"from {prev_id}, pass a fresh --out", file=sys.stderr)
                return 2
        if done and not side.exists():
            log(f"[gen] warn: {len(done)} rows in {out} without a sidecar, this run's settings are recorded "
                "for them as well")
        if not done or not side.exists():
            # the settings land before the first request, so a run cut
            # short still leaves what a resume compares against; a sidecar
            # beside a deleted output is replaced, one beside rows already
            # generated is kept for its run totals
            write_json_atomic(side, {**sidecar_settings(opts, rows, prompt_hash, with_context, base_url, model_id),
                                     "run": None})
        lock = threading.Lock()
        t0 = time.perf_counter()
        n_ok = n_err = 0
        gen_tokens = stops = budget_hits = longest = unenforced = 0

        def work(i, r):
            return i, r, complete(base_url, model_id, r["messages"], opts, opts.seed + i, tokenizer)

        ex = ThreadPoolExecutor(max_workers=opts.concurrency)
        queue = iter(todo)
        pending: set = set()
        with open(out, "a", encoding="utf-8") as ofh:
            try:
                # twice the concurrency is in flight at a time, refilled as
                # replies land, so a long run holds no finished reply in
                # the pool and an interrupt has little to cancel
                while True:
                    while len(pending) < 2 * opts.concurrency:
                        item = next(queue, None)
                        if item is None:
                            break
                        pending.add(ex.submit(work, *item))
                    if not pending:
                        break
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        try:
                            i, r, c = fut.result()
                        except Exception as e:  # noqa: BLE001 - one bad request never ends the run
                            with lock:
                                n_err += 1
                            log(f"[gen] warn: request failed: {type(e).__name__}: {e}")
                            continue
                        row = reply_row(r, c, opts.seed + i)
                        with lock:
                            ofh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            ofh.flush()
                            n_ok += 1
                            gen_tokens += int(c["completion_tokens"] or 0)
                            stops += int(c["finish_reason"] == "stop")
                            budget_hits += int(bool(c["budget_hit"]))
                            if c.get("budget_unenforced"):
                                unenforced += 1
                                if unenforced == 1:
                                    log("[gen] warn: a reasoning trace ran past --thinking-budget, the server is "
                                        "not enforcing it (a drafter at --concurrency above 1 drops the budget); "
                                        "such rows are marked budget_unenforced, not budget_hit")
                            if c["finish_reason"] == "stop":
                                longest = max(longest, int(c["completion_tokens"] or 0))
                            if n_ok % opts.report_every == 0:
                                el = time.perf_counter() - t0
                                log(f"[gen] {n_ok}/{len(todo)} done, {gen_tokens} tokens, {gen_tokens / el:.0f} "
                                    f"tok/s aggregate, stop {stops / n_ok:.3f}, mean {gen_tokens / n_ok:.0f} "
                                    f"tokens per reply, {n_err} failed ({el:.0f}s)")
            except BaseException:
                # an interrupt: the queued requests are cancelled first,
                # then the server stops so the requests in flight fail fast
                ex.shutdown(wait=False, cancel_futures=True)
                if opts.base_url:
                    log(f"[gen] interrupted, waiting for the requests in flight on {opts.base_url} "
                        f"(up to --timeout {opts.timeout}s each)")
                stop_server(opts, proc)
                proc = None
                raise
            ex.shutdown(wait=True)
        el = time.perf_counter() - t0
        sidecar = {
            **sidecar_settings(opts, rows, prompt_hash, with_context, base_url, model_id),
            # the totals come from the rows, so they cover every run that
            # wrote to the file; failed is this run's, since a resume
            # retries every earlier failure; wall_s adds up over the runs
            # that ended (an interrupted run records none), and
            # tok_s_aggregate is this run's rate
            "run": {**row_totals(out), "failed": n_err, "wall_s": el, "tok_s_aggregate": gen_tokens / max(el, 1e-9),
                    "concurrency": opts.concurrency}}
        try:
            prev = _read_side(side) if side.exists() else None
        except ValueError:
            prev = None     # this run's settings were already written over it
        # a resume already matched the settings; prompts added since the
        # last run change the prompt set but not what the wall time counts
        if prev and isinstance(prev.get("run"), dict):
            sidecar["run"]["wall_s"] += prev["run"].get("wall_s", 0)
        write_json_atomic(side, sidecar)
        log(f"[gen] done: {n_ok} replies, {gen_tokens} tokens, {gen_tokens / max(el, 1e-9):.0f} tok/s aggregate, "
            f"stop fraction {stops / max(n_ok, 1):.3f}, {budget_hits} budget hits, {unenforced} past the budget, "
            f"{n_err} failed, sidecar {side}")
        return 0 if n_err == 0 else 1
    except PortInUse as e:
        print(f"[gen] refuse: {e}", file=sys.stderr)
        return 2
    except ServerError as e:
        print(f"[gen] error: {e}", file=sys.stderr)
        return 2
    finally:
        stop_server(opts, proc)
