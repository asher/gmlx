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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import corpus as _corpus
from .constants import log
from .format import write_json_atomic
from .frames import CONTINUE_INSTRUCTION

GEN_VERSION = "3"
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


def prompt_rows(opts: GenOptions) -> list[dict]:
    """``[{id, messages, student_messages?, ...}]`` from the prompt file or
    built from a text corpus. ``messages`` is the teacher's list with any
    context applied; ``student_messages`` is the prompt as given and is
    present only when a context was applied."""
    if opts.context and not Path(opts.context).expanduser().is_file():
        raise ValueError(f"no context file at {opts.context}")
    shared = Path(opts.context).expanduser().read_text(encoding="utf-8") if opts.context else None
    rows: list[dict] = []
    if opts.prompts:
        path = Path(opts.prompts).expanduser()
        if not path.is_file():
            raise ValueError(f"no prompts file at {path}")
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r.get("messages")
            if not msgs or msgs[-1].get("role") != "user":
                raise ValueError(f"prompt {i} of {path.name}: messages must end on a user turn")
            own = r.get("context")
            ctx = own if isinstance(own, str) and own.strip() else shared
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
    pass


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
        raise ServerError(f"HTTP {e.code} from {url}: {body}") from None


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


def wait_ready(base_url: str, proc, timeout: float) -> str:
    """Poll ``/models`` until it lists a model, then require one 1-token
    completion (the model list answers while the model still preloads).
    Returns the served model id."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise ServerError(f"server exited with code {proc.returncode} before becoming ready")
        try:
            data = _get_json(base_url + "/models", timeout=3).get("data") or []
            if data:
                model_id = data[0]["id"]
                body = {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
                _post_json(base_url + "/chat/completions", body, timeout=120)
                return model_id
        except (urllib.error.URLError, OSError, ValueError, ServerError, KeyError):
            pass
        time.sleep(2)
    raise ServerError(f"server not ready after {timeout:.0f}s")


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


def complete(base_url: str, model_id: str, messages: list[dict], opts: GenOptions, seed: int,
             tokenizer=None) -> dict:
    """One chat completion. With a thinking budget, ``budget_hit`` is True
    when the trace reached the budget and the server forced it closed:
    the server's reasoning token count when it reports one, else the
    trace re-tokenized with ``tokenizer``. None without a budget or a
    way to count."""
    t0 = time.perf_counter()
    body = dict(_sampling(opts, seed), model=model_id, messages=messages, max_tokens=opts.max_tokens)
    obj = _post_json(base_url + "/chat/completions", body, timeout=opts.timeout)
    ch = obj["choices"][0]
    msg = ch.get("message") or {}
    usage = obj.get("usage") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    rt = _reasoning_tokens(obj)
    if rt is None and tokenizer is not None and opts.thinking_budget:
        rt = trace_tokens(tokenizer, reasoning)
    budget_hit: bool | None = None
    if opts.thinking_budget and rt is not None:
        # the criteria forces the close once the count exceeds the budget, so a
        # cut trace re-tokenizes to about budget + 1; a merge or two of slack
        budget_hit = rt >= opts.thinking_budget
    return {"content": msg.get("content") or "", "reasoning": reasoning,
            "finish_reason": ch.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "reasoning_tokens": rt,
            "budget_hit": budget_hit, "wall_s": time.perf_counter() - t0}


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
                  "context": bool(r.get("student_messages")), "seed": seed, "wall_s": round(c["wall_s"], 3)}
    return row


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _done_ids(out: Path) -> set[str]:
    """The ids already in ``out``. A torn final line (a kill mid-write) is
    cut off and logged; a bad line anywhere else raises ValueError."""
    done: set[str] = set()
    if not out.exists():
        return done
    text = out.read_text(encoding="utf-8")
    lines = text.split("\n")
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            done.add(str(json.loads(line)["id"]))
        except (ValueError, KeyError, TypeError) as e:
            if k == len(lines) - 1:
                out.write_text(text[:len(text) - len(line)], encoding="utf-8")
                log(f"[gen] dropped a torn last line of {out} ({len(line)} chars)")
                break
            raise ValueError(f"{out}: line {k + 1} is not a JSON row with an id ({e})") from e
    return done


def run_settings(opts: GenOptions) -> dict:
    """The settings every row of one output file shares, recorded in the
    sidecar; a resume must match them."""
    return {"sampling": {"temperature": opts.temperature, "top_p": opts.top_p, "top_k": opts.top_k,
                         "min_p": opts.min_p, "max_tokens": opts.max_tokens},
            "seed": opts.seed,
            "chat_template_kwargs": json.loads(opts.chat_template_kwargs) if opts.chat_template_kwargs else None,
            "thinking": bool(opts.thinking), "thinking_budget": opts.thinking_budget}


def resume_conflict(prev: dict, opts: GenOptions) -> str | None:
    """What differs between an earlier run's sidecar and this run's
    settings, or None."""
    diffs = [f"{k} {prev.get(k)!r} -> {v!r}" for k, v in run_settings(opts).items()
             if k in prev and prev.get(k) != v]
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
            if not isinstance(json.loads(opts.chat_template_kwargs), dict):
                raise ValueError("not an object")
        except ValueError as e:
            print(f"[gen] refuse: --chat-template-kwargs is not a JSON object: {e}", file=sys.stderr)
            return 2
    if opts.thinking_budget and not opts.thinking:
        print("[gen] refuse: --thinking-budget needs --thinking", file=sys.stderr)
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
    out = Path(opts.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = prompt_rows(opts)
    except ValueError as e:
        print(f"[gen] refuse: {e}", file=sys.stderr)
        return 2
    prompt_hash = prompt_set_sha256(rows)
    side = out.with_suffix(out.suffix + ".gen.json")
    # the settings check comes before the torn-line cut, so a refused
    # resume leaves the file as it found it
    if out.exists() and out.stat().st_size > 0 and side.exists():
        conflict = resume_conflict(json.loads(side.read_text(encoding="utf-8")), opts)
        if conflict:
            print(f"[gen] refuse: {out} was generated with other settings ({conflict}), pass a fresh --out",
                  file=sys.stderr)
            return 2
    try:
        done = _done_ids(out)
    except ValueError as e:
        print(f"[gen] refuse: {e}", file=sys.stderr)
        return 2
    todo = [(i, r) for i, r in enumerate(rows) if r["id"] not in done]
    log(f"[gen] {len(rows)} prompts (sha256 {prompt_hash[:12]}), {len(done)} done, {len(todo)} to run at "
        f"concurrency {opts.concurrency}, max_tokens {opts.max_tokens}, T {opts.temperature} top_p {opts.top_p}")
    if not todo:
        return 0
    base_url = opts.base_url.rstrip("/") if opts.base_url else f"http://{opts.host}:{opts.port}/v1"
    proc = None
    try:
        if not opts.base_url:
            proc = spawn_server(opts, out.with_suffix(out.suffix + ".server.log"))
        model_id = wait_ready(base_url, proc, opts.startup_timeout)
        log(f"[gen] server ready: model {model_id}")
        if done and not side.exists():
            log(f"[gen] warn: {len(done)} rows in {out} without a sidecar, this run's settings are recorded "
                "for them as well")
        if not done or not side.exists():
            # the settings land before the first request, so a run cut
            # short still leaves what a resume compares against; a sidecar
            # beside a deleted output is replaced, one beside rows already
            # generated is kept for its run totals
            write_json_atomic(side, {"gen_version": GEN_VERSION, "model": opts.teacher or base_url,
                                     "served_model_id": model_id, **run_settings(opts),
                                     "prompt_set_sha256": prompt_hash, "run": None})
        lock = threading.Lock()
        t0 = time.perf_counter()
        n_ok = n_err = 0
        gen_tokens = stops = budget_hits = longest = 0

        def work(i, r):
            return i, r, complete(base_url, model_id, r["messages"], opts, opts.seed + i, tokenizer)

        with open(out, "a", encoding="utf-8") as ofh, ThreadPoolExecutor(max_workers=opts.concurrency) as ex:
            futs = [ex.submit(work, i, r) for i, r in todo]
            for fut in as_completed(futs):
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
                    if c["finish_reason"] == "stop":
                        longest = max(longest, int(c["completion_tokens"] or 0))
                    if n_ok % opts.report_every == 0:
                        el = time.perf_counter() - t0
                        log(f"[gen] {n_ok}/{len(todo)} done, {gen_tokens} tokens, {gen_tokens / el:.0f} tok/s "
                            f"aggregate, stop {stops / n_ok:.3f}, mean {gen_tokens / n_ok:.0f} tokens per reply, "
                            f"{n_err} failed ({el:.0f}s)")
        el = time.perf_counter() - t0
        with_context = any(r.get("student_messages") for r in rows)
        sidecar = {
            "gen_version": GEN_VERSION, "model": opts.teacher or base_url, "served_model_id": model_id,
            **run_settings(opts),
            "serve_args": list(opts.serve_arg), "prompt_source": opts.prompts or opts.corpus,
            "prompt_set_sha256": prompt_hash, "prompts": len(rows),
            "instruction": opts.instruction if opts.corpus else None,
            "prefix_chars": opts.prefix_chars if opts.corpus else None, "filter_version": None,
            "context": opts.context or ("per-prompt" if with_context else None),
            "context_format": opts.context_format if with_context else None,
            "run": {"completed": n_ok, "failed": n_err, "generated_tokens": gen_tokens, "wall_s": el,
                    "tok_s_aggregate": gen_tokens / max(el, 1e-9), "stop_fraction": stops / max(n_ok, 1),
                    "budget_hits": budget_hits, "longest_stopped_reply_tokens": longest,
                    "concurrency": opts.concurrency}}
        prev = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
        if prev and prev.get("prompt_set_sha256") == prompt_hash and isinstance(prev.get("run"), dict):
            for k in ("completed", "failed", "generated_tokens", "wall_s", "budget_hits"):
                sidecar["run"][k] += prev["run"].get(k, 0)
            sidecar["run"]["longest_stopped_reply_tokens"] = max(
                longest, prev["run"].get("longest_stopped_reply_tokens", 0))
            sidecar["run"]["tok_s_aggregate"] = sidecar["run"]["generated_tokens"] / max(sidecar["run"]["wall_s"], 1e-9)
        write_json_atomic(side, sidecar)
        log(f"[gen] done: {n_ok} replies, {gen_tokens} tokens, {gen_tokens / max(el, 1e-9):.0f} tok/s aggregate, "
            f"stop fraction {stops / max(n_ok, 1):.3f}, {budget_hits} budget hits, {n_err} failed, sidecar {side}")
        return 0 if n_err == 0 else 1
    except PortInUse as e:
        print(f"[gen] refuse: {e}", file=sys.stderr)
        return 2
    except ServerError as e:
        print(f"[gen] error: {e}", file=sys.stderr)
        return 2
    finally:
        stop_server(opts, proc)
