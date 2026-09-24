"""``distill gen`` against a stub chat server on localhost, and ``distill
filter`` on crafted rows: prompt building, the two message lists a
context produces, the gen block and sidecar, resume by prompt id, the
thinking-budget flag, every filter reason in order, the verify command
seam, and the context flag that prepares an on-policy round. No model,
no network beyond the loopback."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gmlx.distill import filter as flt
from gmlx.distill import gen

# ---------------------------------------------------------------------------
# a stub OpenAI-style server: the reply echoes the last user turn's first
# word; a prompt containing LONG runs its trace up to the budget
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    calls: list = []
    model_id: str = "stub-teacher"
    model_ids: list | None = None       # several served models, in place of model_id
    fail_once: set = set()              # first words whose first request fails with a 500
    refuse_status: int | None = None    # the status every request is refused with, when set

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        ids = type(self).model_ids or [type(self).model_id]
        body = json.dumps({"data": [{"id": i} for i in ids]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n))
        type(self).calls.append(req)
        if type(self).refuse_status is not None:
            self.send_response(type(self).refuse_status)
            self.end_headers()
            self.wfile.write(b"model_not_found")
            return
        last = req["messages"][-1]["content"]
        word = last.split()[0] if last.split() else "empty"
        if word in type(self).fail_once:
            type(self).fail_once.discard(word)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"stub failure")
            return
        content = f"reply about {word} " + "and more words " * 8
        msg: dict = {"role": "assistant", "content": content}
        budget = req.get("thinking_budget")
        reasoning_tokens = 0
        if budget:
            reasoning_tokens = budget if "LONG" in last else max(1, budget // 2)
            msg["reasoning_content"] = "thinking " * reasoning_tokens
            if "WRAP" in last:
                # a gmlx server ends a cut trace with its wrap phrase
                from gmlx.gen.thinking_budget import BUDGET_WRAP_PHRASE
                msg["reasoning_content"] += BUDGET_WRAP_PHRASE
        finish = "length" if "CUT" in last else "stop"
        obj = {"choices": [{"message": msg, "finish_reason": finish}],
               "usage": {"prompt_tokens": 12, "completion_tokens": 20 + reasoning_tokens},
               "timings": {"reasoning_tokens": reasoning_tokens}}
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_server():
    _Handler.calls = []
    _Handler.model_id = "stub-teacher"
    _Handler.model_ids = None
    _Handler.fail_once = set()
    _Handler.refuse_status = None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()
        srv.server_close()


def _prompts(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _ThinkTokenizer(_WordTokenizer):
    """A word tokenizer with a thinking-end token, so the forced close
    resolves: the wrap phrase's words plus that one token."""
    think_end_tokens = (99,)


# ---------------------------------------------------------------------------
# gen
# ---------------------------------------------------------------------------

def test_gen_writes_rows_with_two_lists_and_a_sidecar(tmp_path, stub_server):
    ctx = tmp_path / "schema.txt"
    ctx.write_text("CREATE TABLE crews (id INT);\n")
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "a", "messages": [{"role": "user", "content": "alpha question"}], "family": "f1"},
        {"id": "b", "messages": [{"role": "user", "content": "beta question"}],
         "context": "its own context"},
    ])
    out = tmp_path / "corpus.jsonl"
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, context=str(ctx),
                          concurrency=2, seed=5, report_every=1)
    assert gen.run_gen(opts) == 0
    rows = {r["id"]: r for r in _rows(out)}
    assert set(rows) == {"a", "b"}
    a = rows["a"]
    assert a["messages"][0]["content"] == "CREATE TABLE crews (id INT);\n\nalpha question"
    assert a["student_messages"][0]["content"] == "alpha question"
    assert a["messages"][-1] == a["student_messages"][-1]
    assert a["messages"][-1]["content"].startswith("reply about CREATE")
    assert a["family"] == "f1"
    assert a["gen"]["finish_reason"] == "stop" and a["gen"]["context"] is True
    assert a["gen"]["completion_tokens"] == 20 and a["gen"]["budget_hit"] is None
    assert a["gen"]["seed"] == 5
    b = rows["b"]
    assert b["messages"][0]["content"] == "its own context\n\nbeta question"
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["gen_version"] == gen.GEN_VERSION
    assert side["served_model_id"] == "stub-teacher"
    assert side["context"] == str(ctx) and side["context_format"] == gen.DEFAULT_CONTEXT_FORMAT
    assert side["filter_version"] is None
    assert side["run"]["completed"] == 2 and side["run"]["failed"] == 0
    assert side["prompts"] == 2 and len(side["prompt_set_sha256"]) == 64
    assert side["thinking"] is False and side["thinking_budget"] is None
    # the switch is sent off explicitly as serve's own control, so a
    # template that thinks by default does not think behind a sidecar
    # that says off; no template kwarg is invented for it
    sampled = [c for c in _Handler.calls if "seed" in c]
    assert sampled and all(c["thinking"] == "off" and "enable_thinking" not in c and "chat_template_kwargs" not in c
                           for c in sampled)
    assert side["chat_template_kwargs"] is None


def test_gen_resumes_by_prompt_id_and_accumulates_the_run(tmp_path, stub_server):
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
        {"id": "b", "messages": [{"role": "user", "content": "beta"}]},
    ])
    out = tmp_path / "corpus.jsonl"
    out.write_text(json.dumps({"id": "a", "messages": [{"role": "user", "content": "alpha"},
                                                   {"role": "assistant", "content": "x"}],
                               "gen": {}}) + "\n")
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)
    assert gen.run_gen(opts) == 0
    assert [r["id"] for r in _rows(out)] == ["a", "b"]
    assert len(_Handler.calls) == 2                  # the readiness probe and one reply
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    # the totals describe the file: the row found before the run counts
    assert side["run"]["completed"] == 2 and side["run"]["stops"] == 1
    # everything done: no server contact at all
    _Handler.calls = []
    assert gen.run_gen(opts) == 0
    assert _Handler.calls == []


def test_gen_marks_the_thinking_budget_hit(tmp_path, stub_server, monkeypatch):
    """The stub reports reasoning tokens in its timings, which win over the
    tokenizer count; the tokenizer is still required with --base-url."""
    import gmlx.distill.tokens as tokens_mod
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda path: _WordTokenizer())
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "short", "messages": [{"role": "user", "content": "brief one"}]},
        {"id": "long", "messages": [{"role": "user", "content": "LONG one"}]},
    ])
    out = tmp_path / "corpus.jsonl"
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, thinking=True,
                          thinking_budget=40, chat_template_kwargs='{"preserve_thinking": true}',
                          tokenizer="teacher.gguf")
    assert gen.run_gen(opts) == 0
    rows = {r["id"]: r for r in _rows(out)}
    assert rows["short"]["gen"]["budget_hit"] is False and rows["short"]["gen"]["reasoning_tokens"] == 20
    assert rows["long"]["gen"]["budget_hit"] is True and rows["long"]["gen"]["reasoning_tokens"] == 40
    assert rows["long"]["messages"][-1]["reasoning_content"].startswith("thinking")
    sent = [c for c in _Handler.calls if c.get("thinking_budget")]
    assert sent and all(c["thinking_budget"] == 40 for c in sent)
    # a server gen did not start sees the switch and the kwargs on the request itself
    assert all(c["thinking"] == "on" for c in sent)
    assert all(c["chat_template_kwargs"] == {"preserve_thinking": True} for c in sent)
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["chat_template_kwargs"] == {"preserve_thinking": True}
    assert side["thinking_budget"] == 40 and side["run"]["budget_hits"] == 1


def test_gen_counts_the_trace_itself_when_the_server_reports_no_count(tmp_path, stub_server, monkeypatch):
    """The stub reports reasoning tokens in its timings; strip them and the
    verb re-tokenizes the trace with the teacher's tokenizer."""
    real = gen._reasoning_tokens
    monkeypatch.setattr(gen, "_reasoning_tokens", lambda obj: None)
    import gmlx.distill.tokens as tokens_mod
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda path: _WordTokenizer())
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "short", "messages": [{"role": "user", "content": "brief one"}]},
        {"id": "long", "messages": [{"role": "user", "content": "LONG one"}]},
    ])
    out = tmp_path / "corpus.jsonl"
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, thinking=True,
                          thinking_budget=40, tokenizer="teacher.gguf")
    assert gen.run_gen(opts) == 0
    rows = {r["id"]: r for r in _rows(out)}
    assert rows["short"]["gen"] == {**rows["short"]["gen"], "reasoning_tokens": 20, "budget_hit": False}
    assert rows["long"]["gen"] == {**rows["long"]["gen"], "reasoning_tokens": 40, "budget_hit": True}
    assert real is not gen._reasoning_tokens
    # a running server and a budget need a tokenizer to count with
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))
    assert gen.run_gen(gen.GenOptions(out=str(tmp_path / "x.jsonl"), prompts=prompts, base_url=stub_server,
                                      thinking=True, thinking_budget=40)) == 2
    assert gen.run_gen(gen.GenOptions(out=str(tmp_path / "y.jsonl"), prompts=prompts, base_url=stub_server,
                                      thinking=True, thinking_budget=40, tokenizer="missing.gguf")) == 2


def test_gen_builds_continuation_prompts_from_a_corpus(tmp_path, stub_server):
    corpus = tmp_path / "docs.jsonl"
    long_doc = " ".join(f"w{i}" for i in range(400))
    corpus.write_text(json.dumps({"text": long_doc}) + "\n" + json.dumps({"text": "too short"}) + "\n")
    out = tmp_path / "corpus.jsonl"
    opts = gen.GenOptions(out=str(out), corpus=str(corpus), base_url=stub_server, prefix_chars=100,
                          min_chars=500)
    assert gen.run_gen(opts) == 0
    rows = _rows(out)
    assert len(rows) == 1
    user = rows[0]["messages"][0]["content"]
    assert user.startswith(gen.CONTINUE_INSTRUCTION + "\n\n")
    prefix = user.split("\n\n", 1)[1]
    assert len(prefix) <= 100 and not prefix.endswith(" ") and long_doc.startswith(prefix)
    assert "student_messages" not in rows[0]
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["instruction"] == gen.CONTINUE_INSTRUCTION and side["prefix_chars"] == 100


def test_gen_refuses_before_any_request(tmp_path, stub_server, capsys):
    out = str(tmp_path / "c.jsonl")
    assert gen.run_gen(gen.GenOptions(out=out, base_url=stub_server)) == 2
    assert "--prompts or --corpus" in capsys.readouterr().err
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "x"}]}])
    assert gen.run_gen(gen.GenOptions(out=out, prompts=prompts)) == 2
    assert "--teacher or --base-url" in capsys.readouterr().err
    assert gen.run_gen(gen.GenOptions(out=out, prompts=prompts, base_url=stub_server, thinking_budget=5)) == 2
    assert "--thinking" in capsys.readouterr().err
    bad = _prompts(tmp_path / "bad.jsonl", [{"id": "a", "messages": [{"role": "assistant", "content": "x"}]}])
    assert gen.run_gen(gen.GenOptions(out=out, prompts=bad, base_url=stub_server)) == 2
    assert "end on a user turn" in capsys.readouterr().err
    assert _Handler.calls == []


def test_gen_refuses_a_dataset_id_without_datasets_and_an_out_it_cannot_write(tmp_path, stub_server, capsys,
                                                                               monkeypatch):
    """Exit 1 is for requests to rerun, so an input gen cannot read is
    refused with exit 2."""
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "x"}]}])
    monkeypatch.setitem(sys.modules, "datasets", None)
    assert gen.run_gen(gen.GenOptions(out=str(tmp_path / "c.jsonl"), corpus="someorg/ds",
                                      base_url=stub_server)) == 2
    assert "needs the datasets package" in capsys.readouterr().err
    ro = tmp_path / "ro"
    ro.mkdir()
    unread = tmp_path / "unread.jsonl"
    unread.write_text("")
    ro.chmod(0o500)
    unread.chmod(0o000)
    try:
        assert gen.run_gen(gen.GenOptions(out=str(ro / "sub" / "c.jsonl"), prompts=prompts,
                                          base_url=stub_server)) == 2
        assert "[gen] refuse:" in capsys.readouterr().err
        assert gen.run_gen(gen.GenOptions(out=str(unread), prompts=prompts, base_url=stub_server)) == 2
        assert "[gen] refuse:" in capsys.readouterr().err
    finally:
        ro.chmod(0o700)
        unread.chmod(0o600)
    assert _Handler.calls == []


def test_gen_refuses_an_empty_prompt_set_and_names_a_file_that_is_not_utf8(tmp_path, stub_server, capsys):
    """A prompt set with nothing to run exited 0 and wrote no --out, so the
    next step failed away from the cause, and a prompts or context file
    that is not UTF-8 was refused with a codec message naming no file."""
    corpus = tmp_path / "docs.jsonl"
    corpus.write_text(json.dumps({"id": "d", "text": "too short"}) + "\n")
    out = tmp_path / "c.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), corpus=str(corpus), base_url=stub_server)) == 2
    assert (f"[gen] refuse: {corpus} yields no prompts (blank documents and those under --min-chars 2000 are "
            "skipped)") in capsys.readouterr().err
    latin = tmp_path / "p.jsonl"
    latin.write_bytes(b'{"id": "a", "messages": [{"role": "user", "content": "caf\xe9"}]}\n')
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=str(latin), base_url=stub_server)) == 2
    assert f"[gen] refuse: {latin}: not UTF-8 (byte " in capsys.readouterr().err
    prompts = _prompts(tmp_path / "ok.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "x"}]}])
    ctx = tmp_path / "ctx.txt"
    ctx.write_bytes(b"the caf\xe9 doc\n")
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, context=str(ctx), base_url=stub_server)) == 2
    assert f"[gen] refuse: {ctx}: not UTF-8 (byte 7), convert it" in capsys.readouterr().err
    assert _Handler.calls == [] and not out.exists()


def test_gen_counts_a_failed_request_and_keeps_the_rest(tmp_path, stub_server, monkeypatch):
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
        {"id": "b", "messages": [{"role": "user", "content": "beta"}]},
    ])
    real = gen.complete

    def flaky(base_url, model_id, messages, opts, seed, tokenizer=None):
        if messages[-1]["content"] == "beta":
            raise gen.ServerError("HTTP 500 from stub")
        return real(base_url, model_id, messages, opts, seed, tokenizer)
    monkeypatch.setattr(gen, "complete", flaky)
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 1
    assert [r["id"] for r in _rows(out)] == ["a"]
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["run"] == {**side["run"], "completed": 1, "failed": 1}


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

def _row(id_, content, finish="stop", tokens=30, budget_hit=False, reasoning=None, extra=None):
    reply = {"role": "assistant", "content": content}
    if reasoning:
        reply["reasoning_content"] = reasoning
    row = {"id": id_, "messages": [{"role": "user", "content": f"prompt {id_}"}, reply],
           "gen": {"finish_reason": finish, "completion_tokens": tokens, "budget_hit": budget_hit}}
    row.update(extra or {})
    return row


GOOD = " ".join(f"word{i}" for i in range(40))


def test_filter_reasons_in_order_and_the_sidecar(tmp_path):
    rows = [
        _row("ok", GOOD),
        _row("cut", GOOD, finish="length"),
        _row("budget", GOOD, budget_hit=True),
        _row("empty", "three short words"),
        _row("marker", GOOD + " <|im_end|> " + GOOD),
        _row("repeat", ("same eight words repeated here again and again " * 6)),
        _row("lines", "\n".join(["one distinct line of enough words to pass the count"] * 4 + [GOOD])),
        _row("ascii", GOOD + " " + chr(0xE9) * 100),
        _row("tokens", GOOD, tokens=2000),
    ]
    src = tmp_path / "gen.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "gen.jsonl.gen.json").write_text(json.dumps({"gen_version": "3", "filter_version": None}))
    out = tmp_path / "ok.jsonl"
    opts = flt.FilterOptions(inputs=[str(src)], out=str(out), report=str(tmp_path / "r.json"),
                             rejects=str(tmp_path / "rej.jsonl"), max_non_ascii=0.2, max_reply_tokens=1180)
    assert flt.run_filter(opts) == 0
    assert [r["id"] for r in _rows(out)] == ["ok"]
    rej = {r["id"]: r["reason"] for r in _rows(tmp_path / "rej.jsonl")}
    assert rej == {"cut": "length", "budget": "budget", "empty": "empty", "marker": "marker",
                   "repeat": "repeat", "lines": "repeat", "ascii": "ascii", "tokens": "tokens"}
    side = json.loads((tmp_path / "ok.jsonl.gen.json").read_text())
    assert side["filter_version"] == flt.FILTER_VERSION
    assert side["filter"]["kept"] == 1 and side["filter"]["dropped"]["repeat"] == 2
    assert side["filter"]["params"]["max_reply_tokens"] == 1180 and side["filter"]["verify"] is None
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["kept"] == 1 and report["dropped"] == side["filter"]["dropped"]
    # a second pass chains the version and keeps a budget hit on request
    (tmp_path / "ok.jsonl").write_text(json.dumps(_row("b2", GOOD, budget_hit=True)) + "\n")
    out2 = tmp_path / "ok2.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(out)], out=str(out2), keep_budget_hit=True)) == 0
    assert [r["id"] for r in _rows(out2)] == ["b2"]
    side2 = json.loads((tmp_path / "ok2.jsonl.gen.json").read_text())
    assert side2["filter_version"] == "4+4"
    assert [p["params"]["max_reply_tokens"] for p in side2["filter_history"]] == [1180]
    assert side2["filter"]["params"]["max_trace_repeat"] == 0.5


def test_filter_runs_the_verify_command_over_the_survivors(tmp_path):
    rows = [_row("a", GOOD), _row("b", GOOD, finish="length"), _row("c", GOOD)]
    src = tmp_path / "gen.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    checker = tmp_path / "check.py"
    checker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    row = json.loads(line)\n"
        "    print('ok' if row['id'] != 'c' else 'wrong answer')\n")
    out = tmp_path / "ok.jsonl"
    cmd = f"{sys.executable} {checker}"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=str(out), verify=cmd,
                                            rejects=str(tmp_path / "rej.jsonl"))) == 0
    assert [r["id"] for r in _rows(out)] == ["a"]
    rej = {r["id"]: r for r in _rows(tmp_path / "rej.jsonl")}
    assert rej["c"] == {"id": "c", "reason": "verify", "detail": "wrong answer"}
    assert rej["b"]["reason"] == "length"
    side = json.loads((tmp_path / "ok.jsonl.gen.json").read_text())
    assert side["filter"]["verify"] == cmd and side["filter"]["dropped"] == {"length": 1, "verify": 1}


def test_filter_refuses_what_it_cannot_read_or_write_with_the_file_named(tmp_path, capsys):
    """An input or a context that is not UTF-8 or cannot be opened, and an
    out whose folder cannot be made, exit 2 with a refusal, where a
    traceback or a bare codec message named no file."""
    src = tmp_path / "gen.jsonl"
    src.write_text(json.dumps(_row("a", GOOD)) + "\n")
    out = tmp_path / "ok.jsonl"
    latin = tmp_path / "latin.jsonl"
    latin.write_bytes(b'{"id": "b", "note": "caf\xe9"}\n')
    assert flt.run_filter(flt.FilterOptions(inputs=[str(latin)], out=str(out))) == 2
    assert f"[filter] refuse: {latin}: not UTF-8 (byte 24), convert it" in capsys.readouterr().err
    ctx = tmp_path / "ctx.txt"
    ctx.write_bytes(b"the caf\xe9 context\n")
    assert flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=str(out), context=str(ctx))) == 2
    assert f"[filter] refuse: {ctx}: not UTF-8 (byte 7), convert it" in capsys.readouterr().err
    ctx.write_text("the context\n")
    ro = tmp_path / "ro"
    ro.mkdir()
    unread = tmp_path / "unread.jsonl"
    unread.write_text(json.dumps(_row("c", GOOD)) + "\n")
    ro.chmod(0o500)
    unread.chmod(0o000)
    try:
        assert flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=str(ro / "sub" / "ok.jsonl"))) == 2
        assert "[filter] refuse:" in capsys.readouterr().err
        ctx.chmod(0o000)
        assert flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=str(out), context=str(ctx))) == 2
        assert "[filter] refuse:" in capsys.readouterr().err
        ctx.chmod(0o600)
        assert flt.run_filter(flt.FilterOptions(inputs=[str(unread)], out=str(out))) == 2
        assert f"[filter] refuse: cannot read {unread}" in capsys.readouterr().err
    finally:
        ro.chmod(0o700)
        unread.chmod(0o600)
        ctx.chmod(0o600)
    assert not out.exists()
    blocker = tmp_path / "file"
    blocker.write_text("")
    for flag, kw in (("--report", {"report": str(blocker / "r.json")}), ("--rejects", {"rejects": str(blocker / "r")})):
        assert flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=str(out), **kw)) == 2
        assert f"[filter] refuse: cannot write {flag} {blocker}" in capsys.readouterr().err
    assert not out.exists()


def test_filter_refuses_a_verify_command_that_fails_or_miscounts(tmp_path, capsys):
    src = tmp_path / "gen.jsonl"
    src.write_text(json.dumps(_row("a", GOOD)) + "\n")
    out = str(tmp_path / "ok.jsonl")
    assert flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=out, verify="exit 3")) == 2
    assert "exited 3" in capsys.readouterr().err
    assert flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=out, verify="cat >/dev/null")) == 2
    assert "0 verdicts for 1 rows" in capsys.readouterr().err
    assert flt.run_filter(flt.FilterOptions(inputs=["/nonexistent/x.jsonl"], out=out)) == 2


def test_filter_context_puts_the_context_on_the_teacher_side(tmp_path, capsys):
    ctx = tmp_path / "schema.txt"
    ctx.write_text("CREATE TABLE crews (id INT);\n")
    a = tmp_path / "student-a.jsonl"
    b = tmp_path / "student-b.jsonl"
    a.write_text(json.dumps(_row("a", GOOD, extra={"family": "f1"})) + "\n")
    (tmp_path / "student-a.jsonl.gen.json").write_text(json.dumps({"gen_version": "3", "model": "student"}))
    b.write_text(json.dumps(_row("b", GOOD)) + "\n")
    out = tmp_path / "onp.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a), str(b)], out=str(out), context=str(ctx))) == 0
    rows = _rows(out)
    assert [r["id"] for r in rows] == ["a", "b"]
    r = rows[0]
    assert r["messages"][0]["content"] == "CREATE TABLE crews (id INT);\n\nprompt a"
    assert r["student_messages"][0]["content"] == "prompt a"
    assert r["messages"][-1] == r["student_messages"][-1]
    assert r["gen"]["context"] is True and r["family"] == "f1"
    side = json.loads((tmp_path / "onp.jsonl.gen.json").read_text())
    assert side["model"] == "student" and side["context"] == str(ctx)
    assert side["recontext_from"] == [str(a), str(b)] and side["prompts"] == 2
    # a row that already carries a student list cannot take a second context
    out.write_text(json.dumps(rows[0]) + "\n")
    assert flt.run_filter(flt.FilterOptions(inputs=[str(out)], out=str(tmp_path / "x.jsonl"),
                                            context=str(ctx))) == 2
    assert "already carries student_messages" in capsys.readouterr().err


def test_recontext_row_needs_a_user_turn_before_the_reply():
    with pytest.raises(ValueError, match="user turn followed by the reply"):
        flt.recontext_row({"id": "x", "messages": [{"role": "assistant", "content": "r"}]}, "ctx")
    row = {"id": "y", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                                   {"role": "assistant", "content": "r"}]}
    out = flt.recontext_row(row, "ctx", "{context} | {prompt}")
    assert out["messages"][1]["content"] == "ctx | q" and out["messages"][0]["content"] == "s"
    assert out["student_messages"] == row["messages"]


def test_gen_sends_the_thinking_switch_both_ways(monkeypatch, tmp_path):
    """The request carries serve's own thinking control on or off, and the
    spawned server gets --thinking the same way, so serve maps the switch
    onto whatever variable the teacher's template reads (enable_thinking
    for Qwen, thinking_mode for MiniMax, and so on). Template kwargs the
    user passes go through verbatim."""
    off = gen.GenOptions(out="x.jsonl", prompts="p.jsonl")
    body = gen._sampling(off, 1)
    assert body["thinking"] == "off" and "enable_thinking" not in body and "chat_template_kwargs" not in body
    on = gen.GenOptions(out="x.jsonl", prompts="p.jsonl", thinking=True,
                        chat_template_kwargs='{"preserve_thinking": true}')
    body = gen._sampling(on, 1)
    assert body["thinking"] == "on" and body["chat_template_kwargs"] == {"preserve_thinking": True}
    from gmlx.serve import lifecycle
    seen: dict = {}

    def start(args, **kw):
        seen["args"] = list(args)
        return object(), None

    monkeypatch.setattr(gen, "port_listening", lambda host, port: False)
    monkeypatch.setattr(lifecycle, "start_background_nowait", start)
    gen.spawn_server(gen.GenOptions(out="x.jsonl", prompts="p.jsonl", teacher="t.gguf", serve_arg=["--kv-bits", "8"]),
                     tmp_path / "log")
    assert seen["args"] == ["t.gguf", "--thinking", "off", "--kv-bits", "8"]
    gen.spawn_server(gen.GenOptions(out="x.jsonl", prompts="p.jsonl", teacher="t.gguf", thinking=True,
                                    chat_template_kwargs='{"preserve_thinking": true}'), tmp_path / "log")
    assert seen["args"] == ["t.gguf", "--thinking", "on", "--chat-template-config", '{"preserve_thinking": true}']


def test_gen_resume_drops_a_torn_last_line_and_refuses_other_settings(tmp_path, stub_server, capsys):
    """A kill mid-write leaves a torn last line, which a resume cuts off
    and regenerates; a bad line elsewhere and a resume under other
    sampling settings are refused."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
            {"id": "b", "messages": [{"role": "user", "content": "beta"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    with open(out, "a", encoding="utf-8") as fh:
        fh.write('{"id": "c", "mess')
    prompts = _prompts(tmp_path / "p.jsonl", rows + [{"id": "c", "messages": [{"role": "user", "content": "gamma"}]}])
    # the settings check runs before the cut, so a refused resume leaves the file as it was
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, temperature=0.2))
    assert rc == 2 and out.read_text(encoding="utf-8").endswith('{"id": "c", "mess')
    capsys.readouterr()
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    assert "dropped a torn last line" in capsys.readouterr().err
    assert sorted(r["id"] for r in _rows(out)) == ["a", "b", "c"]
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, temperature=0.2))
    err = capsys.readouterr().err
    assert rc == 2 and "generated with other settings" in err and "sampling" in err
    text = out.read_text(encoding="utf-8")
    out.write_text(text.replace('"id": "b"', 'oops', 1), encoding="utf-8")
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server))
    assert rc == 2 and "is not a JSON row" in capsys.readouterr().err
    # a kill inside a multi-byte character leaves bytes no text read decodes
    out.write_text(text, encoding="utf-8")
    with open(out, "ab") as fh:
        fh.write(b'{"id": "d", "messages": [{"role": "user", "content": "\xe4\xb8')
    prompts = _prompts(tmp_path / "p.jsonl", rows + [{"id": "c", "messages": [{"role": "user", "content": "gamma"}]},
                                                     {"id": "d", "messages": [{"role": "user", "content": "delta"}]}])
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    assert "dropped a torn last line" in capsys.readouterr().err
    assert sorted(r["id"] for r in _rows(out)) == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# prompt ids, the start sidecar, several inputs, rows that are not rows
# ---------------------------------------------------------------------------


def test_directory_corpus_ids_carry_the_relative_path_and_prompt_ids_are_unique(tmp_path, capsys):
    from gmlx.distill import corpus as _corpus

    for sub in ("a", "b"):
        (tmp_path / "dc" / sub).mkdir(parents=True)
        (tmp_path / "dc" / sub / "data.jsonl").write_text(json.dumps({"text": f"doc {sub}"}) + "\n")
    ids = [did for did, _ in _corpus.iter_corpus(str(tmp_path / "dc"))]
    assert ids == ["a/data.jsonl:0", "b/data.jsonl:0"]
    rows = [{"id": "x", "messages": [{"role": "user", "content": "one"}]},
            {"id": "x", "messages": [{"role": "user", "content": "two"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    with pytest.raises(ValueError, match="appears twice"):
        gen.prompt_rows(gen.GenOptions(out="o.jsonl", prompts=prompts))
    rc = gen.run_gen(gen.GenOptions(out=str(tmp_path / "o.jsonl"), prompts=prompts, base_url="http://127.0.0.1:1"))
    assert rc == 2 and "appears twice" in capsys.readouterr().err


def test_gen_start_sidecar_replaces_a_stale_one(tmp_path, stub_server, monkeypatch):
    """A sidecar left beside a deleted output is replaced by this run's
    settings before the first request, so a run cut short leaves what a
    resume must compare against."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    side = out.with_suffix(out.suffix + ".gen.json")
    stale = {"gen_version": gen.GEN_VERSION, "model": "old", "seed": 9, "thinking": True, "thinking_budget": None,
             "sampling": {"temperature": 0.2, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 8},
             "chat_template_kwargs": None, "run": {"completed": 5}}
    side.write_text(json.dumps(stale), encoding="utf-8")

    def cut(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(gen, "complete", cut)
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)
    with pytest.raises(KeyboardInterrupt):
        gen.run_gen(opts)
    got = json.loads(side.read_text(encoding="utf-8"))
    assert got["run"] is None and got["seed"] == opts.seed
    assert {k: got[k] for k in gen.run_settings(opts)} == gen.run_settings(opts)


def test_filter_refuses_inputs_generated_with_other_settings(tmp_path, capsys):
    def corpus(name, thinking):
        p = tmp_path / name
        p.write_text(json.dumps(_row(name, "one two three four five six seven eight nine ten eleven twelve "
                                     "thirteen fourteen fifteen sixteen seventeen")) + "\n")
        side = {"gen_version": gen.GEN_VERSION, "model": "t.gguf", "seed": 1, "thinking": thinking,
                "thinking_budget": None, "chat_template_kwargs": None,
                "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "max_tokens": 64},
                "run": {"completed": 1}}
        p.with_suffix(p.suffix + ".gen.json").write_text(json.dumps(side))
        return str(p)

    a, b, c = corpus("a.jsonl", True), corpus("b.jsonl", False), corpus("c.jsonl", True)
    out = tmp_path / "out.jsonl"
    rc = flt.run_filter(flt.FilterOptions(inputs=[a, b], out=str(out)))
    err = capsys.readouterr().err
    assert rc == 2 and "generated with other settings" in err and "thinking" in err
    assert not out.exists()
    assert flt.run_filter(flt.FilterOptions(inputs=[a, c], out=str(out))) == 0
    side = json.loads(out.with_suffix(out.suffix + ".gen.json").read_text())
    assert side["thinking"] is True and side["filter"]["inputs"] == [a, c]


def test_filter_refuses_a_json_line_that_is_not_a_row(tmp_path, capsys):
    p = tmp_path / "in.jsonl"
    p.write_text(json.dumps(_row("a", "fine " * 20)) + "\n[1, 2]\n")
    with pytest.raises(ValueError, match="line 2: not a corpus row"):
        flt._read_rows(p)
    p.write_text(json.dumps(_row("a", "fine " * 20)) + "\n" + json.dumps({"id": "b"}) + "\n")
    rc = flt.run_filter(flt.FilterOptions(inputs=[str(p)], out=str(tmp_path / "o.jsonl")))
    assert rc == 2 and "not a corpus row" in capsys.readouterr().err


def test_gen_warns_on_rows_found_without_a_sidecar(tmp_path, stub_server, capsys):
    """An output with rows and no sidecar is continued under this run's
    settings, with a warning that names the rows it labels."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
            {"id": "b", "messages": [{"role": "user", "content": "beta"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    out.write_text(json.dumps({"id": "a", "messages": rows[0]["messages"] + [{"role": "assistant", "content": "x"}],
                               "gen": {"finish_reason": "stop"}}) + "\n")
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server))
    err = capsys.readouterr().err
    assert rc == 0 and "[gen] warn: 1 rows in" in err and "without a sidecar" in err
    assert out.with_suffix(out.suffix + ".gen.json").exists()
    assert [r["id"] for r in _rows(out)] == ["a", "b"]


def test_filter_refuses_a_row_whose_message_is_not_an_object(tmp_path, capsys):
    p = tmp_path / "in.jsonl"
    p.write_text(json.dumps(_row("a", "fine " * 20)) + "\n"
                 + json.dumps({"id": "b", "messages": ["not a message"]}) + "\n")
    with pytest.raises(ValueError, match="line 2: not a corpus row"):
        flt._read_rows(p)
    rc = flt.run_filter(flt.FilterOptions(inputs=[str(p)], out=str(tmp_path / "o.jsonl")))
    assert rc == 2 and "not a corpus row" in capsys.readouterr().err


def test_filter_names_the_sidecar_it_compares_against_and_warns_on_a_missing_one(tmp_path, capsys):
    """The refusal names the first input that carries a sidecar, and an
    input without one is joined under that sidecar with a warning."""
    def corpus(name, thinking):
        p = tmp_path / name
        p.write_text(json.dumps(_row(name, "fine " * 20)) + "\n")
        if thinking is not None:
            side = {"gen_version": gen.GEN_VERSION, "model": "t.gguf", "seed": 1, "thinking": thinking,
                    "thinking_budget": None, "chat_template_kwargs": None,
                    "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "max_tokens": 64},
                    "run": {"completed": 1}}
            p.with_suffix(p.suffix + ".gen.json").write_text(json.dumps(side))
        return str(p)

    plain, a, b = corpus("plain.jsonl", None), corpus("a.jsonl", True), corpus("b.jsonl", False)
    out = tmp_path / "out.jsonl"
    rc = flt.run_filter(flt.FilterOptions(inputs=[plain, a, b], out=str(out)))
    err = capsys.readouterr().err
    assert rc == 2 and f"than {a}" in err and "warn" not in err
    rc = flt.run_filter(flt.FilterOptions(inputs=[plain, a], out=str(out)))
    err = capsys.readouterr().err
    assert rc == 0 and "[filter] warn:" in err and "plain.jsonl has no sidecar" in err and "a.jsonl" in err
    side = json.loads(out.with_suffix(out.suffix + ".gen.json").read_text())
    assert side["thinking"] is True and side["filter"]["inputs"] == [plain, a]


# ---------------------------------------------------------------------------
# an interrupt stops the queue, a resume compares the
# teacher and the context
# ---------------------------------------------------------------------------


def test_gen_interrupt_cancels_the_queued_requests(tmp_path, stub_server, monkeypatch):
    """A Ctrl-C while replies are still queued must not let the pool drain
    the queue behind the user's back."""
    rows = [{"id": f"r{i}", "messages": [{"role": "user", "content": f"prompt {i}"}]} for i in range(12)]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    calls = []
    real = gen.complete

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    real_wait = gen.wait
    waits = []

    def interrupted(futs, **kw):
        waits.append(1)
        if len(waits) > 1:
            raise KeyboardInterrupt
        return real_wait(futs, **kw)

    monkeypatch.setattr(gen, "complete", counting)
    monkeypatch.setattr(gen, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, concurrency=1))
    assert len(calls) <= 3, len(calls)


def test_gen_resume_refuses_another_teacher_or_context(tmp_path, stub_server, capsys):
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("the context\n", encoding="utf-8")
    base = dict(out=str(out), prompts=prompts, base_url=stub_server, teacher="a.gguf")
    assert gen.run_gen(gen.GenOptions(**base)) == 0
    more = _prompts(tmp_path / "p.jsonl", rows + [{"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    rc = gen.run_gen(gen.GenOptions(**dict(base, prompts=more, teacher="b.gguf")))
    err = capsys.readouterr().err
    assert rc == 2 and "generated with other settings" in err and "model" in err
    rc = gen.run_gen(gen.GenOptions(**dict(base, prompts=more, context=str(ctx))))
    err = capsys.readouterr().err
    assert rc == 2 and "generated with other settings" in err and "context" in err
    assert gen.run_gen(gen.GenOptions(**dict(base, prompts=more))) == 0
    assert sorted(r["id"] for r in _rows(out)) == ["a", "b"]


def test_gen_resume_compares_the_context_format_as_recorded(tmp_path, stub_server, capsys):
    """Rows that carry their own context still go through the format, so
    another --context-format is refused; a context file that reaches no
    row records no format and resumes under the same flags; a teacher
    named by another spelling of its path is the same teacher."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}], "context": "own context"}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    more = _prompts(tmp_path / "p2.jsonl", rows + [{"id": "b", "messages": [{"role": "user", "content": "beta"}],
                                                    "context": "other context"}])
    out = tmp_path / "corpus.jsonl"
    base = dict(out=str(out), prompts=prompts, base_url=stub_server, teacher="a.gguf")
    assert gen.run_gen(gen.GenOptions(**base)) == 0
    rc = gen.run_gen(gen.GenOptions(**dict(base, prompts=more, context_format="CTX={context} Q={prompt}")))
    err = capsys.readouterr().err
    assert rc == 2 and "generated with other settings" in err and "context_format" in err
    assert gen.run_gen(gen.GenOptions(**dict(base, prompts=more, teacher=str(Path("a.gguf").absolute())))) == 0
    assert sorted(r["id"] for r in _rows(out)) == ["a", "b"]
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    out2 = tmp_path / "corpus2.jsonl"
    plain = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}]
    p3 = _prompts(tmp_path / "p3.jsonl", plain)
    # a blank context file is a mistake, not a run without a context
    assert gen.run_gen(gen.GenOptions(out=str(out2), prompts=p3, base_url=stub_server, context=str(empty))) == 2
    assert f"[gen] refuse: context file {empty} is blank" in capsys.readouterr().err
    assert not out2.exists()


def test_gen_keeps_a_bounded_window_of_requests_in_flight(tmp_path, stub_server, monkeypatch):
    """The pool never holds every prompt's future at once, so a long run
    does not keep every finished reply in memory."""
    rows = [{"id": f"r{i}", "messages": [{"role": "user", "content": f"prompt {i}"}]} for i in range(24)]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    peak = [0]
    live = [0]
    lock = threading.Lock()
    real_pool = gen.ThreadPoolExecutor

    class Counting(real_pool):
        def submit(self, fn, *a, **k):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])

            def done(_f):
                with lock:
                    live[0] -= 1
            f = super().submit(fn, *a, **k)
            f.add_done_callback(done)
            return f

    monkeypatch.setattr(gen, "ThreadPoolExecutor", Counting)
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, concurrency=2)) == 0
    assert len(_rows(out)) == 24
    assert peak[0] <= 4, peak[0]


def test_filter_keeps_the_old_corpus_when_the_write_fails(tmp_path, capsys):
    """The output is written beside its final name and moved into place,
    so a failed write leaves the earlier corpus intact."""
    import os
    import stat

    src = tmp_path / "gen.jsonl"
    src.write_text(json.dumps(_row("ok", GOOD)) + "\n")
    outdir = tmp_path / "locked"
    outdir.mkdir()
    out = outdir / "corpus.jsonl"
    out.write_text("earlier rows\n", encoding="utf-8")
    os.chmod(outdir, stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(outdir, os.W_OK):
            pytest.skip("the directory stays writable (running as root)")
        rc = flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=str(out)))
        err = capsys.readouterr().err
        assert rc == 2 and "refuse" in err and "corpus.jsonl" in err
        assert out.read_text(encoding="utf-8") == "earlier rows\n"
    finally:
        os.chmod(outdir, stat.S_IRWXU)


def test_filter_and_tables_leave_no_temp_file_when_the_move_fails(tmp_path, monkeypatch):
    """A write that fails after its temp file exists removes the temp
    file, so a rerun does not find a stale `.tmp` beside the output."""
    import numpy as np

    from gmlx.distill import align as _align

    src = tmp_path / "gen.jsonl"
    src.write_text(json.dumps(_row("ok", GOOD)) + "\n")
    out = tmp_path / "corpus.jsonl"

    def fail(*a, **k):
        raise OSError("moved nothing")

    monkeypatch.setattr(flt.os, "replace", fail)
    assert flt.run_filter(flt.FilterOptions(inputs=[str(src)], out=str(out))) == 2
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))
    monkeypatch.undo()
    t = _align.identity_tables(8, np.zeros(8, dtype=bool), "t", "s")
    monkeypatch.setattr(_align.os, "replace", fail)
    with pytest.raises(OSError, match="moved nothing"):
        _align.save_tables(tmp_path / "tables", t)
    assert not list((tmp_path / "tables").glob("*.tmp"))


def test_gen_records_absolute_names_and_refuses_a_shared_context_added_to_per_prompt_rows(tmp_path, stub_server,
                                                                                         capsys, monkeypatch):
    """The sidecar records the teacher and the context file as absolute
    paths, and a rerun that adds --context to rows that carried their
    own context is another run, since rows without one would now get
    it."""
    monkeypatch.chdir(tmp_path)
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("shared context", encoding="utf-8")
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}], "context": "own context"}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts="p.jsonl", base_url=stub_server, teacher="t.gguf")) == 0
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["model"] == str(tmp_path / "t.gguf") and side["context"] == "per-prompt"
    assert side["shared_context"] is None
    more = _prompts(tmp_path / "p2.jsonl", rows + [{"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=more, base_url=stub_server, teacher="t.gguf",
                                    context="ctx.txt"))
    err = capsys.readouterr().err
    assert rc == 2 and "generated with other settings" in err and "shared_context" in err
    out2 = tmp_path / "c2.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out2), prompts=prompts, base_url=stub_server, teacher="t.gguf",
                                      context="ctx.txt")) == 0
    side = json.loads((tmp_path / "c2.jsonl.gen.json").read_text())
    assert side["context"] == str(ctx) and side["shared_context"] == str(ctx)
    assert gen.run_gen(gen.GenOptions(out=str(out2), prompts=more, base_url=stub_server, teacher=str(tmp_path / "t.gguf"),
                                      context=str(ctx))) == 0
    assert sorted(r["id"] for r in _rows(out2)) == ["a", "b"]


def test_gen_refuses_a_context_added_to_an_older_per_prompt_run(tmp_path, stub_server, capsys):
    """A sidecar written before the shared-context key existed recorded
    per-prompt when no --context was given, so adding one now is another
    run."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}], "context": "own context"}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    side_path = tmp_path / "corpus.jsonl.gen.json"
    side = json.loads(side_path.read_text())
    del side["shared_context"]
    side_path.write_text(json.dumps(side))
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("shared", encoding="utf-8")
    more = _prompts(tmp_path / "p2.jsonl", rows + [{"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=more, base_url=stub_server, context=str(ctx)))
    err = capsys.readouterr().err
    assert rc == 2 and "generated with other settings" in err and "per-prompt" in err
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=more, base_url=stub_server)) == 0


def test_filter_joins_files_whose_sidecars_name_one_teacher_two_ways(tmp_path, monkeypatch):
    """A sidecar with relative names and one with absolute names for the
    same teacher and context file join; the filter's own context lands
    as an absolute path on both context keys."""
    monkeypatch.chdir(tmp_path)
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("CREATE TABLE crews (id INT);\n")
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps(_row("a", GOOD)) + "\n")
    b.write_text(json.dumps(_row("b", GOOD)) + "\n")
    common = {"gen_version": "3", "seed": 1, "thinking": False}
    (tmp_path / "a.jsonl.gen.json").write_text(json.dumps({**common, "model": "t.gguf", "context": "ctx.txt"}))
    (tmp_path / "b.jsonl.gen.json").write_text(json.dumps({**common, "model": str(tmp_path / "t.gguf"),
                                                            "context": str(ctx), "shared_context": str(ctx)}))
    out = tmp_path / "joined.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a), str(b)], out=str(out))) == 0
    assert [r["id"] for r in _rows(out)] == ["a", "b"]
    out2 = tmp_path / "ctx.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(b)], out=str(out2), context="ctx.txt")) == 0
    side = json.loads((tmp_path / "ctx.jsonl.gen.json").read_text())
    assert side["context"] == str(ctx) and side["shared_context"] == str(ctx)


def test_gen_resume_refuses_a_done_id_whose_prompt_changed(tmp_path, stub_server, capsys):
    """Ids from line numbers shift when a line is inserted, and an explicit
    id can be reused for another prompt; a resume compares each done id's
    prompt to the prompt file and refuses a mismatch instead of skipping
    the new prompt and answering the old one twice."""
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"messages": [{"role": "user", "content": "alpha"}]},
        {"messages": [{"role": "user", "content": "beta"}]},
    ])
    out = tmp_path / "corpus.jsonl"
    base = dict(out=str(out), prompts=prompts, base_url=stub_server)
    assert gen.run_gen(gen.GenOptions(**base)) == 0
    assert sorted(r["id"] for r in _rows(out)) == ["0", "1"]
    shifted = _prompts(tmp_path / "p.jsonl", [
        {"messages": [{"role": "user", "content": "gamma"}]},
        {"messages": [{"role": "user", "content": "alpha"}]},
        {"messages": [{"role": "user", "content": "beta"}]},
    ])
    capsys.readouterr()
    rc = gen.run_gen(gen.GenOptions(**dict(base, prompts=shifted)))
    err = capsys.readouterr().err
    assert rc == 2 and "[gen] refuse:" in err and "prompt" in err and "'0'" in err, err
    assert sorted(r["id"] for r in _rows(out)) == ["0", "1"]
    named = _prompts(tmp_path / "n.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}])
    out2 = tmp_path / "named.jsonl"
    assert gen.run_gen(gen.GenOptions(**dict(base, out=str(out2), prompts=named))) == 0
    renamed = _prompts(tmp_path / "n.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "delta"}]},
                                              {"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    rc = gen.run_gen(gen.GenOptions(**dict(base, out=str(out2), prompts=renamed)))
    err = capsys.readouterr().err
    assert rc == 2 and "'a'" in err, err
    gone = _prompts(tmp_path / "n.jsonl", [{"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    rc = gen.run_gen(gen.GenOptions(**dict(base, out=str(out2), prompts=gone)))
    err = capsys.readouterr().err
    assert rc == 2 and "'a'" in err and "not in" in err, err
    assert [r["id"] for r in _rows(out2)] == ["a"]


def test_gen_and_filter_compare_the_serve_args(tmp_path, stub_server, capsys):
    """The server flags shape every reply (a KV quantization, a draft
    model), so they are part of the settings a resume and a join compare."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    base = dict(out=str(out), prompts=prompts, base_url=stub_server, serve_arg=["--kv-bits", "4"])
    assert gen.run_gen(gen.GenOptions(**base)) == 0
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["serve_args"] == ["--kv-bits", "4"]
    more = _prompts(tmp_path / "p.jsonl", rows + [{"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    capsys.readouterr()
    rc = gen.run_gen(gen.GenOptions(**dict(base, prompts=more, serve_arg=[])))
    err = capsys.readouterr().err
    assert rc == 2 and "generated with other settings" in err and "serve_args" in err, err
    assert gen.run_gen(gen.GenOptions(**dict(base, prompts=more))) == 0
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps(_row("a", GOOD)) + "\n")
    b.write_text(json.dumps(_row("b", GOOD)) + "\n")
    common = {"gen_version": "3", "model": "t.gguf", "seed": 1, "thinking": False}
    (tmp_path / "a.jsonl.gen.json").write_text(json.dumps({**common, "serve_args": ["--kv-bits", "4"]}))
    (tmp_path / "b.jsonl.gen.json").write_text(json.dumps({**common, "serve_args": []}))
    rc = flt.run_filter(flt.FilterOptions(inputs=[str(a), str(b)], out=str(tmp_path / "j.jsonl")))
    err = capsys.readouterr().err
    assert rc == 2 and "serve_args" in err, err


def test_gen_refuses_a_missing_corpus_and_a_context_format_without_both_fields(tmp_path, stub_server, capsys):
    """A corpus path that does not exist and a context format that would
    drop the prompt or name an unknown field are refused before any
    request, in gen and in filter alike."""
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("the context\n", encoding="utf-8")
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}])
    out = tmp_path / "corpus.jsonl"
    rc = gen.run_gen(gen.GenOptions(out=str(out), corpus=str(tmp_path / "missing.jsonl"), base_url=stub_server))
    err = capsys.readouterr().err
    assert rc == 2 and "no corpus" in err, err
    for fmt in ("{context}\n\n", "{ctx}\n\n{prompt}", "{context} {prompt} {extra}"):
        rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, context=str(ctx),
                                        context_format=fmt))
        err = capsys.readouterr().err
        assert rc == 2 and "--context-format" in err, (fmt, err)
        a = tmp_path / "a.jsonl"
        a.write_text(json.dumps(_row("a", GOOD)) + "\n")
        rc = flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(tmp_path / "f.jsonl"), context=str(ctx),
                                              context_format=fmt))
        err = capsys.readouterr().err
        assert rc == 2 and "--context-format" in err, (fmt, err)
    assert not out.exists() and _Handler.calls == []


def test_filter_refuses_an_id_two_inputs_share(tmp_path, capsys):
    """Rows are keyed by id downstream (census pairs, gen resume), so two
    inputs that carry one id are refused instead of joined."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps(_row("a", GOOD)) + "\n" + json.dumps(_row("b", GOOD)) + "\n")
    b.write_text(json.dumps(_row("b", GOOD)) + "\n")
    out = tmp_path / "j.jsonl"
    rc = flt.run_filter(flt.FilterOptions(inputs=[str(a), str(b)], out=str(out)))
    err = capsys.readouterr().err
    assert rc == 2 and "[filter] refuse:" in err and "'b'" in err and str(b) in err, err
    assert not out.exists()
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(out))) == 0


# ---------------------------------------------------------------------------
# a resume against another served model, the sidecar totals after a
# resume, a corpus row without the text key
# ---------------------------------------------------------------------------

def test_gen_resume_refuses_a_server_serving_another_model(tmp_path, stub_server, capsys):
    """The sidecar records the model the server named; a resume against a
    server naming another refuses before any request, since the done
    replies and the new ones would come from different teachers."""
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
    ])
    out = tmp_path / "corpus.jsonl"
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)
    assert gen.run_gen(opts) == 0
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["served_model_id"] == "stub-teacher"
    _prompts(tmp_path / "p.jsonl", [
        {"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
        {"id": "b", "messages": [{"role": "user", "content": "beta"}]},
    ])
    _Handler.model_id = "other-teacher"
    _Handler.calls = []
    capsys.readouterr()
    assert gen.run_gen(opts) == 2
    err = capsys.readouterr().err
    assert "the server now serves other-teacher" in err and "came from stub-teacher" in err
    assert len(_Handler.calls) == 1 and [r["id"] for r in _rows(out)] == ["a"]
    assert json.loads((tmp_path / "corpus.jsonl.gen.json").read_text()) == side


def test_gen_sidecar_totals_after_a_resume(tmp_path, stub_server):
    """A prompt that failed is retried by the resume, so the sidecar's
    failed count is the last run's; completed and stops accumulate and the
    stop fraction is over every completed reply, not the last run's."""
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
        {"id": "b", "messages": [{"role": "user", "content": "beta CUT"}]},
    ])
    out = tmp_path / "corpus.jsonl"
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)
    _Handler.fail_once = {"beta"}
    assert gen.run_gen(opts) == 1
    run = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())["run"]
    assert run["completed"] == 1 and run["failed"] == 1 and run["stops"] == 1 and run["stop_fraction"] == 1.0
    assert gen.run_gen(opts) == 0
    run = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())["run"]
    assert [r["id"] for r in _rows(out)] == ["a", "b"]
    assert run["completed"] == 2 and run["failed"] == 0 and run["stops"] == 1
    assert run["stop_fraction"] == pytest.approx(0.5)


def test_gen_refuses_a_corpus_row_without_the_text_key(tmp_path, stub_server, capsys):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"txt": "the cat"}) + "\n")
    out = tmp_path / "corpus.jsonl"
    rc = gen.run_gen(gen.GenOptions(out=str(out), corpus=str(corpus), base_url=stub_server))
    err = capsys.readouterr().err
    assert rc == 2 and "c.jsonl line 1: no 'text' key" in err


def test_gen_sidecar_totals_survive_an_interrupted_run(tmp_path, stub_server, monkeypatch):
    """An interrupt leaves the sidecar with no run block; the resume's
    totals still cover every row in the file, since they are read from
    the rows."""
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
        {"id": "b", "messages": [{"role": "user", "content": "beta"}]},
    ])
    out = tmp_path / "corpus.jsonl"
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, concurrency=1)
    real = gen.reply_row

    def interrupt_on_b(r, c, seed):
        if r["id"] == "b":
            raise KeyboardInterrupt
        return real(r, c, seed)

    monkeypatch.setattr(gen, "reply_row", interrupt_on_b)
    with pytest.raises(KeyboardInterrupt):
        gen.run_gen(opts)
    assert [r["id"] for r in _rows(out)] == ["a"]
    assert json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())["run"] is None
    monkeypatch.setattr(gen, "reply_row", real)
    assert gen.run_gen(opts) == 0
    run = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())["run"]
    assert [r["id"] for r in _rows(out)] == ["a", "b"]
    assert run["completed"] == 2 and run["stops"] == 2 and run["stop_fraction"] == 1.0 and run["failed"] == 0


def test_gen_refuses_a_corpus_row_whose_text_is_not_a_string(tmp_path, stub_server, capsys):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"text": None}) + "\n")
    rc = gen.run_gen(gen.GenOptions(out=str(tmp_path / "o.jsonl"), corpus=str(corpus), base_url=stub_server))
    err = capsys.readouterr().err
    assert rc == 2 and "c.jsonl line 1: 'text' is not a string" in err


def test_prompt_rows_name_the_line_of_a_bad_prompt(tmp_path, capsys):
    """A prompt line that is not an object, or whose messages are not a
    list of messages, is refused with its line rather than a crash."""
    import re

    p = tmp_path / "p.jsonl"
    cases = (("[1, 2]", "prompt 0 of p.jsonl: not a JSON object"),
             ('{"messages": "hi"}', "prompt 0 of p.jsonl: 'messages' is not a list of messages"),
             ('{"messages": [{"role": "user", "content": null}]}',
              "prompt 0 of p.jsonl: message 0 of 'messages' has no content"),
             ('{"messages": [{"role": "user", "content": "hi"}]}\n{"messages": [{"role": "assistant", "content": "x"}]}',
              "prompt 1 of p.jsonl: messages must end on a user turn"),
             ("{not json", "prompt 0 of p.jsonl: not JSON ("))
    for text, why in cases:
        p.write_text(text + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=re.escape(why)):
            gen.prompt_rows(gen.GenOptions(out="o.jsonl", prompts=str(p)))
    p.write_text("[1, 2]\n", encoding="utf-8")
    rc = gen.run_gen(gen.GenOptions(out=str(tmp_path / "o.jsonl"), prompts=str(p), base_url="http://127.0.0.1:1"))
    assert rc == 2 and "[gen] refuse: prompt 0 of p.jsonl: not a JSON object" in capsys.readouterr().err


def test_filter_refuses_a_message_whose_content_is_not_a_string(tmp_path, capsys):
    """A reply whose content is a list of parts, or a user turn with null
    content, is refused by the reader with its line, so filter exits 2
    instead of crashing on the word count."""
    p = tmp_path / "in.jsonl"
    p.write_text(json.dumps(_row("a", "fine " * 20)) + "\n" + json.dumps({"id": "b", "messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": ["fine " * 20]}]}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: message 1 of 'messages' content is not a string"):
        flt._read_rows(p)
    rc = flt.run_filter(flt.FilterOptions(inputs=[str(p)], out=str(tmp_path / "o.jsonl")))
    assert rc == 2 and "[filter] refuse:" in capsys.readouterr().err
    p.write_text(json.dumps({"id": "b", "messages": [{"role": "user", "content": None},
                                                     {"role": "assistant", "content": "fine " * 20}]}) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: message 0 of 'messages' has no content"):
        flt._read_rows(p)


def test_gen_sidecar_wall_time_carries_over_a_resume_with_added_prompts(tmp_path, stub_server):
    """Prompts appended to the file change the prompt set, not the runs
    the wall time counts."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    side = out.with_suffix(out.suffix + ".gen.json")
    first = json.loads(side.read_text())
    rows.append({"id": "b", "messages": [{"role": "user", "content": "beta"}]})
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    second = json.loads(side.read_text())
    assert second["prompt_set_sha256"] != first["prompt_set_sha256"]
    assert second["run"]["completed"] == 2 and second["run"]["wall_s"] > first["run"]["wall_s"] > 0


def test_filter_checks_the_reasoning_trace_for_markers_and_repeats():
    """reply-think trains on the trace, so a leaked marker or a looping
    trace fails the row the way the answer would; the think tags that
    delimit a trace are not a leak."""
    opts = flt.FilterOptions(inputs=[], out="o.jsonl")
    assert flt.reason(_row("a", GOOD, reasoning="thinking <|im_start|> more"), opts) == "marker"
    assert flt.reason(_row("b", GOOD, reasoning="the same line\n" * 5), opts) == "repeat"
    assert flt.reason(_row("c", GOOD, reasoning="<think>a short thought</think>"), opts) is None
    assert flt.reason(_row("d", GOOD), opts) is None


def test_prompt_rows_refuse_a_blank_or_non_string_context(tmp_path):
    """A prompt's own context is a non-empty string; a blank or numeric
    one is refused rather than silently replaced by the shared context,
    and a null one means the row has none."""
    import re

    shared = tmp_path / "ctx.txt"
    shared.write_text("shared context", encoding="utf-8")
    p = tmp_path / "p.jsonl"
    msgs = [{"role": "user", "content": "alpha"}]
    for bad in ("", "  ", 5):
        p.write_text(json.dumps({"id": "a", "messages": msgs, "context": bad}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=re.escape("prompt 0 of p.jsonl: context must be a non-empty string")):
            gen.prompt_rows(gen.GenOptions(out="o.jsonl", prompts=str(p), context=str(shared)))
    p.write_text(json.dumps({"id": "a", "messages": msgs, "context": None}) + "\n"
                 + json.dumps({"id": "b", "messages": msgs, "context": "own"}) + "\n", encoding="utf-8")
    rows = gen.prompt_rows(gen.GenOptions(out="o.jsonl", prompts=str(p), context=str(shared)))
    assert "shared context" in rows[0]["messages"][-1]["content"]
    assert rows[1]["messages"][-1]["content"].startswith("own")



def test_gen_refuses_a_blank_context_file_and_a_prompt_row_carrying_student_messages(tmp_path, stub_server, capsys):
    """A blank --context is a mistake, not a run without a context, and a
    prompt row that already carries student_messages would be overwritten
    silently; both refuse before the output is touched."""
    blank = tmp_path / "blank.txt"
    blank.write_text(" \n", encoding="utf-8")
    p = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}])
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=p, base_url=stub_server, context=str(blank))) == 2
    assert f"[gen] refuse: context file {blank} is blank" in capsys.readouterr().err
    with pytest.raises(ValueError, match="is blank"):
        gen.prompt_rows(gen.GenOptions(out=str(out), prompts=p, context=str(blank)))
    q = _prompts(tmp_path / "q.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}],
                                        "student_messages": [{"role": "user", "content": "alpha"}]}])
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=q, base_url=stub_server)) == 2
    assert "prompt 0 of q.jsonl: student_messages is written by gen, not read" in capsys.readouterr().err
    assert not out.exists()


def test_filter_refuses_a_blank_context_file(tmp_path, capsys):
    blank = tmp_path / "blank.txt"
    blank.write_text("\n\n", encoding="utf-8")
    a = tmp_path / "a.jsonl"
    a.write_text(json.dumps(_row("a", GOOD)) + "\n")
    (tmp_path / "a.jsonl.gen.json").write_text(json.dumps({"gen_version": "3", "model": "student"}))
    out = tmp_path / "o.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(out), context=str(blank))) == 2
    assert f"[filter] refuse: context file {blank} is blank" in capsys.readouterr().err
    assert not out.exists()


def test_line_run_check_skips_lines_without_a_letter_or_digit():
    """Nested code and JSON close with the same bracket line several times
    in a row; those lines are structure, not a loop."""
    opts = flt.FilterOptions(inputs=[], out="o.jsonl")
    nested = GOOD + "\n{\n  \"a\": {\n    \"b\": {\n      \"c\": {}\n    }\n  }\n}\n"
    assert flt.max_line_run(nested) == 1
    assert flt.reason(_row("j", nested), opts) is None
    assert flt.max_line_run("x\n---\n---\n---\nyes\nyes\nyes\n") == 3
    assert flt.reason(_row("k", GOOD + "\nyes\nyes\nyes\n"), opts) == "repeat"


def test_repeat_and_word_checks_count_characters_on_cjk_text():
    """CJK prose has no spaces, so whitespace n-grams never repeat and the
    word count is one; both checks count characters there."""
    opts = flt.FilterOptions(inputs=[], out="o.jsonl")
    phrase = "\u4eca\u5929\u5929\u6c14\u5f88\u597d"
    looping = phrase * 12
    distinct = "".join(chr(0x4E00 + 7 * i) for i in range(60))
    assert flt.repeat_fraction(looping, 8) > 0.5
    assert flt.repeat_fraction(distinct, 8) == 0.0
    assert flt.word_count(distinct) == 60 and flt.word_count(GOOD) == 40
    assert flt.reason(_row("cjk", looping), opts) == "repeat"
    assert flt.reason(_row("ok", distinct), opts) is None
    assert flt.reason(_row("short", phrase), opts) == "empty"


def test_trace_repeat_threshold_is_its_own_flag():
    """A reasoning trace restates and rechecks, so its repeat threshold is
    looser than the answer's and set on its own."""
    trace = ("let me check this again and again for the answer " * 4) + " ".join(f"step{i}" for i in range(60))
    frac = flt.repeat_fraction(trace, 8)
    assert 0.2 < frac < 0.5
    assert flt.reason(_row("a", GOOD, reasoning=trace), flt.FilterOptions(inputs=[], out="o.jsonl")) is None
    assert flt.reason(_row("a", GOOD, reasoning=trace),
                      flt.FilterOptions(inputs=[], out="o.jsonl", max_trace_repeat=0.2)) == "repeat"
    assert flt.reason(_row("b", trace), flt.FilterOptions(inputs=[], out="o.jsonl")) == "repeat"



def test_cjk_units_split_ideographs_and_kana_only_and_leave_hangul_and_code_alone():
    """Ideographs and kana split per character, with the iteration marks,
    ideographic zero and half-width kana among them; code, Hangul and
    punctuation stay whitespace tokens, so an unspaced Chinese answer with
    inline code is not empty and a dashed rule line is not a loop."""
    opts = flt.FilterOptions(inputs=[], out="o.jsonl")
    cjk_code = ("\u8fd9\u662f\u4e00\u4e2a\u4f8b\u5b50\uff0c\u8bf7\u770b\u4ee3\u7801 `x = foo(bar)` "
                "\u7136\u540e\u8fd0\u884c `python run.py --fast` \u5c31\u53ef\u4ee5\u4e86\u3002")
    assert flt.word_count(cjk_code) >= 20
    assert flt.reason(_row("a", cjk_code), opts) is None
    rule = "\u7b2c\u4e00\u6b65\u5b8c\u6210\u3002\n" + "-" * 40 + "\n" + "".join(
        chr(0x4E00 + 3 * i) for i in range(40))
    assert flt.repeat_fraction(rule, 8) == 0.0
    assert flt.reason(_row("b", rule), opts) is None
    hangul = " ".join(f"\ud55c\uae00{i}" for i in range(20))
    assert flt.word_count(hangul) == 20
    assert flt.word_count("a b c") == 3 and flt.word_count("\u4eca\u5929 \u5929\u6c14") == 4
    assert flt.word_count("\u4eba\u3005\u306f\u6642\u3005\uff71\uff72\uff73\uff74\uff75\u3007") == 11
    assert flt.word_count("\u300c\u5f15\u7528\u300d\uff61") == 2
    loop = "\u4eca\u5929\u5929\u6c14\u5f88\u597d" * 12
    assert flt.repeat_fraction(loop, 8) > 0.5 and flt.reason(_row("c", loop), opts) == "repeat"


def test_filter_sidecar_records_every_pass_and_refuses_inputs_filtered_differently(tmp_path, capsys):
    """The sidecar becomes the cache manifest's generator block, so it
    records the trace threshold, keeps each earlier pass under
    filter_history, and never joins an input filtered under one version
    with one filtered under another or not at all."""
    a = tmp_path / "a.jsonl"
    a.write_text(json.dumps(_row("a", GOOD)) + "\n")
    (tmp_path / "a.jsonl.gen.json").write_text(json.dumps({"gen_version": "3", "model": "m"}))
    first = tmp_path / "first.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(first), max_trace_repeat=0.3)) == 0
    side = json.loads((tmp_path / "first.jsonl.gen.json").read_text())
    assert side["filter_version"] == flt.FILTER_VERSION == "4"
    assert side["filter"]["params"]["max_trace_repeat"] == 0.3 and "filter_history" not in side
    second = tmp_path / "second.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(first)], out=str(second), max_trace_repeat=0.4)) == 0
    side2 = json.loads((tmp_path / "second.jsonl.gen.json").read_text())
    assert side2["filter_version"] == "4+4" and side2["filter"]["params"]["max_trace_repeat"] == 0.4
    assert [p["params"]["max_trace_repeat"] for p in side2["filter_history"]] == [0.3]
    b = tmp_path / "b.jsonl"
    b.write_text(json.dumps(_row("b", GOOD)) + "\n")
    (tmp_path / "b.jsonl.gen.json").write_text(json.dumps({"gen_version": "3", "model": "m"}))
    assert flt.run_filter(flt.FilterOptions(inputs=[str(first), str(b)], out=str(tmp_path / "j.jsonl"))) == 2
    err = capsys.readouterr().err
    assert f"[filter] refuse: {b} was filtered differently than {first}" in err
    assert not (tmp_path / "j.jsonl").exists()


def test_filter_and_gen_refuse_a_torn_sidecar_with_exit_2(tmp_path, stub_server, capsys):
    """Exit 1 is reserved for requests that failed and can be rerun; a
    sidecar that is not a JSON object is a refusal."""
    a = tmp_path / "a.jsonl"
    a.write_text(json.dumps(_row("a", GOOD)) + "\n")
    (tmp_path / "a.jsonl.gen.json").write_text("{\"gen_version\": ")
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(tmp_path / "o.jsonl"))) == 2
    assert f"[filter] refuse: {a}.gen.json is not a JSON object" in capsys.readouterr().err
    (tmp_path / "a.jsonl.gen.json").write_text("[1, 2]")
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(tmp_path / "o.jsonl"))) == 2
    assert f"[filter] refuse: {a}.gen.json is not a JSON object" in capsys.readouterr().err
    p = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}])
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=p, base_url=stub_server)) == 0
    side = tmp_path / "corpus.jsonl.gen.json"
    side.write_text("{\"gen_version\": ")
    capsys.readouterr()
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=p, base_url=stub_server)) == 2
    assert f"[gen] refuse: {side} is not a JSON object" in capsys.readouterr().err


def test_gen_refuses_a_prompt_source_that_cannot_be_read_with_exit_2(tmp_path, stub_server, capsys, monkeypatch):
    """A Hugging Face id that does not exist raises an OSError subclass
    from the datasets library; that is a refusal, not a failed request."""
    def boom(opts):
        raise FileNotFoundError("Dataset 'data/texts' doesn't exist on the Hub")

    monkeypatch.setattr(gen, "prompt_rows", boom)
    rc = gen.run_gen(gen.GenOptions(out=str(tmp_path / "o.jsonl"), corpus="data/texts", base_url=stub_server))
    assert rc == 2 and "[gen] refuse: Dataset 'data/texts' doesn't exist" in capsys.readouterr().err


def test_filter_refuses_a_corpus_without_gen_blocks(tmp_path, capsys):
    """Rows without a gen block would all drop as length and the filter
    would write an empty corpus with exit 0."""
    a = tmp_path / "a.jsonl"
    rows = [{"id": "a", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": GOOD}]}]
    a.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "o.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(out))) == 2
    assert f"[filter] refuse: no row of {a} carries a gen block" in capsys.readouterr().err
    assert not out.exists()


def test_gen_adds_the_thinking_budget_to_the_answer_budget(tmp_path, stub_server, monkeypatch):
    """The server counts the reasoning trace in max_tokens, so with a
    thinking budget the request carries answer budget plus trace budget
    plus the forced close (the wrap phrase and the closing marker, as the
    tokenizer counts them, or the allowance when it resolves no marker)
    and the answer keeps the budget --max-tokens names. Without thinking
    the request carries --max-tokens as given."""
    import gmlx.distill.tokens as tokens_mod
    from gmlx.gen.thinking_budget import BUDGET_WRAP_PHRASE
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda path: _ThinkTokenizer())
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "q"}]}])
    assert gen.run_gen(gen.GenOptions(out=str(tmp_path / "a.jsonl"), prompts=prompts, base_url=stub_server,
                                      thinking=True, thinking_budget=5, max_tokens=7,
                                      tokenizer="teacher.gguf")) == 0
    close = len(BUDGET_WRAP_PHRASE.split()) + 1
    sent = [c["max_tokens"] for c in _Handler.calls if "messages" in c and c.get("max_tokens") != 1]
    assert sent == [7 + 5 + close] and close > gen.CLOSE_ALLOWANCE == 4
    side = json.loads((tmp_path / "a.jsonl.gen.json").read_text())
    assert side["request_max_tokens"] == 7 + 5 + close
    _Handler.calls = []
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda path: _WordTokenizer())
    assert gen.run_gen(gen.GenOptions(out=str(tmp_path / "a2.jsonl"), prompts=prompts, base_url=stub_server,
                                      thinking=True, thinking_budget=5, max_tokens=7,
                                      tokenizer="teacher.gguf")) == 0
    sent = [c["max_tokens"] for c in _Handler.calls if "messages" in c and c.get("max_tokens") != 1]
    assert sent == [7 + 5 + gen.CLOSE_ALLOWANCE]
    _Handler.calls = []
    assert gen.run_gen(gen.GenOptions(out=str(tmp_path / "b.jsonl"), prompts=prompts, base_url=stub_server,
                                      max_tokens=7)) == 0
    sent = [c["max_tokens"] for c in _Handler.calls if "messages" in c and c.get("max_tokens") != 1]
    assert sent == [7]


def test_filter_join_refuses_other_filter_settings_and_sums_the_run_totals(tmp_path, capsys):
    """Two filtered inputs joined under one sidecar must have been filtered
    with the same settings, and the joined sidecar's prompt count, run
    totals and prompt fields cover every input, not the first alone."""
    def gen_file(name, n, wall, stops, longest):
        p = tmp_path / f"{name}.jsonl"
        p.write_text("".join(json.dumps(_row(f"{name}{i}", GOOD, tokens=10)) + "\n" for i in range(n)))
        (tmp_path / f"{name}.jsonl.gen.json").write_text(json.dumps({
            "gen_version": "3", "model": "m", "prompts": n, "prompt_source": f"{name}-prompts.jsonl",
            "prompt_set_sha256": name * 4, "run": {"completed": n, "failed": 1, "wall_s": wall,
                                                   "stops": stops, "stop_fraction": stops / n,
                                                   "longest_stopped_reply_tokens": longest,
                                                   "tok_s_aggregate": 9.0, "concurrency": 2}}))
        return p

    a, b = gen_file("a", 2, 3.0, 1, 10), gen_file("b", 3, 4.0, 2, 30)
    fa, fb = tmp_path / "fa.jsonl", tmp_path / "fb.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(fa), min_words=4)) == 0
    assert flt.run_filter(flt.FilterOptions(inputs=[str(b)], out=str(fb), min_words=5)) == 0
    assert flt.run_filter(flt.FilterOptions(inputs=[str(fa), str(fb)], out=str(tmp_path / "j.jsonl"))) == 2
    err = capsys.readouterr().err
    assert f"[filter] refuse: {fb} was filtered with other settings than {fa} (min_words)" in err
    assert not (tmp_path / "j.jsonl").exists()
    assert flt.run_filter(flt.FilterOptions(inputs=[str(b)], out=str(fb), min_words=4)) == 0
    assert flt.run_filter(flt.FilterOptions(inputs=[str(fa), str(fb)], out=str(tmp_path / "j.jsonl"))) == 0
    side = json.loads((tmp_path / "j.jsonl.gen.json").read_text())
    assert side["prompts"] == 5 and side["prompt_source"] == ["a-prompts.jsonl", "b-prompts.jsonl"]
    assert side["joined"] == [str(fa), str(fb)]
    assert side["prompt_set_sha256"] not in ("aaaa", "bbbb") and len(side["prompt_set_sha256"]) == 64
    run = side["run"]
    assert run["completed"] == 5 and run["failed"] == 2 and run["wall_s"] == 7.0
    assert run["stops"] == 3 and run["stop_fraction"] == pytest.approx(0.6)
    assert run["longest_stopped_reply_tokens"] == 30
    assert run["kept"] == {**run["kept"], "completed": 5, "generated_tokens": 50}
    assert run["concurrency"] == 2 and "tok_s_aggregate" not in run
    # a single input keeps its gen run block as it was
    single = json.loads((tmp_path / "fa.jsonl.gen.json").read_text())
    assert single["run"]["tok_s_aggregate"] == 9.0 and single["prompts"] == 2 and "joined" not in single


def test_filter_accepts_an_empty_input(tmp_path, capsys):
    """A corpus with no rows (every gen request failed, or an earlier pass
    kept nothing) is not "not a generated corpus": the filter writes an
    empty output and exits 0."""
    a = tmp_path / "a.jsonl"
    a.write_text("")
    out = tmp_path / "o.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(out))) == 0
    assert out.exists() and out.read_text() == ""
    assert "[filter] kept 0" in capsys.readouterr().err


def test_gen_counts_a_retokenized_trace_one_short_of_the_budget_as_a_hit(tmp_path, stub_server, monkeypatch):
    """A cut trace re-tokenized by gen can lose a token or two to merges,
    so a count within two of the budget is a hit; a count the server
    reports is exact and one below the budget is not."""
    import gmlx.distill.tokens as tokens_mod
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda path: _WordTokenizer())
    monkeypatch.setattr(gen, "_reasoning_tokens", lambda obj: None)
    monkeypatch.setattr(gen, "trace_tokens", lambda tok, text: len(text.split()) - 1)
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "long", "messages": [{"role": "user", "content": "LONG one"}]},
        {"id": "short", "messages": [{"role": "user", "content": "brief"}]},
    ])
    out = tmp_path / "a.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, thinking=True,
                                      thinking_budget=40, tokenizer="teacher.gguf")) == 0
    rows = {r["id"]: r for r in _rows(out)}
    assert rows["long"]["gen"]["reasoning_tokens"] == 39 and rows["long"]["gen"]["budget_hit"] is True
    assert rows["short"]["gen"]["reasoning_tokens"] == 19 and rows["short"]["gen"]["budget_hit"] is False
    monkeypatch.setattr(gen, "_reasoning_tokens", lambda obj: 39)
    out2 = tmp_path / "b.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out2), prompts=prompts, base_url=stub_server, thinking=True,
                                      thinking_budget=40, tokenizer="teacher.gguf")) == 0
    assert all(r["gen"]["budget_hit"] is False and r["gen"]["reasoning_tokens"] == 39 for r in _rows(out2))


def test_gen_refuses_serve_args_that_change_the_prompt_or_the_budget(tmp_path, stub_server, capsys):
    """The budget is sent with every request, so a server-wide one under
    --serve-arg would silently cap what --thinking-budget records. The
    thinking switch, the template, its variables and a server system
    prompt change what the teacher is prompted with, which the rows would
    not record. Each is refused by any spelling serve's parser accepts."""
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "q"}]}])
    cases = [(["--thinking-budget", "40"], "--thinking-budget is gen's own flag"),
             (["--kv-bits", "4", "--thinking-budget=40"], "--thinking-budget is gen's own flag"),
             (["--thinking-b=512"], "--thinking-budget (given as --thinking-b) is gen's own flag"),
             (["--thinking", "on"], "--thinking is gen's own flag"),
             (["--chat-template-config", "{}"], "--chat-template-config is set from gen's --chat-template-kwargs"),
             (["--chat-template", "t.jinja"], "--chat-template renders the teacher's prompt"),
             (["--reas=high"], "--reasoning-effort (given as --reas) changes the teacher's prompt"),
             (["--system-prompt", "be terse"], "--system-prompt changes the teacher's prompt")]
    for arg, want in cases:
        rc = gen.run_gen(gen.GenOptions(out=str(tmp_path / "a.jsonl"), prompts=prompts, base_url=stub_server,
                                        serve_arg=arg))
        err = capsys.readouterr().err
        assert rc == 2 and f"[gen] refuse: {want}" in err, (arg, err)
    assert not (tmp_path / "a.jsonl").exists()
    for arg in ("--thinking-start-token", "--thinking-s=<t>", "--kv-bits", "r1.gguf", "--adapter"):
        assert gen.prompt_flag(arg) is None, arg


def test_gen_resume_and_filter_join_refuse_another_gen_version(tmp_path, stub_server, capsys):
    """The request a version sends differs (the answer budget changed
    across versions), so rows of two versions are not one run."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    side_path = tmp_path / "corpus.jsonl.gen.json"
    side = json.loads(side_path.read_text())
    assert side["gen_version"] == gen.GEN_VERSION == "4" and side["request_max_tokens"] == 1024
    side["gen_version"] = "3"
    side_path.write_text(json.dumps(side))
    more = _prompts(tmp_path / "p.jsonl", rows + [{"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    capsys.readouterr()
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=more, base_url=stub_server))
    err = capsys.readouterr().err
    assert rc == 2 and "gen_version '3' -> '4'" in err, err
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(_row("a", GOOD)) + "\n")
    b.write_text(json.dumps(_row("b", GOOD)) + "\n")
    common = {"model": "t.gguf", "seed": 1, "thinking": False}
    (tmp_path / "a.jsonl.gen.json").write_text(json.dumps({**common, "gen_version": "3"}))
    (tmp_path / "b.jsonl.gen.json").write_text(json.dumps({**common, "gen_version": "4"}))
    rc = flt.run_filter(flt.FilterOptions(inputs=[str(a), str(b)], out=str(tmp_path / "j.jsonl")))
    assert rc == 2 and "gen_version" in capsys.readouterr().err


def test_filter_join_compares_the_verify_command_and_the_corpus_prompting_and_a_single_input_recounts(
        tmp_path, capsys):
    """Two filtered inputs joined under one sidecar must have run the same
    verify command, two generated inputs the same corpus instruction; and
    a single input's run block counts the rows written, as a join's does."""
    def gen_file(name, n, **extra):
        p = tmp_path / f"{name}.jsonl"
        p.write_text("".join(json.dumps(_row(f"{name}{i}", GOOD if i else "too short", tokens=10)) + "\n"
                             for i in range(n)))
        (tmp_path / f"{name}.jsonl.gen.json").write_text(json.dumps({
            "gen_version": gen.GEN_VERSION, "model": "m", "prompts": n,
            "run": {"completed": n, "failed": 0, "wall_s": 1.0, "tok_s_aggregate": 9.0}, **extra}))
        return p

    a = gen_file("a", 3)
    fa = tmp_path / "fa.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(fa), min_words=4)) == 0
    single = json.loads((tmp_path / "fa.jsonl.gen.json").read_text())
    assert single["run"]["completed"] == 3 and single["run"]["tok_s_aggregate"] == 9.0
    assert single["run"]["kept"] == {**single["run"]["kept"], "completed": 2, "generated_tokens": 20}
    b = gen_file("b", 2)
    fb = tmp_path / "fb.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(b)], out=str(fb), min_words=4)) == 0
    sb = json.loads((tmp_path / "fb.jsonl.gen.json").read_text())
    sb["filter"]["verify"] = "true"
    (tmp_path / "fb.jsonl.gen.json").write_text(json.dumps(sb))
    rc = flt.run_filter(flt.FilterOptions(inputs=[str(fa), str(fb)], out=str(tmp_path / "j.jsonl")))
    err = capsys.readouterr().err
    assert rc == 2 and f"{fb} was filtered with other settings than {fa} (verify)" in err, err
    c = gen_file("c", 2, instruction="Continue.", prefix_chars=100)
    d = gen_file("d", 2, instruction="Go on.", prefix_chars=100)
    rc = flt.run_filter(flt.FilterOptions(inputs=[str(c), str(d)], out=str(tmp_path / "j.jsonl")))
    err = capsys.readouterr().err
    assert rc == 2 and "(instruction)" in err, err
    # an input generated before the corpus keys were recorded joins one that has them
    e = gen_file("e", 2)
    assert flt.run_filter(flt.FilterOptions(inputs=[str(c), str(e)], out=str(tmp_path / "j.jsonl"))) == 0


def test_gen_refuses_a_thinking_budget_with_a_drafter_and_flags_a_budget_the_server_ignored(tmp_path, stub_server,
                                                                                            capsys, monkeypatch):
    """A drafted teacher overshoots a cut trace by a draft block and drops
    the budget when requests batch, so --thinking-budget with a drafter
    flag in --serve-arg is refused. Against any server, a trace that ran
    well past the budget shows the server did not enforce it: the row is
    marked unenforced rather than a hit, and one warning is printed. A
    trace ending in the server's wrap phrase was cut whatever its count,
    and filter names the unenforced rows it keeps."""
    import gmlx.distill.tokens as tokens_mod
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "long", "messages": [{"role": "user", "content": "LONG one"}]},
        {"id": "short", "messages": [{"role": "user", "content": "brief"}]},
        {"id": "wrap", "messages": [{"role": "user", "content": "LONG WRAP"}]},
    ])
    for flag in (["--native-mtp"], ["--draft-gguf", "d.gguf"], ["--speculative=1"], ["--spec"],
                 ["--draft=d.gguf"], ["--native"]):
        rc = gen.run_gen(gen.GenOptions(out=str(tmp_path / "a.jsonl"), prompts=prompts, base_url=stub_server,
                                        thinking=True, thinking_budget=40, tokenizer="teacher.gguf",
                                        serve_arg=flag))
        err = capsys.readouterr().err
        assert rc == 2 and "[gen] refuse: --thinking-budget with a drafter (--native-mtp, --speculative" in err, err
    assert not gen.drafter_flag("--mtp") and not gen.drafter_flag("--no-speculative") and not gen.drafter_flag("--d")
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda path: _WordTokenizer())
    monkeypatch.setattr(gen, "_reasoning_tokens", lambda obj: None)
    monkeypatch.setattr(gen, "trace_tokens", lambda tok, text: len(text.split()) * 3)
    out = tmp_path / "b.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server, thinking=True,
                                      thinking_budget=40, tokenizer="teacher.gguf")) == 0
    err = capsys.readouterr().err
    rows = {r["id"]: r for r in _rows(out)}
    assert rows["long"]["gen"]["reasoning_tokens"] == 120
    assert rows["long"]["gen"]["budget_hit"] is False and rows["long"]["gen"]["budget_unenforced"] is True
    assert rows["short"]["gen"]["reasoning_tokens"] == 60
    assert rows["short"]["gen"]["budget_hit"] is True and rows["short"]["gen"]["budget_unenforced"] is False
    assert rows["wrap"]["gen"]["reasoning_tokens"] > 120
    assert rows["wrap"]["gen"]["budget_hit"] is True and rows["wrap"]["gen"]["budget_unenforced"] is False
    assert err.count("[gen] warn: a reasoning trace ran past --thinking-budget") == 1
    side = json.loads((tmp_path / "b.jsonl.gen.json").read_text())
    assert side["run"]["budget_hits"] == 2 and side["run"]["budget_unenforced"] == 1
    kept = tmp_path / "kept.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(out)], out=str(kept), min_words=4, max_repeat=1.0,
                                            max_trace_repeat=1.0, max_line_repeats=99)) == 0
    err = capsys.readouterr().err
    assert "[filter] warn: 1 kept rows carry budget_unenforced" in err
    assert [r["id"] for r in _rows(kept)] == ["long"]
    fside = json.loads((tmp_path / "kept.jsonl.gen.json").read_text())
    assert fside["filter"]["budget_unenforced"] == 1 and fside["filter"]["dropped"] == {"budget": 2}


def test_the_close_allowance_matches_the_spawned_server_and_a_base_url_takes_the_drafted_close():
    """The server gen spawns closes a cut trace with mlx-vlm's criteria,
    a newline and the end marker: CLOSE_ALLOWANCE is that sequence plus
    the over-budget token and an opening marker. A server behind
    --base-url may be drafted, so its close is the wrap phrase plus the
    marker as the tokenizer counts them."""
    from gmlx.gen.thinking_budget import BUDGET_WRAP_PHRASE
    from gmlx.serve.patches import chat_behavior as cb

    class _IdTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [abs(hash(w)) % 1000 for w in text.split()] or [7]

    crit = cb._armed_thinking_budget_criteria_cls()(
        tokenizer=_IdTokenizer(), thinking_budget=5, thinking_end_token="</think>",
        thinking_start_token="<think>", enable_thinking=True, prompt_open_thinking=False)
    assert gen.CLOSE_ALLOWANCE == len(crit._forced_sequence) + 2
    assert gen.close_tokens(_ThinkTokenizer(), spawned=True) == gen.CLOSE_ALLOWANCE
    assert gen.close_tokens(_ThinkTokenizer(), spawned=False) == len(BUDGET_WRAP_PHRASE.split()) + 1
    assert gen.close_tokens(_WordTokenizer(), spawned=False) == gen.CLOSE_ALLOWANCE


def test_gen_refuses_a_thinking_key_in_the_template_kwargs(tmp_path, stub_server, capsys):
    """serve maps the thinking switch gen sends onto the template's own
    variable and the mapped switch wins over a same-named kwarg, so a
    kwarg naming one is refused before any request."""
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}])
    out = tmp_path / "corpus.jsonl"
    for kw in ('{"enable_thinking": true}', '{"thinking": false}', '{"thinking_mode": "on", "x": 1}'):
        rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server,
                                        chat_template_kwargs=kw))
        err = capsys.readouterr().err
        assert rc == 2 and "[gen] refuse: --chat-template-kwargs sets " in err and "use --thinking" in err
    assert _Handler.calls == [] and not out.exists()
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server,
                                      chat_template_kwargs='{"x": 1}')) == 0
    assert _Handler.calls[-1]["chat_template_kwargs"] == {"x": 1} and _Handler.calls[-1]["thinking"] == "off"


def test_gen_resume_ends_a_last_row_lacking_its_newline_and_refuses_a_row_without_an_id(tmp_path, stub_server,
                                                                                         capsys):
    """A complete last row with no newline after it (an editor's save)
    gets one before the next row is appended, so the rows stay one per
    line; a complete last row with no id is a bad row, not a torn one."""
    rows = [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}]
    prompts = _prompts(tmp_path / "p.jsonl", rows)
    out = tmp_path / "corpus.jsonl"
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    data = out.read_bytes()
    assert data.endswith(b"\n")
    out.write_bytes(data[:-1])
    prompts = _prompts(tmp_path / "p.jsonl", rows + [{"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    capsys.readouterr()
    assert gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)) == 0
    assert "ended the last row of" in capsys.readouterr().err
    assert [r["id"] for r in _rows(out)] == ["a", "b"] and out.read_bytes().count(b"\n") == 2
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"messages": [{"role": "user", "content": "gamma"}]}))
    before = out.read_bytes()
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server))
    err = capsys.readouterr().err
    assert rc == 2 and "line 3 is not a JSON row with an id" in err and "torn" not in err
    assert out.read_bytes() == before


def test_gen_rerun_with_nothing_to_do_writes_the_run_block(tmp_path, stub_server):
    """A rerun that finds every prompt answered contacts no server, but a
    sidecar left without a run block (a run cut short) or missing
    altogether still gains one from the rows, so filter and cache read
    the corpus as generated."""
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}])
    out = tmp_path / "corpus.jsonl"
    side = out.with_suffix(out.suffix + ".gen.json")
    out.write_text(json.dumps({"id": "a", "messages": [{"role": "user", "content": "alpha"},
                                                   {"role": "assistant", "content": "x"}],
                               "gen": {"finish_reason": "stop", "completion_tokens": 7}}) + "\n")
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)
    assert gen.run_gen(opts) == 0 and _Handler.calls == []
    got = json.loads(side.read_text())
    assert got["run"] == {**got["run"], "completed": 1, "generated_tokens": 7, "stops": 1, "failed": 0}
    assert {k: got[k] for k in gen.run_settings(opts)} == gen.run_settings(opts) and got["prompts"] == 1
    got["run"] = None
    got["served_model_id"] = "earlier"
    side.write_text(json.dumps(got))
    assert gen.run_gen(opts) == 0 and _Handler.calls == []
    again = json.loads(side.read_text())
    assert again["served_model_id"] == "earlier" and again["run"]["completed"] == 1
    # a run block already there is left alone
    again["run"]["completed"] = 5
    side.write_text(json.dumps(again))
    assert gen.run_gen(opts) == 0
    assert json.loads(side.read_text())["run"]["completed"] == 5


def test_gen_picks_the_served_model_named_like_the_teacher_among_several(tmp_path, stub_server, capsys):
    """A --base-url server listing several models: the one named like
    --teacher serves the run, and none or several matching is an error,
    since the first id listed is an arbitrary model."""
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}])
    out = tmp_path / "corpus.jsonl"
    _Handler.model_ids = ["other-model", "teacher-Q4_K_M.gguf"]
    rc = gen.run_gen(gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server))
    err = capsys.readouterr().err
    assert rc == 2 and "[gen] error: the server lists 2 models (other-model, teacher-Q4_K_M.gguf) and none match" in err
    assert _Handler.calls == []
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server,
                          teacher=str(tmp_path / "models" / "teacher-Q4_K_M.gguf"))
    assert gen.run_gen(opts) == 0
    assert all(c["model"] == "teacher-Q4_K_M.gguf" for c in _Handler.calls) and len(_Handler.calls) == 2
    assert json.loads(out.with_suffix(out.suffix + ".gen.json").read_text())["served_model_id"] == "teacher-Q4_K_M.gguf"
    _Handler.model_ids = ["teacher-Q4_K_M.gguf", "teacher-Q4_K_M"]
    rc = gen.run_gen(gen.GenOptions(out=str(tmp_path / "c2.jsonl"), prompts=prompts, base_url=stub_server,
                                    teacher="teacher-Q4_K_M.gguf"))
    assert rc == 2 and "several match" in capsys.readouterr().err


def test_filter_counts_an_input_whose_sidecar_has_no_run_block_from_its_rows(tmp_path):
    """A gen cut short leaves its sidecar with run null. filter counts
    that input's rows itself, alone or in a join, instead of writing a
    null run block or counting the input as zero."""
    def gen_file(name, n, run):
        p = tmp_path / f"{name}.jsonl"
        p.write_text("".join(json.dumps(_row(f"{name}{i}", GOOD, tokens=10)) + "\n" for i in range(n)))
        (tmp_path / f"{name}.jsonl.gen.json").write_text(json.dumps({
            "gen_version": "3", "model": "m", "prompts": n, "prompt_source": f"{name}-prompts.jsonl",
            "prompt_set_sha256": name * 4, "run": run}))
        return p

    a = gen_file("a", 2, None)
    b = gen_file("b", 3, {"completed": 3, "failed": 0, "wall_s": 4.0, "stops": 3, "stop_fraction": 1.0, "generated_tokens": 30,
                          "longest_stopped_reply_tokens": 10, "tok_s_aggregate": 9.0, "concurrency": 2})
    fa = tmp_path / "fa.jsonl"
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a)], out=str(fa))) == 0
    run = json.loads((tmp_path / "fa.jsonl.gen.json").read_text())["run"]
    assert run == {**run, "completed": 2, "generated_tokens": 20, "stops": 2, "wall_s": 0.0}
    for inputs in ([str(a), str(b)], [str(b), str(a)]):
        j = tmp_path / "j.jsonl"
        assert flt.run_filter(flt.FilterOptions(inputs=inputs, out=str(j))) == 0
        run = json.loads((tmp_path / "j.jsonl.gen.json").read_text())["run"]
        assert run["completed"] == 5 and run["generated_tokens"] == 50 and run["wall_s"] == 4.0
        assert run["kept"]["completed"] == 5


def test_gen_waits_through_a_readiness_completion_that_fails_while_the_server_loads(stub_server, monkeypatch):
    """A server lists its model while it still loads it and answers the
    one-token readiness completion with an HTTP error (a 503 while the
    load is deferred) until the load is done; gen polls again instead of
    ending the run. A server that never gets ready names its last error
    in the timeout."""
    _Handler.fail_once = {"hi"}
    assert gen.wait_ready(stub_server, None, 30) == "stub-teacher"
    assert [c["messages"][-1]["content"] for c in _Handler.calls] == ["hi", "hi"]

    def always_503(url, body, timeout):
        raise gen.ServerError("HTTP 503 from x: loading")

    monkeypatch.setattr(gen, "_post_json", always_503)
    with pytest.raises(gen.ServerError, match=r"not ready after 0s \(last error: HTTP 503 from x: loading\)"):
        gen.wait_ready(stub_server, None, 0.5)


def test_gen_ends_the_readiness_wait_on_a_request_the_server_refuses(stub_server):
    """A 4xx answer to the readiness completion (no such model, a missing
    model file) does not clear by waiting, so the wait ends on the first
    one. A 429 is polled again like a 5xx."""
    _Handler.refuse_status = 404
    with pytest.raises(gen.ServerError, match=r"HTTP 404 from \S+: model_not_found") as ei:
        gen.wait_ready(stub_server, None, 30)
    assert ei.value.status == 404 and len(_Handler.calls) == 1
    _Handler.refuse_status = 429
    with pytest.raises(gen.ServerError, match=r"not ready after 1s \(last error: HTTP 429 from "):
        gen.wait_ready(stub_server, None, 1)


def test_filter_joins_a_rerun_sidecar_that_never_saw_the_server(tmp_path, stub_server, capsys):
    """A gen rerun that finds every prompt answered and no sidecar writes
    one without the served model id, since it contacted no server. That
    file joins another gen output of the same settings, while two
    sidecars naming different served models still refuse."""
    prompts = _prompts(tmp_path / "p.jsonl", [{"id": "a", "messages": [{"role": "user", "content": "alpha"}]}])
    a = tmp_path / "a.jsonl"
    a.write_text(json.dumps({"id": "a", "messages": [{"role": "user", "content": "alpha"},
                                                     {"role": "assistant", "content": GOOD}],
                             "gen": {"finish_reason": "stop", "completion_tokens": 30}}) + "\n")
    assert gen.run_gen(gen.GenOptions(out=str(a), prompts=prompts, base_url=stub_server)) == 0
    assert _Handler.calls == []
    side_a = json.loads((tmp_path / "a.jsonl.gen.json").read_text())
    assert side_a["served_model_id"] is None
    b = tmp_path / "b.jsonl"
    b.write_text(json.dumps(_row("b", GOOD)) + "\n")
    (tmp_path / "b.jsonl.gen.json").write_text(json.dumps({**side_a, "served_model_id": "stub-teacher"}))
    out = tmp_path / "joined.jsonl"
    capsys.readouterr()
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a), str(b)], out=str(out))) == 0, capsys.readouterr().err
    assert [r["id"] for r in _rows(out)] == ["a", "b"]
    assert json.loads((tmp_path / "joined.jsonl.gen.json").read_text())["served_model_id"] == "stub-teacher"
    # the sidecar without the id sits first, so the two that name one
    # are still checked against each other
    c = tmp_path / "c.jsonl"
    c.write_text(json.dumps(_row("c", GOOD)) + "\n")
    (tmp_path / "c.jsonl.gen.json").write_text(json.dumps({**side_a, "served_model_id": "other-model"}))
    capsys.readouterr()
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a), str(b), str(c)], out=str(out))) == 2
    err = capsys.readouterr().err
    assert "served_model_id" in err and str(b) in err, err
    (tmp_path / "a.jsonl.gen.json").write_text(json.dumps({**side_a, "served_model_id": "other-model"}))
    capsys.readouterr()
    assert flt.run_filter(flt.FilterOptions(inputs=[str(a), str(b)], out=str(out))) == 2
    assert "served_model_id" in capsys.readouterr().err
