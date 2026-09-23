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

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "stub-teacher"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n))
        type(self).calls.append(req)
        last = req["messages"][-1]["content"]
        word = last.split()[0] if last.split() else "empty"
        content = f"reply about {word} " + "and more words " * 8
        msg: dict = {"role": "assistant", "content": content}
        budget = req.get("thinking_budget")
        reasoning_tokens = 0
        if budget:
            reasoning_tokens = budget if "LONG" in last else max(1, budget // 2)
            msg["reasoning_content"] = "thinking " * reasoning_tokens
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
    # the switch is sent off explicitly, so a template that thinks by
    # default does not think behind a sidecar that says off
    sampled = [c for c in _Handler.calls if "seed" in c]
    assert sampled and all(c["enable_thinking"] is False and c["chat_template_kwargs"] == {"enable_thinking": False}
                           for c in sampled)
    assert side["chat_template_kwargs"] == {"enable_thinking": False}


def test_gen_resumes_by_prompt_id_and_accumulates_the_run(tmp_path, stub_server):
    prompts = _prompts(tmp_path / "p.jsonl", [
        {"id": "a", "messages": [{"role": "user", "content": "alpha"}]},
        {"id": "b", "messages": [{"role": "user", "content": "beta"}]},
    ])
    out = tmp_path / "corpus.jsonl"
    out.write_text(json.dumps({"id": "a", "messages": [], "gen": {}}) + "\n")
    opts = gen.GenOptions(out=str(out), prompts=prompts, base_url=stub_server)
    assert gen.run_gen(opts) == 0
    assert [r["id"] for r in _rows(out)] == ["a", "b"]
    assert len(_Handler.calls) == 2                  # the readiness probe and one reply
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["run"]["completed"] == 1
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
    assert all(c["enable_thinking"] is True for c in sent)
    assert all(c["chat_template_kwargs"] == {"preserve_thinking": True, "enable_thinking": True} for c in sent)
    side = json.loads((tmp_path / "corpus.jsonl.gen.json").read_text())
    assert side["chat_template_kwargs"] == {"preserve_thinking": True, "enable_thinking": True}
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
    assert json.loads((tmp_path / "ok2.jsonl.gen.json").read_text())["filter_version"] == "2+2"


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
    """Without --thinking the request and the served template carry
    enable_thinking false, so a template whose default is thinking does
    not think behind a sidecar that says off; the switch wins over a
    same-named kwarg."""
    off = gen.GenOptions(out="x.jsonl", prompts="p.jsonl")
    body = gen._sampling(off, 1)
    assert body["enable_thinking"] is False and body["chat_template_kwargs"] == {"enable_thinking": False}
    on = gen.GenOptions(out="x.jsonl", prompts="p.jsonl", thinking=True,
                        chat_template_kwargs='{"enable_thinking": false, "preserve_thinking": true}')
    assert gen._sampling(on, 1)["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    from gmlx.serve import lifecycle
    seen: dict = {}

    def start(args, **kw):
        seen["args"] = list(args)
        return object(), None

    monkeypatch.setattr(gen, "port_listening", lambda host, port: False)
    monkeypatch.setattr(lifecycle, "start_background_nowait", start)
    gen.spawn_server(gen.GenOptions(out="x.jsonl", prompts="p.jsonl", teacher="t.gguf", serve_arg=["--kv-bits", "8"]),
                     tmp_path / "log")
    assert seen["args"] == ["t.gguf", "--chat-template-config", '{"enable_thinking": false}', "--kv-bits", "8"]
