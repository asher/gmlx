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
    fail_once: set = set()              # first words whose first request fails with a 500

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": type(self).model_id}]}).encode()
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
    _Handler.fail_once = set()
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
    p4 = _prompts(tmp_path / "p4.jsonl", plain + [{"id": "b", "messages": [{"role": "user", "content": "beta"}]}])
    assert gen.run_gen(gen.GenOptions(out=str(out2), prompts=p3, base_url=stub_server, context=str(empty))) == 0
    assert gen.run_gen(gen.GenOptions(out=str(out2), prompts=p4, base_url=stub_server, context=str(empty))) == 0
    assert sorted(r["id"] for r in _rows(out2)) == ["a", "b"]


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
