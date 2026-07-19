#!/usr/bin/env python3
"""Served assistant aliases: routing, the server-side tool loop, streaming
shape, capacity, and the chat-completions-only guards. CPU-only - the inner
server is a fake stream seam; no live server, no model load."""
from __future__ import annotations

import importlib
import inspect
import json
import threading

import pytest

pytest.importorskip("mlx_vlm")

from fastapi import Request as _Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from gmlx import assistant_serve as aserve  # noqa: E402
from gmlx import server_patches as sp  # noqa: E402
from gmlx.server_patches import _common as sp_common  # noqa: E402
from gmlx import talk_mcp  # noqa: E402
from gmlx.assistant_brain import Tool, ToolRegistry  # noqa: E402
from gmlx.config import build_config  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_SCHEMAS = importlib.import_module("mlx_vlm.server.schemas")


@pytest.fixture(autouse=True)
def _restore_routes():
    saved = list(_APP.app.router.routes)
    yield
    _APP.app.router.routes[:] = saved


def _cfg(assistants, *, mcp=None, host="127.0.0.1", allow_remote=False,
         api_key=None):
    doc = {
        "server": {"model_dirs": ["/models"], "host": host, "port": 8080,
                   "assistants": assistants},
        "models": {"m-a": {"path": "/abs/a.gguf"},
                   "m-b": {"path": "/abs/b.gguf"}},
    }
    if mcp:
        doc["assistant"] = {"mcp": mcp}
    if allow_remote:
        doc["server"]["assistant_allow_remote"] = True
    if api_key:
        doc["server"]["api_key"] = api_key
    return build_config(doc)


def _stub_chat(record):
    """A stand-in original chat handler with real-class annotations (the
    module's future-annotations would stringize inline ones)."""

    async def stub(request, http_request):
        record.append(request)
        return _SCHEMAS.ChatResponse(choices=[_SCHEMAS.ChatChoice(
            finish_reason="stop",
            message=_SCHEMAS.ChatMessage(role="assistant",
                                         content="stub answer"))])

    stub.__signature__ = inspect.Signature([
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=_SCHEMAS.ChatRequest),
        inspect.Parameter("http_request",
                          inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=_Request),
    ])
    return stub


def _install(cfg, monkeypatch, *, stream=None, registry=None):
    """Stub chat route + models override + install_assistant_serve, with the
    loopback stream seam and (optionally) the MCP registries faked."""
    record: list = []
    sp_common._remove_routes(_APP.app, *sp_common._CHAT_PATHS)
    _APP.app.add_api_route("/v1/chat/completions", _stub_chat(record),
                           methods=["POST"])
    sp.install_models_endpoint_override()
    if stream is not None:
        monkeypatch.setattr(aserve, "stream_chat", stream)
    if registry is not None:
        monkeypatch.setattr(
            talk_mcp, "connect_servers",
            lambda servers, **kw: (None, registry, []))
    aserve.install_assistant_serve(cfg)
    return record


class FakeStream:
    """Callable stand-in for the loopback stream_chat: one scripted delta
    list per round (the last repeats), recording every call's kwargs."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls: list = []

    def __call__(self, base_url, *, model, messages, max_tokens,
                 api_key=None, tools=None, timeout=600.0, extra=None):
        self.calls.append(dict(
            base_url=base_url, model=model,
            messages=[dict(m) for m in messages], max_tokens=max_tokens,
            api_key=api_key, tools=tools, extra=extra))
        idx = min(len(self.calls) - 1, len(self.rounds) - 1)
        yield from iter(self.rounds[idx])


_PROSE_ROUND = [{"content": "It is "}, {"content": "noon."},
                {"_usage": {"prompt_tokens": 100, "completion_tokens": 7}},
                {"_finish": "stop"}]
_TOOL_ROUND = [{"tool_calls": [{"index": 0, "id": "c1",
                                "function": {"name": "clock",
                                             "arguments": "{}"}}]},
               {"_usage": {"prompt_tokens": 90, "completion_tokens": 10}},
               {"_finish": "tool_calls"}]


def _clock_registry(log=None):
    return ToolRegistry([Tool(
        name="clock", description="time",
        call=lambda args: (log.append(args) if log is not None else None)
        or "12:00")])


def _post(client, model, *, stream=False, tools=None, messages=None, **kw):
    if messages is None:
        messages = [{"role": "user", "content": "what time is it?"}]
    body = {"model": model, "stream": stream, "messages": messages}
    if tools is not None:
        body["tools"] = tools
    body.update(kw)
    return client.post("/v1/chat/completions", json=body)


# -- routing -----------------------------------------------------------------

def test_alias_without_tools_runs_loop_stub_untouched(monkeypatch):
    fake = FakeStream([_PROSE_ROUND])
    record = _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
                      stream=fake)
    r = _post(TestClient(_APP.app), "helper")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "helper"
    assert body["choices"][0]["message"]["content"] == "It is noon."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert record == []                     # original handler never ran
    assert fake.calls[0]["model"] == "m-a"  # loop used the underlying id


def test_alias_with_empty_tools_list_also_runs_loop(monkeypatch):
    fake = FakeStream([_PROSE_ROUND])
    record = _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
                      stream=fake)
    r = _post(TestClient(_APP.app), "helper", tools=[])
    assert r.status_code == 200
    assert record == [] and len(fake.calls) == 1


def test_alias_with_tools_rewrites_model_and_passes_through(monkeypatch):
    fake = FakeStream([_PROSE_ROUND])
    record = _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
                      stream=fake)
    tools = [{"type": "function", "function": {"name": "t"}}]
    r = _post(TestClient(_APP.app), "helper", tools=tools)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "stub answer"
    assert len(record) == 1 and record[0].model == "m-a"
    assert fake.calls == []                 # no server-side loop


def test_non_alias_untouched(monkeypatch):
    fake = FakeStream([_PROSE_ROUND])
    record = _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
                      stream=fake)
    r = _post(TestClient(_APP.app), "m-b")
    assert r.status_code == 200
    assert len(record) == 1 and record[0].model == "m-b"
    assert fake.calls == []


def test_models_lists_alias(monkeypatch):
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
             stream=FakeStream([_PROSE_ROUND]))
    data = TestClient(_APP.app).get("/v1/models").json()["data"]
    entry = next(e for e in data if e["id"] == "helper")
    assert entry["assistant"] is True and entry["alias_of"] == "m-a"


def test_responses_and_messages_reject_alias(monkeypatch):
    async def echo(request):
        return await request.json()

    # Real-class annotation: future-annotations would stringize an inline
    # one and the guard's signature copy couldn't resolve it.
    echo.__signature__ = inspect.Signature([
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=_Request)])

    # Stub /v1/responses BEFORE install: the real guard wraps it, and the
    # non-alias passthrough proves the guard's body read left it re-parseable.
    sp_common._remove_routes(_APP.app, "/v1/responses")
    _APP.app.add_api_route("/v1/responses", echo, methods=["POST"])
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
             stream=FakeStream([_PROSE_ROUND]))
    client = TestClient(_APP.app)

    r = client.post("/v1/responses", json={"model": "helper", "input": "x"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "chat-completions only" in r.json()["error"]["message"]

    r = client.post("/v1/messages", json={
        "model": "helper", "max_tokens": 8,
        "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400
    assert r.json()["type"] == "error"     # anthropic error shape
    assert "chat-completions only" in r.json()["error"]["message"]

    r = client.post("/v1/responses", json={"model": "m-b", "input": "x"})
    assert r.status_code == 200 and r.json()["model"] == "m-b"


# -- the loop (multi-round, usage, sampling passthrough) ----------------------

def test_multi_round_tool_loop_and_summed_usage(monkeypatch):
    fake = FakeStream([_TOOL_ROUND, _PROSE_ROUND])
    args_log: list = []
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch, stream=fake,
             registry=_clock_registry(args_log))
    r = _post(TestClient(_APP.app), "helper")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "It is noon."
    assert args_log == [{}]                          # tool actually ran
    assert len(fake.calls) == 2
    round2 = fake.calls[1]["messages"]
    assert round2[-1]["role"] == "tool"
    assert round2[-1]["content"] == "12:00"          # result fed back
    assert fake.calls[0]["tools"]                    # offered on round 1
    # usage: completion summed; prompt = final round's, never summed
    assert body["usage"] == {"prompt_tokens": 100,
                             "completion_tokens": 17,
                             "total_tokens": 117}


def test_history_seeded_and_sampling_forwarded(monkeypatch):
    fake = FakeStream([_PROSE_ROUND])
    cfg = _cfg({"helper": {"model": "m-a"}}, api_key="sek")
    _install(cfg, monkeypatch, stream=fake)
    r = _post(TestClient(_APP.app), "helper", messages=[
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what time is it?"},
    ], temperature=0.2, top_k=40, stop=["END"], max_tokens=64,
        response_format={"type": "json_object"})
    assert r.status_code == 200
    call = fake.calls[0]
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert call["messages"][0]["content"] == "be brief"
    assert call["max_tokens"] == 64
    assert call["api_key"] == "sek"                  # self-call carries key
    extra = call["extra"]
    assert extra["temperature"] == 0.2 and extra["top_k"] == 40
    assert extra["stop"] == ["END"]
    assert extra["stream_options"] == {"include_usage": True}
    assert "response_format" not in extra            # never forwarded
    assert "top_p" not in extra                      # unset stays unset


def test_default_max_tokens_when_unset(monkeypatch):
    fake = FakeStream([_PROSE_ROUND])
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch, stream=fake)
    _post(TestClient(_APP.app), "helper")
    assert fake.calls[0]["max_tokens"] == aserve._DEFAULT_MAX_TOKENS


def test_upstream_error_is_502(monkeypatch):
    def broken(base_url, **kw):
        from gmlx.talk_client import TalkClientError
        raise TalkClientError("chat failed: HTTP 500")
        yield  # pragma: no cover

    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch, stream=broken)
    r = _post(TestClient(_APP.app), "helper")
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["code"] == "assistant_upstream_error"
    assert "HTTP 500" in err["message"]


# -- request validation -------------------------------------------------------

@pytest.mark.parametrize("messages,match", [
    ([], "must not be empty"),
    ([{"role": "user", "content": "x"},
      {"role": "assistant", "content": "y"}], "role 'user'"),
    ([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://x/i.png"}}]}],
     "text-only"),
])
def test_bad_requests_400(monkeypatch, messages, match):
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
             stream=FakeStream([_PROSE_ROUND]))
    r = _post(TestClient(_APP.app), "helper", messages=messages)
    assert r.status_code == 400
    assert match in r.json()["error"]["message"]


def test_text_parts_last_message_ok(monkeypatch):
    fake = FakeStream([_PROSE_ROUND])
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch, stream=fake)
    r = _post(TestClient(_APP.app), "helper", messages=[
        {"role": "user", "content": [{"type": "text", "text": "what "},
                                     {"type": "text", "text": "time?"}]}])
    assert r.status_code == 200
    assert fake.calls[0]["messages"][-1]["content"] == "what time?"


# -- streaming ----------------------------------------------------------------

def _sse_events(text):
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            out.append(line[6:])
    return out


def test_stream_chunk_sequence_no_null_keys(monkeypatch):
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
             stream=FakeStream([_PROSE_ROUND]))
    r = _post(TestClient(_APP.app), "helper", stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    said = "".join(c["choices"][0]["delta"].get("content", "")
                   for c in chunks[1:-1])
    assert said == "It is noon."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    for c in chunks:
        assert c["model"] == "helper"
        assert "timings" not in json.dumps(c)
        assert "null" not in json.dumps(c["choices"])   # no null-valued keys
    assert not any("usage" in c for c in chunks)         # not requested


def test_stream_usage_chunk_iff_requested(monkeypatch):
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
             stream=FakeStream([_PROSE_ROUND]))
    r = _post(TestClient(_APP.app), "helper", stream=True,
              stream_options={"include_usage": True})
    chunks = [json.loads(e) for e in _sse_events(r.text)[:-1]]
    usage = [c for c in chunks if "usage" in c]
    assert len(usage) == 1
    assert usage[0]["usage"] == {"prompt_tokens": 100,
                                 "completion_tokens": 7,
                                 "total_tokens": 107}
    assert usage[0]["choices"] == []


def test_stream_tool_status_is_comment_line(monkeypatch):
    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch,
             stream=FakeStream([_TOOL_ROUND, _PROSE_ROUND]),
             registry=_clock_registry())
    r = _post(TestClient(_APP.app), "helper", stream=True)
    assert ": assistant using clock" in r.text
    events = _sse_events(r.text)
    for e in events[:-1]:
        assert "tool_calls" not in e         # tool rounds never surface


def test_stream_upstream_error_in_band(monkeypatch):
    def broken(base_url, **kw):
        from gmlx.talk_client import TalkClientError
        raise TalkClientError("boom")
        yield  # pragma: no cover

    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch, stream=broken)
    r = _post(TestClient(_APP.app), "helper", stream=True)
    assert r.status_code == 200              # committed before the body ran
    events = _sse_events(r.text)
    err = json.loads(events[-2])
    assert err["error"]["code"] == "assistant_upstream_error"
    assert events[-1] == "[DONE]"


# -- concurrency and capacity --------------------------------------------------

def test_two_turns_run_concurrently(monkeypatch):
    barrier = threading.Barrier(2, timeout=10)

    def stream(base_url, **kw):
        barrier.wait()                       # both brains in-flight at once
        yield from iter(_PROSE_ROUND)

    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch, stream=stream)
    client = TestClient(_APP.app)
    results: list = []

    def post():
        results.append(_post(client, "helper").status_code)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert results == [200, 200]


def test_capacity_cap_is_real_429_on_stream_path(monkeypatch):
    monkeypatch.setattr(aserve, "_MAX_CONCURRENT_TURNS", 1)
    entered = threading.Event()              # first brain holds the slot
    gate = threading.Event()

    def stream(base_url, **kw):
        entered.set()
        yield {"content": "started"}
        gate.wait(10)
        yield {"_finish": "stop"}

    _install(_cfg({"helper": {"model": "m-a"}}), monkeypatch, stream=stream)
    client = TestClient(_APP.app)
    first: dict = {}

    def run_first():
        first["r"] = _post(client, "helper", stream=True)

    t = threading.Thread(target=run_first)
    t.start()
    try:
        assert entered.wait(10)              # first stream is mid-body
        r2 = _post(client, "helper", stream=True)
        assert r2.status_code == 429         # a real status, not in-band
        assert r2.json()["error"]["type"] == "rate_limit_error"
    finally:
        gate.set()
        t.join(timeout=15)
    assert first["r"].status_code == 200
    assert "started" in first["r"].text
    r3 = _post(client, "helper")             # slot released after stream end
    assert r3.status_code == 200


# -- cancellation (seam unit) ---------------------------------------------------

def test_seam_cancel_stops_the_loop(monkeypatch):
    yielded = {"n": 0}

    def endless(base_url, **kw):
        while True:
            yielded["n"] += 1
            yield {"content": "x"}

    monkeypatch.setattr(aserve, "stream_chat", endless)
    cfg = _cfg({"helper": {"model": "m-a"}})
    state = aserve._AssistantState(cfg, "http://127.0.0.1:8080/v1")
    request = _SCHEMAS.ChatRequest(
        model="helper", messages=[{"role": "user", "content": "hi"}])
    cancel = threading.Event()
    usage = {"rounds": 0, "prompt_tokens": 0, "completion_tokens": 0}
    alias = cfg.assistants["helper"]
    brain = aserve._build_brain(state, "helper", alias, ToolRegistry(),
                                request, cancel, usage)
    turn = brain.turn("hi")
    next(turn)                               # first spoken span
    cancel.set()
    with pytest.raises(aserve._AssistantCancelled):
        for _ in turn:
            pass
    assert yielded["n"] < 5                  # stopped promptly, not endless
    assert brain.messages[-1]["role"] == "assistant"   # partial committed


# -- startup wiring --------------------------------------------------------------

def test_memory_default_off_and_per_alias_store(monkeypatch, tmp_path):
    created: list = []

    class FakeStore:
        def __init__(self, **kw):
            created.append(kw)

    import gmlx.talk_memory as tm
    monkeypatch.setattr(tm, "MemoryStore", FakeStore)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(
        talk_mcp, "connect_servers",
        lambda servers, **kw: (None, ToolRegistry(), []))

    state = aserve._build_state(_cfg({"helper": {"model": "m-a"}}))
    assert state.memories == {} and created == []      # default OFF

    state = aserve._build_state(
        _cfg({"helper": {"model": "m-a", "memory": True},
              "other": {"model": "m-b", "memory": True}}))
    assert set(state.memories) == {"helper", "other"}
    paths = sorted(c["path"] for c in created)
    assert paths[0].endswith("assistant-helper.db")
    assert paths[1].endswith("assistant-other.db")     # never the talk store


def test_tool_scoping_distinct_registries(monkeypatch):
    calls: list = []

    def fake_connect(servers, **kw):
        calls.append([s.name for s in servers])
        return None, ToolRegistry(
            [Tool(name=f"{s.name}_tool", description="")
             for s in servers]), []

    monkeypatch.setattr(talk_mcp, "connect_servers", fake_connect)
    cfg = _cfg(
        {"helper": {"model": "m-a"},                      # inherits shared
         "scoped": {"model": "m-b",
                    "mcp": [{"name": "w", "url": "http://127.0.0.1:1/mcp"}]},
         "locked": {"model": "m-a", "mcp": []}},          # zero tools
        mcp=[{"name": "s", "command": ["srv"]}])
    state = aserve._build_state(cfg)
    _, helper_reg = state.aliases["helper"]
    _, scoped_reg = state.aliases["scoped"]
    _, locked_reg = state.aliases["locked"]
    assert helper_reg.names() == ["s_tool"]
    assert scoped_reg.names() == ["w_tool"]
    assert len(locked_reg) == 0
    assert helper_reg is not scoped_reg
    assert sorted(map(tuple, calls)) == [("s",), ("w",)]  # one connect each


def test_self_base_url_shapes():
    class C:
        port = 9090

    for host, expect in (("0.0.0.0", "http://127.0.0.1:9090/v1"),
                         ("::", "http://127.0.0.1:9090/v1"),
                         ("127.0.0.1", "http://127.0.0.1:9090/v1"),
                         ("localhost", "http://127.0.0.1:9090/v1"),
                         ("::1", "http://[::1]:9090/v1"),
                         ("10.0.0.5", "http://10.0.0.5:9090/v1")):
        C.host = host
        assert aserve._self_base_url(C) == expect


def test_install_is_noop_without_aliases():
    n = len(_APP.app.router.routes)

    class C:
        assistants = {}

    aserve.install_assistant_serve(C)
    assert len(_APP.app.router.routes) == n
