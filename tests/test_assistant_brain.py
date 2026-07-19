#!/usr/bin/env python3
"""AssistantBrain tool-loop tests: scripted SSE rounds through an injected stream
seam - no server, no sockets, no MCP. Covers delta accumulation across chunks,
tool execution (unknown / bad-args / timeout), the round cap's tool-less final
call, atomic history commits under cancellation, and the memory seam."""
from __future__ import annotations

import time

import pytest

from gmlx.assistant_brain import (AssistantBrain, Tool, ToolRegistry,
                                  _call_with_timeout)


def _stream_script(*rounds):
    """A stream_chat stand-in that replays one scripted delta list per call
    and records what each call was asked (messages snapshot + tools)."""
    remaining = list(rounds)
    calls: list = []

    def fake(base_url, *, model, messages, max_tokens, api_key=None,
             tools=None, timeout=600.0):
        calls.append({"messages": [dict(m) for m in messages],
                      "tools": tools})
        deltas = remaining.pop(0)
        return iter(list(deltas))

    fake.calls = calls
    return fake


def _clock_tool(log=None):
    log = log if log is not None else []

    def call(args):
        log.append(args)
        return "5pm"

    t = Tool(name="get_time", description="current time",
             parameters={"type": "object",
                         "properties": {"tz": {"type": "string"}}},
             call=call)
    t.log = log
    return t


def _tool_round(name="get_time", args='{"tz": "UTC"}', text=None,
                call_id="call_1", index=0):
    """Deltas for one round that ends in a tool call, with the name in one
    chunk and the arguments split across two (the accumulation contract)."""
    out = []
    if text:
        out.append({"content": text})
    out.append({"tool_calls": [{"index": index, "id": call_id,
                                "function": {"name": name}}]})
    half = len(args) // 2
    out.append({"tool_calls": [{"index": index,
                                "function": {"arguments": args[:half]}}]})
    out.append({"tool_calls": [{"index": index,
                                "function": {"arguments": args[half:]}}]})
    out.append({"_finish": "tool_calls"})
    return out


_PROSE = [{"content": "It is "}, {"content": "5pm."},
          {"_finish": "stop"}, {"_usage": {"total_tokens": 7}}]


def _brain(stream, **kw):
    kw.setdefault("tools", ToolRegistry([_clock_tool()]))
    return AssistantBrain(base_url="http://h:1/v1", model="m", stream=stream,
                      **kw)


# -- plain answers ------------------------------------------------------------
def test_prose_answer_no_tools_touched():
    stream = _stream_script(_PROSE)
    b = _brain(stream)
    events = list(b.turn("what time is it?"))
    assert ("say", "It is ") in events and ("say", "5pm.") in events
    assert events[-1] == ("done", {"total_tokens": 7})
    assert b.messages == [{"role": "user", "content": "what time is it?"},
                          {"role": "assistant", "content": "It is 5pm."}]
    assert b.tools.get("get_time").log == []      # never invoked


def test_system_prompt_sent_but_not_in_history():
    stream = _stream_script(_PROSE)
    b = _brain(stream, system="You speak tersely.")
    list(b.turn("hi"))
    sent = stream.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": "You speak tersely."}
    assert all(m.get("role") != "system" for m in b.messages)


def test_reasoning_deltas_become_status():
    stream = _stream_script([{"reasoning": "hmm"}, {"reasoning": "hm2"}]
                            + _PROSE)
    events = list(_brain(stream).turn("hi"))
    assert events.count(("status", "thinking")) == 2
    assert ("say", "It is ") in events


# -- the tool loop -------------------------------------------------------------
def test_tool_round_executes_and_recalls():
    stream = _stream_script(_tool_round(text="Checking. "), _PROSE)
    b = _brain(stream)
    events = list(b.turn("time?"))

    assert b.tools.get("get_time").log == [{"tz": "UTC"}]
    assert ("status", "using get_time") in events
    assert ("say", "Checking. ") in events and ("say", "5pm.") in events
    assert events[-1][0] == "done"

    # first call offered the tool specs; both requests carried the history
    assert stream.calls[0]["tools"][0]["function"]["name"] == "get_time"
    second = stream.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "get_time"
    assert second[-2]["content"] == "Checking. "  # pre-tool speech committed
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1",
                          "content": "5pm"}
    # final prose is its own assistant message - no duplicated text
    assert b.messages[-1] == {"role": "assistant", "content": "It is 5pm."}


def test_parallel_tool_calls_run_in_index_order():
    seen = []
    reg = ToolRegistry([
        Tool(name="a", description="", call=lambda ar: seen.append("a") or "ra"),
        Tool(name="b", description="", call=lambda ar: seen.append("b") or "rb"),
    ])
    round1 = [{"tool_calls": [
                  {"index": 1, "id": "c2", "function": {"name": "b"}},
                  {"index": 0, "id": "c1", "function": {"name": "a"}}]},
              {"_finish": "tool_calls"}]
    stream = _stream_script(round1, _PROSE)
    b = _brain(stream, tools=reg)
    list(b.turn("go"))
    assert seen == ["a", "b"]
    tool_msgs = [m for m in stream.calls[1]["messages"]
                 if m.get("role") == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == \
        [("c1", "ra"), ("c2", "rb")]


def test_unknown_tool_and_bad_args_become_error_results():
    round1 = _tool_round(name="launch_rocket")
    round2 = _tool_round(args='{oops')
    stream = _stream_script(round1, round2, _PROSE)
    b = _brain(stream)
    list(b.turn("go"))
    results = [m["content"] for msgs in (c["messages"] for c in stream.calls)
               for m in msgs if m.get("role") == "tool"]
    assert any(r.startswith("error: unknown tool 'launch_rocket'")
               for r in results)
    assert any(r.startswith("error: invalid arguments JSON") for r in results)
    assert b.messages[-1]["content"] == "It is 5pm."   # still answered


def test_round_cap_forces_toolless_final_call():
    rounds = [_tool_round(call_id=f"c{i}") for i in range(2)] + [_PROSE]
    stream = _stream_script(*rounds)
    b = _brain(stream, max_tool_rounds=2)
    events = list(b.turn("go"))
    assert len(stream.calls) == 3
    assert stream.calls[0]["tools"] and stream.calls[1]["tools"]
    assert stream.calls[2]["tools"] is None        # the forced final call
    assert events[-1][0] == "done"


def test_cancellation_commits_spoken_text_never_dangling_tool_calls():
    stream = _stream_script(_tool_round(text="Checking. "), _PROSE)
    b = _brain(stream)
    gen = b.turn("time?")
    assert next(gen) == ("say", "Checking. ")
    gen.close()                                    # barge-in
    assert b.messages == [{"role": "user", "content": "time?"},
                          {"role": "assistant", "content": "Checking. "}]
    assert b.tools.get("get_time").log == []       # tool never ran
    # the next turn proceeds on the clean history
    list(b.turn("again?"))
    assert b.messages[-1]["content"] == "It is 5pm."


def test_cancellation_after_committed_round_keeps_tool_results():
    stream = _stream_script(_tool_round(), [{"content": "It is "}])
    b = _brain(stream)
    gen = b.turn("time?")
    evs = [next(gen) for _ in range(2)]            # status using, say "It is "
    assert evs[0] == ("status", "using get_time")
    gen.close()                                    # cancel in round 2
    roles = [m["role"] for m in b.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert b.messages[1]["tool_calls"]             # round 1 stayed atomic
    assert b.messages[-1]["content"] == "It is "   # heard text committed


# -- memory seam ---------------------------------------------------------------
class FakeMemory:
    def __init__(self):
        self.recalled = []
        self.remembered = []

    def recall(self, text):
        self.recalled.append(text)
        return ["user likes green tea"]

    def remember(self, user_text, answer):
        self.remembered.append((user_text, answer))


def test_memory_injected_per_request_not_into_history():
    stream = _stream_script(_PROSE)
    mem = FakeMemory()
    b = _brain(stream, memory=mem, system="Be brief.")
    list(b.turn("tea?"))
    sent = stream.calls[0]["messages"]
    assert sent[1]["role"] == "system"
    assert "green tea" in sent[1]["content"]
    assert all("green tea" not in str(m.get("content"))
               for m in b.messages)
    assert mem.remembered == [("tea?", "It is 5pm.")]


def test_memory_not_written_on_cancellation():
    stream = _stream_script(_PROSE)
    mem = FakeMemory()
    b = _brain(stream, memory=mem)
    gen = b.turn("tea?")
    next(gen)
    gen.close()
    assert mem.remembered == []


# -- registry + timeout helpers -------------------------------------------------
def test_registry_rejects_duplicates_and_builds_specs():
    reg = ToolRegistry([_clock_tool()])
    with pytest.raises(ValueError, match="duplicate"):
        reg.add(_clock_tool())
    spec = reg.specs()[0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "get_time"
    assert spec["function"]["parameters"]["type"] == "object"


def test_call_with_timeout_bounds_and_reports():
    slow = Tool(name="slow", description="",
                call=lambda a: time.sleep(0.5) or "late")
    out = _call_with_timeout(slow, {}, 0.05)
    assert "timed out" in out
    boom = Tool(name="boom", description="",
                call=lambda a: (_ for _ in ()).throw(ValueError("nope")))
    assert _call_with_timeout(boom, {}, 1.0) == "error: ValueError: nope"


def test_execute_survives_pathological_json_nesting():
    # A RecursionError out of json.loads (prompt-injected 10k-deep arguments)
    # must become a tool-error string, never unwind the REPL.
    b = _brain(_stream_script(_PROSE))
    deep = "[" * 20000 + "]" * 20000
    out = b._execute({"name": "get_time", "args": [deep]})
    assert out.startswith("error: invalid arguments JSON")
