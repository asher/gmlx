"""The built-in tool-loop assistant (``talk.brain: assistant``, ``chat
--assistant``, ``server.assistants``).

The standard OpenAI tool loop run against the server's own
``/v1/chat/completions``: stream a turn, accumulate ``tool_calls`` deltas,
execute the named tools, append the results as ``role: tool`` messages, and
re-call - until the model answers in prose or the round cap forces it to.
Implements the same :class:`~gmlx.talk_client.Brain` protocol as
``ServerChatBrain``, so the audio loop and TUI are unchanged; tool activity
surfaces as ``("status", ...)`` events.

Cancellation contract: the loop cancels a turn by closing the generator, so
GeneratorExit lands at the most recent yield. This brain yields a status
event immediately before every tool invocation and bounds each invocation
with ``tool_timeout_s``, keeping close points frequent. A turn canceled
mid-round commits only the text already spoken to history - never a dangling
``tool_calls`` message without its results (which would break chat templates
on the next turn).

Tools come from a :class:`ToolRegistry` - filled from MCP servers (see
``talk_mcp``) or directly in tests. Long-term memory is a constructor seam
(``recall(text) -> [str]`` / ``remember(user_text, answer)``): recalled facts
are injected as a transient system message per request, never stored in the
rolling history.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from collections.abc import Callable, Iterator

from .talk_client import BrainEvent, stream_chat


@dataclass
class Tool:
    """One callable tool: OpenAI function spec + the function itself.
    ``call`` takes the parsed arguments dict and returns the result string
    the model sees."""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)   # JSON schema
    call: Callable[[dict], str] = lambda args: ""

    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters or
            {"type": "object", "properties": {}}}}


class ToolRegistry:
    """Name -> :class:`Tool`, with the OpenAI ``tools`` payload derived."""

    def __init__(self, tools=()):
        self._tools: dict = {}
        for t in tools:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)


def _call_with_timeout(tool: Tool, args: dict, timeout_s: float) -> str:
    """Run one tool bounded by ``timeout_s``. On timeout the worker thread is
    abandoned (daemon) and the model gets an error string - the loop must
    never wedge on a stuck tool."""
    box: dict = {}

    def work():
        try:
            box["out"] = str(tool.call(args))
        except Exception as e:                    # noqa: BLE001 - to the model
            box["out"] = f"error: {type(e).__name__}: {e}"

    t = threading.Thread(target=work, daemon=True, name=f"tool-{tool.name}")
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return f"error: tool {tool.name!r} timed out after {timeout_s:g}s"
    return box.get("out", "")


class AssistantBrain:
    """Tool-loop brain behind the ``Brain`` protocol (see module docstring)."""

    def __init__(self, *, base_url: str, model: str,
                 api_key: str | None = None,
                 system: str | None = None, max_tokens: int | None = 512,
                 tools: ToolRegistry | None = None,
                 max_tool_rounds: int = 8, tool_timeout_s: float = 60.0,
                 memory=None, stream=None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.system = system
        self.max_tokens = max_tokens
        self.tools = tools or ToolRegistry()
        self.max_tool_rounds = max_tool_rounds
        self.tool_timeout_s = tool_timeout_s
        self.memory = memory
        self._stream = stream or stream_chat
        self.messages: list = []                  # clean history, no system
        self.reset()

    def reset(self) -> None:
        self.messages = []

    def close(self) -> None:
        if self.memory is not None and hasattr(self.memory, "close"):
            self.memory.close()

    # -- request assembly --------------------------------------------------
    def _request_messages(self, memory_facts: list) -> list:
        """System + transient memory block + rolling history. Memory is
        re-derived per turn and never enters ``self.messages``."""
        msgs: list = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        if memory_facts:
            block = "\n".join(f"- {f}" for f in memory_facts)
            msgs.append({"role": "system",
                         "content": "Relevant memories of this user from "
                                    f"past conversations:\n{block}"})
        return msgs + self.messages

    # -- the tool loop -------------------------------------------------------
    def turn(self, user_text: str) -> Iterator[BrainEvent]:
        from .reasoning import ReasoningFilter

        memory_facts: list = []
        if self.memory is not None:
            try:
                memory_facts = list(self.memory.recall(user_text) or [])
            except Exception:                     # noqa: BLE001 - best-effort
                memory_facts = []

        self.messages.append({"role": "user", "content": user_text})
        spoken: list = []                # all answer text this turn (memory)
        text_parts: list = []            # current round's uncommitted text
        stats: dict = {}
        completed = False
        committed_tool_round = False     # any assistant+tool round appended?
        try:
            # Round r < max_tool_rounds may call tools; the last request is
            # made tool-less so a tool-happy model still produces an answer.
            for r in range(self.max_tool_rounds + 1):
                offer_tools = bool(self.tools) and r < self.max_tool_rounds
                rf = ReasoningFilter()
                text_parts = []
                calls: dict = {}                  # index -> {id, name, args}
                finish = None
                deltas = self._stream(
                    self.base_url, model=self.model,
                    messages=self._request_messages(memory_facts),
                    max_tokens=self.max_tokens, api_key=self.api_key,
                    tools=self.tools.specs() if offer_tools else None)
                try:
                    for delta in deltas:
                        if "_usage" in delta:
                            stats = delta["_usage"] or {}
                            continue
                        if "_finish" in delta:
                            finish = delta["_finish"]
                            continue
                        if delta.get("reasoning"):
                            yield ("status", "thinking")
                            continue
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(
                                tc.get("index", 0),
                                {"id": None, "name": "", "args": []})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["args"].append(fn["arguments"])
                        text = delta.get("content")
                        if not text:
                            continue
                        for span, mode in rf.feed(text):
                            if mode == "answer" and span:
                                text_parts.append(span)
                                yield ("say", span)
                            elif span:
                                yield ("status", "thinking")
                    for span, mode in rf.flush():
                        if mode == "answer" and span:
                            text_parts.append(span)
                            yield ("say", span)
                finally:
                    close = getattr(deltas, "close", None)
                    if close is not None:         # generators; plain iters ok
                        close()
                spoken.extend(text_parts)

                if (not offer_tools or not calls
                        or finish not in (None, "tool_calls")):
                    # A prose answer, a hard stop, or the tool-less final
                    # round: the turn is complete.
                    completed = True
                    break

                # Execute this round's tool calls, then commit the round to
                # history atomically (assistant tool_calls + every result) so
                # a cancellation can never leave dangling tool_calls behind.
                ordered = [calls[i] for i in sorted(calls)]
                results: list = []
                for c in ordered:
                    name = c["name"] or "?"
                    yield ("status", f"using {name}")   # close point pre-call
                    results.append(self._execute(c))
                assistant: dict = {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                    "tool_calls": [
                        {"id": c["id"] or f"call_{i}", "type": "function",
                         "function": {"name": c["name"],
                                      "arguments": "".join(c["args"]) or "{}"}}
                        for i, c in enumerate(ordered)],
                }
                self.messages.append(assistant)
                for i, (c, result) in enumerate(zip(ordered, results)):
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": c["id"] or f"call_{i}",
                        "content": result})
                committed_tool_round = True
                text_parts = []      # committed inside the tool_calls message
        finally:
            # Completion and cancellation land here (no yields - a closing
            # generator may not yield). Commit the last round's uncommitted text
            # so the next turn's context matches what the user heard. Always
            # append (even if empty) when the turn produced text, committed a
            # tool round, or completed - an empty trailing assistant closes a
            # tool round and keeps role alternation intact. But on a fully-failed
            # turn (stream error / cancellation before any answer or tool round),
            # drop the paired user message instead, so the next turn doesn't send
            # two consecutive user messages that strict chat templates reject.
            if text_parts or committed_tool_round or completed:
                self.messages.append({"role": "assistant",
                                      "content": "".join(text_parts)})
            elif self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()
            if self.memory is not None and completed:
                try:
                    self.memory.remember(user_text, "".join(spoken))
                except Exception:                 # noqa: BLE001 - best-effort
                    pass
        if completed:
            yield ("done", stats)

    def _execute(self, call: dict) -> str:
        """One accumulated tool call -> the result string the model sees.
        Unknown tools and malformed argument JSON come back as error strings
        (the model can retry), never exceptions."""
        tool = self.tools.get(call["name"])
        if tool is None:
            return (f"error: unknown tool {call['name']!r}; available: "
                    f"{', '.join(self.tools.names())}")
        raw = "".join(call["args"]).strip() or "{}"
        try:
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (ValueError, RecursionError) as e:
            # RecursionError: pathologically nested JSON from the model must
            # degrade to a tool error, not unwind the REPL.
            return f"error: invalid arguments JSON ({e})"
        return _call_with_timeout(tool, args, self.tool_timeout_s)
