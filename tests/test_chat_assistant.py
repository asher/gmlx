#!/usr/bin/env python3
"""``gmlx chat --assistant`` - the server-backed tool-loop turn engine.

CPU-only and serverless: ``_setup_assistant`` is monkeypatched to hand the
REPL a scripted fake brain (the AssistantBrain surface: ``messages``,
``turn()`` events, ``reset``/``close``), so the tests drive the *real*
``cmd_chat`` loop - turn rendering, ledger/undo/retry with multi-message tool
rounds, session round-trips seeding ``brain.messages``, the /memory command,
sampling forwarding via the shared ``extra`` dict, and the flag gate."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("mlx_lm")  # noqa: F841 - cmd_chat imports sampler modules

from gmlx import chat  # noqa: E402
from gmlx.assistant_brain import ToolRegistry  # noqa: E402


class _Scripted:
    """Scripted keyboard: one line per call, EOFError (Ctrl-D) when done."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.calls = []

    def __call__(self, prompt, pending):
        self.calls.append((prompt, pending))
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


class _FakeBrain:
    """AssistantBrain stand-in: owns ``messages``, yields the event stream,
    and (like the real one) appends a whole tool round per turn when asked."""

    def __init__(self, replies=("Hello there",), tool_rounds=0):
        self.messages = []
        self.system = None
        self.max_tokens = 512
        self.memory = None
        self.tools = ToolRegistry()
        self.replies = list(replies)
        self.tool_rounds = tool_rounds
        self.turns = []
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1
        self.messages.clear()

    def close(self):
        self.closed = True

    def turn(self, user_text):
        self.turns.append(user_text)
        self.messages.append({"role": "user", "content": user_text})
        reply = self.replies.pop(0) if self.replies else "ok"
        for i in range(self.tool_rounds):
            yield ("status", f"using tool{i}")
            self.messages.append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": f"c{i}"}]})
            self.messages.append({"role": "tool", "content": "42",
                                  "tool_call_id": f"c{i}"})
        for j, word in enumerate(reply.split(" ")):
            yield ("say", word if j == 0 else " " + word)
        self.messages.append({"role": "assistant", "content": reply})
        yield ("done", {"completion_tokens": 7, "prompt_tokens": 11,
                        "total_tokens": 18, "rounds": self.tool_rounds + 1})


def _run(monkeypatch, tmp_path, lines, *, brain=None, extra_argv=(),
         model="served-model"):
    """Run the real REPL under --assistant with the setup seam faked.
    Returns ``(rc, scripted, brain, extra)`` - ``extra`` is the same live
    dict the turn engine forwards, so tests can read what rode along."""
    scripted = _Scripted(lines)
    brain = brain if brain is not None else _FakeBrain()
    extra = {"stream_options": {"include_usage": True}}
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(chat, "_wire_input", lambda no_history: chat.ChatState(
        history_enabled=True, history_loaded=True, input_fn=scripted))
    monkeypatch.setattr(
        chat, "_setup_assistant",
        lambda args: (brain, model, "http://127.0.0.1:8080/v1", extra))
    rc = chat.cmd_chat(["--assistant", *extra_argv])
    return rc, scripted, brain, extra


# ------------------------------ turn engine ------------------------------

def test_turn_renders_reply_and_stats(monkeypatch, tmp_path, capsys):
    rc, _s, brain, _e = _run(monkeypatch, tmp_path, ["hi there"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Hello there" in out
    assert "7 tok" in out                     # the per-reply stat line ran
    assert brain.turns == ["hi there"]
    assert "assistant mode: served-model via http://127.0.0.1:8080/v1" in out


def test_status_lines_are_transient(monkeypatch, tmp_path, capsys):
    brain = _FakeBrain(tool_rounds=1)
    rc, _s, brain, _e = _run(monkeypatch, tmp_path, ["go"], brain=brain)
    out = capsys.readouterr().out
    assert rc == 0
    assert "[assistant] using tool0..." in out
    assert "\r\x1b[K" in out                  # cleared before the spoken text
    assert "ok" in out


def test_system_prompt_reaches_brain(monkeypatch, tmp_path):
    seen = []
    brain = _FakeBrain()
    real_turn = brain.turn

    def spy_turn(user_text):
        seen.append(brain.system)
        return real_turn(user_text)

    brain.turn = spy_turn
    rc, _s, brain, _e = _run(monkeypatch, tmp_path,
                             ["/system be terse", "hello"], brain=brain)
    assert rc == 0
    assert seen == ["be terse"]


def test_reset_resets_brain(monkeypatch, tmp_path):
    rc, _s, brain, _e = _run(monkeypatch, tmp_path, ["one", "/reset", "two"])
    assert rc == 0
    assert brain.resets == 1
    # After the reset, only turn two's round remains in the history.
    assert [m["content"] for m in brain.messages
            if m["role"] == "user"] == ["two"]


# ----------------------------- undo / retry ------------------------------

def test_undo_truncates_whole_tool_round(monkeypatch, tmp_path, capsys):
    brain = _FakeBrain(replies=["first", "second"], tool_rounds=2)
    rc, _s, brain, _e = _run(monkeypatch, tmp_path,
                             ["one", "two", "/undo"], brain=brain)
    out = capsys.readouterr().out
    assert rc == 0
    assert "last exchange removed" in out
    # Turn two appended 1 user + 2x(assistant+tool) + 1 assistant = 6 messages;
    # /undo must drop all of them, not just a user/assistant pair.
    assert [m["content"] for m in brain.messages
            if m["role"] == "user"] == ["one"]
    assert brain.messages[-1] == {"role": "assistant", "content": "first"}
    assert len(brain.messages) == 6


def test_retry_regenerates_from_brain_checkpoint(monkeypatch, tmp_path):
    brain = _FakeBrain(replies=["first", "second"])
    rc, _s, brain, _e = _run(monkeypatch, tmp_path, ["ask", "/retry"],
                             brain=brain)
    assert rc == 0
    assert brain.turns == ["ask", "ask"]      # same user text re-sent
    # The retried round replaced the first (no duplicate history).
    assert [m["content"] for m in brain.messages] == ["ask", "second"]


# ------------------------------- sessions --------------------------------

def test_session_roundtrip_seeds_brain(monkeypatch, tmp_path, capsys):
    rc, _s, brain, _e = _run(
        monkeypatch, tmp_path,
        ["remember me", "/save roundtrip", "/reset",
         "/load-session roundtrip"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "resumed 'roundtrip': 1 turn" in out
    assert brain.messages == [
        {"role": "user", "content": "remember me"},
        {"role": "assistant", "content": "Hello there"},
    ]


def test_load_session_other_model_refused(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, ["hi", "/save owned"])
    rc, _s, brain, _e = _run(monkeypatch, tmp_path,
                             ["/load-session owned"], model="other-model")
    assert rc == 0
    out = capsys.readouterr().out
    assert "recorded with served-model" in out
    assert brain.messages == []


# ------------------------- sampling passthrough --------------------------

def test_cli_sampling_flags_forward(monkeypatch, tmp_path):
    rc, _s, _b, extra = _run(monkeypatch, tmp_path, ["hi"],
                             extra_argv=["--temp", "0.9", "--seed", "7",
                                         "--stop", "END"])
    assert rc == 0
    assert extra["temperature"] == pytest.approx(0.9)
    assert extra["seed"] == 7
    assert extra["stop"] == ["END"]
    assert "top_p" not in extra               # untouched knobs stay server-side
    assert extra["stream_options"] == {"include_usage": True}


def test_slash_command_moves_knob_live(monkeypatch, tmp_path):
    rc, _s, _b, extra = _run(monkeypatch, tmp_path,
                             ["/top-p 0.5", "hi"])
    assert rc == 0
    assert extra["top_p"] == pytest.approx(0.5)
    assert "temperature" not in extra


def test_max_tokens_flag_reaches_brain(monkeypatch, tmp_path):
    rc, _s, brain, _e = _run(monkeypatch, tmp_path, ["hi"],
                             extra_argv=["--max-tokens", "99"])
    assert rc == 0
    assert brain.max_tokens == 99


# ------------------------- guards + flag gating --------------------------

def test_media_guards(monkeypatch, tmp_path, capsys):
    rc, _s, _b, _e = _run(monkeypatch, tmp_path,
                          ["/image cat.png", "/audio a.wav",
                           "/thinking-budget 100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "image attachments are not available with --assistant" in out
    assert "audio attachments are not available with --assistant" in out
    assert "not available with --assistant" in out  # thinking-budget note


def test_xtc_flags_forward_to_server(monkeypatch, tmp_path):
    rc, _s, _b, extra = _run(monkeypatch, tmp_path, ["hi"],
                             extra_argv=["--xtc-probability", "0.5",
                                         "--xtc-threshold", "0.1"])
    assert rc == 0
    assert extra["xtc_probability"] == pytest.approx(0.5)
    assert extra["xtc_threshold"] == pytest.approx(0.1)


def test_xtc_slash_command_forwards(monkeypatch, tmp_path):
    rc, _s, _b, extra = _run(monkeypatch, tmp_path,
                             ["/xtc-probability 0.5", "hi"])
    assert rc == 0
    assert extra["xtc_probability"] == pytest.approx(0.5)
    assert "xtc_threshold" not in extra        # untouched knob stays server-side


def test_reject_flags_exit_2(monkeypatch, tmp_path, capsys):
    rc = chat.cmd_chat(["--assistant", "--adapter", "x.gguf"])
    assert rc == 2
    assert "--adapter is not supported with --assistant" in (
        capsys.readouterr().err)


def test_noop_flags_print_one_note(monkeypatch, tmp_path, capsys):
    rc, _s, _b, _e = _run(monkeypatch, tmp_path, [],
                          extra_argv=["--speculative", "--kv-bits", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "server owns --speculative, --kv-bits - ignored" in out


def test_gguf_required_without_assistant(capsys):
    with pytest.raises(SystemExit):
        chat.cmd_chat([])
    assert "required without --assistant" in capsys.readouterr().err


# --------------------------- setup ergonomics ----------------------------

def _fake_server(monkeypatch, served, default=None):
    """Fake the endpoint + capability probe under the real _setup_assistant."""
    def ensure(ns):
        ns.base_url = "http://127.0.0.1:8080/v1"
        return None

    monkeypatch.setattr("gmlx.launch._ensure_server", ensure)
    monkeypatch.setattr(
        "gmlx.talk_client.probe_capabilities",
        lambda base_url, api_key=None, timeout=5.0: {
            "stt": False, "tts": False, "chat_ids": list(served),
            "default": default,
        })


def test_file_arg_rejected(monkeypatch, tmp_path, capsys):
    _fake_server(monkeypatch, ["served-model"])
    rc = chat.cmd_chat(["model.gguf", "--assistant"])
    assert rc == 2
    assert "pass a served model id, not a file" in capsys.readouterr().err


def test_unserved_id_rejected(monkeypatch, capsys):
    _fake_server(monkeypatch, ["alpha", "beta"])
    rc = chat.cmd_chat(["gamma", "--assistant"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "'gamma' is not served" in err and "alpha, beta" in err


def test_no_model_no_default_rejected(monkeypatch, capsys):
    _fake_server(monkeypatch, ["alpha", "beta"])
    rc = chat.cmd_chat(["--assistant"])
    assert rc == 2
    assert "no model selected and the server has no default" in (
        capsys.readouterr().err)


# ------------------------------- /memory ---------------------------------

class _StubStore:
    def __init__(self):
        self.rows = [
            {"id": 1, "text": "user likes terse answers",
             "created": time.time() - 7200, "recalled": 3},
            {"id": 2, "text": "project is gmlx",
             "created": time.time() - 90000, "recalled": 0},
        ]

    def count(self):
        return len(self.rows)

    def list_all(self, limit=None):
        return list(reversed(self.rows))[:limit]

    def delete(self, mem_id):
        n = len(self.rows)
        self.rows = [r for r in self.rows if r["id"] != mem_id]
        return len(self.rows) < n

    def clear(self):
        n = len(self.rows)
        self.rows = []
        return n


def test_memory_cmd_list_forget_clear(monkeypatch, tmp_path, capsys):
    brain = _FakeBrain()
    brain.memory = _StubStore()
    rc, _s, brain, _e = _run(
        monkeypatch, tmp_path,
        ["/memory", "/memory forget 1", "/memory clear",
         "/memory clear yes", "/memory"], brain=brain)
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 memories" in out
    assert "user likes terse answers" in out
    assert "forgot #1" in out
    assert "this deletes 1 memories - confirm" in out
    assert "cleared 1 memories" in out
    assert "no memories stored" in out


def test_memory_cmd_without_store(monkeypatch, tmp_path, capsys):
    rc, _s, _b, _e = _run(monkeypatch, tmp_path, ["/memory"])
    assert rc == 0
    assert "no memory store" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
