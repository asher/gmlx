#!/usr/bin/env python3
"""End-to-end drivers for the `gmlx chat` REPL.

Two tiers, both CPU-only and deterministic - no GPU, no real terminal, no model:

* **Tier 1 - loop driver.** Stub the model load + streaming generator and feed a
  scripted line list through the REPL's one input seam (``state.input_fn``),
  then run the real :func:`gmlx.chat.cmd_chat`. This exercises the actual
  multi-turn loop: KV-cache continuity across turns, every ``/command`` dispatched
  through the loop (not just the helper in isolation), ``/!`` shell staging ->
  next-turn composition, ``/load`` prefill, ``/image``/``/audio`` media staging and
  per-turn marker pinning, ``/drop``, ``r``/``q``/Ctrl-D, stop-sequence trimming,
  and the per-reply stat line.

* **Tier 2 - prompt_toolkit keystroke harness.** Drive the *real* session that
  :func:`gmlx.chat._wire_ptk` builds, over a pipe input + dummy output, feeding
  raw keystrokes and asserting on the accepted line / live toolbar. This covers the
  editing features the scripted path can't: completion-as-you-type, fish-style ghost
  auto-suggest from history, the bottom toolbar, and bracketed paste.

The genuinely-needs-a-tty bits (the termios Esc-cancel of a streaming reply, real
prompt_toolkit rendering) live in the model-loading pty smoke under ``tests/e2e/``.
"""

from __future__ import annotations

import collections
import importlib

import pytest

mlx_lm = pytest.importorskip("mlx_lm")  # noqa: F841 - gate: real sampler/cache modules

from gmlx import chat  # noqa: E402

# A streaming chunk carries only the three fields _stream_reply reads.
_Chunk = collections.namedtuple("_Chunk", "text generation_tokens generation_tps")


class _Scripted:
    """A scripted stand-in for one human at the keyboard.

    Each call returns the next line; it records ``(prompt, pending)`` so a test
    can assert what the prompt string looked like (e.g. the ``(+1) >>`` staged
    indicator) and that a ``/load`` prefill reached the input layer. Running out
    of lines raises ``EOFError`` - the REPL treats that as Ctrl-D and exits 0.
    """

    def __init__(self, lines):
        self.lines = list(lines)
        self.calls = []  # [(prompt, pending), ...]

    def __call__(self, prompt, pending):
        self.calls.append((prompt, pending))
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)

    @property
    def prompts(self):
        return [p for p, _ in self.calls]


class _FakeStream:
    """Stand-in for mlx-lm's ``stream_generate``: records each turn's prompt +
    KV-cache identity + sampling kwargs, then yields the canned reply word by
    word (so the streaming + stat-line path runs for real)."""

    def __init__(self, reply="Hello there"):
        self.reply = reply
        self.turns = []  # one dict per generate call

    def __call__(self, model, tok, prompt, *, max_tokens=None, sampler=None,
                 logits_processors=None, prompt_cache=None, **kv):
        self.turns.append({
            "prompt": prompt,
            "cache_id": id(prompt_cache),
            "max_tokens": max_tokens,
            "logits_processors": logits_processors,
            "kv": kv,
        })
        words = self.reply.split(" ")
        for i, w in enumerate(words):
            yield _Chunk(w if i == 0 else " " + w, i + 1, 42.0)


class _FakeTok:
    chat_template = "{{ messages }}"
    eos_token_ids = [0]

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        # Return the messages verbatim so a test can read back exactly what the
        # turn built (system-on-first-turn, staged-block composition, ...).
        return {"messages": messages, "kwargs": kw}

    def encode(self, s, **kw):
        # The turn path renders once then encodes the render; pass the dict
        # through so tests keep inspecting the turn's messages via the prompt.
        return s if isinstance(s, dict) else [1]


class _TTYStdin:
    """Just enough stdin to make the background-load gate see a tty. Input
    still flows through the scripted ``input_fn``; Esc-cancel is stubbed."""

    def isatty(self):
        return True


class _InertEsc:
    """Esc-cancel stand-in for the fake-tty tests (pytest's stdin has no
    terminal fd to put into cbreak)."""

    def __init__(self, on_toggle=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def pressed(self):
        return False


def _run_text_chat(monkeypatch, tmp_path, lines, *, extra_argv=(), reply="Hello there",
                   tty=False, load_model=None):
    """Run the real REPL over a scripted line list with the model layer faked.

    Returns ``(rc, scripted, stream)`` - the exit code, the scripted keyboard
    (``.turns`` it never has; use ``.calls``/``.prompts``) and the fake streamer
    (``.turns`` = one record per generated reply). ``tty=True`` makes stdin
    report a tty (arming chat's background load); ``load_model`` overrides the
    stock loader fake.
    """
    scripted = _Scripted(lines)
    stream = _FakeStream(reply)
    caches = []

    def fake_mpc(model, max_kv_size=None):
        c = ["cache", len(caches)]          # a fresh, identity-distinct object
        caches.append(c)
        return c

    monkeypatch.setattr(chat, "_wire_input", lambda no_history: chat.ChatState(
        history_enabled=True, history_loaded=True, input_fn=scripted))
    monkeypatch.setattr("gmlx.cli.maybe_load_from_config", lambda *a, **k: None)
    monkeypatch.setattr("gmlx.loader.load_model",
                        load_model or (lambda *a, **k: (object(), {}, _FakeTok())))
    monkeypatch.setattr("gmlx.loader._resolve_prefill_step",
                        lambda model, step: (None, False))
    monkeypatch.setattr("gmlx.cli._apply_placement", lambda args, model: None)
    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", fake_mpc)
    if tty:
        monkeypatch.setattr(chat.sys, "stdin", _TTYStdin())
        monkeypatch.setattr(chat, "_EscCancel", _InertEsc)
    # `mlx_lm.generate` is re-exported as a *function* on the package, shadowing
    # the submodule attribute - so both a string target and `import ... as` resolve
    # to the function. `import_module` returns the real submodule from sys.modules
    # (the same object `from mlx_lm.generate import ...` reads), so patch that.
    monkeypatch.setattr(
        importlib.import_module("mlx_lm.generate"), "stream_generate", stream)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    rc = chat.cmd_chat([str(gguf), *extra_argv])
    return rc, scripted, stream


# ------------------------- Tier 1 - loop driver (text) -------------------------

class _KV:
    """A trimmable fake layer cache (offset semantics like mlx-lm KVCache)."""

    def __init__(self):
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= min(self.offset, n)


class _OffsetStream(_FakeStream):
    """A fake streamer that also appends 10 tokens to the prompt cache per
    reply, so the retry/undo KV-trim path runs for real."""

    def __call__(self, model, tok, prompt, *, prompt_cache=None, **kw):
        for c in prompt_cache or []:
            c.offset += 10
        yield from super().__call__(model, tok, prompt, prompt_cache=prompt_cache, **kw)


class _FrozenKV(_KV):
    """An offset-bearing layer cache that cannot rewind (recurrent state)."""

    def is_trimmable(self):
        return False


def _run_text_chat_kv(monkeypatch, tmp_path, lines, *, extra_argv=(), kv_cls=_KV):
    """`_run_text_chat` variant with offset-bearing caches (``kv_cls=_FrozenKV``
    models a gated-delta hybrid whose state cache can't trim)."""
    scripted = _Scripted(lines)
    stream = _OffsetStream("Hello there")
    caches = []

    def fake_mpc(model, max_kv_size=None):
        c = [kv_cls(), kv_cls()]
        caches.append(c)
        return c

    monkeypatch.setattr(chat, "_wire_input", lambda no_history: chat.ChatState(
        history_enabled=True, history_loaded=True, input_fn=scripted))
    monkeypatch.setattr("gmlx.cli.maybe_load_from_config", lambda *a, **k: None)
    monkeypatch.setattr("gmlx.loader.load_model",
                        lambda *a, **k: (object(), {}, _FakeTok()))
    monkeypatch.setattr("gmlx.loader._resolve_prefill_step",
                        lambda model, step: (None, False))
    monkeypatch.setattr("gmlx.cli._apply_placement", lambda args, model: None)
    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", fake_mpc)
    monkeypatch.setattr(
        importlib.import_module("mlx_lm.generate"), "stream_generate", stream)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    rc = chat.cmd_chat([str(gguf), *extra_argv])
    return rc, scripted, stream, caches


def test_retry_trims_kv_and_resends_same_content(monkeypatch, tmp_path, capsys):
    rc, _s, stream, caches = _run_text_chat_kv(
        monkeypatch, tmp_path, ["hi there", "/retry", "/exit"])
    assert rc == 0
    assert len(stream.turns) == 2                       # original + regenerated
    assert (stream.turns[0]["prompt"]["messages"]
            == stream.turns[1]["prompt"]["messages"])   # identical composed turn
    # the cache was trimmed back to the pre-turn checkpoint before the re-send
    assert caches[0][0].offset == 10                    # not 20
    assert "regenerating" in capsys.readouterr().out


def test_undo_removes_exchange_and_restores_first_turn(monkeypatch, tmp_path, capsys):
    rc, _s, stream, caches = _run_text_chat_kv(
        monkeypatch, tmp_path, ["hi", "/undo", "again", "/exit"],
        extra_argv=["--system-prompt", "SYS"])
    assert rc == 0
    assert len(stream.turns) == 2
    # undo restored first_turn, so the system prompt is re-sent on the retry
    assert stream.turns[1]["prompt"]["messages"][0] == {
        "role": "system", "content": "SYS"}
    assert caches[0][0].offset == 10
    assert "last exchange removed" in capsys.readouterr().out


def test_undo_with_no_history_notices(monkeypatch, tmp_path, capsys):
    rc, _s, stream, _c = _run_text_chat_kv(monkeypatch, tmp_path, ["/undo", "/exit"])
    assert rc == 0 and not stream.turns
    assert "nothing to undo" in capsys.readouterr().out


def test_retry_refused_while_staged(monkeypatch, tmp_path, capsys):
    rc, _s, stream, _c = _run_text_chat_kv(
        monkeypatch, tmp_path, ["hi", "/! echo x", "/retry", "/exit"])
    assert rc == 0
    assert len(stream.turns) == 1                       # retry refused, no regen
    assert "staged items pending" in capsys.readouterr().out



def test_multi_turn_shares_one_kv_cache(monkeypatch, tmp_path, capsys):
    rc, _scripted, stream = _run_text_chat(monkeypatch, tmp_path, ["hi", "again", "/exit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert len(stream.turns) == 2                       # two replies generated
    # Both turns prefill onto the same persistent cache (no reset between them).
    assert stream.turns[0]["cache_id"] == stream.turns[1]["cache_id"]
    assert stream.turns[0]["prompt"]["messages"] == [{"role": "user", "content": "hi"}]
    assert stream.turns[1]["prompt"]["messages"] == [{"role": "user", "content": "again"}]
    assert out.count("tok @") == 2                      # a stat line per reply
    assert "Hello there" in out


def test_system_prompt_only_on_first_turn(monkeypatch, tmp_path):
    _rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["hi", "again", "/exit"],
        extra_argv=["--system-prompt", "BE BRIEF"])
    first = stream.turns[0]["prompt"]["messages"]
    second = stream.turns[1]["prompt"]["messages"]
    assert first[0] == {"role": "system", "content": "BE BRIEF"}
    assert first[1]["role"] == "user"
    assert [m["role"] for m in second] == ["user"]      # system not resent


def test_reset_command_starts_a_fresh_cache(monkeypatch, tmp_path, capsys):
    _rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["hi", "/reset", "again", "/exit"])
    out = capsys.readouterr().out
    assert len(stream.turns) == 2                       # /reset is not a turn
    assert stream.turns[0]["cache_id"] != stream.turns[1]["cache_id"]
    assert "conversation reset" in out


def test_exit_quit_and_ctrl_d_all_exit_cleanly(monkeypatch, tmp_path):
    rc_x, _s, sx = _run_text_chat(monkeypatch, tmp_path, ["hi", "/exit"])
    rc_q, _s1, sq = _run_text_chat(monkeypatch, tmp_path, ["hi", "/quit"])
    rc_eof, _s2, se = _run_text_chat(monkeypatch, tmp_path, ["hi"])   # script runs dry
    assert rc_x == rc_q == rc_eof == 0
    assert len(sx.turns) == len(sq.turns) == len(se.turns) == 1


def test_single_letters_are_chat_text_now(monkeypatch, tmp_path):
    _rc, _s, stream = _run_text_chat(monkeypatch, tmp_path, ["q", "r", "h"])
    assert len(stream.turns) == 3                       # each is an ordinary message
    sent = [t["prompt"]["messages"][-1]["content"] for t in stream.turns]
    assert sent == ["q", "r", "h"]


def test_system_command_sets_prompt_and_resets(monkeypatch, tmp_path, capsys):
    _rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path,
        ["hi", "/system BE TERSE", "again", "/system", "/exit"])
    out = capsys.readouterr().out
    assert len(stream.turns) == 2
    assert stream.turns[0]["prompt"]["messages"] == [
        {"role": "user", "content": "hi"}]
    assert stream.turns[1]["prompt"]["messages"][0] == {
        "role": "system", "content": "BE TERSE"}          # applied after the set
    assert stream.turns[0]["cache_id"] != stream.turns[1]["cache_id"]  # reset
    assert "system prompt set (conversation reset)" in out
    assert "system prompt: BE TERSE" in out               # the bare /system report


def test_sampling_command_is_not_a_turn_and_reports(monkeypatch, tmp_path, capsys):
    _rc, _s, stream = _run_text_chat(monkeypatch, tmp_path, ["/temp 0.7", "hi", "/exit"])
    out = capsys.readouterr().out
    assert len(stream.turns) == 1                       # the /command didn't generate
    assert "temp = 0.7" in out


def test_shell_staging_composes_into_next_message(monkeypatch, tmp_path, capsys):
    _rc, scripted, stream = _run_text_chat(
        monkeypatch, tmp_path, ["/! echo hi", "explain this", "/exit"])
    out = capsys.readouterr().out
    assert len(stream.turns) == 1                       # /! stages, doesn't generate
    content = stream.turns[0]["prompt"]["messages"][0]["content"]
    assert "$ echo hi" in content and "hi" in content   # staged shell block ...
    assert "[exit 0 in" in content
    assert "explain this" in content                    # ... prepended to the typed line
    assert "```" in content
    # The prompt for the second line shows the staged-block indicator.
    assert any(p.startswith("(+1)") for p in scripted.prompts)
    assert "staged" in out


def test_empty_line_with_staged_sends_blocks_alone(monkeypatch, tmp_path):
    _rc, _s, stream = _run_text_chat(monkeypatch, tmp_path, ["/! echo hi", "", "/exit"])
    assert len(stream.turns) == 1
    content = stream.turns[0]["prompt"]["messages"][0]["content"]
    assert "$ echo hi" in content
    assert content.strip().endswith("```")              # nothing typed after the block


def test_empty_line_with_nothing_staged_is_skipped(monkeypatch, tmp_path):
    _rc, _s, stream = _run_text_chat(monkeypatch, tmp_path, ["", "  ", "hi", "/exit"])
    assert len(stream.turns) == 1                        # blank lines generate nothing


def test_drop_discards_staged_so_next_empty_line_is_inert(monkeypatch, tmp_path, capsys):
    _rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["/! echo hi", "/drop", "", "/exit"])
    out = capsys.readouterr().out
    assert len(stream.turns) == 0                        # dropped -> empty line inert
    assert "dropped 1 staged item" in out


def test_load_prefills_the_next_prompt(monkeypatch, tmp_path):
    doc = tmp_path / "snippet.txt"
    doc.write_text("def f():\n    return 42\n")
    _rc, scripted, _stream = _run_text_chat(
        monkeypatch, tmp_path, [f"/load {doc}", "and explain it", "/exit"])
    # The /load contents reached the input layer as a pending prefill.
    pendings = [pending for _p, pending in scripted.calls if pending is not None]
    assert pendings == ["def f():\n    return 42"]      # trailing newline trimmed


def test_thinking_budget_reaches_generation(monkeypatch, tmp_path):
    # The --thinking-budget promise: the cap is installed into generation, not
    # just constructible - a ThinkingBudgetProcessor rides the logits_processors
    # handed to stream_generate.
    from gmlx.thinking_budget import ThinkingBudgetProcessor
    _rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["hi", "/exit"],
        extra_argv=["--thinking-budget", "64"])
    assert len(stream.turns) == 1
    procs = [p for p in (stream.turns[0]["logits_processors"] or [])
             if isinstance(p, ThinkingBudgetProcessor)]
    assert len(procs) == 1                              # installed into the turn
    assert procs[0].budget == 64


def test_stop_sequence_trims_the_reply(monkeypatch, tmp_path, capsys):
    _rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["go", "/exit"],
        extra_argv=["--stop", "beta"], reply="alpha beta gamma")
    out = capsys.readouterr().out
    assert len(stream.turns) == 1
    assert "alpha" in out and "gamma" not in out         # ended at the stop sequence


# ------------------- Tier 1 - loop driver (background load) -------------------
# With a tty stdin the text path loads on a worker while the REPL comes up;
# the join happens when the model is first needed: a turn, a cache-touching
# /command verb (reset/retry/undo/load-session), or --resume restore. Pure
# state /commands run without joining.

def _recording_load(seen):
    def fake_load(*a, **k):
        import threading

        seen["thread"] = threading.current_thread().name
        return (object(), {}, _FakeTok())

    return fake_load


def test_bg_load_worker_thread_and_deferred_summary(monkeypatch, tmp_path, capsys):
    seen = {}
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["hi", "/exit"], tty=True,
        load_model=_recording_load(seen))
    out = capsys.readouterr().out
    assert rc == 0
    assert seen["thread"] == "gmlx-chat-load"
    assert len(stream.turns) == 1                       # turn ran after the join
    # the REPL banner came up before the deferred [load] summary printed
    assert out.index("'/help' lists commands") < out.index("[load] model.gguf")


def test_bg_load_exit_skips_the_join(monkeypatch, tmp_path, capsys):
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["/exit"], tty=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert stream.turns == []
    assert "[load]" not in out                           # never joined


def test_bg_load_state_command_does_not_join(monkeypatch, tmp_path, capsys):
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["/history off", "/exit"], tty=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert stream.turns == []
    assert "history" in out                              # the command ran
    assert "[load]" not in out                           # without joining


def test_bg_load_joins_at_turn_after_state_commands(monkeypatch, tmp_path, capsys):
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["/history off", "hi", "/exit"], tty=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert len(stream.turns) == 1
    # the state command got its output (and a fresh prompt) before the join
    assert out.index("history") < out.index("[load] model.gguf")


def test_bg_load_cache_verb_joins(monkeypatch, tmp_path, capsys):
    seen = {}
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["/reset", "/exit"], tty=True,
        load_model=_recording_load(seen))
    out = capsys.readouterr().out
    assert rc == 0
    assert stream.turns == []
    assert seen["thread"] == "gmlx-chat-load"
    assert "[load] model.gguf" in out       # /reset touches the cache -> joined


def test_bg_load_stray_writes_deferred_to_join(monkeypatch, tmp_path, capsys):
    def noisy_load(*a, **k):
        import sys

        print("W: added a special token", file=sys.stderr)
        return (object(), {}, _FakeTok())

    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["hi", "/exit"], tty=True,
        load_model=noisy_load)
    assert rc == 0
    assert len(stream.turns) == 1
    # the loader's stray stderr write surfaced at the join, not over the prompt
    assert "W: added a special token" in capsys.readouterr().err


def test_bg_load_error_reraised_at_first_submit(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("no such tensor")

    with pytest.raises(RuntimeError, match="no such tensor"):
        _run_text_chat(monkeypatch, tmp_path, ["hi"], tty=True, load_model=boom)


def test_bg_load_resume_joins_before_restore(monkeypatch, tmp_path, capsys):
    seen = {}
    rc, _s, _stream = _run_text_chat(
        monkeypatch, tmp_path, ["/exit"], tty=True,
        load_model=_recording_load(seen), extra_argv=["--resume"])
    out = capsys.readouterr().out
    assert rc == 0
    assert seen["thread"] == "gmlx-chat-load"
    assert "[load] model.gguf" in out                    # joined despite /exit
    assert "no saved session" in out


def test_bg_load_kill_switch_restores_sync_order(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GMLX_CHAT_BG_LOAD", "0")
    seen = {}
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["hi", "/exit"], tty=True,
        load_model=_recording_load(seen))
    out = capsys.readouterr().out
    assert rc == 0
    assert seen["thread"] == "MainThread"
    assert len(stream.turns) == 1
    assert out.index("[load] model.gguf") < out.index("'/help' lists commands")


# --------------------- Tier 1 - loop driver (multimodal / VLM) -----------------

class _FakeVlmStream:
    """Records the image/audio lists each VLM turn is generated with."""

    def __init__(self, reply="A cat."):
        self.reply = reply
        self.turns = []

    def __call__(self, model, processor, prompt, *, image=None, audio=None, **kw):
        self.turns.append({"image": image, "audio": audio, "prompt": prompt})
        for i, w in enumerate(self.reply.split(" ")):
            yield _Chunk(w if i == 0 else " " + w, i + 1, 30.0)


def _run_vlm_chat(monkeypatch, tmp_path, lines):
    pytest.importorskip("mlx_vlm")
    scripted = _Scripted(lines)
    vstream = _FakeVlmStream()
    msg_calls = []

    def fake_vlm_message(model_type, content, role="user", n_images=0, n_audios=0):
        rec = {"role": role, "content": content,
               "n_images": n_images, "n_audios": n_audios}
        msg_calls.append(rec)
        return rec

    monkeypatch.setattr(chat, "_wire_input", lambda no_history: chat.ChatState(
        history_enabled=True, history_loaded=True, input_fn=scripted))
    monkeypatch.setattr("gmlx.cli.maybe_load_from_config", lambda *a, **k: None)
    monkeypatch.setattr("gmlx.vlm.load_vlm_model",
                        lambda *a, **k: (object(), {"model_type": "fake"}, object()))
    monkeypatch.setattr(chat, "_vlm_message", fake_vlm_message)
    monkeypatch.setattr("mlx_vlm.prompt_utils.get_chat_template",
                        lambda processor, msgs, add_generation_prompt=True, **kw: msgs)
    monkeypatch.setattr(
        importlib.import_module("mlx_vlm.generate"), "stream_generate", vstream)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    proj = tmp_path / "mmproj.gguf"
    proj.write_bytes(b"GGUF")
    rc = chat.cmd_chat([str(gguf), "--mmproj", str(proj)])
    return rc, scripted, vstream, msg_calls


def test_vlm_pins_media_markers_per_turn(monkeypatch, tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG")
    snd = tmp_path / "clip.wav"
    snd.write_bytes(b"RIFF")
    rc, _s, vstream, msgs = _run_vlm_chat(monkeypatch, tmp_path, [
        f"/image {img}", "what is this", f"/audio {snd}", "and this", "/exit",
    ])
    assert rc == 0
    assert len(vstream.turns) == 2
    # Turn 1: the image rode along; no audio yet.
    assert vstream.turns[0]["image"] == [str(img)]
    assert vstream.turns[0]["audio"] is None
    # Turn 2: the earlier image is still attached (markers pinned), audio added.
    assert vstream.turns[1]["image"] == [str(img)]
    assert vstream.turns[1]["audio"] == [str(snd)]
    # Each user turn carries exactly the media staged for *that* turn.
    users = [m for m in msgs if m["role"] == "user"]
    assert (users[0]["n_images"], users[0]["n_audios"]) == (1, 0)
    assert (users[1]["n_images"], users[1]["n_audios"]) == (0, 1)


def test_vlm_dragged_in_path_stages_without_a_turn(monkeypatch, tmp_path):
    img = tmp_path / "drag.png"
    img.write_bytes(b"\x89PNG")
    _rc, _s, vstream, msgs = _run_vlm_chat(monkeypatch, tmp_path, [
        str(img), "describe", "/exit",                       # bare path == Finder drag
    ])
    assert len(vstream.turns) == 1                        # the drag line isn't a turn
    assert vstream.turns[0]["image"] == [str(img)]
    assert [m for m in msgs if m["role"] == "user"][0]["n_images"] == 1


# -------------- Tier 1 - loop driver (VLM x MTP: text-only turns) --------------

class _FakeSpecStream:
    """Stand-in for generation.stream_generate_speculative: records each text-only MTP
    turn's prompt + KV-cache identity, then streams the canned reply word by word."""

    def __init__(self, reply="Sure thing"):
        self.reply = reply
        self.turns = []

    def __call__(self, model, drafter, tok, prompt, *, prompt_cache=None, **kw):
        self.turns.append({"prompt": prompt, "cache_id": id(prompt_cache)})
        for i, w in enumerate(self.reply.split(" ")):
            yield _Chunk(w if i == 0 else " " + w, i + 1, 99.0)


def _run_vlm_mtp_chat(monkeypatch, tmp_path, lines, *, native=False):
    """A --mmproj base with a drafter: text-only turns route through MTP, media
    turns through the plain VLM stream - all in one scripted session.

    ``native=False`` supplies a --draft-gguf assistant (gemma4); ``native=True``
    drops it and pretends the LLM GGUF carries a native head (qwen3.5/3.6)."""
    pytest.importorskip("mlx_vlm")
    scripted = _Scripted(lines)
    vstream = _FakeVlmStream()
    sstream = _FakeSpecStream()
    msg_calls = []
    caches = []

    def fake_vlm_message(model_type, content, role="user", n_images=0, n_audios=0):
        rec = {"role": role, "content": content,
               "n_images": n_images, "n_audios": n_audios}
        msg_calls.append(rec)
        return rec

    def fake_mpc(model):                          # mlx_vlm make_prompt_cache(lm)
        c = ["vlm-text-cache", len(caches)]       # fresh, identity-distinct
        caches.append(c)
        return c

    class _Model:
        language_model = object()

    monkeypatch.setattr(chat, "_wire_input", lambda no_history: chat.ChatState(
        history_enabled=True, history_loaded=True, input_fn=scripted))
    monkeypatch.setattr("gmlx.cli.maybe_load_from_config", lambda *a, **k: None)
    monkeypatch.setattr(
        "gmlx.mtp_load.load_vlm_mtp_model",
        lambda *a, **k: (_Model(), object(), {"model_type": "fake"}, _FakeTok(),
                         object()))
    monkeypatch.setattr("gmlx.generation.stream_generate_speculative", sstream)
    monkeypatch.setattr(chat, "_vlm_message", fake_vlm_message)
    monkeypatch.setattr("mlx_vlm.prompt_utils.get_chat_template",
                        lambda processor, msgs, add_generation_prompt=True, **kw: msgs)
    monkeypatch.setattr("mlx_vlm.models.cache.make_prompt_cache", fake_mpc)
    monkeypatch.setattr(
        importlib.import_module("mlx_vlm.generate"), "stream_generate", vstream)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    proj = tmp_path / "mmproj.gguf"
    proj.write_bytes(b"GGUF")
    argv = [str(gguf), "--mmproj", str(proj)]
    if native:
        monkeypatch.setattr("gmlx.cli._has_native_mtp_head",
                            lambda *a, **k: True)
    else:
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF")
        argv += ["--draft-gguf", str(draft)]
    rc = chat.cmd_chat(argv)
    return rc, scripted, vstream, sstream, msg_calls


def test_vlm_mtp_text_turns_use_mtp_then_image_falls_back(monkeypatch, tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG")
    rc, _s, vstream, sstream, msgs = _run_vlm_mtp_chat(monkeypatch, tmp_path, [
        "hello", "again", f"/image {img}", "what is this", "more text", "/exit",
    ])
    assert rc == 0
    # The two text-only turns ran on the MTP engine over one persistent text cache.
    assert len(sstream.turns) == 2
    assert sstream.turns[0]["cache_id"] == sstream.turns[1]["cache_id"]
    # Incremental templating: each MTP turn sends only the current user message
    # (the cache holds the history), exactly like the plain text MTP path.
    assert sstream.turns[0]["prompt"]["messages"] == [
        {"role": "user", "content": "hello"}]
    assert sstream.turns[1]["prompt"]["messages"] == [
        {"role": "user", "content": "again"}]
    # The image turn and the text turn after it both used the plain VLM stream.
    assert len(vstream.turns) == 2
    assert vstream.turns[0]["image"] == [str(img)]
    # History survived the switch: the VLM re-prefill carries the earlier text turns.
    image_turn_history = vstream.turns[0]["prompt"]
    contents = [m["content"] for m in image_turn_history if m["role"] == "user"]
    assert "hello" in contents and "again" in contents
    # The MTP text turns recorded their user message into vlm_msgs as plain text.
    users = [m for m in msgs if m["role"] == "user"]
    assert (users[0]["n_images"], users[1]["n_images"]) == (0, 0)
    assert users[2]["n_images"] == 1                  # the image turn


def test_vlm_mtp_native_head_text_turns_use_mtp(monkeypatch, tmp_path):
    # A qwen3.5/3.6 VLM (native head, no --draft-gguf): text-only turns still take
    # the MTP engine; with no media the plain VLM stream is never touched.
    rc, _s, vstream, sstream, _msgs = _run_vlm_mtp_chat(
        monkeypatch, tmp_path, ["hello", "again", "/exit"], native=True)
    assert rc == 0
    assert len(sstream.turns) == 2
    assert sstream.turns[0]["cache_id"] == sstream.turns[1]["cache_id"]
    assert len(vstream.turns) == 0


# ------------------ Tier 2 - prompt_toolkit session harness ------------------
#
# These drive the REAL session that `_wire_ptk` builds, over a pipe input + dummy
# output, with no terminal. Two kinds of assertion:
#   * synchronous keystroke round-trips through the live session (plain accept,
#     bracketed paste) - the session reads + accepts piped input end to end;
#   * the components we wrote, exercised through the exact objects the session
#     installs (the slash completer, the toggleable history + the ghost suggestion
#     it feeds, the bottom toolbar).
# The async completion-menu / ghost-*accept* rendering is prompt_toolkit's own
# background-task machinery (it needs an event-loop turn that a single key feed
# doesn't give) - that interactive layer is covered by the pty smoke in tests/e2e.

import asyncio  # noqa: E402
import contextlib  # noqa: E402

pytest.importorskip("prompt_toolkit")
from prompt_toolkit.application import create_app_session  # noqa: E402
from prompt_toolkit.buffer import Buffer  # noqa: E402
from prompt_toolkit.completion import CompleteEvent  # noqa: E402
from prompt_toolkit.document import Document  # noqa: E402
from prompt_toolkit.input.defaults import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402


def _ptk_state(monkeypatch, tmp_path, *, enabled=True, history=()):
    """A chat state ready for ``_wire_ptk``, with the history file pre-seeded.

    The ``.ptk`` file the session reads is written first so the toggleable
    history (and the auto-suggest it feeds) has prior lines to draw on.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    state = chat.ChatState(
        history_enabled=enabled,
        sampling={"temp": 0.0, "top_p": 0.95, "max_tokens": 1024})
    if history:
        from prompt_toolkit.history import FileHistory
        hist_file = chat._history_path().with_suffix(".ptk")
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        fh = FileHistory(str(hist_file))
        for line in history:
            fh.store_string(line)
    return state


@contextlib.contextmanager
def _session(state):
    """The real ptk session, constructed *inside* a pipe app-session so it binds
    to the pipe rather than the real stdin. Yields ``(session, pipe)``."""
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            assert chat._wire_ptk(state) is True
            yield state.ptk_session, pipe


def _accept(state, keys):
    """Feed ``keys`` to the live session and return the line it accepts."""
    with _session(state) as (session, pipe):
        pipe.send_text(keys)
        return session.prompt(">> ")


async def _load_history(history):
    async for _ in history.load():   # populate get_strings() from the file
        pass


# Synchronous keystroke round-trips through the live session.

def test_ptk_session_round_trips_a_typed_line(monkeypatch, tmp_path):
    state = _ptk_state(monkeypatch, tmp_path)
    assert _accept(state, "hello there\r") == "hello there"


def test_ptk_bracketed_paste_keeps_newlines(monkeypatch, tmp_path):
    state = _ptk_state(monkeypatch, tmp_path)
    # Bracketed-paste markers turn the embedded newline into buffer text instead
    # of an accept, so a multi-line paste lands as one message.
    assert _accept(state, "\x1b[200~line one\nline two\x1b[201~\r") == "line one\nline two"


# The slash completer the session installs (our `_completion_options`).

def test_ptk_completer_drives_slash_and_arg_completion(monkeypatch, tmp_path):
    state = _ptk_state(monkeypatch, tmp_path)
    with _session(state) as (session, _pipe):
        comp = session.completer

        def complete(line):
            doc = Document(line, len(line))
            return sorted(c.text for c in comp.get_completions(doc, CompleteEvent()))

        # A unique prefix yields exactly one command; the bare "/" yields all of them.
        assert [(c.text, c.start_position) for c in
                comp.get_completions(Document("/te", 3), CompleteEvent())] == [("/temp ", -3)]
        allcmds = complete("/")
        assert "/temp " in allcmds and "/load " in allcmds and "/image " in allcmds
        assert len(allcmds) == len(chat._ALL_COMMANDS)
        # /history's enum arguments complete after the space.
        assert complete("/history ") == ["clear", "off", "on"]
        # Ordinary chat text stays inert - no candidates outside the command surface.
        assert list(comp.get_completions(Document("hello", 5), CompleteEvent())) == []


# The toggleable history + the ghost suggestion it feeds (our `/history on|off`).

def _suggestion(state):
    with _session(state) as (session, _pipe):
        asyncio.run(_load_history(session.history))
        buf = Buffer(history=session.history, auto_suggest=session.auto_suggest)
        sug = session.auto_suggest.get_suggestion(buf, Document("hel", 3))
        return (sug.text if sug else None), list(session.history.get_strings())


def test_ptk_enabled_history_feeds_ghost_suggestion(monkeypatch, tmp_path):
    state = _ptk_state(monkeypatch, tmp_path, history=["hello world"])
    text, strings = _suggestion(state)
    assert strings == ["hello world"]
    assert text == "lo world"          # "hel" + "lo world" == the prior line


def test_ptk_disabled_history_suppresses_load_and_suggestion(monkeypatch, tmp_path):
    state = _ptk_state(monkeypatch, tmp_path, enabled=False, history=["hello world"])
    text, strings = _suggestion(state)
    assert strings == []               # history off -> nothing loaded ...
    assert text is None                # ... so no ghost suggestion either


def test_ptk_enabled_history_persists_accepted_line(monkeypatch, tmp_path):
    state = _ptk_state(monkeypatch, tmp_path)
    _accept(state, "remember me\r")
    hist_file = chat._history_path().with_suffix(".ptk")
    assert "remember me" in hist_file.read_text()


def test_ptk_disabled_history_does_not_persist_accepted_line(monkeypatch, tmp_path):
    state = _ptk_state(monkeypatch, tmp_path, enabled=False)
    _accept(state, "do not save me\r")
    hist_file = chat._history_path().with_suffix(".ptk")
    saved = hist_file.read_text() if hist_file.exists() else ""
    assert "do not save me" not in saved


# The bottom toolbar closure, bound to live state.

def test_ptk_toolbar_reflects_live_state(monkeypatch, tmp_path):
    state = _ptk_state(monkeypatch, tmp_path)
    with _session(state) as (session, _pipe):
        toolbar = session.bottom_toolbar             # the real closure over `state`
        assert "temp=0" in toolbar()
        state.sampling["temp"] = 0.7
        state.staged = ["block"]
        state.staged_images = ["a.png", "b.png"]
        state.last_tps = 12.3
        text = toolbar()
    assert "temp=0.7" in text
    assert "+1 staged" in text
    assert "+2 img" in text
    assert "12.3 tok/s" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ------------------------- B-inc-F - session persistence -------------------------


def _saved_sessions():
    from gmlx import sessions as ss

    return {s["name"]: s for s in ss.list_sessions()}


def test_autosave_writes_after_each_turn(monkeypatch, tmp_path):
    from gmlx import sessions as ss

    _rc, _s, stream = _run_text_chat(monkeypatch, tmp_path, ["hi", "again", "/exit"])
    saved = _saved_sessions()
    assert len(saved) == 1
    doc, _ = ss.load_session(next(iter(saved)))
    msgs = doc["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "hi" and msgs[2]["content"] == "again"
    assert doc["model"]["path"].endswith("model.gguf")


def test_no_autosave_flag(monkeypatch, tmp_path):
    _run_text_chat(monkeypatch, tmp_path, ["hi", "/exit"],
                   extra_argv=["--no-autosave"])
    assert not _saved_sessions()


def test_undo_updates_the_autosaved_file(monkeypatch, tmp_path):
    from gmlx import sessions as ss

    _run_text_chat_kv(monkeypatch, tmp_path, ["hi", "again", "/undo", "/exit"])
    doc, _ = ss.load_session(next(iter(_saved_sessions())))
    assert [m["content"] for m in doc["messages"]] == ["hi", "Hello there"]


def test_save_and_export_commands(monkeypatch, tmp_path, capsys):
    from gmlx import sessions as ss

    md = tmp_path / "out.md"
    _run_text_chat(monkeypatch, tmp_path,
                   ["hi", "/save mychat", f"/export {md}", "/exit"],
                   extra_argv=["--no-autosave"])
    out = capsys.readouterr().out
    assert "saved" in out and "exported" in out
    doc, _ = ss.load_session("mychat")
    assert doc["messages"][0]["content"] == "hi"
    text = md.read_text()
    assert "## User" in text and "Hello there" in text


def test_load_session_restores_history_and_replays(monkeypatch, tmp_path, capsys):
    from gmlx import sessions as ss

    gguf = tmp_path / "model.gguf"
    ss.save_session({
        "model": {"path": str(gguf)},
        "system_prompt": "SYS",
        "sampling": {"temp": 0.9},
        "messages": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer",
             "stats": {"gen_tokens": 7}},
        ],
    }, "prior")
    _rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["/sessions", "/load-session prior", "new q", "/exit"],
        extra_argv=["--no-autosave"])
    out = capsys.readouterr().out
    assert "prior" in out and "resumed 'prior': 1 turn" in out
    assert len(stream.turns) == 1
    # the deferred replay prepends the full history before the new turn
    assert stream.turns[0]["prompt"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new q"},
    ]


def test_resume_flag_picks_latest_for_model(monkeypatch, tmp_path, capsys):
    from gmlx import sessions as ss

    gguf = tmp_path / "model.gguf"
    ss.save_session({
        "model": {"path": str(gguf)},
        "messages": [{"role": "user", "content": "before"},
                     {"role": "assistant", "content": "yes"}],
    }, "latest")
    ss.save_session({"model": {"path": "/elsewhere/other.gguf"}, "messages": []},
                    "othermodel")
    _rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["follow up", "/exit"],
        extra_argv=["--resume", "--no-autosave"])
    assert "resumed 'latest'" in capsys.readouterr().out
    msgs = stream.turns[0]["prompt"]["messages"]
    assert msgs[0]["content"] == "before" and msgs[-1]["content"] == "follow up"


def test_resume_model_mismatch_refuses(monkeypatch, tmp_path, capsys):
    from gmlx import sessions as ss

    ss.save_session({"model": {"path": "/elsewhere/other.gguf"}, "messages": []},
                    "wrongmodel")
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["hi", "/exit"],
        extra_argv=["--resume", "wrongmodel", "--no-autosave"])
    assert rc == 2 and not stream.turns
    assert "recorded with /elsewhere/other.gguf" in capsys.readouterr().err


def test_resume_with_no_sessions_starts_fresh(monkeypatch, tmp_path, capsys):
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["hi", "/exit"],
        extra_argv=["--resume", "--no-autosave"])
    assert rc == 0 and len(stream.turns) == 1
    assert "starting fresh" in capsys.readouterr().out
    assert stream.turns[0]["prompt"]["messages"] == [
        {"role": "user", "content": "hi"}]


def test_undo_on_restored_turn_rebuilds_cleanly(monkeypatch, tmp_path, capsys):
    from gmlx import sessions as ss

    gguf = tmp_path / "model.gguf"
    ss.save_session({
        "model": {"path": str(gguf)},
        "messages": [{"role": "user", "content": "old"},
                     {"role": "assistant", "content": "reply"}],
    }, "prior")
    rc, _s, stream = _run_text_chat(
        monkeypatch, tmp_path, ["/undo", "fresh q", "/exit"],
        extra_argv=["--resume", "prior", "--no-autosave"])
    assert rc == 0
    assert "last exchange removed" in capsys.readouterr().out
    # the resume replay was dropped with the undone turn: a clean fresh start
    assert len(stream.turns) == 1
    assert stream.turns[0]["prompt"]["messages"] == [
        {"role": "user", "content": "fresh q"}]


def test_undo_on_untrimmable_cache_rebuilds_with_replay(monkeypatch, tmp_path, capsys):
    rc, _s, stream, caches = _run_text_chat_kv(
        monkeypatch, tmp_path, ["hi", "again", "/undo", "more", "/exit"],
        kv_cls=_FrozenKV)
    assert rc == 0
    assert "can't rewind in place - rebuilt" in capsys.readouterr().out
    assert len(caches) == 2                             # fresh cache built
    assert len(stream.turns) == 3
    assert stream.turns[2]["cache_id"] == id(caches[1])
    # surviving history re-prefills ahead of the next message
    assert stream.turns[2]["prompt"]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "more"},
    ]


def test_retry_on_untrimmable_cache_rebuilds_and_resends(monkeypatch, tmp_path, capsys):
    rc, _s, stream, caches = _run_text_chat_kv(
        monkeypatch, tmp_path, ["hi", "again", "/retry", "/exit"],
        kv_cls=_FrozenKV)
    assert rc == 0
    assert "regenerating" in capsys.readouterr().out
    assert len(caches) == 2 and len(stream.turns) == 3
    assert stream.turns[2]["prompt"]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "again"},           # identical re-send
    ]


def test_reset_drops_pending_rebuild_replay(monkeypatch, tmp_path, capsys):
    """An /undo rebuild leaves replay_messages pending; a following /reset
    must drop it or the pre-reset history re-prefills into the fresh chat."""
    rc, _s, stream, caches = _run_text_chat_kv(
        monkeypatch, tmp_path,
        ["hi", "again", "/undo", "/reset", "fresh", "/exit"],
        kv_cls=_FrozenKV)
    assert rc == 0
    assert stream.turns[2]["prompt"]["messages"] == [
        {"role": "user", "content": "fresh"},
    ]


def test_ptk_history_on_backfills_the_file_mid_session(monkeypatch, tmp_path):
    """`--no-history` then `/history on`: prompt_toolkit caches the (empty) load
    result at the first prompt, so enabling has to drop that cache or the prior
    sessions' lines never come back (the readline shim re-reads the file)."""
    state = _ptk_state(monkeypatch, tmp_path, enabled=False,
                       history=["hello world"])
    with _session(state) as (session, _pipe):
        asyncio.run(_load_history(session.history))     # the first prompt's load
        assert list(session.history.get_strings()) == []

        assert chat._handle_slash("/history on", state) is None
        assert state.history_enabled is True

        asyncio.run(_load_history(session.history))     # the next prompt's load
        assert list(session.history.get_strings()) == ["hello world"]
