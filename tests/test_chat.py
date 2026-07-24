#!/usr/bin/env python3
"""`gmlx chat` terminal shim + `run` config/error-path additions.
CPU-only: exercises the pure helpers (slash commands, completion, history
wiring, template-config parsing) and the pre-load validation in `cli.main` -
no model is ever loaded.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from gmlx import chat, cli  # noqa: E402


@pytest.fixture
def state():
    return chat.ChatState(
        history_enabled=True,
        sampling={
            "temp": 0.0,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.05,
            "max_tokens": 1024,
            "xtc_probability": 0.0,
            "xtc_threshold": 0.0,
            "repetition_penalty": 0.0,
            "repetition_context_size": 20,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    )


# parse_template_config
def test_template_config_parses():
    assert chat.parse_template_config('{"enable_thinking": false}') == {
        "enable_thinking": False
    }


def test_template_config_none_and_empty():
    assert chat.parse_template_config(None) == {}
    assert chat.parse_template_config("") == {}


def test_template_config_bad_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        chat.parse_template_config("{enable_thinking: false}")


def test_template_config_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        chat.parse_template_config('["a"]')


def test_chat_template_config_reaches_apply_chat_template():
    """A parsed --chat-template-config dict flows through generate() into the
    tokenizer's apply_chat_template - the generic passthrough forwards ANY key
    (e.g. preserve_thinking for Qwen3.6 / Gemma-4), not just enable_thinking."""
    pytest.importorskip("mlx_lm")
    from gmlx import generation

    template_kwargs = chat.parse_template_config('{"preserve_thinking": true}')
    captured: dict = {}

    class _Stop(Exception):
        pass

    class _Tok:
        chat_template = "{{ x }}"
        eos_token_ids = [0]

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True, **kw):
            captured.update(kw)
            raise _Stop()                       # short-circuit before generation

    with pytest.raises(_Stop):
        generation.generate(object(), _Tok(), "hi", template_kwargs=template_kwargs)
    assert captured == {"preserve_thinking": True}


# /commands
def test_slash_sets_sampling_value(state, capsys):
    assert chat._handle_slash("/temp 0.8", state) is None
    assert state.sampling["temp"] == 0.8
    assert "temp = 0.8" in capsys.readouterr().out


def test_slash_bare_command_prints_current(state, capsys):
    chat._handle_slash("/top-k", state)
    assert "top-k = 0" in capsys.readouterr().out


def test_slash_bad_value_keeps_state(state, capsys):
    chat._handle_slash("/top-k banana", state)
    assert state.sampling["top_k"] == 0
    assert "needs a int" in capsys.readouterr().out


def test_slash_sampling_lists_knobs(state, capsys):
    chat._handle_slash("/sampling", state)
    out = capsys.readouterr().out
    assert "temp=0.0" in out and "frequency-penalty=0.0" in out


def test_slash_clear_requests_reset(state, capsys):
    assert chat._handle_slash("/clear", state) == "reset"
    assert "conversation reset" in capsys.readouterr().out


def test_slash_load_missing_file(state, capsys):
    chat._handle_slash("/load /no/such/file.txt", state)
    assert state.pending_insert is None
    assert "/load:" in capsys.readouterr().out


def test_slash_load_prefills(state, tmp_path, capsys):
    f = tmp_path / "prompt.txt"
    f.write_text("hello from a file\n")
    chat._handle_slash(f"/load {f}", state)
    assert state.pending_insert == "hello from a file"
    assert "edit, then Enter" in capsys.readouterr().out


def test_slash_history_without_readline(state, capsys):
    chat._handle_slash("/history on", state)
    assert "history unavailable" in capsys.readouterr().out


def test_slash_unknown_prints_help(state, capsys):
    chat._handle_slash("/frobnicate", state)
    assert "/history [on|off|clear]" in capsys.readouterr().out


# /! - shell command staging
def test_shell_stages_block_with_header_and_footer(state, capsys):
    chat._handle_slash("/! echo hello", state)
    out = capsys.readouterr().out
    (block,) = state.staged
    assert block.startswith("$ echo hello\nhello\n[exit 0 in ")
    assert block.endswith("s]")
    assert "staged (1 block)" in out


def test_shell_captures_stderr_and_exit_code(state):
    chat._handle_slash("/! echo oops >&2; exit 3", state)
    (block,) = state.staged
    assert "oops" in block and "[exit 3 in " in block


def test_shell_blocks_stack(state):
    chat._handle_slash("/! true", state)
    chat._handle_slash("/! false", state)
    assert len(state.staged) == 2


def test_shell_empty_command_usage(state, capsys):
    chat._handle_slash("/!", state)
    assert not state.staged
    assert "usage: /!" in capsys.readouterr().out


def test_drop_discards_staged(state, capsys):
    chat._handle_slash("/! true", state)
    chat._handle_slash("/drop", state)
    assert not state.staged
    assert "dropped 1 staged item" in capsys.readouterr().out


def test_truncate_middle_keeps_head_and_tail():
    text = "a" * 10_000 + "MID" + "b" * 10_000
    out = chat._truncate_middle(text, limit=1_000)
    assert out.startswith("a" * 500) and out.endswith("b" * 500)
    assert "chars truncated" in out and "MID" not in out


def test_truncate_middle_short_passthrough():
    assert chat._truncate_middle("short", limit=1_000) == "short"


def test_compose_prepends_staged_blocks(state):
    state.staged = ["$ ls\nfoo\n[exit 0 in 0.01s]"]
    content = chat._compose_user_content(state, "what is foo?")
    assert content.startswith("```\n$ ls\n")
    assert content.endswith("\n```\n\nwhat is foo?")
    assert not state.staged  # consumed


def test_compose_empty_line_sends_blocks_alone(state):
    state.staged = ["$ true\n[exit 0 in 0.00s]"]
    content = chat._compose_user_content(state, "")
    assert content == "```\n$ true\n[exit 0 in 0.00s]\n```"


def test_compose_without_staged_is_identity(state):
    assert chat._compose_user_content(state, "hi") == "hi"


# media staging + dropped-path detection
def test_dropped_media_detects_images_and_audio(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"x")
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"x")
    imgs, auds = chat._detect_dropped_media(f"{img} {wav}")
    assert imgs == [str(img)] and auds == [str(wav)]


def test_dropped_media_unescapes_dragged_path(tmp_path):
    img = tmp_path / "My Photo.png"
    img.write_bytes(b"x")
    dragged = str(img).replace(" ", "\\ ")  # what Terminal pastes on drop
    imgs, auds = chat._detect_dropped_media(dragged)
    assert imgs == [str(img)]


def test_dropped_media_rejects_prose_and_missing(tmp_path):
    assert chat._detect_dropped_media("tell me about entropy") is None
    assert chat._detect_dropped_media("/no/such/file.png") is None
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    assert chat._detect_dropped_media(str(txt)) is None  # not a media ext


def test_stage_media_requires_vlm(state, capsys):
    chat._stage_media(state, ["/tmp/x.png"], [])
    assert not state.staged_images
    assert "--mmproj" in capsys.readouterr().out


def test_image_command_stages(state, tmp_path, capsys):
    state.vlm = True
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    chat._handle_slash(f"/image {img}", state)
    assert state.staged_images == [str(img)]
    assert "staged 1 image" in capsys.readouterr().out


def test_image_command_missing_file(state, capsys):
    state.vlm = True
    chat._handle_slash("/image /no/such.png", state)
    assert not state.staged_images
    assert "no such file" in capsys.readouterr().out


def test_drop_clears_media_too(state, capsys):
    state.vlm = True
    state.staged = ["block"]
    state.staged_images = ["a.png", "b.png"]
    state.staged_audio = ["c.wav"]
    chat._handle_slash("/drop", state)
    assert "dropped 4 staged items" in capsys.readouterr().out
    assert not (state.staged or state.staged_images or state.staged_audio)


def test_esc_cancel_inert_without_tty():
    with chat._EscCancel() as esc:  # pytest stdin is not a tty
        assert esc.fd is None
        assert esc.pressed() is False


def test_stream_reply_collects_text_and_stats(state, capsys):
    from types import SimpleNamespace

    chunks = [
        SimpleNamespace(text=t, generation_tokens=i + 1, generation_tps=42.0)
        for i, t in enumerate(["Hel", "lo"])
    ]
    text, canceled = chat._stream_reply(iter(chunks), state)
    assert text == "Hello" and not canceled
    assert state.last_tps == 42.0
    assert state.last_stats["gen_tokens"] == 2
    assert state.last_stats["gen_tps"] == 42.0
    out = capsys.readouterr().out
    assert "Hello" in out
    # the stat line itself is printed by _end_turn, not the stream
    state.turn_checkpoint = {"user_content": "hi"}
    chat._end_turn(state, text, False)
    out = capsys.readouterr().out
    assert "gen 2 tok @ 42.0 tok/s" in out
    assert state.transcript[-1]["assistant"]["content"] == "Hello"


def test_stream_reply_cap_notice(state, capsys):
    from types import SimpleNamespace

    state.sampling["max_tokens"] = 2
    chunks = [
        SimpleNamespace(text=t, generation_tokens=i + 1, generation_tps=1.0)
        for i, t in enumerate(["a", "b"])
    ]
    text, canceled = chat._stream_reply(iter(chunks), state)
    assert text == "ab" and not canceled
    err = capsys.readouterr().err
    assert "max-tokens cap (2)" in err and "/max-tokens 0" in err
    # uncapped (0 = until the model stops): never a notice
    state.sampling["max_tokens"] = 0
    chunks = [SimpleNamespace(text="x", generation_tokens=1, generation_tps=1.0)]
    chat._stream_reply(iter(chunks), state)
    assert "max-tokens cap" not in capsys.readouterr().err


def test_stream_reply_keyboard_interrupt_cancels(state, capsys):
    def chunks():
        from types import SimpleNamespace

        yield SimpleNamespace(text="par", generation_tokens=1, generation_tps=1.0)
        raise KeyboardInterrupt

    text, canceled = chat._stream_reply(chunks(), state)
    assert text == "par" and canceled
    assert "reply canceled" in capsys.readouterr().out
    assert state.last_tps == 1.0    # last chunk's stats are kept for the ledger


def test_vlm_message_pins_markers_per_turn():
    pytest.importorskip("mlx_vlm")
    m_img = chat._vlm_message("gemma4", "what is this?", "user", n_images=1)
    m_txt = chat._vlm_message("gemma4", "and a follow-up", "user")
    assert m_img != m_txt
    assert "image" in str(m_img).lower()
    assert "image" not in str(m_txt).lower()
    unknown = chat._vlm_message("not-a-model", "hi", "user")
    assert unknown == {"role": "user", "content": "hi"}


# completion + history path
class _FakeReadline:
    def __init__(self, buffer=""):
        self._buffer = buffer

    def get_line_buffer(self):
        return self._buffer


def test_completer_matches_commands(state):
    rl = _FakeReadline("/te")
    complete = chat._make_completer(rl, state)
    assert complete("/te", 0) == "/temp "
    assert complete("/te", 1) is None


def test_completer_history_args(state):
    rl = _FakeReadline("/history o")
    complete = chat._make_completer(rl, state)
    assert {complete("o", i) for i in range(2)} == {"on", "off"}


def test_completer_inert_on_chat_text(state):
    rl = _FakeReadline("tell me about entropy")
    complete = chat._make_completer(rl, state)
    assert complete("entropy", 0) is None


def test_history_path_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert chat._history_path() == tmp_path / "gmlx" / "chat_history"


def test_completion_options_shared_logic(tmp_path):
    assert "/temp " in chat._completion_options("/te", "/te")
    assert chat._completion_options("/history o", "o") == ["on", "off"]
    assert chat._completion_options("hello there", "there") is None
    (tmp_path / "notes.txt").write_text("x")
    opts = chat._completion_options(
        "/! cat " + str(tmp_path / "no"), str(tmp_path / "no")
    )
    assert opts == [str(tmp_path / "notes.txt")]


# prompt_toolkit tier (skipped when the [chat] extra is not installed)
def test_ptk_wires_session(monkeypatch, tmp_path):
    pytest.importorskip("prompt_toolkit")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    state = chat.ChatState(history_enabled=True)
    assert chat._wire_ptk(state)
    assert state.ptk_session.completer is not None


def test_ptk_completer_matches_commands(monkeypatch, tmp_path):
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.document import Document

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    state = chat.ChatState(history_enabled=True, sampling={})
    assert chat._wire_ptk(state)
    comp = state.ptk_session.completer
    texts = [c.text for c in comp.get_completions(Document("/te"), None)]
    assert "/temp " in texts
    assert not list(comp.get_completions(Document("plain text"), None))


def test_ptk_history_respects_enabled_flag(monkeypatch, tmp_path):
    pytest.importorskip("prompt_toolkit")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    state = chat.ChatState(history_enabled=True)
    assert chat._wire_ptk(state)
    hist = state.ptk_session.history
    hist.append_string("kept line")
    f = tmp_path / "gmlx" / "chat_history.ptk"
    assert "kept line" in f.read_text()
    state.history_enabled = False  # /history off
    hist.append_string("dropped line")
    assert "dropped line" not in f.read_text()


def test_ptk_history_clear_unlinks_both(monkeypatch, tmp_path):
    pytest.importorskip("prompt_toolkit")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    state = chat.ChatState(history_enabled=True)
    assert chat._wire_ptk(state)
    state.ptk_session.history.append_string("x")
    chat._handle_slash("/history clear", state)
    assert not (tmp_path / "gmlx" / "chat_history.ptk").exists()


# parser + pre-load validation (no model load)
def test_chat_parser_defaults():
    args = chat._build_parser().parse_args(["m.gguf"])
    assert args.max_tokens == 0 and args.temp == 0.0
    assert args.seed is None and not args.no_history


def test_chat_missing_file_errors(capsys):
    assert chat.cmd_chat(["/no/such/model.gguf"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_chat_server_flags_require_assistant(capsys):
    # --base-url/--host/--port target a server; without --assistant chat loads
    # in-process, so accepting them silently loads a surprise second copy.
    with pytest.raises(SystemExit) as ei:
        chat.cmd_chat(["m.gguf", "--base-url", "http://127.0.0.1:8210/v1"])
    assert ei.value.code == 2
    assert "--assistant" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        chat.cmd_chat(["m.gguf", "--port", "8210"])


def test_run_missing_file_errors(capsys):
    assert cli.main(["/no/such/model.gguf"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_run_remote_ref_hints_pull(capsys):
    assert cli.main(["hf:org/repo/model.gguf"]) == 2
    assert "gmlx pull" in capsys.readouterr().err


def test_run_bad_template_config_fails_before_load(tmp_path, monkeypatch, capsys):
    fake = tmp_path / "m.gguf"
    fake.write_bytes(b"GGUF")
    from gmlx import loader

    def no_load(*a, **k):
        raise AssertionError("load_model must not run on a config typo")

    monkeypatch.setattr(loader, "load_model", no_load)
    rc = cli.main([str(fake), "--chat-template-config", "{bad json}"])
    assert rc == 1
    assert "not valid JSON" in capsys.readouterr().err


# new sampling/config surface: stop sequences, logit bias, resize shape
def test_stop_scanner_basic_and_trim():
    from gmlx.generation import StopScanner

    sc = StopScanner(["STOP"])
    out1, hit1 = sc.feed("hello ")
    out2, hit2 = sc.feed("world STOP more")
    assert not hit1 and hit2
    assert out1 + out2 == "hello world "  # trimmed at the match


def test_stop_scanner_split_across_segments():
    from gmlx.generation import StopScanner

    sc = StopScanner(["<|end|>"])
    text = ""
    for seg in ["abc<|", "en", "d|>xyz"]:
        out, hit = sc.feed(seg)
        text += out
        if hit:
            break
    assert hit and text == "abc"


def test_stop_scanner_no_match_flush():
    from gmlx.generation import StopScanner

    sc = StopScanner(["NEVER"])
    out, hit = sc.feed("short")
    assert not hit
    assert out + sc.flush() == "short"


def test_stop_scanner_earliest_of_multiple():
    from gmlx.generation import StopScanner

    sc = StopScanner(["BBB", "A"])
    out, hit = sc.feed("xyABBB")
    assert hit and out == "xy"  # "A" matches first


def test_stream_reply_stop_sequence_trims(state, capsys):
    from types import SimpleNamespace

    chunks = [
        SimpleNamespace(text=t, generation_tokens=i + 1, generation_tps=10.0)
        for i, t in enumerate(["The answer", " is END not this"])
    ]
    text, canceled = chat._stream_reply(iter(chunks), state, stops=["END"])
    assert text == "The answer is " and not canceled
    out = capsys.readouterr().out
    assert "not this" not in out
    assert state.last_stats["gen_tps"] == 10.0


def test_parse_logit_bias():
    assert chat.parse_logit_bias(None) is None
    assert chat.parse_logit_bias('{"128001": -100}') == {128001: -100.0}
    with pytest.raises(ValueError, match="not valid JSON"):
        chat.parse_logit_bias("{bad}")
    with pytest.raises(ValueError, match="token ids"):
        chat.parse_logit_bias('{"notanint": 1}')


def test_parse_resize_shape():
    assert chat.parse_resize_shape(None) is None
    assert chat.parse_resize_shape("448") == [448]
    assert chat.parse_resize_shape("672x448") == [672, 448]
    with pytest.raises(ValueError, match="N or WxH"):
        chat.parse_resize_shape("six-seventy-two")


def test_chat_parser_new_flags():
    args = chat._build_parser().parse_args(
        [
            "m.gguf",
            "--kv-bits",
            "4",
            "--stop",
            "END",
            "--stop",
            "###",
            "--xtc-probability",
            "0.5",
            "--repetition-context-size",
            "64",
            "--prefill-step-size",
            "512",
            "--resize-shape",
            "448",
        ]
    )
    assert args.kv_bits == 4 and args.kv_group_size == 64
    assert args.stop == ["END", "###"]
    assert args.xtc_probability == 0.5
    assert args.repetition_context_size == 64
    assert args.prefill_step_size == 512


def test_xtc_runtime_command(state, capsys):
    state.sampling["xtc_probability"] = 0.0
    chat._handle_slash("/xtc-probability 0.3", state)
    assert state.sampling["xtc_probability"] == 0.3


# Offload flags (same surface as `run`; pre-load validation only)
def test_chat_parser_accepts_offload_flags():
    ap = chat._build_parser()
    args = ap.parse_args(
        [
            "m.gguf",
            "--stream-cpu",
            "--moe-experts",
            "4",
        ]
    )
    assert args.stream_cpu and not args.stream_experts
    assert args.moe_experts == 4
    assert ap.parse_args(["m.gguf", "--stream-experts"]).stream_experts
    # --stream-cpu and --stream-experts are mutually exclusive
    import pytest

    with pytest.raises(SystemExit):
        ap.parse_args(["m.gguf", "--stream-cpu", "--stream-experts"])

    # adaptive fan-out flags: mass and probe are mutually exclusive
    args = ap.parse_args(
        ["m.gguf", "--stream-cpu", "--moe-experts", "6",
         "--moe-expert-mass", "0.9"]
    )
    assert args.moe_experts == 6 and args.moe_expert_mass == 0.9
    assert ap.parse_args(
        ["m.gguf", "--stream-cpu", "--moe-expert-probe"]).moe_expert_probe
    with pytest.raises(SystemExit):
        ap.parse_args(
            ["m.gguf", "--stream-cpu", "--moe-expert-mass", "0.9",
             "--moe-expert-probe"]
        )

    # an out-of-range P fails at parse time, before any model load
    for bad in ("1.5", "0", "-0.2", "nan"):
        with pytest.raises(SystemExit):
            ap.parse_args(["m.gguf", "--stream-cpu", "--moe-expert-mass", bad])


def test_chat_refuses_offload_flags_with_mmproj(tmp_path, capsys):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"")
    rc = chat.cmd_chat([str(gguf), "--mmproj", "p.gguf", "--stream-cpu"])
    assert rc == 2
    assert "text-only" in capsys.readouterr().err


# /render + /theme
def test_render_command_sets_mode(state, capsys):
    state.render = "lite"
    assert chat._handle_slash("/render", state) is None
    assert "render = lite" in capsys.readouterr().out
    chat._handle_slash("/render plain", state)
    assert state.render == "plain"
    chat._handle_slash("/render nonsense", state)
    assert state.render == "plain"


def test_render_command_rich_requires_rich(state, monkeypatch):
    from gmlx import render as rd

    monkeypatch.setattr(rd, "_rich_ok", False)
    state.render = "lite"
    chat._handle_slash("/render rich", state)
    assert state.render == "lite"
    monkeypatch.setattr(rd, "_rich_ok", True)
    chat._handle_slash("/render rich", state)
    assert state.render == "rich"


def test_theme_command_switches_and_validates(state, capsys):
    chat._handle_slash("/theme nord", state)
    assert state.theme.name == "nord"
    chat._handle_slash("/theme dark cb", state)
    assert state.theme.name == "dark" and state.theme.colorblind
    assert state.colorblind is True
    chat._handle_slash("/theme neon-zebra", state)
    assert state.theme.name == "dark"          # unchanged on bad name
    out = capsys.readouterr().out
    assert "unknown theme" in out


def test_theme_and_render_completions():
    assert "nord" in chat._completion_options("/theme n", "n")[0]
    assert chat._completion_options("/theme dark c", "c") == ["cb"]
    assert chat._completion_options("/render l", "l") == ["lite"]


def test_chat_parser_render_theme_flags():
    ap = chat._build_parser()
    ns = ap.parse_args(["m.gguf"])
    # theme defaults to None so the config's `theme:` can fill it (dark last).
    assert ns.render == "auto" and ns.theme is None and ns.colorblind is False
    ns = ap.parse_args(["m.gguf", "--render", "lite", "--theme", "nord", "--colorblind"])
    assert ns.render == "lite" and ns.theme == "nord" and ns.colorblind


# turn ledger + stat line v2 + info commands
class _FakeKV:
    def __init__(self, offset=0):
        self.offset = offset


def test_cache_tokens_reads_max_offset():
    assert chat._cache_tokens(None) == 0
    assert chat._cache_tokens([]) == 0
    assert chat._cache_tokens([_FakeKV(3), _FakeKV(7), _FakeKV(5)]) == 7


def test_begin_end_turn_ledger(state, capsys):
    state.session_stats = {
        "t0": 0.0, "turns": 0, "prompt_tok": 0, "gen_tok": 0,
        "gen_time": 0.0, "accepted": 0, "drafted": 0, "rounds": 0,
    }
    cache = [_FakeKV(100)]
    chat._begin_turn(state, cache=cache, first_turn=True, vlm_lens=(1, 0, 0))
    composed = chat._compose_user_content(state, "hello there")
    assert state.turn_checkpoint["user_content"] == composed
    state.last_stats = {
        "prompt_tokens": 50, "prompt_tps": 800.0,
        "gen_tokens": 20, "gen_tps": 40.0,
        "accepted": 15, "drafted": 20, "rounds": 8,
        "accept_rate": 0.75, "mean_accept_len": 1.9,
    }
    cache[0].offset = 170
    chat._end_turn(state, "hi!", False, cache=cache)
    entry = state.transcript[-1]
    assert entry["user"]["content"] == "hello there"
    assert entry["assistant"]["content"] == "hi!"
    assert entry["cache_before"] == 100
    assert entry["first_turn_before"] is True
    assert entry["vlm_lens"] == (1, 0, 0)
    assert state.ctx_used == 170
    ss = state.session_stats
    assert ss["turns"] == 1 and ss["prompt_tok"] == 50 and ss["gen_tok"] == 20
    out = capsys.readouterr().out
    assert "prompt 50 tok @ 800 tok/s" in out
    assert "gen 20 tok @ 40.0 tok/s" in out
    assert "accept 75% · 1.9/round" in out
    assert "ctx 170" in out


def test_end_turn_canceled_records_but_no_stat_line(state, capsys):
    state.turn_checkpoint = {"user_content": "q"}
    state.last_stats = {"gen_tokens": 5, "gen_tps": 10.0}
    chat._end_turn(state, "part", True)
    assert state.transcript[-1]["assistant"]["canceled"] is True
    assert "tok/s" not in capsys.readouterr().out


def test_stats_command(state, capsys):
    chat._handle_slash("/stats", state)
    assert "no completed turns" in capsys.readouterr().out
    state.session_stats = {
        "t0": 0.0, "turns": 3, "prompt_tok": 1500, "gen_tok": 900,
        "gen_time": 30.0, "accepted": 80, "drafted": 100, "rounds": 40,
    }
    chat._handle_slash("/stats", state)
    out = capsys.readouterr().out
    assert "3 turns" in out and "prompt 1.5k tok" in out
    assert "30.0 tok/s avg" in out and "accept 80%" in out


def test_model_command(state, capsys):
    chat._handle_slash("/model", state)
    assert "unavailable" in capsys.readouterr().out
    state.model_name = "qwen3.6-27b"
    state.ctx_max = 32768
    state.model_info = {
        "path": "/m/x.gguf", "arch": "qwen35moe", "model_type": "qwen3_5_moe",
        "n_params": 35_200_000_000, "n_tensors": 1136,
        "codecs": {"Q6_K": 80, "Q8_0": 302}, "size_bytes": 21_400_000_000,
        "n_shards": 1, "drafter": "native-head MTP (block 3)",
    }
    chat._handle_slash("/model", state)
    out = capsys.readouterr().out
    assert "qwen3.6-27b" in out and "35.2B params" in out
    assert "Q6_K x80" in out and "21.4 GB" in out
    assert "32.8k max" in out and "native-head MTP" in out

# -- B-inc-E: /exit /quit /reset /system /thinking-budget /copy ------------------


def test_exit_quit_reset_verbs(state, capsys):
    assert chat._handle_slash("/exit", state) == "exit"
    assert chat._handle_slash("/quit", state) == "exit"
    assert chat._handle_slash("/reset", state) == "reset"
    assert "conversation reset" in capsys.readouterr().out


def test_system_show_set_and_off(state, capsys):
    assert chat._handle_slash("/system", state) is None
    assert "no system prompt" in capsys.readouterr().out
    assert chat._handle_slash("/system You are terse.", state) == "reset"
    assert state.system_prompt == "You are terse."
    assert "set (conversation reset)" in capsys.readouterr().out
    chat._handle_slash("/system", state)
    assert "system prompt: You are terse." in capsys.readouterr().out
    assert chat._handle_slash("/system off", state) == "reset"
    assert state.system_prompt is None
    assert "cleared" in capsys.readouterr().out


def test_thinking_budget_command(state, capsys):
    chat._handle_slash("/thinking-budget", state)
    assert "thinking-budget = off" in capsys.readouterr().out
    assert chat._handle_slash("/thinking-budget 512", state) is None
    assert state.thinking_budget == 512
    assert "= 512 (next reply)" in capsys.readouterr().out
    chat._handle_slash("/thinking-budget off", state)
    assert state.thinking_budget is None
    chat._handle_slash("/thinking-budget lots", state)
    assert "needs an int" in capsys.readouterr().out
    assert state.thinking_budget is None


def test_strip_thinking():
    raw = "<think>step one\nstep two</think>The answer is 42."
    assert chat._strip_thinking(raw) == "The answer is 42."
    # template pre-opened the block: only the close marker is in the reply
    raw = "pondering...</think>\n\nUse a heap."
    assert chat._strip_thinking(raw, start_in_thinking=True) == "Use a heap."
    assert chat._strip_thinking("plain reply") == "plain reply"


def test_copy_command_strips_thinking(state, capsys):
    copied = []
    state.clipboard_runner = lambda argv, text: copied.append((argv[0], text))
    chat._handle_slash("/copy", state)
    assert "nothing to copy" in capsys.readouterr().out
    state.transcript = [{
        "assistant": {"content": "<think>hmm</think>Final.", "think_open": False},
    }]
    chat._handle_slash("/copy", state)
    assert "copied 6 chars" in capsys.readouterr().out
    assert copied and copied[0][1] == "Final."


def test_copy_runner_failure_falls_through(state, capsys, monkeypatch):
    def boom(argv, text):
        raise OSError("no clipboard")

    monkeypatch.setattr(chat.sys.stderr, "isatty", lambda: False, raising=False)
    state.clipboard_runner = boom
    state.transcript = [{"assistant": {"content": "hi", "think_open": False}}]
    chat._handle_slash("/copy", state)
    assert "no clipboard mechanism" in capsys.readouterr().out


def test_ptk_newline_binding_inserts_not_submits():
    pytest.importorskip("prompt_toolkit")
    kb = chat._ptk_key_bindings()

    class _Buf:
        def __init__(self):
            self.text = ""

        def insert_text(self, s):
            self.text += s

    class _Event:
        current_buffer = _Buf()

    (binding,) = kb.bindings
    assert [str(k) for k in binding.keys] == ["Keys.Escape", "Keys.ControlM"]
    binding.handler(_Event())
    assert _Event.current_buffer.text == "\n"


def test_new_commands_complete(state):
    opts = chat._completion_options("/e", "/e")
    assert "/exit " in opts
    opts = chat._completion_options("/", "/")
    for c in ("/quit ", "/reset ", "/system ", "/thinking-budget ", "/copy "):
        assert c in opts
    assert chat._completion_options("/thinking-budget o", "o") == ["off"]


def test_esc_typed_before_cbreak_entry_still_cancels(monkeypatch):
    # Regression: an Esc typed during the prompt->generation handoff sits in
    # the tty input queue; _EscCancel's cbreak entry must not flush it
    # (tty.setcbreak's default TCSAFLUSH discards pending input, so a reply
    # on a fast model streamed to completion uncancelable).
    import pty
    import threading

    from gmlx.chat import _EscCancel

    import select

    master, slave = pty.openpty()
    stop = threading.Event()

    def _drain_master():
        # A real terminal emulator always reads the master side; without a
        # reader the echoed byte blocks TCSADRAIN's output drain. Poll with a
        # stop flag: a blocking read would deadlock the close() below.
        try:
            while not stop.is_set():
                if select.select([master], [], [], 0.05)[0]:
                    os.read(master, 1024)
        except OSError:
            pass

    t = threading.Thread(target=_drain_master, daemon=True)
    t.start()
    try:
        os.write(master, b"\x1b")           # queued before cbreak entry
        stdin = os.fdopen(os.dup(slave), "rb", buffering=0)
        monkeypatch.setattr(sys, "stdin", stdin)
        with _EscCancel() as esc:
            deadline = time.time() + 2.0
            hit = False
            while time.time() < deadline and not hit:
                hit = esc.pressed()
            assert hit, "queued Esc was flushed by cbreak entry"
    finally:
        stop.set()
        t.join(timeout=2.0)
        os.close(master)
        os.close(slave)


def test_handoff_esc_drains_typeahead():
    # An Esc pressed between prompt submit and streaming ends up in
    # prompt_toolkit's typeahead store, not the tty queue; _handoff_esc must
    # treat it as a cancel request while putting typed-ahead text back.
    from types import SimpleNamespace

    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.input.typeahead import get_typeahead, store_typeahead
    from prompt_toolkit.key_binding.key_processor import KeyPress
    from prompt_toolkit.keys import Keys

    from gmlx.chat import _handoff_esc

    with create_pipe_input() as inp:
        state = chat.ChatState(ptk_session=SimpleNamespace(input=inp))
        assert _handoff_esc(state) is False           # nothing stored
        store_typeahead(inp, [KeyPress(Keys.Escape, "\x1b")])
        assert _handoff_esc(state) is True            # Esc -> cancel
        assert get_typeahead(inp) == []               # and consumed
        store_typeahead(inp, [KeyPress("h", "h"), KeyPress("i", "i")])
        assert _handoff_esc(state) is False           # text is not a cancel
        assert [k.key for k in get_typeahead(inp)] == ["h", "i"]
        # A lone \x1b read by pt's teardown is held in the vt100 parser as an
        # ambiguous sequence prefix, never reaching typeahead; the flush_keys
        # drain must still surface it as a cancel.
        inp.send_bytes(b"\x1b")
        inp.read_keys()                               # parser holds the Esc
        assert _handoff_esc(state) is True
    assert _handoff_esc(chat.ChatState()) is False    # readline editor: inert


def test_thinking_budget_zero_is_set_not_off(state, capsys):
    chat._handle_slash("/thinking-budget 0", state)
    assert state.thinking_budget == 0
    capsys.readouterr()
    chat._handle_slash("/thinking-budget", state)
    assert "thinking-budget = 0" in capsys.readouterr().out


def test_slash_sampling_marks_assistant_touched(state, capsys):
    """A /command that lands back on the session baseline must still ride
    along to the server (explicit override, not the server default)."""
    state.sampling["repetition_context_size"] = 20
    state.assistant_brain = object()
    state.assistant_extra = {}
    state.assistant_baseline = dict(state.sampling)
    state.assistant_touched = set()
    chat._handle_slash("/temp 1.0", state)
    chat._handle_slash("/temp 0", state)
    capsys.readouterr()
    chat._sync_assistant_extra(state)
    assert state.assistant_extra["temperature"] == 0.0


def test_xtc_rides_along_to_the_assistant_server(state, capsys):
    """XTC is a server-side sampler extra (server_patches._attach_xtc), so the
    flags and /commands forward instead of being silently dropped."""
    state.assistant_brain = object()
    state.assistant_extra = {}
    state.assistant_baseline = dict(state.sampling)
    state.assistant_touched = set()

    chat._sync_assistant_extra(state)
    assert "xtc_probability" not in state.assistant_extra   # untouched

    chat._handle_slash("/xtc-probability 0.5", state)
    capsys.readouterr()
    chat._sync_assistant_extra(state)
    assert state.assistant_extra["xtc_probability"] == 0.5
    assert "xtc_threshold" not in state.assistant_extra
