#!/usr/bin/env python3
"""Streaming reasoning/thinking segmentation + display (``gmlx.reasoning``).

CPU-only and model-free: drives the filter/printer with the raw transcripts the
common reasoning formats actually stream, including marker splits across feed
boundaries.
"""

from __future__ import annotations

import pytest

from gmlx import chat
from gmlx.reasoning import ReasoningFilter, ReasoningPrinter


def _segment(text: str, *, start: bool = False, chunk: int = 0):
    """Feed ``text`` through the filter (optionally a char at a time to exercise
    the partial-marker holdback) and return ``(reason, answer, spans)``."""
    f = ReasoningFilter(start_in_thinking=start)
    spans: list[tuple[str, str]] = []
    if chunk:
        for i in range(0, len(text), chunk):
            spans += f.feed(text[i : i + chunk])
    else:
        spans += f.feed(text)
    spans += f.flush()
    reason = "".join(t for t, m in spans if m == "reason")
    answer = "".join(t for t, m in spans if m == "answer")
    return reason, answer, spans


# Real-shape transcripts (markers as the detokenizer emits them).
_GPT_OSS = (
    '<|channel|>analysis<|message|>The user says "hi!" It is a greeting.'
    "<|end|><|start|>assistant<|channel|>final<|message|>Hello! How is it going?"
)
_GEMMA = "<|channel>thought\nThinking Process:\n1. analyze\n<channel|>Hi! How can I help?"
_QWEN = "Here's a thinking process:\n1. Analyze\n</think>\n\nHi! How can I help you today?"


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_gpt_oss_harmony(chunk):
    reason, answer, _ = _segment(_GPT_OSS, chunk=chunk)
    assert reason == 'The user says "hi!" It is a greeting.'
    assert answer == "Hello! How is it going?"
    # No control marker fragment survives in either stream.
    for frag in ("<|", "|>", "channel", "message", "analysis", "final"):
        assert frag not in reason + answer


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_gemma_channel(chunk):
    reason, answer, _ = _segment(_GEMMA, chunk=chunk)
    assert "Thinking Process:" in reason
    assert answer == "Hi! How can I help?"
    assert "channel" not in reason + answer


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_qwen_missing_open_tag(chunk):
    # Qwen pre-opens <think> in the prompt, so only the close is streamed -
    # the filter must be seeded with start_in_thinking=True.
    reason, answer, _ = _segment(_QWEN, start=True, chunk=chunk)
    assert reason.startswith("Here's a thinking process:")
    assert "</think>" not in reason + answer
    assert answer.strip() == "Hi! How can I help you today?"


_HY3 = "<think:opensource>Plan:\n1. greet\n</think:opensource>Hi there!"


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_hy3_suffixed_think_tags(chunk):
    reason, answer, _ = _segment(_HY3, chunk=chunk)
    assert reason == "Plan:\n1. greet\n"
    assert answer == "Hi there!"
    assert "opensource" not in reason + answer


@pytest.mark.parametrize("chunk", [0, 1, 5])
def test_hy3_preopened_close_only(chunk):
    # Hy3 pre-fills '<think:opensource>' in the prompt at reasoning_effort
    # low/high, so only the close tag streams - the filter is seeded.
    text = "Weighing options.\n</think:opensource>Done."
    reason, answer, _ = _segment(text, start=True, chunk=chunk)
    assert reason.startswith("Weighing options.")
    assert answer == "Done."
    assert "opensource" not in reason + answer


_KIMI_K3 = ("User greets; respond warmly.<|close|>think<|sep|>"
            "<|open|>response<|sep|>Hi! How can I help?"
            "<|close|>response<|sep|><|close|>message<|sep|>")


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_kimi_k3_xtml_sections(chunk):
    # K3's prompt pre-opens the think section, so generation streams the
    # close, the response section, and the message close (the <|end_of_msg|>
    # turn terminator is the EOS token and never reaches the filter).
    reason, answer, _ = _segment(_KIMI_K3, start=True, chunk=chunk)
    assert reason == "User greets; respond warmly."
    assert answer == "Hi! How can I help?"
    for frag in ("<|", "|>", "think", "response", "message"):
        assert frag not in reason + answer


@pytest.mark.parametrize("chunk", [0, 3])
def test_kimi_k3_self_opened_think(chunk):
    # Defensive: if the model re-emits the opener itself (unseeded filter),
    # the sections still segment correctly.
    text = ("<|open|>think<|sep|>Plan.<|close|>think<|sep|>"
            "<|open|>response<|sep|>Done.")
    reason, answer, _ = _segment(text, chunk=chunk)
    assert reason == "Plan."
    assert answer == "Done."


def test_qwen_unseeded_still_strips_close():
    # Without the seed the leading text is mistagged as answer, but the close
    # marker is still stripped (no </think> leaks) and the answer is recognised.
    reason, answer, _ = _segment(_QWEN, start=False)
    assert "</think>" not in reason + answer
    assert "Hi! How can I help you today?" in answer


def test_plain_answer_is_all_answer():
    reason, answer, spans = _segment("Just a normal answer, no thinking here.")
    assert reason == ""
    assert answer == "Just a normal answer, no thinking here."
    assert all(m == "answer" for _, m in spans)


def test_bare_channel_waits_for_disambiguation():
    # A lone "<|channel|>" must not be dropped early - it could still become
    # "<|channel|>analysis<|message|>". Held back until the next feed resolves it.
    f = ReasoningFilter()
    assert f.feed("<|channel|>") == []  # nothing emitted yet
    spans = f.feed("final<|message|>done")
    assert spans == [("done", "answer")]


def _render(spans, *, display="show", color=False):
    out: list[str] = []
    p = ReasoningPrinter(display=display, color=color, write=out.append)
    p.feed_spans(spans)
    p.close()
    return "".join(out)


class _Clock:
    """Deterministic monotonic clock: advances a fixed step per call."""

    def __init__(self, step: float = 0.1):
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _stream(printer, text, *, start=False, chunk=6, toggle_at=None, toggle_to="hide"):
    """Drive the printer token-by-token (feed + tick), as the REPL does. With
    ``toggle_at`` set, call ``set_display(toggle_to)`` after that chunk index."""
    f = ReasoningFilter(start_in_thinking=start)
    for n, i in enumerate(range(0, len(text), chunk)):
        printer.feed_spans(f.feed(text[i : i + chunk]))
        printer.tick()
        if n == toggle_at:
            printer.set_display(toggle_to)
    printer.feed_spans(f.flush())
    printer.close()


def test_printer_show_frames_and_pays_off():
    out: list[str] = []
    p = ReasoningPrinter(display="show", color=False, write=out.append, clock=_Clock())
    _stream(p, _GPT_OSS)
    rendered = "".join(out)
    assert "┌ thinking" in rendered            # opens the block
    assert "│ The user says" in rendered        # gutter-framed body
    assert "└ thought for 0.1s · " in rendered  # payoff footer (elapsed + tokens)
    assert "tok" in rendered
    assert rendered.rstrip().endswith("Hello! How is it going?")
    assert "<|" not in rendered                 # markers gone


def test_printer_hide_collapses_to_payoff():
    out: list[str] = []
    p = ReasoningPrinter(display="hide", color=False, write=out.append, clock=_Clock())
    _stream(p, _GPT_OSS)
    rendered = "".join(out)
    assert "The user says" not in rendered       # body suppressed
    assert "thought for 0.1s · " in rendered      # payoff still shown
    assert rendered.rstrip().endswith("Hello! How is it going?")


def test_printer_hide_color_animates_spinner():
    out: list[str] = []
    p = ReasoningPrinter(display="hide", color=True, write=out.append, clock=_Clock())
    _stream(p, _GPT_OSS)
    rendered = "".join(out)
    assert "thinking…" in rendered               # live spinner label
    assert any(f in rendered for f in "⠋⠙⠹⠸")     # at least one spinner frame
    assert "✓ thought for" in rendered            # resolves on the answer
    assert "\r" in rendered                       # repainted in place


def test_printer_token_count_matches_reason_steps():
    out: list[str] = []
    p = ReasoningPrinter(display="show", color=False, write=out.append, clock=_Clock())
    # One reason token per tick; 4 chunks of pure reasoning then the close.
    _stream(p, "<think>aaa bbb ccc</think>answer", chunk=4)
    rendered = "".join(out)
    # tokens are counted only while reasoning - a positive, bounded count.
    import re

    m = re.search(r"· (\d+) tok", rendered)
    assert m and int(m.group(1)) >= 1


def test_printer_raw_is_verbatim():
    out: list[str] = []
    p = ReasoningPrinter(display="raw", color=True, write=out.append, clock=_Clock())
    p.feed_spans([(_GPT_OSS, "answer")])
    p.tick()
    p.close()
    assert "".join(out) == _GPT_OSS


def test_printer_set_display_collapses_live():
    out: list[str] = []
    p = ReasoningPrinter(display="show", color=True, write=out.append, clock=_Clock())
    _stream(p, _GPT_OSS, chunk=8, toggle_at=5)  # Ctrl-O once the body is streaming
    rendered = "".join(out)
    assert "┌ thinking" in rendered      # started expanded
    assert "thinking…" in rendered       # then a live spinner took over
    assert "✓ thought for" in rendered
    assert rendered.rstrip().endswith("Hello! How is it going?")


def test_printer_color_answer_not_dimmed():
    out: list[str] = []
    p = ReasoningPrinter(display="show", color=True, write=out.append, clock=_Clock())
    _stream(p, _GPT_OSS)
    rendered = "".join(out)
    assert "\x1b[2m" in rendered and "\x1b[0m" in rendered
    assert rendered.rstrip().endswith("Hello! How is it going?")  # answer after reset


def test_reasoning_slash_command():
    state = chat.ChatState(history_enabled=True)
    assert chat._handle_slash("/reasoning", state) is None
    assert chat._handle_slash("/reasoning hide", state) is None
    assert state.reasoning == "hide"
    # An invalid mode is rejected without changing state.
    chat._handle_slash("/reasoning bogus", state)
    assert state.reasoning == "hide"


def test_reasoning_in_command_completion():
    assert "/reasoning" in chat._ALL_COMMANDS
    assert set(chat._completion_options("/reasoning ", "")) == {"show", "hide", "raw"}


# answer_sink routing (the markdown renderer seam)
def test_answer_sink_receives_answer_spans_only():
    out, sunk = [], []
    p = ReasoningPrinter(
        display="show", color=False, write=out.append, answer_sink=sunk.append
    )
    p.feed_spans([("pondering", "reason"), ("the answer", "answer")])
    p.close()
    assert "".join(sunk) == "the answer"
    joined = "".join(out)
    assert "pondering" in joined          # reasoning stays on the writer
    assert "the answer" not in joined     # answer went to the sink


def test_answer_sink_sees_lstripped_answer():
    sunk = []
    p = ReasoningPrinter(
        display="show", color=False, write=lambda s: None, answer_sink=sunk.append
    )
    p.feed_spans([("think", "reason"), ("\n\nanswer", "answer")])
    assert "".join(sunk) == "answer"      # payoff lstrip runs before the sink


def test_raw_display_bypasses_answer_sink():
    out, sunk = [], []
    p = ReasoningPrinter(
        display="raw", color=False, write=out.append, answer_sink=sunk.append
    )
    p.feed_spans([("<think>x</think>ans", "answer")])
    p.close()
    assert sunk == []                     # raw = verbatim to the writer
    assert "".join(out) == "<think>x</think>ans"


def test_themed_printer_uses_theme_slot():
    from gmlx.theme import resolve_theme

    out = []
    t = resolve_theme("nord", depth=1 << 24)
    p = ReasoningPrinter(display="show", color=True, write=out.append, theme=t)
    p.feed_spans([("mull", "reason")])
    p.close()
    joined = "".join(out)
    assert t.thinking in joined           # theme slot, not the bare \x1b[2m dim


# StreamRenderer: the run CLI's (write, close) glue over filter + printer.
def test_stream_renderer_show_styles_and_strips(capsys):
    from gmlx.reasoning import StreamRenderer

    r = StreamRenderer("show", color=False)
    text = "<think>" + _QWEN
    for i in range(0, len(text), 7):
        r.write(text[i : i + 7])
    r.close()
    out = capsys.readouterr().out
    assert "┌ thinking" in out and "└ thought for" in out
    assert "</think>" not in out
    assert out.rstrip().endswith("Hi! How can I help you today?")


def test_stream_renderer_prefill_close_only(capsys):
    from gmlx.reasoning import StreamRenderer

    # Pre-fill template: the prompt opened the block, the stream only closes it.
    r = StreamRenderer("hide", start_in_thinking=True, color=False)
    for chunk in ("mulling", "</think>", "Answer."):
        r.write(chunk)
    r.close()
    out = capsys.readouterr().out
    assert "mulling" not in out and "</think>" not in out
    assert "thought for" in out and "Answer." in out


# -- provider-doc thinking switch spellings ------------------------------------

def test_thinking_flag_zai_shapes():
    from gmlx.reasoning import thinking_flag
    assert thinking_flag({"type": "disabled"}) is False
    assert thinking_flag({"type": "enabled"}) is True
    assert thinking_flag({"type": "auto"}) is None       # unknown -> untouched
    assert thinking_flag({"type": "disabled", "x": 1}) is None
    assert thinking_flag("disabled") is None
    assert thinking_flag(None) is None


def test_normalize_template_kwargs_translates_thinking():
    from gmlx.reasoning import normalize_template_kwargs
    assert normalize_template_kwargs({"thinking": {"type": "disabled"}}) == \
        {"enable_thinking": False}
    # An explicit enable_thinking wins over the translated spelling.
    assert normalize_template_kwargs(
        {"thinking": {"type": "disabled"}, "enable_thinking": True}) == \
        {"enable_thinking": True}
    # Non-API values stay as plain template variables.
    assert normalize_template_kwargs({"thinking": "deep"}) == {"thinking": "deep"}


def test_parse_template_config_translates_thinking():
    from gmlx.chat import parse_template_config
    assert parse_template_config('{"thinking": {"type": "disabled"}}') == \
        {"enable_thinking": False}


def test_fold_thinking_flag(monkeypatch):
    from types import SimpleNamespace
    import gmlx.chat as chat

    monkeypatch.setattr(chat, "_template_text",
                        lambda a: "{% if enable_thinking %}...{% endif %}")
    args = SimpleNamespace(thinking=None, reasoning_effort=None, gguf="/m/x.gguf")
    assert chat.fold_thinking_flag(args, {"a": 1}) == {"a": 1}
    args.thinking = "off"
    assert chat.fold_thinking_flag(args, {}) == {"enable_thinking": False}
    # The dedicated flag wins over a --chat-template-config value.
    assert chat.fold_thinking_flag(args, {"enable_thinking": True}) == \
        {"enable_thinking": False}
    args.thinking = "on"
    assert chat.fold_thinking_flag(args, {}) == {"enable_thinking": True}


def test_fold_thinking_flag_hy3_dialect(monkeypatch):
    """Hy3 has no enable_thinking; its template grades reasoning_effort with
    a no_think level - --thinking maps onto that spelling."""
    from types import SimpleNamespace
    import gmlx.chat as chat

    monkeypatch.setattr(chat, "_template_text",
                        lambda a: "reasoning_effort in ['low','high','no_think']")
    args = SimpleNamespace(thinking="off", reasoning_effort=None, gguf="/m/x.gguf")
    assert chat.fold_thinking_flag(args, {}) == {"reasoning_effort": "no_think"}
    args.thinking = "on"
    assert chat.fold_thinking_flag(args, {}) == {"reasoning_effort": "high"}
    # An explicit level wins over the mapped one.
    args.reasoning_effort = "low"
    assert chat.fold_thinking_flag(args, {}) == {"reasoning_effort": "low"}


def test_fold_thinking_flag_gpt_oss_warns(monkeypatch, capsys):
    """gpt-oss grades reasoning_effort but cannot disable reasoning."""
    from types import SimpleNamespace
    import gmlx.chat as chat

    monkeypatch.setattr(chat, "_template_text",
                        lambda a: 'set reasoning_effort = "medium"')
    args = SimpleNamespace(thinking="off", reasoning_effort=None, gguf="/m/x.gguf")
    assert chat.fold_thinking_flag(args, {}) == {}   # no kwarg forced
    assert "reasoning_effort" in capsys.readouterr().err


def test_fold_reasoning_effort_passthrough_and_noop_warning(monkeypatch, capsys):
    from types import SimpleNamespace
    import gmlx.chat as chat

    monkeypatch.setattr(chat, "_template_text",
                        lambda a: 'set reasoning_effort = "medium"')
    args = SimpleNamespace(thinking=None, reasoning_effort="high", gguf="/m/x.gguf")
    assert chat.fold_thinking_flag(args, {}) == {"reasoning_effort": "high"}
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(chat, "_template_text", lambda a: "enable_thinking only")
    assert chat.fold_thinking_flag(args, {}) == {"reasoning_effort": "high"}
    assert "no-op" in capsys.readouterr().err


def test_fold_thinking_flag_minimax_thinking_mode(monkeypatch):
    """MiniMax-M3: three-state thinking_mode (enabled/disabled/adaptive)."""
    from types import SimpleNamespace
    import gmlx.chat as chat

    monkeypatch.setattr(chat, "_template_text",
                        lambda a: 'thinking_mode == "adaptive"')
    args = SimpleNamespace(thinking="off", reasoning_effort=None, gguf="/m/x.gguf")
    assert chat.fold_thinking_flag(args, {}) == {"thinking_mode": "disabled"}
    args.thinking = "on"
    assert chat.fold_thinking_flag(args, {}) == {"thinking_mode": "enabled"}
    args.thinking = "adaptive"
    assert chat.fold_thinking_flag(args, {}) == {"thinking_mode": "adaptive"}


def test_fold_thinking_adaptive_elsewhere_warns(monkeypatch, capsys):
    from types import SimpleNamespace
    import gmlx.chat as chat

    monkeypatch.setattr(chat, "_template_text",
                        lambda a: "{% if enable_thinking %}...{% endif %}")
    args = SimpleNamespace(thinking="adaptive", reasoning_effort=None,
                           gguf="/m/x.gguf")
    assert chat.fold_thinking_flag(args, {}) == {}
    assert "adaptive" in capsys.readouterr().err


def test_map_thinking_controls_base_is_verbatim():
    """An explicitly passed chat_template_kwargs dict is never reinterpreted:
    keys named like the controls pass through untouched; only the dedicated
    control arguments are mapped."""
    from gmlx.reasoning import map_thinking_controls

    base = {"thinking": "deep", "reasoning_effort": "medium", "x": 1}
    assert map_thinking_controls(base, template="enable_thinking") == base
    out = map_thinking_controls(base, thinking="off",
                                template="{% if enable_thinking %}{% endif %}")
    assert out == {**base, "enable_thinking": False}


def test_map_thinking_controls_value_spellings(capsys):
    """Bools, enabled/disabled, and the z.ai dict shape all normalize; an
    unrecognized value warns and is dropped."""
    from gmlx.reasoning import map_thinking_controls

    tmpl = "{% if enable_thinking %}{% endif %}"
    warns = []
    for v in (False, "disabled", {"type": "disabled"}):
        assert map_thinking_controls({}, thinking=v, template=tmpl) == \
            {"enable_thinking": False}
    assert map_thinking_controls({}, thinking="sideways", template=tmpl,
                                 warn=warns.append) == {}
    assert warns and "unrecognized" in warns[0]


@pytest.mark.parametrize("chunk", [0, 1, 3, 7])
def test_harmony_annotated_commentary_tool_header(chunk):
    # Tool preludes annotate the header; everything up to <|message|> is
    # header, the payload routes to reasoning, nothing reaches the answer.
    text = ('<|channel|>commentary to=functions.get_weather '
            '<|constrain|>json<|message|>{"location": "SF"}<|call|>')
    reason, answer, _ = _segment(text, chunk=chunk)
    assert reason == '{"location": "SF"}'
    assert answer == ""
    assert "functions" not in reason + answer


@pytest.mark.parametrize("chunk", [0, 1, 3, 7])
def test_harmony_annotated_final_header(chunk):
    # Structured output constrains the final channel; the annotation is
    # header, the payload is the answer.
    text = '<|channel|>final <|constrain|>json<|message|>{"answer": 4}'
    reason, answer, _ = _segment(text, chunk=chunk)
    assert answer == '{"answer": 4}'
    assert reason == ""


def test_harmony_unknown_channel_routes_to_reasoning():
    reason, answer, _ = _segment(
        "<|channel|>critique<|message|>too wordy<|end|>")
    assert reason == "too wordy"
    assert answer == ""


def test_harmony_capped_mid_header_drops_header_remnant():
    reason, answer, _ = _segment("<|channel|>commentary to=functi")
    assert reason == ""
    assert answer == ""


def test_swallow_budget_bounds_literal_channel_mention():
    # A non-harmony reply quoting the literal marker must not lose the rest
    # of the message: the swallow gives up past its budget.
    tail = "x" * 300
    reason, answer, _ = _segment("see <|channel|>" + tail)
    assert answer.startswith("see ")
    assert tail[-40:] in reason + answer


def test_split_harmony_reply_shapes():
    from gmlx.reasoning import split_harmony_reply
    full = ('<|channel|>analysis<|message|>Count the entries.'
            "<|end|><|start|>assistant<|channel|>final<|message|>Six.")
    assert split_harmony_reply(full) == ("Count the entries.", "Six.")
    capped = "<|channel|>analysis<|message|>Count the"
    assert split_harmony_reply(capped) == ("Count the", "")
    plain = "Six."
    assert split_harmony_reply(plain) == (None, "Six.")


def test_split_harmony_reply_keeps_first_line_code_indent():
    from gmlx.reasoning import split_harmony_reply
    full = ('<|channel|>analysis<|message|>Recall the body.'
            "<|end|><|start|>assistant<|channel|>final<|message|>"
            '\n    """\n    Docstring.\n')
    r, c = split_harmony_reply(full)
    assert r == "Recall the body."
    assert c == '    """\n    Docstring.'
    # whitespace-only content collapses to empty
    ws = ('<|channel|>analysis<|message|>x'
          "<|end|><|start|>assistant<|channel|>final<|message|>\n  \n")
    assert split_harmony_reply(ws) == ("x", "")


# --- muse-glimmer ATEM channel ------------------------------------------------
#
# Routing is on the message HEADER, not on a "to=self" marker: any marker whose
# text can occur in prose would misclassify other models, and one starting with
# a space or letter would make _partial_tail_start hold that character back at
# every chunk boundary. The header is swallowed whole and only an explicit
# "to=self" recipient routes to reasoning.

_MUSE = (
    "<|start|>assistant to=self<|message|>Work out the capital.<|eom|>"
    "<|start|>assistant<|message|>The capital of France is Paris.<|eot|>"
)
# The generation prompt already ends with "<|start|>assistant", so the FIRST
# generated turn streams the header tail only - hence the start_in_header seed.
_MUSE_FIRST = (
    " to=self<|message|>Work out the capital.<|eom|>"
    "<|start|>assistant to=user<|message|>The capital of France is Paris.<|eot|>"
)
_MUSE_TOOL = (
    "<|start|>assistant to=self<|message|>Need the weather.<|eom|>"
    "<|start|>assistant to=weather.get<|message|>"
    '<atem:function_calls><atem:invoke name="get_weather">'
    '<atem:parameter name="city">Paris</atem:parameter>'
    "</atem:invoke></atem:function_calls><|eot|>"
)


def _segment_header(text: str, *, chunk: int = 0):
    f = ReasoningFilter(start_in_header=True)
    spans: list[tuple[str, str]] = []
    if chunk:
        for i in range(0, len(text), chunk):
            spans += f.feed(text[i : i + chunk])
    else:
        spans += f.feed(text)
    spans += f.flush()
    return ("".join(t for t, m in spans if m == "reason"),
            "".join(t for t, m in spans if m == "answer"))


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_muse_glimmer_self_channel_then_answer(chunk):
    reason, answer, _ = _segment(_MUSE, chunk=chunk)
    assert reason == "Work out the capital."
    assert answer == "The capital of France is Paris."
    for frag in ("<|", "|>", "to=self", "message", "eom", "eot"):
        assert frag not in reason + answer


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_muse_glimmer_first_turn_needs_the_header_seed(chunk):
    reason, answer = _segment_header(_MUSE_FIRST, chunk=chunk)
    assert reason == "Work out the capital."
    assert answer == "The capital of France is Paris."
    # " to=user" is header too - it must be swallowed, not leaked as answer text
    assert "to=user" not in reason + answer
    assert not answer.startswith(" to=")


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_muse_glimmer_tool_recipient_is_swallowed_and_call_is_answer(chunk):
    reason, answer, _ = _segment(_MUSE_TOOL, chunk=chunk)
    assert reason == "Need the weather."
    assert "to=weather.get" not in reason + answer
    # the tool block itself is answer-side, so the tool parser still sees it
    assert answer.startswith("<atem:function_calls>")
    assert 'name="get_weather"' in answer


@pytest.mark.parametrize("chunk", [0, 1, 3, 7, 13])
def test_muse_glimmer_eom_returns_to_answer_without_a_self_header(chunk):
    """A non-final message that is not addressed to self stays answer-side."""
    text = ("<|start|>assistant to=self<|message|>think<|eom|>"
            "<|start|>assistant<|message|>first<|eom|>"
            "<|start|>assistant<|message|>second<|eot|>")
    reason, answer, _ = _segment(text, chunk=chunk)
    assert reason == "think"
    assert answer == "firstsecond"
