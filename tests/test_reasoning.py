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
