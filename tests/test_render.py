#!/usr/bin/env python3
"""`gmlx.render`: block segmentation, lite styling/wrapping, and the
in-place streaming repaint machinery. CPU-only - injected write/size/clock."""
from __future__ import annotations

import re

import pytest

from gmlx import render as rd
from gmlx.theme import resolve_theme

_THEME = resolve_theme("dark", depth=1 << 24)
_PLAIN = resolve_theme("dark", color=False)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip(s: str) -> str:
    return _ANSI_RE.sub("", s)


# -- BlockBuffer ---------------------------------------------------------------


def _feed_all(buf: rd.BlockBuffer, text: str, chunk: int) -> list[str]:
    done = []
    for i in range(0, len(text), chunk):
        done.extend(buf.feed(text[i : i + chunk]))
    return done


@pytest.mark.parametrize("chunk", [1, 3, 7, 1000])
def test_blocks_paragraphs_split_on_blank(chunk):
    buf = rd.BlockBuffer()
    done = _feed_all(buf, "para one\nstill one\n\npara two\n\n", chunk)
    assert done == ["para one\nstill one\n", "para two\n"]
    assert buf.flush() == ""


@pytest.mark.parametrize("chunk", [1, 5, 1000])
def test_blocks_fence_holds_blank_lines(chunk):
    src = "```python\na = 1\n\nb = 2\n```\nafter\n\n"
    buf = rd.BlockBuffer()
    done = _feed_all(buf, src, chunk)
    assert done == ["```python\na = 1\n\nb = 2\n```\n", "after\n"]


def test_blocks_tilde_fence_and_longer_close():
    buf = rd.BlockBuffer()
    done = _feed_all(buf, "~~~\nx\n~~~\n", 1)
    assert done == ["~~~\nx\n~~~\n"]
    buf = rd.BlockBuffer()
    done = _feed_all(buf, "```\nx\n````\n", 1)
    assert done == ["```\nx\n````\n"]


def test_blocks_heading_and_hr_complete_immediately():
    buf = rd.BlockBuffer()
    done = buf.feed("## Title\ntext\n---\n")
    assert done[0] == "## Title\n"
    assert done[1] == "text\n"          # hr interrupts the paragraph
    assert done[2] == "---\n"


def test_blocks_fence_interrupts_paragraph():
    buf = rd.BlockBuffer()
    done = buf.feed("intro\n```\ncode\n```\n")
    assert done == ["intro\n", "```\ncode\n```\n"]


def test_blocks_current_includes_partial_line():
    buf = rd.BlockBuffer()
    buf.feed("hello wo")
    assert buf.current == "hello wo"
    buf.feed("rld\nnext ")
    assert buf.current == "hello world\nnext "


def test_blocks_flush_returns_unterminated_fence():
    buf = rd.BlockBuffer()
    buf.feed("```py\nx = 1\n")
    assert buf.flush() == "```py\nx = 1\n"


def test_blocks_list_run_stays_one_block():
    buf = rd.BlockBuffer()
    done = buf.feed("- a\n- b\n- c\n\n")
    assert done == ["- a\n- b\n- c\n"]


# -- lite renderer ---------------------------------------------------------------


def test_lite_inline_styles():
    lines = rd._render_lite("**bold** and `code` and *ital*\n", 60, _THEME)
    joined = "\n".join(lines)
    assert _THEME.bold + "bold" in joined
    assert _THEME.inline_code + "code" in joined
    assert _THEME.italic + "ital" in joined
    assert _strip(joined) == "bold and code and ital"


def test_lite_heading_underline():
    lines = rd._render_lite("# Big Title\n", 40, _THEME)
    assert _strip(lines[0]) == "Big Title"
    assert set(_strip(lines[1])) == {"─"}


def test_lite_bullets_and_ordered():
    lines = rd._render_lite("- alpha\n- beta\n", 40, _THEME)
    assert _strip(lines[0]) == "• alpha"
    lines = rd._render_lite("1. one\n2. two\n", 40, _THEME)
    assert _strip(lines[0]) == "1. one"


def test_lite_blockquote_gutter():
    lines = rd._render_lite("> quoted text\n", 40, _THEME)
    assert _strip(lines[0]) == "│ quoted text"


def test_lite_fence_box_drops_backticks():
    lines = rd._render_lite("```python\nx = 1\n```\n", 30, _THEME)
    stripped = [_strip(ln) for ln in lines]
    assert stripped[0].startswith("┌─ python ")
    assert stripped[1] == "│ x = 1"
    assert stripped[-1].startswith("└")
    assert "```" not in "\n".join(stripped)


def test_lite_wrap_respects_width():
    text = " ".join(["word"] * 30) + "\n"
    for width in (20, 40, 79):
        lines = rd._render_lite(text, width, _THEME)
        assert len(lines) > 1
        assert all(rd._visible_width(_strip(ln)) <= width for ln in lines)


def test_lite_wrap_counts_wide_chars():
    lines = rd._render_lite("宽" * 30 + "\n", 20, _PLAIN)
    assert all(rd._visible_width(ln) <= 20 for ln in lines)


def test_lite_plain_theme_has_no_escapes():
    out = "\n".join(rd._render_lite("**b** `c` # x\n", 40, _PLAIN))
    assert "\x1b" not in out


# -- resolve_render_mode -----------------------------------------------------------


def test_mode_auto_ladder(monkeypatch):
    monkeypatch.setattr(rd, "_rich_ok", True)
    assert rd.resolve_render_mode("auto", tty=True, color=True) == ("rich", None)
    monkeypatch.setattr(rd, "_rich_ok", False)
    assert rd.resolve_render_mode(None, tty=True, color=True) == ("lite", None)
    assert rd.resolve_render_mode(None, tty=False, color=True) == ("plain", None)


def test_mode_explicit_downgrades_note(monkeypatch):
    monkeypatch.setattr(rd, "_rich_ok", False)
    mode, note = rd.resolve_render_mode("rich", tty=True, color=True)
    assert mode == "lite" and "rich" in note
    mode, note = rd.resolve_render_mode("lite", tty=False, color=True)
    assert mode == "plain" and note


# -- StreamRenderer ------------------------------------------------------------------


class _Size:
    def __init__(self, columns=41, lines=24):
        self.columns = columns
        self.lines = lines


class _Term:
    """Captures writes; injectable clock and size."""

    def __init__(self, columns=41, lines=24):
        self.out: list[str] = []
        self.now = 0.0
        self.size = _Size(columns, lines)

    def write(self, s):
        self.out.append(s)

    def clock(self):
        self.now += 1.0  # every call advances past the throttle window
        return self.now

    def text(self):
        return "".join(self.out)

    def renderer(self, mode="lite"):
        return rd.StreamRenderer(
            mode,
            _THEME if mode != "plain" else _PLAIN,
            write=self.write,
            size_fn=lambda: self.size,
            clock=self.clock,
        )


def test_stream_plain_passthrough():
    t = _Term()
    r = t.renderer("plain")
    r.feed("raw **text**")
    r.finalize()
    assert t.text() == "raw **text**"


def test_stream_single_paragraph_finalize():
    t = _Term()
    r = t.renderer()
    r.feed("hello ")
    r.feed("world")
    r.finalize()
    out = t.text()
    assert "hello world" in _strip(out)
    assert out.endswith("\n")


def test_stream_repaint_moves_up_and_clears():
    t = _Term()
    r = t.renderer()
    r.feed("one two\n")     # paints the live block (1 line)
    r.feed("three\n")       # repaint: must move up over the painted line
    r.finalize()
    out = t.text()
    assert "\x1b[1A" in out
    assert "\x1b[2K" in out
    final = _strip(out).rstrip("\n").split("\n")
    assert final[-1] == "one two three"  # wrapped as one paragraph at width 40


def test_stream_completed_block_then_next():
    t = _Term()
    r = t.renderer()
    r.feed("first para\n\nsecond para")
    r.finalize()
    stripped = _strip(t.text())
    assert "first para\n" in stripped
    assert "second para" in stripped
    first_end = stripped.index("first para") + len("first para")
    assert "\n\n" in stripped[first_end : stripped.index("second para")]


def test_stream_shrinking_render_clears_leftovers():
    t = _Term()
    r = t.renderer()
    r.feed("- a\n- b\n- c\n")       # 3 painted lines
    r.feed("\n")                     # completes; final render is the same 3
    r.feed("tiny")
    r.finalize()
    assert "tiny" in _strip(t.text())


def test_stream_oversize_block_freezes_to_raw():
    t = _Term(columns=41, lines=8)   # threshold: >= 6 rendered lines freezes
    r = t.renderer()
    r.feed("```\n")
    for i in range(10):
        r.feed(f"line {i}\n")
    r.feed("```\n")
    r.feed("after\n")
    r.finalize()
    out = t.text()
    stripped = _strip(out)
    assert "line 9" in stripped                  # everything still reached the screen
    assert "after" in stripped
    # once frozen, appends inside the block are raw: no escapes between the
    # late lines (the next repaint only comes with the "after" block)
    frozen_part = out.split("line 7", 1)[1].split("line 9", 1)[0]
    assert "\x1b[" not in frozen_part


def test_stream_resize_freezes_current_block():
    t = _Term(columns=60)
    r = t.renderer()
    r.feed("some words here\n")
    t.size = _Size(columns=30)       # window narrowed mid-block
    r.feed("more words\n")
    r.finalize()
    assert "more words" in _strip(t.text())


def test_stream_finalize_paints_partial_block_once():
    t = _Term()
    r = t.renderer()
    r.feed("unfinished sentence")   # sub-throttle: nothing painted yet?
    r.finalize()                     # must still paint it
    assert "unfinished sentence" in _strip(t.text())


def test_stream_rich_mode_smoke():
    pytest.importorskip("rich")
    t = _Term()
    r = rd.StreamRenderer(
        "rich", _THEME, write=t.write, size_fn=lambda: t.size, clock=t.clock
    )
    r.feed("# Title\n\n- item one\n- item two\n\n```python\nx = 1\n```\n")
    r.finalize()
    stripped = _strip(t.text())
    assert "Title" in stripped
    assert "item one" in stripped
    assert "x = 1" in stripped
    assert "```" not in stripped     # fences rendered, not echoed


def test_feed_strips_model_ansi_and_control_chars():
    # Model output is untrusted terminal input: ESC/OSC/C0 must not reach the
    # user's terminal through any render mode.
    from gmlx.render import StreamRenderer
    out = []
    r = StreamRenderer("plain", write=out.append)
    r.feed("hi \x1b[31mred\x1b]0;evil\x07\x00 there\rok\n")
    r.finalize()
    text = "".join(out)
    assert "\x1b" not in text and "\x00" not in text and "\x07" not in text
    assert "\r" not in text
    assert "hi " in text and "red" in text and "ok" in text
