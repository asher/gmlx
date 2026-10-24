"""The picker widget: its state and rendering with no terminal, and its
key handling through a pipe."""

from __future__ import annotations

import pytest

from gmlx.tui import pick
from gmlx.tui.pick import Group, Picker, Row, Rung, Toggle, plain_text


def _rung(id, tier, quant, size, state="resident", on_disk=False, default=False):
    return Rung(id=id, tier=tier, quant=quant, size_label=size, state=state,
                on_disk=on_disk, default=default)


VISION = Toggle("vision", "image input", "0.9 GB", True, on_when_streaming=False,
                streaming_note="Images work only with gmlx serve when the model runs from the SSD.")
DRAFTER = Toggle("drafter", "faster replies", "1.9 GB", True, hide_when_streaming=True)

DENSE = Row("dense", "Qwen3.8-27B", "27B", ("images", "faster replies"), "18.1 GB",
            "Strong at long documents, code and tool calls.", "Steady for a 27B.", "",
            "Apache 2.0", "",
            (_rung("fast", "fastest", "Q4_0", "15.0 GB"),
             _rung("bal", "balanced", "Q4_K_M", "15.3 GB", default=True),
             _rung("high", "high", "Q6_K", "20.5 GB", state="streams", on_disk=True)),
            "bal", (VISION, DRAFTER))
GEMMA = Row("gemma", "gemma-4-12B", "12B", ("images", "faster replies"), "7.2 GB",
            "Strong general chat with image input.", "Steady for a 12B.", "", "Apache 2.0",
            "needs 20 GB free",
            (_rung("g4", "balanced", "Q4_K_M", "6.6 GB", default=True),
             _rung("g6", "high", "Q6_K", "9.1 GB")),
            "g4", (Toggle("vision", "image input", "0.2 GB", True),))
PLAIN = Row("plain", "K2-Horizon-7B", "7B", (), "6.9 GB", "A chat model with a 512K context.",
            "Quick: a 7B dense model.", "", "Apache 2.0", "",
            (_rung("p6", "high", "Q6_K", "6.9 GB", default=True),), "p6", ())
MOE = Row("flashy", "Qwen3.8-Flash-Next", "Flash-Next", ("images", "faster replies"), "90.7 GB",
          "Strong at long documents.", "Quick for its size.", "", "Qwen Community License", "",
          (_rung("c", "compact", "UD-IQ3_XXS", "76.3 GB"),
           _rung("f", "fastest", "UD-IQ4_XS", "87.2 GB", default=True)),
          "f", (VISION, DRAFTER))

GROUPS = (Group("Recommended for this Mac", (DENSE, GEMMA)),
          Group("Runs in memory", (PLAIN,)),
          Group("Larger than memory: runs from the SSD, much slower", (MOE,)))


def _label(row, rung, on_ids):
    room = 100 - 10 * len(on_ids)
    if rung.state == "streams":
        return "from SSD, slow", ""
    if rung.tier == "compact":
        return f"compact, room for ~{room}K", "smaller, lower quality"
    return f"room for ~{room}K", ""


def _footer(row, rung, on_ids):
    cmd = f"gmlx pull {row.id}"
    if not rung.default:
        cmd += f" --quant {rung.tier}"
    for t in row.toggles:
        if t.on and t.id not in on_ids:
            cmd += f" --no-{t.id}"
        if not t.on and t.id in on_ids:
            cmd += f" --{t.id}"
    return [f"Download {rung.size_label}", cmd]


def _picker(groups=GROUPS):
    return Picker(groups, ram_label="This Mac has 128 GB of memory.", footer=_footer,
                  rung_label=_label, explain_line="Context is how much text the model keeps in view.",
                  empty_line="No catalog model fits this Mac.",
                  no_match_line="No model matches '{filter}'")


def _lines(p, width=80, rows=24):
    return plain_text(p.render(width, rows)).split("\n")


# -- the list screen -------------------------------------------------------------

def test_list_columns_align_and_widen():
    p = _picker()
    lines = _lines(p, 80)
    assert lines[0] == "This Mac has 128 GB of memory."
    rows = [ln for ln in lines if ln[1:4].strip().isdigit() or ln.startswith(">")]
    assert rows[0].startswith(">  1 Qwen3.8-27B")
    assert rows[1].startswith("   2 gemma-4-12B")
    # Every row puts its size in the same column.
    ends = {ln.rindex(" GB") for ln in rows}
    assert len(ends) == 1
    assert "images, faster replies" in rows[0] and "18.1 GB" in rows[0]
    wide = [ln for ln in _lines(p, 120) if ln.startswith(">")][0]
    assert wide.index(" 27B") > rows[0].index(" 27B")
    assert "Recommended for this Mac" in lines and "Runs in memory" in lines
    assert lines[-1] == pick.LIST_HINTS
    # The dim row carries its note in place of its tags.
    assert "needs 20 GB free" in rows[1] and "images" not in rows[1]
    assert all(len(ln) <= 80 for ln in lines)
    styles = [s for s, t in p.render(80, 24) if "gemma-4-12B" in t]
    assert styles == ["class:pick.dim"]
    assert [s for s, t in p.render(80, 24) if "Qwen3.8-27B" in t and t.startswith(">")] == ["class:pick.cursor"]


def test_detail_pane_is_three_lines_for_the_cursor_row():
    p = _picker()
    lines = _lines(p)
    i = lines.index(pick.LIST_HINTS)
    detail = lines[i - 4:i - 1]
    assert detail[0] == "  Strong at long documents, code and tool calls. Apache 2.0"
    assert detail[1] == "  Steady for a 27B."
    assert detail[2] == "  fastest 15.0 GB  balanced 15.3 GB (default)  high 20.5 GB"
    assert all(len(ln) <= 76 for ln in detail)
    assert "Context is how much" not in "\n".join(lines)
    p.move(1)
    assert _lines(p)[i - 4].startswith("  Strong general chat")


def test_filter_is_case_insensitive_and_drops_the_numbers():
    p = _picker()
    for ch in "GEMMA":
        p.type(ch)
    lines = _lines(p)
    assert lines[1] == "filter: GEMMA"
    rows = [ln for ln in lines if ln.startswith(">")]
    assert len(rows) == 1 and "gemma-4-12B" in rows[0] and not rows[0][2:4].strip()
    assert "Runs in memory" not in lines                       # empty groups collapse
    assert p.current() is GEMMA
    p.jump(1)                                                 # digits do nothing while filtering
    assert p.current() is GEMMA
    p.backspace()
    assert p.filter == "GEMM"
    p.clear_filter()
    assert p.filter == "" and p.current() is GEMMA           # the cursor clamps, not resets
    p.jump(3)
    assert p.current() is PLAIN
    for ch in "images":                                       # tags filter too
        p.type(ch)
    assert [r.id for r in p.visible()] == ["dense", "gemma", "flashy"]
    p.clear_filter()
    for ch in "xyz":
        p.type(ch)
    assert "  No model matches 'xyz'" in _lines(p)
    assert p.current() is None


def test_empty_groups_show_the_empty_line_and_cursor_clamps():
    p = _picker((Group("Recommended for this Mac", ()), Group("Runs in memory", ())))
    lines = _lines(p)
    assert "  No catalog model fits this Mac." in lines
    assert "Recommended for this Mac" not in lines
    assert p.select() is False
    p = _picker()
    p.move(-5)
    assert p.current() is DENSE
    p.move(50)
    assert p.current() is MOE


def test_bounded_height_with_more_markers():
    rows = tuple(Row(f"r{i}", f"Model {i}", "7B", (), "1 GB", "Blurb.", "Quick.", "", "MIT", "",
                     (_rung(f"q{i}", "high", "Q6_K", "1 GB", default=True),), f"q{i}", ())
                 for i in range(12))
    p = _picker((Group("Runs in memory", rows),))
    lines = _lines(p, 80, 12)
    assert len(lines) <= 12 + 1
    body = [ln for ln in lines if ln.startswith("  ...")]
    assert body == ["  ... 9 more"] or body[-1].endswith("more")
    p.move(11)
    lines = _lines(p, 80, 12)
    assert any(ln.startswith("  ...") and "more" in ln for ln in lines[:4])
    assert any(ln.startswith("> ") and "Model 11" in ln for ln in lines)


# -- the configure screen -----------------------------------------------------------

def test_select_seeds_the_default_rung_and_toggles():
    p = _picker()
    assert p.select() is True and p.screen == "configure"
    assert p.rung().id == "bal"
    lines = _lines(p)
    assert lines[0] == "Qwen3.8-27B   27B   Apache 2.0"
    assert "Quality" in lines and "Extras" in lines
    bal = [ln for ln in lines if "balanced" in ln][0]
    assert bal.startswith("> (*) balanced") and "Q4_K_M" in bal and "room for ~80K" in bal
    assert bal.rstrip().endswith("default")
    fast = [ln for ln in lines if "fastest" in ln][0]
    assert fast.startswith("  ( ) fastest") and "room for ~80K" in fast
    high = [ln for ln in lines if "high" in ln and "Q6_K" in ln][0]
    assert "from SSD, slow" in high and "on disk" in high
    assert "  Context is how much text the model keeps in view." in lines
    assert "  [x] image input" in "\n".join(lines) and "  [x] faster replies" in "\n".join(lines)
    assert "Download 15.3 GB" in lines and "gmlx pull dense" in lines
    assert lines[-1] == pick.CONFIGURE_HINTS
    assert p.plan() == pick.Plan(DENSE, DENSE.rungs[1], ("vision", "drafter"))


def test_toggles_change_labels_and_footer_but_not_the_default_mark():
    p = _picker()
    p.select()
    assert p.ccursor == 1           # the cursor opens on the default rung
    p.cmove(1)                      # onto the streaming rung: moving chooses nothing
    p.cmove(1)                      # the first toggle: vision
    assert p._items()[p.ccursor][1].id == "vision"
    assert p.rung().id == "bal"
    # Turn the drafter off on the resident set.
    p.cmove(1)
    assert p._items()[p.ccursor][1].id == "drafter"
    p.toggle()
    lines = _lines(p)
    assert "[ ] faster replies" in "\n".join(lines)
    assert all("room for ~90K" in ln for ln in lines if "( )" in ln or "(*)" in ln if "SSD" not in ln)
    assert "gmlx pull dense --no-drafter" in lines
    assert [ln for ln in lines if "balanced" in ln][0].rstrip().endswith("default")
    assert p.plan().toggles == ("vision",)
    # The higher tier with images off reads --quant high --no-vision.
    p2 = _picker((Group("Runs in memory", (Row(
        "d2", "Dense 2", "27B", ("images",), "1 GB", "B.", "Q.", "", "MIT", "",
        (_rung("lo", "balanced", "Q4_K_M", "1 GB", default=True), _rung("hi", "high", "Q6_K", "2 GB")),
        "lo", (Toggle("vision", "image input", "0.9 GB", True),)),)),))
    p2.select()
    p2.cmove(1)
    p2.toggle()                     # space chooses the high tier
    p2.cmove(1)
    p2.toggle()                     # and turns images off
    lines = _lines(p2)
    assert "gmlx pull d2 --quant high --no-vision" in lines
    assert [ln for ln in lines if "balanced" in ln][0].rstrip().endswith("default")
    assert "(*) high" in "\n".join(lines)


def test_streaming_rung_switches_toggle_sets_and_hides_the_drafter():
    p = _picker()
    p.select()
    p.cmove(3)                       # drafter on the resident set
    p.toggle()                       # off
    p.cmove(-2)                      # high: streams
    assert p.rung().id == "bal"
    p.toggle()                       # space chooses it
    assert p.rung().id == "high" and p.streaming()
    lines = _lines(p)
    assert "faster replies" not in "\n".join(lines)
    vision = [ln for ln in lines if "image input" in ln][0]
    assert vision.startswith("  [ ] image input") and "Images work only with gmlx serve" in vision
    assert p.on_ids() == ()
    p.cmove(1)                       # the vision toggle on the streaming set
    assert p._items()[p.ccursor][1].id == "vision"
    p.toggle()
    assert p.on_ids() == ("vision",)
    assert p.plan().toggles == ("vision",)
    # Back on the resident rung the resident set is as it was left.
    p.cmove(-2)
    p.toggle()
    assert p.rung().id == "bal" and p.on_ids() == ("vision",)
    lines = _lines(p)
    assert "[x] image input" in "\n".join(lines) and "[ ] faster replies" in "\n".join(lines)
    assert "Images work only" not in "\n".join(lines)
    # And the streaming set remembers its own vision choice.
    p.cmove(1)
    p.toggle()
    assert p.rung().id == "high" and p.on_ids() == ("vision",)
    assert p.plan().toggles == ("vision",)


def test_configure_clips_with_markers_and_back_keeps_state():
    rungs = tuple(_rung(f"q{i}", f"t{i}", f"Q{i}_K", "1 GB", default=(i == 0)) for i in range(10))
    row = Row("many", "Many", "1B", (), "1 GB", "B.", "Q.", "", "MIT", "", rungs, "q0", (VISION,))
    p = _picker((Group("Runs in memory", (row,)),))
    p.select()
    lines = _lines(p, 80, 10)
    assert len(lines) <= 11
    assert any("more" in ln for ln in lines)
    p.cmove(9)
    p.toggle()
    assert p.rung().id == "q9"
    lines = _lines(p, 80, 10)
    assert any("> (*) t9" in ln for ln in lines)
    # Back keeps the filter and the cursor.
    p = _picker()
    for ch in "gem":
        p.type(ch)
    p.select()
    p.back()
    assert p.screen == "list" and p.filter == "gem" and p.current() is GEMMA
    # A row with no toggles shows no Extras.
    p = _picker()
    p.jump(3)
    p.select()
    assert "Extras" not in _lines(p)
    p.cmove(5)
    assert p.ccursor == 0
    p.toggle()                       # nothing to toggle
    assert p.plan().toggles == ()


# -- through the terminal ---------------------------------------------------------------

def _run(keys: str, groups=GROUPS, color=False):
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        pipe.close()
        return pick.pick(groups, ram_label="This Mac has 128 GB of memory.", footer=_footer,
                         rung_label=_label, explain_line="Context.", empty_line="Nothing.",
                         no_match_line="No model matches '{filter}'", color=color,
                         input=pipe, output=DummyOutput())


def test_enter_twice_takes_the_default_plan():
    plan = _run("\r\r")
    assert plan == pick.Plan(DENSE, DENSE.rungs[1], ("vision", "drafter"))


def test_keys_move_select_and_toggle():
    # down, enter, down x3 (onto the drafter), space, enter
    plan = _run("\x1b[B\r\x1b[B\x1b[B\x1b[B \r")
    assert plan.row is GEMMA
    assert plan.rung.id == "g4" and plan.toggles == ()
    # Right opens configure, Left goes back, then enter twice.
    plan = _run("\x1b[C\x1b[D\r\r")
    assert plan.row is DENSE and plan.rung.id == "bal"
    # Filter then select.
    plan = _run("gemma\r\r")
    assert plan.row is GEMMA
    # A digit jumps.
    plan = _run("3\r\r")
    assert plan.row is PLAIN
    # j and k move on the configure screen; enter on a tier chooses it.
    plan = _run("\rj\r")
    assert plan.row is DENSE and plan.rung.id == "high" and plan.toggles == ()
    plan = _run("\rk\r")
    assert plan.rung.id == "fast"
    # Space on a tier chooses it, and enter on a toggle keeps that choice.
    plan = _run("\rk \x1b[B\x1b[B\x1b[B\r")
    assert plan.rung.id == "fast" and plan.toggles == ("vision", "drafter")


def test_escape_clears_the_filter_first_then_quits():
    assert _run("qw\x1b\r\r") == pick.Plan(DENSE, DENSE.rungs[1], ("vision", "drafter"))
    assert _run("\x1bx") is None
    assert _run("\x1b") is None
    assert _run("\rq") is None            # q quits only on the configure screen
    assert _run("q") is None              # an unmatched filter, then the pipe closes
    assert _run("\x03") is None
    assert _run("\x04") is None
    assert _run("") is None               # a closed pipe


def test_non_tty_without_seams_is_unavailable(monkeypatch):
    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(pick.PickerUnavailable):
        pick.pick(GROUPS, ram_label="", footer=_footer, rung_label=_label, explain_line="",
                  empty_line="", no_match_line="")


def test_color_false_is_an_empty_style():
    assert pick._style(False).style_rules == []
    assert pick._style(True).style_rules
