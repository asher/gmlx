#!/usr/bin/env python3
"""`gmlx.theme`: slot resolution across color depths, colorblind remap,
NO_COLOR emptiness, and the rich Theme bridge. CPU-only, no terminal needed."""
from __future__ import annotations

import pytest

from gmlx import theme as th


def test_all_themes_resolve_every_slot():
    for name in th.list_themes():
        t = th.resolve_theme(name, depth=1 << 24)
        for slot in th._SLOTS:
            assert isinstance(t.sgr[slot], str)
        assert t.reset == th.RESET
        assert t.code_theme


def test_unknown_theme_names_options():
    with pytest.raises(ValueError, match="dark-hc"):
        th.resolve_theme("neon-zebra")


def test_color_off_empties_every_slot():
    t = th.resolve_theme("nord", color=False)
    assert all(v == "" for v in t.sgr.values())
    assert t.reset == ""
    assert t.paint("heading", "x") == "x"


def test_depth_ladder_truecolor_256_16():
    tc = th.resolve_theme("nord", depth=1 << 24)
    c256 = th.resolve_theme("nord", depth=256)
    c16 = th.resolve_theme("nord", depth=16)
    assert "38;2;136;192;208" in tc.heading          # exact RGB
    assert "38;5;" in c256.heading                   # cube-mapped
    assert "38;" not in c16.heading                  # RGB theme degrades to attrs
    assert "1" in c16.heading                        # bold survives


def test_16color_theme_uses_basic_codes():
    t = th.resolve_theme("dark", depth=1 << 24)
    assert t.heading == "\x1b[1;36m"                 # bold cyan, palette-respecting


def test_colorblind_remaps_accents_to_okabe_ito():
    t = th.resolve_theme("dark", colorblind=True, depth=1 << 24)
    assert "38;2;86;180;233" in t.heading            # sky blue
    assert "38;2;230;159;0" in t.bullet              # orange
    assert "38;2;213;94;0" in t.error                # vermillion, not red
    # attribute-only slots keep their look
    assert t.bold == "\x1b[1m"
    assert t.thinking == th.resolve_theme("dark", depth=1 << 24).thinking


def test_colorblind_applies_to_named_themes_too():
    plain = th.resolve_theme("dracula", depth=1 << 24)
    cb = th.resolve_theme("dracula", colorblind=True, depth=1 << 24)
    assert plain.error != cb.error
    assert "38;2;213;94;0" in cb.error


def test_paint_wraps_with_reset():
    t = th.resolve_theme("dark", depth=1 << 24)
    assert t.paint("heading", "Hi") == "\x1b[1;36mHi\x1b[0m"


def test_detect_depth():
    assert th.detect_depth({"COLORTERM": "truecolor"}) == 1 << 24
    assert th.detect_depth({"TERM": "xterm-256color"}) == 256
    assert th.detect_depth({"TERM": "vt100"}) == 16
    assert th.detect_depth({}) == 16


def test_rich_theme_bridge():
    pytest.importorskip("rich")
    t = th.resolve_theme("nord", depth=1 << 24)
    rt = t.rich_theme()
    assert "markdown.h1" in rt.styles
    assert rt.styles["markdown.h1"].color.name == "#88c0d0"


def test_colorblind_keeps_color_on_a_16_color_terminal():
    """`--colorblind` on TERM=linux used to strip every remapped accent to no
    color at all (RGB-only, fg16=None), leaving the user who asked for the
    accessibility flag with less differentiation than the default theme."""
    for name, remap in (("dark", th._CB_DARK), ("light", th._CB_LIGHT)):
        t = th.resolve_theme(name, colorblind=True, depth=16)
        for slot, rgb in remap.items():
            prefix = getattr(t, slot)
            codes = prefix.removeprefix("\x1b[").removesuffix("m").split(";")
            assert str(th._OI_FG16[rgb]) in codes, f"{name}.{slot}: {prefix!r}"
            assert "38;2" not in prefix and "38;5" not in prefix
