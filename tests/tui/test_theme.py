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


@pytest.fixture
def user_themes():
    """Registry isolation: user themes registered in a test don't leak."""
    saved = dict(th._USER_THEMES)
    yield th._USER_THEMES
    th._USER_THEMES.clear()
    th._USER_THEMES.update(saved)


def test_user_theme_registers_and_resolves(user_themes):
    warnings = th.register_user_themes({
        "my-black": {
            "thinking": {"italic": True, "fg16": 94},
            "heading": {"bold": True, "rgb": "#88c0d0"},
            "stat": {"rgb": [146, 131, 116]},
        },
    })
    assert warnings == []
    assert "my-black" in th.list_themes()
    t = th.resolve_theme("my-black", depth=1 << 24)
    assert t.thinking == "\x1b[3;94m"
    assert t.heading == "\x1b[1;38;2;136;192;208m"   # hex parsed
    assert t.stat == "\x1b[38;2;146;131;116m"        # list parsed
    # Unspecified slots inherit from dark.
    assert t.error == th.resolve_theme("dark", depth=1 << 24).error


def test_user_theme_extends_named_theme_and_shadows_builtins(user_themes):
    assert th.register_user_themes({
        "nord": {"extends": "nord", "thinking": {"fg16": 96}},   # shadow
        "kid": {"extends": "nord", "error": {"bold": True, "fg16": 91}},
    }) == []
    t = th.resolve_theme("nord", depth=1 << 24)
    assert t.thinking == "\x1b[96m"                  # overridden slot
    assert t.code_theme == "nord"                    # meta inherited
    kid = th.resolve_theme("kid", depth=1 << 24)
    assert kid.thinking == "\x1b[96m"                # extends the shadowed nord
    assert kid.error == "\x1b[1;91m"


def test_user_theme_colorblind_remap_applies(user_themes):
    th.register_user_themes({"my-black": {"thinking": {"fg16": 94}}})
    cb = th.resolve_theme("my-black", colorblind=True, depth=1 << 24)
    assert "38;2;213;94;0" in cb.error               # vermillion
    assert cb.thinking == "\x1b[94m"                 # non-accent slot untouched


def test_user_theme_bad_definitions_warn_and_skip(user_themes):
    warnings = th.register_user_themes({
        "bad-slot": {"thinkng": {"fg16": 94}},
        "bad-fg": {"thinking": {"fg16": 12}},
        "bad-rgb": {"thinking": {"rgb": "#12345"}},
        "bad-base": {"extends": "neon-zebra"},
        "good": {"thinking": {"fg16": 94}},
    })
    assert len(warnings) == 4
    assert any("thinkng" in w for w in warnings)
    assert any("30-37 or 90-97" in w for w in warnings)
    assert any("#rrggbb" in w for w in warnings)
    assert any("neon-zebra" in w for w in warnings)
    assert "good" in th.list_themes()
    for name in ("bad-slot", "bad-fg", "bad-rgb", "bad-base"):
        assert name not in th.list_themes()


def test_config_carries_theme_keys():
    from gmlx.config import build_config

    cfg = build_config({
        "theme": "my-black",
        "themes": {"my-black": {"thinking": {"fg16": 94}}},
    })
    assert cfg.theme == "my-black"
    assert cfg.themes == {"my-black": {"thinking": {"fg16": 94}}}
    assert build_config({}).theme is None


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
