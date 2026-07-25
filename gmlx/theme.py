"""Color themes for the chat TUI.

A ``Theme`` resolves named style slots (heading, inline code, thinking gutter,
...) to ANSI SGR prefixes at the terminal's color depth. The ``dark`` /
``light`` / ``dark-hc`` themes use the 16-color palette so they follow the
terminal's own colors; the named themes (nord, dracula, solarized-dark,
gruvbox) carry explicit RGB values with a 256-color fallback.

``colorblind=True`` remaps every accent slot onto the Okabe-Ito palette
(blue / orange / sky / vermillion - no red-green opposition), giving a
colorblind-friendly variant of any theme.

User-defined themes (a ``themes:`` mapping in gmlx.yaml, registered by chat
startup via :func:`register_user_themes`) resolve exactly like built-ins and
may shadow them; unspecified slots inherit from the theme they ``extends``
(default ``dark``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

RESET = "\x1b[0m"

# Okabe-Ito colorblind-safe palette (shared accent set for -cb variants).
_OI_ORANGE = (230, 159, 0)
_OI_SKY = (86, 180, 233)
_OI_GREEN = (0, 158, 115)
_OI_YELLOW = (240, 228, 66)
_OI_BLUE = (0, 114, 178)
_OI_VERMILLION = (213, 94, 0)
_OI_PURPLE = (204, 121, 167)

# Nearest ANSI-16 partner per Okabe-Ito color. A 16-color terminal cannot show
# the RGB value, and dropping to no color at all would leave the colorblind
# user with *less* differentiation than the default theme gives.
_OI_FG16 = {
    _OI_ORANGE: 33,       # yellow
    _OI_SKY: 36,          # cyan
    _OI_GREEN: 32,
    _OI_YELLOW: 33,
    _OI_BLUE: 34,
    _OI_VERMILLION: 31,   # red
    _OI_PURPLE: 35,       # magenta
}


@dataclass(frozen=True)
class Style:
    """One slot's look: attributes + an optional 16-color code or RGB value."""

    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False
    fg16: int | None = None  # SGR 30-37 / 90-97, used when no rgb or at depth 16
    rgb: tuple[int, int, int] | None = None


def _rgb_to_256(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    if r == g == b:  # grayscale ramp
        if r < 8:
            return 16
        if r >= 248:            # 248 would map to 256, past the 232-255 ramp
            return 231
        return 232 + (r - 8) * 24 // 240
    def q(v):
        return 0 if v < 48 else (1 if v < 114 else (v - 35) // 40)
    return 16 + 36 * q(r) + 6 * q(g) + q(b)


def _sgr(style: Style, depth: int) -> str:
    """Resolve a Style to an SGR escape prefix ('' for the empty style)."""
    codes: list[str] = []
    if style.bold:
        codes.append("1")
    if style.dim:
        codes.append("2")
    if style.italic:
        codes.append("3")
    if style.underline:
        codes.append("4")
    if style.rgb is not None and depth >= 256:
        if depth >= 1 << 24:
            r, g, b = style.rgb
            codes.append(f"38;2;{r};{g};{b}")
        else:
            codes.append(f"38;5;{_rgb_to_256(style.rgb)}")
    elif style.fg16 is not None:
        codes.append(str(style.fg16))
    if not codes:
        return ""
    return "\x1b[" + ";".join(codes) + "m"


# Slot names -> what they style. Accent slots are the ones the colorblind
# modifier remaps; attribute-only slots (bold/italic/thinking) pass through.
_SLOTS = (
    "thinking",     # reasoning gutter + payoff
    "heading",
    "bold",
    "italic",
    "inline_code",
    "code_block",   # fence body
    "code_border",  # fence frame + language tag
    "bullet",       # list markers
    "blockquote",
    "link",
    "hr",
    "stat",         # end-of-reply stat line
    "info",         # [chat] notices
    "error",
)


@dataclass(frozen=True)
class ThemeSpec:
    """Palette definition, depth-independent."""

    name: str
    slots: dict[str, Style]
    code_theme: str = "monokai"        # pygments style for rich fences
    code_theme_cb: str | None = None   # override under colorblind
    ptk_toolbar: str | None = None     # prompt_toolkit style, None = default


def _s(**kw) -> Style:
    return Style(**kw)


THEMES: dict[str, ThemeSpec] = {
    # Terminal-palette-respecting defaults (16-color).
    "dark": ThemeSpec(
        name="dark",
        slots={
            "thinking": _s(italic=True, fg16=94),     # bright blue; dim is illegible on many dark palettes
            "heading": _s(bold=True, fg16=36),        # cyan
            "bold": _s(bold=True),
            "italic": _s(italic=True),
            "inline_code": _s(fg16=33),               # yellow
            "code_block": _s(fg16=37),
            "code_border": _s(dim=True, fg16=36),
            "bullet": _s(fg16=36),
            "blockquote": _s(dim=True, italic=True),
            "link": _s(underline=True, fg16=34),      # blue
            "hr": _s(dim=True),
            "stat": _s(dim=True),
            "info": _s(dim=True),
            "error": _s(fg16=31),                     # red
        },
        code_theme="monokai",
    ),
    "light": ThemeSpec(
        name="light",
        slots={
            "thinking": _s(dim=True),
            "heading": _s(bold=True, fg16=34),
            "bold": _s(bold=True),
            "italic": _s(italic=True),
            "inline_code": _s(fg16=35),               # magenta reads on white
            "code_block": _s(fg16=30),
            "code_border": _s(dim=True, fg16=34),
            "bullet": _s(fg16=34),
            "blockquote": _s(dim=True, italic=True),
            "link": _s(underline=True, fg16=34),
            "hr": _s(dim=True),
            "stat": _s(dim=True),
            "info": _s(dim=True),
            "error": _s(fg16=31),
        },
        code_theme="default",
    ),
    # High-contrast dark: no dim anywhere, bright 16-color + bold.
    "dark-hc": ThemeSpec(
        name="dark-hc",
        slots={
            "thinking": _s(fg16=90),                  # bright black, not dim
            "heading": _s(bold=True, underline=True, fg16=97),
            "bold": _s(bold=True, fg16=97),
            "italic": _s(italic=True, fg16=97),
            "inline_code": _s(bold=True, fg16=93),
            "code_block": _s(fg16=97),
            "code_border": _s(fg16=96),
            "bullet": _s(bold=True, fg16=96),
            "blockquote": _s(fg16=96),
            "link": _s(bold=True, underline=True, fg16=94),
            "hr": _s(fg16=90),
            "stat": _s(fg16=90),
            "info": _s(fg16=90),
            "error": _s(bold=True, fg16=91),
        },
        code_theme="github-dark",
    ),
    # Named palettes (truecolor with 256 fallback).
    "nord": ThemeSpec(
        name="nord",
        slots={
            "thinking": _s(dim=True, rgb=(97, 110, 136)),
            "heading": _s(bold=True, rgb=(136, 192, 208)),
            "bold": _s(bold=True, rgb=(229, 233, 240)),
            "italic": _s(italic=True),
            "inline_code": _s(rgb=(235, 203, 139)),
            "code_block": _s(rgb=(216, 222, 233)),
            "code_border": _s(rgb=(76, 86, 106)),
            "bullet": _s(rgb=(129, 161, 193)),
            "blockquote": _s(italic=True, rgb=(143, 188, 187)),
            "link": _s(underline=True, rgb=(136, 192, 208)),
            "hr": _s(rgb=(76, 86, 106)),
            "stat": _s(rgb=(97, 110, 136)),
            "info": _s(rgb=(97, 110, 136)),
            "error": _s(rgb=(191, 97, 106)),
        },
        code_theme="nord",
        ptk_toolbar="bg:#3b4252 #d8dee9",
    ),
    "dracula": ThemeSpec(
        name="dracula",
        slots={
            "thinking": _s(dim=True, rgb=(98, 114, 164)),
            "heading": _s(bold=True, rgb=(189, 147, 249)),
            "bold": _s(bold=True, rgb=(248, 248, 242)),
            "italic": _s(italic=True, rgb=(241, 250, 140)),
            "inline_code": _s(rgb=(80, 250, 123)),
            "code_block": _s(rgb=(248, 248, 242)),
            "code_border": _s(rgb=(98, 114, 164)),
            "bullet": _s(rgb=(255, 121, 198)),
            "blockquote": _s(italic=True, rgb=(98, 114, 164)),
            "link": _s(underline=True, rgb=(139, 233, 253)),
            "hr": _s(rgb=(98, 114, 164)),
            "stat": _s(rgb=(98, 114, 164)),
            "info": _s(rgb=(98, 114, 164)),
            "error": _s(rgb=(255, 85, 85)),
        },
        code_theme="dracula",
        ptk_toolbar="bg:#44475a #f8f8f2",
    ),
    "solarized-dark": ThemeSpec(
        name="solarized-dark",
        slots={
            "thinking": _s(dim=True, rgb=(88, 110, 117)),
            "heading": _s(bold=True, rgb=(38, 139, 210)),
            "bold": _s(bold=True, rgb=(147, 161, 161)),
            "italic": _s(italic=True),
            "inline_code": _s(rgb=(181, 137, 0)),
            "code_block": _s(rgb=(147, 161, 161)),
            "code_border": _s(rgb=(88, 110, 117)),
            "bullet": _s(rgb=(42, 161, 152)),
            "blockquote": _s(italic=True, rgb=(88, 110, 117)),
            "link": _s(underline=True, rgb=(38, 139, 210)),
            "hr": _s(rgb=(88, 110, 117)),
            "stat": _s(rgb=(88, 110, 117)),
            "info": _s(rgb=(88, 110, 117)),
            "error": _s(rgb=(220, 50, 47)),
        },
        code_theme="solarized-dark",
        ptk_toolbar="bg:#073642 #93a1a1",
    ),
    "gruvbox": ThemeSpec(
        name="gruvbox",
        slots={
            "thinking": _s(dim=True, rgb=(146, 131, 116)),
            "heading": _s(bold=True, rgb=(250, 189, 47)),
            "bold": _s(bold=True, rgb=(235, 219, 178)),
            "italic": _s(italic=True),
            "inline_code": _s(rgb=(184, 187, 38)),
            "code_block": _s(rgb=(235, 219, 178)),
            "code_border": _s(rgb=(146, 131, 116)),
            "bullet": _s(rgb=(254, 128, 25)),
            "blockquote": _s(italic=True, rgb=(146, 131, 116)),
            "link": _s(underline=True, rgb=(131, 165, 152)),
            "hr": _s(rgb=(146, 131, 116)),
            "stat": _s(rgb=(146, 131, 116)),
            "info": _s(rgb=(146, 131, 116)),
            "error": _s(rgb=(251, 73, 52)),
        },
        code_theme="gruvbox-dark",
        ptk_toolbar="bg:#3c3836 #ebdbb2",
    ),
}


def list_themes() -> list[str]:
    return sorted(set(THEMES) | set(_USER_THEMES))


# User-defined themes (``themes:`` in gmlx.yaml), registered at chat startup.
# A user name shadows a built-in; unspecified slots inherit from ``extends``.
_USER_THEMES: dict[str, ThemeSpec] = {}

_STYLE_KEYS = frozenset({"bold", "dim", "italic", "underline", "fg16", "rgb"})
_THEME_META_KEYS = frozenset(
    {"extends", "code_theme", "code_theme_cb", "ptk_toolbar"})
_FG16_CODES = frozenset(range(30, 38)) | frozenset(range(90, 98))


def _parse_rgb(theme: str, slot: str, value) -> tuple[int, int, int]:
    if isinstance(value, str):
        v = value.lstrip("#")
        if len(v) == 6:
            try:
                return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
    elif (isinstance(value, (list, tuple)) and len(value) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in value)):
        return tuple(value)
    raise ValueError(
        f"theme {theme!r} slot {slot!r}: rgb must be '#rrggbb' or "
        f"[r, g, b] 0-255, got {value!r}")


def _parse_user_style(theme: str, slot: str, raw) -> Style:
    if not isinstance(raw, dict):
        raise ValueError(
            f"theme {theme!r} slot {slot!r}: expected a mapping of style keys "
            f"({', '.join(sorted(_STYLE_KEYS))}), got {type(raw).__name__}")
    unknown = set(raw) - _STYLE_KEYS
    if unknown:
        raise ValueError(
            f"theme {theme!r} slot {slot!r}: unknown style keys "
            f"{sorted(unknown)} (valid: {', '.join(sorted(_STYLE_KEYS))})")
    kw: dict = {k: bool(raw[k])
                for k in ("bold", "dim", "italic", "underline") if k in raw}
    fg16 = raw.get("fg16")
    if fg16 is not None:
        if not isinstance(fg16, int) or fg16 not in _FG16_CODES:
            raise ValueError(
                f"theme {theme!r} slot {slot!r}: fg16 must be an ANSI "
                f"foreground code 30-37 or 90-97, got {fg16!r}")
        kw["fg16"] = fg16
    if raw.get("rgb") is not None:
        kw["rgb"] = _parse_rgb(theme, slot, raw["rgb"])
    return Style(**kw)


def define_user_theme(name: str, spec) -> ThemeSpec:
    """Build a :class:`ThemeSpec` from a config mapping.

    Keys are slot names (see ``_SLOTS``) mapping to style keys, plus optional
    ``extends`` (the theme filling unspecified slots, default ``dark``),
    ``code_theme`` / ``code_theme_cb`` (pygments styles for rich fences) and
    ``ptk_toolbar`` (prompt_toolkit toolbar style). Raises ValueError with the
    offending key on any malformed piece."""
    if not isinstance(spec, dict):
        raise ValueError(
            f"theme {name!r}: expected a mapping of slots, "
            f"got {type(spec).__name__}")
    unknown = set(spec) - set(_SLOTS) - _THEME_META_KEYS
    if unknown:
        raise ValueError(
            f"theme {name!r}: unknown keys {sorted(unknown)} (slots: "
            f"{', '.join(_SLOTS)}; meta: {', '.join(sorted(_THEME_META_KEYS))})")
    base_name = str(spec.get("extends", "dark"))
    base = _USER_THEMES.get(base_name) or THEMES.get(base_name)
    if base is None:
        raise ValueError(
            f"theme {name!r}: extends unknown theme {base_name!r} "
            f"(choose from: {', '.join(list_themes())})")
    slots = dict(base.slots)
    for slot in _SLOTS:
        if slot in spec:
            slots[slot] = _parse_user_style(name, slot, spec[slot])
    return ThemeSpec(
        name=name,
        slots=slots,
        code_theme=str(spec.get("code_theme", base.code_theme)),
        code_theme_cb=spec.get("code_theme_cb", base.code_theme_cb),
        ptk_toolbar=spec.get("ptk_toolbar", base.ptk_toolbar),
    )


def register_user_themes(themes: dict | None) -> list[str]:
    """Register the config's ``themes:`` mapping. Returns a warning string per
    definition that failed validation; the valid ones still register (later
    definitions may extend earlier ones)."""
    warnings: list[str] = []
    for name, spec in (themes or {}).items():
        try:
            _USER_THEMES[str(name)] = define_user_theme(str(name), spec)
        except ValueError as e:
            warnings.append(str(e))
    return warnings


# Colorblind remap: accent slots -> Okabe-Ito, keyed by slot semantics so it
# applies uniformly to every theme. Dark themes get the lighter partners
# (sky/yellow), light the darker (blue/vermillion). Attribute-only slots and
# the neutral dim slots keep their look.
_CB_DARK = {
    "heading": _OI_SKY,
    "inline_code": _OI_YELLOW,
    "code_border": _OI_SKY,
    "bullet": _OI_ORANGE,
    "blockquote": _OI_SKY,
    "link": _OI_SKY,
    "error": _OI_VERMILLION,
}
_CB_LIGHT = {
    "heading": _OI_BLUE,
    "inline_code": _OI_VERMILLION,
    "code_border": _OI_BLUE,
    "bullet": _OI_ORANGE,
    "blockquote": _OI_BLUE,
    "link": _OI_BLUE,
    "error": _OI_VERMILLION,
}


def _cb_slots(spec: ThemeSpec) -> dict[str, Style]:
    remap = _CB_LIGHT if spec.name == "light" else _CB_DARK
    out = {}
    for slot, style in spec.slots.items():
        if slot in remap:
            out[slot] = replace(style, rgb=remap[slot],
                                fg16=_OI_FG16[remap[slot]])
        else:
            out[slot] = style
    return out


def detect_depth(env: dict | None = None) -> int:
    """Terminal color depth: 2^24 (truecolor), 256, or 16."""
    env = os.environ if env is None else env
    ct = env.get("COLORTERM", "").lower()
    if "truecolor" in ct or "24bit" in ct:
        return 1 << 24
    term = env.get("TERM", "")
    if "256" in term or term.startswith(("iterm", "xterm", "screen", "tmux")):
        return 256
    return 16


@dataclass(frozen=True)
class Theme:
    """Depth-resolved theme: one SGR prefix per slot ('' when color is off)."""

    name: str
    colorblind: bool
    sgr: dict[str, str] = field(repr=False)
    reset: str = RESET
    code_theme: str = "monokai"
    ptk_toolbar: str | None = None

    def __getattr__(self, slot: str) -> str:  # theme.heading -> its SGR prefix
        try:
            return self.sgr[slot]
        except KeyError:
            raise AttributeError(slot) from None

    def paint(self, slot: str, text: str) -> str:
        pre = self.sgr.get(slot, "")
        return f"{pre}{text}{self.reset}" if pre else text

    def rich_theme(self):
        """Build the matching rich Theme (lazy import; rich mode only)."""
        from rich.style import Style as RStyle
        from rich.theme import Theme as RTheme

        spec = THEMES[self.name]
        slots = _cb_slots(spec) if self.colorblind else spec.slots

        def conv(slot: str) -> RStyle:
            s = slots[slot]
            color = None
            if s.rgb is not None:
                color = "#%02x%02x%02x" % s.rgb
            elif s.fg16 is not None:
                base = (
                    "black red green yellow blue magenta cyan white".split()
                )
                idx = s.fg16 % 10
                color = ("bright_" if s.fg16 >= 90 else "") + base[idx]
            return RStyle(
                color=color, bold=s.bold or None, dim=s.dim or None,
                italic=s.italic or None, underline=s.underline or None,
            )

        return RTheme(
            {
                "markdown.h1": conv("heading"),
                "markdown.h2": conv("heading"),
                "markdown.h3": conv("heading"),
                "markdown.h4": conv("heading"),
                "markdown.h5": conv("heading"),
                "markdown.h6": conv("heading"),
                "markdown.strong": conv("bold"),
                "markdown.em": conv("italic"),
                "markdown.code": conv("inline_code"),
                "markdown.block_quote": conv("blockquote"),
                "markdown.item.bullet": conv("bullet"),
                "markdown.item.number": conv("bullet"),
                "markdown.link": conv("link"),
                "markdown.link_url": conv("link"),
                "markdown.hr": conv("hr"),
                "markdown.code_block": conv("code_block"),
            }
        )


def resolve_theme(
    name: str = "dark",
    *,
    colorblind: bool = False,
    color: bool = True,
    depth: int | None = None,
) -> Theme:
    """Resolve a theme name to depth-appropriate SGR prefixes.

    ``color=False`` (non-TTY / NO_COLOR) empties every slot so callers can
    style unconditionally. Unknown names raise ValueError naming the options.
    """
    spec = _USER_THEMES.get(name) or THEMES.get(name)
    if spec is None:
        raise ValueError(
            f"unknown theme {name!r} (choose from: {', '.join(list_themes())})"
        )
    slots = _cb_slots(spec) if colorblind else spec.slots
    if not color:
        sgr = {slot: "" for slot in _SLOTS}
        reset = ""
    else:
        d = detect_depth() if depth is None else depth
        sgr = {slot: _sgr(slots[slot], d) for slot in _SLOTS}
        reset = RESET
    code_theme = spec.code_theme
    if colorblind and spec.code_theme_cb:
        code_theme = spec.code_theme_cb
    return Theme(
        name=spec.name,
        colorblind=colorblind,
        sgr=sgr,
        reset=reset,
        code_theme=code_theme,
        ptk_toolbar=spec.ptk_toolbar,
    )
