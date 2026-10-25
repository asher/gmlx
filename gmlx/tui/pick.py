"""An inline picker: rows in groups, then a configure screen for the chosen
row's quality rungs and toggles.

The widget knows nothing about models. The caller hands it ``Group`` rows
with their ``Rung`` and ``Toggle`` choices, a ``rung_label`` callback that
words each rung for the toggles that are on, and a ``footer`` callback
that prints the live command. ``pick`` runs it on prompt_toolkit and
returns the ``Plan`` the user chose, or None when they quit. ``Picker``
is the state and the rendering, with no terminal, so tests drive it
directly. ASCII only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

RESIDENT = "resident"
STREAMS = "streams"

LIST_HINTS = "up/down move  enter choose  type to filter  esc quit"
CONFIGURE_HINTS = "up/down move  space choose or toggle  enter download  left back"
MAX_ROWS = 24           # the tallest the widget draws, in lines
MIN_LIST_LINES = 3


class PickerUnavailable(Exception):
    """No terminal to draw on."""


@dataclass(frozen=True)
class Rung:
    id: str
    tier: str                 # the shown name: a tier, "compact", or a quant
    quant: str                # dim beside the name when it differs
    size_label: str
    state: str                # RESIDENT or STREAMS; the caller never passes a rung that cannot run
    on_disk: bool = False
    default: bool = False


@dataclass(frozen=True)
class Toggle:
    id: str
    label: str
    size_label: str
    on: bool                  # the default plan's value on a resident rung
    on_when_streaming: bool = False
    note: str = ""
    streaming_note: str = ""
    on_disk: bool = False
    hide_when_streaming: bool = False


@dataclass(frozen=True)
class Row:
    id: str
    title: str
    params: str
    tags: tuple
    size_label: str
    blurb: str
    speed: str
    note: str
    license: str
    dim_note: str             # "" or why the row is dim (needs N GB free)
    rungs: tuple
    default_rung: str         # the Rung id the default mark sits on
    toggles: tuple
    use: str = ""             # a few words: what the model is good for
    explain: str | None = None   # replaces the picker's explain line under Quality; "" shows none


@dataclass(frozen=True)
class Group:
    title: str
    rows: tuple


@dataclass(frozen=True)
class Plan:
    row: Row
    rung: Rung
    toggles: tuple            # the ids that are on


def _clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[:max(0, width - 3)] + "..." if width > 3 else text[:width]


def _pad(text: str, width: int) -> str:
    return _clip(text, width).ljust(width)


def _columns(width: int) -> dict:
    """Column widths for the list screen at ``width``."""
    wide = width >= 100
    return {"title": 36 if wide else 24, "size": 9, "images": 6}


class Picker:
    """The picker's state and rendering, with no terminal attached."""

    def __init__(self, groups, *, ram_label: str, footer, rung_label, explain_line: str,
                 empty_line: str, no_match_line: str, other_line: str = ""):
        self.groups = tuple(groups)
        self.ram_label = ram_label
        self.footer = footer
        self.rung_label = rung_label
        self.explain_line = explain_line
        self.empty_line = empty_line
        self.no_match_line = no_match_line
        self.other_line = other_line      # a dim line above the list screen's key hints
        self.filter = ""
        self.cursor = 0
        self.screen = "list"
        self.row: Row | None = None
        self.rung_idx = 0
        self.ccursor = 0
        self._sets: dict = {}          # (row id, streaming) -> set of toggle ids on

    # -- the list screen ----------------------------------------------------

    def _matches(self, row: Row) -> bool:
        f = self.filter.lower()
        if not f:
            return True
        hay = " ".join([row.id, row.title, row.params, row.use, *row.tags]).lower()
        return f in hay

    def groups_visible(self) -> list:
        out = []
        for g in self.groups:
            rows = [r for r in g.rows if self._matches(r)]
            if rows:
                out.append((g.title, rows))
        return out

    def visible(self) -> list:
        return [r for _t, rows in self.groups_visible() for r in rows]

    def _clamp(self) -> None:
        n = len(self.visible())
        self.cursor = 0 if n == 0 else max(0, min(self.cursor, n - 1))

    def current(self) -> Row | None:
        rows = self.visible()
        self._clamp()
        return rows[self.cursor] if rows else None

    def move(self, delta: int) -> None:
        self.cursor += delta
        self._clamp()

    def jump(self, n: int) -> None:
        if not self.filter and 1 <= n <= len(self.visible()):
            self.cursor = n - 1

    def _refilter(self, text: str) -> None:
        """Change the filter, keeping the cursor on its row when that row
        still shows, else on the first match."""
        cur = self.current()
        self.filter = text
        rows = self.visible()
        self.cursor = rows.index(cur) if cur in rows else 0

    def type(self, ch: str) -> None:
        self._refilter(self.filter + ch)

    def backspace(self) -> None:
        self._refilter(self.filter[:-1])

    def clear_filter(self) -> None:
        self._refilter("")

    def select(self) -> bool:
        row = self.current()
        if row is None:
            return False
        self.row = row
        ids = [r.id for r in row.rungs]
        self.rung_idx = ids.index(row.default_rung) if row.default_rung in ids else 0
        self.ccursor = self.rung_idx
        self.screen = "configure"
        return True

    def back(self) -> None:
        self.screen = "list"

    # -- the configure screen ------------------------------------------------

    def rung(self) -> Rung:
        return self.row.rungs[self.rung_idx]

    def streaming(self) -> bool:
        return self.rung().state == STREAMS

    def _set(self) -> set:
        key = (self.row.id, self.streaming())
        if key not in self._sets:
            on = {t.id for t in self.row.toggles
                  if (t.on_when_streaming if self.streaming() else t.on)}
            self._sets[key] = on
        return self._sets[key]

    def toggles_shown(self) -> list:
        if self.streaming():
            return [t for t in self.row.toggles if not t.hide_when_streaming]
        return list(self.row.toggles)

    def on_ids(self) -> tuple:
        shown = {t.id for t in self.toggles_shown()}
        return tuple(t.id for t in self.row.toggles if t.id in self._set() and t.id in shown)

    def _items(self) -> list:
        return [("rung", r) for r in self.row.rungs] + [("toggle", t) for t in self.toggles_shown()]

    def cmove(self, delta: int) -> None:
        """Move the configure cursor; moving never changes the chosen rung."""
        items = self._items()
        self.ccursor = max(0, min(self.ccursor + delta, len(items) - 1))

    def toggle(self) -> None:
        """Space: choose the rung under the cursor, or flip the toggle."""
        items = self._items()
        kind, item = items[self.ccursor]
        if kind == "rung":
            self.rung_idx = self.row.rungs.index(item)
            # The toggle list may have shrunk: keep the cursor on an item.
            self.ccursor = min(self.ccursor, len(self._items()) - 1)
            return
        on = self._set()
        if item.id in on:
            on.discard(item.id)
        else:
            on.add(item.id)

    def choose(self) -> Plan:
        """Enter: a rung under the cursor is chosen first; then the plan."""
        kind, item = self._items()[self.ccursor]
        if kind == "rung":
            self.rung_idx = self.row.rungs.index(item)
        return self.plan()

    def plan(self) -> Plan:
        return Plan(self.row, self.rung(), self.on_ids())

    # -- rendering --------------------------------------------------------------

    def render(self, width: int, rows: int) -> list:
        """Style-and-text tuples for the current screen, ``rows`` lines at most."""
        rows = max(8, min(rows, MAX_ROWS))
        if self.screen == "list":
            return self._render_list(width, rows)
        return self._render_configure(width, rows)

    @staticmethod
    def _window(lines: list, area: int, cursor_line: int) -> list:
        """``lines`` (each ``(style, text, is_row)``) clipped to ``area``
        with the cursor line visible and ``... N more`` markers."""
        if len(lines) <= area:
            return [(s, t) for s, t, _r in lines]
        inner = area - 2
        start = max(0, min(cursor_line - inner // 2, len(lines) - inner))
        end = start + inner
        out = []
        above = sum(1 for _s, _t, r in lines[:start] if r)
        below = sum(1 for _s, _t, r in lines[end:] if r)
        out.append(("class:pick.dim", f"  ... {above} more" if above else ""))
        out.extend((s, t) for s, t, _r in lines[start:end])
        out.append(("class:pick.dim", f"  ... {below} more" if below else ""))
        return out

    def _render_list(self, width: int, rows: int) -> list:
        col = _columns(width)
        out = [("class:pick.header", _clip(self.ram_label, width)), ("", "\n")]
        if self.filter:
            out.append(("class:pick.filter", _clip(f"filter: {self.filter}", width)))
        else:
            head = (f"     {_pad('NAME', col['title'])} {'DOWNLOAD':>{col['size']}}  "
                    f"{_pad('IMAGES', col['images'])}  GOOD FOR")
            out.append(("class:pick.dim", _clip(head, width)))
        out.append(("", "\n"))
        area = max(MIN_LIST_LINES, rows - 8 - (1 if self.other_line else 0))
        groups = self.groups_visible()
        lines = []
        if not groups:
            text = (self.no_match_line.format(filter=self.filter) if self.filter
                    else self.empty_line)
            lines.append(("", "  " + _clip(text, width - 2), True))
            cursor_line = 0
        else:
            self._clamp()
            n = 0
            cursor_line = 0
            for title, grp in groups:
                lines.append(("class:pick.group", _clip(title, width), False))
                for row in grp:
                    here = n == self.cursor
                    if here:
                        cursor_line = len(lines)
                    num = "  " if self.filter else f"{n + 1:>2}"
                    # A row that cannot download here shows why in place of its use.
                    images = "yes" if "images" in row.tags else ""
                    text = (f"{'>' if here else ' '} {num} {_pad(row.id, col['title'])} "
                            f"{row.size_label:>{col['size']}}  "
                            f"{_pad(images, col['images'])}  "
                            f"{row.dim_note or row.use}")
                    style = ("class:pick.cursor" if here
                             else "class:pick.dim" if row.dim_note else "")
                    lines.append((style, _clip(text, width), True))
                    n += 1
        for style, text in self._window(lines, area, cursor_line):
            out.append((style, text))
            out.append(("", "\n"))
        out.append(("", "\n"))
        row = self.current()
        detail = ["", "", ""]
        if row is not None:
            detail = [f"  {row.blurb} {row.license}".rstrip(), f"  {row.speed}",
                      "  " + "  ".join(
                          f"{r.tier} {r.size_label}" + (" (default)" if r.default else "")
                          for r in row.rungs)]
        for line in detail:
            out.append(("", _clip(line, width)))
            out.append(("", "\n"))
        out.append(("", "\n"))
        if self.other_line:
            out.append(("class:pick.dim", _clip(self.other_line, width)))
            out.append(("", "\n"))
        out.append(("class:pick.dim", _clip(LIST_HINTS, width)))
        return out

    def _render_configure(self, width: int, rows: int) -> list:
        row = self.row
        out = [("class:pick.header",
                _clip(f"{row.title}   {row.params}   {row.license}".rstrip(), width)),
               ("", "\n"), ("", "\n")]
        on_ids = self.on_ids()
        streaming = self.streaming()
        items = self._items()
        lines = []
        lines.append(("class:pick.group", "Quality", False))
        for i, (kind, item) in enumerate(items):
            here = i == self.ccursor
            style = "class:pick.cursor" if here else ""
            if kind == "rung":
                label, note = self.rung_label(row, item, on_ids)
                marks = [m for m, on in (("default", item.default), ("on disk", item.on_disk)) if on]
                quant = item.quant if item.quant != item.tier else ""
                text = (f"{'>' if here else ' '} ({'*' if item is self.rung() else ' '}) "
                        f"{_pad(item.tier, 10)} {_pad(quant, 10)} {item.size_label:>9}  {label}")
                if marks:
                    text += "   " + ", ".join(marks)
                if note:
                    text += f"  ({note})"
                lines.append((style, _clip(text, width), True))
                if i == len(row.rungs) - 1:
                    explain = row.explain if row.explain is not None else self.explain_line
                    if explain:
                        lines.append(("class:pick.dim", _clip("  " + explain, width), False))
                    lines.append(("class:pick.group", "Extras", False))
            else:
                on = item.id in on_ids
                note = item.streaming_note if streaming else item.note
                text = (f"{'>' if here else ' '} [{'x' if on else ' '}] {_pad(item.label, 18)} "
                        f"{item.size_label:>9}")
                if item.on_disk:
                    text += "  on disk"
                if note:
                    text += f"  {note}"
                lines.append((style, _clip(text, width), True))
        if not self.toggles_shown():
            lines = [ln for ln in lines if ln[1] != "Extras"]
        area = max(MIN_LIST_LINES, rows - 8)
        cursor_line = next((i for i, ln in enumerate(lines) if ln[0] == "class:pick.cursor"), 0)
        for style, text in self._window(lines, area, cursor_line):
            out.append((style, text))
            out.append(("", "\n"))
        out.append(("", "\n"))
        footer = list(self.footer(row, self.rung(), on_ids))[:2] + ["", ""]
        out.append(("", _clip(footer[0], width)))
        out.append(("", "\n"))
        out.append(("class:pick.cmd", _clip(footer[1], width)))
        out.append(("", "\n"))
        out.append(("", "\n"))
        out.append(("class:pick.dim", _clip(CONFIGURE_HINTS, width)))
        return out


def plain_text(fragments) -> str:
    return "".join(text for _style, text in fragments)


def _style(color: bool):
    from prompt_toolkit.styles import Style

    if not color:
        return Style([])
    return Style([("pick.header", "bold"), ("pick.group", "bold"),
                  ("pick.cursor", "bold ansicyan"), ("pick.dim", "#888888"),
                  ("pick.cmd", "ansiyellow"), ("pick.filter", "bold")])


def pick(groups, *, ram_label: str, footer, rung_label, explain_line: str,
         empty_line: str, no_match_line: str, other_line: str = "", color=None, input=None,
         output=None):
    """Run the picker; the chosen ``Plan``, or None on quit. Raises
    PickerUnavailable off a terminal when no input and output are given."""
    if input is None and output is None:
        try:
            tty = sys.stdin.isatty() and sys.stdout.isatty()
        except Exception:
            tty = False
        if not tty:
            raise PickerUnavailable("gmlx pull needs a terminal to pick a model")
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    if color is None:
        from gmlx.tui.reasoning import want_color
        color = want_color()
    state = Picker(groups, ram_label=ram_label, footer=footer, rung_label=rung_label,
                   explain_line=explain_line, empty_line=empty_line, no_match_line=no_match_line,
                   other_line=other_line)
    app_box: dict = {}

    def size():
        try:
            s = app_box["app"].output.get_size()
            return s.columns, s.rows
        except Exception:
            return 80, 24

    def text():
        w, h = size()
        return state.render(w, min(h - 1, MAX_ROWS))

    kb = KeyBindings()
    on_list = lambda: state.screen == "list"  # noqa: E731

    @kb.add("up")
    @kb.add("c-p")
    def _up(event):
        state.move(-1) if on_list() else state.cmove(-1)

    @kb.add("down")
    @kb.add("c-n")
    def _down(event):
        state.move(1) if on_list() else state.cmove(1)

    @kb.add("enter")
    def _enter(event):
        if on_list():
            state.select()
        else:
            event.app.exit(result=state.choose())

    @kb.add("right")
    def _right(event):
        if on_list():
            state.select()

    @kb.add("left")
    def _left(event):
        if not on_list():
            state.back()

    @kb.add("escape", eager=True)
    def _escape(event):
        if not on_list():
            state.back()
        elif state.filter:
            state.clear_filter()
        else:
            event.app.exit(result=None)

    @kb.add("backspace")
    @kb.add("c-h")
    def _backspace(event):
        if on_list():
            state.backspace()

    @kb.add("c-u")
    def _clear(event):
        if on_list():
            state.clear_filter()

    @kb.add(" ")
    def _space(event):
        if on_list():
            state.type(" ")
        else:
            state.toggle()

    @kb.add("c-c")
    @kb.add("c-d")
    @kb.add(Keys.SIGINT)
    def _quit(event):
        event.app.exit(result=None)

    @kb.add(Keys.Any)
    def _any(event):
        data = event.data
        if len(data) != 1 or not data.isprintable():
            return
        if on_list():
            if data.isdigit() and not state.filter:
                state.jump(int(data))
            else:
                state.type(data)
        elif data == "j":
            state.cmove(1)
        elif data == "k":
            state.cmove(-1)
        elif data == "q":
            event.app.exit(result=None)

    window = Window(FormattedTextControl(text, show_cursor=False),
                    always_hide_cursor=True, wrap_lines=False)
    app = Application(layout=Layout(window), key_bindings=kb, style=_style(color),
                      full_screen=False, erase_when_done=True, mouse_support=False,
                      input=input, output=output)
    app.ttimeoutlen = 0.1
    app_box["app"] = app
    try:
        return app.run()
    except EOFError:
        return None
