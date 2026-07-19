"""Per-block streaming markdown rendering for the chat TUI.

The answer channel streams token-by-token; markdown structure only settles at
block boundaries. ``StreamRenderer`` therefore segments the stream into
top-level blocks (``BlockBuffer``), live-repaints only the in-progress block in
place (throttled), and prints each completed block's final render permanently -
native scrollback is preserved (no alternate screen, no full-screen app).

Three modes behind one interface:
  plain - raw pass-through (non-TTY / NO_COLOR fallback),
  lite  - hand-rolled ANSI markdown (zero deps),
  rich  - per-block ``rich.Markdown`` (tables, syntax highlighting; optional dep).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
import unicodedata

from .theme import Theme, resolve_theme

_rich_ok: bool | None = None


def rich_available() -> bool:
    global _rich_ok
    if _rich_ok is None:
        try:
            import rich.markdown  # noqa: F401

            _rich_ok = True
        except ImportError:
            _rich_ok = False
    return _rich_ok


def resolve_render_mode(
    requested: str | None, *, tty: bool, color: bool
) -> tuple[str, str | None]:
    """Pick the effective mode; returns ``(mode, note)`` with a user-facing
    note when the request was downgraded."""
    req = requested or "auto"
    if req == "auto":
        if tty and color and rich_available():
            return "rich", None
        if tty and color:
            return "lite", None
        return "plain", None
    if req == "plain":
        return "plain", None
    if not (tty and color):
        return "plain", f"render={req} needs a color terminal; using plain"
    if req == "rich" and not rich_available():
        return "lite", "rich not installed (pip install 'gmlx[chat]'); using lite"
    return req, None


# -- block segmentation --------------------------------------------------------

_FENCE_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s")
_HR_RE = re.compile(r"^ {0,3}((-\s*){3,}|(\*\s*){3,}|(_\s*){3,})$")


class BlockBuffer:
    """Incremental top-level markdown block segmentation (line-oriented).

    ``feed`` returns the sources of newly COMPLETED blocks; ``current`` is the
    in-progress block including the partial last line (for live painting);
    ``flush`` returns the unfinished tail at end-of-reply.
    """

    def __init__(self):
        self._lines: list[str] = []  # complete lines of the current block
        self._partial = ""           # text after the last newline
        self._fence: str | None = None  # opening fence marker inside a fence

    @property
    def current(self) -> str:
        return "".join(self._lines) + self._partial

    def feed(self, text: str) -> list[str]:
        done: list[str] = []
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._line(line + "\n", done)
        return done

    def flush(self) -> str:
        src = self.current
        self._lines, self._partial, self._fence = [], "", None
        return src

    # one complete line
    def _line(self, line: str, done: list[str]) -> None:
        stripped = line.strip()
        if self._fence is not None:
            self._lines.append(line)
            if stripped.startswith(self._fence) and stripped == stripped[0] * len(
                stripped
            ):
                self._complete(done)  # closing fence
            return
        m = _FENCE_RE.match(line.rstrip("\n"))
        if m:
            if self._lines:  # fence interrupts a paragraph/list
                self._complete(done)
            self._fence = m.group(2)
            self._lines.append(line)
            return
        if not stripped:
            if self._lines:
                self._complete(done)
            return  # collapse extra blank lines
        if _HEADING_RE.match(line) or _HR_RE.match(stripped):
            if self._lines:
                self._complete(done)
            self._lines.append(line)
            self._complete(done)  # single-line block
            return
        self._lines.append(line)

    def _complete(self, done: list[str]) -> None:
        if self._lines:
            done.append("".join(self._lines))
        self._lines = []
        self._fence = None


# -- lite renderer --------------------------------------------------------------

_INLINE_RE = re.compile(
    r"(?P<code>`+)(?P<code_t>.+?)(?P=code)"
    r"|\*\*(?P<bold>[^*]+)\*\*"
    r"|__(?P<bold2>[^_]+)__"
    r"|\*(?P<ital>[^*\s][^*]*)\*"
    r"|(?<![\w`])_(?P<ital2>[^_\s][^_]*)_(?![\w`])"
    r"|\[(?P<lt>[^\]]+)\]\((?P<lu>[^)\s]+)\)"
)

_BULLET_RE = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d{1,3}[.)])\s+(.*)$")
_QUOTE_RE = re.compile(r"^ {0,3}>\s?(.*)$")


def _cw(ch: str) -> int:
    """Display columns of one character."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _visible_width(text: str) -> int:
    return sum(_cw(c) for c in text)


def _inline_runs(text: str, theme: Theme) -> list[tuple[str, str]]:
    """Split a line into ``(text, sgr_prefix)`` runs by inline markdown."""
    runs: list[tuple[str, str]] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            runs.append((text[pos : m.start()], ""))
        if m.group("code_t") is not None:
            runs.append((m.group("code_t"), theme.inline_code))
        elif m.group("bold") is not None or m.group("bold2") is not None:
            runs.append((m.group("bold") or m.group("bold2"), theme.bold))
        elif m.group("ital") is not None or m.group("ital2") is not None:
            runs.append((m.group("ital") or m.group("ital2"), theme.italic))
        elif m.group("lt") is not None:
            runs.append((m.group("lt"), theme.link))
            runs.append((f" ({m.group('lu')})", theme.stat))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], ""))
    return runs


def _wrap_runs(
    runs: list[tuple[str, str]],
    width: int,
    theme: Theme,
    *,
    first_prefix: str = "",
    prefix: str = "",
    prefix_sgr: str = "",
) -> list[str]:
    """Greedy word-wrap styled runs to ``width`` display columns."""
    words: list[tuple[str, str]] = []
    for text, sgr in runs:
        for piece in re.split(r"(\s+)", text):
            if piece:
                words.append((piece, sgr))

    lines: list[str] = []
    cur: list[str] = []
    cur_w = _visible_width(first_prefix)
    lead = first_prefix

    def emit():
        nonlocal cur, cur_w, lead
        painted = (
            f"{prefix_sgr}{lead}{theme.reset}" if prefix_sgr and lead else lead
        )
        lines.append(painted + "".join(cur))
        cur, cur_w, lead = [], _visible_width(prefix), prefix

    def push(word: str, sgr: str) -> None:
        nonlocal cur_w
        cur.append(f"{sgr}{word}{theme.reset}" if sgr else word)
        cur_w += _visible_width(word)

    for word, sgr in words:
        if word.isspace():
            if cur:
                cur.append(" ")
                cur_w += 1
            continue
        w = _visible_width(word)
        if cur and cur_w + w > width:
            if cur[-1] == " ":
                cur.pop()
                cur_w -= 1
            emit()
        if w > width:  # unbreakable overlong word (CJK, URLs): hard-split
            piece = ""
            for ch in word:
                if cur_w + _visible_width(piece + ch) > width:
                    push(piece, sgr)
                    emit()
                    piece = ""
                piece += ch
            if piece:
                push(piece, sgr)
            continue
        push(word, sgr)
    if cur or not lines:
        if cur and cur[-1] == " ":
            cur.pop()
        emit()
    return lines


def _render_lite(src: str, width: int, theme: Theme) -> list[str]:
    raw_lines = src.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    if not raw_lines:
        return []

    m = _FENCE_RE.match(raw_lines[0])
    if m:
        return _render_lite_fence(raw_lines, m.group(3).strip(), width, theme)

    first = raw_lines[0]
    if _HEADING_RE.match(first):
        level = len(first) - len(first.lstrip("#"))
        text = first.strip().lstrip("#").strip()
        line = theme.paint("heading", text)
        if level <= 2:
            under = theme.paint("hr", "─" * min(width, max(4, _visible_width(text))))
            return [line, under]
        return [line]
    if _HR_RE.match(first.strip()):
        return [theme.paint("hr", "─" * max(4, width // 2))]

    out: list[str] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            out.extend(_wrap_runs(_inline_runs(" ".join(para), theme), width, theme))
            para.clear()

    for line in raw_lines:
        line = line.rstrip("\n")
        q = _QUOTE_RE.match(line)
        if q:
            flush_para()
            out.extend(
                _wrap_runs(
                    _inline_runs(q.group(1), theme),
                    width,
                    theme,
                    first_prefix="│ ",
                    prefix="│ ",
                    prefix_sgr=theme.blockquote,
                )
            )
            continue
        b = _BULLET_RE.match(line)
        o = _ORDERED_RE.match(line)
        if b:
            flush_para()
            indent, _, rest = b.groups()
            marker = "• "
            pad = " " * len(indent)
            out.extend(
                _wrap_runs(
                    _inline_runs(rest, theme),
                    width,
                    theme,
                    first_prefix=f"{pad}{theme.paint('bullet', marker)}",
                    prefix=pad + "  ",
                )
            )
            continue
        if o:
            flush_para()
            indent, num, rest = o.groups()
            pad = " " * len(indent)
            out.extend(
                _wrap_runs(
                    _inline_runs(rest, theme),
                    width,
                    theme,
                    first_prefix=f"{pad}{theme.paint('bullet', num + ' ')}",
                    prefix=pad + " " * (len(num) + 1),
                )
            )
            continue
        if line.lstrip().startswith("|"):  # table rows: never reflow
            flush_para()
            out.extend(_wrap_runs(_inline_runs(line, theme), width, theme))
            continue
        para.append(line.strip())
    flush_para()
    return out


def _render_lite_fence(
    raw_lines: list[str], lang: str, width: int, theme: Theme
) -> list[str]:
    body = raw_lines[1:]
    # Drop the closing fence when present (an unterminated fence streams live).
    if body:
        close = _FENCE_RE.match(body[-1])
        if close and not close.group(3).strip():
            body = body[:-1]
    tag = f"─ {lang} " if lang else "─"
    head = "┌" + tag + "─" * max(0, width - 1 - _visible_width(tag))
    out = [theme.paint("code_border", head)]
    border = theme.paint("code_border", "│ ")
    for line in body:
        out.append(border + theme.paint("code_block", line.rstrip("\n")))
    out.append(theme.paint("code_border", "└" + "─" * (width - 1)))
    return out


# -- rich renderer ---------------------------------------------------------------


class _RichBackend:
    """Cached rich Console keyed on width; captures per-block renders."""

    def __init__(self, theme: Theme):
        self._theme = theme
        self._console = None
        self._width = None

    def render(self, src: str, width: int) -> list[str]:
        from rich.console import Console
        from rich.markdown import Markdown

        if self._console is None or self._width != width:
            depth_ok = "truecolor" in os.environ.get("COLORTERM", "").lower()
            self._console = Console(
                width=width,
                force_terminal=True,
                color_system="truecolor" if depth_ok else "256",
                theme=self._theme.rich_theme(),
                highlight=False,
                emoji=False,
            )
            self._width = width
        with self._console.capture() as cap:
            self._console.print(
                Markdown(src, code_theme=self._theme.code_theme), end=""
            )
        lines = cap.get().split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        return [ln.rstrip() for ln in lines]


# -- streaming renderer -----------------------------------------------------------


def _term_size():
    return shutil.get_terminal_size(fallback=(80, 24))


# Model output is untrusted terminal input: strip ESC + C0 controls (keep
# \n and \t) so an emitted ANSI/OSC sequence cannot restyle, retitle, or
# clear the user's terminal through the renderer. Stripped ESCs leave their
# printable tail visible (e.g. "[31m"), which is the honest rendering.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class StreamRenderer:
    """Answer-channel sink: repaint the current block in place, print completed
    blocks permanently. Plain mode passes text straight through."""

    def __init__(
        self,
        mode: str,
        theme: Theme | None = None,
        *,
        write=None,
        size_fn=None,
        clock=None,
        min_repaint_s: float = 0.04,
    ):
        self.mode = mode
        self.theme = theme or resolve_theme(color=(mode != "plain"))
        self._w = write or (lambda s: (sys.stdout.write(s), sys.stdout.flush()))
        self._size = size_fn or _term_size
        self._clock = clock or time.monotonic
        self._min_repaint = min_repaint_s
        self._buf = BlockBuffer()
        self._painted = 0          # terminal lines the live block occupies
        self._frozen = False       # oversize/resize: current block appends raw
        self._raw_emitted = 0      # chars of the current block already raw-written
        self._last_paint = 0.0
        self._width = None
        self._backend = _RichBackend(self.theme) if mode == "rich" else None

    # -- public ------------------------------------------------------------------

    def feed(self, text: str) -> None:
        if not text:
            return
        text = _CONTROL_CHARS.sub("", text)
        if self.mode == "plain":
            self._w(text)
            return
        for src in self._buf.feed(text):
            self._finish_block(src)
        self._paint_live()

    def finalize(self) -> None:
        """End of reply: last unthrottled paint of the tail; cursor ends on a
        fresh line below the rendered output."""
        if self.mode == "plain":
            return
        tail = self._buf.flush()
        if self._frozen:
            self._w(tail[self._raw_emitted :])
            if tail and not tail.endswith("\n"):
                self._w("\n")
            self._reset_block()
            return
        if tail.strip():
            self._repaint(self._render(tail))
        self._painted = 0

    # -- internals ----------------------------------------------------------------

    def _render(self, src: str) -> list[str]:
        width = max(20, self._size().columns - 1)
        if self._backend is not None:
            return self._backend.render(src, width)
        return _render_lite(src, width, self.theme)

    def _finish_block(self, src: str) -> None:
        if self._frozen:
            self._w(src[self._raw_emitted :])
            self._w("\n")  # block separator
            self._reset_block()
            return
        if src.strip():
            self._repaint(self._render(src))
            self._w("\n")
        self._painted = 0
        self._raw_emitted = 0

    def _paint_live(self) -> None:
        cur = self._buf.current
        if self._frozen:
            self._w(cur[self._raw_emitted :])
            self._raw_emitted = len(cur)
            return
        if not cur.strip():
            return
        now = self._clock()
        if now - self._last_paint < self._min_repaint:
            return
        self._last_paint = now
        size = self._size()
        width = max(20, size.columns - 1)
        if self._width is not None and width != self._width and self._painted:
            self._freeze(cur)
            return
        self._width = width
        lines = self._render(cur)
        if len(lines) >= max(4, size.lines - 2):
            self._freeze(cur)
            return
        self._repaint(lines)

    def _repaint(self, lines: list[str]) -> None:
        out = []
        if self._painted:
            out.append(f"\x1b[{self._painted}A")
        for ln in lines:
            out.append("\x1b[2K" + ln + "\n")
        extra = self._painted - len(lines)
        if extra > 0:
            out.append("\x1b[2K\n" * extra)
            out.append(f"\x1b[{extra}A")
        self._w("".join(out))
        self._painted = len(lines)

    def _freeze(self, cur: str) -> None:
        """Oversize or resized mid-block: swap the painted render for the raw
        source and append-only from here to the end of this block."""
        head, _, partial = cur.rpartition("\n")
        self._repaint(head.split("\n") if head else [])
        if partial:
            self._w(partial)  # stays open; later appends continue the line
        self._frozen = True
        self._raw_emitted = len(cur)
        self._painted = 0

    def _reset_block(self) -> None:
        self._frozen = False
        self._raw_emitted = 0
        self._painted = 0
        self._width = None
