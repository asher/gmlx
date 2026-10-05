"""Text that launch prints to your terminal.

Paths and messages that launch prints can hold names the container chose,
such as a git folder a guest-written file names. A control character in
them could move the cursor, rewrite lines above, or reach the terminal's
clipboard through OSC 52, and a format character such as U+202E reverses
the text after it, so every such line passes through :func:`printable`
first.
"""

from __future__ import annotations

import unicodedata


def _escape(ch: str) -> str:
    if ch == " ":
        return ch
    code = ord(ch)
    if 0xDC80 <= code <= 0xDCFF:
        # A byte of a name that is not UTF-8, which Python keeps as a
        # surrogate. Printed raw, a byte such as 0x9b starts a CSI sequence.
        return f"\\x{code - 0xDC00:02x}"
    if unicodedata.category(ch)[0] not in "CZ":
        return ch
    if code < 0x100:
        return f"\\x{code:02x}"
    return f"\\u{code:04x}" if code < 0x10000 else f"\\U{code:08x}"


def printable(text: object) -> str:
    """``text`` with every control, format, unassigned, private-use and
    surrogate character and every separator but the space shown as an
    escape, so it prints as one line of plain text that reads in the order
    it is stored. A byte of a name that is not UTF-8 shows as ``\\xNN``."""
    return "".join(_escape(ch) for ch in str(text))


def printable_lines(text: object) -> str:
    """Like :func:`printable`, but each newline stays, for text that launch
    itself composes from several lines."""
    return "\n".join(printable(line) for line in str(text).split("\n"))
