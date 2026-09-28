"""Text that launch prints to your terminal.

Paths and messages that launch prints can hold names the container chose,
such as a git folder a guest-written file names. A control character in
them could move the cursor, rewrite lines above, or reach the terminal's
clipboard through OSC 52, so every such line passes through
:func:`printable` first.
"""

from __future__ import annotations


def _escape(ch: str) -> str:
    code = ord(ch)
    if code < 0x20 or code == 0x7F or 0x80 <= code < 0xA0:
        return f"\\x{code:02x}"
    if 0xDC80 <= code <= 0xDCFF:
        # A byte of a name that is not UTF-8, which Python keeps as a
        # surrogate. Printed raw, a byte such as 0x9b starts a CSI sequence.
        return f"\\x{code - 0xDC00:02x}"
    if 0xD800 <= code <= 0xDFFF:
        return f"\\u{code:04x}"
    return ch


def printable(text: object) -> str:
    """``text`` with every C0 control character, including newline and
    carriage return, DEL, every C1 control character and every byte of a
    name that is not UTF-8 shown as a ``\\xNN`` escape, so it prints as one
    line of plain text."""
    return "".join(_escape(ch) for ch in str(text))


def printable_lines(text: object) -> str:
    """Like :func:`printable`, but each newline stays, for text that launch
    itself composes from several lines."""
    return "\n".join(printable(line) for line in str(text).split("\n"))
