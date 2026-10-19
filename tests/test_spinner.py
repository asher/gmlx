#!/usr/bin/env python3
"""`gmlx.spinner.Spinner`: TTY-gated animation, elapsed suffix, clean shutdown.
CPU-only - a StringIO stands in for the terminal; a subclass fakes ``isatty``."""
from __future__ import annotations

import io
import re
import threading

import pytest

from gmlx import spinner

_FRAME = re.compile(f"[{spinner._FRAMES}] ")


class _FakeTTY(io.StringIO):
    """A terminal stand-in that wakes waiters on every write."""

    def __init__(self):
        super().__init__()
        self._cond = threading.Condition()

    def isatty(self):                     # StringIO.isatty() is False; force the TTY path
        return True

    def write(self, s):
        with self._cond:
            n = super().write(s)
            self._cond.notify_all()
        return n

    def wait_for(self, pred, timeout=30.0) -> bool:
        """Wait until ``pred(text written so far)`` holds."""
        with self._cond:
            return self._cond.wait_for(lambda: pred(self.getvalue()), timeout)

    def wait_frame(self) -> bool:
        return self.wait_for(lambda out: _FRAME.search(out) is not None)


def test_non_tty_prints_label_once_no_escapes():
    buf = io.StringIO()                   # isatty() False -> plain path
    with spinner.Spinner("loading model", stream=buf):
        pass
    out = buf.getvalue()
    assert out == "loading model\n"       # printed once, plainly
    assert "\x1b" not in out and "\r" not in out


def test_non_tty_starts_no_thread():
    buf = io.StringIO()
    with spinner.Spinner("x", stream=buf) as s:
        assert s._thread is None          # no animation thread off a TTY


def test_non_tty_update_swaps_label():
    buf = io.StringIO()
    with spinner.Spinner("first", stream=buf) as s:
        s.update("second")
    out = buf.getvalue()
    assert "first\n" in out and "second\n" in out


def test_tty_animates_with_elapsed():
    buf = _FakeTTY()
    with spinner.Spinner("loading", stream=buf, show_elapsed=True) as s:
        assert buf.wait_for(lambda out: re.search(r"loading \d+s", out))
    assert s._thread is not None and not s._thread.is_alive()   # joined on exit
    out = buf.getvalue()
    assert "\r" in out                    # animated in place
    assert _FRAME.search(out)
    assert re.search(r"loading \d+s", out)    # elapsed suffix


def test_tty_clears_line_on_exit():
    buf = _FakeTTY()
    with spinner.Spinner("x", stream=buf):
        assert buf.wait_frame()
    assert buf.getvalue().endswith("\r\x1b[K")   # line cleared after the block


def test_tty_no_elapsed_when_disabled():
    buf = _FakeTTY()
    with spinner.Spinner("loading", stream=buf, show_elapsed=False):
        assert buf.wait_for(lambda out: _FRAME.search(out) and "loading" in out)
    # The label animates but no "<N>s" elapsed token is appended.
    assert "loading" in buf.getvalue()
    assert not re.search(r" \d+s", buf.getvalue())


def test_exit_joins_thread_on_exception():
    buf = _FakeTTY()
    s = spinner.Spinner("x", stream=buf)
    with pytest.raises(ValueError):
        with s:
            assert buf.wait_frame()
            raise ValueError("boom")
    assert not s._thread.is_alive()              # thread joined even on exception
    assert buf.getvalue().endswith("\r\x1b[K")   # and the line cleared


def test_println_on_tty_clears_frame_and_emits_line():
    buf = _FakeTTY()
    def frame_after_line() -> bool:
        parts = buf.getvalue().split("WARNING: odd tensor\n", 1)
        return len(parts) == 2 and any(f in parts[1] for f in spinner._FRAMES)

    with spinner.Spinner("loading", stream=buf) as s:
        assert buf.wait_frame()
        s.println("WARNING: odd tensor")
        assert buf.wait_for(lambda out: frame_after_line()), "no frame after println"
    out = buf.getvalue()
    assert "\r\x1b[KWARNING: odd tensor\n" in out
    assert frame_after_line()                    # frames landed after the line


def test_println_off_tty_is_plain_line():
    buf = io.StringIO()
    with spinner.Spinner("x", stream=buf) as s:
        s.println("WARNING: odd tensor")
    out = buf.getvalue()
    assert "WARNING: odd tensor\n" in out
    assert "\x1b" not in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
