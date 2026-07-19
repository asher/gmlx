#!/usr/bin/env python3
"""`gmlx.spinner.Spinner`: TTY-gated animation, elapsed suffix, clean shutdown.
CPU-only - a StringIO stands in for the terminal; a subclass fakes ``isatty``."""
from __future__ import annotations

import io
import time

import pytest

from gmlx import spinner


class _FakeTTY(io.StringIO):
    def isatty(self):                     # StringIO.isatty() is False; force the TTY path
        return True


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
        time.sleep(0.25)                  # let a few frames land
    assert s._thread is not None and not s._thread.is_alive()   # joined on exit
    out = buf.getvalue()
    assert "loading" in out
    assert "\r" in out                    # animated in place
    assert any(f in out for f in spinner._FRAMES)
    assert "0s" in out                    # elapsed suffix


def test_tty_clears_line_on_exit():
    buf = _FakeTTY()
    with spinner.Spinner("x", stream=buf):
        time.sleep(0.12)
    assert buf.getvalue().endswith("\r\x1b[K")   # line cleared after the block


def test_tty_no_elapsed_when_disabled():
    buf = _FakeTTY()
    with spinner.Spinner("loading", stream=buf, show_elapsed=False):
        time.sleep(0.12)
    # The label animates but no "<N>s" elapsed token is appended.
    assert "loading" in buf.getvalue() and "0s" not in buf.getvalue()


def test_exit_joins_thread_on_exception():
    buf = _FakeTTY()
    s = spinner.Spinner("x", stream=buf)
    with pytest.raises(ValueError):
        with s:
            time.sleep(0.12)
            raise ValueError("boom")
    assert not s._thread.is_alive()              # thread joined even on exception
    assert buf.getvalue().endswith("\r\x1b[K")   # and the line cleared


def test_println_on_tty_clears_frame_and_emits_line():
    buf = _FakeTTY()
    with spinner.Spinner("loading", stream=buf) as s:
        time.sleep(0.12)
        s.println("WARNING: odd tensor")
        time.sleep(0.12)                         # animation resumes after
    out = buf.getvalue()
    assert "\r\x1b[KWARNING: odd tensor\n" in out
    after = out.split("WARNING: odd tensor\n", 1)[1]
    assert any(f in after for f in spinner._FRAMES)   # frames landed after the line


def test_println_off_tty_is_plain_line():
    buf = io.StringIO()
    with spinner.Spinner("x", stream=buf) as s:
        s.println("WARNING: odd tensor")
    out = buf.getvalue()
    assert "WARNING: odd tensor\n" in out
    assert "\x1b" not in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
