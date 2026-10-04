"""gmlx/container/terminal.py: the input parser and the terminal relay that
sits between your terminal and ``container run -t`` or ``container exec -t``.
The relay tests use a pseudo-terminal as your terminal and read what the
relay passes on from a copy of the CLI's end."""

from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import textwrap

import pytest

from gmlx.container import terminal

START, END = terminal.PASTE_START, terminal.PASTE_END


def _joined(events):
    """The events with the data events next to each other joined, so a
    split of the input does not change what the test compares."""
    out = []
    for kind, value in events:
        if kind == "data" and out and out[-1][0] == "data":
            out[-1] = ("data", out[-1][1] + value)
        elif kind != "data" or value:
            out.append((kind, value))
    return out


def _feed(parser, *chunks):
    events = []
    for chunk in chunks:
        events += parser.feed(chunk)
    return events


# The parser

def test_ordinary_bytes_pass_unchanged_and_in_order():
    data = bytes(b for b in range(256) if b not in (0x16, 0x1b)) * 3
    parser = terminal.InputParser()
    assert _joined(parser.feed(data)) == [("data", data)]
    assert not parser.waiting()


def test_other_escape_sequences_pass_unchanged():
    data = b"\x1b[A\x1b[1;5C\x1bOP\x1b[?2004h\x1b]52;c;?\x07\x1b\x1bx\x1b[27;5;13~"
    assert _joined(terminal.InputParser().feed(data)) == [("data", data)]


def test_a_bracketed_paste_split_at_every_byte_boundary():
    stream = b"ab" + START + b"/Users/u/a b.png\n/x" + END + b"cd\x16e"
    want = [("data", b"ab"), ("paste", b"/Users/u/a b.png\n/x"), ("data", b"cd"),
            ("key", "ctrl+v"), ("data", b"\x16e")]
    for cut in range(1, len(stream)):
        parser = terminal.InputParser()
        events = _feed(parser, stream[:cut], stream[cut:])
        assert _joined(events) == want, cut
        assert not parser.waiting()


def test_a_paste_fed_one_byte_at_a_time():
    stream = START + b"x\x16y" + END
    parser = terminal.InputParser()
    events = _feed(parser, *(stream[i:i + 1] for i in range(len(stream))))
    assert _joined(events) == [("paste", b"x\x16y")]           # a ^V in a paste is no key


def test_a_cut_off_sequence_waits_for_its_rest_or_the_idle_time():
    parser = terminal.InputParser()
    assert parser.feed(b"a\x1b[11") == [("data", b"a")]
    assert parser.waiting()
    assert parser.feed(b"8;5u") == [("key", "ctrl+v"), ("data", b"\x1b[118;5u")]
    assert parser.feed(b"\x1b") == [] and parser.waiting()
    assert parser.idle() == [("data", b"\x1b")]                 # a lone Escape key
    assert not parser.waiting()


def test_a_sequence_too_long_to_be_a_key_passes_at_once():
    long = b"\x1b[" + b"1;" * terminal.SEQUENCE_MAX
    parser = terminal.InputParser()
    assert _joined(parser.feed(long)) == [("data", long)] and not parser.waiting()


def test_an_unterminated_paste_is_held_until_its_end():
    parser = terminal.InputParser()
    assert parser.feed(START + b"/a.png") == []
    assert not parser.waiting()                    # no timer ends a paste
    assert parser.feed(b" more") == []
    assert parser.end() == [("data", START + b"/a.png more")]


def test_a_start_marker_inside_a_paste_is_part_of_it():
    parser = terminal.InputParser()
    events = parser.feed(START + b"a" + START + b"b" + END + b"c")
    assert events == [("paste", b"a" + START + b"b"), ("data", b"c")]


def test_a_paste_past_the_cap_passes_through_as_it_came(monkeypatch):
    monkeypatch.setattr(terminal, "PASTE_MAX", 16)
    body = b"0123456789\x16abcdefghijklmnop"
    stream = b"<" + START + body + END + b">\x16"
    for cut in range(1, len(stream)):
        parser = terminal.InputParser()
        events = _joined(_feed(parser, stream[:cut], stream[cut:]) + parser.idle())
        assert events == [("data", stream[:-1]), ("key", "ctrl+v"), ("data", b"\x16")], cut


@pytest.mark.parametrize("seq, name", [
    (b"\x16", "ctrl+v"),
    (b"\x1b[118;5u", "ctrl+v"),                    # kitty keyboard protocol
    (b"\x1b[118;5:1u", "ctrl+v"),                  # with the press event type
    (b"\x1b[118;5:2u", "ctrl+v"),                  # a repeat
    (b"\x1b[118:86;5u", "ctrl+v"),                 # with the shifted key
    (b"\x1b[118;69u", "ctrl+v"),                   # caps lock on
    (b"\x1b[118;133u", "ctrl+v"),                  # num lock on
    (b"\x1b[27;5;118~", "ctrl+v"),                 # xterm modifyOtherKeys
    (b"\x1b[118;6u", "ctrl+shift+v"),
    (b"\x1b[118:86;6u", "ctrl+shift+v"),
    (b"\x1b[27;6;86~", "ctrl+shift+v"),
    (b"\x1bv", "alt+v"),
    (b"\x1b[118;3u", "alt+v"),
    (b"\x1b[27;3;118~", "alt+v"),
    (b"\x1b[118;5:3u", None),                      # a release
    (b"\x1b[118u", None),                          # v alone
    (b"\x1b[119;5u", None),                        # ctrl+w
    (b"\x1b[27;5;119~", None),
    (b"\x1bV", None),
])
def test_key_names(seq, name):
    assert terminal.key_name(seq) == name


@pytest.mark.parametrize("seq", [b"\x16", b"\x1b[118;5u", b"\x1b[118;5:1u",
                                 b"\x1b[27;5;118~"])
def test_the_paste_key_comes_before_its_bytes_in_every_encoding(seq):
    for cut in range(1, len(seq) + 1):
        parser = terminal.InputParser()
        events = _feed(parser, b"x" + seq[:cut], seq[cut:] + b"y")
        assert _joined(events) == [("data", b"x"), ("key", "ctrl+v"), ("data", seq + b"y")]


def test_the_paste_keys_are_the_union_of_the_clients():
    """claude-code, opencode, pi and omp paste an image on Ctrl-V, and
    hermes on Ctrl-V and Alt-V."""
    assert terminal.PASTE_KEYS == {"ctrl+v", "alt+v"}
    parser = terminal.InputParser()
    assert _joined(parser.feed(b"\x1bv")) == [("key", "alt+v"), ("data", b"\x1bv")]


def test_keys_outside_the_set_are_no_press():
    parser = terminal.InputParser(frozenset({"ctrl+v"}))
    data = b"\x1bv\x1b[118;6u"
    assert _joined(parser.feed(data)) == [("data", data)]
    parser = terminal.InputParser(frozenset({"ctrl+v", "alt+v"}))
    assert _joined(parser.feed(data)) == [("key", "alt+v"), ("data", data)]


def test_unbracketed_input_is_never_a_paste():
    """Only the paste markers make a paste, so long typed text that holds a
    path passes as it is."""
    text = b"/Users/u/a.png " + b"x" * 5000
    assert _joined(terminal.InputParser().feed(text)) == [("data", text)]


# The relay

class _Terminal:
    """A pseudo-terminal that plays your terminal: the test types into
    ``keys`` and the relay reads ``tty``."""

    def __init__(self, rows=30, cols=100):
        self.keys, self.tty = pty.openpty()
        fcntl.ioctl(self.tty, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.mode = termios.tcgetattr(self.tty)

    def close(self):
        for fd in (self.keys, self.tty):
            try:
                os.close(fd)
            except OSError:
                pass


def _read(fd, want: bytes, timeout=10.0) -> bytes:
    """Read ``fd`` until ``want`` arrived, or fail after ``timeout``."""
    got = b""
    while want not in got:
        assert select.select([fd], [], [], timeout)[0], f"waited for {want!r}, got {got!r}"
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            chunk = b""
        assert chunk, f"the end came before {want!r}, got {got!r}"
        got += chunk
    return got


def _ended(fd, timeout=10.0) -> bool:
    """Whether reading ``fd`` reaches the end of the pseudo-terminal."""
    while select.select([fd], [], [], timeout)[0]:
        try:
            if not os.read(fd, 4096):
                return True
        except OSError:
            return True
    return False


@pytest.fixture
def term():
    t = _Terminal()
    yield t
    t.close()


def _relay(term, **kw):
    """A started relay and a raw copy of the CLI's end of its terminal."""
    out_r, out_w = os.pipe()
    relay = terminal.TerminalRelay(term=term.tty, out=out_w, **kw)
    cli = os.dup(relay.slave)
    termios.tcsetattr(cli, termios.TCSANOW, terminal.raw_mode(termios.tcgetattr(cli)))
    relay.start()
    return relay, cli, (out_r, out_w)


def _close(*fds):
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def test_typing_ctrl_c_and_the_paste_key_pass_through(term):
    pressed = []
    relay, cli, pipe = _relay(term, on_key=lambda: pressed.append(True))
    try:
        assert not termios.tcgetattr(term.tty)[3] & termios.ICANON    # raw while it runs
        os.write(term.keys, b"hello\r\x03")
        assert _read(cli, b"hello\r\x03") == b"hello\r\x03"
        os.write(term.keys, b"\x16")
        assert _read(cli, b"\x16") == b"\x16"
        assert pressed == [True]                   # before the key reached the client
        # An empty paste reads a clipboard image in opencode, omp and hermes.
        os.write(term.keys, START + END)
        assert _read(cli, START + END) == START + END
        assert pressed == [True, True]
        os.write(term.keys, START + b"text" + END)
        assert _read(cli, END) == START + b"text" + END
        assert pressed == [True, True]
    finally:
        relay.close()
        _close(cli, *pipe)
    assert termios.tcgetattr(term.tty) == term.mode


def test_a_paste_reaches_the_client_rewritten(term):
    class Pastes:
        def rewrite(self, body):
            return body.replace(b"/mac/a.png", b"/home/.gmlx/pastes/x.png")
    relay, cli, pipe = _relay(term, pastes=Pastes())
    try:
        os.write(term.keys, b"a" + START + b"/mac/a.png")
        os.write(term.keys, END + b"b")
        assert _read(cli, END + b"b") == b"a" + START + b"/home/.gmlx/pastes/x.png" + END + b"b"
    finally:
        relay.close()
        _close(cli, *pipe)


def test_paths_typed_or_sent_without_paste_markers_pass_as_they_came(term):
    """Only a bracketed paste goes through the paste rewrite. Input that
    holds nothing but a path, typed or sent in one write, passes as it came,
    so a path you type is never copied into the container."""
    seen = []

    class Pastes:
        def rewrite(self, body):
            seen.append(body)
            return body
    relay, cli, pipe = _relay(term, pastes=Pastes())
    try:
        os.write(term.keys, b"/mac/a.png ")
        assert _read(cli, b"a.png ") == b"/mac/a.png "
        for key in b"/mac/b.png":
            os.write(term.keys, bytes([key]))
            _read(cli, bytes([key]))
    finally:
        relay.close()
        _close(cli, *pipe)
    assert seen == []


def test_a_paste_that_fails_to_rewrite_passes_as_it_came(term):
    class Pastes:
        def rewrite(self, body):
            raise RuntimeError("broken")
    lines = []
    relay, cli, pipe = _relay(term, pastes=Pastes(), log=lines.append)
    try:
        os.write(term.keys, START + b"/mac/a.png" + END)
        assert _read(cli, END) == START + b"/mac/a.png" + END
        # Typing goes on after the failed paste.
        os.write(term.keys, b"more")
        assert _read(cli, b"more") == b"more"
    finally:
        relay.close()
        _close(cli, *pipe)
    assert lines == ["terminal: a paste passed as it came (RuntimeError: broken)"]


def test_a_lone_escape_passes_after_the_idle_time(term):
    relay, cli, pipe = _relay(term, esc_wait=0)
    try:
        os.write(term.keys, b"\x1b")
        assert _read(cli, b"\x1b") == b"\x1b"
    finally:
        relay.close()
        _close(cli, *pipe)


def test_the_echo_before_raw_mode_reaches_your_terminal(term):
    """Until the CLI sets raw mode, its terminal has your settings, so what
    you type is echoed there and the relay copies the echo to your screen."""
    out_r, out_w = os.pipe()
    relay = terminal.TerminalRelay(term=term.tty, out=out_w)
    cli = os.dup(relay.slave)
    relay.start()
    try:
        os.write(term.keys, b"hi")
        assert _read(out_r, b"hi") == b"hi"
    finally:
        relay.close()
        _close(cli, out_r, out_w)


def test_close_twice_and_close_before_start_are_safe(term):
    relay = terminal.TerminalRelay(term=term.tty)
    relay.close()
    relay.close()
    assert termios.tcgetattr(term.tty) == term.mode


def test_no_terminal_no_relay():
    r, w = os.pipe()
    try:
        with pytest.raises(OSError):
            terminal.TerminalRelay(term=r)
    finally:
        _close(r, w)


def test_a_failing_relay_gives_the_terminal_back_and_hangs_up(term, monkeypatch):
    lines = []
    relay, cli, pipe = _relay(term, log=lines.append)

    def broken(data):
        raise RuntimeError("parser bug")
    monkeypatch.setattr(relay.parser, "feed", broken)
    try:
        os.write(term.keys, b"x")
        assert _ended(cli)                         # the client sees its terminal close
        assert termios.tcgetattr(term.tty) == term.mode
    finally:
        relay.close()
        _close(cli, *pipe)
    assert lines == ["terminal: the input relay stopped (RuntimeError: parser bug)"]


def test_a_closed_terminal_hangs_up_the_client(term):
    relay, cli, pipe = _relay(term)
    try:
        os.close(term.keys)                        # the window closes
        assert _ended(cli)
    finally:
        relay.close()
        _close(cli, *pipe)


# The relay with a real child that leads a session on the pseudo-terminal

_CHILD = textwrap.dedent("""
    import fcntl, os, signal, struct, sys, termios
    report = int(sys.argv[1])
    def size(signum, frame):
        rows, cols = struct.unpack("HHHH", fcntl.ioctl(0, termios.TIOCGWINSZ, b"\\0" * 8))[:2]
        os.write(report, f"size {rows} {cols}\\n".encode())
    signal.signal(signal.SIGWINCH, size)
    os.write(report, f"leader {os.getsid(0) == os.getpid()} "
                     f"foreground {os.tcgetpgrp(0) == os.getpgrp()}\\n".encode())
    while True:
        signal.pause()
""")


def _child(relay, report_w):
    return subprocess.Popen([sys.executable, "-c", _CHILD, str(report_w)], stdin=relay.slave,
                            pass_fds=(report_w,), start_new_session=True,
                            preexec_fn=terminal.take_terminal)


def test_the_child_leads_a_session_on_the_relay_and_follows_a_resize(term):
    report_r, report_w = os.pipe()
    relay = terminal.TerminalRelay(term=term.tty)
    child = _child(relay, report_w)
    relay.start()
    try:
        assert b"leader True foreground True" in _read(report_r, b"\n")
        fcntl.ioctl(term.tty, termios.TIOCSWINSZ, struct.pack("HHHH", 41, 133, 0, 0))
        os.kill(os.getpid(), signal.SIGWINCH)      # as the terminal sends it to launch
        assert b"size 41 133" in _read(report_r, b"size 41 133")
    finally:
        relay.close()
        child.kill()
        child.wait()
        _close(report_r, report_w)
    assert signal.getsignal(signal.SIGWINCH) in (signal.SIG_DFL, None)
    assert termios.tcgetattr(term.tty) == term.mode


def test_input_typed_after_a_resize_reaches_the_child_after_the_new_size(term):
    """The signal of a resize can reach launch after the input typed in the
    new window, so the relay checks the size before each input."""
    relay, cli, pipe = _relay(term)
    try:
        fcntl.ioctl(term.tty, termios.TIOCSWINSZ, struct.pack("HHHH", 52, 151, 0, 0))
        os.write(term.keys, b"x")                  # no SIGWINCH reaches launch
        assert _read(cli, b"x") == b"x"
        size = struct.unpack("HHHH", fcntl.ioctl(cli, termios.TIOCGWINSZ, b"\0" * 8))[:2]
        assert size == (52, 151)
    finally:
        relay.close()
        _close(cli, *pipe)


def test_a_closed_terminal_sends_the_child_sighup(term):
    report_r, report_w = os.pipe()
    relay = terminal.TerminalRelay(term=term.tty)
    child = _child(relay, report_w)
    relay.start()
    try:
        _read(report_r, b"\n")
        os.close(term.keys)                        # the window closes
        assert child.wait(10) == -signal.SIGHUP
    finally:
        relay.close()
        if child.poll() is None:
            child.kill()
            child.wait()
        _close(report_r, report_w)
