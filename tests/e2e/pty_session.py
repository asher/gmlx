"""Drive an interactive console program inside a pseudo-terminal.

The pty analog of :mod:`server_proc`: spawn a child (``gmlx chat ...``) attached to a
real pty so it sees an interactive tty - which is what makes the program engage its
full terminal UI (prompt_toolkit editing, the bottom toolbar, and the termios
``_EscCancel`` that watches for Esc while a reply streams). Plain pipes can't do this:
the program detects the non-tty and falls back to the line-editor shim.

Pure stdlib (``pty``) - no ``pexpect`` dependency. Reads accumulate into ``transcript``;
:meth:`expect` waits for a substring with a moving cursor so the same marker (e.g. the
per-reply ``tok @`` stat line) can be awaited once per turn without re-matching.

Synchronisation note: the caller must wait for a reply to *finish* (its ``tok @`` stat
line) before sending the next line. While a reply streams, the child holds stdin in
cbreak and drains every pending byte looking for Esc - anything typed early is eaten.
``tok @`` prints only after that loop exits and cooked mode is restored, so it is the
correct barrier between turns. (The Esc-cancel arm relies on exactly this drain.)
"""
from __future__ import annotations

import fcntl
import os
import pty
import select
import struct
import subprocess
import termios
import time


class PtyProcess:
    def __init__(self, argv, *, env=None, rows=40, cols=120, log=None):
        self.argv = list(argv)
        self.env = env if env is not None else dict(os.environ)
        self.rows, self.cols = rows, cols
        self.log = log                      # optional writable file to tee raw output
        self._raw = b""
        self.transcript = ""
        self._cursor = 0                    # search position for expect()
        self.master = None
        self.proc = None

    # ---- lifecycle -------------------------------------------------------------

    def __enter__(self):
        self.master, slave = pty.openpty()
        # Give the child a sane window size, else prompt_toolkit assumes 0x0.
        fcntl.ioctl(slave, termios.TIOCSWINSZ,
                    struct.pack("HHHH", self.rows, self.cols, 0, 0))
        self.proc = subprocess.Popen(
            self.argv, stdin=slave, stdout=slave, stderr=slave,
            env=self.env, close_fds=True, start_new_session=True)
        os.close(slave)                     # the child owns the slave now
        return self

    def __exit__(self, *exc):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        finally:
            if self.master is not None:
                os.close(self.master)

    # ---- io --------------------------------------------------------------------

    def _drain(self, timeout):
        """Pull whatever output is available for up to ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            r, _, _ = select.select([self.master], [], [], min(0.25, remaining))
            if not r:
                continue
            try:
                chunk = os.read(self.master, 65536)
            except OSError:                 # pty closed on child exit (EIO)
                return
            if not chunk:                   # EOF
                return
            self._raw += chunk
            self.transcript = self._raw.decode("utf-8", "replace")
            if self.log:
                self.log.write(chunk.decode("utf-8", "replace"))
                self.log.flush()

    def expect(self, needle, timeout=60.0):
        """Wait until ``needle`` appears past the last match. Returns True/False."""
        deadline = time.monotonic() + timeout
        while True:
            idx = self.transcript.find(needle, self._cursor)
            if idx != -1:
                self._cursor = idx + len(needle)
                return True
            if time.monotonic() >= deadline:
                return False
            self._drain(min(0.5, max(0.0, deadline - time.monotonic())))

    def send(self, text):
        """Write raw bytes to the child (use ``\\r`` for Enter, ``\\x1b`` for Esc)."""
        os.write(self.master, text.encode())

    def sendline(self, text=""):
        self.send(text + "\r")

    def wait_exit(self, timeout=30.0):
        """Drain trailing output and wait for exit. Returns the code, or None on
        timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self._drain(0.3)            # flush the final bytes
                return self.proc.returncode
            self._drain(0.3)
        return None

    def tail(self, n=2000):
        return self.transcript[-n:]
