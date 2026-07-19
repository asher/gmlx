"""A reusable, stdlib-only terminal spinner.

TTY-gated: when the stream is a terminal a daemon thread animates braille frames in
place; otherwise the label is printed once and nothing animates (clean pipes/CI/logs).
Used by ``launch`` to show progress while a background server starts, but deliberately
generic so other long waits can reuse it.
"""
from __future__ import annotations

import sys
import threading
import time

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """Context manager that animates ``text`` on a terminal while the block runs.

    On a TTY a background thread rewrites the current line (``\\r``) ~10x/s with the
    next frame, the label, and - when ``show_elapsed`` - the whole seconds since the
    block was entered. Off a TTY it prints ``text`` once and animates nothing, so the
    label is still captured in logs without escape-code noise.

    ``update(new_text)`` swaps the label live. ``__exit__`` stops and *joins* the
    writer thread before clearing the line, so no frame can land after the ``with``
    block closes (e.g. just before an ``os.execvpe``). The stop/clear runs on
    exceptions and ``KeyboardInterrupt`` too.
    """

    def __init__(self, text: str, *, stream=None, show_elapsed: bool = True):
        self._text = text
        self._stream = stream if stream is not None else sys.stderr
        self._show_elapsed = show_elapsed
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0
        self._lock = threading.Lock()

    def update(self, new_text: str) -> None:
        """Swap the animated label (the writer snapshots it per frame)."""
        self._text = new_text
        if not self._tty:
            print(new_text, file=self._stream, flush=True)

    def println(self, msg: str) -> None:
        """Emit a full line without corrupting the animated frame; the next
        frame redraws below it."""
        if not self._tty:
            print(msg, file=self._stream, flush=True)
            return
        with self._lock:
            self._stream.write("\r\x1b[K" + msg + "\n")
            self._stream.flush()

    def __enter__(self) -> "Spinner":
        self._start = time.monotonic()
        if not self._tty:
            print(self._text, file=self._stream, flush=True)
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._stream.write("\r\x1b[K")
            self._stream.flush()
        return False                      # never suppress exceptions

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = _FRAMES[i % len(_FRAMES)]
            line = f"{frame} {self._text}"
            if self._show_elapsed:
                line += f" {int(time.monotonic() - self._start)}s"
            with self._lock:
                self._stream.write("\r\x1b[K" + line)
                self._stream.flush()
            i += 1
            self._stop.wait(0.1)
