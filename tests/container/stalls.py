"""The lock of scripts/stress_tests.py's staller, for the moments of a
test that a stall would break."""

from __future__ import annotations

import contextlib
import fcntl
import os


@contextlib.contextmanager
def no_stall():
    """Holds the lock of scripts/stress_tests.py's staller, so no stall is
    under way. The kernel drops a stop sent to a process that a stall holds
    stopped, such as the SIGTSTP of a Ctrl-Z. A stall's SIGCONT undoes a
    stop that has taken effect, and discards one still pending, as a
    SIGTSTP is until its process runs or unblocks it. So a test holds the
    lock from the stop until it sees what the stop did. macOS also leaves a
    signal that only a thread asleep in the kernel can take pending on that
    thread across a stop and continue, until the thread next runs."""
    path = os.environ.get("GMLX_STRESS_LOCK")
    if not path:
        yield
        return
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
