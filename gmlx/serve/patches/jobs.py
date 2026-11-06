"""The book a job route keeps: its jobs in flight, the seconds of its
recent jobs and its counts, under one lock. The images API and the video
jobs each keep one."""

from __future__ import annotations

import statistics
import threading

# The recent jobs whose seconds a route's median reads.
RECENT = 16


class JobBook:
    def __init__(self, *counts: str):
        self.lock = threading.Lock()
        self.in_flight = 0
        self.durations: list = []           # seconds of recent jobs
        self.counts = dict.fromkeys(counts, 0)

    def admit(self, waiting: int) -> bool:
        """Take a slot, or refuse when ``waiting`` jobs wait behind the
        one running."""
        with self.lock:
            if self.in_flight > waiting:
                return False
            self.in_flight += 1
            return True

    def leave(self, seconds: float | None = None) -> None:
        with self.lock:
            self.in_flight = max(0, self.in_flight - 1)
            if seconds is not None:
                self.timed(seconds)

    def timed(self, seconds: float) -> None:
        """Add a job's seconds to the recent ones. Callers serialize:
        ``leave`` holds the book's lock and VideoJobs its own. The list is
        replaced, not changed in place, so ``typical`` reads the old one or
        the new one."""
        self.durations = (self.durations + [float(seconds)])[-RECENT:]

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + n

    def typical(self) -> float | None:
        """The median seconds of the recent jobs, None with none."""
        with self.lock:
            return (statistics.median(self.durations) if self.durations
                    else None)
