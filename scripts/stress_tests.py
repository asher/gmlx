#!/usr/bin/env python3
"""Run tests under the stalls of a busy, shared machine to find races.

  python scripts/stress_tests.py --preset concurrency --minutes 10
  python scripts/stress_tests.py tests/x.py::test_y --count 200 --rounds 10

Each round is one pytest session with every selected test repeated --count
times (pytest-repeat) over twice as many xdist workers as cores. While it
runs, a staller stops one random process of the session's tree at a time
for up to --stall-ms, then continues it. A parent that would see the stop,
which a busy machine never shows, stops first and goes on last. Fast idle
machines never reproduce the lost-signal and FIFO races that a stalled CI
VM shows; these stalls do.

A stall ends with SIGCONT, which would resume a process that job control
or a test stopped, so the staller skips a stall in which any of its
processes is stopped. The kernel drops a stop signal sent to a process that is already
stopped, so a stop that lands during a stall is lost. Each stall holds the
lock file that GMLX_STRESS_LOCK names, and a test that stops a process,
or types Ctrl-Z in its terminal, holds that lock while it does.

A test that hangs fails at --timeout with every thread's stack. A failed
round keeps its log, a `ps` listing of the tree and any FIFOs left in its
temporary folders under --out. The failing test ids go to
<out>/flaky.json with their failure messages, which `scripts/ci_tests.py
issues` reads. A round whose
pytest exits nonzero with no failing test, as when a plugin is missing or
nothing is collected, fails too.

Needs pytest-xdist, pytest-repeat and pytest-timeout (the `test` group).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The tests of signals, ptys, FIFOs, sockets and worker threads, where the
# races of a loaded machine show.
PRESETS = {"concurrency": [
    "tests/container/test_container_entry.py", "tests/container/test_container_session.py",
    "tests/container/test_container_relay.py", "tests/container/test_container_terminal.py",
    "tests/stream/test_decode_feeder.py", "tests/stream/test_prefill_feeder.py",
    "tests/stream/test_keepwarm.py", "tests/serve/test_lifecycle.py",
    "tests/serve/test_engine_jobs.py", "tests/serve/test_session_sockets.py",
    "tests/talk/test_talk.py", "tests/test_spinner.py",
]}


# The processes that wait for their children with WUNTRACED, and so would
# see a stall as a stop. A shell or Python sees no stop of its child.
WATCHERS = ("gmlx-entry",)


def tree(root: int) -> dict[int, tuple[int, str]]:
    """``root`` and every live descendant, each with its parent and name."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,comm="], capture_output=True, text=True).stdout
    children: dict[int, list[tuple[int, str]]] = {}
    for line in out.splitlines():
        pid, ppid, *comm = line.split(None, 2)
        children.setdefault(int(ppid), []).append((int(pid), os.path.basename(comm[0]) if comm else ""))
    found, todo = {}, [(root, 0, "")]
    while todo:
        pid, ppid, name = todo.pop()
        found[pid] = (ppid, name)
        todo += [(child, pid, child_name) for child, child_name in children.get(pid, [])]
    return found


def stalled(procs: dict[int, tuple[int, str]], pid: int) -> list[int]:
    """What a stall of ``pid`` stops, in order: each ancestor that would see
    it stop, from the top, then ``pid``."""
    out = [pid]
    while (parent := procs[out[0]][0]) in procs and procs[parent][1] in WATCHERS:
        out.insert(0, parent)
    return out


class _BSDInfo(ctypes.Structure):
    """The head of macOS's struct proc_bsdinfo, padded to its full size."""
    _fields_ = [("flags", ctypes.c_uint32), ("status", ctypes.c_uint32),
                ("rest", ctypes.c_char * 128)]


_LIBC = ctypes.CDLL(None) if sys.platform == "darwin" else None
_PROC_PIDTBSDINFO, _SSTOP = 3, 4


def stopped(pid: int) -> bool:
    """Whether ``pid`` is stopped, read without starting a process, since a
    stall check runs before every stall. A process whose state cannot be
    read, such as a zombie, a setuid process or one gone, counts as
    stopped, so the staller leaves it alone."""
    if _LIBC is not None:
        info = _BSDInfo()
        size = ctypes.sizeof(info)
        got = _LIBC.proc_pidinfo(pid, _PROC_PIDTBSDINFO, ctypes.c_uint64(0),
                                 ctypes.byref(info), size)
        return got != size or info.status == _SSTOP
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rpartition(")")[2].split()[0] in ("T", "t")
    except (OSError, IndexError):
        return True


def tree_pids(root: int) -> list[int]:
    """``root`` and every live descendant."""
    return list(tree(root))


class Staller(threading.Thread):
    """Stops one random process of a tree at a time, for up to ``stall_ms``.
    Each stall holds an flock on ``lock`` when one is given."""

    def __init__(self, root: int, stall_ms: float, seed: int, lock: Path | None = None):
        super().__init__(daemon=True)
        self.root, self.stall_ms, self.rnd = root, stall_ms, random.Random(seed)
        self.lock = lock
        self.done = threading.Event()
        self.stalls = 0

    def run(self) -> None:
        procs: dict[int, tuple[int, str]] = {}
        refreshed = 0.0
        while not self.done.is_set():
            if time.monotonic() - refreshed > 0.2:
                procs, refreshed = tree(self.root), time.monotonic()
            if self.done.wait(self.rnd.uniform(0, 2 * self.stall_ms) / 1000) or not procs:
                continue
            self.stall(stalled(procs, self.rnd.choice(list(procs))),
                       self.rnd.uniform(0, self.stall_ms) / 1000)

    def stall(self, pids: list[int], seconds: float) -> bool:
        """Stops ``pids`` in order for ``seconds``, or until :meth:`stop`,
        then continues them in the reverse order, unless one of them is
        stopped already. Returns whether it stalled them."""
        with contextlib.ExitStack() as held:
            if self.lock is not None:
                fcntl.flock(held.enter_context(open(self.lock, "a")), fcntl.LOCK_EX)
            if any(stopped(p) for p in pids):
                return False
            try:
                for p in pids:
                    os.kill(p, signal.SIGSTOP)
                self.done.wait(seconds)
                self.stalls += 1
                return True
            except ProcessLookupError:
                return False
            finally:
                for p in reversed(pids):
                    try:
                        os.kill(p, signal.SIGCONT)
                    except ProcessLookupError:
                        pass

    def stop(self) -> None:
        self.done.set()
        self.join()
        for pid in tree_pids(self.root):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


def base_id(classname: str, name: str, count: int) -> str:
    """A junit case as a node id, without the pytest-repeat suffix."""
    name = re.sub(rf"\[(?:(.*)-)?\d+-{count}\]$", lambda m: f"[{m.group(1)}]" if m.group(1) else "", name)
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        path = Path(*parts[:i]).with_suffix(".py")
        if (ROOT / path).is_file():
            return "::".join([path.as_posix(), *parts[i:], name])
    return f"{classname}::{name}"


def failures(junit: Path, count: int) -> dict[str, str]:
    """Each failed test id with its first failure message."""
    try:
        cases = ET.parse(junit).iter("testcase")
    except (OSError, ET.ParseError):
        return {}
    out: dict[str, str] = {}
    for c in cases:
        for child in c:
            if child.tag in ("failure", "error"):
                out.setdefault(base_id(c.get("classname", ""), c.get("name", ""), count),
                               child.get("message") or (child.text or "").strip())
                break
    return out


def _leftovers(root: int, tmp: Path, log: Path) -> None:
    ps = subprocess.run(["ps", "-o", "pid,ppid,stat,wchan,command", "-p",
                         ",".join(str(p) for p in tree_pids(root))], capture_output=True, text=True)
    fifos = subprocess.run(["find", str(tmp), "-type", "p"], capture_output=True, text=True)
    with open(log, "a") as f:
        f.write("\n--- process tree\n" + ps.stdout + "\n--- FIFOs left\n" + (fifos.stdout or "none\n"))


def one_round(n: int, a, out: Path) -> tuple[int, dict[str, str], int]:
    # A short folder: tests put unix sockets under their tmp_path.
    tmp = Path(tempfile.mkdtemp(prefix=f"gst{n}-", dir="/tmp"))
    log = out / f"round-{n}.log"
    junit = out / f"round-{n}.xml"
    workers = a.workers or 2 * (os.cpu_count() or 4)
    cmd = [sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider",
           "-n", str(workers), "--dist", "worksteal", "--count", str(a.count),
           f"--timeout={a.timeout}", f"--basetemp={tmp}", f"--junitxml={junit}", *a.tests]
    # pytest empties --basetemp first, so the lock lives beside it.
    lock = tmp.with_name(tmp.name + ".lock")
    env = {k: v for k, v in os.environ.items() if k != "KQUANT_TEST_GGUF_DIR"}
    env["GMLX_STRESS_LOCK"] = str(lock)
    with open(log, "w") as f:
        f.write("+ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        staller = Staller(proc.pid, a.stall_ms, seed=n, lock=lock)
        staller.start()
        try:
            code = proc.wait(timeout=a.round_timeout)
        except subprocess.TimeoutExpired:
            staller.stop()
            _leftovers(proc.pid, tmp, log)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            lock.unlink(missing_ok=True)
            return -1, {f"round {n} hung past {a.round_timeout} s": ""}, staller.stalls
        staller.stop()
    failed = failures(junit, a.count)
    if code != 0 and not failed:
        failed = {f"round {n}: pytest exit {code}": ""}
    if code != 0:
        _leftovers(proc.pid, tmp, log)
    else:
        log.unlink()
        junit.unlink(missing_ok=True)
    subprocess.run(["rm", "-rf", str(tmp)])
    lock.unlink(missing_ok=True)
    return code, failed, staller.stalls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tests", nargs="*", help="pytest paths or node ids")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="add a named list of test files")
    ap.add_argument("--count", type=int, default=20, help="repeats of each test per round")
    ap.add_argument("--rounds", type=int, default=0, help="stop after this many rounds")
    ap.add_argument("--minutes", type=float, default=0, help="start no round after this many minutes")
    ap.add_argument("--stall-ms", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=0, help="default: twice the cores")
    ap.add_argument("--timeout", type=int, default=120, help="per-test limit, seconds")
    ap.add_argument("--round-timeout", type=int, default=3600)
    ap.add_argument("--out", default="stress-out", help="logs of failed rounds and flaky.json")
    a = ap.parse_args(argv)
    a.tests = [*a.tests, *PRESETS.get(a.preset, [])]
    if not a.tests:
        ap.error("name tests or a --preset")
    if not a.rounds and not a.minutes:
        a.rounds = 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start, n, bad, messages = time.monotonic(), 0, Counter(), {}
    while (not a.rounds or n < a.rounds) and (not a.minutes or time.monotonic() - start < 60 * a.minutes):
        n += 1
        code, failed, stalls = one_round(n, a, out)
        bad.update(failed.keys())
        for test, message in failed.items():
            messages.setdefault(test, message)
        print(f"round {n}: exit {code}, {len(failed)} failed, {stalls} stalls", flush=True)
    flaky = sorted(t for t in bad if "::" in t)
    (out / "flaky.json").write_text(json.dumps(
        {"flaky": flaky, "messages": {t: messages[t] for t in flaky if messages[t]}}))
    for test, times in bad.most_common():
        print(f"FAILED {times}x {test}")
    print(f"{n} rounds, {sum(bad.values())} failures; logs of failed rounds in {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
