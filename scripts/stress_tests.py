#!/usr/bin/env python3
"""Run tests under the stalls of a busy, shared machine to find races.

  python scripts/stress_tests.py --preset concurrency --minutes 10
  python scripts/stress_tests.py tests/x.py::test_y --count 200 --rounds 10

Each round is one pytest session with every selected test repeated --count
times (pytest-repeat) over twice as many xdist workers as cores. While it
runs, a staller stops one random process of the session's tree at a time
for up to --stall-ms, then continues it. Fast idle machines never reproduce
the lost-signal and FIFO races that a stalled CI VM shows; these stalls do.

A test that hangs fails at --timeout with every thread's stack. A failed
round keeps its log, a `ps` listing of the tree and any FIFOs left in its
temporary folders under --out. The failing test ids go to
<out>/flaky.json, which `scripts/ci_tests.py issues` reads.

Needs pytest-xdist, pytest-repeat and pytest-timeout (the `test` group).
"""

from __future__ import annotations

import argparse
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


def tree_pids(root: int) -> list[int]:
    """``root`` and every live descendant."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        pid, ppid = (int(x) for x in line.split())
        children.setdefault(ppid, []).append(pid)
    found, todo = [], [root]
    while todo:
        pid = todo.pop()
        found.append(pid)
        todo += children.get(pid, [])
    return found


class Staller(threading.Thread):
    """Stops one random process of a tree at a time, for up to ``stall_ms``."""

    def __init__(self, root: int, stall_ms: float, seed: int):
        super().__init__(daemon=True)
        self.root, self.stall_ms, self.rnd = root, stall_ms, random.Random(seed)
        self.done = threading.Event()
        self.stalls = 0

    def run(self) -> None:
        pids: list[int] = []
        refreshed = 0.0
        while not self.done.is_set():
            if time.monotonic() - refreshed > 0.2:
                pids, refreshed = tree_pids(self.root), time.monotonic()
            if self.done.wait(self.rnd.uniform(0, 2 * self.stall_ms) / 1000) or not pids:
                continue
            pid = self.rnd.choice(pids)
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                continue
            try:
                time.sleep(self.rnd.uniform(0, self.stall_ms) / 1000)
            finally:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            self.stalls += 1

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


def failures(junit: Path, count: int) -> list[str]:
    try:
        cases = ET.parse(junit).iter("testcase")
    except (OSError, ET.ParseError):
        return []
    return [base_id(c.get("classname", ""), c.get("name", ""), count)
            for c in cases if any(child.tag in ("failure", "error") for child in c)]


def _leftovers(root: int, tmp: Path, log: Path) -> None:
    ps = subprocess.run(["ps", "-o", "pid,ppid,stat,wchan,command", "-p",
                         ",".join(str(p) for p in tree_pids(root))], capture_output=True, text=True)
    fifos = subprocess.run(["find", str(tmp), "-type", "p"], capture_output=True, text=True)
    with open(log, "a") as f:
        f.write("\n--- process tree\n" + ps.stdout + "\n--- FIFOs left\n" + (fifos.stdout or "none\n"))


def one_round(n: int, a, out: Path) -> tuple[int, list[str], int]:
    # A short folder: tests put unix sockets under their tmp_path.
    tmp = Path(tempfile.mkdtemp(prefix=f"gst{n}-", dir="/tmp"))
    log = out / f"round-{n}.log"
    junit = out / f"round-{n}.xml"
    workers = a.workers or 2 * (os.cpu_count() or 4)
    cmd = [sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider",
           "-n", str(workers), "--dist", "worksteal", "--count", str(a.count),
           f"--timeout={a.timeout}", f"--basetemp={tmp}", f"--junitxml={junit}", *a.tests]
    env = {k: v for k, v in os.environ.items() if k != "KQUANT_TEST_GGUF_DIR"}
    with open(log, "w") as f:
        f.write("+ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        staller = Staller(proc.pid, a.stall_ms, seed=n)
        staller.start()
        try:
            code = proc.wait(timeout=a.round_timeout)
        except subprocess.TimeoutExpired:
            staller.stop()
            _leftovers(proc.pid, tmp, log)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return -1, [f"round {n} hung past {a.round_timeout} s"], staller.stalls
        staller.stop()
    failed = failures(junit, a.count)
    if code != 0:
        _leftovers(proc.pid, tmp, log)
    else:
        log.unlink()
        junit.unlink(missing_ok=True)
    subprocess.run(["rm", "-rf", str(tmp)])
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
    start, n, bad = time.monotonic(), 0, Counter()
    while (not a.rounds or n < a.rounds) and (not a.minutes or time.monotonic() - start < 60 * a.minutes):
        n += 1
        code, failed, stalls = one_round(n, a, out)
        bad.update(failed)
        print(f"round {n}: exit {code}, {len(failed)} failed, {stalls} stalls", flush=True)
    (out / "flaky.json").write_text(json.dumps({"flaky": sorted(t for t in bad if "::" in t)}))
    for test, times in bad.most_common():
        print(f"FAILED {times}x {test}")
    print(f"{n} rounds, {sum(bad.values())} failures; logs of failed rounds in {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
