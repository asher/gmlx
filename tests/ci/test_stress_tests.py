"""scripts/stress_tests.py: a round that runs no test is a failure, and a
stall never resumes a stopped process or overlaps a test's stop."""
from __future__ import annotations

import fcntl
import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_stress_tests", _ROOT / "scripts" / "stress_tests.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stress = _load()


def test_a_round_that_runs_no_test_fails(tmp_path):
    # A missing path, like a missing plugin, ends pytest nonzero with no
    # failing test in its report.
    out = tmp_path / "out"
    rc = stress.main([str(tmp_path / "no_such_test.py"), "--count", "1", "--workers", "1",
                      "--stall-ms", "1", "--out", str(out)])
    assert rc == 1
    assert (out / "round-1.log").exists()
    assert (out / "flaky.json").read_text() == '{"flaky": []}'


def test_a_round_gives_its_staller_and_its_tests_one_lock_and_removes_it(
        tmp_path, monkeypatch):
    """Tests that stop a process take the staller's lock from
    GMLX_STRESS_LOCK, so a staller without it brings back the stops that
    stalls drop."""
    given = []

    class Recorder:
        stalls = 0

        def __init__(self, root, stall_ms, seed, lock=None):
            given.append(lock)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(stress, "Staller", Recorder)
    seen = tmp_path / "seen"
    test = tmp_path / "test_lock.py"
    test.write_text("import os\n\n\ndef test_lock():\n"
                    "    lock = os.environ['GMLX_STRESS_LOCK']\n"
                    "    open(lock, 'a').close()\n"
                    f"    with open({str(seen)!r}, 'w') as f:\n"
                    "        f.write(lock)\n")
    rc = stress.main([str(test), "--count", "1", "--workers", "1", "--out", str(tmp_path / "out")])
    assert rc == 0
    assert len(given) == 1 and given[0] is not None
    assert seen.read_text() == str(given[0])
    assert not Path(given[0]).exists()


def _until(check, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline
        time.sleep(0.005)


_TREE = """\
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import subprocess, time\\n"
                          "print(subprocess.Popen(['sleep', '60']).pid, flush=True)\\n"
                          "time.sleep(60)"], stdout=subprocess.PIPE)
print(child.pid, child.stdout.readline().decode().strip(), flush=True)
time.sleep(60)
"""


@pytest.fixture
def chain():
    """A process, its child and its grandchild, each running."""
    root = subprocess.Popen([sys.executable, "-c", _TREE], stdout=subprocess.PIPE,
                            start_new_session=True)
    pids = [root.pid, *map(int, root.stdout.readline().split())]
    yield pids
    os.killpg(root.pid, signal.SIGKILL)
    for pid in pids[1:]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    root.wait(10)


def test_stopped_reads_the_state_and_counts_an_unreadable_process_as_stopped(chain):
    """The staller resumes what it stalls, so a process whose state it
    cannot read, which might be stopped, is left alone."""
    _, child, _ = chain
    assert not stress.stopped(child)
    os.kill(child, signal.SIGSTOP)
    _until(lambda: stress.stopped(child))
    os.kill(child, signal.SIGCONT)
    _until(lambda: not stress.stopped(child))
    assert stress.stopped(2 ** 22 + 1)               # beyond any pid
    if sys.platform == "darwin":
        assert stress.stopped(1)                     # launchd, another user's


def test_a_stall_never_resumes_a_process_that_was_stopped(chain):
    """A stall ends with SIGCONT, which would resume a client that Ctrl-Z or
    a test stopped. The staller skips a pair when either one is stopped."""
    root, child, grandchild = chain
    staller = stress.Staller(root, stall_ms=1, seed=0)
    os.kill(child, signal.SIGSTOP)
    _until(lambda: stress.stopped(child))
    assert not staller.stall([root, child], 0.01)     # the chosen pid is stopped
    assert not staller.stall([child, grandchild], 0.01)   # its parent is
    assert stress.stopped(child) and not stress.stopped(grandchild)
    os.kill(child, signal.SIGCONT)
    _until(lambda: not stress.stopped(child))
    assert staller.stall([child, grandchild], 0.01)
    assert staller.stalls == 1
    _until(lambda: not stress.stopped(child) and not stress.stopped(grandchild))


def test_a_stall_holds_the_lock_from_its_check_to_its_continue(tmp_path, chain, monkeypatch):
    """A test stops a process while it holds the same lock, so its stop
    never lands while a stall holds that process stopped, where the kernel
    would drop it and the stall's SIGCONT would undo it."""
    _, child, grandchild = chain
    lock = tmp_path / "stall.lock"

    def free() -> bool:
        with open(lock, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

    checked, stopped = [], stress.stopped
    monkeypatch.setattr(stress, "stopped", lambda pid: checked.append(free()) or stopped(pid))
    staller = stress.Staller(chain[0], stall_ms=1, seed=0, lock=lock)
    stalling = threading.Thread(target=staller.stall, args=([child, grandchild], 60.0))
    stalling.start()
    try:
        _until(lambda: stopped(grandchild))
        assert not free()
    finally:
        staller.done.set()                          # ends the stall
        stalling.join()
    assert checked == [False, False]
    assert staller.stalls == 1 and free()
    _until(lambda: not stopped(child) and not stopped(grandchild))
