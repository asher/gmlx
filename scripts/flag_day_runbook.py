#!/usr/bin/env python3
"""Flag-day runbook for the pinned-invariant probes.

Runs every probe arm, including the destructive negatives that the
pytest suite (tests/test_pinned_invariants.py) skips on wheels where
they abort the whole process, and checks each outcome against the
expectation for the installed mlx wheel. Run this once per mlx
upgrade, on an idle box: the abort arms intentionally drive the GPU
into a command-buffer timeout, which can take a couple of minutes to
resolve and briefly wedges other GPU clients.

Exit status is nonzero if any arm's outcome does not match the
expectation table. A mismatch is the flag-day finding, not a runbook
bug: read the printed transcript before touching the table.

Usage:
  python scripts/flag_day_runbook.py           run everything
  python scripts/flag_day_runbook.py --safe    positive arms only
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROBES = Path(__file__).resolve().parents[1] / "tests" / "invariant_probes"


def mlx_version():
    import mlx.core as mx

    parts = []
    for p in mx.__version__.split(".")[:3]:
        m = re.match(r"\d+", p)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts), mx.__version__


# Each row: (script, arm-or-None, destructive, expectation pre-fix,
# expectation on 0.32.1+). Destructive rows abort the child process
# on pre-fix wheels; they are still safe for the parent, but skip
# them with --safe if the box is busy.
TABLE = [
    ("sync_survivor.py", None, False, "CLEAN", "CLEAN"),
    ("async_survivor.py", "item", False, "HANG", "CLEAN"),
    ("async_survivor.py", "samestream", False, "CLEAN", "CLEAN"),
    ("async_survivor.py", "fresh", False, "CLEAN", "CLEAN"),
    ("async_survivor.py", "xstream", True, "ABORT", "CLEAN"),
    ("gil_release.py", "hang", False, "EXIT42", "CLEAN"),
    ("gil_release.py", "busy", False, "BUSY", "BUSY"),
    ("multistream_drain.py", "orderfix", False, "CLEAN", "CLEAN"),
    ("multistream_drain.py", "thrower-first", True, "ABORT", "CLEAN"),
    ("reverse_topology.py", "trip-only", True, "CAUGHT-ABORT", "CAUGHT"),
    ("reverse_topology.py", "producer-first", False, "CLEAN", "CLEAN"),
    ("reverse_topology.py", "consumer-first", True, "ABORT", "CLEAN"),
    ("thread_rules.py", None, False, "MATRIX", "MATRIX"),
]

THREAD_WANT = {"a": "ok", "b": "ok", "c": "THROW", "d": "THROW", "e": "THROW"}


def classify(script, arm, timeout):
    cmd = [sys.executable, str(PROBES / script)]
    if arm:
        cmd.append(arm)
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        return "HANG", e.stdout or ""
    out = r.stdout + ("\n" + r.stderr if r.stderr.strip() else "")
    if r.returncode == 42 and "outcome=ALIVE" in r.stdout:
        return "EXIT42", out
    if r.returncode < 0:
        # A caught trip followed by an abort at teardown is its own
        # class: the catch worked, the poisoned stream killed exit.
        if "outcome=CAUGHT" in r.stdout:
            return "CAUGHT-ABORT", out
        return "ABORT", out
    if r.returncode != 0:
        return "ERROR", out
    if script == "thread_rules.py":
        got = dict(
            re.findall(r"^case=(\w) outcome=(\S+)", r.stdout, re.MULTILINE)
        )
        ok = all(
            got.get(c, "").startswith(w) for c, w in THREAD_WANT.items()
        )
        return ("MATRIX" if ok else "MATRIX-DIFF"), out
    if "outcome=CLEAN" in r.stdout:
        return "CLEAN", out
    if "outcome=CAUGHT" in r.stdout:
        return "CAUGHT", out
    if "outcome=BUSY-DONE" in r.stdout:
        return "BUSY", out
    return "OTHER", out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safe", action="store_true",
                    help="skip the destructive abort arms")
    args = ap.parse_args()

    ver, ver_str = mlx_version()
    fixed = ver >= (0, 32, 1)
    col = 4 if fixed else 3
    print(f"mlx {ver_str}: expectations for the "
          f"{'0.32.1+' if fixed else 'pre-0.32.1'} column\n")

    failures = []
    for row in TABLE:
        script, arm, destructive, *_ = row
        expect = row[col]
        name = script.replace(".py", "") + (f" {arm}" if arm else "")
        if args.safe and destructive:
            print(f"{name:38s} SKIP (destructive; --safe)")
            continue
        timeout = 25 if expect == "HANG" else 180
        got, out = classify(script, arm, timeout)
        status = "ok" if got == expect else "MISMATCH"
        print(f"{name:38s} expect={expect:7s} got={got:11s} {status}")
        if got != expect:
            failures.append((name, expect, got, out))

    if failures:
        print(f"\n{len(failures)} mismatch(es):")
        for name, expect, got, out in failures:
            print(f"\n--- {name}: expected {expect}, got {got} ---")
            print(out.rstrip())
        return 1
    print("\nall arms match the expectation table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
