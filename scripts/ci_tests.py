#!/usr/bin/env python3
"""Run the CI test step, keep the flake record, and report suite growth.

  python scripts/ci_tests.py run --select full|docs [--skip-numerics] [--nightly]
  python scripts/ci_tests.py issues --flaky flaky.json --junit junit.xml [--count] [--daily]
  python scripts/ci_tests.py growth --junit junit.xml [--baseline main.xml]

`run` runs pytest once with a junit report. When only tests failed, it
reruns them alone, serially. A test that passes on the rerun is flaky: the
step passes with a warning, and its id goes to flaky.json for `issues`.

`issues` opens or reopens one GitHub issue labelled `flaky` per flaky test.
With `--count` it also counts, for each open issue, the runs where the test
passed, and closes the issue after 20 in a row. With `--daily`, a run that
does not contain the test counts towards closing it after 5, since a daily
run deselects nothing.

`growth` writes the test count, total test time and the slowest tests to
the job summary, with the change against the main baseline.

Stdlib only; `issues` needs the gh CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from changed_paths import DOCS_TESTS, NUMERICS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LASTFAILED = ROOT / ".pytest_cache" / "v" / "cache" / "lastfailed"
GREEN_TO_CLOSE = 20
ABSENT_TO_CLOSE = 5
NOTICE_CASES = 100
NOTICE_SECONDS = 60.0
LABEL = "flaky"
_ID = re.compile(r"<!-- flaky-id: (.+?) -->")
_STATE = re.compile(r"<!-- flaky-state: (\{.*?\}) -->")


def selection(select: str, *, skip_numerics: bool, nightly: bool) -> list[str]:
    """The pytest arguments that pick the tests, without output options."""
    if select == "docs":
        return sorted(str(p.relative_to(ROOT)) for g in DOCS_TESTS for p in ROOT.glob(g))
    args = [] if nightly else ["-m", "not nightly"]
    if skip_numerics:
        args += [f"--ignore=tests/{d}" for d in NUMERICS]
    return args


def junit_key(nodeid: str) -> tuple[str, str]:
    """The (classname, name) pair pytest's junit report gives a node id."""
    path, bracket, params = nodeid.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + params
    return ".".join(names[:-1]), names[-1]


def read_junit(path: Path) -> dict[tuple[str, str], tuple[str, float]]:
    """Each test case's outcome (passed, failed, skipped) and time."""
    out = {}
    for case in ET.parse(path).iter("testcase"):
        tags = {child.tag for child in case}
        outcome = ("failed" if tags & {"failure", "error"}
                   else "skipped" if "skipped" in tags else "passed")
        out[(case.get("classname", ""), case.get("name", ""))] = (outcome, float(case.get("time") or 0))
    return out


def _summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


def _lastfailed() -> list[str]:
    try:
        return sorted(json.loads(LASTFAILED.read_text()))
    except (OSError, ValueError):
        return []


def _pytest(args: list[str]) -> int:
    print("+ python -m pytest " + " ".join(args), flush=True)
    return subprocess.run([sys.executable, "-m", "pytest", *args], cwd=ROOT).returncode


def run(a) -> int:
    picked = selection(a.select, skip_numerics=a.skip_numerics, nightly=a.nightly)
    report = ["-q", "-rfE", "--durations=25"]
    LASTFAILED.unlink(missing_ok=True)
    code = _pytest([*report, "-n", "auto", "--dist", "worksteal", f"--junitxml={a.junit}", *picked])
    failed = _lastfailed()
    Path(a.flaky_out).write_text(json.dumps({"flaky": []}))
    if code != 1 or not failed:
        return code
    print(f"\n{len(failed)} failed; rerunning them alone to tell flaky from broken", flush=True)
    rerun = _pytest([*report, "-p", "no:xdist", "--lf", "--lfnf=none",
                     f"--junitxml={Path(a.junit).with_suffix('.rerun.xml')}", *picked])
    if rerun != 0:
        return rerun
    Path(a.flaky_out).write_text(json.dumps({"flaky": failed}))
    rows = ["### Flaky tests", "", "Failed, then passed when rerun alone:", ""]
    for nodeid in failed:
        print(f"::warning title=Flaky test::{nodeid} failed, then passed when rerun alone")
        rows.append(f"- `{nodeid}`")
    _summary("\n".join(rows))
    return 0


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def parse_issue(body: str) -> tuple[str | None, dict]:
    nodeid = _ID.search(body or "")
    state = _STATE.search(body or "")
    return (nodeid.group(1) if nodeid else None,
            json.loads(state.group(1)) if state else {"green": 0, "absent": 0})


def issue_body(nodeid: str, state: dict, note: str) -> str:
    return (f"`{nodeid}` failed in CI, then passed when rerun alone.\n\n{note}\n\n"
            f"This issue closes after {GREEN_TO_CLOSE} green runs in a row on main, or after "
            f"{ABSENT_TO_CLOSE} daily runs without the test.\n\n"
            f"<!-- flaky-id: {nodeid} -->\n<!-- flaky-state: {json.dumps(state)} -->\n")


def tally(state: dict, outcome: str | None, *, daily: bool) -> tuple[dict, str | None]:
    """The counters after one run, and why the issue closes, if it does.
    ``outcome`` is the test's junit outcome in this run, None when absent."""
    green, absent = state.get("green", 0), state.get("absent", 0)
    if outcome == "passed":
        green, absent = green + 1, 0
    elif outcome == "failed":
        green, absent = 0, 0
    elif outcome is None and daily:
        absent += 1
    new = {"green": green, "absent": absent}
    if green >= GREEN_TO_CLOSE:
        return new, f"passed in {GREEN_TO_CLOSE} runs in a row"
    if absent >= ABSENT_TO_CLOSE:
        return new, f"missing from {ABSENT_TO_CLOSE} daily runs: removed, renamed or consolidated"
    return new, None


def issues(a) -> int:
    flaky = json.loads(Path(a.flaky).read_text())["flaky"] if Path(a.flaky).exists() else []
    if not flaky and not a.count:
        return 0
    run_url = a.run_url or "this run"
    listed = json.loads(_gh("issue", "list", "--label", LABEL, "--state", "all", "--limit", "1000",
                            "--json", "number,state,body"))
    known = {}
    for issue in listed:
        nodeid, state = parse_issue(issue["body"])
        if nodeid:
            known[nodeid] = (issue, state)
    if flaky:
        _gh("label", "create", LABEL, "--color", "d93f0b", "--force",
            "--description", "A test that failed in CI, then passed when rerun alone")
    for nodeid in flaky:
        note = f"Last seen: {run_url}."
        fresh = {"green": 0, "absent": 0}
        if nodeid in known:
            number = str(known[nodeid][0]["number"])
            if known[nodeid][0]["state"] != "OPEN":
                _gh("issue", "reopen", number)
            _gh("issue", "edit", number, "--body", issue_body(nodeid, fresh, note))
            _gh("issue", "comment", number, "--body", f"Flaky again in {run_url}.")
        else:
            _gh("issue", "create", "--label", LABEL, "--title", f"flaky: {nodeid}"[:250],
                "--body", issue_body(nodeid, fresh, note))
    if not a.count:
        return 0
    outcomes = {k: v[0] for k, v in read_junit(Path(a.junit)).items()}
    for nodeid, (issue, state) in known.items():
        if issue["state"] != "OPEN" or nodeid in flaky:
            continue
        new, reason = tally(state, outcomes.get(junit_key(nodeid)), daily=a.daily)
        if new == state and reason is None:
            continue
        number = str(issue["number"])
        _gh("issue", "edit", number, "--body", issue_body(nodeid, new, f"Last counted: {run_url}."))
        if reason:
            _gh("issue", "close", number, "--comment", f"Closed: the test {reason}.")
    return 0


def growth_report(cases: dict, base: dict | None, *, slowest: int = 25) -> tuple[str, str | None]:
    """The summary markdown and the notice text, if the change is large."""
    total = sum(t for _, t in cases.values())
    lines = ["### Test suite size", ""]
    notice = None
    if base is None:
        lines.append(f"{len(cases)} tests, {total:.0f} s of test time. No main baseline to compare.")
    else:
        new = [k for k in cases if k not in base]
        new_time = sum(cases[k][1] for k in new)
        base_total = sum(t for _, t in base.values())
        delta = len(cases) - len(base)
        lines += ["| | main | this run | change |", "|---|---|---|---|",
                  f"| tests | {len(base)} | {len(cases)} | {delta:+d} |",
                  f"| test time | {base_total:.0f} s | {total:.0f} s | {total - base_total:+.0f} s |",
                  "", f"{len(new)} tests are new here; they take {new_time:.0f} s."]
        if delta >= NOTICE_CASES or new_time >= NOTICE_SECONDS:
            notice = f"This change adds {delta:+d} tests; the {len(new)} new ones take {new_time:.0f} s."
    top = sorted(cases.items(), key=lambda kv: -kv[1][1])[:slowest]
    lines += ["", f"Slowest {len(top)}:", "", "| s | test |", "|---|---|"]
    lines += [f"| {t:.1f} | `{c}::{n}` |" for (c, n), (_, t) in top]
    return "\n".join(lines), notice


def growth(a) -> int:
    cases = read_junit(Path(a.junit))
    base = read_junit(Path(a.baseline)) if a.baseline and Path(a.baseline).exists() else None
    text, notice = growth_report(cases, base)
    _summary(text)
    if notice:
        print(f"::notice title=Suite growth::{notice}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--select", choices=["full", "docs"], default="full")
    r.add_argument("--skip-numerics", action="store_true")
    r.add_argument("--nightly", action="store_true", help="also run the tests marked nightly")
    r.add_argument("--junit", default="junit.xml")
    r.add_argument("--flaky-out", default="flaky.json")
    i = sub.add_parser("issues")
    i.add_argument("--flaky", default="flaky.json")
    i.add_argument("--junit", default="junit.xml")
    i.add_argument("--count", action="store_true", help="count green runs for open issues")
    i.add_argument("--daily", action="store_true", help="a run that deselects nothing")
    i.add_argument("--run-url", default="")
    g = sub.add_parser("growth")
    g.add_argument("--junit", default="junit.xml")
    g.add_argument("--baseline", default="")
    a = ap.parse_args(argv)
    return {"run": run, "issues": issues, "growth": growth}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
