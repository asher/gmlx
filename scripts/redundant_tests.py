#!/usr/bin/env python3
"""List tests whose coverage another test in the same file already has.

  python scripts/redundant_tests.py                  # the last full gate run
  python scripts/redundant_tests.py --db .coverage --junit junit.xml
  python scripts/redundant_tests.py tests/image      # one folder

It reads coverage data recorded per test (`--cov=gmlx --cov-context=test`,
as `scripts/gate.py run --full` records it; add `--cov-branch` for arcs).
In each test file it lists:

- a test that covers the same lines or arcs as another, or a subset of
  another's;
- a parametrized test whose cases cover fewer distinct sets than it has
  cases.

A listed test is a candidate, not a verdict: two tests with the same
coverage can assert different things. Python coverage also cannot see
kernel paths, so in the numerics folders a case whose shape or dtype picks
another kernel can cover the same Python lines as its neighbour.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from changed_paths import NUMERICS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def default_db(root: Path | None = None) -> Path:
    root = root or ROOT
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=root, capture_output=True, text=True).stdout.strip()
    return Path(common) / "gmlx-gate" / "coverage" / ".coverage"


def read_masks(db: Path) -> dict[str, int]:
    """Each test's coverage as one bit per (file, line) or (file, arc)."""
    from coverage.numbits import numbits_to_nums

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tests = {}
    for cid, ctx in con.execute("SELECT id, context FROM context"):
        nodeid, _, phase = ctx.rpartition("|")
        if nodeid.startswith("tests/") and phase == "run":
            tests[cid] = nodeid
    index: dict[tuple, int] = {}
    masks: dict[str, int] = defaultdict(int)
    for fid, cid, bits in con.execute("SELECT file_id, context_id, numbits FROM line_bits"):
        if cid in tests:
            for line in numbits_to_nums(bits):
                masks[tests[cid]] |= 1 << index.setdefault((fid, line), len(index))
    for fid, cid, a, b in con.execute("SELECT file_id, context_id, fromno, tono FROM arc"):
        if cid in tests:
            masks[tests[cid]] |= 1 << index.setdefault((fid, a, b), len(index))
    con.close()
    return dict(masks)


def read_times(junit: Path) -> dict[str, float]:
    out = {}
    for case in ET.parse(junit).iter("testcase"):
        parts = case.get("classname", "").split(".")
        # The module path ends before the first class name.
        i = len(parts)
        while i > 1 and parts[i - 1][:1].isupper():
            i -= 1
        nodeid = "/".join(parts[:i]) + ".py::" + "::".join([*parts[i:], case.get("name", "")])
        out[nodeid] = float(case.get("time") or 0)
    return out


def candidates(masks: dict[str, int]) -> dict[str, list[tuple[str, str, str]]]:
    """Per test file, (test, how, other test): "same" when the two cover
    the same set, "inside" when the test's set is a strict subset. A test
    with no coverage is left out, since it ran its code elsewhere, such as
    in a child process."""
    by_file: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for nodeid, mask in masks.items():
        if mask:
            by_file[nodeid.partition("::")[0]].append((nodeid, mask))
    out: dict[str, list[tuple[str, str, str]]] = {}
    for path, tests in sorted(by_file.items()):
        found = []
        first: dict[int, str] = {}
        reps = []
        for nodeid, mask in tests:
            if mask in first:
                found.append((nodeid, "same", first[mask]))
            else:
                first[mask] = nodeid
                reps.append((mask.bit_count(), nodeid, mask))
        reps.sort()
        for i, (count, nodeid, mask) in enumerate(reps):
            for other_count, other, other_mask in reps[i + 1:]:
                if other_count > count and mask & ~other_mask == 0:
                    found.append((nodeid, "inside", other))
                    break
        if found:
            out[path] = found
    return out


def param_sets(masks: dict[str, int], min_cases: int = 4) -> list[tuple[str, int, int]]:
    """(test function, cases, distinct coverage sets) where the cases cover
    fewer sets than there are cases."""
    groups: dict[str, list[int]] = defaultdict(list)
    for nodeid, mask in masks.items():
        if "[" in nodeid and mask:
            groups[nodeid.partition("[")[0]].append(mask)
    rows = [(fn, len(ms), len(set(ms))) for fn, ms in groups.items()]
    return sorted((r for r in rows if r[1] >= min_cases and r[2] < r[1]),
                  key=lambda r: r[2] - r[1])


def _numerics(path: str) -> bool:
    return any(path.startswith(f"{folder}/") for folder in NUMERICS)


def report(masks: dict[str, int], times: dict[str, float], prefixes: list[str]) -> str:
    keep = {k: v for k, v in masks.items() if not prefixes or k.startswith(tuple(prefixes))}
    lines = []
    total, saved = 0, 0.0
    for path, found in candidates(keep).items():
        n = sum(1 for k in keep if k.startswith(path + "::"))
        lines.append(f"{path}: {len(found)} of {n} tests"
                     + ("  (numerics: kernel paths not seen)" if _numerics(path) else ""))
        for nodeid, how, other in found:
            t = times.get(nodeid)
            saved += t or 0
            what = "same coverage as" if how == "same" else "inside"
            lines.append(f"  {nodeid.partition('::')[2]}  {what} {other.partition('::')[2]}"
                         + (f"  ({t:.2f} s)" if t is not None else ""))
        total += len(found)
    rows = param_sets(keep)
    if rows:
        lines += ["", "Parametrized tests whose cases share coverage:"]
        lines += [f"  {fn}: {cases} cases, {sets} distinct" for fn, cases, sets in rows]
    lines += ["", f"{total} candidate tests" + (f", {saved:.0f} s of test time" if times else "")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prefixes", nargs="*", help="only test ids that start with one of these")
    ap.add_argument("--db", type=Path, help="coverage data (default: the last full gate run's)")
    ap.add_argument("--junit", type=Path, help="a junit report, for each candidate's time")
    a = ap.parse_args(argv)
    db = a.db or default_db()
    if not db.exists():
        print(f"no coverage data at {db}; run scripts/gate.py run --full first", file=sys.stderr)
        return 2
    times = read_times(a.junit) if a.junit else {}
    print(report(read_masks(db), times, a.prefixes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
