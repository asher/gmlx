#!/usr/bin/env python3
"""Fail on tests that wait a fixed time or bound a wall-clock time.

A sleep used to let another thread or process get somewhere, and an assert
that elapsed time stays under a bound, both pass on an idle Mac and fail on
a loaded CI runner. Tests synchronize on events, pipes, injected clocks or
hooks instead, as docs/internals/testing.md says.

This flags, in the tests pytest and the ComfyUI pack job collect (not the
GPU harnesses in tests/e2e or the probes in tests/invariant_probes):

- a call to time.sleep or asyncio.sleep with a positive number literal;
- an assert whose comparison puts a clock reading, or a name computed
  from one, under an upper bound.

A sleep or a deadline assert inside a poll is allowed: a while loop, or a
for loop that can end early with break or return.

A function may keep these when it stands in for a hung service or bounds a
hang guard: list it in scripts/test_timing_allowlist.txt as
`path::function`, with a comment saying why.

  python scripts/check-test-timing.py           # check tests/
  python scripts/check-test-timing.py --list    # print every finding

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "scripts" / "test_timing_allowlist.txt"
SKIP = ("tests/e2e/", "tests/invariant_probes/")
CLOCKS = {"monotonic", "time", "perf_counter", "monotonic_ns", "perf_counter_ns"}
SLEEPS = {"sleep"}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute):
            return f.attr
        if isinstance(f, ast.Name):
            return f.id
    return None


def _reads_clock(node: ast.AST, derived: set[str]) -> bool:
    for sub in ast.walk(node):
        if _call_name(sub) in CLOCKS and isinstance(sub.func, (ast.Attribute, ast.Name)):
            owner = sub.func.value.id if isinstance(sub.func, ast.Attribute) and isinstance(
                sub.func.value, ast.Name) else None
            if owner in (None, "time", "loop") or isinstance(sub.func, ast.Name):
                return True
        if isinstance(sub, ast.Name) and sub.id in derived:
            return True
    return False


def _names(target: ast.AST):
    """The names an assignment target binds. An attribute or an item of
    another object binds none."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _names(elt)
    elif isinstance(target, ast.Starred):
        yield from _names(target.value)


def _positive_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool) and node.value > 0


class _Finder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.derived: list[set[str]] = [set()]
        self.loops: list[int] = [0]
        self.found: list[tuple[int, str, str]] = []

    def _function(self, node) -> None:
        self.stack.append(node.name)
        self.derived.append(set())
        self.loops.append(0)
        self.generic_visit(node)
        self.loops.pop()
        self.derived.pop()
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _function

    def _loop(self, node) -> None:
        polls = isinstance(node, ast.While) or any(
            isinstance(sub, (ast.Break, ast.Return)) for sub in ast.walk(node))
        self.loops[-1] += polls
        self.generic_visit(node)
        self.loops[-1] -= polls

    visit_While = visit_For = visit_AsyncFor = _loop

    def _where(self) -> str:
        return self.stack[0] if self.stack else "<module>"

    def visit_Assign(self, node: ast.Assign) -> None:
        if _reads_clock(node.value, self.derived[-1]):
            for target in node.targets:
                self.derived[-1].update(_names(target))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (_call_name(node) in SLEEPS and node.args and _positive_literal(node.args[0])
                and not self.loops[-1]):
            self.found.append((node.lineno, self._where(), "sleep"))
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if self.loops[-1]:
            return self.generic_visit(node)
        for cmp in ast.walk(node.test):
            if not isinstance(cmp, ast.Compare):
                continue
            operands = [cmp.left, *cmp.comparators]
            for i, op in enumerate(cmp.ops):
                small, large = operands[i], operands[i + 1]
                if isinstance(op, (ast.Gt, ast.GtE)):
                    small, large = large, small
                elif not isinstance(op, (ast.Lt, ast.LtE)):
                    continue
                if _reads_clock(small, self.derived[-1]) and not _reads_clock(large, self.derived[-1]):
                    self.found.append((node.lineno, self._where(), "upper bound"))
        self.generic_visit(node)


def findings(path: Path) -> list[tuple[int, str, str]]:
    finder = _Finder()
    finder.visit(ast.parse(path.read_text(), filename=str(path)))
    return finder.found


def allowed() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    return {line.split("#")[0].strip() for line in ALLOWLIST.read_text().splitlines()
            if line.split("#")[0].strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print every finding, allowed or not")
    a = ap.parse_args(argv)
    keep = allowed()
    bad, seen = [], set()
    for path in sorted((ROOT / "tests").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(SKIP):
            continue
        for line, where, kind in findings(path):
            key = f"{rel}::{where}"
            seen.add(key)
            if a.list or key not in keep:
                bad.append(f"{rel}:{line}: {kind} in {where}" + (" (allowed)" if key in keep else ""))
    stale = sorted(keep - seen)
    for line in bad:
        print(line)
    for key in stale:
        print(f"{ALLOWLIST.relative_to(ROOT)}: {key} has no finding left; remove it")
    if a.list:
        return 0
    if bad:
        print(f"\n{len(bad)} timing-based wait(s) or bound(s). Synchronize on an event, a pipe, an "
              "injected clock or a hook instead; see docs/internals/testing.md.")
    return 1 if bad or stale else 0


if __name__ == "__main__":
    sys.exit(main())
