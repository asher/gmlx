#!/usr/bin/env python3
"""Pick the tests a change can break, and run them.

  python scripts/affected_tests.py               # list the test files
  python scripts/affected_tests.py --run         # run them with xdist
  python scripts/affected_tests.py --run -- -x   # extra pytest arguments

The change is everything since the merge-base with main, committed or not,
plus untracked files. A changed gmlx file selects the tests that ran it in
the last `scripts/gate.py run --full`, from the test map that run writes.
A static scan of the tests adds the files that import it among those the
map cannot speak for: test files added or changed since the map, and test
files that run a child interpreter. A gmlx file the map does not know, or
knows only from import time, selects every test file that imports it.

A changed test file selects itself and the tests that import it, a changed
conftest its folder, a docs change the docs tests, and any other file the
tests that name it. This is the inner loop; the gate runs before a push.

--run drops KQUANT_TEST_GGUF_DIR, as the gate does, so no real-model test
runs. Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changed_paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORKERS = 8
MAP_AGE = timedelta(days=28)
# gmlx imports in code a test runs in a child interpreter
_CODE_IMPORT = re.compile(r"^\s*(?:from\s+(gmlx[\w.]*)\s+import\s+([\w, ]+)|import\s+(gmlx[\w.]*))", re.M)


def _git(*args: str, root: Path | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=root or ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def changed(base: str, root: Path | None = None) -> list[str]:
    """Paths changed since the merge-base with ``base``, in the work tree too."""
    root = root or ROOT
    fork = _git("merge-base", "HEAD", base, root=root).strip()
    if not fork:
        raise SystemExit(f"affected_tests: no merge-base with {base}")
    return _changed_since(fork, root)


def _changed_since(commit: str, root: Path) -> list[str]:
    paths = _git("diff", "--name-only", "--no-renames", commit, root=root).splitlines()
    paths += _git("ls-files", "--others", "--exclude-standard", root=root).splitlines()
    return sorted({p for p in paths if p})


def module_name(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.suffix != ".py" or p.parts[0] != "gmlx":
        return None
    parts = list(p.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def is_test(path: str) -> bool:
    return path.startswith("tests/") and PurePosixPath(path).name.startswith("test_") \
        and path.endswith(".py")


class Scan:
    """Which test files import which gmlx modules and test helpers."""

    def __init__(self, root: Path | None = None, extra_modules: list[str] = ()) -> None:
        root = root or ROOT
        self.root = root
        gmlx = [p.relative_to(root).as_posix() for p in (root / "gmlx").rglob("*.py")
                if "__pycache__" not in p.parts]
        self.files = sorted(p.relative_to(root).as_posix() for p in (root / "tests").rglob("*.py")
                            if "__pycache__" not in p.parts)
        self.modules = {m: f for f in [*gmlx, *extra_modules] if (m := module_name(f))}
        # Tests import helpers from any test folder by basename.
        self.helpers: dict[str, str] = {}
        for f in self.files:
            if PurePosixPath(f).parent.as_posix() != "tests/fixtures":
                self.helpers.setdefault(PurePosixPath(f).stem, f)
        self._known = set(self.files)
        self.texts = {f: (root / f).read_text(errors="replace") for f in self.files}
        self.imports = {f: self._imports(f) for f in self.files}
        self.reverse: dict[str, set[str]] = {}
        for f, deps in self.imports.items():
            for d in deps:
                self.reverse.setdefault(d, set()).add(f)

    def _resolve(self, dotted: str) -> list[str]:
        """The gmlx module ``dotted`` names, and the packages above it."""
        parts = dotted.split(".")
        return [self.modules[n] for i in range(len(parts), 0, -1)
                if (n := ".".join(parts[:i])) in self.modules]

    def _helper(self, dotted: str) -> list[str]:
        top = dotted.split(".")[0]
        if top == "tests":
            top = dotted.split(".")[1] if "." in dotted else ""
        return [self.helpers[top]] if top in self.helpers else []

    def _imports(self, path: str) -> set[str]:
        try:
            tree = ast.parse(self.texts[path], filename=path)
        except SyntaxError:
            return set()
        here = PurePosixPath(path).parent
        deps: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps.update(self._resolve(alias.name) or self._helper(alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level:
                base = here
                for _ in range(node.level - 1):
                    base = base.parent
                target = base.joinpath(*(node.module or "").split(".")).as_posix()
                for name in (target, *(f"{target}/{a.name}" for a in node.names)):
                    deps.update(c for c in (f"{name}.py", f"{name}/__init__.py") if c in self._known)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                deps.update(self._resolve(mod) or self._helper(mod))
                for alias in node.names:
                    deps.update(self._resolve(f"{mod}.{alias.name}"))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("gmlx.") and " " not in node.value:
                    # monkeypatch targets and import_module names
                    deps.update(self._resolve(node.value))
                for m in _CODE_IMPORT.finditer(node.value):
                    mod = m.group(1) or m.group(3)
                    deps.update(self._resolve(mod))
                    for name in (m.group(2) or "").split(","):
                        if name.strip():
                            deps.update(self._resolve(f"{mod}.{name.strip()}"))
        deps.discard(path)
        return deps

    def importers(self, target: str) -> set[str]:
        """Test files and conftests that import ``target``, directly or
        through test helpers."""
        seen, todo = {target}, [target]
        while todo:
            for f in self.reverse.get(todo.pop(), ()):
                if f not in seen:
                    seen.add(f)
                    todo.append(f)
        seen.discard(target)
        return seen

    def runs_a_child(self, path: str) -> bool:
        return "sys.executable" in self.texts.get(path, "")


def map_file(root: Path | None = None) -> Path:
    root = root or ROOT
    common = _git("rev-parse", "--path-format=absolute", "--git-common-dir", root=root).strip()
    return Path(common) / "gmlx-gate" / "test-map.json"


def build_map(db: Path, commit: str, root: Path | None = None) -> dict:
    """For each file under ``root``, the test files whose coverage contexts
    ran it. A file that ran only at import, under the empty context, gets
    an empty list."""
    root = root or ROOT
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    top = root.resolve()
    rel: dict[int, str | None] = {}
    for fid, path in con.execute("SELECT id, path FROM file"):
        try:
            rel[fid] = Path(path).resolve().relative_to(top).as_posix()
        except ValueError:
            rel[fid] = None
    contexts = {cid: ctx.split("::")[0] for cid, ctx in con.execute("SELECT id, context FROM context")}
    tests: dict[str, int] = {}
    files: dict[str, set[int]] = {}
    for table in ("line_bits", "arc"):
        try:
            rows = con.execute(f"SELECT DISTINCT file_id, context_id FROM {table}").fetchall()
        except sqlite3.OperationalError:
            continue
        for fid, cid in rows:
            if rel.get(fid) is None:
                continue
            entry = files.setdefault(rel[fid], set())
            name = contexts.get(cid, "")
            if name.startswith("tests/"):
                entry.add(tests.setdefault(name, len(tests)))
    con.close()
    return {"commit": commit, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tests": list(tests), "files": {f: sorted(files[f]) for f in sorted(files)}}


def write_map(db: Path, commit: str, root: Path | None = None) -> Path:
    root = root or ROOT
    out = map_file(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(build_map(db, commit, root)))
    tmp.replace(out)
    return out


def load_map(root: Path | None = None, now: datetime | None = None) -> tuple[dict | None, list[str]]:
    """The test map with the test files changed since it, and notes to print."""
    root = root or ROOT
    path = map_file(root)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, ["no test map; selecting by imports only (scripts/gate.py run --full writes one)"]
    commit = data.get("commit", "")
    if subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root,
                      capture_output=True).returncode != 0:
        return None, [f"the test map's commit {commit[:10]} is not here; selecting by imports only"]
    data["since"] = {p for p in _changed_since(commit, root) if p.startswith("tests/")}
    age = (now or datetime.now(timezone.utc)) - datetime.fromisoformat(data["when"])
    notes = [f"test map from {commit[:10]}, {age.days} days old"]
    if age > MAP_AGE:
        notes.append("the test map is over four weeks old; refresh it with scripts/gate.py run --full")
    return data, notes


def _mirror(path: str, root: Path) -> set[str]:
    parts = PurePosixPath(path).parts
    if len(parts) > 2 and (root / "tests" / parts[1]).is_dir():
        return {f"tests/{parts[1]}"}
    flat = f"tests/test_{PurePosixPath(path).stem}.py"
    return {flat} if len(parts) == 2 and (root / flat).is_file() else set()


def _named(path: str, scan: Scan) -> set[str]:
    """Test files that name ``path``, or import a helper that does."""
    name = PurePosixPath(path).name
    needles = {path, name} if len(name) >= 8 else {path}
    if path.startswith("crates/"):
        needles.add(path.split("/")[1])
    hits = {f for f, text in scan.texts.items() if f != path and any(n in text for n in needles)}
    found = {f for h in hits for f in (h, *scan.importers(h)) if is_test(f)}
    return found | {PurePosixPath(h).parent.as_posix() for h in hits if PurePosixPath(h).name == "conftest.py"}


def select(paths: list[str], root: Path | None = None, test_map: dict | None = None) -> tuple[list[str], list[str]]:
    """Test files and folders for ``paths``, and the paths that selected none."""
    root = root or ROOT
    scan = Scan(root, [p for p in paths if module_name(p)])
    picked: set[str] = set()
    folders: set[str] = set()
    unmatched: list[str] = []
    for path in paths:
        name = PurePosixPath(path).name
        found: set[str] = set()
        if name == "conftest.py" and path.startswith("tests/"):
            folders.add(PurePosixPath(path).parent.as_posix())
            continue
        if path.endswith(".py") and (path.startswith("tests/") or module_name(path)):
            static = scan.importers(path)
            ran = []
            if test_map and module_name(path):
                ran = [test_map["tests"][i] for i in test_map["files"].get(path, [])]
            if ran:
                # The map speaks for every test file but those it predates and
                # those whose gmlx code runs in a child, which coverage misses.
                found = {f for f in ran if (root / f).is_file()}
                found |= {f for f in static if f in test_map["since"] or scan.runs_a_child(f)}
            else:
                found = static
            folders |= {PurePosixPath(f).parent.as_posix() for f in found
                        if PurePosixPath(f).name == "conftest.py"}
            found = {f for f in found if is_test(f)}
            if is_test(path) and (root / path).exists():
                found.add(path)
            if not found and module_name(path):
                found = _mirror(path, root)
            elif not found and not is_test(path):
                # a helper run as a script, such as a fake binary
                found = _named(path, scan) or {PurePosixPath(path).parent.as_posix()}
        elif changed_paths.is_docs(path):
            found = {p.relative_to(root).as_posix() for g in changed_paths.DOCS_TESTS for p in root.glob(g)}
        else:
            found = _named(path, scan)
        folders |= {f for f in found if (root / f).is_dir()}
        if found or path in folders:
            picked |= found
        else:
            unmatched.append(path)
    if "tests" in folders:
        return ["tests"], unmatched
    picked = {f for f in picked if not any(f == d or f.startswith(d + "/") for d in folders)}
    return sorted(folders | picked), unmatched


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="main", help="the branch the change forks from (default main)")
    ap.add_argument("--run", action="store_true", help="run the selected tests")
    ap.add_argument("--workers", type=int, default=WORKERS, help=f"xdist workers (default {WORKERS})")
    a = ap.parse_args(argv)
    paths = changed(a.base)
    test_map, notes = load_map()
    for note in notes:
        print(note, file=sys.stderr)
    tests, unmatched = select(paths, test_map=test_map)
    for path in unmatched:
        print(f"no tests found for {path}", file=sys.stderr)
    if not a.run:
        print("\n".join(tests))
        return 0
    if not tests:
        print("affected_tests: nothing to run", file=sys.stderr)
        return 0
    env = {k: v for k, v in os.environ.items() if k != "KQUANT_TEST_GGUF_DIR"}
    cmd = [sys.executable, "-m", "pytest", "-q", "-n", str(a.workers), "--dist", "worksteal",
           "-m", "not nightly", *tests, *extra]
    print(f"{len(tests)} test files or folders for {len(paths)} changed paths", file=sys.stderr)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
